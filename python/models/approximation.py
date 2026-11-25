"""
Approximation framework.

STATUS: mechanism NOT frozen. This module implements multiple CANDIDATE
mechanisms behind one interface so experiments can compare them, per the
project brief. Nothing here should be read as a final decision.

Each candidate function has the signature:

    approx_multiply(a: int, b: int, level: int) -> int

where level in {0,1,2,3}: level 0 must always be bit-exact (equal to
golden_multiply), and level increases with approximation strength.

Candidates implemented:

  RECOMMENDED (see models/pp_truncation.py): `pp_column_truncation` --
      the actual hardware-grounded PP-column-drop mechanism, with a
      derived compensation constant. This supersedes the item below.

  CANDIDATE 1 (SUPERSEDED, kept only for old-vs-new comparison, exposed
      here as `old_output_truncation_proxy`): "partial-product LSB
      truncation" -- the k least-significant columns of the partial-product
      array are dropped before reduction (equivalent to not generating them
      in hardware), with a small constant compensation term to reduce bias.
      Investigation in the next project stage showed this was actually
      computing the EXACT product and then zeroing output bits (product-bit
      truncation), not true PP-column truncation -- see pp_truncation.py.

  CANDIDATE 2: "operand LSB rounding/truncation before multiply" -- round
      A and B to fewer significant bits before the (still-exact) multiply.
      Simpler to reason about mathematically, but less representative of
      an actual approximate-multiplier datapath (it looks more like
      requantization than approximate arithmetic), so it is kept only as
      a comparison baseline, NOT the primary candidate.

  CANDIDATE 3: "approximate reduction via lower-part OR-based compression"
      -- a common approximate-adder idea (e.g. lower-part-OR adder, LOA)
      applied to the final product formation. Included as a second
      structured candidate for comparison; more complex to map cleanly to
      a truncation-width control knob, so ranked below Candidate 1.

Do NOT treat the numeric level->k mapping below as final; it is a
starting parameterization to be revisited once one candidate is selected.
"""

from config import config as cfg
from models.multiplier import golden_multiply, clip_to_width
from models import pp_truncation

# level -> number of LSB partial-product columns dropped (Candidate 1)
# and equivalently number of LSBs rounded off before multiply (Candidate 2).
# Chosen so each level roughly doubles the number of affected columns;
# NOT yet justified by an actual PPA study -- placeholder ladder to be
# revisited once a mechanism is selected (see SECTION E of the review).
LEVEL_TO_K = {0: 0, 1: 2, 2: 4, 3: 6}


def _k_for_level(level: int) -> int:
    if level not in LEVEL_TO_K:
        raise ValueError(f"undefined approximation level: {level}")
    return LEVEL_TO_K[level]


def candidate1_pp_truncation(a: int, b: int, level: int) -> int:
    """Drop the k LSB columns of the partial-product array, then add a
    constant compensation term (mean of the dropped contribution over
    random operands) to reduce systematic bias -- a standard trick for
    truncated multipliers. k=0 is bit-exact."""
    k = _k_for_level(level)
    exact = golden_multiply(a, b)
    if k == 0:
        return exact
    # Dropping the k LSBs of the *product* is the numerical effect of not
    # generating PP columns below weight k (for this simple compensated
    # model; a full bit-accurate PP-column model belongs in RTL/gate sim).
    truncated = (exact >> k) << k
    # constant correction: add half the truncation quantum, standard for
    # reducing mean truncation bias (equivalent to round-to-nearest bias
    # correction, implementable as a fixed added constant in hardware).
    correction = 1 << (k - 1)
    return truncated + correction


def candidate2_operand_rounding(a: int, b: int, level: int) -> int:
    """Round A and B to fewer significant bits before an otherwise-exact
    multiply. Comparison baseline only -- see module docstring."""
    k = _k_for_level(level)
    if k == 0:
        return golden_multiply(a, b)

    def round_off(x, k):
        half = 1 << (k - 1)
        rounded = ((x + half) >> k) << k
        return clip_to_width(rounded, cfg.OPERAND_WIDTH + 1, signed=True)

    a_r = round_off(a, k)
    b_r = round_off(b, k)
    return a_r * b_r


def candidate3_lower_part_or(a: int, b: int, level: int) -> int:
    """Lower-part-OR-style approximation: for the low k bits of each
    operand's magnitude, replace exact addition in that region with a
    bitwise-OR-based approximate sum (as in the LOA adder literature),
    implemented here at the numerical level as: exact product, then the
    low k bits of the result are replaced by (a_low OR b_low) instead of
    their carry-correct contribution. Included for comparison only."""
    k = _k_for_level(level)
    exact = golden_multiply(a, b)
    if k == 0:
        return exact
    mask = (1 << k) - 1
    a_low = a & mask
    b_low = b & mask
    approx_low = a_low | b_low
    return (exact & ~mask) | (approx_low & mask)


def pp_column_truncation(a: int, b: int, level: int) -> int:
    """HARDWARE-GROUNDED mechanism (see models/pp_truncation.py for full
    derivation): drop the k least-significant PP-array columns (no PP
    generation, no carry from that region) with a derived, k-dependent
    compensation constant. This is the FINAL recommended mechanism,
    replacing candidate1_pp_truncation (kept below, renamed, only for the
    explicit old-vs-new comparison in experiments/compare_old_new.py)."""
    return pp_truncation.level_multiply(a, b, level, compensate=True)


def old_output_truncation_proxy(a: int, b: int, level: int) -> int:
    """RENAMED alias for the previous stage's candidate1_pp_truncation.
    This is product-BIT truncation (compute exact, then zero low output
    bits), NOT partial-product-column truncation -- see the module
    docstring in models/pp_truncation.py for the proof that these differ.
    Retained ONLY so experiments/compare_old_new.py can quantify the gap
    between the old numerical proxy and the new hardware-grounded model."""
    return candidate1_pp_truncation(a, b, level)


CANDIDATES = {
    "pp_column_truncation": pp_column_truncation,
    "old_output_truncation_proxy": old_output_truncation_proxy,
    "candidate2_operand_rounding": candidate2_operand_rounding,
    "candidate3_lower_part_or": candidate3_lower_part_or,
}

RECOMMENDED_CANDIDATE = "pp_column_truncation"


def approx_mac_step(acc_before: int, a: int, b: int, level: int,
                     candidate: str = RECOMMENDED_CANDIDATE):
    """One approximate ACC = ACC + approx(A*B) step, same width policy as
    the exact MAC (see models/exact_mac.py) applied to the accumulator."""
    fn = CANDIDATES[candidate]
    product = fn(a, b, level)
    acc_math = acc_before + product
    if cfg.ACC_OVERFLOW_POLICY == "wrap":
        acc_hw = clip_to_width(acc_math, cfg.ACC_WIDTH, signed=True)
    else:
        acc_hw = max(cfg.ACC_MIN, min(cfg.ACC_MAX, acc_math))
    return acc_hw
