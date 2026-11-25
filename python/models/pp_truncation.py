"""
Hardware-grounded 8x8 signed partial-product (PP) truncation.

This replaces the OLD numerical proxy (exact_product >> k << k, see
models/approximation.py candidate1_pp_truncation, kept only for the
old-vs-new comparison in experiments/compare_old_new.py) with a model
that actually reflects dropping PP-array columns before reduction.

------------------------------------------------------------------
WHY THE OLD MODEL WAS NOT PP-COLUMN TRUNCATION
------------------------------------------------------------------
`(exact >> k) << k` computes the FULL exact product (all PPs generated,
fully reduced with carry-propagate addition), and only THEN zeroes the
k least-significant output bits. Any carry that the low k bits would
have produced into bit k is already baked into the exact result before
the shift -- so this is PRODUCT-BIT (output) truncation, not PP-column
truncation. It corresponds to hardware that builds the full exact
multiplier and only saves gates in a final output register/mask -- no
PP generation or reduction hardware is actually removed. That is why it
was flagged in the previous stage as a numerical proxy, not a design.

------------------------------------------------------------------
THE 8x8 SIGNED PP ARRAY
------------------------------------------------------------------
For two's-complement 8-bit operands, bit i of A contributes a signed
weight:

    w_A(i) = -A_i * 2^i   if i == 7 (the sign bit)
    w_A(i) =  A_i * 2^i   otherwise

so that A = sum_i w_A(i) exactly reproduces two's-complement value.
Likewise for B. Then, since A*B = (sum_i w_A(i)) * (sum_j w_B(j)):

    A*B = sum_i sum_j  w_A(i) * w_B(j)

Each term T(i,j) = w_A(i)*w_B(j) is one PP-array cell, and it sits in
COLUMN c = i+j (0..14), with the standard population per column for an
8x8 array: 1,2,3,4,5,6,7,8,7,6,5,4,3,2,1 (columns 0..14). Terms
involving the sign bit (i==7 or j==7) get a sign flip, and the single
term with i=j=7 gets a double flip (net positive) -- this is exactly
the Baugh-Wooley sign-correction structure, derived here per-term
rather than assumed.

------------------------------------------------------------------
DEFINING "DROP THE k LSB PP COLUMNS" (chosen: OPTION A/C -- discard,
no carry preserved from the dropped region)
------------------------------------------------------------------
We define dropping k columns as: do not generate T(i,j) for any (i,j)
with i+j < k, and inject NO carry from that region into column k. This
is Option A (don't generate) combined with Option C (no carry) from
the brief -- the simplest and cheapest hardware interpretation: the low
k columns' AND-gate PP generators and their reduction adders/
compressors are physically absent. The approximate product is:

    P_approx(k) = sum_{i+j >= k} T(i,j)  [+ compensation, see below]

We rejected "generate but don't reduce" (Option B) and "approximate
reduction but preserve some carry" (Option D) as the FIRST architecture
because they still require generating the low PP rows (no AND-gate
savings) and/or a probabilistic carry estimator (extra logic that can
erode the savings the mechanism is supposed to provide) -- see
SECTION 12 hardware-cost discussion in the accompanying review.
A middle-ground carry estimator is noted as a candidate REFINEMENT,
not adopted here.

------------------------------------------------------------------
SIGNED CASE / WHY UNSIGNED INTUITION ALONE IS NOT ENOUGH
------------------------------------------------------------------
For k <= 7 (our L1..L3 range: k in {2,4,6}), every dropped term has
i,j <= 6 (since i+j < k <= 7 forces both indices below the sign-bit
index 7), so the sign bit of neither operand ever participates in the
dropped region. This means, for our chosen k values, magnitude-only PP
cells are removed and the Baugh-Wooley sign-correction terms (which
live in higher columns, since they involve i=7 or j=7 => column >= 7)
are always fully preserved. This is an important, explicitly checked
property -- it would NOT hold for k >= 8, which is why k is capped
below 8 in this design.

------------------------------------------------------------------
COMPENSATION -- DERIVED, NOT ASSUMED
------------------------------------------------------------------
Dropping magnitude-only terms T(i,j) = A_i*B_j*2^(i+j) (i,j<=6, always
non-negative bit contributions) makes the approximate product a
systematic UNDER-estimate of the exact magnitude contribution from that
region (we never drop a negative contribution when k<=7, since no sign
term is ever in the dropped region). Under the standard assumption that
each magnitude bit is an independent uniform Bernoulli(0.5) variable
(reasonable as a first-order model; NOT verified against your actual
operand/activation distribution -- flagged as a limitation), each
dropped term has expectation E[A_i*B_j] = 0.25, so:

    E[dropped_sum(k)] = 0.25 * sum_{i+j<k, i,j in 0..6} 2^(i+j)

This is computed exactly (not guessed) by `expected_drop()` below and
rounded to the nearest integer to obtain a per-k compensation constant
-- a single small constant ADDED to the reduced result, implementable
as a hard-wired constant input to the final adder (cheap: a handful of
extra 1-bits in the addition, not proportional to k in a way that
threatens the savings from dropping PP generation itself). This is
explicitly a DERIVED, k-dependent, distribution-assumption-dependent
constant -- not the ad hoc 2^(k-1) used previously.
"""

from functools import lru_cache

from config import config as cfg


def to_bits(value: int, width: int):
    if not (-(1 << (width - 1)) <= value <= (1 << (width - 1)) - 1):
        raise ValueError(f"{value} does not fit in signed {width}-bit range")
    u = value & ((1 << width) - 1)
    return [(u >> i) & 1 for i in range(width)]


def _weight(bits, idx, width):
    if idx == width - 1:
        return -bits[idx] * (1 << idx)
    return bits[idx] * (1 << idx)


def pp_terms(a: int, b: int, width: int = cfg.OPERAND_WIDTH):
    """Return list of (i, j, column, term_value) for every PP-array cell.
    Sum of all term_value == a*b exactly (this is checked exhaustively
    in tests)."""
    a_bits = to_bits(a, width)
    b_bits = to_bits(b, width)
    terms = []
    for i in range(width):
        wa = _weight(a_bits, i, width)
        for j in range(width):
            wb = _weight(b_bits, j, width)
            terms.append((i, j, i + j, wa * wb))
    return terms


def column_population(width: int = cfg.OPERAND_WIDTH):
    """Number of PP-array cells landing in each column 0..2*(width-1)."""
    pops = [0] * (2 * width - 1)
    for i in range(width):
        for j in range(width):
            pops[i + j] += 1
    return pops


@lru_cache(maxsize=None)
def expected_drop(k: int, width: int = cfg.OPERAND_WIDTH) -> int:
    """Expected value of the sum of dropped magnitude-only terms
    (i+j < k, i,j <= width-2) under independent uniform Bernoulli(0.5)
    magnitude bits. Returns the constant rounded to nearest integer.
    Requires k <= width - 1 (guaranteed magnitude-only dropped region,
    see module docstring)."""
    if k > width - 1:
        raise ValueError(
            f"k={k} would drop columns touching the sign bit for width={width}; "
            "this design is only valid for k <= width-1"
        )
    total = 0.0
    for i in range(width - 1):       # magnitude bits only: 0..width-2
        for j in range(width - 1):
            if i + j < k:
                total += 0.25 * (1 << (i + j))
    return round(total)


def pp_truncated_multiply(a: int, b: int, k: int, width: int = cfg.OPERAND_WIDTH,
                           compensate: bool = True) -> int:
    """The hardware-grounded approximate product: drop all PP terms with
    column < k (no carry from that region), optionally add the derived
    compensation constant for that k.

    COMPENSATION GATING (fix applied after audit):
    The additive constant approximates E[dropped_sum(k)] as an ENSEMBLE
    average -- it is not a per-sample correction. Applied unconditionally,
    it produces a nonzero result for 0*x or x*0, which is wrong for any
    k>0 (if either operand is exactly zero, every dropped PP term is
    exactly zero too, since a dropped term T(i,j) with i+j<k always has
    i<k and j<k, so it needs a nonzero bit from BOTH operands within
    their low k bits to be nonzero at all).

    Minimal-hardware fix: gate the constant on
        gate = OR(A[0:k]) AND OR(B[0:k])
    i.e. two k-input OR gates + one AND gate on the operand's own low-k
    bits -- cheap, and it is a hardware-exact (not heuristic) test for
    "the discarded region cannot contain any nonzero term." It does NOT
    fix the deeper issue that the constant's magnitude still assumes a
    uniform bit distribution (a documented limitation for real, non-
    uniform DNN operand distributions), but it removes the pathological
    zero-operand case, which is what the audit flagged as a defect
    rather than a limitation.
    """
    if k == 0:
        # no columns dropped: must be bit-exact
        return sum(t[3] for t in pp_terms(a, b, width))
    kept = sum(t[3] for t in pp_terms(a, b, width) if t[2] >= k)
    if compensate:
        low_mask = (1 << k) - 1
        a_low_nonzero = (a & low_mask) != 0
        b_low_nonzero = (b & low_mask) != 0
        gate = a_low_nonzero and b_low_nonzero
        if gate:
            kept += expected_drop(k, width)
    return kept


# ---- Approximation ladder, hardware-grounded ----
# k = number of LSB PP columns dropped. Capped at width-1=7 so the
# dropped region never touches the sign bit (see docstring). Dropped
# term counts (out of 64 total 8x8 PP cells) for the chosen k values,
# using column_population(): k=2 -> 3 terms, k=4 -> 10 terms,
# k=6 -> 21 terms -- an increasing, physically meaningful fraction of
# the PP array removed at each level.
LEVEL_TO_K = {0: 0, 1: 2, 2: 4, 3: 6}


def level_multiply(a: int, b: int, level: int, compensate: bool = True) -> int:
    if level not in LEVEL_TO_K:
        raise ValueError(f"undefined level {level}")
    k = LEVEL_TO_K[level]
    return pp_truncated_multiply(a, b, k, compensate=compensate)
