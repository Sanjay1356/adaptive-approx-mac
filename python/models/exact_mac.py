"""
Exact signed-INT8 MAC behavioral model.

    ACC_new = ACC_old + (A * B)

Three explicitly separated notions, per the project spec:

    mathematical result   -> unlimited-precision Python integer
    hardware-width product -> clipped/checked to signed INT16
    hardware-width accumulator -> clipped/wrapped or saturated to signed INT32

We do NOT silently rely on Python's unlimited integers: every width
transition below is an explicit, auditable step, and product overflow
(which should be IMPOSSIBLE for INT8xINT8->INT16 given correct widths)
is checked with an assertion rather than assumed.
"""

from dataclasses import dataclass

from config import config as cfg
from models.multiplier import golden_multiply, clip_to_width


@dataclass
class MacResult:
    a: int
    b: int
    product_math: int      # unlimited-precision A*B
    product_hw: int        # A*B after clipping to PRODUCT_WIDTH (should never actually clip)
    acc_before: int
    acc_math: int           # unlimited-precision acc_before + product
    acc_hw: int              # acc_before + product after ACC_WIDTH policy applied
    overflowed: bool         # whether acc_math fell outside the ACC_WIDTH range


def exact_multiply(a: int, b: int) -> int:
    """A * B -> hardware-width signed product (INT16)."""
    product_math = golden_multiply(a, b)
    # An 8x8 signed multiply can never exceed the signed 16-bit range
    # ((-128)*(-128)=16384 needs 15 bits + sign -> fits INT16 exactly),
    # so this assertion should never fire; it exists to catch a config
    # mistake (e.g. someone widening OPERAND_WIDTH without widening
    # PRODUCT_WIDTH) rather than to "handle" real overflow.
    assert cfg.PRODUCT_MIN <= product_math <= cfg.PRODUCT_MAX, (
        f"product {product_math} does not fit signed INT{cfg.PRODUCT_WIDTH}; "
        "PRODUCT_WIDTH is misconfigured for this OPERAND_WIDTH"
    )
    return product_math  # already within width; no bits are actually dropped


def exact_mac_step(acc_before: int, a: int, b: int) -> MacResult:
    """One ACC = ACC + A*B step with explicit ACC_WIDTH policy."""
    product_hw = exact_multiply(a, b)
    acc_math = acc_before + product_hw

    if cfg.ACC_OVERFLOW_POLICY == "wrap":
        acc_hw = clip_to_width(acc_math, cfg.ACC_WIDTH, signed=True)
    elif cfg.ACC_OVERFLOW_POLICY == "saturate":
        acc_hw = max(cfg.ACC_MIN, min(cfg.ACC_MAX, acc_math))
    else:
        raise ValueError(f"unknown ACC_OVERFLOW_POLICY: {cfg.ACC_OVERFLOW_POLICY}")

    overflowed = not (cfg.ACC_MIN <= acc_math <= cfg.ACC_MAX)

    return MacResult(
        a=a,
        b=b,
        product_math=golden_multiply(a, b),
        product_hw=product_hw,
        acc_before=acc_before,
        acc_math=acc_math,
        acc_hw=acc_hw,
        overflowed=overflowed,
    )


def exact_mac_sequence(a_list, b_list, acc_init: int = 0):
    """Run a full accumulation sequence (a realistic dot product), returning
    the final MacResult list (one per step) so overflow/wrap behavior at any
    point in the sequence is inspectable, not just the final value."""
    assert len(a_list) == len(b_list)
    acc = acc_init
    steps = []
    for a, b in zip(a_list, b_list):
        r = exact_mac_step(acc, a, b)
        steps.append(r)
        acc = r.acc_hw
    return steps
