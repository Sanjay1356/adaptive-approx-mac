"""
Two independent references for signed INT8 x INT8 multiplication:

1. golden_multiply()      -- simple numerical reference (PRIMARY golden model)
2. bitlevel_multiply()    -- explicit partial-product generation + reduction,
                             modelling the actual hardware decomposition.

Both must agree exactly across the full 65,536-pair space. This is checked
in tests/test_mac.py and again in experiments/exhaustive_analysis.py.

IMPORTANT ON SIGNED BIT-LEVEL MULTIPLICATION:
Naively summing Ai*Bj*2^(i+j) for unsigned bit patterns gives the UNSIGNED
product of the two's-complement bit patterns, which is wrong for signed
operands. We use the standard two's-complement decomposition (equivalent to
the Baugh-Wooley sign handling) rather than a "simplified" model that would
silently give wrong answers for negative operands. Specifically, for an
N-bit two's-complement number X with bits x_{N-1}...x_0:

    X = -x_{N-1} * 2^(N-1) + sum_{i=0}^{N-2} x_i * 2^i

Multiplying two such expansions and collecting terms gives every partial
product's correct sign. That is what bitlevel_multiply() implements.
"""

from config import config as cfg


def to_bits(value: int, width: int):
    """Return the two's-complement bit list [b0, b1, ..., b_{width-1}] (LSB first)
    for `value`, which must be in the representable signed range."""
    if not (-(1 << (width - 1)) <= value <= (1 << (width - 1)) - 1):
        raise ValueError(f"{value} does not fit in signed {width}-bit range")
    u = value & ((1 << width) - 1)  # two's-complement bit pattern as unsigned int
    return [(u >> i) & 1 for i in range(width)]


def golden_multiply(a: int, b: int) -> int:
    """Primary golden reference: Python's arbitrary-precision integers.
    This is mathematically exact and independent of any bit-level model."""
    if not (cfg.OPERAND_MIN <= a <= cfg.OPERAND_MAX):
        raise ValueError(f"a={a} out of signed INT8 range")
    if not (cfg.OPERAND_MIN <= b <= cfg.OPERAND_MAX):
        raise ValueError(f"b={b} out of signed INT8 range")
    return a * b  # exact, unlimited precision -> hardware width applied by caller


def bitlevel_multiply(a: int, b: int, width: int = cfg.OPERAND_WIDTH) -> int:
    """Explicit two's-complement partial-product multiplication.

    Implements: X = -x_{N-1}*2^(N-1) + sum_{i<N-1} x_i * 2^i, then
    P = A * B expanded term-by-term over all width*width bit-pairs,
    each weighted with the correct sign depending on whether it involves
    the sign bit of A and/or B. This mirrors what an 8x8 partial-product
    array (e.g. Baugh-Wooley) must implement in hardware, so it is a
    meaningful reference for later RTL comparison -- not just a second
    call to '*'.
    """
    a_bits = to_bits(a, width)
    b_bits = to_bits(b, width)

    # A = sum_i weight(a_bits,i), B = sum_j weight(b_bits,j), P = A*B.
    # Expanding the width*width term-by-term product directly is easy to get
    # sign-wrong for the (sign,sign) cross term, so we use the standard,
    # auditable Baugh-Wooley-equivalent closed form instead of an inline
    # double loop:
    a_mag_bits = a_bits[:-1]
    b_mag_bits = b_bits[:-1]
    a_sign = a_bits[-1]
    b_sign = b_bits[-1]

    unsigned_part = sum(
        a_mag_bits[i] * b_mag_bits[j] * (1 << (i + j))
        for i in range(width - 1)
        for j in range(width - 1)
    )
    cross_a = sum(a_mag_bits[i] * (1 << (i + width - 1)) for i in range(width - 1))
    cross_b = sum(b_mag_bits[j] * (1 << (j + width - 1)) for j in range(width - 1))
    sign_term = (1 << (2 * (width - 1)))

    total = (
        unsigned_part
        - b_sign * cross_a
        - a_sign * cross_b
        + a_sign * b_sign * sign_term
    )
    return total


def clip_to_width(value: int, width: int, signed: bool = True) -> int:
    """Apply hardware-width truncation to `value` (two's-complement wrap)."""
    mask = (1 << width) - 1
    v = value & mask
    if signed and v >= (1 << (width - 1)):
        v -= (1 << width)
    return v
