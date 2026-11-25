"""Illustrative Python reference for the approximate INT8 multiplier.

Restates the arithmetic of the RTL (partial-product column truncation with
operand-gated statistical compensation). This is a readable illustration, not
the original golden-vector generator; it reproduces vectors/golden_vectors.txt.

    python3 model/approx_mul.py          # self-check against the golden vectors
"""
LEVEL_K = {0: 0, 1: 2, 2: 4, 3: 6}   # L0..L3 -> K


def approx_mul(a: int, b: int, K: int, W: int = 8) -> int:
    """Signed W-bit x W-bit product with all cells i + j < K removed."""
    a &= (1 << W) - 1
    b &= (1 << W) - 1
    s = 0
    for i in range(W):
        for j in range(W):
            if i + j < K:                       # removed partial-product cell
                continue
            t = ((a >> i) & 1) & ((b >> j) & 1)
            neg = (i == W - 1) ^ (j == W - 1)   # Baugh-Wooley-style sign weight
            s += -(t << (i + j)) if neg else (t << (i + j))
    C = sum(2 ** (i + j) for i in range(W) for j in range(W) if i + j < K) // 4
    gate = K > 0 and (a & ((1 << K) - 1)) != 0 and (b & ((1 << K) - 1)) != 0
    p = (s + (C if gate else 0)) & 0xFFFF       # 16-bit two's-complement wrap
    return p - 0x10000 if p & 0x8000 else p


if __name__ == "__main__":
    import pathlib
    vec = pathlib.Path(__file__).resolve().parent.parent / "vectors" / "golden_vectors.txt"
    bad = n = 0
    for line in vec.read_text().split("\n"):
        if not line.strip():
            continue
        aa, bb, lv, pp = line.split()
        a, b = int(aa, 16), int(bb, 16)
        a = a - 256 if a & 0x80 else a
        b = b - 256 if b & 0x80 else b
        got = approx_mul(a, b, LEVEL_K[int(lv)]) & 0xFFFF
        n += 1
        bad += got != int(pp, 16)
    print(f"{n} vectors, {bad} mismatches -> {'PASS' if bad == 0 else 'FAIL'}")
    print("127 x 127: L0 =", approx_mul(127, 127, 0), " L3 =", approx_mul(127, 127, 6))
