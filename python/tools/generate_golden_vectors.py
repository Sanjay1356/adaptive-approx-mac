"""Regenerate ../vectors/golden_vectors.txt from the Python golden model.

Each line: "AA BB L PPPP" (hex) = 8-bit A, 8-bit B, level 0-3, expected 16-bit
product, all two's-complement encoded. 256 x 256 x 4 = 262,144 lines.
These are the vectors tb/tb_approx_mac.sv and tb/tb_fixed_k.sv check against.

Run from python/:   python -m tools.generate_golden_vectors
"""
import os

from models.pp_truncation import level_multiply

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "vectors", "golden_vectors.txt")


def main():
    lines = []
    for a in range(-128, 128):
        for b in range(-128, 128):
            for level in range(4):
                p = level_multiply(a, b, level)
                lines.append(f"{a & 0xFF:02x} {b & 0xFF:02x} {level} {p & 0xFFFF:04x}\n")
    with open(OUT, "w") as f:
        f.writelines(lines)
    print(f"wrote {len(lines)} vectors to {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
