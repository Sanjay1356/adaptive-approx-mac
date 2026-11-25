"""
Exhaustive analysis over all 256x256 = 65,536 signed INT8 operand pairs.
Verifies golden vs bitlevel agreement and dumps the full result table.

Run from adaptive_approx_mac/:  python -m experiments.exhaustive_analysis
"""
import csv
import os

from config import config as cfg
from models.multiplier import golden_multiply, bitlevel_multiply


def run():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    path = os.path.join(cfg.RESULTS_DIR, "exhaustive_results.csv")

    mismatches = 0
    total = 0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["a", "b", "golden_product", "bitlevel_product", "match"])
        for a in range(cfg.OPERAND_MIN, cfg.OPERAND_MAX + 1):
            for b in range(cfg.OPERAND_MIN, cfg.OPERAND_MAX + 1):
                g = golden_multiply(a, b)
                bl = bitlevel_multiply(a, b)
                match = g == bl
                total += 1
                if not match:
                    mismatches += 1
                writer.writerow([a, b, g, bl, match])

    print(f"Exhaustive analysis complete: {total} pairs, {mismatches} mismatches")
    print(f"Results written to {path}")
    return mismatches == 0


if __name__ == "__main__":
    ok = run()
    raise SystemExit(0 if ok else 1)
