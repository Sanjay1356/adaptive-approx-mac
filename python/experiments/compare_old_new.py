"""
Quantifies the gap between:
  OLD  -- exact product then zero low output bits (old_output_truncation_proxy)
  NEW  -- true PP-column drop with derived compensation (pp_column_truncation)

Run from adaptive_approx_mac/: python -m experiments.compare_old_new
"""
import csv
import os

from config import config as cfg
from models.multiplier import golden_multiply
from models.approximation import old_output_truncation_proxy, pp_column_truncation


def run():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    path = os.path.join(cfg.RESULTS_DIR, "old_vs_new_comparison.csv")

    identical_count = {l: 0 for l in range(4)}
    old_abs_err_sum = {l: 0 for l in range(4)}
    new_abs_err_sum = {l: 0 for l in range(4)}
    n = 65536

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["a", "b", "exact", "level", "old_approx", "new_approx",
                          "old_abs_err", "new_abs_err", "identical"])
        for a in range(cfg.OPERAND_MIN, cfg.OPERAND_MAX + 1):
            for b in range(cfg.OPERAND_MIN, cfg.OPERAND_MAX + 1):
                exact = golden_multiply(a, b)
                for level in range(4):
                    old = old_output_truncation_proxy(a, b, level)
                    new = pp_column_truncation(a, b, level)
                    identical = old == new
                    if identical:
                        identical_count[level] += 1
                    old_abs_err_sum[level] += abs(exact - old)
                    new_abs_err_sum[level] += abs(exact - new)
                    writer.writerow([a, b, exact, level, old, new,
                                      abs(exact - old), abs(exact - new), identical])

    print("Old (output-bit truncation) vs New (PP-column truncation), 65,536 pairs/level:")
    for level in range(4):
        pct_identical = 100.0 * identical_count[level] / n
        old_mae = old_abs_err_sum[level] / n
        new_mae = new_abs_err_sum[level] / n
        verdict = "underestimates" if old_mae < new_mae else "overestimates"
        print(f"  L{level}: identical on {pct_identical:.1f}% of pairs | "
              f"old MAE={old_mae:.2f} new MAE={new_mae:.2f} "
              f"(old {verdict} true PP-truncation error)")
    print(f"Full per-pair comparison written to {path}")
    print()
    print("RECOMMENDATION: the old model is NOT a valid stand-in for the "
          "hardware mechanism at any level > 0 -- see per-level identical%% "
          "above. It should be discarded as the design reference and kept "
          "only as a documented historical baseline.")


if __name__ == "__main__":
    run()
