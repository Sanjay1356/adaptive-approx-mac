"""
Exhaustive (all 65,536 signed INT8xINT8 pairs) arithmetic characterization
of the hardware-grounded PP-column-truncation mechanism, for every level
L0..L3.

Run from adaptive_approx_mac/:  python -m experiments.exhaustive_pp_characterization
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import config as cfg
from models.multiplier import golden_multiply
from models.pp_truncation import level_multiply, LEVEL_TO_K


def relative_error(exact, approx):
    if exact == 0:
        return None
    return abs(exact - approx) / abs(exact)


def run():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    os.makedirs(cfg.PLOTS_DIR, exist_ok=True)

    per_level = {lvl: {"abs_err": [], "signed_err": [], "rel_err": []} for lvl in range(4)}
    csv_path = os.path.join(cfg.RESULTS_DIR, "pp_exhaustive_results.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["a", "b", "exact"] + [f"L{l}_approx" for l in range(4)]
                         + [f"L{l}_signed_err" for l in range(4)])
        for a in range(cfg.OPERAND_MIN, cfg.OPERAND_MAX + 1):
            for b in range(cfg.OPERAND_MIN, cfg.OPERAND_MAX + 1):
                exact = golden_multiply(a, b)
                approxes, errs = [], []
                for level in range(4):
                    approx = level_multiply(a, b, level)
                    err = exact - approx
                    per_level[level]["abs_err"].append(abs(err))
                    per_level[level]["signed_err"].append(err)
                    rel = relative_error(exact, approx)
                    if rel is not None:
                        per_level[level]["rel_err"].append(rel)
                    approxes.append(approx)
                    errs.append(err)
                writer.writerow([a, b, exact] + approxes + errs)

    n_samples = 65536
    summary = []
    for level in range(4):
        d = per_level[level]
        abs_err = np.array(d["abs_err"])
        signed_err = np.array(d["signed_err"])
        rel_err = np.array(d["rel_err"]) if d["rel_err"] else np.array([np.nan])
        summary.append({
            "level": f"L{level}",
            "k_columns_dropped": LEVEL_TO_K[level],
            "mae": float(np.mean(abs_err)),
            "mse": float(np.mean(signed_err.astype(np.float64) ** 2)),
            "max_abs_err": int(np.max(abs_err)),
            "max_pos_err": int(np.max(signed_err)),
            "max_neg_err": int(np.min(signed_err)),
            "mean_signed_error": float(np.mean(signed_err)),
            "mred": float(np.mean(rel_err)) if d["rel_err"] else float("nan"),
            "pct_exact_outputs": 100.0 * np.sum(abs_err == 0) / n_samples,
            "pct_nonzero_error": 100.0 * np.sum(abs_err != 0) / n_samples,
        })

    metrics_path = os.path.join(cfg.RESULTS_DIR, "pp_exhaustive_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    # plot: MAE / mean-signed-error / max-abs-error vs level
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    levels = [s["level"] for s in summary]
    ax[0].bar(levels, [s["mae"] for s in summary], color="steelblue")
    ax[0].set_title("MAE by level")
    ax[1].bar(levels, [s["mean_signed_error"] for s in summary], color="firebrick")
    ax[1].set_title("Mean signed error (bias) by level")
    ax[2].bar(levels, [s["max_abs_err"] for s in summary], color="darkorange")
    ax[2].set_title("Max abs error by level")
    fig.suptitle("Exhaustive 8x8 PP-column-truncation characterization")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.PLOTS_DIR, "pp_exhaustive_summary.png"), dpi=150)
    plt.close(fig)

    print("Exhaustive PP-column-truncation characterization (65,536 pairs):")
    for s in summary:
        print(f"  {s['level']} (k={s['k_columns_dropped']}): MAE={s['mae']:.3f} "
              f"MSE={s['mse']:.1f} MaxAbsErr={s['max_abs_err']} "
              f"Bias={s['mean_signed_error']:.4f} MRED={s['mred']:.4f} "
              f"ExactOutputs={s['pct_exact_outputs']:.1f}%")
    print(f"Per-pair results: {csv_path}")
    print(f"Summary metrics: {metrics_path}")
    return summary


if __name__ == "__main__":
    run()
