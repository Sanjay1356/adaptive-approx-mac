"""
Compare each approximation level/candidate against the L0 (exact) reference.

Metrics: MAE, MSE, Max Abs Error, MRED (with a defensible zero-reference
policy), exact-output percentage, non-zero-error percentage.

Plots: error histogram, approx-vs-exact scatter, error-vs-operand-magnitude.

Run from adaptive_approx_mac/:  python -m experiments.error_characterization
"""
import csv
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import config as cfg
from models.multiplier import golden_multiply
from models.approximation import CANDIDATES, RECOMMENDED_CANDIDATE


def relative_error(exact, approx):
    """MRED-style relative error with an explicit zero-reference policy:
    when exact==0, relative error is undefined (0/0 or x/0). Rather than
    silently dividing by zero (NaN/inf) or silently skipping the sample
    (which would hide real approximation error at zero), we report it
    separately: return None so callers can exclude it from MRED but still
    count it under absolute-error metrics."""
    if exact == 0:
        return None
    return abs(exact - approx) / abs(exact)


def characterize(candidate_name, n_samples, seed):
    fn = CANDIDATES[candidate_name]
    rng = random.Random(seed)

    rows = []  # for CSV export
    per_level = {lvl: {"abs_err": [], "signed_err": [], "rel_err": [],
                        "exact_vals": [], "approx_vals": []} for lvl in range(4)}

    for _ in range(n_samples):
        a = rng.randint(cfg.OPERAND_MIN, cfg.OPERAND_MAX)
        b = rng.randint(cfg.OPERAND_MIN, cfg.OPERAND_MAX)
        exact = golden_multiply(a, b)
        row = {"a": a, "b": b, "exact": exact}
        for level in range(4):
            approx = fn(a, b, level)
            err = exact - approx
            per_level[level]["abs_err"].append(abs(err))
            per_level[level]["signed_err"].append(err)
            per_level[level]["exact_vals"].append(exact)
            per_level[level]["approx_vals"].append(approx)
            rel = relative_error(exact, approx)
            if rel is not None:
                per_level[level]["rel_err"].append(rel)
            row[f"L{level}_approx"] = approx
            row[f"L{level}_abs_err"] = abs(err)
        rows.append(row)

    return rows, per_level


def summarize(per_level, n_samples):
    summary = []
    for level in range(4):
        d = per_level[level]
        abs_err = np.array(d["abs_err"])
        signed_err = np.array(d["signed_err"])
        rel_err = np.array(d["rel_err"]) if d["rel_err"] else np.array([np.nan])
        n_zero_ref = n_samples - len(d["rel_err"])
        exact_pct = 100.0 * np.sum(abs_err == 0) / n_samples
        nonzero_err_pct = 100.0 * np.sum(abs_err != 0) / n_samples
        summary.append({
            "level": f"L{level}",
            "mae": float(np.mean(abs_err)),
            "mse": float(np.mean(signed_err.astype(np.float64) ** 2)),
            "max_abs_err": int(np.max(abs_err)),
            "mred": float(np.mean(rel_err)) if d["rel_err"] else float("nan"),
            "n_zero_reference_excluded_from_mred": n_zero_ref,
            "pct_exact_outputs": exact_pct,
            "pct_nonzero_error_outputs": nonzero_err_pct,
            "mean_signed_error": float(np.mean(signed_err)),  # bias indicator
        })
    return summary


def write_csv_summary(summary, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def write_plots(per_level, plots_dir, candidate_name):
    os.makedirs(plots_dir, exist_ok=True)

    # 1. absolute-error histogram, one subplot per level
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for level in range(4):
        axes[level].hist(per_level[level]["abs_err"], bins=40, color="steelblue")
        axes[level].set_title(f"L{level} abs error")
        axes[level].set_xlabel("|exact - approx|")
    axes[0].set_ylabel("count")
    fig.suptitle(f"Absolute error histograms ({candidate_name})")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "error_histogram.png"), dpi=150)
    plt.close(fig)

    # 2. approx vs exact scatter (L3, worst case, most informative)
    fig, ax = plt.subplots(figsize=(6, 6))
    exact_vals = np.array(per_level[3]["exact_vals"])
    approx_vals = np.array(per_level[3]["approx_vals"])
    ax.scatter(exact_vals, approx_vals, s=2, alpha=0.3)
    lims = [exact_vals.min(), exact_vals.max()]
    ax.plot(lims, lims, "r--", linewidth=1, label="ideal (y=x)")
    ax.set_xlabel("exact product")
    ax.set_ylabel("approximate product (L3)")
    ax.set_title(f"Approx vs exact output, L3 ({candidate_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "output_comparison.png"), dpi=150)
    plt.close(fig)

    # 3. error vs operand magnitude (uses |a|+|b| as a magnitude proxy), L2
    fig, ax = plt.subplots(figsize=(6, 5))
    # recompute magnitude alongside stored per-level arrays isn't stored;
    # approximate with exact_vals magnitude as a proxy for operand magnitude
    mags = np.abs(np.array(per_level[2]["exact_vals"]))
    errs = np.array(per_level[2]["abs_err"])
    ax.scatter(mags, errs, s=2, alpha=0.3, color="darkorange")
    ax.set_xlabel("|exact product| (magnitude proxy)")
    ax.set_ylabel("absolute error")
    ax.set_title(f"Error vs magnitude, L2 ({candidate_name})")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "error_vs_magnitude.png"), dpi=150)
    plt.close(fig)


def run(candidate_name=RECOMMENDED_CANDIDATE):
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    os.makedirs(cfg.PLOTS_DIR, exist_ok=True)

    rows, per_level = characterize(candidate_name, cfg.N_RANDOM_ERROR_SAMPLES, cfg.RANDOM_SEED)
    summary = summarize(per_level, cfg.N_RANDOM_ERROR_SAMPLES)

    metrics_path = os.path.join(cfg.RESULTS_DIR, "arithmetic_metrics.csv")
    write_csv_summary(summary, metrics_path)

    write_plots(per_level, cfg.PLOTS_DIR, candidate_name)

    print(f"Candidate mechanism: {candidate_name}")
    print(f"Samples per level: {cfg.N_RANDOM_ERROR_SAMPLES}")
    for s in summary:
        print(
            f"  {s['level']}: MAE={s['mae']:.2f} MSE={s['mse']:.1f} "
            f"MaxAbsErr={s['max_abs_err']} MRED={s['mred']:.4f} "
            f"ExactOutputs={s['pct_exact_outputs']:.1f}% "
            f"MeanSignedErr={s['mean_signed_error']:.3f} "
            f"(zero-ref excluded from MRED: {s['n_zero_reference_excluded_from_mred']})"
        )
    print(f"Metrics written to {metrics_path}")
    print(f"Plots written to {cfg.PLOTS_DIR}/")


if __name__ == "__main__":
    run()
