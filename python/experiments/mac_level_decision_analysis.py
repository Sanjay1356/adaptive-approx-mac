"""
MAC-LEVEL DECISION ANALYSIS

Analyzes the REAL ResNet-18 MAC workload already collected by:

    results/real_dnn_mac_results.csv

This script does NOT:
    - run ResNet-18
    - collect images
    - generate MAC operands
    - modify the multiplier
    - run another profiling experiment

It only analyzes the existing 20,000 real convolution dot products.

Primary metrics:
    - Normalized MAE
    - Normalized absolute bias
    - P50 absolute error
    - P95 absolute error
    - P99 absolute error
    - Exact dot-product percentage
    - Affected dot-product percentage

It also identifies:
    - layers where L2 beats L1
    - layers where L3 exceeds 100% normalized MAE
    - best approximation level per layer
    - 3x3 vs 1x1 behavior

Run:
    python -m experiments.mac_level_decision_analysis
"""

import csv
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import config as cfg


INPUT_FILE = os.path.join(
    cfg.RESULTS_DIR,
    "real_dnn_mac_results.csv"
)

LAYER_OUTPUT = os.path.join(
    cfg.RESULTS_DIR,
    "mac_level_layer_analysis.csv"
)

SUMMARY_OUTPUT = os.path.join(
    cfg.RESULTS_DIR,
    "mac_level_decision_summary.csv"
)

LEVELS = ["L1", "L2", "L3"]


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

def load_results():

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Could not find:\n{INPUT_FILE}\n\n"
            "Run the real DNN MAC profiler first."
        )

    with open(INPUT_FILE, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(
            f"{INPUT_FILE} contains no data."
        )

    required = {
        "layer",
        "sample",
        "K",
        "exact_mac",
        "L1_error",
        "L2_error",
        "L3_error",
    }

    missing = required - set(rows[0].keys())

    if missing:
        raise ValueError(
            "CSV schema is missing required columns:\n"
            + ", ".join(sorted(missing))
        )

    return rows


# ============================================================
# LAYER TYPE
# ============================================================

def infer_layer_type(layer_name, K):

    # ResNet-18 downsample convolutions are 1x1.
    if "downsample" in layer_name:
        return "1x1"

    # K = Cin * kernel_height * kernel_width.
    #
    # For this ResNet-18 workload:
    #   1x1 layers: K = 64, 128, 256
    #   3x3 layers: K = 147, 576, 1152, 2304, 4608
    #
    if K in {64, 128, 256}:
        return "1x1"

    if K in {147, 576, 1152, 2304, 4608}:
        return "3x3"

    return "unknown"


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(rows, level):

    error_column = f"{level}_error"

    errors = np.array(
        [float(row[error_column]) for row in rows],
        dtype=np.float64,
    )

    exact = np.array(
        [float(row["exact_mac"]) for row in rows],
        dtype=np.float64,
    )

    abs_error = np.abs(errors)

    mean_abs_exact = np.mean(np.abs(exact))

    if mean_abs_exact > 0:

        normalized_mae = (
            np.mean(abs_error)
            / mean_abs_exact
            * 100.0
        )

        normalized_bias = (
            abs(np.mean(errors))
            / mean_abs_exact
            * 100.0
        )

    else:

        normalized_mae = np.nan
        normalized_bias = np.nan

    return {
        "MAE": float(np.mean(abs_error)),

        "RMSE": float(
            np.sqrt(np.mean(errors ** 2))
        ),

        "MSE": float(
            np.mean(errors ** 2)
        ),

        "MaxAbsError": float(
            np.max(abs_error)
        ),

        "Bias": float(
            np.mean(errors)
        ),

        "Std": float(
            np.std(errors)
        ),

        "P50AbsError": float(
            np.percentile(abs_error, 50)
        ),

        "P95AbsError": float(
            np.percentile(abs_error, 95)
        ),

        "P99AbsError": float(
            np.percentile(abs_error, 99)
        ),

        "MeanAbsExactMAC": float(
            mean_abs_exact
        ),

        "NormalizedMAE": float(
            normalized_mae
        ),

        "NormalizedAbsBias": float(
            normalized_bias
        ),

        "ExactDotPct": float(
            np.mean(abs_error == 0) * 100.0
        ),

        "AffectedDotPct": float(
            np.mean(abs_error != 0) * 100.0
        ),
    }


# ============================================================
# GROUP BY LAYER
# ============================================================

def group_layers(rows):

    grouped = {}

    for row in rows:

        layer = row["layer"]

        if layer not in grouped:
            grouped[layer] = []

        grouped[layer].append(row)

    return grouped


# ============================================================
# BUILD LAYER TABLE
# ============================================================

def build_layer_analysis(grouped):

    analysis = []

    for layer, rows in grouped.items():

        K = int(rows[0]["K"])

        layer_type = infer_layer_type(
            layer,
            K
        )

        metrics = {}

        for level in LEVELS:

            metrics[level] = calculate_metrics(
                rows,
                level
            )

        # ----------------------------------------------------
        # Determine best approximation level by normalized MAE
        # ----------------------------------------------------

        nmae = {
            level: metrics[level]["NormalizedMAE"]
            for level in LEVELS
        }

        best_level = min(
            nmae,
            key=nmae.get
        )

        # ----------------------------------------------------
        # L2 vs L1
        # ----------------------------------------------------

        l1_nmae = metrics["L1"]["NormalizedMAE"]
        l2_nmae = metrics["L2"]["NormalizedMAE"]

        l2_beats_l1 = (
            l2_nmae < l1_nmae
        )

        improvement_l2_vs_l1 = (
            (l1_nmae - l2_nmae)
            / l1_nmae
            * 100.0
            if l1_nmae != 0
            else 0.0
        )

        # ----------------------------------------------------
        # L3 severity
        # ----------------------------------------------------

        l3_nmae = metrics["L3"]["NormalizedMAE"]

        l3_over_100 = (
            l3_nmae > 100.0
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        result = {
            "layer": layer,
            "layer_type": layer_type,
            "K": K,
            "samples": len(rows),
            "best_level": best_level,

            "L2_beats_L1": l2_beats_l1,

            "L2_improvement_vs_L1_pct":
                improvement_l2_vs_l1,

            "L3_over_100pct_NMAE":
                l3_over_100,
        }

        for level in LEVELS:

            m = metrics[level]

            result[f"{level}_MAE"] = m["MAE"]
            result[f"{level}_RMSE"] = m["RMSE"]
            result[f"{level}_MSE"] = m["MSE"]
            result[f"{level}_MaxAbsError"] = m["MaxAbsError"]
            result[f"{level}_Bias"] = m["Bias"]
            result[f"{level}_Std"] = m["Std"]

            result[f"{level}_P50AbsError"] = (
                m["P50AbsError"]
            )

            result[f"{level}_P95AbsError"] = (
                m["P95AbsError"]
            )

            result[f"{level}_P99AbsError"] = (
                m["P99AbsError"]
            )

            result[f"{level}_MeanAbsExactMAC"] = (
                m["MeanAbsExactMAC"]
            )

            result[f"{level}_NormalizedMAE"] = (
                m["NormalizedMAE"]
            )

            result[f"{level}_NormalizedAbsBias"] = (
                m["NormalizedAbsBias"]
            )

            result[f"{level}_ExactDotPct"] = (
                m["ExactDotPct"]
            )

            result[f"{level}_AffectedDotPct"] = (
                m["AffectedDotPct"]
            )

        analysis.append(result)

    return analysis


# ============================================================
# WRITE CSV
# ============================================================

def write_csv(rows, path):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    if not rows:
        return

    with open(path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# NETWORK SUMMARY
# ============================================================

def build_summary(analysis):

    summary = []

    for level in LEVELS:

        nmae = np.array([
            row[f"{level}_NormalizedMAE"]
            for row in analysis
        ])

        bias = np.array([
            row[f"{level}_NormalizedAbsBias"]
            for row in analysis
        ])

        p95 = np.array([
            row[f"{level}_P95AbsError"]
            for row in analysis
        ])

        exact = np.array([
            row[f"{level}_ExactDotPct"]
            for row in analysis
        ])

        summary.append({

            "level": level,

            "mean_layer_NMAE_pct":
                np.mean(nmae),

            "median_layer_NMAE_pct":
                np.median(nmae),

            "max_layer_NMAE_pct":
                np.max(nmae),

            "mean_layer_abs_bias_pct":
                np.mean(bias),

            "mean_layer_P95_abs_error":
                np.mean(p95),

            "mean_exact_dot_pct":
                np.mean(exact),

            "layers_where_best":
                sum(
                    row["best_level"] == level
                    for row in analysis
                ),

            "layers_where_L3_over_100pct":
                sum(
                    row["L3_over_100pct_NMAE"]
                    for row in analysis
                ),
        })

    return summary


# ============================================================
# PRINT DECISION REPORT
# ============================================================

def print_report(analysis, summary):

    print()
    print("=" * 78)
    print("MAC-LEVEL APPROXIMATION DECISION ANALYSIS")
    print("=" * 78)

    print()
    print(f"Convolution layers : {len(analysis)}")
    print(
        f"Total dot products: "
        f"{sum(row['samples'] for row in analysis)}"
    )

    print()
    print("NETWORK SUMMARY")
    print("-" * 78)

    print(
        f"{'Level':<8}"
        f"{'Mean NMAE':>14}"
        f"{'Median NMAE':>16}"
        f"{'Max NMAE':>14}"
        f"{'Best Layers':>14}"
    )

    for row in summary:

        print(
            f"{row['level']:<8}"
            f"{row['mean_layer_NMAE_pct']:>13.2f}%"
            f"{row['median_layer_NMAE_pct']:>15.2f}%"
            f"{row['max_layer_NMAE_pct']:>13.2f}%"
            f"{row['layers_where_best']:>14}"
        )

    # --------------------------------------------------------
    # L2 beats L1
    # --------------------------------------------------------

    l2_better = [
        row for row in analysis
        if row["L2_beats_L1"]
    ]

    print()
    print("L2 BEATS L1")
    print("-" * 78)

    print(
        f"{len(l2_better)} / {len(analysis)} layers "
        "have lower normalized MAE at L2 than L1."
    )

    if l2_better:

        print()

        for row in sorted(
            l2_better,
            key=lambda x:
                x["L2_improvement_vs_L1_pct"],
            reverse=True
        ):

            print(
                f"{row['layer']:<30}"
                f"L1={row['L1_NormalizedMAE']:.2f}% "
                f"L2={row['L2_NormalizedMAE']:.2f}% "
                f"improvement="
                f"{row['L2_improvement_vs_L1_pct']:.1f}%"
            )

    # --------------------------------------------------------
    # L3
    # --------------------------------------------------------

    l3_bad = [
        row for row in analysis
        if row["L3_over_100pct_NMAE"]
    ]

    print()
    print("L3 SEVERITY")
    print("-" * 78)

    print(
        f"{len(l3_bad)} / {len(analysis)} layers "
        "have L3 normalized MAE > 100%."
    )

    # --------------------------------------------------------
    # Best level
    # --------------------------------------------------------

    print()
    print("BEST LEVEL BY LAYER")
    print("-" * 78)

    for row in analysis:

        print(
            f"{row['layer']:<30}"
            f"K={row['K']:<5} "
            f"type={row['layer_type']:<4} "
            f"best={row['best_level']}"
        )

    # --------------------------------------------------------
    # 3x3 vs 1x1
    # --------------------------------------------------------

    print()
    print("3x3 VS 1x1")
    print("-" * 78)

    for layer_type in ["3x3", "1x1"]:

        subset = [
            row
            for row in analysis
            if row["layer_type"] == layer_type
        ]

        if not subset:
            continue

        print(
            f"\n{layer_type} layers: {len(subset)}"
        )

        for level in LEVELS:

            values = [
                row[f"{level}_NormalizedMAE"]
                for row in subset
            ]

            print(
                f"  {level}: "
                f"mean NMAE="
                f"{np.mean(values):.2f}%"
            )

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print()
    print("INTERPRETATION")
    print("-" * 78)

    if len(l3_bad) >= len(analysis) / 2:

        print(
            "L3 is broadly aggressive: more than half of the "
            "profiled layers exceed 100% normalized MAE."
        )

    if l2_better:

        print(
            "L2 outperforms L1 in some real workload layers. "
            "This confirms that workload-dependent signed error "
            "behavior matters; approximation strength does not "
            "translate into strictly monotonic normalized MAE "
            "for every sampled layer."
        )

    print()
    print(
        "IMPORTANT: These results characterize MAC-level error only."
    )

    print(
        "They do NOT establish final DNN accuracy, energy, area, "
        "or timing."
    )

    print(
        "The next experiment should therefore evaluate the "
        "surviving approximation candidates at full-network level."
    )


# ============================================================
# PLOT 1: NORMALIZED MAE
# ============================================================

def plot_normalized_mae(analysis):

    layers = [
        row["layer"]
        for row in analysis
    ]

    x = np.arange(len(layers))

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    for level in LEVELS:

        y = [
            row[f"{level}_NormalizedMAE"]
            for row in analysis
        ]

        ax.plot(
            x,
            y,
            marker="o",
            label=level
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        layers,
        rotation=75,
        ha="right"
    )

    ax.set_ylabel(
        "Normalized MAE (%)"
    )

    ax.set_title(
        "Real ResNet-18 MAC Workload — "
        "Normalized MAE by Layer"
    )

    ax.legend()
    ax.grid(
        True,
        alpha=0.25
    )

    fig.tight_layout()

    path = os.path.join(
        cfg.PLOTS_DIR,
        "mac_level_normalized_mae.png"
    )

    fig.savefig(
        path,
        dpi=160
    )

    plt.close(fig)


# ============================================================
# PLOT 2: P95 ERROR
# ============================================================

def plot_p95(analysis):

    layers = [
        row["layer"]
        for row in analysis
    ]

    x = np.arange(len(layers))

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    for level in LEVELS:

        y = [
            row[f"{level}_P95AbsError"]
            for row in analysis
        ]

        ax.plot(
            x,
            y,
            marker="o",
            label=level
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        layers,
        rotation=75,
        ha="right"
    )

    ax.set_ylabel(
        "P95 Absolute Error"
    )

    ax.set_title(
        "Real ResNet-18 MAC Workload — "
        "P95 Absolute Error by Layer"
    )

    ax.legend()
    ax.grid(
        True,
        alpha=0.25
    )

    fig.tight_layout()

    path = os.path.join(
        cfg.PLOTS_DIR,
        "mac_level_p95_error.png"
    )

    fig.savefig(
        path,
        dpi=160
    )

    plt.close(fig)


# ============================================================
# PLOT 3: HEATMAP
# ============================================================

def plot_heatmap(analysis):

    layers = [
        row["layer"]
        for row in analysis
    ]

    matrix = np.array([

        [
            row[f"{level}_NormalizedMAE"]
            for level in LEVELS
        ]

        for row in analysis
    ])

    fig, ax = plt.subplots(
        figsize=(
            7,
            max(7, len(layers) * 0.35)
        )
    )

    image = ax.imshow(
        matrix,
        aspect="auto"
    )

    ax.set_xticks(
        np.arange(len(LEVELS))
    )

    ax.set_xticklabels(
        LEVELS
    )

    ax.set_yticks(
        np.arange(len(layers))
    )

    ax.set_yticklabels(
        layers
    )

    ax.set_xlabel(
        "Approximation Level"
    )

    ax.set_ylabel(
        "Convolution Layer"
    )

    ax.set_title(
        "Real ResNet-18 MAC Workload — "
        "Normalized MAE Heatmap"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Normalized MAE (%)"
    )

    fig.tight_layout()

    path = os.path.join(
        cfg.PLOTS_DIR,
        "mac_level_layer_heatmap.png"
    )

    fig.savefig(
        path,
        dpi=160
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        cfg.RESULTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        cfg.PLOTS_DIR,
        exist_ok=True
    )

    print(
        "Loading existing real-DNN MAC results..."
    )

    print(
        f"Input: {INPUT_FILE}"
    )

    rows = load_results()

    print(
        f"Loaded {len(rows)} MAC result rows."
    )

    grouped = group_layers(rows)

    print(
        f"Found {len(grouped)} convolution layers."
    )

    analysis = build_layer_analysis(
        grouped
    )

    summary = build_summary(
        analysis
    )

    write_csv(
        analysis,
        LAYER_OUTPUT
    )

    write_csv(
        summary,
        SUMMARY_OUTPUT
    )

    plot_normalized_mae(
        analysis
    )

    plot_p95(
        analysis
    )

    plot_heatmap(
        analysis
    )

    print_report(
        analysis,
        summary
    )

    print()
    print("=" * 78)
    print("OUTPUT FILES")
    print("=" * 78)

    print(
        f"Layer analysis : {LAYER_OUTPUT}"
    )

    print(
        f"Decision summary: {SUMMARY_OUTPUT}"
    )

    print(
        f"Plot: {cfg.PLOTS_DIR}/mac_level_normalized_mae.png"
    )

    print(
        f"Plot: {cfg.PLOTS_DIR}/mac_level_p95_error.png"
    )

    print(
        f"Plot: {cfg.PLOTS_DIR}/mac_level_layer_heatmap.png"
    )


if __name__ == "__main__":
    main()
