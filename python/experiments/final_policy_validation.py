"""
FINAL POLICY VALIDATION
=======================

Purpose
-------
Validate the candidate layer-wise adaptive approximation policy derived
from the layer-isolation sensitivity experiment.

The policy is NOT hard-coded. It is derived from:

    results/layer_isolation_sensitivity.csv

For every convolution layer, choose the strongest level among
L1/L2/L3 whose isolated reference agreement is >= 90%.

If no approximate level satisfies the threshold, use L0.

Then compare:

    L0       : all convolution layers exact
    L1       : all convolution layers L1
    L2       : all convolution layers L2
    L3       : all convolution layers L3
    ADAPTIVE : layer-wise policy derived above

IMPORTANT
---------
This experiment validates the candidate policy. With 100 images,
the result is a smoke/validation run, not a statistically final result.
"""

import csv
import os
import time

import numpy as np
import pandas as pd
import torch

from torchvision import models

# Reuse the already-corrected DNN evaluation infrastructure.
#
# IMPORTANT:
# These imports intentionally come from the existing implementation.
# We do NOT duplicate quantization, convolution, LUT, or multiplier
# logic here.
from experiments.dnn_accuracy_eval import (
    DEVICE,
    RESULTS_DIR,
    load_model,
    load_dataset,
    calculate_weight_scales,
    collect_activation_ranges,
    build_multiplier_lut,
    replace_convolutions_exact_int8,
    get_reference_predictions,
    ApproxConv2d,
)


# ================================================================
# CONFIGURATION
# ================================================================

NUM_IMAGES = 100

POLICY_THRESHOLD = 90.0

ISOLATION_CSV = os.path.join(
    RESULTS_DIR,
    "layer_isolation_sensitivity.csv",
)

OUTPUT_CSV = os.path.join(
    RESULTS_DIR,
    "final_policy_validation.csv",
)

OUTPUT_SUMMARY = os.path.join(
    RESULTS_DIR,
    "final_policy_summary.txt",
)

# Number of PP cells in an 8x8 multiplier.
TOTAL_PP_CELLS = 64

# Number of PP cells removed by each approximation level.
#
# k = 0 -> 0 dropped cells
# k = 2 -> 3 dropped cells
# k = 4 -> 10 dropped cells
# k = 6 -> 21 dropped cells
PP_CELLS_DROPPED = {
    "L0": 0,
    "L1": 3,
    "L2": 10,
    "L3": 21,
}


# ================================================================
# LOAD ISOLATION RESULTS
# ================================================================

def load_isolation_results():
    if not os.path.exists(ISOLATION_CSV):
        raise FileNotFoundError(
            f"Missing layer-isolation results:\n"
            f"  {ISOLATION_CSV}\n\n"
            f"Run the layer-isolation experiment first."
        )

    df = pd.read_csv(ISOLATION_CSV)

    required_columns = {
        "layer",
        "level",
        "reference_agreement_pct",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Isolation CSV is missing columns: {sorted(missing)}"
        )

    return df


# ================================================================
# DERIVE ADAPTIVE POLICY
# ================================================================

def derive_adaptive_policy(df, threshold):
    """
    For each layer choose the strongest approximation level whose
    isolated reference agreement meets the threshold.

    Priority:

        L3 > L2 > L1 > L0

    If no approximate level satisfies the threshold, choose L0.
    """

    layers = list(df["layer"].drop_duplicates())

    policy = {}

    for layer in layers:

        layer_df = df[df["layer"] == layer]

        agreement = {}

        for level in ["L1", "L2", "L3"]:
            matches = layer_df[layer_df["level"] == level]

            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one {level} result for "
                    f"layer '{layer}', found {len(matches)}"
                )

            agreement[level] = float(
                matches.iloc[0]["reference_agreement_pct"]
            )

        chosen = "L0"

        # Strongest acceptable approximation wins.
        for level in ["L1", "L2", "L3"]:
            if agreement[level] >= threshold:
                chosen = level

        policy[layer] = chosen

    return policy


# ================================================================
# PRINT POLICY
# ================================================================

def print_policy(policy, df):

    print()
    print("=" * 72)
    print("CANDIDATE ADAPTIVE POLICY")
    print("=" * 72)

    print(
        f"Selection rule: strongest level with isolated agreement "
        f">= {POLICY_THRESHOLD:.1f}%"
    )

    print()

    for layer, level in policy.items():

        layer_df = df[df["layer"] == layer]

        values = {}

        for l in ["L1", "L2", "L3"]:
            values[l] = float(
                layer_df[layer_df["level"] == l]
                ["reference_agreement_pct"]
                .iloc[0]
            )

        print(
            f"{layer:<30} "
            f"L1={values['L1']:5.1f}% "
            f"L2={values['L2']:5.1f}% "
            f"L3={values['L3']:5.1f}% "
            f"-> {level}"
        )

    print()


# ================================================================
# POLICY STATISTICS
# ================================================================

def policy_statistics(policy):

    counts = {
        "L0": 0,
        "L1": 0,
        "L2": 0,
        "L3": 0,
    }

    for level in policy.values():
        counts[level] += 1

    total_layers = len(policy)

    total_pp_cells = total_layers * TOTAL_PP_CELLS

    dropped_pp_cells = sum(
        PP_CELLS_DROPPED[level]
        for level in policy.values()
    )

    reduction_pct = (
        100.0 * dropped_pp_cells / total_pp_cells
        if total_pp_cells > 0
        else 0.0
    )

    return {
        "counts": counts,
        "total_layers": total_layers,
        "total_pp_cells": total_pp_cells,
        "dropped_pp_cells": dropped_pp_cells,
        "pp_reduction_pct": reduction_pct,
    }


# ================================================================
# BUILD MODEL
# ================================================================

def build_policy_model(
    policy,
    lut,
    weight_scales,
    activation_scales,
):
    """
    Construct a complete ResNet-18 where each convolution layer uses
    the level specified by policy.
    """

    model = load_model()

    # First convert every convolution to the exact INT8 reference
    # implementation.
    model = replace_convolutions_exact_int8(
        model=model,
        weight_scales=weight_scales,
        activation_scales=activation_scales,
    )

    # If the policy is completely exact, we're done.
    if all(level == "L0" for level in policy.values()):
        return model

    # Load pristine weights once for reconstructing ApproxConv2d layers.
    pristine = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    for layer_name, level in policy.items():

        if level == "L0":
            continue

        # Locate layer in the exact model.
        parts = layer_name.split(".")

        parent = model

        for part in parts[:-1]:
            parent = getattr(parent, part)

        child_name = parts[-1]

        # Locate corresponding original Conv2d.
        pristine_parent = pristine

        for part in parts[:-1]:
            pristine_parent = getattr(pristine_parent, part)

        pristine_conv = getattr(
            pristine_parent,
            child_name,
        )

        replacement = ApproxConv2d(
            original_conv=pristine_conv,
            layer_name=layer_name,
            level_name=level,
            lut=lut,
            weight_scale=weight_scales[layer_name],
            activation_scale=activation_scales[layer_name],
        )

        setattr(
            parent,
            child_name,
            replacement,
        )

    del pristine

    return model


# ================================================================
# UNIFORM MODEL
# ================================================================

def build_uniform_model(
    level,
    lut,
    weight_scales,
    activation_scales,
):
    """
    Build a model in which every convolution uses the same level.
    """

    model = load_model()

    model = replace_convolutions_exact_int8(
        model=model,
        weight_scales=weight_scales,
        activation_scales=activation_scales,
    )

    if level == "L0":
        return model

    pristine = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    for layer_name in weight_scales.keys():

        parts = layer_name.split(".")

        parent = model

        for part in parts[:-1]:
            parent = getattr(parent, part)

        child_name = parts[-1]

        pristine_parent = pristine

        for part in parts[:-1]:
            pristine_parent = getattr(
                pristine_parent,
                part,
            )

        pristine_conv = getattr(
            pristine_parent,
            child_name,
        )

        replacement = ApproxConv2d(
            original_conv=pristine_conv,
            layer_name=layer_name,
            level_name=level,
            lut=lut,
            weight_scale=weight_scales[layer_name],
            activation_scale=activation_scales[layer_name],
        )

        setattr(
            parent,
            child_name,
            replacement,
        )

    del pristine

    return model


# ================================================================
# EVALUATION
# ================================================================

@torch.no_grad()
def evaluate_model(
    model,
    dataset,
    reference_predictions,
    num_images,
    policy_name,
):
    model.eval()

    predictions = []

    start = time.time()

    for i in range(num_images):

        image, _ = dataset[i]

        image = image.unsqueeze(0).to(DEVICE)

        output = model(image)

        pred = int(
            torch.argmax(
                output,
                dim=1,
            ).item()
        )

        predictions.append(pred)

        if (i + 1) % 10 == 0:
            print(
                f"{policy_name}: "
                f"{i + 1}/{num_images}"
            )

    elapsed = time.time() - start

    predictions = np.asarray(predictions)

    reference_predictions = np.asarray(
        reference_predictions
    )

    agreement = 100.0 * np.mean(
        predictions == reference_predictions
    )

    differing = int(
        np.sum(
            predictions != reference_predictions
        )
    )

    return {
        "policy": policy_name,
        "reference_agreement_pct": float(agreement),
        "n_differing_predictions": differing,
        "inference_time_s": float(elapsed),
    }


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print("=" * 72)
    print("FINAL ADAPTIVE POLICY VALIDATION")
    print("=" * 72)

    print(f"Device                 : {DEVICE}")
    print(f"Images                 : {NUM_IMAGES}")
    print(
        f"Policy threshold       : "
        f"{POLICY_THRESHOLD:.1f}% isolated agreement"
    )

    if NUM_IMAGES < 500:
        print()
        print("WARNING")
        print("-------")
        print(
            "This is a 100-image candidate-policy validation."
        )
        print(
            "Do NOT treat small percentage differences as "
            "statistically final."
        )
        print()

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # 1. Read layer-isolation results
    # ------------------------------------------------------------

    print()
    print("Loading layer-isolation results...")
    print(ISOLATION_CSV)

    isolation_df = load_isolation_results()

    policy = derive_adaptive_policy(
        isolation_df,
        POLICY_THRESHOLD,
    )

    print_policy(
        policy,
        isolation_df,
    )

    stats = policy_statistics(policy)

    print("=" * 72)
    print("ADAPTIVE POLICY SUMMARY")
    print("=" * 72)

    counts = stats["counts"]

    print(f"L0 layers : {counts['L0']}")
    print(f"L1 layers : {counts['L1']}")
    print(f"L2 layers : {counts['L2']}")
    print(f"L3 layers : {counts['L3']}")

    print(
        f"PP cells removed : "
        f"{stats['dropped_pp_cells']} / "
        f"{stats['total_pp_cells']}"
    )

    print(
        f"PP-cell reduction: "
        f"{stats['pp_reduction_pct']:.2f}%"
    )

    # ------------------------------------------------------------
    # 2. Load model and data
    # ------------------------------------------------------------

    print()
    print("Loading ResNet-18...")
    model_for_scales = load_model()

    weights = models.ResNet18_Weights.DEFAULT

    print("Loading dataset...")
    dataset = load_dataset(weights)

    conv_layers = [
        name
        for name, module
        in model_for_scales.named_modules()
        if isinstance(module, torch.nn.Conv2d)
    ]

    print(
        f"Found {len(conv_layers)} convolution layers."
    )

    if set(conv_layers) != set(policy.keys()):
        missing = set(conv_layers) - set(policy.keys())
        extra = set(policy.keys()) - set(conv_layers)

        raise RuntimeError(
            "Layer-policy mismatch.\n"
            f"Missing policy layers: {sorted(missing)}\n"
            f"Unknown policy layers: {sorted(extra)}"
        )

    # ------------------------------------------------------------
    # 3. Quantization calibration
    # ------------------------------------------------------------

    print()
    print("Calculating fixed layer-wide weight scales...")

    weight_scales = calculate_weight_scales(
        model_for_scales
    )

    print()
    print("Collecting activation ranges...")

    activation_scales = collect_activation_ranges(
        model_for_scales,
        dataset,
        NUM_IMAGES,
    )

    del model_for_scales

    # ------------------------------------------------------------
    # 4. Build multiplier LUT
    # ------------------------------------------------------------

    print()
    print("Building INT8 multiplier LUT...")

    lut = build_multiplier_lut()

    # ------------------------------------------------------------
    # 5. Reference model
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("REFERENCE MODEL")
    print("=" * 72)

    reference_model = load_model()

    reference_model = replace_convolutions_exact_int8(
        model=reference_model,
        weight_scales=weight_scales,
        activation_scales=activation_scales,
    )

    reference_predictions, labels = get_reference_predictions(
        reference_model,
        dataset,
        NUM_IMAGES,
    )

    del reference_model

    # ------------------------------------------------------------
    # 6. Evaluate policies
    # ------------------------------------------------------------

    results = []

    policies = {
        "L0": "L0",
        "L1": "L1",
        "L2": "L2",
        "L3": "L3",
    }

    # ------------------------------------------------------------
    # Uniform policies
    # ------------------------------------------------------------

    for policy_name, level in policies.items():

        print()
        print("=" * 72)
        print(f"UNIFORM {policy_name}")
        print("=" * 72)

        model = build_uniform_model(
            level,
            lut,
            weight_scales,
            activation_scales,
        )

        result = evaluate_model(
            model,
            dataset,
            reference_predictions,
            NUM_IMAGES,
            policy_name,
        )

        result["pp_cells_dropped_per_multiplier"] = (
            PP_CELLS_DROPPED[level]
        )

        result["total_pp_cells_dropped"] = (
            PP_CELLS_DROPPED[level]
            * len(conv_layers)
        )

        result["pp_cell_reduction_pct"] = (
            100.0
            * result["total_pp_cells_dropped"]
            / (TOTAL_PP_CELLS * len(conv_layers))
        )

        results.append(result)

        del model

    # ------------------------------------------------------------
    # Adaptive policy
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("ADAPTIVE POLICY")
    print("=" * 72)

    adaptive_model = build_policy_model(
        policy,
        lut,
        weight_scales,
        activation_scales,
    )

    adaptive_result = evaluate_model(
        adaptive_model,
        dataset,
        reference_predictions,
        NUM_IMAGES,
        "ADAPTIVE",
    )

    adaptive_result[
        "pp_cells_dropped_per_multiplier"
    ] = np.nan

    adaptive_result[
        "total_pp_cells_dropped"
    ] = stats["dropped_pp_cells"]

    adaptive_result[
        "pp_cell_reduction_pct"
    ] = stats["pp_reduction_pct"]

    results.append(adaptive_result)

    del adaptive_model

    # ------------------------------------------------------------
    # 7. Print final table
    # ------------------------------------------------------------

    result_df = pd.DataFrame(results)

    print()
    print("=" * 72)
    print("FINAL POLICY COMPARISON")
    print("=" * 72)

    print()

    print(
        result_df[
            [
                "policy",
                "reference_agreement_pct",
                "n_differing_predictions",
                "pp_cell_reduction_pct",
                "inference_time_s",
            ]
        ].to_string(
            index=False,
            formatters={
                "reference_agreement_pct":
                    lambda x: f"{x:.2f}%",
                "pp_cell_reduction_pct":
                    lambda x: f"{x:.2f}%",
                "inference_time_s":
                    lambda x: f"{x:.2f}",
            },
        )
    )

    # ------------------------------------------------------------
    # 8. Save CSV
    # ------------------------------------------------------------

    result_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print(f"Saved: {OUTPUT_CSV}")

    # ------------------------------------------------------------
    # 9. Save policy + summary
    # ------------------------------------------------------------

    with open(
        OUTPUT_SUMMARY,
        "w",
    ) as f:

        f.write(
            "FINAL ADAPTIVE POLICY VALIDATION\n"
        )
        f.write(
            "=" * 72 + "\n\n"
        )

        f.write(
            f"Images: {NUM_IMAGES}\n"
        )

        f.write(
            f"Isolation threshold: "
            f"{POLICY_THRESHOLD:.1f}%\n\n"
        )

        f.write(
            "ADAPTIVE POLICY\n"
        )
        f.write(
            "-" * 72 + "\n"
        )

        for layer, level in policy.items():
            f.write(
                f"{layer:<30} -> {level}\n"
            )

        f.write("\n")

        f.write(
            "POLICY COUNTS\n"
        )
        f.write(
            "-" * 72 + "\n"
        )

        for level in ["L0", "L1", "L2", "L3"]:
            f.write(
                f"{level}: "
                f"{counts[level]} layers\n"
            )

        f.write("\n")

        f.write(
            f"PP cells removed: "
            f"{stats['dropped_pp_cells']} / "
            f"{stats['total_pp_cells']}\n"
        )

        f.write(
            f"PP-cell reduction: "
            f"{stats['pp_reduction_pct']:.2f}%\n\n"
        )

        f.write(
            "NETWORK RESULTS\n"
        )
        f.write(
            "-" * 72 + "\n"
        )

        f.write(
            result_df.to_string(
                index=False
            )
        )

        f.write("\n\n")

        f.write(
            "IMPORTANT LIMITATIONS\n"
        )
        f.write(
            "-" * 72 + "\n"
        )

        f.write(
            "1. This run uses 100 images and is therefore "
            "candidate-policy validation, not final statistical "
            "validation.\n"
        )

        f.write(
            "2. Reference agreement measures agreement with the "
            "exact INT8 reference network.\n"
        )

        f.write(
            "3. The CIFAR-10 accuracy remains diagnostic because "
            "the model uses an ImageNet 1000-class classifier.\n"
        )

        f.write(
            "4. PP-cell reduction is a structural multiplier-level "
            "estimate, not synthesized chip area or energy savings.\n"
        )

    print(
        f"Saved: {OUTPUT_SUMMARY}"
    )

    print()
    print("=" * 72)
    print("VALIDATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
