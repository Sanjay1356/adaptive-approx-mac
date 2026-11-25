"""
LAYER-ISOLATION DNN SENSITIVITY EXPERIMENT
===========================================

Answers the question the MAC-level metric CANNOT answer:

    "How much does approximating THIS ONE layer, with every other
     layer held EXACT, change the network's predictions relative to
     the exact/reference network?"

This is the direct measurement Part 5/7 of the audit calls for. It is
built entirely on top of the already-audited infrastructure in
dnn_accuracy_eval.py (ApproxConv2d, ExactInt8Conv2d, replace_convolutions,
build_multiplier_lut, load_model, load_dataset, calculate_weight_scales,
collect_activation_ranges, get_reference_predictions) -- it does NOT
reimplement quantization, convolution, or LUT logic, so it inherits the
LUT-encoding fix and every other correctness property already verified
there. It does NOT replace real_dnn_mac_profile.py or
mac_level_decision_analysis.py -- it answers a different question
(network-level sensitivity, not MAC-level error) that those files were
never designed to answer.

Method:
    For each convolution layer L and each level in {L1, L2, L3}:
        1. Start from a model where EVERY conv is ExactInt8Conv2d
           (i.e. identical to the reference network).
        2. Replace ONLY layer L with ApproxConv2d at that level.
        3. Run inference over the same image set used for the
           reference predictions.
        4. Record: agreement with reference, number of differing
           predictions, mean/median |output logit error|.

This isolates each layer's OWN contribution to prediction change,
uncontaminated by every other layer also being approximated -- which
is what the uniform L1/L2/L3 and current ADAPTIVE numbers conflate.

Recommended sample size: >=500 images (see note in main() about why
100 images gives too much sampling noise to trust small agreement
differences between levels).

Run (from the same directory as dnn_accuracy_eval.py):
    python layer_isolation_sensitivity.py

Output:
    results/layer_isolation_sensitivity.csv
      columns: layer, level, reference_agreement_pct,
               n_differing_predictions, mean_abs_logit_error,
               median_abs_logit_error
    results/layer_isolation_summary.txt
      human-readable per-layer "max tolerable level" table
"""

import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn

from experiments. dnn_accuracy_eval import (
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
from torchvision import models


# ================================================================
# CONFIGURATION
# ================================================================

# NOTE ON SAMPLE SIZE:
# The original 100-image DNN-level evaluation has real statistical
# limitations for THIS experiment. With N=100, the standard error of
# a proportion near p=0.5 is sqrt(0.5*0.5/100) ~= 5 percentage points,
# and worse in relative terms near the extremes (p close to 0 or 1).
# A single-layer perturbation's true effect on agreement could easily
# be a few percent either way -- indistinguishable from noise at N=100.
# Use at least 500 images (matching the N_IMAGES already used for MAC
# collection in real_dnn_mac_profile.py) for any number that will be
# used to freeze the final policy. 100 is left as the default here
# only so the experiment is runnable quickly as a first pass /
# smoke test -- do NOT treat a 100-image run as the final policy input.
NUM_IMAGES = 100

LEVELS_TO_TEST = [1, 2, 3]  # L1, L2, L3 (L0 is the reference itself)

RESULTS_CSV = os.path.join(RESULTS_DIR, "layer_isolation_sensitivity.csv")
SUMMARY_TXT = os.path.join(RESULTS_DIR, "layer_isolation_summary.txt")


# ================================================================
# BUILD A MODEL WITH ONE LAYER APPROXIMATED, EVERYTHING ELSE EXACT
# ================================================================

def build_single_layer_approx_model(target_layer, level, lut,
                                     weight_scales, activation_scales):
    """Every conv is ExactInt8Conv2d EXCEPT target_layer, which is
    ApproxConv2d at the given level. This isolates that one layer's
    contribution to prediction change."""
    model = load_model()
    model = replace_convolutions_exact_int8(
        model=model,
        weight_scales=weight_scales,
        activation_scales=activation_scales,
    )

    # Now swap ONLY the target layer to the approximate version.
    parts = target_layer.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    child_name = parts[-1]

    original_exact_conv = getattr(parent, child_name)

    # ApproxConv2d needs the ORIGINAL nn.Conv2d (for stride/padding/etc
    # and unquantized weights) -- rebuild it from the pristine model
    # rather than from the already-quantized ExactInt8Conv2d.
    pristine = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    pristine_conv = pristine
    for part in parts[:-1]:
        pristine_conv = getattr(pristine_conv, part)
    pristine_conv = getattr(pristine_conv, child_name)

    level_name = f"L{level}"
    replacement = ApproxConv2d(
        original_conv=pristine_conv,
        layer_name=target_layer,
        level_name=level_name,
        lut=lut,
        weight_scale=weight_scales[target_layer],
        activation_scale=activation_scales[target_layer],
    )

    setattr(parent, child_name, replacement)
    del original_exact_conv
    return model


# ================================================================
# EVALUATE ONE (layer, level) COMBINATION
# ================================================================

@torch.no_grad()
def evaluate_single_layer(model, dataset, num_images,
                           reference_predictions, reference_logits):
    model.eval()
    predictions = []
    logit_errors = []

    for i in range(num_images):
        image, _ = dataset[i]
        image = image.unsqueeze(0).to(DEVICE)
        output = model(image)

        pred = int(torch.argmax(output, dim=1).item())
        predictions.append(pred)

        logit_err = torch.abs(output - reference_logits[i]).mean().item()
        logit_errors.append(logit_err)

    predictions = np.array(predictions)
    logit_errors = np.array(logit_errors)

    agreement = 100.0 * np.mean(predictions == reference_predictions)
    n_differing = int(np.sum(predictions != reference_predictions))

    return {
        "reference_agreement_pct": agreement,
        "n_differing_predictions": n_differing,
        "mean_abs_logit_error": float(np.mean(logit_errors)),
        "median_abs_logit_error": float(np.median(logit_errors)),
    }


# ================================================================
# MAIN
# ================================================================

def main():
    print()
    print("=" * 72)
    print("LAYER-ISOLATION DNN SENSITIVITY EXPERIMENT")
    print("=" * 72)
    print(f"Images: {NUM_IMAGES}")
    if NUM_IMAGES < 500:
        print()
        print("WARNING: NUM_IMAGES < 500. Per-level agreement differences")
        print("of a few percent are within sampling noise at this size.")
        print("Treat this run as a smoke test, not a final policy input.")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    original_model = load_model()
    weights = models.ResNet18_Weights.DEFAULT
    dataset = load_dataset(weights)

    conv_layers = [
        name for name, module in original_model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    print(f"Found {len(conv_layers)} convolution layers.")

    weight_scales = calculate_weight_scales(original_model)
    activation_scales = collect_activation_ranges(original_model, dataset, NUM_IMAGES)

    lut = build_multiplier_lut()

    # Reference: every conv exact, using the SAME scales as the
    # approximate models (isolates approximation error from ordinary
    # quantization error -- consistent with the already-audited design).
    reference_model = load_model()
    reference_model = replace_convolutions_exact_int8(
        model=reference_model,
        weight_scales=weight_scales,
        activation_scales=activation_scales,
    )

    reference_predictions, labels = get_reference_predictions(
        reference_model, dataset, NUM_IMAGES
    )

    # Also capture reference logits per image for the logit-error metric.
    reference_logits = []
    with torch.no_grad():
        for i in range(NUM_IMAGES):
            image, _ = dataset[i]
            image = image.unsqueeze(0).to(DEVICE)
            reference_logits.append(reference_model(image))
    del reference_model

    rows = []
    start_all = time.time()

    for idx, layer_name in enumerate(conv_layers, start=1):
        for level in LEVELS_TO_TEST:
            print(f"[{idx}/{len(conv_layers)}] {layer_name} -> L{level} ...")
            model = build_single_layer_approx_model(
                layer_name, level, lut, weight_scales, activation_scales
            )
            metrics = evaluate_single_layer(
                model, dataset, NUM_IMAGES, reference_predictions, reference_logits
            )
            rows.append({
                "layer": layer_name,
                "level": f"L{level}",
                **metrics,
            })
            print(
                f"    agreement={metrics['reference_agreement_pct']:.2f}%  "
                f"differing={metrics['n_differing_predictions']}  "
                f"mean_abs_logit_err={metrics['mean_abs_logit_error']:.4f}"
            )
            del model

    elapsed = time.time() - start_all
    print(f"\nTotal time: {elapsed:.1f}s")

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {RESULTS_CSV}")

    # ------------------------------------------------------------
    # Derive "max tolerable level per layer" using an explicit,
    # justified constraint. Constraint used here (state clearly in
    # the PPT): the strongest level whose ISOLATED reference
    # agreement is >= 99%, i.e. isolating that layer's approximation
    # changes at most 1% of predictions on its own. This threshold is
    # a starting point, not a law -- if your final images-per-run
    # count changes, recompute rather than reuse this number blindly,
    # since 1% of N images is only meaningful once N is large enough
    # that 1% corresponds to more than sampling noise (again, N>=500
    # recommended).
    # ------------------------------------------------------------
    THRESHOLD_PCT = 99.0

    by_layer = {name: {} for name in conv_layers}
    for row in rows:
        by_layer[row["layer"]][row["level"]] = row["reference_agreement_pct"]

    lines = []
    lines.append(f"Per-layer isolation sensitivity (threshold: agreement >= {THRESHOLD_PCT}%)")
    lines.append("=" * 78)
    for layer_name in conv_layers:
        levels = by_layer[layer_name]
        chosen = "L0"
        for level in ["L1", "L2", "L3"]:
            if levels.get(level, 0.0) >= THRESHOLD_PCT:
                chosen = level
            else:
                break
        line = f"{layer_name:<30} " + " ".join(
            f"{lvl}={levels.get(lvl, float('nan')):.1f}%" for lvl in ["L1", "L2", "L3"]
        ) + f"  -> max tolerable: {chosen}"
        lines.append(line)
        print(line)

    with open(SUMMARY_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
