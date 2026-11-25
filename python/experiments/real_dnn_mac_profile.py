"""
REAL RESNET-18 MAC-LEVEL WORKLOAD PROFILER — FULL PASS

Purpose
-------
Characterize the V3 approximate MAC on REAL INT8 ResNet-18
convolution workloads.

This is a MAC-level experiment.
It is NOT a multiplier characterization experiment.

Pipeline:

    CIFAR-10 workload images
            |
            v
    ImageNet-pretrained ResNet-18
            |
            v
    real Conv2D input activations
            |
            v
    fixed layer-wide INT8 activation quantization
            |
            v
    fixed layer-wide INT8 weight quantization
            |
            v
    real convolution dot products
            |
            +--------------------+
            |                    |
            v                    v
      Exact INT8 MAC       Approximate INT8 MAC
                               L0/L1/L2/L3
            |                    |
            +---------+----------+
                      |
                      v
             MAC-level error

Approximation:
    L0 = exact
    L1 = drop PP columns 0-1   (k=2)
    L2 = drop PP columns 0-3   (k=4)
    L3 = drop PP columns 0-5   (k=6)

Reference:
    Exact INT8 MAC

Important:
    Quantization scales are calculated ONCE per layer and then
    reused for every sampled dot product from that layer.

    We do NOT calculate a new scale for every dot product.
"""

import csv
import hashlib
import os
import random

import numpy as np
import torch
from torch import nn
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights

from config import config as cfg
from models.exact_mac import exact_mac_sequence
from models.approximation import approx_mac_step


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

N_IMAGES = 500

# Number of actual convolution dot products sampled from each
# convolution layer.
DOT_PRODUCTS_PER_LAYER = 1000

SEED = cfg.RANDOM_SEED


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 70)
print("REAL RESNET-18 MAC-LEVEL WORKLOAD PROFILING")
print("=" * 70)

print()
print("Loading ImageNet-pretrained ResNet-18...")

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()

preprocess = weights.transforms()

print("Model loaded.")


# ============================================================
# DATASET
# ============================================================

dataset = datasets.CIFAR10(
    root=os.path.expanduser("~/datasets"),
    train=False,
    download=False,
)

print(
    f"Using {N_IMAGES} CIFAR-10 test images "
    "as workload stimulus."
)


# ============================================================
# FIND ALL CONVOLUTION LAYERS
# ============================================================

layers = {}

for name, module in model.named_modules():

    if isinstance(module, nn.Conv2d):
        layers[name] = module


print()
print(f"Found {len(layers)} convolution layers.")

print()
print("Convolution structure:")

for name, layer in layers.items():

    cin = layer.in_channels
    kh, kw = layer.kernel_size

    K = cin * kh * kw

    print(
        f"{name}: "
        f"Cin={cin}, "
        f"Cout={layer.out_channels}, "
        f"kernel={kh}x{kw}, "
        f"stride={layer.stride}, "
        f"K={K}"
    )


# ============================================================
# INT8 QUANTIZATION
# ============================================================

def symmetric_scale(tensor):
    """
    Symmetric INT8 scale:

        scale = max(|x|) / 127

    Zero-point = 0.
    """

    max_abs = tensor.abs().max().item()

    if max_abs == 0:
        return 1.0

    return max_abs / 127.0


def quantize_int8(tensor, scale):
    """
    Quantize using a FIXED scale.
    """

    q = torch.round(tensor / scale)

    q = torch.clamp(
        q,
        -128,
        127
    )

    return q.to(torch.int8)


# ============================================================
# WEIGHT QUANTIZATION
# ============================================================

print()
print("=" * 70)
print("BUILDING FIXED LAYER-WIDE INT8 WEIGHTS")
print("=" * 70)

weight_int8 = {}
weight_scales = {}

for name, layer in layers.items():

    w_fp32 = layer.weight.detach().cpu().float()

    scale = symmetric_scale(w_fp32)

    w_q = quantize_int8(
        w_fp32,
        scale
    )

    weight_scales[name] = scale
    weight_int8[name] = w_q

    print(
        f"{name:25s} "
        f"scale={scale:.10f} "
        f"range=[{int(w_q.min())}, {int(w_q.max())}] "
        f"zero={((w_q == 0).float().mean().item() * 100):.2f}%"
    )


# ============================================================
# ACTIVATION STORAGE
# ============================================================

activation_samples = {
    name: []
    for name in layers
}


def make_hook(name):
    """
    Capture the REAL FP32 input tensor of every convolution.

    This input tensor contains the activation values that
    participate in the convolution MACs.
    """

    def hook(module, inputs, output):

        x = inputs[0].detach().cpu()

        activation_samples[name].append(x)

    return hook


hooks = []

for name, layer in layers.items():

    hooks.append(
        layer.register_forward_hook(
            make_hook(name)
        )
    )


# ============================================================
# RUN REAL WORKLOAD
# ============================================================

print()
print("=" * 70)
print(
    f"COLLECTING REAL ACTIVATIONS FROM {N_IMAGES} IMAGES"
)
print("=" * 70)

with torch.no_grad():

    for i in range(N_IMAGES):

        image, _ = dataset[i]

        image = preprocess(image)

        image = image.unsqueeze(0)

        model(image)

        if (i + 1) % 50 == 0:

            print(
                f"Processed {i + 1}/{N_IMAGES}"
            )


# Remove hooks.
for hook in hooks:
    hook.remove()


# ============================================================
# BUILD FIXED ACTIVATION SCALES
# ============================================================

print()
print("=" * 70)
print("BUILDING FIXED LAYER-WIDE INT8 ACTIVATION SCALES")
print("=" * 70)

activation_scales = {}

for name in layers:

    tensors = activation_samples[name]

    if not tensors:

        raise RuntimeError(
            f"No activation data collected for {name}"
        )

    global_min = min(
        x.min().item()
        for x in tensors
    )

    global_max = max(
        x.max().item()
        for x in tensors
    )

    max_abs = max(
        abs(global_min),
        abs(global_max)
    )

    if max_abs == 0:

        scale = 1.0

    else:

        scale = max_abs / 127.0

    activation_scales[name] = scale

    print(
        f"{name:25s} "
        f"FP32=[{global_min:.6f}, {global_max:.6f}] "
        f"scale={scale:.10f}"
    )


# ============================================================
# DETERMINISTIC RNG PER LAYER
# ============================================================

def layer_seed(name):
    """
    Generate a deterministic seed from the layer name.

    Avoids Python's randomized hash() behaviour between
    separate Python processes.
    """

    digest = hashlib.sha256(
        name.encode("utf-8")
    ).hexdigest()

    return SEED + int(
        digest[:8],
        16
    )


# ============================================================
# EXTRACT REAL CONVOLUTION DOT PRODUCTS
# ============================================================

def extract_real_dot_products(name, n_samples):

    """
    Extract actual convolution dot products.

    For a convolution:

        output[c_out, y, x]

    the MAC is:

        SUM(
            activation[c_in, y+i, x+j]
            *
            weight[c_out, c_in, i, j]
        )

    We extract exactly those real activation/weight pairs.

    Returns:
        list of (activation_vector, weight_vector)
    """

    layer = layers[name]

    cin = layer.in_channels
    kh, kw = layer.kernel_size

    K = cin * kh * kw

    weight_q = weight_int8[name]

    activation_scale = activation_scales[name]

    rng = random.Random(
        layer_seed(name)
    )

    tensors = activation_samples[name]

    pairs = []

    while len(pairs) < n_samples:

        x = rng.choice(tensors)

        batch, channels, H, W = x.shape

        if H < kh or W < kw:
            continue

        # Random valid convolution position.
        y = rng.randrange(
            0,
            H - kh + 1
        )

        xpos = rng.randrange(
            0,
            W - kw + 1
        )

        # Random output channel.
        out_channel = rng.randrange(
            0,
            layer.out_channels
        )

        # Actual activation patch consumed by this convolution.
        patch = x[
            0,
            :,
            y:y + kh,
            xpos:xpos + kw
        ]

        # Fixed layer-wide activation scale.
        patch_q = quantize_int8(
            patch,
            activation_scale
        )

        # Corresponding actual kernel.
        kernel_q = weight_q[
            out_channel
        ]

        a_list = patch_q.flatten().tolist()

        b_list = kernel_q.flatten().tolist()

        if len(a_list) != K:
            raise RuntimeError(
                f"{name}: expected K={K}, "
                f"got {len(a_list)}"
            )

        pairs.append(
            (
                a_list,
                b_list
            )
        )

    return pairs


# ============================================================
# EXACT MAC
# ============================================================

def exact_dot_product(a_list, b_list):

    """
    Exact INT8 MAC:

        ACC = ACC + A*B

    repeated K times.

    The project's exact MAC model controls accumulator width
    and overflow behaviour.
    """

    steps = exact_mac_sequence(
        [int(a) for a in a_list],
        [int(b) for b in b_list],
        acc_init=0
    )

    if not steps:
        return 0

    return steps[-1].acc_hw


# ============================================================
# APPROXIMATE MAC
# ============================================================

def approximate_dot_product(
    a_list,
    b_list,
    level
):

    """
    Approximate INT8 MAC.

    IMPORTANT:
    This evaluates the entire accumulated MAC.

        ACC = ACC + approximate(A*B)

    for every operand pair.

    It does NOT characterize an isolated multiplier.
    """

    acc = 0

    for a, b in zip(
        a_list,
        b_list
    ):

        acc = approx_mac_step(
            acc,
            int(a),
            int(b),
            level
        )

    return acc


# ============================================================
# MAC METRICS
# ============================================================

def calculate_metrics(
    errors,
    exact_values
):

    errors = np.asarray(
        errors,
        dtype=np.float64
    )

    exact_values = np.asarray(
        exact_values,
        dtype=np.float64
    )

    abs_errors = np.abs(errors)

    mae = float(
        np.mean(abs_errors)
    )

    mse = float(
        np.mean(errors ** 2)
    )

    rmse = float(
        np.sqrt(mse)
    )

    max_abs = int(
        np.max(abs_errors)
    )

    bias = float(
        np.mean(errors)
    )

    std = float(
        np.std(errors)
    )

    exact_pct = float(
        np.mean(errors == 0)
        * 100.0
    )

    affected_pct = float(
        np.mean(errors != 0)
        * 100.0
    )

    # --------------------------------------------------------
    # Error percentiles
    # --------------------------------------------------------

    p50 = float(
        np.percentile(
            abs_errors,
            50
        )
    )

    p90 = float(
        np.percentile(
            abs_errors,
            90
        )
    )

    p95 = float(
        np.percentile(
            abs_errors,
            95
        )
    )

    p99 = float(
        np.percentile(
            abs_errors,
            99
        )
    )

    # --------------------------------------------------------
    # Normalized MAE
    #
    # This is more useful for MAC-level comparison than
    # raw MAE alone because MAC magnitudes vary between layers.
    # --------------------------------------------------------

    mean_abs_exact = float(
        np.mean(
            np.abs(exact_values)
        )
    )

    if mean_abs_exact > 0:

        normalized_mae = (
            mae
            / mean_abs_exact
            * 100.0
        )

        normalized_bias = (
            abs(bias)
            / mean_abs_exact
            * 100.0
        )

    else:

        normalized_mae = float("nan")
        normalized_bias = float("nan")

    # --------------------------------------------------------
    # Relative error
    #
    # Exclude exact==0 because relative error is undefined.
    # --------------------------------------------------------

    valid = (
        exact_values != 0
    )

    if np.any(valid):

        relative_errors = (
            abs_errors[valid]
            / np.abs(exact_values[valid])
        )

        mean_relative_error = float(
            np.mean(
                relative_errors
            ) * 100.0
        )

        median_relative_error = float(
            np.median(
                relative_errors
            ) * 100.0
        )

    else:

        mean_relative_error = float("nan")
        median_relative_error = float("nan")

    zero_reference_count = int(
        np.sum(~valid)
    )

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "max_abs_error": max_abs,
        "bias": bias,
        "std": std,

        "p50_abs_error": p50,
        "p90_abs_error": p90,
        "p95_abs_error": p95,
        "p99_abs_error": p99,

        "mean_abs_exact": mean_abs_exact,

        "normalized_mae_pct":
            normalized_mae,

        "normalized_abs_bias_pct":
            normalized_bias,

        "mean_relative_error_pct":
            mean_relative_error,

        "median_relative_error_pct":
            median_relative_error,

        "zero_reference_count":
            zero_reference_count,

        "exact_dot_products_pct":
            exact_pct,

        "affected_dot_products_pct":
            affected_pct,
    }


# ============================================================
# RUN FULL MAC-LEVEL EXPERIMENT
# ============================================================

summary_rows = []
detail_rows = []


for layer_index, layer_name in enumerate(
    layers.keys(),
    start=1
):

    layer = layers[layer_name]

    cin = layer.in_channels
    kh, kw = layer.kernel_size

    K = cin * kh * kw

    print()
    print("=" * 70)
    print(
        f"[{layer_index}/{len(layers)}] "
        f"LAYER: {layer_name}"
    )
    print("=" * 70)

    print(
        f"Cin={cin}, "
        f"Cout={layer.out_channels}, "
        f"Kernel={kh}x{kw}, "
        f"K={K}"
    )

    print(
        f"Real dot products sampled: "
        f"{DOT_PRODUCTS_PER_LAYER}"
    )

    print(
        "Extracting real convolution MAC operands..."
    )

    pairs = extract_real_dot_products(
        layer_name,
        DOT_PRODUCTS_PER_LAYER
    )

    per_level_errors = {
        level: []
        for level in range(4)
    }

    exact_values = []

    # --------------------------------------------------------
    # Evaluate every real dot product.
    # --------------------------------------------------------

    for sample_index, (
        a_list,
        b_list
    ) in enumerate(pairs):

        exact = exact_dot_product(
            a_list,
            b_list
        )

        exact_values.append(
            exact
        )

        detail = {
            "layer": layer_name,
            "sample": sample_index,
            "K": K,
            "exact_mac": exact,
        }

        for level in range(4):

            approx = approximate_dot_product(
                a_list,
                b_list,
                level
            )

            error = (
                exact
                - approx
            )

            per_level_errors[
                level
            ].append(
                error
            )

            detail[
                f"L{level}_approx_mac"
            ] = approx

            detail[
                f"L{level}_error"
            ] = error

        detail_rows.append(
            detail
        )

    # --------------------------------------------------------
    # Print results for this layer.
    # --------------------------------------------------------

    for level in range(4):

        metrics = calculate_metrics(
            per_level_errors[level],
            exact_values
        )

        print()
        print(
            f"L{level}:"
        )

        print(
            f"  MAE                  = "
            f"{metrics['mae']:.3f}"
        )

        print(
            f"  RMSE                 = "
            f"{metrics['rmse']:.3f}"
        )

        print(
            f"  MSE                  = "
            f"{metrics['mse']:.3f}"
        )

        print(
            f"  Max Abs Error        = "
            f"{metrics['max_abs_error']}"
        )

        print(
            f"  Bias                 = "
            f"{metrics['bias']:.3f}"
        )

        print(
            f"  Std                  = "
            f"{metrics['std']:.3f}"
        )

        print(
            f"  P50 Abs Error        = "
            f"{metrics['p50_abs_error']:.3f}"
        )

        print(
            f"  P90 Abs Error        = "
            f"{metrics['p90_abs_error']:.3f}"
        )

        print(
            f"  P95 Abs Error        = "
            f"{metrics['p95_abs_error']:.3f}"
        )

        print(
            f"  P99 Abs Error        = "
            f"{metrics['p99_abs_error']:.3f}"
        )

        print(
            f"  Mean |Exact MAC|     = "
            f"{metrics['mean_abs_exact']:.3f}"
        )

        print(
            f"  Normalized MAE       = "
            f"{metrics['normalized_mae_pct']:.4f}%"
        )

        print(
            f"  Normalized |Bias|    = "
            f"{metrics['normalized_abs_bias_pct']:.4f}%"
        )

        print(
            f"  Mean Relative Error  = "
            f"{metrics['mean_relative_error_pct']:.4f}%"
        )

        print(
            f"  Median Relative Err  = "
            f"{metrics['median_relative_error_pct']:.4f}%"
        )

        print(
            f"  Exact Dot Products   = "
            f"{metrics['exact_dot_products_pct']:.2f}%"
        )

        print(
            f"  Affected Dot Products= "
            f"{metrics['affected_dot_products_pct']:.2f}%"
        )

        summary_rows.append({
            "layer": layer_name,
            "K": K,
            "Cin": cin,
            "Cout": layer.out_channels,
            "kernel_h": kh,
            "kernel_w": kw,
            "activation_scale":
                activation_scales[layer_name],
            "weight_scale":
                weight_scales[layer_name],
            "level": f"L{level}",
            **metrics,
        })


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    cfg.RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# WRITE DETAILED CSV
# ============================================================

detail_path = os.path.join(
    cfg.RESULTS_DIR,
    "real_dnn_mac_results.csv"
)


with open(
    detail_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            detail_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        detail_rows
    )


# ============================================================
# WRITE SUMMARY CSV
# ============================================================

summary_path = os.path.join(
    cfg.RESULTS_DIR,
    "real_dnn_mac_summary.csv"
)


with open(
    summary_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            summary_rows[0].keys()
        )
    )

    writer.writeheader()
    writer.writerows(
        summary_rows
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FULL MAC-LEVEL PROFILING COMPLETE")
print("=" * 70)

print()
print(f"Convolution layers profiled : {len(layers)}")
print(
    f"Dot products per layer     : "
    f"{DOT_PRODUCTS_PER_LAYER}"
)
print(
    f"Total dot products          : "
    f"{len(layers) * DOT_PRODUCTS_PER_LAYER}"
)

print()
print("Reference:")
print("  Exact INT8 MAC")

print()
print("Approximation:")
print("  L0 = exact")
print("  L1 = PP columns 0-1 dropped")
print("  L2 = PP columns 0-3 dropped")
print("  L3 = PP columns 0-5 dropped")

print()
print("Quantization:")
print("  Fixed layer-wide activation scale")
print("  Fixed layer-wide weight scale")

print()
print("Detailed results:")
print(f"  {detail_path}")

print()
print("Summary results:")
print(f"  {summary_path}")

print()
print("NEXT STAGE:")
print("  Use these MAC-level results to identify")
print("  viable approximation levels before moving")
print("  to full DNN-level accuracy evaluation.")
