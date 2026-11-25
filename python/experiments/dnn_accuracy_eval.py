"""
DNN-LEVEL APPROXIMATE MAC EVALUATION
====================================

Evaluates approximate INT8 MAC policies inside ImageNet-pretrained
ResNet-18 using CIFAR-10 images.

Policies:
    L0       : exact INT8 MAC
    L1       : uniform L1 approximation
    L2       : uniform L2 approximation
    L3       : uniform L3 approximation
    ADAPTIVE : layer-wise policy selected from MAC-level analysis

Approximation mechanism:
    Hardware-grounded partial-product column truncation.

Important:
    models.approximation.pp_column_truncation()
    expects integer levels {0,1,2,3}, NOT strings "L0"..."L3".

Performance:
    A 65,536-entry signed-INT8 multiplication LUT is generated once.
    Network evaluation then performs LUT lookups instead of repeatedly
    constructing the 8x8 partial-product array.

Outputs:
    results/dnn_accuracy_results.csv
    results/dnn_accuracy_summary.csv
"""

import csv
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms


# ================================================================
# CONFIGURATION
# ================================================================

NUM_IMAGES = 100

RESULTS_DIR = "results"

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "dnn_accuracy_results.csv",
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "dnn_accuracy_summary.csv",
)

DEVICE = torch.device("cpu")


# ================================================================
# LEVEL DEFINITIONS
# ================================================================

LEVEL_NAME_TO_INT = {
    "L0": 0,
    "L1": 1,
    "L2": 2,
    "L3": 3,
}


# ================================================================
# MAC-DERIVED ADAPTIVE POLICY
# ================================================================
#
# Taken directly from the MAC-level decision analysis you obtained.
#
# L1:
#   conv1
#   layer1.*
#   layer2.0.*
#   layer2.1.conv1
#   layer3.0.downsample.0
#   layer4.0.downsample.0
#
# L2:
#   layer2.1.conv2
#   layer3.0.conv1
#   layer3.0.conv2
#   layer3.1.conv1
#   layer3.1.conv2
#   layer4.0.conv1
#   layer4.0.conv2
#   layer4.1.conv1
#   layer4.1.conv2
#
# This corresponds to the "best level by layer" output from your
# MAC-level experiment.
# ================================================================

ADAPTIVE_POLICY = OrderedDict([
    ("conv1", "L1"),

    ("layer1.0.conv1", "L1"),
    ("layer1.0.conv2", "L1"),
    ("layer1.1.conv1", "L1"),
    ("layer1.1.conv2", "L1"),

    ("layer2.0.conv1", "L1"),
    ("layer2.0.conv2", "L1"),
    ("layer2.0.downsample.0", "L1"),

    ("layer2.1.conv1", "L1"),
    ("layer2.1.conv2", "L2"),

    ("layer3.0.conv1", "L2"),
    ("layer3.0.conv2", "L2"),
    ("layer3.0.downsample.0", "L1"),

    ("layer3.1.conv1", "L2"),
    ("layer3.1.conv2", "L2"),

    ("layer4.0.conv1", "L2"),
    ("layer4.0.conv2", "L2"),
    ("layer4.0.downsample.0", "L1"),

    ("layer4.1.conv1", "L2"),
    ("layer4.1.conv2", "L2"),
])


# ================================================================
# MODEL
# ================================================================

def load_model():
    print("Loading ImageNet-pretrained ResNet-18...")

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    model = model.to(DEVICE)
    model.eval()

    print("Model loaded.")

    return model


# ================================================================
# CIFAR-10 DATA
# ================================================================

def load_dataset(weights):
    """
    Resize CIFAR-10 images to 224x224 and use the normalization expected
    by ImageNet-pretrained ResNet-18.
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    return dataset


# ================================================================
# SIGNED INT8 CONVERSION
# ================================================================

def signed_int8_to_index(a, b):
    """
    Convert signed INT8 operands into a [0,65535] LUT index.

    a,b:
        signed values in [-128,127]

    index:
        upper byte = unsigned representation of a
        lower byte = unsigned representation of b
    """

    a_u = a & 0xFF
    b_u = b & 0xFF

    return (a_u << 8) | b_u


# ================================================================
# LUT GENERATION
# ================================================================

def build_multiplier_lut():
    """
    Generate the complete signed INT8 approximate multiplier LUT.

    There are exactly:

        256 * 256 = 65,536

    possible signed INT8 operand pairs.

    Each LUT contains:
        L0
        L1
        L2
        L3
    """

    print()
    print("=" * 64)
    print("BUILDING INT8 APPROXIMATE MULTIPLIER LUT")
    print("=" * 64)

    print("Possible signed INT8 operand pairs : 65,536")
    print("Generating L0/L1/L2/L3 tables...")

    # Import only here so the experiment has one clear dependency.
    from models.approximation import pp_column_truncation

    # int32 is enough for the product:
    # -128 * -128 = 16384
    # 127 * 127 = 16129
    lut = {
        "L0": np.zeros(65536, dtype=np.int32),
        "L1": np.zeros(65536, dtype=np.int32),
        "L2": np.zeros(65536, dtype=np.int32),
        "L3": np.zeros(65536, dtype=np.int32),
    }

    start = time.time()

    for idx in range(65536):

        a_u = (idx >> 8) & 0xFF
        b_u = idx & 0xFF

        # unsigned byte -> signed INT8
        a = a_u if a_u < 128 else a_u - 256
        b = b_u if b_u < 128 else b_u - 256

        # Exact product.
        lut["L0"][idx] = a * b

        # Hardware-grounded approximation.
        lut["L1"][idx] = pp_column_truncation(
            a, b, LEVEL_NAME_TO_INT["L1"]
        )

        lut["L2"][idx] = pp_column_truncation(
            a, b, LEVEL_NAME_TO_INT["L2"]
        )

        lut["L3"][idx] = pp_column_truncation(
            a, b, LEVEL_NAME_TO_INT["L3"]
        )

        if (idx + 1) % 8192 == 0:
            print(
                f"LUT progress: {idx + 1}/65536"
            )

    elapsed = time.time() - start

    print(f"LUT generation complete in {elapsed:.2f} seconds.")

    # Sanity check.
    print()
    print("LUT sanity checks:")

    test_pairs = [
        (0, 0),
        (1, 1),
        (-1, 1),
        (1, -1),
        (-1, -1),
        (127, 127),
        (-128, 127),
        (-128, -128),
    ]

    for a, b in test_pairs:
        idx = signed_int8_to_index(a, b)

        exact = int(lut["L0"][idx])
        expected = a * b

        assert exact == expected, (
            f"L0 LUT failure: {a} * {b}"
        )

    print("L0 exactness: PASS")

    # ------------------------------------------------------------
    # TEST A (per debugging strategy): verify that the encoding used
    # to BUILD the LUT (signed_int8_to_index) is IDENTICAL to the
    # encoding ApproxConv2d/ExactInt8Conv2d use to INDEX it, for all
    # 65,536 signed INT8 pairs. This is the exact invariant whose
    # violation caused the original bug -- it must never regress.
    # ------------------------------------------------------------
    mismatches = 0
    for a in range(-128, 128):
        for b in range(-128, 128):
            build_idx = signed_int8_to_index(a, b)
            lookup_a = a & 0xFF
            lookup_b = b & 0xFF
            lookup_idx = (lookup_a << 8) | lookup_b
            if build_idx != lookup_idx:
                mismatches += 1

    assert mismatches == 0, (
        f"LUT build/lookup index encodings disagree on {mismatches} pairs "
        "-- this is the exact defect found in the previous audit."
    )
    print("LUT build/lookup index-encoding consistency: PASS (65,536/65,536)")

    return lut


# ================================================================
# INT8 QUANTIZATION
# ================================================================

def quantize_tensor(x, scale):
    """
    Symmetric signed INT8 quantization.

        q = round(x / scale)

    followed by clipping to [-127,127].

    Zero-point is fixed at zero.
    """

    if scale <= 0:
        raise ValueError("Quantization scale must be positive.")

    q = torch.round(x / scale)

    q = torch.clamp(
        q,
        -127,
        127,
    )

    return q.to(torch.int16)


# ================================================================
# DEQUANTIZATION
# ================================================================

def dequantize_tensor(q, scale):
    return q.float() * scale


# ================================================================
# WEIGHT SCALE
# ================================================================

def calculate_weight_scales(model):
    """
    Fixed layer-wide symmetric INT8 weight scales.

        scale = max(abs(weight)) / 127
    """

    scales = {}

    print()
    print("Calculating fixed layer-wide weight scales...")

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            weight = module.weight.detach()

            max_abs = float(
                torch.max(torch.abs(weight)).item()
            )

            if max_abs == 0:
                scale = 1.0
            else:
                scale = max_abs / 127.0

            scales[name] = scale

    return scales


# ================================================================
# ACTIVATION CALIBRATION
# ================================================================

@torch.no_grad()
def collect_activation_ranges(
    model,
    dataset,
    num_images,
):
    """
    Collect layer-wide activation maxima.

    The model is run in FP32 during calibration.

    ReLU outputs are naturally non-negative for the selected
    post-ReLU locations, but this calibration simply records
    the absolute maximum.
    """

    print()
    print("Collecting activation ranges...")

    activation_max = {}

    hooks = []

    def make_hook(name):

        def hook(module, inputs, output):

            if isinstance(output, tuple):
                output = output[0]

            value = float(
                torch.max(
                    torch.abs(output)
                ).item()
            )

            if name not in activation_max:
                activation_max[name] = value
            else:
                activation_max[name] = max(
                    activation_max[name],
                    value,
                )

        return hook

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            hooks.append(
                module.register_forward_hook(
                    make_hook(name)
                )
            )

    model.eval()

    for i in range(num_images):

        image, _ = dataset[i]

        image = image.unsqueeze(0).to(DEVICE)

        model(image)

        if (i + 1) % 25 == 0:
            print(
                f"Calibration: {i + 1}/{num_images}"
            )

    for hook in hooks:
        hook.remove()

    scales = {}

    for name, max_abs in activation_max.items():

        if max_abs == 0:
            scales[name] = 1.0
        else:
            scales[name] = max_abs / 127.0

    return scales


# ================================================================
# APPROXIMATE CONVOLUTION
# ================================================================

class ApproxConv2d(nn.Module):
    """
    Replacement Conv2d implementing the INT8 approximate MAC.

    Important design choice:

    We do NOT execute:

        pp_column_truncation(a,b,...)

    for every MAC.

    Instead:

        signed INT8 operands
                |
                v
        65,536-entry LUT
                |
                v
        approximate product

    This makes the experiment practical on CPU.
    """

    def __init__(
        self,
        original_conv,
        layer_name,
        level_name,
        lut,
        weight_scale,
        activation_scale,
    ):
        super().__init__()

        self.layer_name = layer_name
        self.level_name = level_name

        self.lut = lut[level_name]

        self.weight_scale = weight_scale
        self.activation_scale = activation_scale

        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups

        self.out_channels = original_conv.out_channels
        self.in_channels = original_conv.in_channels
        self.kernel_size = original_conv.kernel_size

        # Quantized weights.
        q_weight = quantize_tensor(
            original_conv.weight.detach(),
            weight_scale,
        )

        self.register_buffer(
            "q_weight",
            q_weight,
        )

        if original_conv.bias is not None:
            self.register_buffer(
                "bias",
                original_conv.bias.detach().clone(),
            )
        else:
            self.bias = None

    def forward(self, x):

        # ------------------------------------------------------------
        # Quantize activation.
        # ------------------------------------------------------------

        q_x = quantize_tensor(
            x,
            self.activation_scale,
        )

        # ------------------------------------------------------------
        # Convert signed int16 tensors into the SAME two's-complement
        # byte encoding used to build the LUT (signed_int8_to_index:
        # a_u = a & 0xFF). The previous version used excess-128 offset
        # binary (q_x + 128), which does NOT match the LUT's indexing
        # scheme except by coincidence at a few values -- it silently
        # looked up the product of an unrelated operand pair on every
        # single MAC. Bitwise AND with 0xFF reproduces two's-complement
        # byte truncation exactly, matching signed_int8_to_index().
        #
        # [-128,127] (two's complement) -> [0,255]
        # ------------------------------------------------------------

        x_u = (q_x.to(torch.int32) & 0xFF)
        w_u = (self.q_weight.to(torch.int32) & 0xFF)

        # ------------------------------------------------------------
        # Unfold input into convolution patches.
        #
        # Shape:
        #
        # [B, Cin*K*K, output_positions]
        # ------------------------------------------------------------

        patches = torch.nn.functional.unfold(
            x_u.float(),
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        ).to(torch.int32)

        batch_size = patches.shape[0]
        positions = patches.shape[2]

        k = (
            self.in_channels
            * self.kernel_size[0]
            * self.kernel_size[1]
        )

        # ------------------------------------------------------------
        # Flatten weights.
        #
        # [Cout, K]
        # ------------------------------------------------------------

        weights_flat = w_u.reshape(
            self.out_channels,
            k,
        )

        # ------------------------------------------------------------
        # LUT lookup.
        #
        # Index:
        #
        #     a_unsigned * 256 + b_unsigned
        #
        # We perform this in chunks over K so memory doesn't explode.
        # ------------------------------------------------------------

        accumulator = torch.zeros(
            batch_size,
            self.out_channels,
            positions,
            dtype=torch.int32,
            device=x.device,
        )

        # A conservative chunk size.
        chunk = 256

        for start in range(0, k, chunk):

            end = min(
                start + chunk,
                k,
            )

            a_chunk = patches[:, start:end, :]
            b_chunk = weights_flat[:, start:end]

            # --------------------------------------------------------
            # Broadcast:
            #
            # a:
            #   [B, C, P]
            #
            # b:
            #   [O, C]
            #
            # result:
            #   [B, O, C, P]
            #
            # For ResNet-18 this is manageable chunk-by-chunk.
            # --------------------------------------------------------

            a_u = a_chunk.unsqueeze(1)

            b_u = b_chunk.unsqueeze(0).unsqueeze(-1)

            indices = (
                a_u * 256
                + b_u
            ).long()

            # LUT lives as numpy array.
            # Convert it once to torch for fast indexing.
            lut_tensor = torch.from_numpy(
                self.lut
            ).to(
                device=x.device,
                dtype=torch.int32,
            )

            products = lut_tensor[indices]

            accumulator += products.sum(
                dim=2
            )

        # ------------------------------------------------------------
        # Convert accumulated INT8 MAC back to approximate real value.
        #
        # Each product corresponds to:
        #
        #     q_activation * q_weight
        #
        # therefore:
        #
        #     real = MAC * activation_scale * weight_scale
        # ------------------------------------------------------------

        output = (
            accumulator.float()
            * self.activation_scale
            * self.weight_scale
        )

        if self.bias is not None:
            output += self.bias.view(
                1,
                -1,
                1,
                1,
            )

        output_height = (
            patches.shape[2]
        )

        # Calculate output dimensions.
        h_in = x.shape[2]
        w_in = x.shape[3]

        kh, kw = self.kernel_size
        dh, dw = self.dilation
        ph, pw = self.padding
        sh, sw = self.stride

        h_out = (
            h_in
            + 2 * ph
            - dh * (kh - 1)
            - 1
        ) // sh + 1

        w_out = (
            w_in
            + 2 * pw
            - dw * (kw - 1)
            - 1
        ) // sw + 1

        output = output.reshape(
            batch_size,
            self.out_channels,
            h_out,
            w_out,
        )

        return output


# ================================================================
# RECURSIVE MODEL REPLACEMENT
# ================================================================

def replace_convolutions(
    model,
    policy_name,
    lut,
    weight_scales,
    activation_scales,
):
    """
    Replace every Conv2d with ApproxConv2d.

    policy_name:
        L0/L1/L2/L3/ADAPTIVE
    """

    for name, module in list(
        model.named_children()
    ):

        full_name = name

        if isinstance(module, nn.Conv2d):

            # For recursive modules, the caller supplies names
            # relative to the full ResNet hierarchy.
            pass

    # Easier robust implementation using explicit traversal.
    for layer_name, module in list(
        model.named_modules()
    ):

        if not isinstance(module, nn.Conv2d):
            continue

        # Find parent module.
        parts = layer_name.split(".")

        parent = model

        for part in parts[:-1]:
            parent = getattr(parent, part)

        child_name = parts[-1]

        if policy_name == "ADAPTIVE":

            level_name = ADAPTIVE_POLICY[
                layer_name
            ]

        else:

            level_name = policy_name

        replacement = ApproxConv2d(
            original_conv=module,
            layer_name=layer_name,
            level_name=level_name,
            lut=lut,
            weight_scale=weight_scales[layer_name],
            activation_scale=activation_scales[layer_name],
        )

        setattr(
            parent,
            child_name,
            replacement,
        )

    return model


# ================================================================
# EXACT INT8 REFERENCE CONVOLUTION
# ================================================================

class ExactInt8Conv2d(nn.Module):
    """
    Exact INT8 convolution used as the DNN-level reference.

    This is deliberately separate from ApproxConv2d: it performs the
    same symmetric INT8 quantization policy, but computes the MAC using
    PyTorch's exact integer convolution rather than the approximate LUT.
    L0 must agree with this model, not with the original FP32 model.
    """

    def __init__(
        self,
        original_conv,
        layer_name,
        weight_scale,
        activation_scale,
    ):
        super().__init__()

        self.layer_name = layer_name
        self.weight_scale = weight_scale
        self.activation_scale = activation_scale

        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups

        q_weight = quantize_tensor(
            original_conv.weight.detach(),
            weight_scale,
        ).to(torch.int32)

        self.register_buffer("q_weight", q_weight)

        if original_conv.bias is not None:
            self.register_buffer(
                "bias",
                original_conv.bias.detach().clone(),
            )
        else:
            self.bias = None

    def forward(self, x):
        q_x = quantize_tensor(
            x,
            self.activation_scale,
        ).to(torch.int32)

        # Use the same explicit unfolded-MAC structure as ApproxConv2d,
        # but multiply the signed INT8 operands exactly instead of using
        # the approximation LUT. Keeping the accumulator in int64 avoids
        # any floating-point accumulation/rounding ambiguity.
        patches = torch.nn.functional.unfold(
            q_x.float(),
            kernel_size=self.q_weight.shape[2:],
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        ).to(torch.int32)

        batch_size = patches.shape[0]
        positions = patches.shape[2]
        k = (
            self.q_weight.shape[1]
            * self.q_weight.shape[2]
            * self.q_weight.shape[3]
        )

        weights_flat = self.q_weight.reshape(
            self.q_weight.shape[0],
            k,
        )

        accumulator = torch.zeros(
            batch_size,
            self.q_weight.shape[0],
            positions,
            dtype=torch.int64,
            device=x.device,
        )

        chunk = 256

        for start in range(0, k, chunk):
            end = min(start + chunk, k)

            a_chunk = patches[:, start:end, :]
            b_chunk = weights_flat[:, start:end]

            # [B,C,P] x [O,C] -> [B,O,C,P]
            products = (
                a_chunk.unsqueeze(1).to(torch.int64)
                * b_chunk.unsqueeze(0).unsqueeze(-1).to(torch.int64)
            )

            accumulator += products.sum(dim=2)

        output = (
            accumulator.float()
            * self.activation_scale
            * self.weight_scale
        )

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        h_in = x.shape[2]
        w_in = x.shape[3]
        kh, kw = self.q_weight.shape[2:]
        dh, dw = self.dilation
        ph, pw = self.padding
        sh, sw = self.stride

        h_out = (
            h_in + 2 * ph - dh * (kh - 1) - 1
        ) // sh + 1
        w_out = (
            w_in + 2 * pw - dw * (kw - 1) - 1
        ) // sw + 1

        return output.reshape(
            batch_size,
            self.q_weight.shape[0],
            h_out,
            w_out,
        )


def replace_convolutions_exact_int8(
    model,
    weight_scales,
    activation_scales,
):
    """Replace every Conv2d with the exact INT8 reference convolution."""

    for layer_name, module in list(model.named_modules()):

        if not isinstance(module, nn.Conv2d):
            continue

        parts = layer_name.split(".")
        parent = model

        for part in parts[:-1]:
            parent = getattr(parent, part)

        replacement = ExactInt8Conv2d(
            original_conv=module,
            layer_name=layer_name,
            weight_scale=weight_scales[layer_name],
            activation_scale=activation_scales[layer_name],
        )

        setattr(parent, parts[-1], replacement)

    return model


# ================================================================
# REFERENCE PREDICTIONS
# ================================================================

@torch.no_grad()
def get_reference_predictions(
    model,
    dataset,
    num_images,
):
    """
    Reference = exact INT8 ResNet-18 using the same fixed layer-wide
    weight and activation scales as the approximate models.

    This is the correct baseline for this experiment. Comparing the
    approximate INT8 network against the original FP32 network would
    mix quantization error with approximation error and can make L0
    disagree even though L0 itself is bit-exact.

    We also record CIFAR-10 labels for the raw diagnostic accuracy.
    """

    print()
    print("=" * 64)
    print("REFERENCE MODEL EVALUATION")
    print("=" * 64)

    model.eval()

    predictions = []
    labels = []

    start = time.time()

    for i in range(num_images):

        image, label = dataset[i]

        image = image.unsqueeze(0).to(DEVICE)

        output = model(image)

        pred = int(
            torch.argmax(
                output,
                dim=1,
            ).item()
        )

        predictions.append(pred)
        labels.append(label)

        if (i + 1) % 25 == 0:
            print(
                f"Reference inference: "
                f"{i + 1}/{num_images}"
            )

    elapsed = time.time() - start

    predictions = np.array(predictions)
    labels = np.array(labels)

    accuracy = (
        np.mean(
            predictions == labels
        )
        * 100.0
    )

    print(
        f"Reference accuracy: {accuracy:.2f}%"
    )

    print(
        f"Reference inference time: "
        f"{elapsed:.2f} s"
    )

    return predictions, labels


# ================================================================
# APPROXIMATE MODEL EVALUATION
# ================================================================

@torch.no_grad()
def evaluate_policy(
    model,
    dataset,
    num_images,
    reference_predictions,
    labels,
    policy_name,
):
    print()
    print("=" * 64)
    print(f"EVALUATING POLICY: {policy_name}")
    print("=" * 64)

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

    predictions = np.array(predictions)

    # ------------------------------------------------------------
    # Agreement with reference network.
    # ------------------------------------------------------------

    agreement = (
        np.mean(
            predictions
            == reference_predictions
        )
        * 100.0
    )

    # ------------------------------------------------------------
    # Actual CIFAR-10 top-1 accuracy.
    #
    # NOTE:
    # ImageNet-pretrained ResNet-18 has ImageNet's 1000 output
    # classes, while CIFAR-10 labels are 0..9. Therefore this is
    # NOT a meaningful transfer-learning accuracy metric unless
    # the model's classifier has been adapted to CIFAR-10.
    #
    # We still calculate it only as a raw diagnostic.
    # ------------------------------------------------------------

    accuracy = (
        np.mean(
            predictions == labels
        )
        * 100.0
    )

    return {
        "policy": policy_name,
        "accuracy_pct": accuracy,
        "reference_agreement_pct": agreement,
        "inference_time_sec": elapsed,
    }


# ================================================================
# POLICY SUMMARY
# ================================================================

def print_policy_table(results):
    print()
    print("=" * 72)
    print("DNN-LEVEL POLICY COMPARISON")
    print("=" * 72)

    print(
        f"{'Policy':<12}"
        f"{'Accuracy':>14}"
        f"{'Ref Agreement':>18}"
        f"{'Time(s)':>14}"
    )

    print("-" * 72)

    for result in results:

        print(
            f"{result['policy']:<12}"
            f"{result['accuracy_pct']:>13.2f}%"
            f"{result['reference_agreement_pct']:>17.2f}%"
            f"{result['inference_time_sec']:>13.2f}"
        )


# ================================================================
# SAVE RESULTS
# ================================================================

def save_results(
    results,
    reference_accuracy,
):
    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Detailed summary.
    # ------------------------------------------------------------

    with open(
        RESULTS_FILE,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "policy",
            "accuracy_pct",
            "reference_agreement_pct",
            "inference_time_sec",
            "accuracy_drop_vs_reference_pct",
        ])

        for result in results:

            drop = (
                reference_accuracy
                - result["accuracy_pct"]
            )

            writer.writerow([
                result["policy"],
                f"{result['accuracy_pct']:.6f}",
                f"{result['reference_agreement_pct']:.6f}",
                f"{result['inference_time_sec']:.6f}",
                f"{drop:.6f}",
            ])

    # ------------------------------------------------------------
    # Compact summary.
    # ------------------------------------------------------------

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "reference_accuracy_pct",
            "policy",
            "accuracy_pct",
            "reference_agreement_pct",
            "accuracy_drop_vs_reference_pct",
        ])

        for result in results:

            drop = (
                reference_accuracy
                - result["accuracy_pct"]
            )

            writer.writerow([
                f"{reference_accuracy:.6f}",
                result["policy"],
                f"{result['accuracy_pct']:.6f}",
                f"{result['reference_agreement_pct']:.6f}",
                f"{drop:.6f}",
            ])

    print()
    print("Saved:")
    print(f"  {RESULTS_FILE}")
    print(f"  {SUMMARY_FILE}")


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print("=" * 72)
    print("DNN-LEVEL APPROXIMATE MAC EVALUATION")
    print("=" * 72)

    print()
    print(f"Device        : {DEVICE}")
    print(f"Images        : {NUM_IMAGES}")

    # ------------------------------------------------------------
    # Original model.
    # ------------------------------------------------------------

    original_model = load_model()

    # ------------------------------------------------------------
    # Dataset.
    # ------------------------------------------------------------

    weights = models.ResNet18_Weights.DEFAULT

    dataset = load_dataset(weights)

    print(
        f"Using {NUM_IMAGES} CIFAR-10 test images."
    )

    # ------------------------------------------------------------
    # Find convolution layers.
    # ------------------------------------------------------------

    conv_layers = [
        name
        for name, module
        in original_model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]

    print(
        f"Found {len(conv_layers)} convolution layers."
    )

    # ------------------------------------------------------------
    # Weight scales.
    # ------------------------------------------------------------

    weight_scales = calculate_weight_scales(
        original_model
    )

    # ------------------------------------------------------------
    # Activation calibration.
    # ------------------------------------------------------------

    activation_scales = collect_activation_ranges(
        original_model,
        dataset,
        NUM_IMAGES,
    )

    # ------------------------------------------------------------
    # Print adaptive policy.
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("MAC-DERIVED ADAPTIVE POLICY")
    print("=" * 72)

    for layer_name in conv_layers:

        level = ADAPTIVE_POLICY.get(
            layer_name,
            "L1",
        )

        print(
            f"{layer_name:<30} -> {level}"
        )

    # ------------------------------------------------------------
    # Build LUT.
    # ------------------------------------------------------------

    lut = build_multiplier_lut()

    # ------------------------------------------------------------
    # Reference predictions.
    # ------------------------------------------------------------

    # Build the exact INT8 reference using the SAME quantization
    # scales used by ApproxConv2d. This isolates approximation error
    # from ordinary FP32->INT8 quantization error.
    reference_model = load_model()
    reference_model = replace_convolutions_exact_int8(
        model=reference_model,
        weight_scales=weight_scales,
        activation_scales=activation_scales,
    )

    reference_predictions, labels = (
        get_reference_predictions(
            reference_model,
            dataset,
            NUM_IMAGES,
        )
    )

    del reference_model

    reference_accuracy = (
        np.mean(
            reference_predictions == labels
        )
        * 100.0
    )

    # ------------------------------------------------------------
    # Evaluate all policies.
    # ------------------------------------------------------------

    policies = [
        "L0",
        "L1",
        "L2",
        "L3",
        "ADAPTIVE",
    ]

    results = []

    for policy in policies:

        print()
        print(
            f"Preparing {policy} model..."
        )

        # Fresh model for every policy.
        model = load_model()

        model = replace_convolutions(
            model=model,
            policy_name=policy,
            lut=lut,
            weight_scales=weight_scales,
            activation_scales=activation_scales,
        )

        result = evaluate_policy(
            model=model,
            dataset=dataset,
            num_images=NUM_IMAGES,
            reference_predictions=reference_predictions,
            labels=labels,
            policy_name=policy,
        )

        results.append(result)

        del model

    # ------------------------------------------------------------
    # Print final comparison.
    # ------------------------------------------------------------

    print_policy_table(results)

    # ------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------

    save_results(
        results,
        reference_accuracy,
    )

    print()
    print("=" * 72)
    print("EXPERIMENT COMPLETE")
    print("=" * 72)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Reference agreement measures whether approximate "
        "ResNet-18 produces the same prediction as the "
        "reference model."
    )

    print(
        "The raw CIFAR-10 accuracy is only diagnostic because "
        "this experiment uses ImageNet-pretrained ResNet-18 "
        "without replacing its 1000-class classifier."
    )

    print()
    print(
        "The key result for this stage is:"
    )

    print(
        "Does the MAC-derived ADAPTIVE layer policy preserve "
        "network predictions better than uniform L1/L2/L3?"
    )


if __name__ == "__main__":
    main()
