import os
import csv
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

# ============================================================
# Configuration
# ============================================================

NUM_IMAGES = 10
OUTPUT_DIR = "results/resnet_workload"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Dataset
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# ============================================================
# Model
# ============================================================

model = torchvision.models.resnet18(weights="DEFAULT")
model.eval()

# ============================================================
# Collect convolution layers
# ============================================================

conv_layers = []

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        conv_layers.append((name, module))

print(f"Found {len(conv_layers)} convolution layers")

# ============================================================
# INT8 quantization
# ============================================================

def quantize_symmetric(x, scale):
    q = torch.round(x / scale)
    q = torch.clamp(q, -128, 127)
    return q.to(torch.int8)


def get_scale(x):
    max_abs = torch.max(torch.abs(x)).item()

    if max_abs == 0:
        return 1.0

    return max_abs / 127.0


# ============================================================
# Storage
# ============================================================

workload_file = os.path.join(
    OUTPUT_DIR,
    "resnet18_int8_workload.csv"
)

# ============================================================
# Hook activations
# ============================================================

activations = {}

hooks = []


def make_hook(name):
    def hook(module, inputs, output):
        activations[name] = inputs[0].detach().cpu()
    return hook


for name, module in conv_layers:
    hooks.append(
        module.register_forward_hook(make_hook(name))
    )


# ============================================================
# Export
# ============================================================

with open(workload_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "image_id",
        "layer_id",
        "layer_name",
        "mac_index",
        "a",
        "b"
    ])

    for image_id in range(NUM_IMAGES):

        image, label = dataset[image_id]

        image = image.unsqueeze(0)

        activations.clear()

        # Run ResNet forward pass
        with torch.no_grad():
            model(image)

        print(
            f"\nProcessing image {image_id + 1}/{NUM_IMAGES}"
        )

        # ----------------------------------------------------
        # Process every convolution layer
        # ----------------------------------------------------

        for layer_id, (name, module) in enumerate(conv_layers):

            x = activations[name]

            weight = module.weight.detach().cpu()

            # ------------------------------------------------
            # Quantize activation
            # ------------------------------------------------

            act_scale = get_scale(x)

            x_q = quantize_symmetric(
                x,
                act_scale
            )

            # ------------------------------------------------
            # Quantize weights
            # ------------------------------------------------

            weight_scale = get_scale(weight)

            w_q = quantize_symmetric(
                weight,
                weight_scale
            )

            # ------------------------------------------------
            # Extract convolution dimensions
            # ------------------------------------------------

            batch, cin, hin, win = x_q.shape

            cout, _, kh, kw = w_q.shape

            stride_h, stride_w = module.stride
            pad_h, pad_w = module.padding

            # ------------------------------------------------
            # Explicit convolution traversal
            # ------------------------------------------------

            mac_index = 0

            for out_ch in range(cout):

                for oh in range(
                    (hin + 2 * pad_h - kh) // stride_h + 1
                ):

                    for ow in range(
                        (win + 2 * pad_w - kw) // stride_w + 1
                    ):

                        for c in range(cin):

                            for ky in range(kh):

                                iy = (
                                    oh * stride_h
                                    + ky
                                    - pad_h
                                )

                                if iy < 0 or iy >= hin:
                                    continue

                                for kx in range(kw):

                                    ix = (
                                        ow * stride_w
                                        + kx
                                        - pad_w
                                    )

                                    if ix < 0 or ix >= win:
                                        continue

                                    a = int(
                                        x_q[0, c, iy, ix].item()
                                    )

                                    b = int(
                                        w_q[
                                            out_ch,
                                            c,
                                            ky,
                                            kx
                                        ].item()
                                    )

                                    writer.writerow([
                                        image_id,
                                        layer_id,
                                        name,
                                        mac_index,
                                        a,
                                        b
                                    ])

                                    mac_index += 1

            print(
                f"  {layer_id:2d} {name:25s} "
                f"MACs exported: {mac_index}"
            )


# ============================================================
# Cleanup
# ============================================================

for hook in hooks:
    hook.remove()

print("\n======================================")
print("WORKLOAD EXPORT COMPLETE")
print("======================================")
print(f"Output: {workload_file}")
