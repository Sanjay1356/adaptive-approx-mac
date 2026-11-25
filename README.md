# Adaptive Approximate MAC Architecture for AI Accelerators

**Layer-adaptive approximate INT8 multiply–accumulate unit, from DNN sensitivity analysis to RTL, synthesis and GDSII.**

![SystemVerilog](https://img.shields.io/badge/HDL-SystemVerilog-blue)
![Verification](https://img.shields.io/badge/multiplier-262%2C144%2F262%2C144%20vectors%20pass-brightgreen)
![Flow](https://img.shields.io/badge/flow-RTL%20%E2%86%92%20GDSII-orange)
![Tech](https://img.shields.io/badge/tech-GPDK180%20%7C%20TSMC%20N65-lightgrey)

B.Tech Final Year Project, Electronics Engineering (VLSI Design and Technology), VIT Chennai.
**Author:** A Sanjay · **Supervisor:** Dr. Sasipriya P

---

## The idea

DNN inference is mostly multiply–accumulate operations, so the MAC unit drives an accelerator's area and power. Approximate multipliers save hardware. But applying **one** approximation level to every layer ignores a basic fact: **layers do not tolerate error equally.** Some can be approximated heavily, while others break the network's prediction with even small errors.

This project builds a **signed INT8 MAC whose approximation level is chosen per layer at run time**. The per-layer levels come from **measured** sensitivity on ResNet-18, not from a proxy.

```
 layer_id ──► ┌────────────────────────────┐  approx_level[1:0]  ┌──────────────────────────────────┐
 layer_valid─►│ adaptive_policy_controller │────────────────────►│ approx_mac_top                   │
              │  20-entry policy table     │   (registered,      │  ┌────────────────────────────┐  │
              │  (derived offline)         │    1-cycle latency) │  │ approx_multiplier          │  │
              └────────────────────────────┘                     │  │ INT8×INT8 → 16b, PP column │  │
 a[7:0], b[7:0] ────────────────────────────────────────────────►│  │ truncation + compensation  │  │
                                                                 │  └─────────────┬──────────────┘  │
                                                                 │        32-bit accumulator        │──► acc_out[31:0]
                                                                 └──────────────────────────────────┘
                                   adaptive_mac_system (top)
```

## How the approximation works

The 8×8 signed multiplier forms all 64 partial-product cells `a_i · b_j · 2^(i+j)`, using Baugh–Wooley-style sign weights: a cell is negative when exactly one of `i`, `j` is the sign bit (bit 7). Approximation **removes every cell in the low-order columns `i + j < K`**:

| Level | `approx_level` | K | PP cells removed (of 64) | Compensation `C_K` |
|:-----:|:--------------:|:-:|:------------------------:|:------------------:|
| L0 | `2'b00` | 0 | 0 (exact) | 0 |
| L1 | `2'b01` | 2 | 3 (4.69%) | 1 |
| L2 | `2'b10` | 4 | 10 (15.63%) | 12 |
| L3 | `2'b11` | 6 | 21 (32.81%) | 80 |

**Statistical compensation.** Truncation only ever loses value, so the error has a non-zero mean. If operand bits are treated as Bernoulli(0.5), each dropped cell is active with probability ¼. So the constant `C_K = ⌊¼ · Σ_{i+j<K} 2^(i+j)⌋` is added back. It is added only when `|a[K-1:0]` and `|b[K-1:0]` are both set, which is a cheap necessary condition for any dropped cell to be active.

*Worked example (127 × 127):* exact = 16129. At L3 the dropped cells hold 321, the gate fires, and the result is 16129 − 321 + 80 = **15888**.

For K ≤ 6 every removed cell has positive weight, so the sign handling of the retained array is unchanged.

## Deriving the per-layer policy

1. Run the exact INT8 ResNet-18 (ImageNet-pretrained, 20 conv layers including 3 downsample 1×1s) on a fixed 100-image set to get reference predictions.
2. For each layer ℓ and each level L1–L3, approximate **only that layer** and measure top-1 **prediction agreement** with the reference. That gives 60 measurements.
3. Assign each layer the **strongest level with ≥ 90% agreement**, or L0 if none qualifies.

Resulting policy (hard-coded in `adaptive_policy_controller.sv`):

| Layer ID | Layer | Level |
|:-:|---|:-:|
| 7  | layer2.0.downsample.0 | **L2** |
| 12 | layer3.0.downsample.0 | **L3** |
| 17 | layer4.0.downsample.0 | **L1** |
| 18 | layer4.1.conv1 | **L2** |
| 19 | layer4.1.conv2 | **L1** |
| all other 15 | conv1, layer1–layer4 main convs | L0 (exact) |

The downsample layers and the last two convolutions are the most tolerant. conv1 and the early layer1 convs are the most sensitive.

## Results

### Network-level prediction agreement (100 images, preliminary)

| Configuration | Agreement with exact INT8 |
|---|:-:|
| L0 uniform (exact) | 100% |
| L1 uniform | 25% |
| L2 uniform | 1% |
| L3 uniform | 0% |
| **Adaptive policy** | **86%** |

Uniform approximation collapses quickly, even at the mildest level. The adaptive policy keeps most of the reference behaviour. Errors from the 5 approximated layers compound, so the network-level result falls below the 90% per-layer threshold. Agreement is measured against the exact INT8 model, not against ImageNet labels.

### Verification

| Unit | Method | Result |
|---|---|---|
| `approx_multiplier` (runtime level) | Exhaustive, 256 × 256 × 4 levels vs. golden model | **262,144 / 262,144 pass** |
| `approx_multiplier` (`FIXED_K` = 0/2/4/6) | Exhaustive vs. same golden vectors | **262,144 / 262,144 pass** |
| `adaptive_policy_controller` | All IDs, invalid IDs, `layer_valid=0`, async reset | **71 / 71 pass** |
| `approx_mac_top` | Accumulation and edge-case sequences | PASS |
| `adaptive_mac_system` | Layer 12 → L3 → 15888; layer 3 → L0 → 16129 | PASS |
| Synthesized netlist | Cadence Conformal LEC, RTL vs. netlist | Equivalent |

### Synthesis (Cadence Genus, 10 ns clock)

**GPDK180** (Δ relative to K0):

| Config | Cells | Area | ΔArea | Power (mW) | ΔPower | Setup slack |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| K0 | 490 | 16997.1 | – | 0.954 | – | +0.149 ns |
| K2 | 447 | 10760.9 | −36.7% | 0.937 | −1.8% | +0.217 ns |
| K4 | 479 | 11090.2 | −34.8% | 0.970 | +1.6% | +0.055 ns |
| K6 | 425 | 9606.6 | −43.5% | 0.849 | −11.0% | +0.103 ns |
| Adaptive | 462 | 10988.4 | −35.4% | 0.894 | −6.3% | ≈ +0.162 ns |

**TSMC N65** (FF 0.88 V / SS 0.90 V):

| Config | ΔArea | ΔPower FF | ΔPower SS |
|---|:-:|:-:|:-:|
| K2 | −2.1% | +1.6% | +1.9% |
| K4 | −12.2% | −9.7% | −9.3% |
| K6 | −19.5% | −12.8% | −11.8% |
| Adaptive | −1.4% | −3.8% | −3.9% |

The N65 trends follow K much more regularly than GPDK180, where re-optimization per variant makes area non-monotonic. Power is the tool's default-activity estimate, not workload-driven.

### Physical design (Cadence Innovus, GPDK180, adaptive design)

| Item | Result |
|---|---|
| Core size | ≈ 130.0 µm × 121.0 µm |
| Placement density | ≈ 69.4% |
| Post-CTS WNS / TNS | ≈ +0.027 ns / 0 (0 of 68 paths violating) |
| Routing overflow | 0.00% |
| DRC / connectivity violations | 0 / 0 |
| GDSII stream-out | Completed |

Flow: init (MMMC, slow lib for setup, fast lib for hold) → floorplan → power rings and stripes → placement → CCOpt CTS → NanoRoute → DRC/connectivity → filler → GDSII.

## Repository layout

```
├── rtl/
│   ├── approx_multiplier.sv          # INT8×INT8→16b PP-truncation multiplier (runtime level or FIXED_K)
│   ├── approx_mac_top.sv             # multiplier + 32-bit wrap-around accumulator
│   ├── adaptive_policy_controller.sv # layer_id → approx_level LUT, registered, L0 fail-safe
│   └── adaptive_mac_system.sv        # top: controller + MAC
├── tb/
│   ├── tb_approx_mac.sv              # exhaustive 262,144-vector multiplier check
│   ├── tb_fixed_k.sv                 # exhaustive check of the FIXED_K synthesis variants
│   ├── tb_mac_top.sv                 # accumulator tests
│   ├── tb_adaptive_policy_controller.sv
│   └── tb_adaptive_mac_system.sv     # integration
├── vectors/golden_vectors.txt        # 262,144 golden vectors: "AA BB L PPPP" (hex)
├── model/approx_mul.py               # readable Python reference; self-checks against the vectors
├── flow/                             # Genus, Conformal LEC and Innovus scripts
└── Makefile
```

### `approx_multiplier`: two uses of one module

- `FIXED_K = -1` (default): the level is selected at run time through `approx_level`. All 64 AND gates exist and the low columns are gated off. This is the variant used in the adaptive system.
- `FIXED_K = 0/2/4/6`: the level is fixed at compile time and dropped cells are **never generated**, so the gates are truly removed. Synthesize each K separately for a fair PPA comparison. Tying `approx_level` to a constant in the runtime variant does not give a real comparison.

## Running the simulations

Requires [Icarus Verilog](https://steveicarus.github.io/iverilog/) ≥ 11 (`-g2012`). Run from the repo root, because the testbenches read `vectors/golden_vectors.txt` by relative path.

```bash
make            # all five testbenches
make mult       # exhaustive multiplier (262,144 vectors)
make fixed_k    # exhaustive FIXED_K variants
make mac        # accumulator
make policy     # policy controller (71 checks)
make system     # integration

python3 model/approx_mul.py   # Python reference vs. golden vectors
```

Icarus prints `sorry: Case unique/unique0 qualities are ignored`. This is harmless. The original results were also run in Cadence Xcelium and inspected in SimVision.

The scripts in `flow/` expect a Cadence install and a GPDK180 PDK. Library paths are read from `GPDK180_LIB` / `GPDK180_ROOT`, so set them for your server.

## Limitations and future work

- The network validation uses 100 images and reports agreement with the exact INT8 model. It is preliminary and is not a full ImageNet accuracy evaluation.
- The policy is chosen from isolated-layer measurements, so interaction between approximated layers is not modelled (86% vs. the 90% per-layer target).
- The policy is static and layer-indexed, not input-adaptive.
- Power numbers are synthesis estimates. A ResNet-18 INT8 workload (4 M MAC records) has been generated for switching-activity power, energy/MAC and energy/inference, but that analysis is still pending.
- Next steps: weight the policy by per-layer MAC counts, search other thresholds and layer combinations, report post-route PPA, and integrate the MAC into a full accelerator datapath.

## Reference

Closest related work: Sayadi et al., *"Layer-specific approximate multipliers for energy-precision trade-offs in convolutional neural networks,"* Scientific Reports, 2025. That work selects multipliers per layer from a weight-variance proxy using operand truncation. This project instead measures sensitivity directly and truncates partial-product columns on full-width operands.

---

*A Sanjay, VIT Chennai, 2026* · [github.com/Sanjay1356](https://github.com/Sanjay1356)
