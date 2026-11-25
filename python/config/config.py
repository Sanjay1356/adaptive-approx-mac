"""
Single source of truth for experimental parameters.
Change values here, not inside individual scripts.
"""

# ---- Numerical format ----
OPERAND_WIDTH = 8          # signed INT8 operands (A, B)
PRODUCT_WIDTH = 16         # signed INT16 full-precision product
ACC_WIDTH = 32             # signed INT32 accumulator

OPERAND_MIN = -(2 ** (OPERAND_WIDTH - 1))        # -128
OPERAND_MAX = 2 ** (OPERAND_WIDTH - 1) - 1        # 127

PRODUCT_MIN = -(2 ** (PRODUCT_WIDTH - 1))
PRODUCT_MAX = 2 ** (PRODUCT_WIDTH - 1) - 1

ACC_MIN = -(2 ** (ACC_WIDTH - 1))
ACC_MAX = 2 ** (ACC_WIDTH - 1) - 1

# ---- Overflow policy ----
# "wrap"      -> two's-complement wraparound (matches a plain RTL adder with no saturation logic)
# "saturate"  -> clamp to [ACC_MIN, ACC_MAX] (requires extra saturation logic in RTL)
ACC_OVERFLOW_POLICY = "wrap"

# ---- Reproducibility ----
RANDOM_SEED = 42

# ---- Experiment sizes ----
N_RANDOM_MAC_SEQUENCES = 2000     # random accumulation sequences for MAC-level testing
MAC_SEQUENCE_LENGTH = 64          # operands accumulated per sequence (mimics a dot product)
N_RANDOM_ERROR_SAMPLES = 200_000  # random (A,B) pairs used for error characterization plots

# ---- Approximation levels to evaluate ----
# NOTE: mechanism + levels are NOT frozen. See models/approximation.py.
# This list only controls which candidate configurations experiments will loop over.
APPROX_LEVELS = ["L0", "L1", "L2", "L3"]

# ---- Output locations ----
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
