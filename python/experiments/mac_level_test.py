"""
MAC-level test: run random accumulation sequences through the exact MAC
and the new approximate MAC (pp_column_truncation) and measure
accumulated error -- NOT per-multiply error, per-sequence final error.

Run from adaptive_approx_mac/: python -m experiments.mac_level_test
"""
import csv
import os
import random

import numpy as np

from config import config as cfg
from models.exact_mac import exact_mac_sequence
from models.approximation import approx_mac_step


def run_sequence_approx(a_list, b_list, level, acc_init=0):
    acc = acc_init
    for a, b in zip(a_list, b_list):
        acc = approx_mac_step(acc, a, b, level)
    return acc


def run():
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    random.seed(cfg.RANDOM_SEED)

    n_sequences = cfg.N_RANDOM_MAC_SEQUENCES
    seq_len = cfg.MAC_SEQUENCE_LENGTH

    per_level_final_err = {l: [] for l in range(4)}

    rows = []
    for seq_idx in range(n_sequences):
        a_list = [random.randint(cfg.OPERAND_MIN, cfg.OPERAND_MAX) for _ in range(seq_len)]
        b_list = [random.randint(cfg.OPERAND_MIN, cfg.OPERAND_MAX) for _ in range(seq_len)]
        exact_steps = exact_mac_sequence(a_list, b_list)
        exact_final = exact_steps[-1].acc_hw

        row = {"sequence": seq_idx, "exact_final": exact_final}
        for level in range(4):
            approx_final = run_sequence_approx(a_list, b_list, level)
            err = exact_final - approx_final
            per_level_final_err[level].append(err)
            row[f"L{level}_approx_final"] = approx_final
            row[f"L{level}_final_err"] = err
        rows.append(row)

    csv_path = os.path.join(cfg.RESULTS_DIR, "mac_level_results.csv")
    with open(csv_path, "w", newline="") as f:
        import csv as _csv
        writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"MAC-level test: {n_sequences} sequences x {seq_len} operations each")
    for level in range(4):
        errs = np.array(per_level_final_err[level])
        print(f"  L{level}: mean_final_err={errs.mean():.2f} "
              f"|mean|={np.abs(errs).mean():.2f} "
              f"max_abs_final_err={np.abs(errs).max():.0f} "
              f"std={errs.std():.2f}")

    print()
    print("NOTE: mean_final_err growing roughly linearly with sequence length "
          "(vs. per-multiply mean_signed_error) would indicate the per-multiply "
          "bias documented in pp_truncation.py is NOT self-cancelling across "
          "accumulation -- check this before trusting these levels for a real "
          "dot-product-sized (hundreds+ term) accumulation.")
    print(f"Full per-sequence results: {csv_path}")

    # explicit edge cases
    print()
    print("Edge cases:")
    edge_cases = [
        ("all zero", [0]*8, [0]*8),
        ("all max", [127]*8, [127]*8),
        ("all min", [-128]*8, [-128]*8),
        ("alternating extremes", [127, -128]*4, [-128, 127]*4),
    ]
    for name, a_list, b_list in edge_cases:
        exact_final = exact_mac_sequence(a_list, b_list)[-1].acc_hw
        for level in range(4):
            approx_final = run_sequence_approx(a_list, b_list, level)
            print(f"  {name} L{level}: exact={exact_final} approx={approx_final} "
                  f"err={exact_final - approx_final}")


if __name__ == "__main__":
    run()
