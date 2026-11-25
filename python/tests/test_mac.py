"""
Correctness tests. Run with:  python -m tests.test_mac
(run from the adaptive_approx_mac/ directory)

Demonstrates correctness rather than asserting it silently: prints a report
of total/passed/failed tests and representative failures.
"""
import random
import sys

from config import config as cfg
from models.multiplier import golden_multiply, bitlevel_multiply
from models.exact_mac import exact_mac_step, exact_mac_sequence


def report(name, total, failed):
    status = "PASS" if not failed else "FAIL"
    print(f"[{status}] {name}: {total - len(failed)}/{total} passed")
    for f in failed[:5]:
        print(f"    FAILURE: {f}")
    return len(failed) == 0


def test_golden_vs_bitlevel_exhaustive():
    failed = []
    total = 0
    for a in range(cfg.OPERAND_MIN, cfg.OPERAND_MAX + 1):
        for b in range(cfg.OPERAND_MIN, cfg.OPERAND_MAX + 1):
            total += 1
            g = golden_multiply(a, b)
            bl = bitlevel_multiply(a, b)
            if g != bl:
                failed.append(f"a={a} b={b} golden={g} bitlevel={bl}")
    return report("golden_multiply vs bitlevel_multiply (exhaustive 65536 pairs)", total, failed)


def test_known_edge_cases():
    cases = [
        (0, 0, 0), (1, 1, 1), (-1, -1, 1), (-1, 1, -1),
        (127, 127, 16129), (-128, 127, -16256), (127, -128, -16256),
        (-128, -128, 16384),
    ]
    failed = []
    for a, b, expected in cases:
        g = golden_multiply(a, b)
        bl = bitlevel_multiply(a, b)
        if g != expected or bl != expected:
            failed.append(f"a={a} b={b} expected={expected} golden={g} bitlevel={bl}")
    return report("known edge-case products", len(cases), failed)


def test_mac_accumulation_matches_python_sum():
    random.seed(cfg.RANDOM_SEED)
    failed = []
    total = 0
    for _ in range(200):
        n = random.randint(1, 32)
        a_list = [random.randint(cfg.OPERAND_MIN, cfg.OPERAND_MAX) for _ in range(n)]
        b_list = [random.randint(cfg.OPERAND_MIN, cfg.OPERAND_MAX) for _ in range(n)]
        steps = exact_mac_sequence(a_list, b_list)
        total += 1
        expected_math = sum(a * b for a, b in zip(a_list, b_list))
        final = steps[-1]
        if not final.overflowed and final.acc_hw != expected_math:
            failed.append(
                f"n={n} expected={expected_math} got={final.acc_hw} "
                f"(no overflow flagged but mismatch)"
            )
        # when overflow occurs the wrapped value must equal expected_math
        # reduced mod 2**ACC_WIDTH into the signed range
        from models.multiplier import clip_to_width
        if cfg.ACC_OVERFLOW_POLICY == "wrap":
            expected_wrapped = clip_to_width(expected_math, cfg.ACC_WIDTH, signed=True)
            if final.acc_hw != expected_wrapped:
                failed.append(
                    f"n={n} wrap mismatch: expected={expected_wrapped} got={final.acc_hw}"
                )
    return report("MAC accumulation vs independent Python sum (200 random sequences)", total, failed)


def test_acc_overflow_boundaries():
    failed = []
    total = 3
    # force accumulator right up to the boundary then push it over
    near_max = cfg.ACC_MAX - 10
    r = exact_mac_step(near_max, 127, 127)  # +16129, will overflow
    if cfg.ACC_OVERFLOW_POLICY == "wrap":
        if not r.overflowed:
            failed.append("expected overflow flag near ACC_MAX, got none")
    r2 = exact_mac_step(0, 0, 0)
    if r2.overflowed:
        failed.append("acc=0 + 0*0 incorrectly flagged as overflow")
    r3 = exact_mac_step(cfg.ACC_MIN, -128, 127)  # large negative product, near min
    # just check it runs and produces an in-range hw value
    if not (cfg.ACC_MIN <= r3.acc_hw <= cfg.ACC_MAX):
        failed.append(f"acc_hw {r3.acc_hw} out of representable range")
    return report("accumulator overflow boundary behavior", total, failed)


def main():
    results = [
        test_golden_vs_bitlevel_exhaustive(),
        test_known_edge_cases(),
        test_mac_accumulation_matches_python_sum(),
        test_acc_overflow_boundaries(),
    ]
    print()
    if all(results):
        print("ALL TEST SUITES PASSED")
        sys.exit(0)
    else:
        print("SOME TEST SUITES FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
