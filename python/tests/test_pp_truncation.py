"""
Correctness tests for models/pp_truncation.py.
Run with: python -m tests.test_pp_truncation
"""
import sys

from config import config as cfg
from models.multiplier import golden_multiply
from models.pp_truncation import pp_terms, level_multiply, LEVEL_TO_K, column_population


def report(name, total, failed):
    status = "PASS" if not failed else "FAIL"
    print(f"[{status}] {name}: {total - len(failed)}/{total} passed")
    for f in failed[:5]:
        print(f"    FAILURE: {f}")
    return len(failed) == 0


def test_pp_terms_sum_to_exact_exhaustive():
    failed = []
    total = 0
    for a in range(cfg.OPERAND_MIN, cfg.OPERAND_MAX + 1):
        for b in range(cfg.OPERAND_MIN, cfg.OPERAND_MAX + 1):
            total += 1
            s = sum(t[3] for t in pp_terms(a, b))
            g = golden_multiply(a, b)
            if s != g:
                failed.append(f"a={a} b={b} pp_sum={s} golden={g}")
    return report("sum of all 64 PP terms == golden_multiply (exhaustive)", total, failed)


def test_level0_is_bit_exact():
    failed = []
    total = 0
    for a in (-128, -1, 0, 1, 127, 42, -77):
        for b in (-128, -1, 0, 1, 127, 42, -77):
            total += 1
            approx = level_multiply(a, b, 0)
            exact = golden_multiply(a, b)
            if approx != exact:
                failed.append(f"a={a} b={b} L0 approx={approx} exact={exact}")
    return report("L0 (k=0) is bit-exact", total, failed)


def test_column_population_matches_known_sequence():
    pops = column_population(8)
    expected = [1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]
    failed = [] if pops == expected else [f"got {pops}, expected {expected}"]
    return report("8x8 PP column population matches 1..8..1", 1, failed)


def test_k_never_exceeds_sign_safe_bound():
    failed = []
    for level, k in LEVEL_TO_K.items():
        if k > cfg.OPERAND_WIDTH - 1:
            failed.append(f"level {level} uses k={k}, exceeds sign-safe bound {cfg.OPERAND_WIDTH-1}")
    return report("all configured k values stay in sign-safe range", len(LEVEL_TO_K), failed)


def main():
    results = [
        test_pp_terms_sum_to_exact_exhaustive(),
        test_level0_is_bit_exact(),
        test_column_population_matches_known_sequence(),
        test_k_never_exceeds_sign_safe_bound(),
    ]
    print()
    if all(results):
        print("ALL PP-TRUNCATION TESTS PASSED")
        sys.exit(0)
    else:
        print("SOME PP-TRUNCATION TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
