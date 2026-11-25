"""
Run the full pre-DNN pipeline in order:
  1. correctness tests
  2. exhaustive multiplier analysis
  3. arithmetic error characterization

Run from adaptive_approx_mac/:  python run_all.py
"""
import sys

from tests import test_mac
from tests import test_pp_truncation
from experiments import exhaustive_analysis
from experiments import error_characterization
from experiments import exhaustive_pp_characterization
from experiments import compare_old_new
from experiments import mac_level_test


def main():
    print("=" * 60)
    print("STEP 1/3: correctness tests")
    print("=" * 60)
    try:
        test_mac.main()
    except SystemExit as e:
        if e.code != 0:
            print("Correctness tests failed -- stopping before further steps.")
            sys.exit(e.code)

    print()
    print("=" * 60)
    print("STEP 2/3: exhaustive multiplier analysis")
    print("=" * 60)
    ok = exhaustive_analysis.run()
    if not ok:
        print("Exhaustive analysis found mismatches -- stopping.")
        sys.exit(1)

    print()
    print("=" * 60)
    print("STEP 3/3: arithmetic error characterization")
    print("=" * 60)
    error_characterization.run()

    print()
    print("=" * 60)
    print("STEP 4/6: PP-truncation correctness tests")
    print("=" * 60)
    try:
        test_pp_truncation.main()
    except SystemExit as e:
        if e.code != 0:
            sys.exit(e.code)

    print()
    print("=" * 60)
    print("STEP 5/6: exhaustive PP-column-truncation characterization")
    print("=" * 60)
    exhaustive_pp_characterization.run()

    print()
    print("=" * 60)
    print("STEP 6/6: old-vs-new comparison + MAC-level test")
    print("=" * 60)
    compare_old_new.run()
    print()
    mac_level_test.run()

    print()
    print("Pipeline complete. See results/ and plots/.")


if __name__ == "__main__":
    main()
