"""Run all automated Morph tests in order: export -> terrain -> batch.

Skips the interactive visualize.py (it opens a window).

Run:
    python tests/run_all.py
"""
import subprocess
import sys
import time

from _common import PY, HERE

TESTS = ["test_export.py", "test_terrain.py", "test_batch.py"]


def main():
    print("=" * 70)
    print("Morph end-to-end test suite")
    print("=" * 70)
    t0 = time.time()
    for t in TESTS:
        print(f"\n>>> Running {t}")
        r = subprocess.run([PY, str(HERE / t)], text=True)
        if r.returncode != 0:
            print(f"\nFAILED at {t} (exit {r.returncode})")
            sys.exit(r.returncode)
    print(f"\n{'=' * 70}")
    print(f"ALL TESTS PASSED in {time.time()-t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
