"""
Reproduce everything with one command (works on Windows, macOS and Linux):

    python run_pipeline.py                 # data -> 4 models -> benchmark
    python run_pipeline.py --from distill  # resume from a given step

Each step is a separate module that can also be run on its own
(python -m src.<step>).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

STEPS = [
    ("data", "src.data_prep"),
    ("baseline", "src.train_baseline"),
    ("nn", "src.train_nn"),
    ("transformer", "src.train_transformer"),
    ("distill", "src.distill_quantize"),
    ("evaluate", "src.evaluate"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="data", choices=[s for s, _ in STEPS])
    ap.add_argument("--only", default=None, choices=[s for s, _ in STEPS])
    args = ap.parse_args()

    names = [s for s, _ in STEPS]
    selected = [STEPS[names.index(args.only)]] if args.only else STEPS[names.index(args.start):]
    for name, module in selected:
        print(f"\n{'=' * 70}\n>>> step '{name}': python -m {module}\n{'=' * 70}", flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable, "-m", module])
        if rc != 0:
            sys.exit(f"step '{name}' failed with exit code {rc}")
        print(f">>> step '{name}' finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
