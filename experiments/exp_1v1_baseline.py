"""1v1 baseline: sanity check that PPO learns basic paddle control.

Not a matched comparison to 2v2 (different geometry) — see TASK.md.
Usage: uv run python experiments/exp_1v1_baseline.py [--seed N] [--timesteps T]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.train import main

if __name__ == "__main__":
    main(["--mode", "1v1", *sys.argv[1:]])
