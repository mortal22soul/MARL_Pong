"""2v2 main MARL condition: all four agents learn concurrently.

Usage: uv run python experiments/exp_2v2_main.py [--seed N] [--timesteps T]
Run once per seed (0, 1, 2) for the 3-seed protocol in TASK.md.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.train import main

if __name__ == "__main__":
    main(["--mode", "2v2", *sys.argv[1:]])
