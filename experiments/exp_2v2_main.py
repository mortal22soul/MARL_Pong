"""2v2 main MARL condition: all four agents learn concurrently.

Usage: uv run python experiments/exp_2v2_main.py [--seed N] [--timesteps T]
Defaults reproduce the reported run (5M steps, seed 0, see README).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.train import main

if __name__ == "__main__":
    main(sys.argv[1:])
