"""Uniform evaluation pass over trained runs.

Plays the same episode count for every run against self, heuristic, and
random opposition so cross-seed and cross-condition comparisons are not
confounded by differing eval sizes. Writes results/eval_<run>.json (self)
and results/eval_<run>_vs_<opponent>.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.config import Config
from evaluation.evaluate import evaluate_weights


def eval_run(weights: str, episodes: int, base_seed: int = 1000) -> list[str]:
    """Evaluate one checkpoint against all three opponent kinds."""
    import torch

    run = os.path.basename(os.path.dirname(weights))
    mode = "1v1" if "1v1" in run else "2v2"
    cfg = dataclasses.replace(Config(), mode=mode)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    written = []
    for opponent in ("self", "heuristic", "random"):
        suffix = "" if opponent == "self" else f"_vs_{opponent}"
        out_path = os.path.join("results", f"eval_{run}{suffix}.json")
        out = evaluate_weights(
            cfg, state, episodes=episodes, base_seed=base_seed, opponent=opponent
        )
        out["opponent"] = opponent
        out["run"] = run
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        written.append(out_path)
        print(
            f"{run} vs {opponent}: win A={out['win_rate_A']:.2f} "
            f"B={out['win_rate_B']:.2f} len={out['mean_ep_len']:.0f}"
        )
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run names or final.pt paths")
    ap.add_argument("--episodes", type=int, default=60)
    args = ap.parse_args()
    for run in args.runs:
        weights = run if run.endswith(".pt") else os.path.join("results", "models", run, "final.pt")
        eval_run(weights, args.episodes)


if __name__ == "__main__":
    main()
