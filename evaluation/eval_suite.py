"""Uniform evaluation pass over trained runs.

Plays the same episode count for every run against self, heuristic, and
random opposition so cross-seed and cross-condition comparisons are not
confounded by differing eval sizes. Writes results/eval_<run>.json (self)
and results/eval_<run>_vs_<opponent>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.config import Config
from evaluation.evaluate import evaluate_weights


def eval_run(
    weights: str,
    episodes: int,
    base_seed: int = 1000,
    paired: bool = False,
) -> list[str]:
    """Evaluate one checkpoint against self, heuristic, and random opposition."""
    import torch

    run = os.path.basename(os.path.dirname(weights))
    cfg = Config()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    written = []
    jobs = [("self", "B")]
    for opponent in ("heuristic", "random"):
        jobs.append((opponent, "B"))
        if paired:
            jobs.append((opponent, "A"))
    for opponent, team in jobs:
        suffix = "" if opponent == "self" else f"_vs_{opponent}"
        if opponent != "self" and team == "A":
            suffix += "_teamA"
        out_path = os.path.join("results", f"eval_{run}{suffix}.json")
        out = evaluate_weights(
            cfg,
            state,
            episodes=episodes,
            base_seed=base_seed,
            opponent=opponent,
            opponent_team=team,
        )
        out["run"] = run
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        written.append(out_path)
        print(
            f"{run} vs {opponent} team {team}: win A={out['win_rate_A']:.2f} "
            f"B={out['win_rate_B']:.2f} len={out['mean_ep_len']:.0f}"
        )
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run names or final.pt paths")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument(
        "--paired",
        action="store_true",
        help="also substitute scripted Team A for side-balanced tests",
    )
    args = ap.parse_args()
    for run in args.runs:
        weights = run if run.endswith(".pt") else os.path.join("results", "models", run, "final.pt")
        eval_run(weights, args.episodes, paired=args.paired)


if __name__ == "__main__":
    main()
