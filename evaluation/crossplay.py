"""Cross-play matrix across independently trained full-team checkpoints."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from environment.config import V2_CALIBRATED, Config
from evaluation.evaluate import evaluate_weights


def checkpoint_path(run: str) -> str:
    return run if run.endswith(".pt") else os.path.join("results", "models", run, "final.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run names or final.pt paths")
    ap.add_argument("--mode", choices=["2v2"], default="2v2")
    ap.add_argument("--env-version", choices=["v1", "v2"], default="v1")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=80_000)
    ap.add_argument("--out", default="results/crossplay.json")
    args = ap.parse_args()

    base = V2_CALIBRATED if args.env_version == "v2" else Config()
    cfg = dataclasses.replace(
        base,
        mode=args.mode,
        collision_resolution="closest_paddle" if args.env_version == "v2" else "sequential_id",
    )
    loaded = {
        run: torch.load(checkpoint_path(run), map_location="cpu", weights_only=True)
        for run in args.runs
    }
    rows = []
    for a_name, a_state in loaded.items():
        for b_name, b_state in loaded.items():
            state = {a: a_state[a] for a in ("A1", "A2")}
            state.update({b: b_state[b] for b in ("B1", "B2")})
            out = evaluate_weights(cfg, state, episodes=args.episodes, base_seed=args.seed)
            rows.append(
                {
                    "team_A_run": a_name,
                    "team_B_run": b_name,
                    "win_rate_A": out["win_rate_A"],
                    "win_rate_B": out["win_rate_B"],
                    "draw_rate": out["draw_rate"],
                    "mean_return_A": out["mean_return_A"],
                    "mean_ep_len": out["mean_ep_len"],
                }
            )
            print(
                f"A={a_name} B={b_name}: Awin={out['win_rate_A']:.2f} Bwin={out['win_rate_B']:.2f}"
            )
    with open(args.out, "w") as f:
        json.dump(
            {
                "mode": args.mode,
                "env_version": args.env_version,
                "episodes": args.episodes,
                "rows": rows,
            },
            f,
            indent=2,
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
