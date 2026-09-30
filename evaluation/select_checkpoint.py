"""Evaluate saved full-team checkpoints against a fixed scripted opponent.

Use this after training to distinguish the final self-play policy from the
best policy reached transiently during a non-stationary run.
"""

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


def checkpoints(run_dir: str) -> list[str]:
    numbered = sorted(
        (f for f in os.listdir(run_dir) if f.startswith("iter_") and f.endswith(".pt")),
        key=lambda f: int(f[5:-3]),
    )
    paths = [os.path.join(run_dir, f) for f in numbered]
    final = os.path.join(run_dir, "final.pt")
    return paths + ([final] if os.path.exists(final) else [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", choices=["1v1", "2v2"], default="2v2")
    ap.add_argument("--env-version", choices=["v1", "v2"], default="v1")
    ap.add_argument("--paddle-overlap", type=float, default=None)
    ap.add_argument("--ball-speed-scale", type=float, default=None)
    ap.add_argument(
        "--opponent",
        choices=["random", "range", "heuristic", "reactive", "predictive"],
        default="reactive",
    )
    ap.add_argument("--opponent-team", choices=["A", "B"], default="B")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=50_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base_cfg = V2_CALIBRATED if args.env_version == "v2" else Config()
    speed_source = Config() if args.ball_speed_scale is not None else base_cfg
    speed_scale = args.ball_speed_scale if args.ball_speed_scale is not None else 1.0
    cfg = dataclasses.replace(
        base_cfg,
        mode=args.mode,
        env_version=args.env_version,
        collision_resolution="closest_paddle" if args.env_version == "v2" else "sequential_id",
        paddle_overlap=(
            args.paddle_overlap if args.paddle_overlap is not None else base_cfg.paddle_overlap
        ),
        serve_speed_min=speed_source.serve_speed_min * speed_scale,
        serve_speed_max=speed_source.serve_speed_max * speed_scale,
        ball_speed_max=speed_source.ball_speed_max * speed_scale,
    )
    rows = []
    for path in checkpoints(args.run_dir):
        state = torch.load(path, map_location="cpu", weights_only=True)
        out = evaluate_weights(
            cfg,
            state,
            args.episodes,
            args.seed,
            opponent=args.opponent,
            opponent_team=args.opponent_team,
        )
        out["checkpoint"] = os.path.basename(path)
        rows.append(out)
        print(
            f"{out['checkpoint']}: return_A={out['mean_return_A']:.2f} win_A={out['win_rate_A']:.2f}"
        )
    best = max(rows, key=lambda r: r["mean_return_A"])
    output = {
        "criterion": "maximum mean_return_A",
        "opponent": args.opponent,
        "rows": rows,
        "best": best,
    }
    out_path = args.out or os.path.join(args.run_dir, f"checkpoint_eval_vs_{args.opponent}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"best={best['checkpoint']} -> {out_path}")


if __name__ == "__main__":
    main()
