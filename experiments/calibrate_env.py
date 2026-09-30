"""Scripted-only v2 environment calibration; never trains PPO.

Sweeps predeclared overlap and speed candidates, then records baseline-ladder
match statistics. Select v2 settings from reachability and scripted behavior,
not from PPO performance, and freeze the chosen Config before main training.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from baselines.agents import HeuristicAgent, PredictiveAgent, RandomAgent, RangeAwareAgent
from environment.config import V2_DEFAULT, Config
from environment.pong_env import PongEnv


def parse_csv_floats(value: str) -> list[float]:
    return [float(v) for v in value.split(",")]


def make_agent(kind: str, agent_id: str, env: PongEnv, seed: int):
    lo, hi = env._allowed_range(agent_id)
    home = (lo + hi) / 2.0
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "range":
        return RangeAwareAgent(home=home)
    if kind == "reactive":
        return HeuristicAgent(team=agent_id[0], home=home)
    if kind == "predictive":
        paddle_x = -env.cfg.paddle_x_offset if agent_id.startswith("A") else env.cfg.paddle_x_offset
        return PredictiveAgent(agent_id[0], paddle_x, env.cfg.ball_radius, home=home)
    raise ValueError(f"unknown scripted policy {kind!r}")


def play(cfg: Config, team_a: str, team_b: str, episodes: int, seed: int) -> dict:
    env = PongEnv(config=cfg, seed=seed)
    agents = {
        a: make_agent(team_a if a.startswith("A") else team_b, a, env, seed + i)
        for i, a in enumerate(env.agent_ids)
    }
    scores_a, scores_b, lengths, draws = [], [], [], 0
    contacts = {a: [] for a in env.agent_ids}
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            actions = {a: agents[a].act(o) for a, o in obs.items()}
            obs, _, term, trunc, info = env.step(actions)
            done = all(term.values()) or all(trunc.values())
        scores_a.append(info["scores"]["A"])
        scores_b.append(info["scores"]["B"])
        lengths.append(info["steps"])
        draws += int(info["scores"]["A"] == info["scores"]["B"])
        for a in env.agent_ids:
            contacts[a].append(info["hits"][a])
    a_wins = sum(a > b for a, b in zip(scores_a, scores_b))
    b_wins = sum(b > a for a, b in zip(scores_a, scores_b))
    return {
        "team_A": team_a,
        "team_B": team_b,
        "episodes": episodes,
        "win_A": a_wins / episodes,
        "win_B": b_wins / episodes,
        "draw": draws / episodes,
        "mean_points_A": float(np.mean(scores_a)),
        "mean_points_B": float(np.mean(scores_b)),
        "mean_length": float(np.mean(lengths)),
        "mean_contacts": {a: float(np.mean(v)) for a, v in contacts.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlaps", default="0.15,0.25,0.35,0.45")
    ap.add_argument("--speed-scales", default="1.0")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="results/calibration_v2.json")
    args = ap.parse_args()

    rows = []
    for overlap in parse_csv_floats(args.overlaps):
        for scale in parse_csv_floats(args.speed_scales):
            cfg = dataclasses.replace(
                V2_DEFAULT,
                paddle_overlap=overlap,
                serve_speed_min=V2_DEFAULT.serve_speed_min * scale,
                serve_speed_max=V2_DEFAULT.serve_speed_max * scale,
                ball_speed_max=V2_DEFAULT.ball_speed_max * scale,
            )
            # Same-policy games calibrate the ladder without PPO selection.
            games = [
                play(cfg, "range", "range", args.episodes, args.seed),
                play(cfg, "reactive", "reactive", args.episodes, args.seed),
                play(cfg, "predictive", "predictive", args.episodes, args.seed),
                play(cfg, "reactive", "random", args.episodes, args.seed),
            ]
            rows.append({"config": dataclasses.asdict(cfg), "games": games})
            print(
                f"overlap={overlap:.2f} speed_scale={scale:.2f} "
                f"reactive draw={games[1]['draw']:.2f} len={games[1]['mean_length']:.0f} "
                f"reactive-v-random Awin={games[3]['win_A']:.2f}",
                flush=True,
            )
    with open(args.out, "w") as f:
        json.dump({"purpose": "scripted-only v2 calibration", "candidates": rows}, f, indent=2)
    print(f"wrote {args.out}; choose and freeze one candidate before PPO training")


if __name__ == "__main__":
    main()
