"""Deterministic evaluation for trained marl-pong teams.

Loads a checkpoint (all agents), plays greedy episodes, and reports:
win rate, average team return, return variance, rally length, plus teammate
positional specialization (per-paddle mean/std/range of y and the overlap
between teammates' coverage histograms). Trajectories are saved to .npz for
heatmap plotting. Win rate alone is NOT treated as coordination evidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agents.multi_agent_ppo import IndependentPPO, PPOConfig
from environment.config import Config
from environment.pong_env import PongEnv


def hist_overlap(y1: np.ndarray, y2: np.ndarray, bins: int = 20) -> float:
    h1, edges = np.histogram(y1, bins=bins, range=(-1, 1))
    h2, _ = np.histogram(y2, bins=bins, range=(edges[0], edges[-1]))
    denom = 0.5 * (h1.sum() + h2.sum())
    return float(np.minimum(h1, h2).sum() / denom) if denom > 0 else 0.0


def evaluate_weights(
    cfg: Config,
    state: dict,
    episodes: int = 20,
    base_seed: int = 1000,
    save_npz: str | None = None,
    opponent: str = "self",
) -> dict:
    """opponent: 'self' (both teams trained), 'heuristic' / 'random' (team B replaced)."""
    from baselines.agents import HeuristicAgent, RandomAgent

    env = PongEnv(config=cfg, seed=base_seed)
    trainer = IndependentPPO(env.agent_ids, cfg=PPOConfig())
    subs = {}
    if opponent != "self":
        for i, b in enumerate([a for a in env.agent_ids if a.startswith("B")]):
            lo, hi = env._allowed_range(b)
            subs[b] = (
                HeuristicAgent(team="B", home=(lo + hi) / 2.0)
                if opponent == "heuristic"
                else RandomAgent(seed=500 + i)
            )
    for a in trainer.ids:
        trainer.nets[a].load_state_dict(state[a])
        trainer.nets[a].eval()
    wins = {"A": 0, "B": 0, "draw": 0}
    returns, lens = [], []
    trajs: dict[str, list[np.ndarray]] = {a: [] for a in env.agent_ids}
    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        ys: dict[str, list[float]] = {a: [] for a in env.agent_ids}
        done = False
        while not done:
            actions = {}
            for a, o in obs.items():
                actions[a] = subs[a].act(o) if a in subs else trainer.nets[a].greedy(o)
            for a in env.agent_ids:
                ys[a].append(env.paddles[a][0])
            obs, _, term, trunc, info = env.step(actions)
            done = all(term.values()) or all(trunc.values())
        for a in env.agent_ids:
            trajs[a].append(np.asarray(ys[a], dtype=np.float32))
        sa, sb = info["scores"]["A"], info["scores"]["B"]
        returns.append(sa - sb)
        lens.append(info["steps"])
        wins["A" if sa > sb else ("B" if sb > sa else "draw")] += 1
    spec: dict[str, dict] = {}
    for team in ("A", "B"):
        mates = [a for a in env.agent_ids if a.startswith(team)]
        pooled = {a: np.concatenate(trajs[a]) for a in mates}
        spec[team] = {
            a: {
                "mean_y": float(y.mean()),
                "std_y": float(y.std()),
                "min_y": float(y.min()),
                "max_y": float(y.max()),
            }
            for a, y in pooled.items()
        }
        if len(mates) == 2:
            spec[team]["coverage_overlap"] = hist_overlap(pooled[mates[0]], pooled[mates[1]])
    out = {
        "mode": cfg.mode,
        "episodes": episodes,
        "win_rate_A": wins["A"] / episodes,
        "win_rate_B": wins["B"] / episodes,
        "draw_rate": wins["draw"] / episodes,
        "mean_return_A": float(np.mean(returns)),
        "var_return_A": float(np.var(returns)),
        "mean_ep_len": float(np.mean(lens)),
        "specialization": spec,
    }
    if save_npz:
        np.savez_compressed(
            save_npz, **{f"{a}_{i}": t for a in trajs for i, t in enumerate(trajs[a])}
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["1v1", "2v2"], default="2v2")
    ap.add_argument("--weights", required=True, help="checkpoint .pt holding all agents")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--points", type=int, default=5)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-npz", default=None)
    ap.add_argument("--opponent", choices=["self", "heuristic", "random"], default="self")
    args = ap.parse_args()
    import torch

    cfg = dataclasses.replace(Config(), mode=args.mode, points_to_win=args.points)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    out = evaluate_weights(cfg, state, args.episodes, args.seed, args.out_npz, args.opponent)
    print(json.dumps(out, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
