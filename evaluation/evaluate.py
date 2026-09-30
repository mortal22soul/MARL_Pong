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
from agents.policies import ActorCritic
from baselines.agents import HeuristicAgent, PredictiveAgent, RandomAgent, RangeAwareAgent
from environment.config import V2_CALIBRATED, Config
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
    opponent_team: str = "B",
) -> dict:
    """Evaluate full or partial checkpoints against a scripted team.

    ``opponent_team`` permits paired side-swapped tests. A partial checkpoint
    is valid only when every missing agent belongs to the scripted team.
    """
    if opponent_team not in ("A", "B"):
        raise ValueError("opponent_team must be 'A' or 'B'")
    env = PongEnv(config=cfg, seed=base_seed)
    subs = {}
    if opponent != "self":
        for i, b in enumerate([a for a in env.agent_ids if a.startswith(opponent_team)]):
            lo, hi = env._allowed_range(b)
            subs[b] = (
                RandomAgent(seed=500 + i)
                if opponent == "random"
                else (
                    RangeAwareAgent(home=(lo + hi) / 2.0)
                    if opponent == "range"
                    else (
                        HeuristicAgent(team=opponent_team, home=(lo + hi) / 2.0)
                        if opponent in ("heuristic", "reactive")
                        else PredictiveAgent(
                            opponent_team,
                            (
                                -env.cfg.paddle_x_offset
                                if opponent_team == "A"
                                else env.cfg.paddle_x_offset
                            ),
                            env.cfg.ball_radius,
                            home=(lo + hi) / 2.0,
                        )
                    )
                )
            )
    learned_ids = list(state)
    if opponent == "self" and set(learned_ids) != set(env.agent_ids):
        raise ValueError("self-play evaluation requires weights for every agent")
    missing = set(env.agent_ids) - set(learned_ids) - set(subs)
    if missing:
        raise ValueError(f"weights/baseline do not supply agents: {sorted(missing)}")
    trainer = IndependentPPO(
        env.agent_ids, cfg=PPOConfig(), trainable_ids=learned_ids, opponents=subs
    )
    for a in trainer.ids:
        sd = ActorCritic._load_state_dict_compat(state[a])
        trainer.nets[a].load_state_dict(sd)
        trainer.nets[a].eval()
    wins = {"A": 0, "B": 0, "draw": 0}
    returns, lens = [], []
    trajs: dict[str, list[np.ndarray]] = {a: [] for a in env.agent_ids}
    defending_y: dict[str, list[float]] = {a: [] for a in env.agent_ids}
    defending_error: dict[str, list[float]] = {a: [] for a in env.agent_ids}
    action_counts = {a: np.zeros(3, dtype=np.int64) for a in env.agent_ids}
    contacts = {a: 0 for a in env.agent_ids}
    region_contacts = {a: {"upper": 0, "middle": 0, "lower": 0} for a in env.agent_ids}
    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        ys: dict[str, list[float]] = {a: [] for a in env.agent_ids}
        done = False
        while not done:
            actions = {}
            for a, o in obs.items():
                actions[a] = subs[a].act(o) if a in subs else trainer.nets[a].greedy(o)
                action_counts[a][actions[a]] += 1
            for a in env.agent_ids:
                ys[a].append(env.paddles[a][0])
            defending_team = "A" if env.ball[2] < 0 else "B"
            for a in env.agent_ids:
                if a.startswith(defending_team):
                    defending_y[a].append(env.paddles[a][0])
                    defending_error[a].append(abs(env.paddles[a][0] - env.ball[1]))
            obs, _, term, trunc, info = env.step(actions)
            done = all(term.values()) or all(trunc.values())
        for a in env.agent_ids:
            trajs[a].append(np.asarray(ys[a], dtype=np.float32))
        sa, sb = info["scores"]["A"], info["scores"]["B"]
        returns.append(sa - sb)
        lens.append(info["steps"])
        for a in env.agent_ids:
            contacts[a] += info["hits"][a]
            for region, count in info["contact_regions"][a].items():
                region_contacts[a][region] += count
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
            spec[team]["home_separation"] = abs(
                spec[team][mates[0]]["mean_y"] - spec[team][mates[1]]["mean_y"]
            )
    behavior = {}
    for a in env.agent_ids:
        total_actions = int(action_counts[a].sum())
        behavior[a] = {
            "action_probs": (
                (action_counts[a] / total_actions).tolist() if total_actions else [0.0, 0.0, 0.0]
            ),
            "defending_mean_y": float(np.mean(defending_y[a])) if defending_y[a] else None,
            "defending_ball_error": (
                float(np.mean(defending_error[a])) if defending_error[a] else None
            ),
            "defending_samples": len(defending_y[a]),
            "contacts": contacts[a],
            "contacts_per_episode": contacts[a] / episodes,
            "contact_regions": dict(region_contacts[a]),
            "contact_region_share": {
                region: region_contacts[a][region] / contacts[a] if contacts[a] else 0.0
                for region in ("upper", "middle", "lower")
            },
        }
    for team in ("A", "B"):
        mates = [a for a in env.agent_ids if a.startswith(team)]
        total_contacts = sum(contacts[a] for a in mates)
        for a in mates:
            behavior[a]["team_contact_share"] = (
                contacts[a] / total_contacts if total_contacts else 0.0
            )
    out = {
        "mode": cfg.mode,
        "episodes": episodes,
        "win_rate_A": wins["A"] / episodes,
        "win_rate_B": wins["B"] / episodes,
        "draw_rate": wins["draw"] / episodes,
        "mean_return_A": float(np.mean(returns)),
        "var_return_A": float(np.var(returns)),
        "mean_ep_len": float(np.mean(lens)),
        "opponent": opponent,
        "opponent_team": opponent_team if opponent != "self" else None,
        "specialization": spec,
        "behavior": behavior,
    }
    if save_npz:
        np.savez_compressed(
            save_npz, **{f"{a}_{i}": t for a in trajs for i, t in enumerate(trajs[a])}
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["1v1", "2v2"], default="2v2")
    ap.add_argument("--env-version", choices=["v1", "v2"], default="v1")
    ap.add_argument("--paddle-overlap", type=float, default=None)
    ap.add_argument("--ball-speed-scale", type=float, default=None)
    ap.add_argument("--weights", required=True, help="checkpoint .pt holding all agents")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--points", type=int, default=5)
    ap.add_argument("--reward-mode", choices=["point_only", "shared_hit"], default="point_only")
    ap.add_argument("--hit-reward", type=float, default=0.05)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-npz", default=None)
    ap.add_argument(
        "--opponent",
        choices=["self", "random", "range", "heuristic", "reactive", "predictive"],
        default="self",
    )
    ap.add_argument("--opponent-team", choices=["A", "B"], default="B")
    args = ap.parse_args()
    import torch

    base_cfg = V2_CALIBRATED if args.env_version == "v2" else Config()
    speed_source = Config() if args.ball_speed_scale is not None else base_cfg
    speed_scale = args.ball_speed_scale if args.ball_speed_scale is not None else 1.0
    cfg = dataclasses.replace(
        base_cfg,
        mode=args.mode,
        points_to_win=args.points,
        env_version=args.env_version,
        collision_resolution="closest_paddle" if args.env_version == "v2" else "sequential_id",
        paddle_overlap=(
            args.paddle_overlap if args.paddle_overlap is not None else base_cfg.paddle_overlap
        ),
        serve_speed_min=speed_source.serve_speed_min * speed_scale,
        serve_speed_max=speed_source.serve_speed_max * speed_scale,
        ball_speed_max=speed_source.ball_speed_max * speed_scale,
        reward_mode=args.reward_mode,
        hit_reward=args.hit_reward,
    )
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    out = evaluate_weights(
        cfg, state, args.episodes, args.seed, args.out_npz, args.opponent, args.opponent_team
    )
    print(json.dumps(out, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
