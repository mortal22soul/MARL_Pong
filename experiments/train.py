"""Independent-PPO training entry point for marl-pong (1v1 and 2v2).

Four (or two) agents learn concurrently from a shared team reward with
synchronized environment stepping. Logs per-iteration stats to CSV,
checkpoints to results/models/<run>/, and runs a greedy evaluation at
the end (results/eval_<run>.json + trajectories .npz).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.frozen_policy import FrozenPolicy
from agents.multi_agent_ppo import IndependentPPO, PPOConfig
from baselines.agents import HeuristicAgent, PredictiveAgent, RandomAgent
from environment.config import V2_CALIBRATED, Config
from environment.pong_env import PongEnv
from evaluation.evaluate import evaluate_weights


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["1v1", "2v2"], default="2v2")
    ap.add_argument("--env-version", choices=["v1", "v2"], default="v1")
    ap.add_argument("--paddle-overlap", type=float, default=None)
    ap.add_argument("--ball-speed-scale", type=float, default=None)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.98)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument(
        "--ent-final",
        type=float,
        default=0.0005,
        help="linearly annealed final entropy coefficient",
    )
    ap.add_argument("--points", type=int, default=5)
    ap.add_argument(
        "--reward-mode", choices=["point_only", "shared_hit", "shaped"], default="shaped"
    )
    ap.add_argument("--hit-reward", type=float, default=0.2)
    ap.add_argument("--team-hit-reward", type=float, default=0.1)
    ap.add_argument("--ckpt-every", type=int, default=20, help="iterations between checkpoints")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--run-name", default=None)
    ap.add_argument(
        "--warm-start-weights",
        default=None,
        help="load model weights only; does not restore optimizer state",
    )
    ap.add_argument("--train-team", choices=["all", "A", "B"], default="all")
    ap.add_argument(
        "--opponent",
        choices=["self", "random", "reactive", "predictive", "checkpoint"],
        default="self",
        help="fixed policy for the non-trainable team",
    )
    ap.add_argument(
        "--opponent-weights",
        default=None,
        help="full-team checkpoint used with --opponent checkpoint",
    )
    return ap.parse_args(argv)


def make_fixed_agents(
    env: PongEnv, kind: str, train_team: str, seed: int, checkpoint: str | None = None
) -> dict[str, object]:
    if train_team == "all" or kind == "self":
        return {}
    opponent_team = "B" if train_team == "A" else "A"
    if kind == "checkpoint":
        if checkpoint is None:
            raise ValueError("--opponent checkpoint requires --opponent-weights")
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        needed = [a for a in env.agent_ids if a.startswith(opponent_team)]
        if not set(needed) <= set(state):
            raise ValueError(f"checkpoint does not contain frozen opponent agents {needed}")
        return {a: FrozenPolicy(state[a]) for a in needed}
    out = {}
    for i, a in enumerate(a for a in env.agent_ids if a.startswith(opponent_team)):
        lo, hi = env._allowed_range(a)
        home = (lo + hi) / 2.0
        if kind == "random":
            out[a] = RandomAgent(seed=seed + i)
        elif kind == "reactive":
            out[a] = HeuristicAgent(team=opponent_team, home=home)
        else:
            paddle_x = -env.cfg.paddle_x_offset if opponent_team == "A" else env.cfg.paddle_x_offset
            out[a] = PredictiveAgent(opponent_team, paddle_x, env.cfg.ball_radius, home=home)
    return out


def main(argv=None) -> str:
    args = parse_args(argv)
    if args.train_team == "all" and args.opponent != "self":
        raise ValueError("--train-team all requires --opponent self")
    if args.train_team != "all" and args.opponent == "self":
        raise ValueError("fixed-team training requires a fixed scripted or checkpoint opponent")
    if args.opponent == "checkpoint" and not args.opponent_weights:
        raise ValueError("--opponent checkpoint requires --opponent-weights")
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    run = args.run_name or f"{args.mode}_seed{args.seed}_{stamp}"
    model_dir = os.path.join("results", "models", run)
    log_dir = os.path.join("results", "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    base_cfg = V2_CALIBRATED if args.env_version == "v2" else Config()
    collision_resolution = "closest_paddle" if args.env_version == "v2" else "sequential_id"
    speed_source = Config() if args.ball_speed_scale is not None else base_cfg
    speed_scale = args.ball_speed_scale if args.ball_speed_scale is not None else 1.0
    cfg = dataclasses.replace(
        base_cfg,
        mode=args.mode,
        points_to_win=args.points,
        env_version=args.env_version,
        collision_resolution=collision_resolution,
        paddle_overlap=(
            args.paddle_overlap if args.paddle_overlap is not None else base_cfg.paddle_overlap
        ),
        serve_speed_min=speed_source.serve_speed_min * speed_scale,
        serve_speed_max=speed_source.serve_speed_max * speed_scale,
        ball_speed_max=speed_source.ball_speed_max * speed_scale,
        reward_mode=args.reward_mode,
        hit_reward=args.hit_reward,
    )
    env = PongEnv(config=cfg, seed=args.seed)
    trainable_ids = (
        env.agent_ids
        if args.train_team == "all"
        else [a for a in env.agent_ids if a.startswith(args.train_team)]
    )
    fixed_agents = make_fixed_agents(
        env, args.opponent, args.train_team, args.seed + 10_000, args.opponent_weights
    )
    iters = math.ceil(args.timesteps / args.rollout)
    ppo = PPOConfig(
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.lam,
        clip_eps=args.clip,
        epochs=args.epochs,
        minibatch=args.minibatch,
        rollout_steps=args.rollout,
        ent_coef=args.ent,
        ent_coef_final=args.ent_final,
        total_updates=iters,
        seed=args.seed,
    )
    trainer = IndependentPPO(
        env.agent_ids, cfg=ppo, trainable_ids=trainable_ids, opponents=fixed_agents
    )
    if args.warm_start_weights:
        import torch

        from agents.policies import ActorCritic

        state = torch.load(args.warm_start_weights, map_location="cpu", weights_only=True)
        for a in trainer.ids:
            sd = ActorCritic._load_state_dict_compat(state[a])
            trainer.nets[a].load_state_dict(sd)
        print(
            f"warm-started model weights from {args.warm_start_weights}; optimizer state is fresh"
        )
    trainer.reset(env)

    csv_path = os.path.join(log_dir, f"{run}.csv")
    agent_columns = [
        f"{a}_{metric}"
        for a in trainer.ids
        for metric in (
            "action_stay",
            "action_up",
            "action_down",
            "entropy",
            "value_loss",
            "approx_kl",
        )
    ]
    with open(os.path.join(model_dir, "run_config.json"), "w") as f:
        json.dump(
            {
                "run": run,
                "requested_timesteps": args.timesteps,
                "config": dataclasses.asdict(cfg),
                "ppo": dataclasses.asdict(ppo),
                "trainable_ids": trainable_ids,
                "fixed_opponent": args.opponent if fixed_agents else None,
                "fixed_opponent_weights": args.opponent_weights if fixed_agents else None,
            },
            f,
            indent=2,
        )
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "iter",
                "steps",
                "episodes",
                "mean_return_A",
                "mean_len",
                "mean_A",
                "mean_B",
                "loss_pg",
                "loss_v",
                "entropy",
                "ent_coef",
                "approx_kl",
                "clip_frac",
                "explained_variance",
                "reward_event_rate",
                "truncated_episodes",
            ]
            + agent_columns
        )
        actual_steps = 0
        for it in range(1, iters + 1):
            collected = min(args.rollout, args.timesteps - actual_steps)
            buf, advs, rets, st = trainer.rollout(env, steps=collected)
            losses = trainer.update(buf, advs, rets)
            actual_steps += collected
            row = [
                it,
                actual_steps,
                st["episodes"],
                f"{st['mean_return_A']:.3f}",
                f"{st['mean_len']:.1f}",
                f"{st['mean_A']:.2f}",
                f"{st['mean_B']:.2f}",
                f"{losses['pg']:.4f}",
                f"{losses['v']:.4f}",
                f"{losses['ent']:.4f}",
                f"{losses['ent_coef']:.5f}",
                f"{losses['kl']:.5f}",
                f"{losses['clip_frac']:.4f}",
                f"{losses['ev']:.4f}",
                f"{st['reward_event_rate']:.5f}",
                st["truncated_episodes"],
            ]
            for a in trainer.ids:
                agent = st["agents"][a]
                update = trainer.last_agent_metrics[a]
                row.extend(
                    [
                        f"{agent['action_stay']:.4f}",
                        f"{agent['action_up']:.4f}",
                        f"{agent['action_down']:.4f}",
                        f"{update['entropy']:.4f}",
                        f"{update['value_loss']:.4f}",
                        f"{update['approx_kl']:.5f}",
                    ]
                )
            w.writerow(row)
            f.flush()
            print(
                f"[{run}] it={it}/{iters} steps={actual_steps} "
                f"eps={st['episodes']} retA={st['mean_return_A']:.2f} "
                f"A:B={st['mean_A']:.1f}:{st['mean_B']:.1f} "
                f"pg={losses['pg']:.3f} v={losses['v']:.3f} ent={losses['ent']:.3f} "
                f"kl={losses['kl']:.4f}",
                flush=True,
            )
            if it % args.ckpt_every == 0 or it == iters:
                trainer.save(os.path.join(model_dir, f"iter_{it}.pt"))
    final = os.path.join(model_dir, "final.pt")
    trainer.save(final)
    print(f"saved {final}; log {csv_path}")

    if args.train_team == "all":
        import torch

        state = torch.load(final, map_location="cpu", weights_only=True)
        out = evaluate_weights(
            cfg,
            state,
            args.eval_episodes,
            base_seed=10_000 + args.seed,
            save_npz=os.path.join("results", f"traj_{run}.npz"),
        )
        out["run"] = run
        eval_path = os.path.join("results", f"eval_{run}.json")
        with open(eval_path, "w") as f:
            json.dump(out, f, indent=2)
        print(
            json.dumps(
                {k: out[k] for k in ("win_rate_A", "win_rate_B", "mean_return_A", "mean_ep_len")},
                indent=2,
            )
        )
    else:
        print(
            "fixed-opponent run saved learner-only weights; evaluate with evaluation/evaluate.py --team-state"
        )
    return run


if __name__ == "__main__":
    main()
