"""Independent-PPO training entry point for marl-pong (2v2).

Four agents learn concurrently from a shared team reward with
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

from agents.multi_agent_ppo import IndependentPPO, PPOConfig
from environment.config import Config
from environment.pong_env import PongEnv
from evaluation.evaluate import evaluate_weights


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout", type=int, default=PPOConfig.rollout_steps)
    ap.add_argument("--lr", type=float, default=PPOConfig.lr)
    ap.add_argument("--epochs", type=int, default=PPOConfig.epochs)
    ap.add_argument("--minibatch", type=int, default=PPOConfig.minibatch)
    ap.add_argument("--gamma", type=float, default=PPOConfig.gamma)
    ap.add_argument("--lam", type=float, default=PPOConfig.gae_lambda)
    ap.add_argument("--clip", type=float, default=PPOConfig.clip_eps)
    ap.add_argument("--ent", type=float, default=PPOConfig.ent_coef)
    ap.add_argument(
        "--ent-final",
        type=float,
        default=PPOConfig.ent_coef_final,
        help="linearly annealed final entropy coefficient",
    )
    ap.add_argument("--points", type=int, default=5)
    ap.add_argument("--ckpt-every", type=int, default=20, help="iterations between checkpoints")
    ap.add_argument("--eval-episodes", type=int, default=60)
    ap.add_argument("--run-name", default=None)
    ap.add_argument(
        "--warm-start-weights",
        default=None,
        help="load model weights only; does not restore optimizer state",
    )
    return ap.parse_args(argv)


def main(argv=None) -> str:
    args = parse_args(argv)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    run = args.run_name or f"2v2_seed{args.seed}_{stamp}"
    model_dir = os.path.join("results", "models", run)
    log_dir = os.path.join("results", "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cfg = dataclasses.replace(Config(), points_to_win=args.points)
    env = PongEnv(config=cfg, seed=args.seed)
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
    trainer = IndependentPPO(env.agent_ids, cfg=ppo)
    if args.warm_start_weights:
        import torch

        state = torch.load(args.warm_start_weights, map_location="cpu", weights_only=True)
        for a in trainer.ids:
            trainer.nets[a].load_checkpoint(state[a])
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
    return run


if __name__ == "__main__":
    main()
