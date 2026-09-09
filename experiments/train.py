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

from agents.multi_agent_ppo import IndependentPPO, PPOConfig
from environment.config import Config
from environment.pong_env import PongEnv
from evaluation.evaluate import evaluate_weights


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["1v1", "2v2"], default="2v2")
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--points", type=int, default=5)
    ap.add_argument("--ckpt-every", type=int, default=20, help="iterations between checkpoints")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--init-weights", default=None, help="checkpoint .pt to warm-start from")
    return ap.parse_args(argv)


def main(argv=None) -> str:
    args = parse_args(argv)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    run = args.run_name or f"{args.mode}_seed{args.seed}_{stamp}"
    model_dir = os.path.join("results", "models", run)
    log_dir = os.path.join("results", "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cfg = dataclasses.replace(Config(), mode=args.mode, points_to_win=args.points)
    env = PongEnv(config=cfg, seed=args.seed)
    ppo = PPOConfig(
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.lam,
        clip_eps=args.clip,
        epochs=args.epochs,
        minibatch=args.minibatch,
        rollout_steps=args.rollout,
        ent_coef=args.ent,
        seed=args.seed,
    )
    trainer = IndependentPPO(env.agent_ids, cfg=ppo)
    if args.init_weights:
        import torch

        state = torch.load(args.init_weights, map_location="cpu", weights_only=True)
        for a in trainer.ids:
            trainer.nets[a].load_state_dict(state[a])
        print(f"warm-started from {args.init_weights}")
    trainer.reset(env)

    iters = math.ceil(args.timesteps / args.rollout)
    csv_path = os.path.join(log_dir, f"{run}.csv")
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
            ]
        )
        for it in range(1, iters + 1):
            buf, advs, rets, st = trainer.rollout(env)
            losses = trainer.update(buf, advs, rets)
            steps = min(it * args.rollout, args.timesteps)
            w.writerow(
                [
                    it,
                    steps,
                    st["episodes"],
                    f"{st['mean_return_A']:.3f}",
                    f"{st['mean_len']:.1f}",
                    f"{st['mean_A']:.2f}",
                    f"{st['mean_B']:.2f}",
                    f"{losses['pg']:.4f}",
                    f"{losses['v']:.4f}",
                    f"{losses['ent']:.4f}",
                ]
            )
            f.flush()
            print(
                f"[{run}] it={it}/{iters} steps={steps} "
                f"eps={st['episodes']} retA={st['mean_return_A']:.2f} "
                f"A:B={st['mean_A']:.1f}:{st['mean_B']:.1f} "
                f"pg={losses['pg']:.3f} v={losses['v']:.3f} ent={losses['ent']:.3f}",
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
            {
                k: out[k]
                for k in (
                    "win_rate_A",
                    "win_rate_B",
                    "mean_return_A",
                    "mean_ep_len",
                    "specialization",
                )
            },
            indent=2,
        )
    )
    return run


if __name__ == "__main__":
    main()
