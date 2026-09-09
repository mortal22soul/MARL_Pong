"""Aggregate training logs and trajectory dumps into report figures.

Three views, all aimed at coordination evidence rather than win rate alone:
learning curves (mean +/- seed spread) from the per-iteration CSV logs,
paddle-coverage heatmaps from evaluation trajectory dumps, and teammate
coverage overlap measured at successive training checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt


def load_run_csv(path: str) -> dict[str, np.ndarray]:
    """Load a training log CSV into float arrays keyed by column name."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cols = rows[0].keys()
    out: dict[str, np.ndarray] = {}
    for c in cols:
        vals = [float(r[c]) for r in rows]
        out[c] = np.asarray(vals, dtype=np.float64)
    return out


def learning_curves(csv_paths: list[str], column: str = "mean_return_A") -> dict:
    """Align runs on their common 'iter' values and aggregate a column.

    Returns {"steps", "mean", "min", "max"} where steps are total env steps
    (iter * rollout) shared by every run; runs may end at different iters.
    """
    if not csv_paths:
        raise ValueError("need at least one CSV")
    common: np.ndarray | None = None
    per_run: list[dict[float, float]] = []
    for p in csv_paths:
        data = load_run_csv(p)
        table = dict(zip(data["iter"].tolist(), data[column].tolist()))
        iters = np.asarray(sorted(table), dtype=np.float64)
        common = iters if common is None else np.intersect1d(common, iters)
        per_run.append(table)
    common = np.asarray(common)
    if len(common) == 0:
        raise ValueError("runs share no common iterations")
    stacked = np.asarray([np.asarray([t[i] for i in common]) for t in per_run])
    rollout = _rollout_from_csv(csv_paths[0])
    return {
        "steps": common * float(rollout),
        "mean": stacked.mean(axis=0),
        "min": stacked.min(axis=0),
        "max": stacked.max(axis=0),
    }


def _rollout_from_csv(csv_path: str) -> float:
    """Recover the rollout size from the CSV 'steps' column (steps / iter)."""
    data = load_run_csv(csv_path)
    iters, steps = data["iter"], data["steps"]
    if len(iters) < 2:
        return 1.0
    return float(np.median(np.diff(steps) / np.diff(iters)))


def plot_learning_curves(
    curves_by_label: dict[str, dict], out_path: str, ylabel: str = "team A return"
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, c in curves_by_label.items():
        steps = c["steps"] / 1000.0
        ax.plot(steps, c["mean"], label=label)
        ax.fill_between(steps, c["min"], c["max"], alpha=0.2)
    ax.set_xlabel("env steps (k)")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title("Training curves (mean ± seed spread)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_y_heatmaps(npz_path: str, out_path: str, y_bins: int = 30, t_bins: int = 30) -> None:
    """Paddle coverage heatmaps: episode-time fraction vs paddle y, per agent.

    Paddles only move vertically, so the informative heatmap is each paddle's
    y occupancy over the (normalized) episode timeline; teammates are stacked
    in team rows for direct comparison.
    """
    data = np.load(npz_path)
    agents = sorted({f.rsplit("_", 1)[0] for f in data.files})
    episodes: dict[str, list[np.ndarray]] = {a: [] for a in agents}
    for f in data.files:
        episodes[f.rsplit("_", 1)[0]].append(data[f])
    fig, axes = plt.subplots(
        1, len(agents), figsize=(2.6 * len(agents), 3.4), sharey=True, squeeze=False
    )
    y_edges = np.linspace(-1, 1, y_bins + 1)
    t_edges = np.linspace(0, 1, t_bins + 1)
    for ax, a in zip(axes[0], agents):
        heat = np.zeros((t_bins, y_bins))
        for ep in episodes[a]:
            ep = np.asarray(ep, dtype=np.float64)
            if len(ep) < 2:
                continue
            t = np.linspace(0, 1, len(ep))
            heat += np.histogram2d(t, ep, bins=[t_edges, y_edges])[0]
        ax.imshow(heat.T, aspect="auto", origin="upper", extent=[0, 1, 1, -1])
        ax.set_title(a)
        ax.set_xlabel("episode time (norm)")
    axes[0][0].set_ylabel("paddle y  (+y down)")
    fig.suptitle("Paddle coverage")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def overlap_at_checkpoints(
    cfg,
    ckpt_paths: list[str],
    episodes: int = 4,
    base_seed: int = 1000,
) -> list[float]:
    """Team A teammate coverage overlap at successive checkpoints.

    Greedy short episodes per checkpoint (see evaluation.evaluate); a falling
    overlap over training is the strongest positional-specialization evidence.
    """
    import torch

    overlaps = []
    for path in ckpt_paths:
        state = torch.load(path, map_location="cpu", weights_only=True)
        overlaps.append(_short_eval(cfg, state, episodes, base_seed))
    return overlaps


def _short_eval(cfg, state, episodes: int, base_seed: int) -> float:
    from agents.multi_agent_ppo import IndependentPPO, PPOConfig
    from environment.pong_env import PongEnv

    env = PongEnv(config=cfg, seed=base_seed)
    trainer = IndependentPPO(env.agent_ids, cfg=PPOConfig())
    for a in trainer.ids:
        trainer.nets[a].load_state_dict(state[a])
        trainer.nets[a].eval()
    ys = {a: [] for a in ("A1", "A2")}
    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        while not done:
            actions = {a: trainer.nets[a].greedy(o) for a, o in obs.items()}
            for a, y in ys.items():
                y.append(env.paddles[a][0])
            obs, _, term, trunc, _ = env.step(actions)
            done = all(term.values()) or all(trunc.values())
    from evaluation.evaluate import hist_overlap

    return hist_overlap(np.asarray(ys["A1"]), np.asarray(ys["A2"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", nargs="+", help="label=csv pairs for learning curves")
    ap.add_argument("--column", default="mean_return_A")
    ap.add_argument("--traj", help="trajectory .npz for coverage heatmaps")
    ap.add_argument("--out-dir", default="results/plots")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.curves:
        by_label = {}
        for pair in args.curves:
            label, path = pair.split("=", 1)
            by_label[label] = learning_curves([path], args.column)
        plot_learning_curves(by_label, os.path.join(args.out_dir, f"curves_{args.column}.png"))
    if args.traj:
        plot_y_heatmaps(args.traj, os.path.join(args.out_dir, "coverage_heatmaps.png"))


if __name__ == "__main__":
    main()
