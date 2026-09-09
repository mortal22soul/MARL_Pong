"""Tests for analysis: CSV aggregation, curve/heatmap rendering, ckpt sweep."""

import csv
import dataclasses
import os

import numpy as np
import torch

from agents.multi_agent_ppo import IndependentPPO, PPOConfig
from analysis.plots import learning_curves, load_run_csv, overlap_at_checkpoints
from environment.config import Config
from environment.pong_env import PongEnv


def _write_csv(path, iters, rets):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "steps", "episodes", "mean_return_A", "mean_len"])
        for i, r in zip(iters, rets):
            w.writerow([i, i * 1024, max(i - 1, 0), r, 500.0])


def test_load_run_csv_types(tmp_path):
    p = str(tmp_path / "run.csv")
    _write_csv(p, [1, 2, 3], [np.nan, 0.5, 1.5])
    data = load_run_csv(p)
    assert data["iter"].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(data["mean_return_A"][0])
    assert data["mean_len"].dtype == np.float64


def test_learning_curves_aligns_on_common_iters(tmp_path):
    p1, p2 = str(tmp_path / "a.csv"), str(tmp_path / "b.csv")
    _write_csv(p1, [1, 2, 3, 4], [0.0, 1.0, 2.0, 3.0])
    _write_csv(p2, [2, 3, 4, 5], [2.0, 4.0, 6.0, 8.0])
    c = learning_curves([p1, p2])
    assert c["steps"].tolist() == [2048.0, 3072.0, 4096.0]
    assert c["mean"].tolist() == [1.5, 3.0, 4.5]
    assert c["min"].tolist() == [1.0, 2.0, 3.0]
    assert c["max"].tolist() == [2.0, 4.0, 6.0]


def test_learning_curves_empty_and_disjoint(tmp_path):
    p1, p2 = str(tmp_path / "a.csv"), str(tmp_path / "b.csv")
    _write_csv(p1, [1, 2], [0.0, 1.0])
    _write_csv(p2, [3, 4], [0.0, 1.0])
    try:
        learning_curves([p1, p2])
        raised = False
    except ValueError:
        raised = True
    assert raised
    try:
        learning_curves([])
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_plot_learning_curves_renders_file(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from analysis.plots import plot_learning_curves

    p = str(tmp_path / "run.csv")
    _write_csv(p, [1, 2, 3], [0.0, 1.0, 2.0])
    out = str(tmp_path / "curves.png")
    plot_learning_curves({"seed0": learning_curves([p])}, out)
    assert os.path.getsize(out) > 0


def test_plot_y_heatmaps_renders_file(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from analysis.plots import plot_y_heatmaps

    rng = np.random.default_rng(0)
    npz = str(tmp_path / "traj.npz")
    arrays = {}
    for a in ("A1", "A2"):
        for ep in range(2):
            arrays[f"{a}_{ep}"] = rng.uniform(-1, 1, size=100).astype(np.float32)
    np.savez_compressed(npz, **arrays)
    out = str(tmp_path / "heat.png")
    plot_y_heatmaps(npz, out)
    assert os.path.getsize(out) > 0


def test_overlap_at_checkpoints_with_fresh_weights(tmp_path):
    cfg = dataclasses.replace(Config(), mode="2v2", max_steps=120, points_to_win=1)
    env = PongEnv(config=cfg, seed=0)
    trainer = IndependentPPO(env.agent_ids, cfg=PPOConfig(seed=0))
    state = {a: trainer.nets[a].state_dict() for a in env.agent_ids}
    path = str(tmp_path / "ckpt.pt")
    torch.save(state, path)
    overlaps = overlap_at_checkpoints(cfg, [path], episodes=1)
    assert len(overlaps) == 1
    assert 0.0 <= overlaps[0] <= 1.0
