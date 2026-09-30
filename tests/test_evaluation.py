"""Tests for deterministic evaluation: overlap metric, metrics, NPZ dump."""

import dataclasses

import numpy as np

from agents.multi_agent_ppo import IndependentPPO, PPOConfig
from environment.config import Config
from environment.pong_env import PongEnv
from evaluation.evaluate import evaluate_weights, hist_overlap


def test_hist_overlap_identical_distributions():
    y = np.linspace(-0.9, 0.9, 200)
    assert hist_overlap(y, y) == 1.0


def test_hist_overlap_disjoint_distributions():
    upper = np.linspace(0.1, 0.9, 100)
    lower = np.linspace(-0.9, -0.1, 100)
    assert hist_overlap(upper, lower) == 0.0


def test_hist_overlap_partial_and_bounds():
    y1 = np.linspace(-1.0, 1.0, 200)
    y2 = np.linspace(0.0, 1.0, 200)  # shares the positive half
    overlap = hist_overlap(y1, y2)
    assert 0.0 < overlap < 1.0
    assert hist_overlap(np.zeros(10), np.ones(10)) >= 0.0


def _tiny_state():
    env = PongEnv(seed=0)
    trainer = IndependentPPO(env.agent_ids, cfg=PPOConfig(seed=0))
    return {a: trainer.nets[a].state_dict() for a in env.agent_ids}


def test_evaluate_weights_metrics_and_npz(tmp_path):
    cfg = dataclasses.replace(Config(), mode="2v2", max_steps=150, points_to_win=1)
    state = _tiny_state()
    npz_path = str(tmp_path / "traj.npz")
    out = evaluate_weights(cfg, state, episodes=2, save_npz=npz_path)

    assert out["mode"] == "2v2"
    assert out["episodes"] == 2
    # Every episode ends in a win for one team or a draw; rates are fractions.
    rate_sum = out["win_rate_A"] + out["win_rate_B"] + out["draw_rate"]
    assert abs(rate_sum - 1.0) < 1e-9
    assert out["mean_return_A"] >= -1.0
    assert out["var_return_A"] >= 0.0
    assert out["mean_ep_len"] > 0
    # 2v2 specialization carries per-paddle stats plus team coverage overlap.
    for team in ("A", "B"):
        spec = out["specialization"][team]
        assert (
            spec["A1" if team == "A" else "B1"]["min_y"]
            <= spec["A1" if team == "A" else "B1"]["max_y"]
        )
        assert 0.0 <= spec["coverage_overlap"] <= 1.0
        assert spec["home_separation"] >= 0.0
    for a in cfg.agent_ids:
        behavior = out["behavior"][a]
        assert abs(sum(behavior["action_probs"]) - 1.0) < 1e-9
        assert behavior["contacts"] >= 0
        assert 0.0 <= behavior["team_contact_share"] <= 1.0
        assert sum(behavior["contact_regions"].values()) == behavior["contacts"]
        expected_region_share = 1.0 if behavior["contacts"] else 0.0
        assert abs(sum(behavior["contact_region_share"].values()) - expected_region_share) < 1e-9
    # Trajectory dump round-trips with one array per agent per episode.
    npz = np.load(npz_path)
    assert set(npz.files) == {f"{a}_{i}" for a in cfg.agent_ids for i in range(2)}
    assert all(npz[k].ndim == 1 for k in npz.files)


def test_evaluate_weights_substituted_random_opponent():
    cfg = dataclasses.replace(Config(), mode="2v2", max_steps=150, points_to_win=1)
    state = _tiny_state()
    out = evaluate_weights(cfg, state, episodes=2, opponent="random")
    # Team B paddles were replaced by scripted agents: no B specialization rows
    # beyond per-paddle stats is still fine, but the run itself must complete.
    assert out["episodes"] == 2
    assert out["win_rate_A"] + out["win_rate_B"] + out["draw_rate"] == 1.0


def test_evaluate_partial_team_checkpoint_against_side_swapped_baseline():
    cfg = dataclasses.replace(Config(), mode="2v2", max_steps=150, points_to_win=1)
    env = PongEnv(config=cfg, seed=0)
    learner = IndependentPPO(
        env.agent_ids, trainable_ids=["B1", "B2"], opponents={"A1": None, "A2": None}
    )
    state = {a: learner.nets[a].state_dict() for a in learner.ids}
    out = evaluate_weights(cfg, state, episodes=2, opponent="random", opponent_team="A")
    assert out["opponent_team"] == "A"
    assert out["episodes"] == 2


def test_evaluate_partial_team_checkpoint_against_frozen_checkpoint():
    cfg = dataclasses.replace(Config(), mode="2v2", max_steps=150, points_to_win=1)
    full = _tiny_state()
    learned_a = {a: full[a] for a in ("A1", "A2")}
    out = evaluate_weights(
        cfg,
        learned_a,
        episodes=2,
        opponent="checkpoint",
        opponent_team="B",
        opponent_state=full,
    )
    assert out["opponent"] == "checkpoint"
    assert out["episodes"] == 2


def test_evaluate_weights_deterministic_for_fixed_seed():
    cfg = dataclasses.replace(Config(), mode="2v2", max_steps=150, points_to_win=1)
    state = _tiny_state()
    a = evaluate_weights(cfg, state, episodes=2, base_seed=42)
    b = evaluate_weights(cfg, state, episodes=2, base_seed=42)
    assert a["win_rate_A"] == b["win_rate_A"]
    assert a["mean_ep_len"] == b["mean_ep_len"]
    assert a["specialization"] == b["specialization"]
