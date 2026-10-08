"""Smoke test for the independent-PPO loop (fast, CPU)."""

import os

import numpy as np
import torch

from agents.multi_agent_ppo import IndependentPPO, PPOConfig, compute_gae
from agents.policies import ActorCritic
from environment.config import Config
from environment.pong_env import PongEnv


def test_ppo_rollout_and_update_runs():
    import dataclasses

    cfg = dataclasses.replace(Config(), max_steps=200)
    env = PongEnv(config=cfg, seed=0)
    ppo = PPOConfig(rollout_steps=128, epochs=1, minibatch=64, seed=0)
    trainer = IndependentPPO(env.agent_ids, cfg=ppo)
    trainer.reset(env)
    buf, advs, rets, _stats = trainer.rollout(env)
    assert trainer.global_steps == 128
    for a in env.agent_ids:
        assert buf[a]["obs"].shape == (128, 8)
        assert np.all(np.isfinite(advs[a]))
    losses = trainer.update(buf, advs, rets)
    for v in losses.values():
        assert np.isfinite(v)


def test_ppo_save_load_roundtrip(tmp_path):
    env = PongEnv(seed=0)
    trainer = IndependentPPO(env.agent_ids, cfg=PPOConfig(seed=0))
    path = os.path.join(str(tmp_path), "ckpt.pt")
    trainer.save(path)
    before = next(iter(trainer.nets["A1"].parameters())).detach().clone()
    trainer2 = IndependentPPO(env.agent_ids, cfg=PPOConfig(seed=1))
    trainer2.load(path)
    after = next(iter(trainer2.nets["A1"].parameters()))
    assert torch_allclose(before, after)


def test_ppo_exact_partial_rollout():
    import dataclasses

    env = PongEnv(config=dataclasses.replace(Config(), max_steps=100), seed=0)
    trainer = IndependentPPO(
        env.agent_ids, cfg=PPOConfig(rollout_steps=64, epochs=1, minibatch=16, seed=0)
    )
    trainer.reset(env)
    buf, advs, rets, stats = trainer.rollout(env, steps=17)
    assert trainer.global_steps == 17
    assert set(buf) == set(env.agent_ids)
    assert all(buf[a]["obs"].shape == (17, 8) for a in buf)
    assert stats["reward_event_rate"] >= 0.0
    losses = trainer.update(buf, advs, rets)
    assert np.isfinite(losses["kl"])


def test_numpy_sampler_matches_torch_forward():
    net = ActorCritic(agent_id="A1")
    sampler = net.numpy_sampler()
    feat = np.linspace(-1.0, 1.0, 8).astype(np.float32)
    action, logp, value = sampler.sample(feat, np.random.default_rng(0))
    with torch.no_grad():
        logits, v = net(torch.as_tensor(feat).unsqueeze(0))
    expected_logp = torch.log_softmax(logits, dim=-1)[0, action]
    assert abs(logp - float(expected_logp)) < 1e-5
    assert abs(value - float(v)) < 1e-5


def test_compute_gae_known_undiscounted_trajectory():
    advantages, returns = compute_gae(
        rewards=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        values=np.zeros(3, dtype=np.float32),
        next_values=np.zeros(3, dtype=np.float32),
        terminated=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert np.allclose(advantages, [3.0, 2.0, 1.0])
    assert np.allclose(returns, advantages)


def test_compute_gae_bootstraps_time_limit_truncation():
    advantages, _ = compute_gae(
        rewards=np.asarray([0.0], dtype=np.float32),
        values=np.asarray([0.0], dtype=np.float32),
        next_values=np.asarray([2.0], dtype=np.float32),
        terminated=np.asarray([0.0], dtype=np.float32),
        gamma=0.5,
        gae_lambda=1.0,
    )
    assert np.allclose(advantages, [1.0])


class _BanditEnv:
    agent_ids = ("A1",)

    def reset(self, seed=None):
        return {"A1": np.zeros(8, dtype=np.float32)}, {}

    def step(self, actions):
        reward = [0.0, 1.0, -1.0][actions["A1"]]
        obs = {"A1": np.zeros(8, dtype=np.float32)}
        return (
            obs,
            {"A1": reward},
            {"A1": True},
            {"A1": False},
            {"scores": {"A": 0, "B": 0}, "steps": 1},
        )


def test_ppo_learns_deterministic_bandit():
    env = _BanditEnv()
    trainer = IndependentPPO(
        env.agent_ids,
        cfg=PPOConfig(lr=0.01, rollout_steps=64, epochs=4, minibatch=32, ent_coef=0.0, seed=0),
    )
    trainer.reset(env)
    for _ in range(15):
        buf, advs, rets, _ = trainer.rollout(env)
        trainer.update(buf, advs, rets)
    assert trainer.nets["A1"].greedy(np.zeros(8, dtype=np.float32)) == 1


def torch_allclose(a, b) -> bool:
    import torch

    return bool(torch.allclose(a, b))
