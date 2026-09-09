"""Smoke test for the independent-PPO loop (fast, CPU)."""

import os

import numpy as np

from agents.multi_agent_ppo import IndependentPPO, PPOConfig
from environment.config import Config
from environment.pong_env import PongEnv


def test_ppo_rollout_and_update_runs():
    import dataclasses

    cfg = dataclasses.replace(Config(), mode="2v2", max_steps=200)
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


def torch_allclose(a, b) -> bool:
    import torch

    return bool(torch.allclose(a, b))
