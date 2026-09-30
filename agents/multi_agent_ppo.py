"""Independent PPO for the marl-pong dict env.

Each agent gets its own actor-critic, rollout buffer, and PPO updates —
no shared critic, no parameter sharing, no cross-agent gradient flow.
Environment stepping is synchronized: one shared env.step per timestep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from .policies import ActorCritic


@dataclass
class PPOConfig:
    lr: float = 4e-4
    gamma: float = 0.995
    gae_lambda: float = 0.98
    clip_eps: float = 0.2
    epochs: int = 4
    minibatch: int = 256
    rollout_steps: int = 1024
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    ent_coef_final: float | None = 0.0005
    total_updates: int | None = None
    max_grad_norm: float = 0.5
    seed: int = 0


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return GAE advantages and value targets for one known rollout.

    Time-limit truncations intentionally remain bootstrap-able: callers store
    their actual final-observation value in ``next_values`` and set only true
    scored terminal transitions in ``terminated``.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        nonterminal = 1.0 - terminated[t]
        delta = rewards[t] + gamma * nonterminal * next_values[t] - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[t] = gae
    return advantages, advantages + values


class IndependentPPO:
    def __init__(
        self,
        agent_ids: list[str],
        obs_dim: int = 8,
        n_actions: int = 3,
        cfg: PPOConfig | None = None,
        trainable_ids: list[str] | None = None,
        opponents: dict[str, object] | None = None,
    ):
        self.cfg = cfg or PPOConfig()
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.agent_ids = list(agent_ids)
        self.ids = list(trainable_ids) if trainable_ids is not None else list(agent_ids)
        unknown = set(self.ids) - set(self.agent_ids)
        if unknown:
            raise ValueError(f"unknown trainable agents: {sorted(unknown)}")
        self.opponents = opponents or {}
        missing = set(self.agent_ids) - set(self.ids) - set(self.opponents)
        if missing:
            raise ValueError(f"missing fixed policies for non-trainable agents: {sorted(missing)}")
        self.nets = {a: ActorCritic(obs_dim, n_actions, agent_id=a) for a in self.ids}
        self.opts = {
            a: torch.optim.Adam(n.parameters(), lr=self.cfg.lr) for a, n in self.nets.items()
        }
        self.cur_obs: dict[str, np.ndarray] | None = None
        self.global_steps = 0
        self.updates = 0
        self.last_agent_metrics: dict[str, dict[str, float]] = {}

    # --- data collection ---
    def reset(self, env) -> None:
        obs, _ = env.reset(seed=self.cfg.seed)
        self.cur_obs = obs

    def rollout(self, env, steps: int | None = None):
        """Collect rollout_steps synchronized transitions. Returns (buf, advs, rets, stats)."""
        cfg, ids = self.cfg, self.ids
        T = steps if steps is not None else cfg.rollout_steps
        assert self.cur_obs is not None, "call reset(env) first"
        buf = {
            a: {
                "obs": np.zeros((T, 8), np.float32),
                "act": np.zeros(T, np.int64),
                "logp": np.zeros(T, np.float32),
                "val": np.zeros(T, np.float32),
                "rew": np.zeros(T, np.float32),
                "terminated": np.zeros(T, np.float32),
                "truncated": np.zeros(T, np.float32),
                "next_val": np.zeros(T, np.float32),
            }
            for a in ids
        }
        ep_returns, ep_lens, ep_a, ep_b = [], [], [], []
        truncated_episodes = 0
        for t in range(T):
            acts: dict[str, int] = {}
            for a in self.agent_ids:
                if a in self.nets:
                    act, lp, v = self.nets[a].sample(self.cur_obs[a])
                    acts[a] = act
                    buf[a]["obs"][t] = self.nets[a]._prep_obs(self.cur_obs[a])
                    buf[a]["act"][t] = act
                    buf[a]["logp"][t] = lp
                    buf[a]["val"][t] = v
                else:
                    acts[a] = self.opponents[a].act(self.cur_obs[a])
            nobs, rews, terms, truncs, info = env.step(acts)
            terminated = all(terms.values())
            done = terminated or all(truncs.values())
            for a in ids:
                buf[a]["rew"][t] = rews[a]
                buf[a]["terminated"][t] = 1.0 if terminated else 0.0
                buf[a]["truncated"][t] = 1.0 if all(truncs.values()) else 0.0
                if terminated:
                    buf[a]["next_val"][t] = 0.0
                else:
                    with torch.no_grad():
                        nfeat = self.nets[a]._prep_obs(nobs[a])
                        _, value = self.nets[a](
                            torch.as_tensor(nfeat, dtype=torch.float32).unsqueeze(0)
                        )
                        buf[a]["next_val"][t] = float(value.item())
            self.global_steps += 1
            if done:
                ep_returns.append(info["scores"]["A"] - info["scores"]["B"])
                ep_lens.append(info["steps"])
                ep_a.append(info["scores"]["A"])
                ep_b.append(info["scores"]["B"])
                truncated_episodes += int(all(truncs.values()))
                self.cur_obs, _ = env.reset()
            else:
                self.cur_obs = nobs
        # GAE uses each transition's actual next observation. Time-limit
        # truncations bootstrap; scored-terminal transitions do not.
        advs, rets = {}, {}
        for a in ids:
            advs[a], rets[a] = compute_gae(
                buf[a]["rew"],
                buf[a]["val"],
                buf[a]["next_val"],
                buf[a]["terminated"],
                cfg.gamma,
                cfg.gae_lambda,
            )
        stats = {
            "episodes": len(ep_returns),
            "mean_return_A": float(np.mean(ep_returns)) if ep_returns else float("nan"),
            "mean_len": float(np.mean(ep_lens)) if ep_lens else float("nan"),
            "mean_A": float(np.mean(ep_a)) if ep_a else float("nan"),
            "mean_B": float(np.mean(ep_b)) if ep_b else float("nan"),
            "truncated_episodes": truncated_episodes,
            "reward_event_rate": float(np.mean([buf[a]["rew"] != 0.0 for a in ids])),
            "agents": {
                a: {
                    "action_stay": float(np.mean(buf[a]["act"] == 0)),
                    "action_up": float(np.mean(buf[a]["act"] == 1)),
                    "action_down": float(np.mean(buf[a]["act"] == 2)),
                }
                for a in ids
            },
        }
        return buf, advs, rets, stats

    # --- optimization ---
    def update(self, buf, advs, rets) -> dict[str, float]:
        cfg = self.cfg
        ent_coef = self._entropy_coef()
        agg = {"pg": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "clip_frac": 0.0, "ev": 0.0}
        agent_metrics = {}
        for a in self.ids:
            obs = torch.as_tensor(buf[a]["obs"])
            act = torch.as_tensor(buf[a]["act"])
            old_logp = torch.as_tensor(buf[a]["logp"])
            old_v = torch.as_tensor(buf[a]["val"])
            adv = torch.as_tensor(advs[a], dtype=torch.float32)
            ret = torch.as_tensor(rets[a], dtype=torch.float32)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            n = obs.shape[0]
            for _ in range(cfg.epochs):
                perm = torch.randperm(n)
                for start in range(0, n, cfg.minibatch):
                    sl = perm[start : start + cfg.minibatch]
                    logits, v = self.nets[a](obs[sl])
                    dist = Categorical(logits=logits)
                    logp = dist.log_prob(act[sl])
                    ratio = torch.exp(logp - old_logp[sl])
                    pg = torch.min(
                        ratio * adv[sl],
                        torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv[sl],
                    )
                    v_clipped = old_v[sl] + torch.clamp(v - old_v[sl], -cfg.clip_eps, cfg.clip_eps)
                    vl = torch.max((v - ret[sl]) ** 2, (v_clipped - ret[sl]) ** 2)
                    loss = -pg.mean() + cfg.vf_coef * vl.mean() - ent_coef * dist.entropy().mean()
                    self.opts[a].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.nets[a].parameters(), cfg.max_grad_norm)
                    self.opts[a].step()
            with torch.no_grad():
                logits, v = self.nets[a](obs)
                dist = Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(act) - old_logp)
                pg = torch.min(
                    ratio * adv, torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
                )
                agg["pg"] += float(-pg.mean())
                agg["v"] += float(((v - ret) ** 2).mean())
                agg["ent"] += float(dist.entropy().mean())
                log_ratio = dist.log_prob(act) - old_logp
                agg["kl"] += float((torch.exp(log_ratio) - 1.0 - log_ratio).mean())
                agg["clip_frac"] += float((torch.abs(ratio - 1.0) > cfg.clip_eps).float().mean())
                var_y = torch.var(ret)
                agg["ev"] += float(1.0 - torch.var(ret - v) / var_y) if var_y > 1e-8 else 0.0
                agent_metrics[a] = {
                    "policy_loss": float(-pg.mean()),
                    "value_loss": float(((v - ret) ** 2).mean()),
                    "entropy": float(dist.entropy().mean()),
                    "approx_kl": float((torch.exp(log_ratio) - 1.0 - log_ratio).mean()),
                    "clip_fraction": float((torch.abs(ratio - 1.0) > cfg.clip_eps).float().mean()),
                    "explained_variance": (
                        float(1.0 - torch.var(ret - v) / var_y) if var_y > 1e-8 else 0.0
                    ),
                }
        self.updates += 1
        self.last_agent_metrics = agent_metrics
        result = {k: v / len(self.ids) for k, v in agg.items()}
        result["ent_coef"] = ent_coef
        return result

    def _entropy_coef(self) -> float:
        if self.cfg.ent_coef_final is None or self.cfg.total_updates is None:
            return self.cfg.ent_coef
        progress = min(1.0, self.updates / max(1, self.cfg.total_updates - 1))
        return self.cfg.ent_coef + progress * (self.cfg.ent_coef_final - self.cfg.ent_coef)

    # --- persistence ---
    def save(self, path: str) -> None:
        torch.save({a: n.state_dict() for a, n in self.nets.items()}, path)

    def load(self, path: str) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        for a, n in self.nets.items():
            n.load_checkpoint(sd[a])
