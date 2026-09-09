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
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 4
    minibatch: int = 256
    rollout_steps: int = 1024
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    seed: int = 0


class IndependentPPO:
    def __init__(
        self,
        agent_ids: list[str],
        obs_dim: int = 8,
        n_actions: int = 3,
        cfg: PPOConfig | None = None,
    ):
        self.cfg = cfg or PPOConfig()
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.ids = list(agent_ids)
        self.nets = {a: ActorCritic(obs_dim, n_actions) for a in self.ids}
        self.opts = {
            a: torch.optim.Adam(n.parameters(), lr=self.cfg.lr) for a, n in self.nets.items()
        }
        self.cur_obs: dict[str, np.ndarray] | None = None
        self.global_steps = 0

    # --- data collection ---
    def reset(self, env) -> None:
        obs, _ = env.reset(seed=self.cfg.seed)
        self.cur_obs = obs

    def rollout(self, env):
        """Collect rollout_steps synchronized transitions. Returns (buf, advs, rets, stats)."""
        cfg, ids, T = self.cfg, self.ids, self.cfg.rollout_steps
        assert self.cur_obs is not None, "call reset(env) first"
        buf = {
            a: {
                "obs": np.zeros((T, 8), np.float32),
                "act": np.zeros(T, np.int64),
                "logp": np.zeros(T, np.float32),
                "val": np.zeros(T, np.float32),
                "rew": np.zeros(T, np.float32),
                "term": np.zeros(T, np.float32),
            }
            for a in ids
        }
        ep_returns, ep_lens, ep_a, ep_b = [], [], [], []
        for t in range(T):
            acts: dict[str, int] = {}
            for a in ids:
                act, lp, v = self.nets[a].sample(self.cur_obs[a])
                acts[a] = act
                buf[a]["obs"][t] = self.cur_obs[a]
                buf[a]["act"][t] = act
                buf[a]["logp"][t] = lp
                buf[a]["val"][t] = v
            nobs, rews, terms, truncs, info = env.step(acts)
            terminated = all(terms.values())
            done = terminated or all(truncs.values())
            for a in ids:
                buf[a]["rew"][t] = rews[a]
                buf[a]["term"][t] = 1.0 if terminated else 0.0
            self.global_steps += 1
            if done:
                ep_returns.append(info["scores"]["A"] - info["scores"]["B"])
                ep_lens.append(info["steps"])
                ep_a.append(info["scores"]["A"])
                ep_b.append(info["scores"]["B"])
                self.cur_obs, _ = env.reset()
            else:
                self.cur_obs = nobs
            if t == T - 1:
                last_nobs, last_term = nobs, terminated
        # GAE per agent (truncations bootstrap; terminations don't).
        advs, rets = {}, {}
        with torch.no_grad():
            last_vals = {}
            for a in ids:
                if last_term:
                    last_vals[a] = 0.0
                else:
                    _, v = self.nets[a](
                        torch.as_tensor(last_nobs[a], dtype=torch.float32).unsqueeze(0)
                    )
                    last_vals[a] = float(v.item())
            for a in ids:
                vals, adv = buf[a]["val"], np.zeros(T, np.float32)
                gae = 0.0
                for t2 in range(T - 1, -1, -1):
                    mask = 1.0 - buf[a]["term"][t2]
                    nv = last_vals[a] if t2 == T - 1 else vals[t2 + 1]
                    delta = buf[a]["rew"][t2] + cfg.gamma * mask * nv - vals[t2]
                    gae = delta + cfg.gamma * cfg.gae_lambda * mask * gae
                    adv[t2] = gae
                advs[a], rets[a] = adv, adv + vals
        stats = {
            "episodes": len(ep_returns),
            "mean_return_A": float(np.mean(ep_returns)) if ep_returns else float("nan"),
            "mean_len": float(np.mean(ep_lens)) if ep_lens else float("nan"),
            "mean_A": float(np.mean(ep_a)) if ep_a else float("nan"),
            "mean_B": float(np.mean(ep_b)) if ep_b else float("nan"),
        }
        return buf, advs, rets, stats

    # --- optimization ---
    def update(self, buf, advs, rets) -> dict[str, float]:
        cfg = self.cfg
        agg = {"pg": 0.0, "v": 0.0, "ent": 0.0}
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
                    loss = (
                        -pg.mean() + cfg.vf_coef * vl.mean() - cfg.ent_coef * dist.entropy().mean()
                    )
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
        return {k: v / len(self.ids) for k, v in agg.items()}

    # --- persistence ---
    def save(self, path: str) -> None:
        torch.save({a: n.state_dict() for a, n in self.nets.items()}, path)

    def load(self, path: str) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        for a, n in self.nets.items():
            n.load_state_dict(sd[a])
