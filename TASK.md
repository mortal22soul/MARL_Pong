# Project Specification: Emergent Coordination in 2v2 Pong via Independent PPO

*Comprehensive technical design document — course project*

---

## 1. Project Overview

**Research question:** Can independently learning agents develop effective cooperative strategies when competing against another team of independently learning agents, without explicit role assignment or communication?

**Approach:** A custom 2D 2v2 Pong environment with four independently trained PPO agents (two per team). Teammates share a team-level reward; there is no centralized critic, parameter sharing, or communication channel between agents in the primary experiment.

**Framing document:** This project is positioned against Li et al. (2025), *Multi-Agent Reinforcement Learning in Games: Research and Applications* (Biomimetics, 10(6), 375), which supplies the taxonomy (value-based / policy-gradient / search-based methods) and the challenge framing (non-stationarity, credit assignment) used throughout this spec and the final report.

---

## 2. Confirmed Technical Decisions

| Decision | Choice | Notes |
|---|---|---|
| Package/environment manager | **uv** | Fast, modern; replaces pip+venv workflow |
| Python version | **3.14** | PyTorch and NumPy both support 3.14 as of current date. SB3 itself switched its `[extra]` install target from `pygame` to `pygame-ce` in v2.8.0, specifically because plain `pygame` does not reliably ship 3.14 wheels while `pygame-ce` does — see dependency list below. **Fallback: 3.12** if any other dependency lacks a 3.14 build on your specific OS/architecture — check this during environment setup (Phase 0) before committing further. |
| Multi-agent training framework | **Shared environment → four agent-specific PPO training interfaces → synchronized environment stepping → independent PPO updates** | The exact PPO integration (thin wrapper around SB3 internals vs. a small custom PyTorch PPO implementation) is an **open implementation decision to be resolved during Phase 4**, not committed to here — see Section 6. RLlib is ruled out for this project (too much config overhead for course scope). |
| Experiment tracking | **MLflow** (local file-based tracking, `./mlruns`) | No remote tracking server needed for a solo course project; local file store is zero-setup and still gives you run comparison, metric plots, and artifact storage. |
| Linting/formatting | **ruff + black** | ruff for linting (fast, replaces flake8/isort), black for formatting |
| Seeds per condition | **3 seeds** | Balances statistical honesty against compute/time budget (see reasoning above) |
| Rendering | **Pygame** (via the environment's own `render()` method) | Headless during training (`render_mode=None`), enabled only for final demo footage capture |

**Open parameter — not yet decided, deliberately:** paddle vertical-coverage overlap between teammates. This is a substantive environment-design choice (too little overlap trivializes coordination; too much makes it meaningless). It should **not** simply be tuned until PPO agents "learn well" — that would introduce a hidden experimental degree of freedom into the results. Instead, Phase 1 tests a small number of candidate overlap values using scripted/heuristic agents only, selects one against an explicit criterion (both teammates can plausibly reach the shared region; neither teammate can cover the full defensive area alone; neither has an obviously dominant region), and **freezes that value before any PPO training begins**.

---

## 3. Environment Design

### 3.1 Arena and Physics

- Rectangular 2D arena, origin top-left, normalized coordinate space internally (e.g., `[-1, 1]` on both axes) for observation stability.
- Team A occupies the left boundary, Team B the right boundary.
- Ball has continuous position and velocity; bounces off top/bottom walls; initial trajectory randomized per episode (angle and speed within a bounded range) to prevent memorized fixed responses.
- Each team has two paddles (vertical bars) with a constrained but *overlapping* vertical movement range (see open parameter above).
- Paddle movement has minimal inertia (not instant teleport) to add a small amount of realistic control friction.

### 3.2 Agents

Four independent agents: A1, A2 (Team A), B1, B2 (Team B). Each agent controls exactly one paddle total 4 (2 on a side). No agent is assigned a fixed region in advance — any positional specialization must emerge from training.

### 3.3 Action Space

Discrete, per agent:
```
0 = stay
1 = move up
2 = move down
```

### 3.4 Observation Space

Per agent, normalized to `[-1, 1]` or similar stable range:
```
own paddle: y position, y velocity
ball: x position, y position, x velocity, y velocity
teammate: y position, y velocity
```
Opponent positions are excluded **from the primary experiment by design, not as a placeholder to be filled in later**: adding opponent observations would gradually turn the research question from "can teammates coordinate using local state + teammate state?" into a globally-informed-controller problem, diluting the specific coordination claim this project is built around. It remains a possible stretch extension (see Section 9), but is not planned for the primary run.

### 3.5 Reward Structure

Shared team-level reward:
```
Team A: +1 if Team B misses the ball, -1 if Team A misses the ball
Team B: +1 if Team A misses the ball, -1 if Team B misses the ball
```
Both teammates receive the identical signal. This is a deliberate simplification (see Section 8, Credit Assignment).

### 3.6 Episode Structure

- Episode ends after a fixed point threshold (e.g., first team to 5 or 11 points) or a maximum timestep cap, whichever comes first — the cap prevents pathological infinite rallies from stalling training.
- `reset()` re-centers paddles and randomizes ball trajectory.

---

## 4. Repository Structure

```
marl-pong/
├── pyproject.toml            # uv-managed dependencies, ruff/black config
├── README.md
├── environment/
│   ├── pong_env.py            # Core environment class (Gym/PettingZoo-style API)
│   ├── physics.py             # Ball/paddle collision and movement logic
│   └── config.py              # Arena size, paddle overlap, speed constants
├── agents/
│   ├── multi_agent_ppo.py     # Custom outer loop: 4x SB3 PPO models, synchronized stepping
│   └── policies.py            # Shared MLP architecture config for policies
├── baselines/
│   ├── random_agent.py
│   └── heuristic_agent.py     # Ball-tracking scripted paddle
├── evaluation/
│   ├── evaluate.py            # Deterministic evaluation episode runner
│   ├── metrics.py             # Win rate, rally length, reward variance calculations
│   └── trajectories.py        # Position logging + heatmap generation
├── experiments/
│   ├── exp_1v1_baseline.py
│   ├── exp_2v2_main.py
│   └── exp_2v2_frozen_opponent.py   # optional, time-permitting
├── tracking/
│   └── mlflow_setup.py        # MLflow experiment/run initialization helpers
├── tests/
│   ├── test_env_sanity.py     # Random-agent validation (Phase 3)
│   └── test_physics.py
├── results/
│   ├── plots/
│   ├── models/
│   └── videos/
└── report/
    └── (final markdown/PDF report, references the framing paper)
```

---

## 5. Dependency Stack

Managed via `uv` (`pyproject.toml`). Core dependencies:

```toml
[project]
name = "marl-pong"
requires-python = ">=3.14"
dependencies = [
    "torch",
    "stable-baselines3",
    "gymnasium",
    "pygame-ce",   # NOT "pygame" — plain pygame does not reliably ship 3.14 wheels;
                   # SB3 itself switched its own [extra] target to pygame-ce in v2.8.0
                   # for exactly this reason.
    "numpy",
    "mlflow",
]

[dependency-groups]
dev = [
    "ruff",
    "black",
    "pytest",
]
```

`uv sync` installs the locked environment; `uv run pytest` runs the sanity-check test suite. Note: since SB3 ships a pure-Python wheel, its own compatibility with 3.14 is governed entirely by whether PyTorch, NumPy, and pygame-ce install cleanly on your system — verify this in Phase 0 before writing any project code.

---

## 6. Multi-Agent Training Architecture

**This section describes a fixed high-level shape, with one internal implementation choice deliberately left open.**

Fixed shape (not in question):

1. Environment's `step()` accepts a dict of four actions (keyed by agent ID) and returns a dict of four observations, rewards, and done flags.
2. Each of the four agents has its own independent PPO training interface — its own policy, its own rollout collection, its own updates. No shared critic, no parameter sharing, no cross-agent gradient flow.
3. Environment stepping is synchronized: one shared environment step per timestep, with all four agents' actions applied simultaneously, then each agent's own observation/reward routed back to its own training interface.
4. Device: `device="cpu"` throughout (see prior discussion — GPU offers no benefit for this network size/observation size).

**Open implementation decision — to be resolved in Phase 4, not here:** *how* each agent's PPO update is actually implemented. Standard Stable-Baselines3 `PPO` objects are not designed to have transitions pushed into their rollout buffers from an external multi-agent loop through a public API — their rollout collection and update logic are coupled to SB3's own single-agent `env.step()` loop. Committing to "four `PPO()` instances with manually-fed buffers" without validating that this is actually feasible through SB3's public interface would risk turning Phase 4 into reverse-engineering SB3 internals rather than building the actual project. Two candidate resolutions, either acceptable:

- **(a) A thin wrapper around SB3 internals** — if a clean way to drive SB3's rollout buffer and `train()` call from outside its own `learn()` loop can be validated early in Phase 4, this preserves SB3's well-tested PPO update logic (clipping, GAE, etc.).
- **(b) A small custom PyTorch PPO implementation** — given that the PPO algorithm itself is not this project's research contribution, and the policy network/observation space here are tiny, a from-scratch PPO loop (actor-critic MLP, GAE advantage computation, clipped surrogate loss) is a bounded, well-understood amount of code, and may actually be *less* effort than fighting SB3's single-agent assumptions.

The decision between (a) and (b) should be made early in Phase 4 based on a quick feasibility spike (a few hours, not days) attempting (a) first, falling back to (b) if it proves awkward. This is flagged explicitly so that timeline risk is visible now rather than discovered mid-implementation.

---

## 7. Experiment Design

| Condition | Purpose | Seeds |
|---|---|---|
| 1v1 baseline | Sanity check that PPO learns basic paddle control in this framework; not a matched-geometry comparison to 2v2 (see Scope/Limitations in the final report) | 3 |
| 2v2 simultaneous learning | Main MARL condition — all four agents learn concurrently | 3 |
| 2v2 frozen-opponent (optional, time-permitting) | Isolates genuine skill improvement from opponent-driven instability, directly probing non-stationarity | 3 (if time allows) |

Each seed run is logged as a separate MLflow run within the same experiment, tagged by condition, so final comparison plots can aggregate across seeds (mean ± spread) rather than reporting single-run curves.

---

## 8. Evaluation Protocol

- Post-training, run deterministic evaluation episodes (exploration disabled) — do not evaluate using training-time rewards alone.
- Metrics per evaluation run:

| Metric | Purpose |
|---|---|
| Win rate | Overall team performance |
| Average episode reward | Learning objective tracking |
| Rally length | Game quality / engagement proxy |
| Reward variance | Training/evaluation stability indicator. **Not** a direct measurement of non-stationarity — it can equally reflect stochasticity, opponent strength, reward sparsity, or seed variation. Non-stationarity itself is discussed qualitatively through learning dynamics, and (if run) compared directly against the frozen-opponent condition. |
| Paddle trajectory / heatmap | Behavioral/coordination evidence |
| **Teammate positional specialization** | The strongest available evidence for or against the actual research question. Compute, per teammate: mean vertical position, spatial coverage (e.g., variance or range of vertical position over an evaluation episode), and the overlap between the two teammates' coverage distributions. This answers *"did A1 and A2 develop complementary defensive roles?"* directly and quantitatively, rather than relying on a qualitative "the heatmap looks coordinated" read. |

- **Credit assignment caveat (carried from the final report):** the shared team reward may produce asymmetric contribution between teammates (one paddle doing most of the defensive work while the other "free-rides"). This is treated as a reportable finding, not a bug to patch mid-project.
- **Coordination claim discipline (carried from the final report):** strong win-rate performance alone does not prove coordination emerged — it could result from game geometry or reward structure. Coordination claims must be backed by trajectory/heatmap evidence (e.g., complementary positioning, reduced overlap over training), not win rate alone.

---

## 9. Execution Timeline

| Phase | Deliverable |
|---|---|
| 0. Environment setup | `uv` project initialized; Python 3.14 (or 3.12 fallback) confirmed working with torch/SB3/pygame installed |
| 1. Environment prototyping | Playable 2v2 Pong with scripted paddles; paddle-overlap parameter chosen and justified |
| 2. Multi-agent interface | Environment exposes dict-based multi-agent step/reset API |
| 3. Random-agent validation | `pytest` suite confirms observations, rewards, termination behave correctly before any training begins |
| 4. Multi-agent PPO infrastructure | Custom outer loop built and tested on a trivial case (e.g., verify 1v1 reduces correctly to standard SB3 usage) |
| 5. Baselines | Random policy and heuristic (ball-tracking) agents implemented for reference performance |
| 6. Main training | 1v1 baseline and 2v2 main condition trained, 3 seeds each, logged to MLflow |
| 7. Evaluation | Deterministic evaluation episodes; metrics computed and logged |
| 8. Behavioral analysis | Trajectories/heatmaps generated; coordination evidence (or its absence) documented honestly |
| 9. Optional frozen-opponent experiment | Only if Phases 1–8 are complete with time remaining |
| 10. Reporting | Final write-up assembled, gameplay footage rendered and captured |

---

## 10. Final Deliverables

**Required (project is successful without any of the "optional" items below):**

1. Working 2v2 Pong environment (custom, Pygame-rendered via `pygame-ce`) with four independently controllable agents; overlap parameter chosen per the frozen-before-training methodology in Section 3.1.
2. Independent PPO training (SB3-wrapped or custom PyTorch — per Phase 4 decision) across 3 seeds.
3. Evaluation: win rate, average reward, rally length, reward variance, teammate positional specialization.
4. Coordination analysis: trajectory/heatmap plus the quantitative specialization metrics from Section 8.
5. Short rendered gameplay footage of trained agents.
6. Final written report connecting results to the MARL challenges framed in Li et al. (2025), explicitly distinguishing genuine coordination evidence (specialization metrics, trajectory analysis) from geometry- or reward-structure-driven performance (win rate alone).

**Optional (do not let these become required for project success):**

7. Frozen-opponent experiment (Section 7), only if time remains after all required items are complete.

---

## 11. Scope and Limitations (carried forward)

- No centralized-critic methods (MAPPO, QMIX, MADDPG), no parameter sharing, no partial-observability ablation, no larger team sizes — these are explicitly out of scope, noted as future work only.
- 1v1 vs. 2v2 is not a controlled causal comparison of team size (differing geometry/coverage confounds it) — 1v1 serves only as an implementation sanity check.
- The environment is a simplified 2D abstraction, not a physically realistic simulation — this is intentional, matching the framing paper's own use of toy environments to isolate strategic/coordination phenomena from implementation complexity.

---

## 12. References

- Li, H., Yang, P., Liu, W., Yan, S., Zhang, X., & Zhu, D. (2025). Multi-Agent Reinforcement Learning in Games: Research and Applications. *Biomimetics*, 10(6), 375. https://doi.org/10.3390/biomimetics10060375
- Pong from Pixels: Andrej Karpathy
