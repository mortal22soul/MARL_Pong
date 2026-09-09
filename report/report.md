# Emergent Coordination in 2v2 Pong via Independent PPO

Course project report — methodology and honest results, including the ones
that did not come out the way the research question hoped for.

## 1. Research question

Can independently learning agents develop effective cooperative strategies
when competing against another team of independently learning agents, without
explicit role assignment or communication? The project is framed against Li
et al. (2025), *Multi-Agent RL in Games* (Biomimetics 10(6):375), whose
taxonomy (value-based / policy-gradient / search-based) and challenge framing
(non-stationarity, credit assignment) structure the analysis below.

## 2. Environment and method

- Custom 2v2 Pong (`environment/`): normalized `[-1, 1]` arena, inertial
  paddles with overlapping but constrained vertical ranges, escalating ball
  speed on hits, shared team-level reward (+1/-1 per point, both teammates
  receive the identical signal).
- Four independent PPO agents (`agents/`), each with its own 64×64
  actor-critic MLP, synchronized environment stepping, GAE + clipped
  surrogate updates, CPU-only torch. No shared critic, no parameter sharing,
  no communication, and — by deliberate design — no opponent observations:
  the coordination question is *local state + teammate state only*.
- Observations per agent (8-dim): own paddle y/vy, ball x/y/vx/vy, teammate
  paddle y/vy.

### Deviations from the original spec (TASK.md)

| Spec'd | Delivered | Why |
|---|---|---|
| SB3-wrapped or custom PPO (open decision, §6) | Custom PyTorch PPO — option (b) | SB3's single-agent rollout buffer coupling made option (a) reverse-engineering, not engineering |
| MLflow tracking | Per-iteration CSV logs | Zero-dependency, sufficient for 3-seed aggregation in `analysis/`; MLflow was process overhead, not signal |
| Paddle overlap swept and frozen before training | Kept at 0.4 (temporary default) | Heuristic sweep never re-opened; **this is a protocol caveat** — the frozen-before-training methodology was not executed |

## 3. Experimental protocol

- Conditions: 1v1 baseline (sanity check, not a matched comparison — the
  geometry differs) at 300k steps; 2v2 main condition at 1M steps.
- 3 seeds per condition (0, 1, 2), freshly initialized — no warm starts in
  the headline numbers. (An earlier seed-0 warm-start continuation run,
  `2v2_seed0_cont1M`, is kept under `results/` as a documented extra.)
- Evaluation: deterministic (greedy) episodes, 60 per run, against three
  kinds of opposition: self-play (both teams trained), the near-perfect
  ball-tracking heuristic team, and the random team. Uniform episode counts
  across all runs and opponents.
- Coordination evidence is taken from positional specialization metrics
  (per-paddle y mean/std/range and teammate coverage overlap), trajectory
  heatmaps, and coverage overlap across training checkpoints — explicitly
  *not* from win rate alone, which can reflect game geometry or reward
  structure rather than coordination.

## 4. Results

### 4.1 Learning (3 seeds, mean ± spread)

See `results/plots/curves_mean_return_A.png` and the aggregated table in
`results/summary.md`. (Filled after the final eval pass.)

### 4.2 Evaluation vs the three opponents

(TBD — table in `results/summary.md`.)

### 4.3 Specialization and coordination evidence

(TBD — heatmap figure `results/plots/coverage_heatmaps.png`, checkpoint
overlap sweep `results/plots/overlap_over_training.png`.)

## 5. Discussion

(TBD — to be written against the actual numbers: team asymmetry, behavior
against perfect scripted defense, non-stationarity dynamics, and whether the
specialization metrics show complementary roles or dominant-paddle
free-riding.)

## 6. Limitations

- 1v1 vs 2v2 is not a controlled causal comparison of team size (differing
  geometry/coverage confounds it).
- No centralized-critic baselines (MAPPO/QMIX/MADDPG), no parameter-sharing
  ablation, no larger teams — out of scope by design (TASK.md §11).
- Simplified 2D abstraction; paddle overlap was not swept before training.
- Single hardware budget (CPU), fixed hyperparameters per condition.

## References

- Li, H., Yang, P., Liu, W., Yan, S., Zhang, Z., & Zhu, D. (2025).
  Multi-Agent Reinforcement Learning in Games: Research and Applications.
  *Biomimetics*, 10(6), 375. https://doi.org/10.3390/biomimetics10060375
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.
  arXiv:1707.06347.
