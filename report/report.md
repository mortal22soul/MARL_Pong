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

All numbers: 3 seeds, 60 deterministic episodes per run per opponent; full
table in `results/summary.md`, records under `results/eval_*.json`.

### 4.1 Learning dynamics (Figure: `results/plots/curves_2v2.png`)

Smoothed self-play returns per seed do **not** converge to an equilibrium.
Instead they oscillate in multi-hundred-thousand-step waves (one team's edge
builds, erodes, reverses), and all three seeds drift downward over their last
~200k steps. This is the non-stationarity signature predicted by TASK.md and
Li et al.: each team's improvement is the other team's environment change, so
the "target" keeps moving. Seed identity matters enormously — which team is
stronger at 1M steps is essentially a coin flip (see 4.2), and no seed's
curve predicts another's.

### 4.2 Evaluation vs the three opponents

| condition | opponent | win A | win B | return A | mean ep len |
|---|---|---|---|---|---|
| 2v2 @1M | random | 0.68 ± 0.27 | 0.28 ± 0.27 | +1.56 ± 2.13 | 1311 |
| 2v2 @1M | self | 0.43 ± 0.24 | 0.46 ± 0.37 | −0.22 ± 1.70 | 1616 |
| 2v2 @1M | heuristic | 0.02 ± 0.03 | 0.87 ± 0.17 | −3.04 ± 1.56 | 1763 |
| 1v1 @300k | random | 0.48 ± 0.15 | 0.52 ± 0.15 | −0.19 ± 1.03 | 951 |
| 1v1 @300k | self | 0.39 ± 0.08 | 0.56 ± 0.09 | −0.47 ± 0.43 | 1016 |
| 1v1 @300k | heuristic | 0.00 ± 0.00 | 0.99 ± 0.02 | −4.29 ± 0.64 | 1329 |

Findings:

1. **2v2 beats random (0.68) but with ±0.27 seed spread** — one seed wins
   ~90%+, another hovers near 50%. Skill relative to a fixed opponent is
   real but seed-fragile at this budget.
2. **The scripted near-perfect tracker beats every trained team** (2v2 0.02,
   1v1 0.00 win rate). PPO-learned play remains qualitatively worse than a
   hand-written tracking policy after 1M steps.
3. **Self-play stays balanced on average** (0.43/0.46) but the balance is
   per-seed asymmetric: seed 1 has team A at 0.67, seeds 0 and 2 have team B
   at 0.58/0.70. With a shared team reward and simultaneous learning,
   accidental asymmetric skill equilibria form and persist — the credit-
   assignment caveat from TASK.md §8 showing up as team-level asymmetry.

### 4.3 Specialization and coordination evidence

(Figures: `results/plots/coverage_heatmaps_seed{0,1,2}.png`,
`results/plots/overlap_over_training.png`.)

- **Positional structure emerges, but not one equilibrium.** In evaluation
  play each paddle develops a home band plus chase excursions — e.g. seed 0:
  A1 anchors mid-upper, A2 mid-lower with excursions to the top wall; the
  mirror structure appears on team B. Mean teammate coverage overlap at
  eval is 0.17–0.28 for seeds 0/2.
- **The strongest partition is seed 1: overlap ≈ 0.01** — A1/A2 split the
  field into near-disjoint bands — and that seed's team A is also the one
  that beats its opponent 0.67 in self-play. Specialization and team success
  co-occur in this run, though n=3 forbids a causal claim.
- **Overlap does not fall monotonically over training.** The checkpoint
  sweep (short greedy probes every ~100k steps) shows overlap *rising* from
  ~0.05–0.15 early to ~0.3–0.4 late in all three seeds. Reading: early
  policies barely move (low, narrow occupancy), while later policies occupy
  more of the field as chasing becomes vigorous — shared coverage grows even
  as distinct home bands persist. "Complementary roles" here means distinct
  anchors with overlapping pursuit, not a clean static partition.
- **Verdict on the research question:** partial yes. Distinct, stable,
  complementary positional roles emerge without communication or role
  assignment (the heatmaps are unambiguous), and the best-partitioned seed
  is also the most successful. But coordination is seed-lottery: it is not
  guaranteed, not monotone over training, and win rate alone would have
  overstated it (beating random is achievable with zero coordination).

## 5. Discussion

### v2 follow-up: calibrated collision fairness and corrected aggregation

After the original v1 report, a versioned v2 condition was run with the
predeclared scripted-only calibration profile: `paddle_overlap=0.15`, ball
speed scale 1.15, and symmetric closest-paddle collision selection. Three
fresh 1M-step 2v2 seeds were evaluated with 60 deterministic episodes each.
The standard Team-A-versus-Team-B baseline results are: random `0.86 ± 0.07`
Team-A win rate, reactive tracker `0.04 ± 0.08`, and self-play `0.44 ± 0.16`
for Team A versus `0.44 ± 0.15` for Team B (all `mean ± sample std`).

The v2 random result is stronger than v1's `0.68 ± 0.24`, while the tracker
remains a much stronger fixed policy. Team-A coverage overlap is `0.13 ±
0.09`; this is evidence of positional differentiation only, not sufficient
evidence of coordination by itself. The v2 evaluator therefore records each
paddle's defense-time position and ball error, action distribution, contact
count/share, and home-position separation. Future coordination claims should
use these functional measures alongside heatmaps.

An earlier draft incorrectly pooled paired side-swapped baseline evaluations
with the three standard seeds, creating a six-row aggregate and an apparent
50% tracker win rate. Those paired games are diagnostic side tests, not extra
independent seeds. The corrected aggregator excludes them from headline
tables. Cross-play and frozen-opponent experiments are now implemented as the
next tests of seed-specific co-adaptation and non-stationarity.

- **Non-stationarity dominates the dynamics.** The oscillating curves and
  the coin-flip team asymmetry at 1M are exactly the challenge Li et al.
  attribute to simultaneous learning: no agent's optimization target is
  stationary, so "progress" is a traveling wave, not a trend.
- **Credit assignment is visible at team level.** Both teammates receive the
  identical reward regardless of who touched the ball; the persistent
  per-seed asymmetric equilibria (one team consistently stronger) and the
  dominance of one paddle in some heatmaps are the free-riding/uneven-
  contribution patterns the spec predicted would be reportable findings.
- **What would strengthen the claim (future work):** the optional frozen-
  opponent condition to separate skill improvement from opponent drift; more
  seeds to de-lottery the specialization/success correlation; opponent-
  observation ablation to test whether coordination survives full information.
- **Honest limitations of this evidence:** 3 seeds; the checkpoint sweep
  probes use shortened (500-step) episodes, so its overlap trend is indicative,
  not directly comparable to the 60-episode eval numbers; paddle overlap was
  never swept (see §2 deviations), so the coordination difficulty was fixed
  by an unvalidated default.

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
