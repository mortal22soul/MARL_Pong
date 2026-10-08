# Emergent Coordination in 2v2 Pong via Independent PPO

Course project report: methodology and honest results, including the parts
that did not come out the way the research question hoped.

## 1. Research question

Can independently learning agents develop effective cooperative strategies
when competing against another team of independently learning agents, without
explicit role assignment or communication? The project is framed against Li et
al. (2025), *Multi-Agent RL in Games* (Biomimetics 10(6):375), whose taxonomy
(value-based / policy-gradient / search-based) and challenge framing
(non-stationarity, credit assignment) structure the analysis below.

## 2. Environment and method

- Custom 2v2 Pong (`environment/`): normalized `[-1, 1]` arena, inertial
  paddles with overlapping but constrained vertical ranges, escalating ball
  speed on hits, shared team-level reward (+1/-1 per point, both teammates
  receive the identical signal, no per-hit shaping).
- Overlapping paddles resolve a simultaneous contact by giving the ball to the
  paddle closest to it, so neither teammate has a fixed-order priority.
- Four independent PPO agents (`agents/`): each has its own 64×64 actor and
  64×64 critic MLP, its own optimizer and its own clipped-surrogate updates.
  Environment stepping is synchronized; there is no shared critic, no parameter
  sharing, no communication and, by design, no opponent observations.
- Observations per agent (8-dim): own paddle y/vy, ball x/y/vx/vy, teammate
  y/vy, re-expressed in a side-invariant frame (distance to ball along the
  paddle's facing direction, relative ball/teammate offsets).

### Environment calibration (before any PPO training)

TASK.md requires the paddle-overlap parameter to be chosen with scripted
agents only and frozen before PPO. `experiments/calibrate_env.py` swept overlap
`{0.15, 0.25, 0.35, 0.45}` × ball-speed scale `{1.0, 1.15}` with range-holding,
reactive, predictive and random scripted teams (60 episodes per game,
`results/calibration_v2.json`). The frozen profile is `paddle_overlap = 0.15`
and speed scale `1.15`: reactive-vs-reactive play is decisive in 68% of
episodes (not a permanent stalemate), predictive trackers still draw every
game, and reactive teams beat random 100%. These values live in
`environment/config.py` and were not changed afterwards.

### Deviations from the original spec (TASK.md)

| Spec'd | Delivered | Why |
|---|---|---|
| SB3-wrapped or custom PPO (open decision, §6) | Custom PyTorch PPO, option (b) | SB3's single-agent rollout coupling made option (a) reverse-engineering rather than engineering |
| MLflow tracking | Per-iteration CSV logs | No extra dependency; sufficient for the analysis in `analysis/` |
| 1v1 sanity baseline, 3 seeds | Dropped | The earlier 1v1 runs used a superseded environment; 1v1 is not a matched comparison to 2v2 anyway (§11) |
| 3 seeds per reported condition | One showcased run (seed 0); a 3-seed run of the same configuration is summarized in §4.4 | See §6 |
| Frozen-opponent condition (optional) | Not run | Optional per TASK.md §10 |

## 3. Experimental protocol

- **Reported run `s1_5M`** (named for "sweep 1, 5M steps"): all four agents
  learn concurrently for 5,000,000 environment steps, seed 0.
- **PPO hyperparameters:** rollout 4096 steps per update, 6 epochs, minibatch
  512, lr 3e-4, γ = 0.995, GAE λ = 0.99, clip 0.2, value coefficient 0.5,
  entropy coefficient annealed linearly 0.01 → 0.001, gradient-norm clip 0.5.
  These are the CLI defaults; `results/models/s1_5M/run_config.json` holds the
  exact record.
- **How the run was chosen:** a single-seed sweep over rollout size, learning
  rate, γ and entropy (1M steps each), then longer runs of the best
  configuration (3 seeds × 3M steps, and seed 0 × 5M steps). Candidates were
  ranked by fixed-opponent win rate and a head-to-head cross-play matrix (§4.4).
  This is a selection step, so the reported numbers are the best observed run,
  not an average.
- **Evaluation:** deterministic (greedy) play, 60 episodes per opponent, episode
  seeds 1000–1059. The opponents are self-play (the four learned policies), random
  paddles, the reactive scripted tracker ("heuristic": follows the ball's
  current y), and the predictive scripted tracker (moves to the ball's
  projected intercept, including wall bounces). An episode ends at 5 points or
  2000 steps; the winner is the team ahead at the end, and a tie counts as a draw.
- **Coordination evidence** comes from positional and functional metrics
  (coverage overlap, home separation, per-paddle contact share and impact band,
  defence-time ball error) and from heatmaps. It is not taken from win rate
  alone, which can reflect geometry or reward structure.

## 4. Results

Records: `results/eval_s1_5M*.json`, table: `results/summary.md`.

### 4.1 Learning dynamics (`results/plots/curves_mean_return_A.png`)

The smoothed self-play Team-A return first collapses to about −3.4 around 400k
steps, because Team B learns to return the ball first. It recovers to positive
by roughly 1.2M steps, peaks near +1 at 2.5M and then settles in a +0.3 to +0.5
band for the last 2M steps. The early swing is the non-stationarity Li et al.
describe: each team's improvement changes the other team's environment.
The run does stabilize, but it stabilizes with Team A ahead, not at a symmetric
equilibrium.

### 4.2 Evaluation against fixed and learned opposition

| Opponent (Team B) | Win A | Win B | Draw | Return A | Mean ep. length |
|---|---|---|---|---|---|
| random | **1.00** | 0.00 | 0.00 | +4.48 | 1330 |
| reactive tracker (heuristic) | **0.63** | 0.07 | 0.30 | +0.68 | 2000 |
| predictive tracker | 0.00 | 0.13 | 0.87 | −0.20 | 2000 |
| self-play (learned Team B) | 0.53 | 0.03 | 0.43 | +0.73 | 2000 |

Findings:

1. **The learned team beats random every time and usually beats the reactive
   tracker.** Against the tracker, every episode hits the 2000-step cap. The
   0.63 win rate means being ahead on points at the cap, from a low-scoring
   game, not first-to-5. Mean return is +0.68 points per episode.
2. **The predictive tracker is not beaten.** 87% of games are scoreless draws
   and the tracker wins 13%. A projected-intercept policy remains a stronger
   defender than anything PPO learned here.
3. **Self-play is asymmetric:** Team A wins 53% and Team B 3%. With a shared team
   reward and simultaneous learning, one team settled into the stronger joint
   strategy and kept it. This is the team-level credit-assignment asymmetry that
   TASK.md §8 anticipated.

### 4.3 Specialization and coordination evidence

(`results/plots/coverage_heatmaps.png`, `results/plots/overlap_over_training.png`)

| Metric (Team A, self-play eval) | A1 | A2 |
|---|---|---|
| mean y (+y is down) | +0.27 | −0.10 |
| share of team contacts | 43% | 57% |
| contacts in its outer band | 74% (lower) | 55% (upper) |
| contacts in the shared middle band | 26% | 45% |
| mean defence-time ball error | 0.37 | 0.26 |

Coverage overlap is 0.32 and home separation is 0.37. The vs-heuristic,
vs-predictive and vs-random evaluations show the same pattern: A2 handles 59–62%
of team contacts, and the overlap stays between 0.30 and 0.34.

- **Both teammates cover their own side, but neither stays home.** The heatmaps show each
  paddle spending much of its time at the inner edge of its range, the
  boundary of the shared middle band, and making excursions towards its wall. The
  same structure appears on Team B. The learned formation is "both guard the
  middle, each sweeps its own side", not a static top/bottom split.
- **The outer-band split is partly forced by geometry.** With overlap 0.15,
  A1 physically cannot reach the upper band and A2 cannot reach the lower band,
  so their 0% cross-band contacts are not evidence of coordination by
  themselves.
- **The shared middle band is where allocation is learned.** Both paddles can
  reach it. In self-play A2 takes about 70% of middle-band returns (2.0 vs 0.9
  per episode), so the team has a de facto primary middle defender, chosen
  without any assignment or communication.
- **Overlap rises over training** (0.04 at 82k steps → about 0.3 at 5M). Early
  policies barely move. Later policies converge on the shared middle and chase
  aggressively, so coverage overlaps more even as functional roles appear.
  Lower overlap is therefore not a usable proxy for better coordination here.
- **Verdict on the research question: partial yes.** Without communication or
  assigned roles, teammates learn a consistent division of labour: each covers
  its own outer band, and one is the main defender of the contested middle.
  The team using it beats random play and the reactive tracker. This is
  functional evidence, not only occupancy. It does not reach the strength of a
  hand-written predictive defender, and it rests on one showcased run.

### 4.4 Selection evidence and robustness (archived runs)

The comparison runs were removed from the repository to keep it focused. Their
checkpoints and records are kept in an off-repo archive. The numbers below come
from the same 60-episode protocol.

| Run | Seeds | Steps | vs random | vs reactive tracker |
|---|---|---|---|---|
| **s1_5M (reported)** | 1 | 5M | **1.00** | **0.63** |
| same config, 3 seeds | 3 | 3M | 0.98 ± 0.01 | 0.24 ± 0.14 |
| same config | 1 | 1M | 0.97 | 0.12 |
| rollout 1024, 4 epochs, minibatch 256 | 1 | 1M | 0.87 | 0.13 |
| entropy 0.03 | 1 | 1M | 0.75 | 0.02 |
| γ = 0.999 | 1 | 1M | 0.73 | 0.05 |
| lr 1e-4 | 1 | 1M | 0.72 | 0.10 |
| earlier protocol, 3 seeds | 3 | 1M | 0.81 ± 0.03 | 0.07 ± 0.09 |

In head-to-head cross-play (60 episodes per pairing), s1_5M's Team A finished
ahead on net against every other candidate's Team B:

| Team B from | 3M seed 0 | 3M seed 1 | 3M seed 2 | earlier protocol 1M |
|---|---|---|---|---|
| s1_5M Team A wins | 0.30 | 0.83 | 0.17 | 0.92 |
| s1_5M Team A loses | 0.13 | 0.00 | 0.10 | 0.00 |

These results show:

- The larger-rollout configuration generalizes across seeds: all three 3M seeds
  beat random 97–98% of the time.
- Performance against the tracker keeps improving with training budget (0.12 at
  1M, 0.24 mean at 3M, 0.63 at 5M for seed 0). It also varies substantially
  across seeds at 3M (0.12–0.40).

## 5. Discussion

- **Non-stationarity is visible but does not prevent learning.** The early
  collapse and recovery in §4.1 is the moving-target problem. A larger rollout
  (4096 steps, about two full-length episodes per update) and more training
  were the changes that most improved fixed-opponent strength in the sweep.
- **Credit assignment shows up at team level.** The identical team reward never
  says which paddle conceded or saved a point, yet the team still splits the
  contested middle unevenly. Self-play ends with one team clearly stronger. Both
  match TASK.md §8: uneven contribution and asymmetric equilibria are findings,
  not bugs.
- **Win rate would have overstated the story.** Beating random needs no
  coordination at all. The functional metrics (middle-band contact allocation)
  are the evidence for coordination, and the geometry-forced outer-band split
  must be discounted.

## 6. Limitations

- **Single showcased seed, chosen after comparing runs.** AGENTS.md treats
  single-seed numbers as non-convergence claims. The 3-seed 3M run of the same
  configuration (§4.4) is the stronger cross-seed estimate. To make the 5M
  result a protocol-grade claim, run seeds 1 and 2 at 5M steps.
- Against the scripted trackers every game reaches the step cap. Win rates
  there come from low-scoring games, so per-point return is the better measure
  of margin.
- No frozen-opponent condition, so skill growth and opponent drift are not
  separated.
- No centralized-critic baselines (MAPPO/QMIX/MADDPG), no parameter-sharing
  ablation and no larger teams; these are out of scope by design (TASK.md §11).
- Simplified 2D abstraction. The overlap value comes from a scripted
  calibration, not from PPO performance.

## References

- Li, H., Yang, P., Liu, W., Yan, S., Zhang, Z., & Zhu, D. (2025).
  Multi-Agent Reinforcement Learning in Games: Research and Applications.
  *Biomimetics*, 10(6), 375. https://doi.org/10.3390/biomimetics10060375
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.
  arXiv:1707.06347.
