# marl-pong — Emergent Coordination in 2v2 Pong via Independent PPO

Complete course project implementing the TASK.md specification: four independent PPO agents learning 2v2 Pong with a shared team reward, no communication, no parameter sharing, and no centralized critic.

## Quickstart

```bash
uv sync
uv run pytest
uv run python demo.py                    # 2v2 heuristic demo, headed
uv run python demo.py --headless         # no window, fast smoke run
uv run python demo.py --human            # control A1 with W/S
uv run python demo.py --human-both       # control A1 (W/S) and A2 (Up/Down)
```

Watch trained v2 agents play each other:

```bash
uv run python demo.py --weights results/models/v2_2v2_seed2_1M/final.pt
```

Play as both A1 and A2 against trained B1/B2:

```bash
uv run python demo.py --human-both --weights results/models/v2_2v2_seed2_1M/final.pt
```

## Demo options

```bash
uv run python demo.py --mode 1v1                    # classic 1v1, full-height paddles
uv run python demo.py --mode 1v1 --human            # play 1v1 yourself
uv run python demo.py --human --opponent random     # beatable opponents
uv run python demo.py --human --points 2            # short match
uv run python demo.py --human-both                  # control both A1 (W/S) and A2 (Up/Down)
uv run python demo.py --human-both --weights results/models/v2_2v2_seed2_1M/final.pt
uv run python demo.py --help                        # all flags
```

In `--human` 2v2 mode you play A1 (upper region); in 1v1 you get the full
height. `--human-both` lets you control both A1 (W/S) and A2 (Up/Down).
ESC or closing the window quits.

## Reinforcement learning (independent PPO)

Each agent has its own actor-critic MLP (64×64, CPU) and PPO updates — no
shared critic, no parameter sharing. One shared env step per timestep.

```bash
# 2v2 main condition (run per seed 0,1,2 for the 3-seed protocol)
uv run python experiments/exp_2v2_main.py --timesteps 1000000 --seed 0 \
  --run-name 2v2_seed0_1M
# 1v1 sanity baseline
uv run python experiments/exp_1v1_baseline.py --timesteps 300000 --seed 0
# continue a run (warm start)
uv run python experiments/exp_2v2_main.py --timesteps 700000 --seed 1 \
  --init-weights results/models/2v2_seed0/final.pt --run-name 2v2_seed0_cont1M
```

Training logs per-iteration stats to `results/logs/<run>.csv`, checkpoints to
`results/models/<run>/`, and auto-runs greedy eval
(`results/eval_<run>.json` + `traj_<run>.npz`).

```bash
# watch trained teams play each other (or --human to play A1 yourself)
uv run python demo.py --weights results/models/v2_2v2_seed2_1M/final.pt
# play as both A1 and A2 against trained B1/B2
uv run python demo.py --human-both --weights results/models/v2_2v2_seed2_1M/final.pt
# record gameplay footage (headless works too)
uv run python demo.py --weights results/models/v2_2v2_seed2_1M/final.pt \
  --episodes 1 --record results/videos/demo.mp4
# uniform eval pass: 60 episodes vs self / heuristic / random per run
uv run python evaluation/eval_suite.py --runs v2_2v2_seed0_1M v2_2v2_seed1_1M v2_2v2_seed2_1M --episodes 60
# cross-play matrix across all 3 seeds
uv run python evaluation/crossplay.py --runs v2_2v2_seed0_1M v2_2v2_seed1_1M v2_2v2_seed2_1M --env-version v2
# figures + cross-seed summary
uv run python analysis/plots.py --curves 2v2=results/logs/v2_2v2_seed0_1M.csv
uv run python analysis/summarize.py   # writes results/summary.md
```

## Results (3-seed protocol, complete)

Headline numbers (60 episodes/run/opponent, mean ± spread across seeds 0-2;
full table in `results/summary.md`, analysis in `report/report.md`):

- **2v2 @ 1M steps** beats random 0.68 ± 0.27, loses to the scripted tracker
  0.02, self-play balanced (0.43 / 0.46) with per-seed team asymmetry.
- **1v1 @ 300k** is ~even vs random (0.48) and loses to the tracker — sanity
  baseline, not a matched comparison.
- **Coordination evidence:** distinct teammate home bands emerge (coverage
  overlap 0.17–0.28 in eval; 0.01 in the best-partitioned seed, which is also
  the strongest). Overlap *rises* over training as chasing intensifies —
  specialization means distinct anchors, not disjoint coverage. Self-play
  returns oscillate without converging: the non-stationarity dynamics
  TASK.md predicted, playing out as predicted.
- Figures under `results/plots/`; footage under `results/videos/`.

### v2 Follow-Up (3 Seeds, 1M Steps)

The calibrated v2 environment uses symmetric closest-paddle collision
selection, `paddle_overlap=0.15`, and a 1.15 ball-speed scale. Standard
Team-A-versus-Team-B evaluation gives:

- **vs random:** `0.86 ± 0.07` Team-A win rate, up from v1's `0.68 ± 0.24`.
- **self-play:** `0.44 ± 0.16` / `0.44 ± 0.15`, balanced on average but
  asymmetric by seed.
- **vs reactive tracker:** `0.04 ± 0.08`; the scripted controller remains a
  strong fixed reference.
- **specialization:** Team-A coverage overlap `0.13 ± 0.09`, lower than v1's
  `0.20 ± 0.17`, but overlap alone is not a coordination claim. Evaluation
  now also records home separation, defense-time position/error, action
  distributions, and contact shares per paddle.

The earlier side-swapped diagnostic evaluations must not be aggregated as
extra seeds. `analysis/summarize.py` now excludes them from headline tables
and reports sample standard deviation (`mean ± std`), not max-distance spread.

## v2 Calibration And PPO Gates

The reported `v1` environment is preserved for reproducibility. New research
runs should use `v2`, which resolves simultaneous overlapping-paddle contacts
by selecting the paddle closest to the ball rather than giving A1/B1 fixed
iteration-order priority. Do not mix v1 and v2 result records.

First calibrate v2 using only scripted policies. This is not PPO training:

```bash
uv run python experiments/calibrate_env.py \
  --overlaps 0.15,0.25,0.35,0.45 \
  --speed-scales 1.0,1.15 \
  --episodes 60 \
  --out results/calibration_v2.json
```

Choose one candidate using the predeclared criteria in `TASK.md`: both
teammates reach the shared region, neither covers the arena alone, there are
no coverage holes or collision-order privilege, and the reactive scripted
tracker is strong but not effectively perfect. Freeze the selected overlap and
speed scale before PPO training.

**Frozen v2 profile:** calibration selected `paddle_overlap=0.15` and
`ball_speed_scale=1.15`. Reactive-vs-reactive play was decisive in 68% of 60
episodes (32% draws; 0.63 vs 0.67 points/episode), while predictive trackers
still drew every game and reactive teams beat random 100%. `--env-version v2`
uses this profile automatically; the explicit flags below document it and make
the run configuration unambiguous.

Then run stationary 1v1 control gates before any long 2v2 job. These keep PPO
independent and point-only, but freeze Team B to diagnose sparse credit and
non-stationarity separately:

```bash
# Run each command for seeds 0, 1, 2.
uv run python experiments/exp_1v1_baseline.py \
  --env-version v2 --paddle-overlap 0.15 \
  --ball-speed-scale 1.15 \
  --train-team A --opponent random \
  --gamma 0.995 --lam 0.99 --ent 0.01 --ent-final 0.001 \
  --timesteps 300000 --seed 0 --run-name v2_1v1_random_seed0

uv run python experiments/exp_1v1_baseline.py \
  --env-version v2 --paddle-overlap 0.15 \
  --ball-speed-scale 1.15 \
  --train-team A --opponent reactive \
  --gamma 0.995 --lam 0.99 --ent 0.01 --ent-final 0.001 \
  --timesteps 300000 --seed 0 --run-name v2_1v1_reactive_seed0
```

Fixed-opponent checkpoints contain only the learned team. Evaluate them
against the same scripted side:

```bash
uv run python evaluation/evaluate.py --mode 1v1 --env-version v2 \
  --paddle-overlap 0.15 \
  --ball-speed-scale 1.15 \
  --weights results/models/v2_1v1_reactive_seed0/final.pt \
  --opponent reactive --opponent-team B --episodes 60
```

Select the best checkpoint rather than assuming final self-play is best:

```bash
uv run python evaluation/select_checkpoint.py \
  --run-dir results/models/<RUN> --mode 2v2 --env-version v2 \
  --paddle-overlap 0.15 --ball-speed-scale 1.15 \
  --opponent reactive --episodes 30
```

Only after the stationary gate clearly beats random and narrows the tracker
gap should you run the fresh three-seed v2 main condition:

```bash
uv run python experiments/exp_2v2_main.py \
  --env-version v2 --paddle-overlap 0.15 \
  --ball-speed-scale 1.15 \
  --gamma 0.995 --lam 0.99 --ent 0.01 --ent-final 0.001 \
  --timesteps 1000000 --seed 0 --run-name v2_2v2_seed0_1M
```

Repeat the final command for seeds `1` and `2`. Training now logs exact
environment transitions, entropy coefficient, approximate KL, clipping
fraction, explained variance, reward-event rate, and truncation count.

If the point-only stationary gate remains poor, run a separate **diagnostic**
ablation, never a replacement for the primary result. It gives both teammates
the same small reward when either returns the ball:

```bash
uv run python experiments/exp_1v1_baseline.py \
  --env-version v2 --paddle-overlap 0.15 \
  --ball-speed-scale 1.15 \
  --train-team A --opponent reactive \
  --reward-mode shared_hit --hit-reward 0.05 \
  --gamma 0.995 --lam 0.99 --timesteps 300000 \
  --seed 0 --run-name v2_diag_shared_hit_seed0
```

If this diagnostic learns reliable tracking while point-only PPO does not,
report sparse temporal credit assignment as the bottleneck; do not present the
shaped condition as the primary TASK.md result.

For side-balanced scripted evaluation of a full-team run:

```bash
uv run python evaluation/eval_suite.py \
  --runs v2_2v2_seed0_1M v2_2v2_seed1_1M v2_2v2_seed2_1M --episodes 60 --paired \
  --env-version v2 --paddle-overlap 0.15 --ball-speed-scale 1.15

Cross-play tests whether a learned Team A from one seed generalizes against a
learned Team B from another rather than only its co-trained counterpart:

```bash
uv run python evaluation/crossplay.py \
  --runs v2_2v2_seed0_1M v2_2v2_seed1_1M v2_2v2_seed2_1M \
  --env-version v2 --episodes 60 \
  --out results/crossplay_v2_2v2_1M.json
```

For the frozen-opponent condition, train one team against the fixed
policies from a full-team checkpoint. This preserves independent PPO while
separating skill improvement from simultaneous-opponent drift:

```bash
uv run python experiments/exp_2v2_main.py \
  --env-version v2 --train-team A --opponent checkpoint \
  --opponent-weights results/models/v2_2v2_seed2_1M/final.pt \
  --gamma 0.995 --lam 0.99 --ent 0.01 --ent-final 0.001 \
  --timesteps 1000000 --seed 0 --run-name v2_frozen_B_seed0
```

Evaluate the frozen-opponent run (learner A vs frozen B):

```bash
uv run python evaluation/evaluate.py \
  --mode 2v2 --env-version v2 \
  --weights results/models/v2_frozen_B_seed0/final.pt \
  --opponent checkpoint --opponent-weights results/models/v2_2v2_seed2_1M/final.pt \
  --episodes 60
```
```

## Layout

- `environment/` — `config.py` (all tuning), `physics.py` (pure helpers),
  `pong_env.py` (dict multi-agent API + pygame-ce rendering)
- `agents/` — independent-PPO learners (`multi_agent_ppo.py`) + actor-critic
  policies (`policies.py`); torch CPU-only
- `experiments/` — `train.py` core loop, `exp_1v1_baseline.py`, `exp_2v2_main.py`
- `evaluation/` — greedy eval, win rate / rally / specialization metrics,
  cross-play vs scripted teams, trajectory dumps, uniform eval suite
- `analysis/` — learning curves, coverage heatmaps, checkpoint overlap
  sweeps, cross-seed summary tables
- `baselines/` — random + ball-tracking heuristic agents
- `tests/` — physics, env, PPO, evaluation, baseline, and analysis suites
- `report/` — final write-up (research question → evidence, honestly)
- `results/` — plots / models / videos / eval records / summary table
