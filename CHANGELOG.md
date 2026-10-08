# Changelog

All notable changes to marl-pong are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
ad-hoc pre-1.0 while the game is built ahead of the RL phase in `TASK.md`.

## [Unreleased]

### Cleanup — single reported result (`s1_5M`)

#### Changed
- Reported result is now `s1_5M`: 5M-step self-play, seed 0, rollout 4096 /
  6 epochs / minibatch 512 / λ 0.99. Evaluated over 60 episodes it wins 1.00
  vs random, 0.63 vs the reactive tracker and 0.00 vs the predictive tracker
  (0.87 draws). Report, deck, README and HOW_TO_RUN are rewritten around it.
- The calibrated profile (overlap 0.15, ball speed ×1.15, closest-paddle
  collisions, point-only reward) is now the only environment and the
  `Config()` default. `PPOConfig` and the training CLI default to the reported
  hyperparameters (5M steps, 60-episode final eval).
- Rollouts sample through a pure-NumPy snapshot of each policy
  (`NumpySampler`), and next-state values are batched per rollout. The
  algorithm is unchanged; this is a large speed-up for single-observation
  sampling.
- `demo.py`: `--opponent self|heuristic|random` works with `--weights`, so a
  trained Team A can face a scripted Team B. Added `--seed`.
- `calibrate_env.py` scales from `BASE_BALL_SPEEDS` (reproduces
  `results/calibration_v2.json` exactly) and writes `calibration_rerun.json`
  by default.

#### Removed
- v1 environment path (sequential-ID collision priority, uncalibrated overlap
  0.4) and all v1 artifacts.
- 1v1 mode (`exp_1v1_baseline.py`, `--mode`) and its results.
- Frozen/scripted-opponent training (`agents/frozen_policy.py`,
  `--train-team`, `--opponent`, `--opponent-weights`), and checkpoint
  opponents in evaluation.
- `shared_hit` / `shaped` reward modes, legacy shared-torso checkpoint loading,
  and `--env-version` / `--paddle-overlap` / `--ball-speed-scale` flags.
- Superseded runs: v2 1M 3-seed runs, 150k gate and shared-hit diagnostics,
  the aborted frozen-opponent run, and the hyperparameter sweep (`s1_*`,
  `big3M_*`). Their numbers are summarized in `report/report.md` §4.4; the
  artifacts are kept in an off-repo archive.
- v1 plots and video; replaced by `s1_5M` figures and footage.

### Added
- `HOW_TO_RUN.md`: manual command reference for demos, dual-paddle controls,
  trained playback, evaluation, calibration, analysis, and PPO conditions.
- Versioned v2 environment path: closest-paddle collision resolution,
  per-paddle contact telemetry, scripted-only calibration harness, and a
  range-aware / reactive / predictive / random baseline ladder.
- Stationary-opponent PPO support (`--train-team`, `--opponent`), exact
  transition budgets, entropy annealing, PPO optimizer diagnostics,
  side-swapped scripted evaluation, and checkpoint selection utility.
- Frozen v2 calibration profile from `results/calibration_v2.json`:
  `paddle_overlap=0.15`, `ball_speed_scale=1.15`.
- Per-agent evaluation behavior: defense-time y/error, action probabilities,
  contacts, team contact share, and home-position separation. Added cross-play
  matrix and frozen-checkpoint opponent support.
- Presentation deck (`report/marl_pong_presentation.pptx`, built by
  `report/build_deck.js`): findings, figures, and eval numbers for the
  final talk, with speaker notes.
- Completed 3-seed protocol: fresh 2v2 runs at 1M steps (seeds 0-2) and 1v1
  at 300k (seeds 0-2), with a uniform 60-episode eval pass per run vs
  self / heuristic / random (`evaluation/eval_suite.py`).
- Cross-seed summary table (`analysis/summarize.py` → `results/summary.md`).
- Report (`report/report.md`): results, coordination analysis, documented
  spec deviations, honest limitations.
- Demo gameplay recording (`demo.py --record out.mp4`): headless capture via
  a dummy SDL display, `--fps` override for fast capture; imageio +
  imageio-ffmpeg dependencies.

### Changed
- v1 remains the reported historical environment; v2 is opt-in and must be
  calibrated and frozen before new PPO headline runs.
- Summary aggregation now excludes paired side-swapped diagnostics from the
  seed count and reports sample standard deviation instead of max-distance
  spread.
- Learning-curve plots are now NaN-aware and rolling-smoothed (per-iteration
  logs are single-episode samples); `plot_learning_curves` takes a title.
- Seed-0 eval records superseded by the uniform 60-episode pass; the earlier
  warm-start `2v2_seed0_cont1M` run is kept as a documented extra, outside
  the headline protocol (fresh 1M per seed).
- Analysis module (`analysis/plots.py`): learning curves with seed spread,
  paddle-coverage heatmaps from trajectory dumps, teammate coverage-overlap
  sweep across checkpoints; matplotlib added as a dev dependency.
- Reward variance (`var_return_A`) in evaluation output per the TASK.md §8
  metric table.
- Central arena config (`environment/config.py`): normalized sim space,
  paddle-overlap parameter, point threshold, step cap.
- Independent-PPO training (`agents/`, `experiments/`): per-agent
  actor-critic MLPs, synchronized rollouts, GAE + clipped updates on CPU
  (torch pinned to the CPU index); `exp_1v1_baseline.py` / `exp_2v2_main.py`
  entry points, CSV logging, checkpoints, warm-start resume.
- Deterministic evaluation (`evaluation/evaluate.py`): win rate, team
  return, rally length, teammate specialization (mean/std/range + coverage
  overlap), trajectory `.npz` dumps, cross-play vs heuristic/random teams.
- Demo can play trained checkpoints (`--weights`), including human-vs-team.
- Seed-0 trained models + logs + eval records under `results/`.
- Pure physics helpers (`environment/physics.py`): inertial paddles, wall
  bounces, paddle reflection, scoring detection.
- 2v2 `PongEnv` (`environment/pong_env.py`): dict multi-agent API, shared
  team rewards, 8-dim observations (no opponent info), pygame-ce rendering.
- Scripted baselines (`baselines/agents.py`): random + ball-tracking
  heuristic agents.
- Playable demo (`demo.py`): headed/headless scripted matches, human control
  of A1 (W/S), `--opponent`, and `--mode` selection.
- Dynamic ball speed: randomized serve speeds, per-hit rally escalation,
  paddle-motion "smash" transfer.
- Adjustable modes: `1v1` (A1 vs B1, full-height paddles) and `2v2`
  (overlapping partial ranges) via `Config(mode=...)`.
- Test suite (`tests/`): physics, observations, rewards, termination,
  determinism, 1v1, escalation, formation, and PPO loop coverage (25 tests).

### Fixed
- Demo initializes the pygame display before polling events (crashed human
  mode with "video system not initialized"); all headed modes poll window
  events so the window stays closable.
- Human input uses polled keyboard state instead of fragile KEYDOWN/KEYUP
  latching; `--human --headless` is rejected instead of silently ignored.
- Doubles formation: teammates no longer spawn stacked at center — upper /
  lower paddles start in their own quartiles and idle at their range
  midpoint.
