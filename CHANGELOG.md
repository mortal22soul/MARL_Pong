# Changelog

All notable changes to marl-pong are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
ad-hoc pre-1.0 while the game is built ahead of the RL phase in `TASK.md`.

## [Unreleased]

### Added
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
