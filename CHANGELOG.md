# Changelog

All notable changes to marl-pong are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
ad-hoc pre-1.0 while the game is built ahead of the RL phase in `TASK.md`.

## [Unreleased]

### Added
- Central arena config (`environment/config.py`): normalized sim space,
  paddle-overlap parameter, point threshold, step cap.
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
  determinism, 1v1, escalation, and formation coverage (23 tests).

### Fixed
- Demo initializes the pygame display before polling events (crashed human
  mode with "video system not initialized"); all headed modes poll window
  events so the window stays closable.
- Human input uses polled keyboard state instead of fragile KEYDOWN/KEYUP
  latching; `--human --headless` is rejected instead of silently ignored.
- Doubles formation: teammates no longer spawn stacked at center — upper /
  lower paddles start in their own quartiles and idle at their range
  midpoint.
