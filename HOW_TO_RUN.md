# How To Run marl-pong

Run every command from the repository root. Commands use `uv`; run `uv sync`
once after cloning or after dependencies change.

## Setup And Checks

```bash
uv sync
uv run pytest
uv run ruff check .
uv run black --check .
```

Show the exact CLI arguments for an entry point:

```bash
uv run python demo.py --help
uv run python experiments/exp_2v2_main.py --help
uv run python evaluation/evaluate.py --help
```

## Play The Game

### Scripted 2v2

```bash
# Four reactive scripted paddles.
uv run python demo.py

# Easier game: Team B is random.
uv run python demo.py --opponent random

# Fast no-window smoke test.
uv run python demo.py --headless --episodes 2 --opponent random
```

### Human Controls

```bash
# Control A1 only: W = up, S = down.
uv run python demo.py --human --opponent random

# Control both Team A paddles in 2v2:
# A1: W/S; A2: Up/Down arrows.
uv run python demo.py --mode 2v2 --human-both --opponent random

# Classic 1v1: control A1 over the full field height.
uv run python demo.py --mode 1v1 --human --opponent random
```

`ESC` or closing the window quits. `--human` and `--human-both` require a
display and cannot be used with `--headless`.

### Watch Or Play Against Trained v2 Agents

Use `--env-version v2` with v2 checkpoints. It loads the frozen calibrated
profile: closest-paddle collisions, overlap `0.15`, and speed scale `1.15`.

```bash
V2=results/models/v2_2v2_seed2_1M/final.pt

# Watch all four learned policies.
uv run python demo.py --mode 2v2 --env-version v2 --weights "$V2"

# You are A1; A2, B1, and B2 use the checkpoint.
uv run python demo.py --mode 2v2 --env-version v2 --human --weights "$V2"

# You control all of Team A; trained B1/B2 are the opponents.
uv run python demo.py --mode 2v2 --env-version v2 --human-both --weights "$V2"

# A short first-to-two match.
uv run python demo.py --mode 2v2 --env-version v2 --human-both --points 2 --weights "$V2"
```

Available full v2 checkpoints:

```text
results/models/v2_2v2_seed0_1M/final.pt
results/models/v2_2v2_seed1_1M/final.pt
results/models/v2_2v2_seed2_1M/final.pt
```

### Record Gameplay

```bash
uv run python demo.py \
  --mode 2v2 --env-version v2 \
  --weights results/models/v2_2v2_seed2_1M/final.pt \
  --episodes 1 --record results/videos/v2_seed2.mp4
```

Recording works without a visible window. Use a higher `--fps` for a faster
capture rate:

```bash
uv run python demo.py --headless --opponent random --episodes 1 \
  --record results/videos/scripted.mp4 --fps 120
```

## Inspect Existing Results

```bash
# Aggregate 3-seed headline table.
uv run python analysis/summarize.py

# Open the generated table and report.
libreoffice results/summary.md
libreoffice report/report.md

# Open the slide deck.
libreoffice --impress report/marl_pong_presentation.pptx

# View the stored v2 cross-play matrix.
libreoffice results/crossplay_v2_2v2_1M.json
```

The main artifacts are:

```text
results/logs/<run>.csv                 Training telemetry
results/models/<run>/final.pt          Final checkpoint
results/models/<run>/run_config.json   Exact environment/PPO config
results/eval_<run>*.json               Evaluation and behavior metrics
results/traj_<run>.npz                 Paddle trajectories
results/plots/                         Curves and heatmaps
results/crossplay_v2_2v2_1M.json       3x3 v2 policy matrix, 60 episodes/cell
```

## Evaluate A Checkpoint

Set a checkpoint once for the following examples:

```bash
RUN=v2_2v2_seed2_1M
WEIGHTS="results/models/$RUN/final.pt"
```

### Self-Play

```bash
uv run python evaluation/evaluate.py \
  --mode 2v2 --env-version v2 \
  --weights "$WEIGHTS" --episodes 60 \
  --out-json "results/eval_${RUN}_manual.json" \
  --out-npz "results/traj_${RUN}_manual.npz"
```

### Team A Against Fixed Baselines

```bash
# Team B is random.
uv run python evaluation/evaluate.py \
  --mode 2v2 --env-version v2 --weights "$WEIGHTS" \
  --opponent random --opponent-team B --episodes 60

# Team B uses the current-y reactive tracker.
uv run python evaluation/evaluate.py \
  --mode 2v2 --env-version v2 --weights "$WEIGHTS" \
  --opponent reactive --opponent-team B --episodes 60

# Team B uses the stronger predicted-intercept tracker.
uv run python evaluation/evaluate.py \
  --mode 2v2 --env-version v2 --weights "$WEIGHTS" \
  --opponent predictive --opponent-team B --episodes 60

# Side-swapped diagnostic: scripted Team A versus trained Team B.
uv run python evaluation/evaluate.py \
  --mode 2v2 --env-version v2 --weights "$WEIGHTS" \
  --opponent reactive --opponent-team A --episodes 60
```

Each evaluation JSON includes global occupancy and functional behavior:

```text
specialization.coverage_overlap
specialization.home_separation
behavior.<agent>.defending_ball_error
behavior.<agent>.action_probs
behavior.<agent>.contacts
behavior.<agent>.team_contact_share
behavior.<agent>.contact_region_share
```

### Uniform Three-Seed Evaluation

```bash
uv run python evaluation/eval_suite.py \
  --runs v2_2v2_seed0_1M v2_2v2_seed1_1M v2_2v2_seed2_1M \
  --episodes 60 --paired --env-version v2
```

`--paired` additionally evaluates the side-swapped scripted-baseline games.
Those are diagnostics; `analysis/summarize.py` excludes them from the standard
three-seed headline table.

### Cross-Play

```bash
uv run python evaluation/crossplay.py \
  --runs v2_2v2_seed0_1M v2_2v2_seed1_1M v2_2v2_seed2_1M \
  --env-version v2 --episodes 60 \
  --out results/crossplay_v2_2v2_1M.json
```

This composes Team A from every seed against Team B from every seed: nine
matchups total. It measures seed-specific co-adaptation, not coordination by
itself.

### Select A Checkpoint

Evaluate numbered checkpoints against a fixed baseline and report the best by
mean Team-A return:

```bash
uv run python evaluation/select_checkpoint.py \
  --run-dir results/models/v2_2v2_seed2_1M \
  --mode 2v2 --env-version v2 \
  --opponent reactive --opponent-team B --episodes 30
```

## Calibrate The v2 Environment

This is scripted-only and does **not** train PPO. It should be run before a
new v2 protocol, not selected based on PPO scores.

```bash
uv run python experiments/calibrate_env.py \
  --overlaps 0.15,0.25,0.35,0.45 \
  --speed-scales 1.0,1.15 \
  --episodes 60 --seed 1000 \
  --out results/calibration_v2.json
```

The committed calibrated v2 profile is:

```text
paddle_overlap = 0.15
ball_speed_scale = 1.15
collision_resolution = closest_paddle
reward_mode = point_only
```

## Train PPO

Training commands can run for minutes or hours. Use a unique `--run-name` so
existing committed artifacts are not overwritten.

### Main 2v2 Independent PPO Self-Play

All four agents learn concurrently. This is the primary MARL condition.

```bash
uv run python experiments/exp_2v2_main.py \
  --env-version v2 --reward-mode point_only \
  --timesteps 1000000 --seed 0 \
  --gamma 0.995 --lam 0.99 \
  --ent 0.01 --ent-final 0.001 \
  --run-name manual_v2_2v2_seed0_1M
```

Repeat with seeds `1` and `2` for the required three-seed condition.

### 1v1 Sanity/Control Condition

```bash
uv run python experiments/exp_1v1_baseline.py \
  --env-version v2 --reward-mode point_only \
  --timesteps 300000 --seed 0 \
  --run-name manual_v2_1v1_seed0
```

1v1 is an implementation/control sanity check, not a matched causal
comparison against 2v2.

### Fixed Scripted Opponent

Train only Team A while Team B remains fixed:

```bash
uv run python experiments/exp_2v2_main.py \
  --env-version v2 --train-team A --opponent reactive \
  --reward-mode point_only --timesteps 300000 --seed 0 \
  --run-name manual_v2_A_vs_reactive_seed0
```

Use `--opponent random` or `--opponent predictive` for the other scripted
baselines.

### Frozen Learned Opponent

Train Team A against Team B loaded from a full checkpoint:

```bash
uv run python experiments/exp_2v2_main.py \
  --env-version v2 --train-team A --opponent checkpoint \
  --opponent-weights results/models/v2_2v2_seed2_1M/final.pt \
  --reward-mode point_only --timesteps 200000 --seed 3 \
  --run-name manual_v2_A_vs_frozen_B2_probe
```

For adaptation rather than fresh learning, optionally warm-start Team A from a
full checkpoint. This restores **model weights only**, not optimizer state:

```bash
uv run python experiments/exp_2v2_main.py \
  --env-version v2 --train-team A --opponent checkpoint \
  --opponent-weights results/models/v2_2v2_seed0_1M/final.pt \
  --warm-start-weights results/models/v2_2v2_seed2_1M/final.pt \
  --reward-mode point_only --ent 0.001 --ent-final 0.0001 \
  --timesteps 200000 --seed 3 \
  --run-name manual_v2_A2_vs_frozen_B0_probe
```

Evaluate a learner-only Team-A checkpoint against its frozen Team B:

```bash
uv run python evaluation/evaluate.py \
  --mode 2v2 --env-version v2 \
  --weights results/models/manual_v2_A_vs_frozen_B2_probe/final.pt \
  --opponent checkpoint \
  --opponent-weights results/models/v2_2v2_seed2_1M/final.pt \
  --opponent-team B --episodes 60
```

### Diagnostic Shared-Hit Reward

This is not the primary TASK.md condition. It is only a sparse-credit
diagnostic, where both teammates receive a small reward whenever either paddle
returns the ball:

```bash
uv run python experiments/exp_1v1_baseline.py \
  --env-version v2 --train-team A --opponent reactive \
  --reward-mode shared_hit --hit-reward 0.05 \
  --timesteps 300000 --seed 0 \
  --run-name manual_v2_shared_hit_diagnostic_seed0
```

## Rebuild Figures

```bash
# One log curve.
uv run python analysis/plots.py \
  --curves v2=results/logs/v2_2v2_seed0_1M.csv \
  --out-dir results/plots

# Heatmaps from an evaluation trajectory file.
uv run python analysis/plots.py \
  --traj results/traj_v2_2v2_seed0_1M.npz \
  --out-dir results/plots

# Checkpoint coverage-overlap sweep.
uv run python analysis/plots.py \
  --sweep-run results/models/v2_2v2_seed0_1M \
  --out-dir results/plots
```

## Troubleshooting

| Problem | Resolution |
|---|---|
| `pygame.error: video system not initialized` | Update the checkout and run the headed demo; the display must initialize before event polling. |
| Human keys do nothing | Click the pygame window so it has keyboard focus. Use `--human-both`, not `--human`, to control both Team-A paddles. |
| Checkpoint architecture error | Use the current code. It loads legacy shared-torso checkpoints compatibly. |
| `--human-both` with `--headless` fails | Expected: human input requires a display. |
| PPO run appears stuck at `nan` return | `nan` means no episode finished during that rollout, not a crash. Inspect `episodes` and the next rows. |
| Frozen fresh Team A loses immediately | Expected against a mature learned opponent; use the warm-start adaptation probe rather than treating it as the main result. |
