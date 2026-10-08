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

There is one environment: the calibrated 2v2 profile in
`environment/config.py` (overlap `0.15`, ball speed ×1.15, closest-paddle
collisions, point-only shared reward).

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

# Control both Team A paddles: A1 W/S, A2 Up/Down arrows.
uv run python demo.py --human-both --opponent random
```

`ESC` or closing the window quits. `--human` and `--human-both` require a
display and cannot be used with `--headless`.

### Watch Or Play Against The Trained Team

```bash
W=results/models/s1_5M/final.pt

# Watch all four learned policies (self-play).
uv run python demo.py --weights "$W"

# Trained Team A against a scripted Team B.
uv run python demo.py --weights "$W" --opponent heuristic
uv run python demo.py --weights "$W" --opponent random

# You are A1; A2, B1 and B2 use the checkpoint.
uv run python demo.py --human --weights "$W"

# You control all of Team A against the trained B1/B2; first to two.
uv run python demo.py --human-both --points 2 --weights "$W"
```

With `--weights`, `--opponent` defaults to `self`. Without weights it defaults
to `heuristic`. `--seed N` sets the first episode's seed.

### Record Gameplay

Recording works without a visible window:

```bash
uv run python demo.py --headless --weights results/models/s1_5M/final.pt \
  --seed 2 --episodes 1 --record results/videos/s1_5M_selfplay.mp4
uv run python demo.py --headless --weights results/models/s1_5M/final.pt \
  --opponent heuristic --seed 5 --episodes 1 --record results/videos/s1_5M_vs_heuristic.mp4
```

The committed clips use these seeds. They were picked because points are
scored in them, so treat them as illustrations; the evaluation tables are the
evidence.

## Inspect Existing Results

```text
results/models/s1_5M/final.pt          Checkpoint (A1, A2, B1, B2)
results/models/s1_5M/run_config.json   Exact environment/PPO config
results/logs/s1_5M.csv                 Training telemetry
results/eval_s1_5M*.json               Evaluation and behavior metrics
results/traj_s1_5M.npz                 Paddle trajectories (60 self-play episodes)
results/summary.md                     Summary table
results/calibration_v2.json            Scripted-only calibration sweep
results/plots/                         Curves, heatmaps, overlap over training
results/videos/                        Footage
```

## Evaluate A Checkpoint

```bash
W=results/models/s1_5M/final.pt

# Self-play, with trajectory dump for heatmaps.
uv run python evaluation/evaluate.py --weights "$W" --episodes 60 \
  --out-json results/eval_manual.json --out-npz results/traj_manual.npz

# Team A against a fixed Team B: random | range | heuristic | predictive.
uv run python evaluation/evaluate.py --weights "$W" --opponent heuristic --episodes 60
uv run python evaluation/evaluate.py --weights "$W" --opponent predictive --episodes 60

# Side-swapped: scripted Team A against the trained Team B.
uv run python evaluation/evaluate.py --weights "$W" --opponent heuristic --opponent-team A --episodes 60
```

Each evaluation JSON includes occupancy and functional behavior:

```text
specialization.<team>.coverage_overlap
specialization.<team>.home_separation
behavior.<agent>.defending_ball_error
behavior.<agent>.action_probs
behavior.<agent>.contacts / team_contact_share
behavior.<agent>.contact_region_share
```

### Uniform Evaluation Pass

Runs self-play, heuristic and random evaluations and writes
`results/eval_<run>{,_vs_heuristic,_vs_random}.json`:

```bash
uv run python evaluation/eval_suite.py --runs s1_5M --episodes 60
# --paired also writes side-swapped (_teamA) diagnostics; summarize.py excludes them.
```

### Cross-Play

Composes Team A from every run against Team B from every run:

```bash
uv run python evaluation/crossplay.py --runs <runA> <runB> ... --episodes 60 \
  --out results/crossplay.json
```

### Select A Checkpoint

Evaluates every `iter_*.pt` + `final.pt` in a run directory against a fixed
baseline and reports the best by mean Team-A return:

```bash
uv run python evaluation/select_checkpoint.py --run-dir results/models/<run> \
  --opponent heuristic --episodes 30
```

## Calibrate The Environment

Scripted-only; it does **not** train PPO. Speeds are scaled from the
pre-calibration base speeds in `environment/config.py`. Write to a new file so
the committed record is kept:

```bash
uv run python experiments/calibrate_env.py \
  --overlaps 0.15,0.25,0.35,0.45 --speed-scales 1.0,1.15 \
  --episodes 60 --seed 1000 --out results/calibration_rerun.json
```

## Train PPO

The CLI defaults are the reported configuration: 5M steps, rollout 4096,
6 epochs, minibatch 512, lr 3e-4, γ 0.995, λ 0.99, entropy 0.01 → 0.001.
Use a unique `--run-name` so committed artifacts are not overwritten.

```bash
uv run python experiments/exp_2v2_main.py --seed 0 --run-name my_run_seed0

# Shorter run or other seeds.
uv run python experiments/exp_2v2_main.py --timesteps 1000000 --seed 1 --run-name my_run_seed1

# Continue from weights (optimizer state starts fresh).
uv run python experiments/exp_2v2_main.py --timesteps 1000000 \
  --warm-start-weights results/models/s1_5M/final.pt --run-name s1_5M_plus1M
```

Training writes `results/logs/<run>.csv`, checkpoints every 20 iterations to
`results/models/<run>/` (`iter_*.pt` are git-ignored), and finishes with a
60-episode self-play evaluation (`results/eval_<run>.json`,
`results/traj_<run>.npz`).

## Rebuild Figures And Summary

```bash
uv run python analysis/plots.py --curves s1_5M=results/logs/s1_5M.csv \
  --title "Self-play training curve (s1_5M, seed 0)"
uv run python analysis/plots.py --traj results/traj_s1_5M.npz
uv run python analysis/plots.py --sweep-run results/models/s1_5M   # needs iter_*.pt
uv run python analysis/summarize.py
```

## Rebuild The Slide Deck

```bash
npm install --prefix /tmp/deckdeps pptxgenjs
NODE_PATH=/tmp/deckdeps/node_modules node report/build_deck.js
```

## Troubleshooting

| Problem | Resolution |
|---|---|
| Human keys do nothing | Click the pygame window so it has keyboard focus. Use `--human-both` to control both Team-A paddles. |
| `--human` with `--headless` fails | Expected: human input requires a display. |
| `--opponent self needs --weights` | Self-play needs a checkpoint; use `--opponent heuristic` or `random` otherwise. |
| PPO log shows `nan` return | No episode finished during that rollout; it is not a crash. |
