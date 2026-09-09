# marl-pong — 2v2 Pong game (RL deferred)

Game-first build of `TASK.md`. RL training lands later; the env already
exposes an RL-ready dict API.

## Quickstart

```bash
uv sync
uv run pytest
uv run python demo.py            # 2v2 heuristic demo, headed
uv run python demo.py --headless # no window, fast smoke run
uv run python demo.py --human    # control A1 with W/S
```

## Demo options

```bash
uv run python demo.py --mode 1v1              # classic 1v1, full-height paddles
uv run python demo.py --mode 1v1 --human      # play 1v1 yourself
uv run python demo.py --human --opponent random  # beatable opponents
uv run python demo.py --human --points 2      # short match
uv run python demo.py --help                  # all flags
```

In `--human` 2v2 mode you play A1 (upper region); in 1v1 you get the full
height. ESC or closing the window quits.

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
uv run python demo.py --weights results/models/2v2_seed0_1M/final.pt
# record gameplay footage (headless works too)
uv run python demo.py --weights results/models/2v2_seed0_1M/final.pt \
  --episodes 1 --record results/videos/demo.mp4
# uniform eval pass: 60 episodes vs self / heuristic / random per run
uv run python evaluation/eval_suite.py --runs 2v2_seed0_1M 1v1_seed0 --episodes 60
# figures + cross-seed summary
uv run python analysis/plots.py --curves 2v2=results/logs/2v2_seed0_1M.csv
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
