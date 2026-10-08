# marl-pong — Emergent Coordination in 2v2 Pong via Independent PPO

Four independent PPO agents learn 2v2 Pong with a shared team reward: no
communication, no parameter sharing, no centralized critic and no opponent
observations. `TASK.md` is the spec, `report/report.md` the write-up, and
[HOW_TO_RUN.md](HOW_TO_RUN.md) the full command reference.

## Quickstart

```bash
uv sync
uv run pytest
uv run python demo.py --weights results/models/s1_5M/final.pt                       # watch the trained team
uv run python demo.py --weights results/models/s1_5M/final.pt --opponent heuristic  # trained A vs scripted tracker
uv run python demo.py --human-both --weights results/models/s1_5M/final.pt          # you play A1 (W/S) + A2 (Up/Down)
uv run python demo.py --headless                                                    # scripted smoke run
```

## Result

The reported run is **`s1_5M`**: four agents trained in self-play for 5M steps
(seed 0), evaluated greedily over 60 episodes per opponent.

| Team B opponent | Win A | Win B | Draw | Return A |
|---|---|---|---|---|
| random | **1.00** | 0.00 | 0.00 | +4.48 |
| reactive scripted tracker | **0.63** | 0.07 | 0.30 | +0.68 |
| predictive scripted tracker | 0.00 | 0.13 | 0.87 | −0.20 |
| self-play (learned B) | 0.53 | 0.03 | 0.43 | +0.73 |

- **Coordination:** teammates split the field functionally. Each takes every
  return in its own outer band, and A2 takes about 70% of returns in the shared
  middle band, all without assigned roles. Coverage overlap is 0.32.
- **Limits:** the predictive tracker is not beaten. Games against both trackers
  run to the 2000-step cap, so wins there are low-scoring. The result is
  **single-seed**: a 3-seed × 3M run of the same configuration averaged 0.98
  vs random and 0.24 vs the reactive tracker (report §4.4).

Artifacts:

```text
results/models/s1_5M/final.pt          checkpoint (all four agents)
results/models/s1_5M/run_config.json   exact env + PPO config
results/logs/s1_5M.csv                 per-iteration training telemetry
results/eval_s1_5M*.json               evaluation records (self / heuristic / predictive / random)
results/traj_s1_5M.npz                 paddle trajectories (60 self-play episodes)
results/summary.md                     summary table
results/calibration_v2.json            scripted-only environment calibration
results/plots/                         learning curve, coverage heatmaps, overlap over training
results/videos/                        self-play and vs-tracker footage
```

## Reproduce

```bash
# Train (CLI defaults are the reported configuration; took ~80 min on a 12-core CPU)
uv run python experiments/exp_2v2_main.py --seed 0 --run-name s1_5M
# Evaluate
uv run python evaluation/eval_suite.py --runs s1_5M --episodes 60
uv run python evaluation/evaluate.py --weights results/models/s1_5M/final.pt --opponent predictive --episodes 60
# Figures and summary
uv run python analysis/plots.py --curves s1_5M=results/logs/s1_5M.csv --traj results/traj_s1_5M.npz --sweep-run results/models/s1_5M
uv run python analysis/summarize.py
```

`--sweep-run` needs the intermediate `iter_*.pt` checkpoints, which training
writes but git ignores.

## Environment

- Normalized `[-1, 1]` arena with inertial paddles and escalating ball speed.
  Teammates have overlapping vertical ranges; the overlap is `0.15`.
- When teammates overlap, the ball goes to the closest paddle, so neither has a
  fixed priority.
- Point-only shared reward: +1/−1 per point, identical for both teammates.
  Episodes end at 5 points or 2000 steps.
- Overlap and ball speed (×1.15) were frozen from a scripted-only calibration
  (`experiments/calibrate_env.py`) before any PPO training, per TASK.md.

## Layout

- `environment/`: `config.py` (all tuning, frozen profile), `physics.py` (pure
  helpers), `pong_env.py` (dict multi-agent API + pygame-ce rendering)
- `agents/`: independent PPO (`multi_agent_ppo.py`) and actor-critic policies
  with a NumPy rollout sampler (`policies.py`); torch CPU-only
- `experiments/`: `train.py` core loop, `exp_2v2_main.py`, `calibrate_env.py`
- `evaluation/`: greedy evaluation with behavior metrics, uniform eval suite,
  cross-play matrix, checkpoint selection
- `analysis/`: learning curves, coverage heatmaps, overlap sweep, summary table
- `baselines/`: random, range-holding, reactive and predictive scripted agents
- `tests/`: physics, environment, PPO, evaluation, baselines, analysis, demo
- `report/`: write-up and slide deck (`build_deck.js` builds the `.pptx`)
