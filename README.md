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
uv run python experiments/exp_2v2_main.py --timesteps 300000 --seed 0
# 1v1 sanity baseline
uv run python experiments/exp_1v1_baseline.py --timesteps 300000 --seed 0
# continue a run (warm start), e.g. to 1M steps total
uv run python experiments/exp_2v2_main.py --timesteps 700000 --seed 1 \
  --init-weights results/models/2v2_seed0/final.pt --run-name 2v2_seed0_cont1M
```

Training logs per-iteration stats to `results/logs/<run>.csv`, checkpoints to
`results/models/<run>/`, and auto-runs greedy eval
(`results/eval_<run>.json` + `traj_<run>.npz`).

```bash
# watch trained teams play each other
uv run python demo.py --weights results/models/2v2_seed0_cont1M/final.pt
# play against a trained team yourself (you are A1)
uv run python demo.py --weights results/models/2v2_seed0_cont1M/final.pt --human
# evaluate a checkpoint, incl. cross-play vs scripted teams
uv run python evaluation/evaluate.py --mode 2v2 \
  --weights results/models/2v2_seed0_cont1M/final.pt --episodes 20
uv run python evaluation/evaluate.py --mode 2v2 \
  --weights results/models/2v2_seed0_cont1M/final.pt --opponent random
```

Seed-0 results so far (honest snapshot, not a convergence claim): 1v1 beats
random 65% (+1.2); 2v2 at 1M steps beats random 70% (+1.4) with ~1750-step
rallies, but loses to perfect scripted defense and shows team asymmetry
(B stronger) — the non-stationarity/credit-assignment dynamics from TASK.md
playing out as predicted.

## Layout

- `environment/` — `config.py` (all tuning), `physics.py` (pure helpers),
  `pong_env.py` (dict multi-agent API + pygame-ce rendering)
- `agents/` — independent-PPO learners (`multi_agent_ppo.py`) + actor-critic
  policies (`policies.py`); torch CPU-only
- `experiments/` — `train.py` core loop, `exp_1v1_baseline.py`, `exp_2v2_main.py`
- `evaluation/` — greedy eval, win rate / rally / specialization metrics,
  cross-play vs scripted teams, trajectory dumps
- `baselines/` — random + ball-tracking heuristic agents
- `tests/` — physics and env sanity suite
- `results/` — plots / models / videos output dirs
