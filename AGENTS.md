# AGENTS.md — marl-pong

Course project: emergent coordination in 2v2 Pong via **independent PPO** (four
separate actor-critic MLPs, one shared team reward, no shared critic, no
parameter sharing, no communication). `TASK.md` is the authoritative spec —
read it before changing env design, rewards, or observation space. `README.md`
holds the current honest results snapshot; `CHANGELOG.md` tracks history.

## Commands

```bash
uv sync                                    # install (Python >=3.14, uv-managed)
uv run pytest                              # tests (testpaths=tests)
uv run ruff check . && uv run black .      # lint + format (line-length 100)
uv run python demo.py --headless           # fast smoke run (also --human, --mode 1v1, --weights <ckpt>)
uv run python experiments/exp_2v2_main.py --timesteps 300000 --seed 0
uv run python evaluation/evaluate.py --mode 2v2 --weights results/models/<run>/final.pt
```

Training writes `results/logs/<run>.csv`, `results/models/<run>/` (checkpoints),
`results/eval_<run>.json` + `traj_<run>.npz`. Continue a run with `--init-weights`.

## Architecture rules

- `environment/config.py` — single source of truth for all tuning; frozen
  dataclass, callers customize via `dataclasses.replace`. Change constants
  here, never inline them elsewhere.
- Internal simulation is in normalized coordinates `[-1, 1]` on both axes.
  `physics.py` is pure helpers and **never touches pixels**; only
  `pong_env.py`'s render maps to screen pixels.
- `pong_env.py` exposes a dict multi-agent API: `step({"A1": 0|1|2, ...})`,
  per-agent obs arrays in `environment/` — not a PettingZoo/Gymnasium
  vectorized API.
- `agents/` is torch **CPU-only** (`device="cpu"` throughout); the
  `pytorch-cpu` wheel index is pinned in `pyproject.toml`. Keep it that way.
- `experiments/train.py` inserts the repo root into `sys.path`; scripts are
  run as `uv run python experiments/<script>.py` from the repo root.
- Reward is shared team-level (+1/-1 per point); both teammates get the
  identical signal. Episodes end at `points_to_win` or `max_steps`.

## Design constraints that are deliberate (don't "fix" them)

- **No opponent observations** in the primary experiment — by design, not a
  placeholder (see TASK.md §3.4). Adding them changes the research question.
- **Paddle overlap** (`config.paddle_overlap`) is an experimental parameter
  frozen per the TASK.md protocol; do not tune it to make PPO learn better.
- 3-seed protocol (seeds 0,1,2) for every condition; single-seed numbers are
  not convergence claims.

## Gotchas

- Dependency is `pygame-ce`, **not** `pygame` (plain pygame lacks reliable
  Python 3.14 wheels). Python fallback if wheels fail: 3.12.
- `results/videos/*.{mp4,avi,mkv}` and `results/models/*/iter_*.pt` are
  gitignored (regenerable/large); `final.pt` files and logs are committed.
- `mlruns/` is regenerable MLflow state; don't commit it.
