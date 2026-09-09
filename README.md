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

## Layout

- `environment/` — `config.py` (all tuning), `physics.py` (pure helpers),
  `pong_env.py` (dict multi-agent API + pygame-ce rendering)
- `baselines/` — random + ball-tracking heuristic agents
- `tests/` — physics and env sanity suite
- `results/` — plots / models / videos output dirs
