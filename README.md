# marl-pong — 2v2 Pong game (RL deferred)

Game-first build of `TASK.md`. RL training lands later; the env already
exposes an RL-ready dict API.

## Quickstart

```bash
uv sync
uv run pytest
uv run python demo.py            # 4x heuristic demo, headed
uv run python demo.py --headless # no window, fast smoke run
uv run python demo.py --human    # control A1 with W/S
```
