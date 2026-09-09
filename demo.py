"""Headed/headless demo: 4x heuristic paddles, optional human control of A1 (W/S)."""

import argparse

from baselines.agents import HeuristicAgent, RandomAgent
from environment.config import Config
from environment.pong_env import PongEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--human", action="store_true", help="control A1 with W/S keys")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--points", type=int, default=5)
    ap.add_argument(
        "--opponent",
        choices=["heuristic", "random"],
        default="heuristic",
        help="Team B policy (random scores fast; heuristic defends near-perfectly).",
    )
    args = ap.parse_args()

    import dataclasses

    cfg = dataclasses.replace(Config(), points_to_win=args.points)
    env = PongEnv(config=cfg, render_mode=None if args.headless else "human", seed=0)
    agents = {}
    for i, a in enumerate(env.agent_ids):
        if a.startswith("B") and args.opponent == "random":
            agents[a] = RandomAgent(seed=100 + i)
        else:
            agents[a] = HeuristicAgent(team=a[0])

    human_up = human_down = False
    headed = not args.headless
    if headed:
        env.render()  # initialize pygame display before any event polling
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            if headed:
                import pygame

                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        env.close()
                        return
                    if e.type in (pygame.KEYDOWN, pygame.KEYUP):
                        pressed = e.type == pygame.KEYDOWN
                        if e.key == pygame.K_w:
                            human_up = pressed
                        elif e.key == pygame.K_s:
                            human_down = pressed
                        elif e.key == pygame.K_ESCAPE and pressed:
                            env.close()
                            return
            actions = {}
            for a, o in obs.items():
                if args.human and a == "A1" and not args.headless:
                    actions[a] = 1 if human_up else (2 if human_down else 0)
                else:
                    actions[a] = agents[a].act(o)
            obs, _rewards, terminated, truncated, info = env.step(actions)
            done = all(terminated.values()) or all(truncated.values())
            if not args.headless:
                env.render()
        print(f"episode {ep}: scores={info['scores']} steps={info['steps']}")
    env.close()


if __name__ == "__main__":
    main()
