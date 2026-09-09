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
    ap.add_argument("--mode", choices=["1v1", "2v2"], default="2v2")
    ap.add_argument(
        "--weights",
        default=None,
        help="checkpoint .pt: all agents play greedy (with --human, A1 stays yours).",
    )
    args = ap.parse_args()
    if args.human and args.headless:
        ap.error("--human needs a display; drop --headless to play.")

    import dataclasses

    cfg = dataclasses.replace(Config(), points_to_win=args.points, mode=args.mode)
    env = PongEnv(config=cfg, render_mode=None if args.headless else "human", seed=0)
    agents = {}
    for i, a in enumerate(env.agent_ids):
        lo, hi = env._allowed_range(a)
        home = (lo + hi) / 2.0
        if a.startswith("B") and args.opponent == "random":
            agents[a] = RandomAgent(seed=100 + i)
        else:
            agents[a] = HeuristicAgent(team=a[0], home=home)
    trained = None
    if args.weights:
        import torch

        from agents.multi_agent_ppo import IndependentPPO

        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        if set(state) != set(env.agent_ids):
            ap.error(
                f"--weights is for {sorted(state)}, but --mode {args.mode} needs {env.agent_ids}"
            )
        trained = IndependentPPO(env.agent_ids)
        for a in env.agent_ids:
            trained.nets[a].load_state_dict(state[a])
            trained.nets[a].eval()

    headed = not args.headless
    if headed:
        env.render()  # initialize pygame display before any event polling
    if args.human and headed:
        if args.mode == "1v1":
            print("You play A1 (left side, full height). W=up, S=down, ESC=quit.")
        else:
            lo, hi = cfg.paddle_range(0)
            print(
                f"You play A1 (left side, upper region y in [{lo:.2f}, {hi:.2f}]). "
                "W=up, S=down, ESC=quit."
            )
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            human_act = 0
            if headed:
                import pygame

                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        env.close()
                        return
                    if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                        env.close()
                        return
                if args.human:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_w] and not keys[pygame.K_s]:
                        human_act = 1
                    elif keys[pygame.K_s] and not keys[pygame.K_w]:
                        human_act = 2
            actions = {}
            for a, o in obs.items():
                if args.human and a == "A1" and headed:
                    actions[a] = human_act
                elif trained is not None:
                    actions[a] = trained.nets[a].greedy(o)
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
