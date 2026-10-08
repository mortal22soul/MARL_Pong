"""Headed/headless demo: 4x heuristic paddles, optional human control of A1 (W/S)."""

import argparse
import os

from baselines.agents import HeuristicAgent, RandomAgent
from environment.config import Config
from environment.pong_env import PongEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--human", action="store_true", help="control A1 with W/S keys")
    ap.add_argument(
        "--human-both", action="store_true", help="control both A1 (W/S) and A2 (Up/Down)"
    )
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0, help="seed of the first episode")
    ap.add_argument("--points", type=int, default=5)
    ap.add_argument(
        "--opponent",
        choices=["self", "heuristic", "random"],
        default=None,
        help="Team B policy: self (checkpoint), heuristic (reactive ball tracker), or random. "
        "Defaults to self with --weights, else heuristic.",
    )
    ap.add_argument(
        "--weights",
        default=None,
        help="checkpoint .pt: all agents play greedy (with --human, A1 stays yours).",
    )
    ap.add_argument(
        "--record",
        default=None,
        metavar="MP4",
        help="write gameplay footage to this .mp4 (headless uses a dummy display).",
    )
    ap.add_argument("--fps", type=int, default=None, help="override render/capture fps")
    args = ap.parse_args()
    if args.human and args.human_both:
        ap.error("use only one of --human or --human-both")
    if args.human and args.headless:
        ap.error("--human needs a display; drop --headless to play.")
    if args.human_both and args.headless:
        ap.error("--human-both needs a display; drop --headless to play.")
    if args.opponent is None:
        args.opponent = "self" if args.weights else "heuristic"
    if args.opponent == "self" and not args.weights:
        ap.error("--opponent self needs --weights")
    if args.record:
        # Dummy video driver lets pygame build a real surface with no window.
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import dataclasses

    cfg = dataclasses.replace(Config(), points_to_win=args.points)
    if args.fps is not None:
        cfg = dataclasses.replace(cfg, fps=args.fps)
    recording = args.record is not None
    env = PongEnv(
        config=cfg,
        render_mode="human" if (not args.headless or recording) else None,
        seed=0,
    )
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
            ap.error(f"--weights holds {sorted(state)}, but the game needs {env.agent_ids}")
        trained = IndependentPPO(env.agent_ids)
        for a in env.agent_ids:
            trained.nets[a].load_checkpoint(state[a])
            trained.nets[a].eval()

    headed = not args.headless
    if headed:
        env.render()  # initialize pygame display before any event polling
    writer = None
    if recording:
        import imageio

        writer = imageio.get_writer(args.record, fps=cfg.fps)
    if (args.human or args.human_both) and headed:
        lo, hi = cfg.paddle_range(0)
        if args.human_both:
            print(
                f"You play A1 (W/S) and A2 (Up/Down) on left side. "
                f"Regions: A1 y in [{lo:.2f}, {hi:.2f}], A2 y in "
                f"[{cfg.paddle_range(1)[0]:.2f}, {cfg.paddle_range(1)[1]:.2f}]. "
                "ESC=quit."
            )
        else:
            print(
                f"You play A1 (left side, upper region y in [{lo:.2f}, {hi:.2f}]). "
                "W=up, S=down, ESC=quit."
            )
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        while not done:
            human_act_a1 = 0
            human_act_a2 = 0
            if headed:
                import pygame

                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        env.close()
                        return
                    if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                        env.close()
                        return
                if args.human or args.human_both:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_w] and not keys[pygame.K_s]:
                        human_act_a1 = 1
                    elif keys[pygame.K_s] and not keys[pygame.K_w]:
                        human_act_a1 = 2
                    if args.human_both:
                        if keys[pygame.K_UP] and not keys[pygame.K_DOWN]:
                            human_act_a2 = 1
                        elif keys[pygame.K_DOWN] and not keys[pygame.K_UP]:
                            human_act_a2 = 2
            actions = {}
            for a, o in obs.items():
                if args.human and a == "A1" and headed:
                    actions[a] = human_act_a1
                elif args.human_both and a in ("A1", "A2") and headed:
                    actions[a] = human_act_a1 if a == "A1" else human_act_a2
                elif trained is not None and (a.startswith("A") or args.opponent == "self"):
                    actions[a] = trained.nets[a].greedy(o)
                else:
                    actions[a] = agents[a].act(o)
            obs, _rewards, terminated, truncated, info = env.step(actions)
            done = all(terminated.values()) or all(truncated.values())
            if not args.headless or recording:
                env.render()
            if recording:
                import pygame

                frame = pygame.surfarray.array3d(pygame.display.get_surface())
                writer.append_data(frame.transpose(1, 0, 2))
        print(f"episode {ep}: scores={info['scores']} steps={info['steps']}")
    if writer is not None:
        writer.close()
        print(f"footage written to {args.record}")
    env.close()


if __name__ == "__main__":
    main()
