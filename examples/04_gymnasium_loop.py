"""Gymnasium API: reset(seed) -> (obs, info); step -> (obs, reward, terminated, truncated, info).

    python examples/04_gymnasium_loop.py
    python examples/04_gymnasium_loop.py --scenario r6_big --scenario-dir my_scenarios
"""
import argparse

import gymnasium as gym

import multiroom_color_goal  # noqa: F401  (registers the MultiRoomColorGoal-* ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rooms", type=int, default=4)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--scenario-dir", default=None)
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    if args.scenario is None:
        env = gym.make("MultiRoomColorGoal-R%d-v0" % args.rooms, split="train")
    else:
        env = gym.make("MultiRoomColorGoal-v0", scenario=args.scenario,
                       scenario_dir=args.scenario_dir, split="train")
    print(env.spec.id, env.observation_space, env.action_space)

    obs, info = env.reset(seed=0)
    for episode in range(args.episodes):
        terminated = truncated = False
        steps, ret = 0, 0.0
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            steps += 1
            ret += reward
        print("episode %d: %s after %d steps, return %+.3f, success %s, target %s" % (
            episode, "terminated" if terminated else "truncated", steps, ret,
            bool(info["success"]), info["target_color"]))
        obs, info = env.reset()
    env.close()


if __name__ == "__main__":
    main()
