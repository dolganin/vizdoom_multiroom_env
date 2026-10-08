"""Random agent, classic API (reset() -> obs; step() -> obs, reward, done, info).

    python examples/01_random_agent.py --rooms 4 --episodes 5
"""
import argparse

import numpy as np

from multiroom_color_goal import ACTION_NAMES, MultiRoomColorGoalEnv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rooms", type=int, default=4, help="preset: 2, 4, 8, 16 or 32")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = MultiRoomColorGoalEnv(args.rooms, split="train", seed=args.seed)
    rng = np.random.default_rng(args.seed)
    print("observation", env.observation_space.shape, env.observation_space.dtype,
          "| actions", ACTION_NAMES, "| step limit", env.max_episode_steps)
    for episode in range(args.episodes):
        obs = env.reset()
        done, ret = False, 0.0
        while not done:
            obs, reward, done, info = env.step(int(rng.integers(env.action_space.n)))
            ret += reward
        outcome = ("success" if info["success"] else
                   "wrong room (%s)" % info["entered_room_color"] if info["wrong_choice"] else
                   "timeout")
        print("episode %d: map %d, target %-13s -> %-22s length %4d, return %+.3f, "
              "left the cue room at step %d" % (
                  episode, info["map_id"], info["target_color"], outcome, info["length"], ret,
                  info["cue_exit_step"]))
    env.close()


if __name__ == "__main__":
    main()
