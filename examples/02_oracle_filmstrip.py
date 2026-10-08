"""Record one episode of the scripted oracle as a PNG film strip (and optionally a GIF).

    python examples/02_oracle_filmstrip.py --rooms 8 --out oracle_r8.png
    python examples/02_oracle_filmstrip.py --rooms 8 --out oracle_r8.png --gif oracle_r8.gif

The oracle is a privileged controller (it reads the map and the target from
``env.privileged_state()``); it only shows what a perfect episode looks like.
"""
import argparse
import os

import cv2
import numpy as np

from multiroom_color_goal import MultiRoomColorGoalEnv, run_scripted_episode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rooms", type=int, default=8)
    ap.add_argument("--scenario", default=None, help="name of a generated scenario")
    ap.add_argument("--scenario-dir", default=None)
    ap.add_argument("--layout", type=int, default=0)
    ap.add_argument("--target", type=int, default=1)
    ap.add_argument("--frames", type=int, default=8, help="frames in the strip")
    ap.add_argument("--out", default=os.path.join("examples", "output", "oracle_filmstrip.png"))
    ap.add_argument("--gif", default=None, help="also write an animated GIF (needs Pillow)")
    args = ap.parse_args()

    env = MultiRoomColorGoalEnv(args.rooms, scenario=args.scenario,
                                scenario_dir=args.scenario_dir, split="all")
    frames, notes = [], []

    def on_step(e, obs, info, k):
        frames.append(e.render().copy())                 # raw 160x120 RGB frame
        notes.append("cue" if info is None or info["cue_phase"] else "sealed")

    info = run_scripted_episode(env, "oracle", layout_id=args.layout, target_id=args.target,
                                on_step=on_step)
    env.close()
    print("target %s, success %s, %d steps (limit %d), cue door triggered at step %d" % (
        info["target_color"], bool(info["success"]), info["length"], env.max_episode_steps,
        info["cue_exit_step"]))

    idx = np.linspace(0, len(frames) - 1, args.frames).round().astype(int)
    panels = []
    for i in idx:
        img = cv2.resize(frames[i], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, "t=%d %s" % (i, notes[i]), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(img)
    strip = np.concatenate(panels, axis=1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print("film strip ->", args.out)
    if args.gif:
        from PIL import Image
        images = [Image.fromarray(f) for f in frames]
        images[0].save(args.gif, save_all=True, append_images=images[1:], duration=110, loop=0)
        print("gif ->", args.gif)


if __name__ == "__main__":
    main()
