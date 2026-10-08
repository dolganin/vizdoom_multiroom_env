"""Inspect a MultiRoomColorGoal scenario and verify it with the scripted oracle.

    python -m multiroom_color_goal.inspect 8                          # shipped preset
    python -m multiroom_color_goal.inspect r6_big --scenario-dir my_scenarios
    python -m multiroom_color_goal.inspect my_scenarios/multiroom_color_goal_r6_big.json

Prints the parameters and the ASCII maps and, if vizdoom is importable, walks
the privileged scripted oracle through every (layout, target) pair and reports
the success rate and the episode lengths against the step limit.  Exit status
is 1 if the oracle fails anywhere, so the command can be used as a check after
generating a new scenario.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .map_generator import SCENARIO_PREFIX, Geometry, ascii_map

LEGEND = ("legend: # wall   c cue room   d door strip   a/b airlock (trigger line between a and b)"
          "   . neutral   r candidate room")


def _find(scenario: str, scenario_dir):
    """Standard-library twin of multiroom_env.resolve_scenario (no numpy needed)."""
    if scenario.isdigit():
        scenario = "r" + scenario
    head, tail = os.path.split(scenario)
    for ext in (".json", ".wad", ".cfg"):
        if tail.endswith(ext):
            tail = tail[:-len(ext)]
            break
    stem = tail if tail.startswith(SCENARIO_PREFIX) else SCENARIO_PREFIX + tail
    if head:
        directory = head if scenario_dir is None or os.path.isabs(head) else os.path.join(
            scenario_dir, head)
    else:
        directory = scenario_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "scenarios")
    path = os.path.join(directory, stem + ".json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no scenario file {path!r}")
    return directory, stem, path


def describe(meta: dict, show_maps: bool = True, out=sys.stdout) -> None:
    geom = Geometry.from_meta(meta)
    p = meta["params"]
    w = out.write
    w("scenario   %s  (%s)\n" % (meta.get("name", "r%d" % meta["scale"]), meta["wad"]))
    w("rooms      %d, colour scheme %s" % (meta["scale"], meta["color_scheme"]))
    if meta["color_scheme"] == "two_tone":
        w(" (%d wall x %d floor colours)" % (p["wall_colors"], p["floor_colors"]))
    w("\n")
    w("geometry   tile %d units, cell %d x %d tiles = %d units, room %d units, opening %d tiles "
      "= %d units (cell tiles %d..%d)\n" % (
          geom.tile, geom.cell_tiles, geom.cell_tiles, geom.cell, geom.content * geom.tile,
          geom.link_w, geom.link_w * geom.tile, geom.link_lo, geom.link_lo + geom.link_w - 1))
    w("parameters n_corridor %d, lead_in %d, branch_prob %s, dead ends %d, distractors %d, "
      "step limit %d, held-out targets %d\n" % (
          p["n_corridor"], p["lead_in"], p["branch_prob"], p["n_dead_ends"], p["n_distractors"],
          p["max_episode_steps"], p["heldout_targets"]))
    w("seeds      train %s, eval %s\n" % (p["train_seeds"], p["eval_seeds"]))
    w("identities %s\n" % ", ".join("%d=%s" % (i["id"], i["name"]) for i in meta["identities"]))
    w("splits\n")
    for name, sp in meta["splits"].items():
        w("  %-17s layouts %s targets %s\n" % (name, sp["layouts"], sp["targets"]))
    w("layouts\n")
    for m in meta["layouts"]:
        s = m["stats"]
        w("  %d %-5s seed %-7d grid %dx%d cells, %dx%d units; rooms at depth %d..%d (mean %.1f) "
          "cells; forks %d, dead ends %d, distractors %d; cue line-of-sight violations %d\n" % (
              m["layout_id"], m["split"], m["map_seed"], s["grid_cells"][0], s["grid_cells"][1],
              s["map_units"][0], s["map_units"][1], s["path_cells_min"], s["path_cells_max"],
              s["path_cells_mean"], s["n_forks"], s["n_dead_ends"], s["n_distractors"],
              m["cue_los_violations"] + m.get("cue_portal_los_violations", 0)))
    if show_maps:
        w(LEGEND + "\n")
        for m in meta["layouts"]:
            w("\nlayout %d (%s, seed %d), north is up\n" % (m["layout_id"], m["split"],
                                                            m["map_seed"]))
            w(_cropped_map(m) + "\n")


def _cropped_map(layout_meta: dict) -> str:
    """ASCII map without the unused rows / columns around the maze."""
    rows = ascii_map(layout_meta).split("\n")
    used = [i for i, r in enumerate(rows) if r.strip("#")]
    rows = rows[max(used[0] - 1, 0):used[-1] + 2]
    cols = [i for i in range(len(rows[0])) if any(r[i] != "#" for r in rows)]
    return "\n".join(r[max(cols[0] - 1, 0):cols[-1] + 2] for r in rows)


def run_oracle(scenario_dir: str, stem: str, split: str = "all", out=sys.stdout) -> dict:
    """Oracle over every (layout, target) of ``split``; returns a summary dict."""
    from .multiroom_env import MultiRoomColorGoalEnv
    from .scripted_policies import run_scripted_episode

    env = MultiRoomColorGoalEnv(scenario=stem, scenario_dir=scenario_dir, split=split,
                                track_visibility=False)
    try:
        meta = env.metadata_dict
        sp = meta["splits"][split]
        lengths, failures, seal = [], [], []
        t0 = time.time()
        for layout_id in sp["layouts"]:
            for target_id in sp["targets"]:
                info = run_scripted_episode(env, "oracle", layout_id=layout_id,
                                            target_id=target_id)
                if info["success"] == 1.0:
                    lengths.append(int(info["length"]))
                    seal.append(int(info["cue_exit_step"]))
                else:
                    failures.append((layout_id, target_id, int(info["length"]),
                                     "timeout" if info["timeout"] else "wrong room"))
        n = len(sp["layouts"]) * len(sp["targets"])
        res = {
            "episodes": n, "successes": len(lengths), "failures": failures,
            "success_rate": len(lengths) / float(n) if n else 0.0,
            "length_min": min(lengths) if lengths else None,
            "length_mean": sum(lengths) / float(len(lengths)) if lengths else None,
            "length_max": max(lengths) if lengths else None,
            "max_episode_steps": env.max_episode_steps,
            "observation_shape": tuple(env.observation_space.shape),
            "seconds": time.time() - t0,
        }
    finally:
        env.close()
    w = out.write
    w("\nscripted oracle, split %r: %d layouts x %d targets = %d episodes (%.1f s)\n" % (
        split, len(sp["layouts"]), len(sp["targets"]), n, res["seconds"]))
    w("  observation shape %s\n" % (res["observation_shape"],))
    w("  success rate      %d / %d = %.3f\n" % (res["successes"], n, res["success_rate"]))
    if lengths:
        w("  episode length    min %d, mean %.1f, max %d agent steps; step limit %d "
          "(longest episode uses %.0f%% of it)\n" % (
              res["length_min"], res["length_mean"], res["length_max"],
              res["max_episode_steps"], 100.0 * res["length_max"] / res["max_episode_steps"]))
        w("  cue door trigger  crossed at step %d..%d\n" % (min(seal), max(seal)))
    for layout_id, target_id, length, why in failures[:20]:
        w("  FAILED layout %d target %d after %d steps: %s\n" % (layout_id, target_id, length, why))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m multiroom_color_goal.inspect",
        description="Print the parameters and maps of a scenario and verify it with the "
                    "scripted oracle.")
    ap.add_argument("scenario", help="number of rooms of a preset (8), a scenario name (r6_big) "
                                     "or a path to one of its files")
    ap.add_argument("--scenario-dir", default=None,
                    help="directory with the scenario files (default: the packaged presets)")
    ap.add_argument("--split", default="all", help="split used for the oracle run (default all)")
    ap.add_argument("--no-maps", action="store_true", help="do not print the ASCII maps")
    ap.add_argument("--no-oracle", action="store_true", help="do not run the scripted oracle")
    args = ap.parse_args(argv)
    try:
        directory, stem, path = _find(args.scenario, args.scenario_dir)
    except FileNotFoundError as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    with open(path) as f:
        meta = json.load(f)
    describe(meta, show_maps=not args.no_maps)
    if args.no_oracle:
        return 0
    if args.split not in meta["splits"]:
        print("error: unknown split %r; available: %s" % (args.split, sorted(meta["splits"])),
              file=sys.stderr)
        return 2
    try:
        import vizdoom  # noqa: F401
    except ImportError:
        print("\nvizdoom is not installed: skipping the scripted oracle run")
        return 0
    res = run_oracle(directory, stem, args.split)
    return 0 if res["successes"] == res["episodes"] else 1


if __name__ == "__main__":
    sys.exit(main())
