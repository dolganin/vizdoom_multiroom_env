"""Generate MultiRoomColorGoal scenarios (command line front end of map_generator).

Examples::

    # 6 rooms, bigger rooms (7 x 7 tiles per cell, 64 units per tile)
    python -m multiroom_color_goal.generate --rooms 6 --cell-tiles 7 --tile 64 \\
        --name r6_big --out-dir my_scenarios --print-maps

    # 12 rooms, default room size, 8 train + 4 eval layouts, own seeds
    python -m multiroom_color_goal.generate --rooms 12 --train-layouts 8 --eval-layouts 4 \\
        --seed-base 500000 --out-dir my_scenarios

    # rebuild a shipped preset / all of them (byte-identical WAD files)
    python -m multiroom_color_goal.generate --preset 8 --out-dir rebuilt
    python -m multiroom_color_goal.generate --all-presets --out-dir rebuilt

Every scenario consists of three files ``multiroom_color_goal_<name>.wad``,
``.json`` and ``.cfg`` in ``--out-dir``.  Load it with
``MultiRoomColorGoalEnv(scenario="<name>", scenario_dir="<out-dir>")``.
Only the Python standard library is needed to generate.
"""
from __future__ import annotations

import argparse
import sys

from .map_generator import (
    DEFAULT_GEOMETRY, MAX_CELL_TILES, MAX_ROOMS, MAX_TILE, MIN_CELL_TILES, MIN_TILE, SCALES,
    Geometry, ascii_map, build_scale, build_scenario, default_scenario_name, derive_params,
    scenario_file_stem,
)

DEFAULT_OUT_DIR = "multiroom_scenarios"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m multiroom_color_goal.generate",
        description="Generate MultiRoomColorGoal scenarios with any number and size of rooms.",
        epilog="Parameters that are not given are derived from --rooms following the trend of "
               "the presets (see multiroom_color_goal.map_generator.derive_params).")
    what = ap.add_argument_group("what to build (one of)")
    what.add_argument("--rooms", type=int, metavar="N",
                      help=f"number of candidate rooms, 2..{MAX_ROOMS}")
    what.add_argument("--preset", type=int, nargs="+", choices=sorted(SCALES), metavar="N",
                      help=f"rebuild shipped preset(s) exactly as shipped: {sorted(SCALES)}")
    what.add_argument("--all-presets", action="store_true", help="rebuild all five presets")

    size = ap.add_argument_group("size of the rooms (with --rooms)")
    size.add_argument("--cell-tiles", type=int, default=DEFAULT_GEOMETRY.cell_tiles, metavar="K",
                      help=f"tiles per cell side, {MIN_CELL_TILES}..{MAX_CELL_TILES}; a room is "
                           f"(K-1) x (K-1) tiles (default {DEFAULT_GEOMETRY.cell_tiles})")
    size.add_argument("--tile", type=int, default=DEFAULT_GEOMETRY.tile, metavar="U",
                      help=f"map units per tile, {MIN_TILE}..{MAX_TILE} "
                           f"(default {DEFAULT_GEOMETRY.tile}; the player is 32 units wide)")
    size.add_argument("--link-w", type=int, default=DEFAULT_GEOMETRY.link_w, metavar="T",
                      help=f"width of the openings between cells in tiles "
                           f"(default {DEFAULT_GEOMETRY.link_w})")

    out = ap.add_argument_group("output")
    out.add_argument("--name", help="scenario name; files are multiroom_color_goal_<name>.* "
                                    "(default: r<N>, or r<N>_c<K>_t<U> for a non-default size)")
    out.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                     help=f"output directory, created if missing (default ./{DEFAULT_OUT_DIR})")
    out.add_argument("--print-maps", action="store_true", help="print an ASCII picture per layout")
    out.add_argument("--show-params", action="store_true",
                     help="print the derived parameters and exit without writing files")
    out.add_argument("--quiet", action="store_true")

    lay = ap.add_argument_group("layouts (with --rooms)")
    lay.add_argument("--train-layouts", type=int, default=4, metavar="K",
                     help="number of train layouts (default 4)")
    lay.add_argument("--eval-layouts", type=int, default=2, metavar="K",
                     help="number of held-out eval layouts (default 2)")
    lay.add_argument("--seed-base", type=int, default=None, metavar="B",
                     help="layout seeds are B+1.. (train) and B+901.. (eval); default 1000*N")

    par = ap.add_argument_group("generator parameters (with --rooms; default: derived from N)")
    par.add_argument("--n-corridor", type=int, help="corridor cells after the airlock")
    par.add_argument("--lead-in", type=int,
                     help="corridor cells before the first room may be attached (memory distance)")
    par.add_argument("--branch-prob", type=float, help="probability of a fork per corridor cell")
    par.add_argument("--dead-ends", type=int, dest="n_dead_ends", help="empty corridor stubs")
    par.add_argument("--distractors", type=int, dest="n_distractors",
                     help="neutral (grey) leaf rooms")
    par.add_argument("--max-episode-steps", type=int, help="step limit stored in the scenario")
    par.add_argument("--heldout-targets", type=int,
                     help="target identities kept out of the train split")
    par.add_argument("--wall-colors", type=int, help="number of wall colours (1..8)")
    par.add_argument("--floor-colors", type=int,
                     help="number of floor colours (2..8) for two-tone rooms; 0 = solid rooms")
    return ap


def _print_maps(meta: dict) -> None:
    for m in meta["layouts"]:
        print("layout %d (%s, seed %d)" % (m["layout_id"], m["split"], m["map_seed"]))
        print(ascii_map(m))
        print()


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    presets = sorted(SCALES) if args.all_presets else sorted(set(args.preset or []))
    if presets and args.rooms is not None:
        ap.error("--rooms cannot be combined with --preset / --all-presets")
    if not presets and args.rooms is None:
        ap.error("give --rooms N, --preset N or --all-presets")
    try:
        if presets:
            custom = [k for k in ("cell_tiles", "tile", "link_w") if
                      getattr(args, k) != getattr(DEFAULT_GEOMETRY, k)]
            custom += [k for k in ("name", "n_corridor", "lead_in", "branch_prob", "n_dead_ends",
                                   "n_distractors", "max_episode_steps", "heldout_targets",
                                   "wall_colors", "floor_colors", "seed_base")
                       if getattr(args, k) is not None]
            if custom or (args.train_layouts, args.eval_layouts) != (4, 2):
                ap.error("presets are fixed; use --rooms N to build a variant with other "
                         "parameters")
            for s in presets:
                if args.show_params:
                    print("preset r%d:" % s, DEFAULT_GEOMETRY, SCALES[s])
                    continue
                meta = build_scale(s, args.out_dir, verbose=not args.quiet)
                if args.print_maps:
                    _print_maps(meta)
            return 0

        geom = Geometry(tile=args.tile, cell_tiles=args.cell_tiles, link_w=args.link_w)
        overrides = {}
        for key in ("n_corridor", "lead_in", "branch_prob", "n_dead_ends", "n_distractors",
                    "max_episode_steps", "heldout_targets", "wall_colors"):
            if getattr(args, key) is not None:
                overrides[key] = getattr(args, key)
        if args.floor_colors is not None:
            overrides["floor_colors"] = args.floor_colors or None
        params = derive_params(args.rooms, geom, n_train_layouts=args.train_layouts,
                               n_eval_layouts=args.eval_layouts, seed_base=args.seed_base,
                               **overrides)
        name = args.name if args.name is not None else default_scenario_name(args.rooms, geom)
        stem = scenario_file_stem(name)
        if args.show_params:
            print("scenario:", stem)
            print("geometry:", geom)
            for k, v in params.items():
                print("  %-18s %s" % (k, v))
            return 0
        meta = build_scenario(params, args.out_dir, name=name, geom=geom, verbose=not args.quiet)
        if args.print_maps:
            _print_maps(meta)
        if not args.quiet:
            print("step limit %d; load with MultiRoomColorGoalEnv(scenario=%r, scenario_dir=%r)"
                  % (params["max_episode_steps"], meta["name"], args.out_dir))
    except (ValueError, RuntimeError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
