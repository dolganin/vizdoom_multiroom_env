"""Generate a scenario with another number and size of rooms, then load it.

    python examples/03_custom_scenario.py --rooms 6 --cell-tiles 7 --tile 64 --out-dir my_scenarios

The same from the command line:

    python -m multiroom_color_goal.generate --rooms 6 --cell-tiles 7 --tile 64 \\
        --name r6_big --out-dir my_scenarios
    python -m multiroom_color_goal.inspect r6_big --scenario-dir my_scenarios
"""
import argparse

from multiroom_color_goal import (
    Geometry, MultiRoomColorGoalEnv, ascii_map, build_scenario, derive_params,
    run_scripted_episode)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rooms", type=int, default=6)
    ap.add_argument("--cell-tiles", type=int, default=7)
    ap.add_argument("--tile", type=int, default=64)
    ap.add_argument("--name", default="r6_big")
    ap.add_argument("--out-dir", default="my_scenarios")
    args = ap.parse_args()

    # 1. size of the rooms and parameters derived from the number of rooms;
    #    any parameter can be overridden (here: a longer way to the first room)
    geom = Geometry(tile=args.tile, cell_tiles=args.cell_tiles)
    default = derive_params(args.rooms, geom)
    params = derive_params(args.rooms, geom, n_train_layouts=4, n_eval_layouts=2,
                           lead_in=default["lead_in"] + 1)
    print(geom)
    print(params)

    # 2. write multiroom_color_goal_<name>.{wad,json,cfg}
    meta = build_scenario(params, args.out_dir, name=args.name, geom=geom, verbose=False)
    print("\nlayout 0 (c cue room, d door, a/b airlock, . corridor, r candidate room):")
    print(ascii_map(meta["layouts"][0]))

    # 3. load it by name and check it with the scripted oracle
    env = MultiRoomColorGoalEnv(scenario=args.name, scenario_dir=args.out_dir, split="all")
    print("\nloaded %r: %d rooms, step limit %d, observation %s" % (
        env.scenario, env.n_rooms, env.max_episode_steps, env.observation_space.shape))
    lengths = []
    for layout in meta["splits"]["all"]["layouts"]:
        for target in meta["splits"]["all"]["targets"]:
            info = run_scripted_episode(env, "oracle", layout_id=layout, target_id=target)
            assert info["success"] == 1.0
            lengths.append(info["length"])
    env.close()
    print("oracle: %d / %d episodes solved, longest %d steps of %d allowed" % (
        len(lengths), len(lengths), max(lengths), env.max_episode_steps))


if __name__ == "__main__":
    main()
