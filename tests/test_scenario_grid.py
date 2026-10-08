"""VizDoom verification of generated scenarios over a grid of room counts and sizes.

For every (rooms, cell_tiles, tile) of the grid a scenario with 2 train + 1 eval
layouts is generated into a temporary directory and checked end to end (see
``helpers.check_scenario``):

* the map loads and the first observation has the documented shape;
* the cue door seals (frames + position) and the generator reports no line of
  sight from the cue zone to a candidate room;
* the scripted oracle reaches the target for every (layout, target) within
  the step limit;
* memoryless policies succeed at 1 / n_rooms.

One test function per (cell_tiles, tile) so that the runner can spread the
grid over two processes.  ``MRCG_GRID_ROOMS="2,6"`` restricts the room counts
(for a quick local run); the default is the full grid.
"""
from __future__ import annotations

import os

import helpers
from helpers import TempDir

from multiroom_color_goal.map_generator import Geometry, build_scenario, derive_params

GRID_ROOMS = (2, 3, 5, 6, 12, 24)
GRID_CELL_TILES = (4, 5, 7)
GRID_TILES = (48, 64, 96)
# corners of the valid range and non-default opening widths: (rooms, cell_tiles, tile, link_w)
EXTREMES = (
    (4, 4, 32, 2),      # smallest cell, smallest tile
    (5, 9, 32, 2),      # largest cell, smallest tile
    (2, 4, 128, 3),     # smallest cell, largest tile, opening = whole wall
    (4, 9, 128, 2),     # largest cell, largest tile
    (3, 5, 48, 1),      # narrowest allowed opening (48 units)
    (5, 6, 64, 1),      # one-tile opening, odd content (opening half a tile off centre)
    (5, 9, 32, 8),      # opening as wide as the room
)


def _rooms():
    env = os.environ.get("MRCG_GRID_ROOMS")
    return tuple(int(v) for v in env.split(",")) if env else GRID_ROOMS


def _check(rooms, cell_tiles, tile, link_w=2):
    geom = Geometry(tile, cell_tiles, link_w)
    params = derive_params(rooms, geom, n_train_layouts=2, n_eval_layouts=1)
    with TempDir() as d:
        meta = build_scenario(params, d, "grid", geom, verbose=False)
        assert meta["params"]["max_episode_steps"] == params["max_episode_steps"]
        res = helpers.check_scenario(d, "grid")
    print(helpers.format_row(res) + ("" if link_w == 2 else " [link_w %d]" % link_w))
    return res


def _make(cell_tiles, tile):
    def test():
        for rooms in _rooms():
            _check(rooms, cell_tiles, tile)
    test.__name__ = "test_grid_cell_tiles_%d_tile_%d" % (cell_tiles, tile)
    test.__doc__ = "rooms %s with cell_tiles=%d, tile=%d" % (GRID_ROOMS, cell_tiles, tile)
    return test


for _ct in GRID_CELL_TILES:
    for _t in GRID_TILES:
        _fn = _make(_ct, _t)
        globals()[_fn.__name__] = _fn
del _ct, _t, _fn


def test_extreme_geometries():
    for rooms, cell_tiles, tile, link_w in EXTREMES:
        _check(rooms, cell_tiles, tile, link_w)


def test_shipped_presets_with_oracle():
    """The five shipped presets: oracle on every (layout, target), all 6 layouts."""
    scenario_dir = os.path.join(helpers.SRC, "multiroom_color_goal", "scenarios")
    for rooms in (2, 4, 8, 16, 32):
        res = helpers.check_scenario(scenario_dir, "r%d" % rooms)
        assert res["layouts"] == 6 and res["oracle_episodes"] == 6 * rooms
        print(helpers.format_row(res))


def test_large_48_rooms_smallest_cells():
    """Beyond the largest preset: 48 rooms (8 x 6 colours), 4 x 4 tiles of 32 units."""
    _check(48, 4, 32)


def test_large_64_rooms():
    """The maximum: 64 rooms (8 wall x 8 floor colours), default room size."""
    _check(64, 5, 64)
