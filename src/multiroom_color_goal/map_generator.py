"""Seeded map generator for the VizDoom MultiRoomColorGoal family.

The generator is the single source of truth for the WAD files shipped in
``scenarios/`` and for every custom scenario.  It needs only the Python
standard library: the maps are written as UDMF ``TEXTMAP`` lumps (the ZDoom
engine inside VizDoom builds the BSP nodes itself on load) and no ACS compiler
is required, because all episode logic lives in the Python wrapper and the
only in-map mechanism is a native line special (``Door_Close``).

Command line: ``python -m multiroom_color_goal.generate --help``.
``python -m multiroom_color_goal.generate --all-presets --out-dir DIR``
rebuilds, byte for byte, ``multiroom_color_goal_r{2,4,8,16,32}.wad``.

Two things can be scaled:

* the NUMBER of candidate rooms - :func:`derive_params` turns ``n_rooms`` into
  a full parameter set (colour scheme, corridor budget, memory distance, ...);
  the five hand-tuned presets live in :data:`SCALES` and are returned unchanged;
* the SIZE of the rooms - :class:`Geometry` (``tile`` map units per tile,
  ``cell_tiles`` tiles per cell, ``link_w`` tiles per opening).

Geometry model
--------------
* A coarse grid of *cells*; every cell is ``cell_tiles`` x ``cell_tiles`` tiles
  of ``tile`` map units.  Tile column/row 0 of a cell is a solid border, the
  remaining ``content`` x ``content`` tiles (``content = cell_tiles - 1``) are
  the walkable content.  Two neighbouring cells are therefore always separated
  by one solid tile unless a *link* (a ``link_w`` tiles wide opening) is carved
  through the border.
* Cell kinds: ``cue`` (the coloured start room), ``airlock`` (first neutral
  cell), ``corridor`` (neutral), ``room`` (coloured candidate room, always a
  leaf), ``distractor`` (neutral leaf room that looks like a room entrance but
  has no colour and no effect), ``deadend`` (empty neutral stub).
* The cue room and the door strip stand 32 units above the rest of the map
  (one-way drop: the player cannot step up more than 24 units).  Two tiles
  after the drop a walk-over line triggers ``Door_Close`` on the door strip.
  After that the cue room is sealed for the rest of the episode.
* The corridor structure is a tree (chains, forks, dead ends).  The first
  corridor step after the airlock is always followed by a 90 degree bend, and
  every generated layout is checked for the absence of a line of sight from
  the cue zone to any candidate room: ``cue_los_violations`` (point samples,
  the check of the original presets) and ``cue_portal_los_violations`` (strict
  opening-to-opening test, see :func:`portal_los_violations`) must both be 0;
  otherwise the next attempt of the same seed is taken.

Random target colour without ACS
--------------------------------
For every layout the WAD holds one map lump per target identity; the lumps
differ only in the colour of the cue room.  The wrapper draws the target at
reset and loads the matching lump.  The target is thus visible to the policy
only through the pixels of the cue room.
"""
from __future__ import annotations

import json
import math
import os
import random
import struct
import re

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
# Default geometry (the one of the five shipped presets).  These module
# constants are only defaults: the geometry actually used by a scenario is a
# ``Geometry`` object (see below) and is stored in the scenario's .json.
TILE = 64                 # map units per tile
CELL_TILES = 5            # tiles per coarse cell (1 border + 4 content)
CONTENT = CELL_TILES - 1  # content tiles per cell side
LINK_LO = 2               # link occupies cell tiles LINK_LO .. LINK_LO+LINK_W-1
LINK_W = 2
CELL = TILE * CELL_TILES  # 320 map units

FLOOR_LOW = 0
CEIL_LOW = 128
CUE_RAISE = 32            # one-way drop height (> 24 = max step-up)
DOOR_TAG = 1
AIRLOCK_A_TILES = 2       # airlock tile columns before the Door_Close trigger line
DOOR_SPEED = 128          # ZDoom speed units (/8 per tic) -> 16 units/tic
DOOR_CLOSE_TICS = int(math.ceil((CEIL_LOW + CUE_RAISE - CUE_RAISE) / (DOOR_SPEED / 8.0))) + 2

GENERATOR_VERSION = 1

# ---- engine facts the valid geometry ranges are derived from ---------------
PLAYER_RADIUS = 16        # Doom player: 32 x 32 units bounding box
MAX_STEP_UP = 24          # highest ledge the player can climb (CUE_RAISE is above it)
MIN_OPENING = 48          # narrowest allowed passage: player width 32 + 8 units each side
MIN_TILE = 32             # see Geometry: one agent step (~28 units) must not exceed a tile
MAX_TILE = 128
MIN_CELL_TILES = 4        # content >= 3: 2 airlock columns + >= 1 column behind the trigger
MAX_CELL_TILES = 9
MAX_MAP_UNITS = 30000     # ZDoom map coordinates are 16.16 fixed point (|x| < 32768)
MAX_ROOMS = 64            # 8 wall colours x 8 floor colours
MAX_LAYOUTS = 100         # map lump name L<2 digits>T<2 digits>
SCENARIO_PREFIX = "multiroom_color_goal_"
assert CUE_RAISE > MAX_STEP_UP


class Geometry:
    """Size of the rooms: ``tile`` map units per tile, ``cell_tiles`` tiles per cell.

    Every cell (cue room, corridor segment, candidate room) is a square of
    ``cell_tiles`` x ``cell_tiles`` tiles; tile row/column 0 is the solid
    border, so the walkable part is ``content = cell_tiles - 1`` tiles, i.e.
    ``content * tile`` map units, wide.  Openings between cells are ``link_w``
    tiles wide and centred in the content area: they start at cell tile
    ``link_lo = 1 + (content - link_w) // 2`` (exactly centred when
    ``content - link_w`` is even, otherwise half a tile towards the low side).
    The defaults (64, 5, 2) give ``link_lo = 2`` - the geometry of the presets.

    Valid ranges and where they come from:

    * ``cell_tiles`` in [4, 9].  The airlock needs ``AIRLOCK_A_TILES`` = 2 tile
      columns in front of the Door_Close trigger line and at least one column
      behind it, so ``content >= 3``.  The upper bound keeps the pure-Python
      line-of-sight proof (quadratic in ``cell_tiles``) and the WAD size sane.
    * ``1 <= link_w <= content`` - the opening must stay inside the content.
    * ``link_w * tile >= 48`` - the player is 32 units wide (radius 16); 8
      units of clearance on each side are the minimum for a turn-then-walk
      controller to get through without sliding along the wall.
    * ``tile`` in [32, 128], integer.  The player is 32 units wide and one
      agent step (frame_skip 4) at full speed covers ~28 units, so with smaller
      tiles a single step would cross a whole tile (the airlock, 2 tiles before
      the trigger line, would be shorter than the stopping distance).  128 is
      simply the largest tile that was verified (a 9 x 128 cell is 1152 units).
    * the whole map must stay below 30000 map units (ZDoom's 16.16 fixed point
      coordinates); this is checked per scenario because it depends on the
      number of rooms.
    """

    __slots__ = ("tile", "cell_tiles", "content", "link_w", "link_lo", "cell")

    def __init__(self, tile: int = TILE, cell_tiles: int = CELL_TILES, link_w: int = LINK_W):
        for label, v in (("tile", tile), ("cell_tiles", cell_tiles), ("link_w", link_w)):
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{label} must be an integer, got {v!r}")
        if not MIN_CELL_TILES <= cell_tiles <= MAX_CELL_TILES:
            raise ValueError(
                f"cell_tiles must be in [{MIN_CELL_TILES}, {MAX_CELL_TILES}], got {cell_tiles}: "
                f"a cell is 1 border tile + (cell_tiles - 1) content tiles and the airlock needs "
                f"{AIRLOCK_A_TILES} tile columns before the door trigger plus one behind it")
        if not MIN_TILE <= tile <= MAX_TILE:
            raise ValueError(
                f"tile must be in [{MIN_TILE}, {MAX_TILE}] map units, got {tile} "
                f"(the player is {2 * PLAYER_RADIUS} units wide and moves ~28 units per agent "
                f"step; {MAX_TILE} is the largest verified tile)")
        content = cell_tiles - 1
        if not 1 <= link_w <= content:
            raise ValueError(
                f"link_w must be in [1, {content}] for cell_tiles={cell_tiles} (the opening has "
                f"to stay inside the {content} content tiles), got {link_w}")
        if link_w * tile < MIN_OPENING:
            raise ValueError(
                f"opening too narrow: link_w * tile = {link_w * tile} map units, need at least "
                f"{MIN_OPENING} (player width {2 * PLAYER_RADIUS} + 8 units clearance per side); "
                f"increase tile or link_w")
        self.tile = tile
        self.cell_tiles = cell_tiles
        self.content = content
        self.link_w = link_w
        self.link_lo = 1 + (content - link_w) // 2
        self.cell = tile * cell_tiles

    @classmethod
    def from_meta(cls, meta: dict) -> "Geometry":
        """Geometry of a scenario from its side-car .json metadata."""
        g = cls(int(meta["tile"]), int(meta["cell_tiles"]), int(meta["link_w"]))
        if g.link_lo != int(meta["link_lo"]) or g.content != int(meta["content_tiles"]):
            raise ValueError("scenario metadata has an inconsistent geometry "
                             "(link_lo / content_tiles); regenerate the scenario")
        return g

    def as_dict(self) -> dict:
        return {"tile": self.tile, "cell_tiles": self.cell_tiles, "link_w": self.link_w}

    def __eq__(self, other):
        return isinstance(other, Geometry) and self.as_dict() == other.as_dict()

    def __repr__(self):
        return (f"Geometry(tile={self.tile}, cell_tiles={self.cell_tiles}, link_w={self.link_w})"
                f"  # cell {self.cell} units, room {self.content * self.tile} units, "
                f"opening {self.link_w * self.tile} units")

    # ---- coordinates (map units) -------------------------------------------
    def cell_center(self, c):
        """Centre of the walkable content of cell ``c``."""
        ct, half = self.cell_tiles, self.content / 2.0
        return ((c[0] * ct + 1 + half) * self.tile, (c[1] * ct + 1 + half) * self.tile)

    def link_center(self, a, b):
        """Centre of the opening between two adjacent cells."""
        lo = a if (a[0] < b[0] or a[1] < b[1]) else b
        hi = b if lo == a else a
        ct = self.cell_tiles
        mid = (self.link_lo + self.link_w / 2.0)
        if hi[0] != lo[0]:   # east-west link: border column of hi
            return ((hi[0] * ct + 0.5) * self.tile, (lo[1] * ct + mid) * self.tile)
        return ((lo[0] * ct + mid) * self.tile, (hi[1] * ct + 0.5) * self.tile)

    def link_tiles(self, a, b):
        """Tile coordinates carved out of the border between two adjacent cells."""
        lo = a if (a[0] < b[0] or a[1] < b[1]) else b
        hi = b if lo == a else a
        ct = self.cell_tiles
        if hi[0] != lo[0]:
            x = hi[0] * ct
            return [(x, lo[1] * ct + self.link_lo + i) for i in range(self.link_w)]
        y = hi[1] * ct
        return [(lo[0] * ct + self.link_lo + i, y) for i in range(self.link_w)]


DEFAULT_GEOMETRY = Geometry()
assert (DEFAULT_GEOMETRY.link_lo, DEFAULT_GEOMETRY.content, DEFAULT_GEOMETRY.cell) == (
    LINK_LO, CONTENT, CELL)

# name -> Doom palette index (freedoom2 / Doom PLAYPAL, identical ordering).
# RGB values are the palette entries; they are stored in the metadata.
COLORS = [
    ("red", 176, (255, 0, 0)),
    ("green", 116, (91, 191, 79)),
    ("blue", 200, (0, 0, 255)),
    ("yellow", 231, (255, 255, 0)),
    ("magenta", 251, (255, 0, 255)),
    ("orange", 215, (255, 127, 27)),
    ("sky", 195, (143, 143, 255)),
    ("pink", 171, (255, 155, 155)),
]
COLOR_NAMES = [c[0] for c in COLORS]

# neutral (pure grey, zero saturation) palette indices
GREY_WALL = 92      # (159,159,159)
GREY_WALL_DARK = 100  # (111,111,111) brick joints
GREY_FLOOR = 104    # (79,79,79)
GREY_CEIL = 84      # (215,215,215)
GREY_DOOR = 108     # darker
FLAT_WALL = "MRNWALL"
FLAT_FLOOR = "MRNFLOOR"
FLAT_CEIL = "MRNCEIL"


def color_flat(name: str) -> str:
    return ("MRC" + name.upper())[:8]


# Per-scale presets.  Everything here is a generator parameter.
#   n_rooms          candidate (coloured) rooms
#   wall_colors /    identity of a room = (wall colour, floor+ceiling colour);
#   floor_colors     if floor_colors is None the room is solid (floor = wall)
#   n_corridor       neutral corridor cells after the airlock (path budget)
#   lead_in          corridor cells between airlock and the first cell a room
#                    may be attached to (controls the memory distance)
#   branch_prob      probability that a new corridor cell starts a fork
#   n_dead_ends      extra empty corridor stubs
#   n_distractors    neutral leaf rooms
#   max_episode_steps  step limit (agent steps)
#   train_seeds /    map seeds of the train / eval layouts (one layout per seed)
#   eval_seeds
#   heldout_targets  number of target identities never used as target in "train"
# Do not edit the presets: the shipped WAD files are generated from them.
# For other room counts use derive_params().
SCALES = {
    2: dict(n_rooms=2, wall_colors=2, floor_colors=None, n_corridor=3, lead_in=2,
            branch_prob=0.0, n_dead_ends=0, n_distractors=0, max_episode_steps=300,
            train_seeds=[2001, 2002, 2003, 2004], eval_seeds=[2901, 2902],
            heldout_targets=0),
    4: dict(n_rooms=4, wall_colors=4, floor_colors=None, n_corridor=6, lead_in=2,
            branch_prob=0.3, n_dead_ends=1, n_distractors=0, max_episode_steps=500,
            train_seeds=[4001, 4002, 4003, 4004], eval_seeds=[4901, 4902],
            heldout_targets=0),
    8: dict(n_rooms=8, wall_colors=8, floor_colors=None, n_corridor=12, lead_in=3,
            branch_prob=0.35, n_dead_ends=2, n_distractors=1, max_episode_steps=800,
            train_seeds=[8001, 8002, 8003, 8004], eval_seeds=[8901, 8902],
            heldout_targets=2),
    16: dict(n_rooms=16, wall_colors=4, floor_colors=4, n_corridor=22, lead_in=3,
             branch_prob=0.4, n_dead_ends=3, n_distractors=2, max_episode_steps=1400,
             train_seeds=[16001, 16002, 16003, 16004], eval_seeds=[16901, 16902],
             heldout_targets=4),
    32: dict(n_rooms=32, wall_colors=8, floor_colors=4, n_corridor=40, lead_in=4,
             branch_prob=0.45, n_dead_ends=4, n_distractors=4, max_episode_steps=2400,
             train_seeds=[32001, 32002, 32003, 32004], eval_seeds=[32901, 32902],
             heldout_targets=8),
}

DIRS = {"E": (1, 0), "W": (-1, 0), "N": (0, 1), "S": (0, -1)}


# --------------------------------------------------------------------------
# identities (colour codes)
# --------------------------------------------------------------------------
def build_identities(wall_colors: int, floor_colors, n_rooms: int | None = None):
    """Return the list of room identities for a scale.

    An identity is ``{"id", "name", "wall", "floor"}`` where wall/floor are
    colour names.  Solid scheme: floor == wall.  Two-tone scheme: the cross
    product wall x floor (rooms that share a wall or a floor colour are the
    'similar looking rooms').  If ``n_rooms`` is smaller than the number of
    combinations only the first ``n_rooms`` (wall-major order) are used, so
    the last wall colour then appears with fewer floor colours.
    """
    if not 1 <= wall_colors <= len(COLORS):
        raise ValueError(f"wall_colors must be in [1, {len(COLORS)}], got {wall_colors}")
    if floor_colors is not None and not 2 <= floor_colors <= len(COLORS):
        raise ValueError(f"floor_colors must be None (solid rooms) or in [2, {len(COLORS)}], "
                         f"got {floor_colors}")
    out = []
    if floor_colors is None:
        for w in COLOR_NAMES[:wall_colors]:
            out.append({"id": len(out), "name": w, "wall": w, "floor": w})
    else:
        for w in COLOR_NAMES[:wall_colors]:
            for f in COLOR_NAMES[:floor_colors]:
                out.append({"id": len(out), "name": f"{w}/{f}", "wall": w, "floor": f})
    if n_rooms is not None:
        if len(out) < n_rooms:
            raise ValueError(
                f"colour scheme wall_colors={wall_colors}, floor_colors={floor_colors} gives only "
                f"{len(out)} distinct room identities, but n_rooms={n_rooms}")
        out = out[:n_rooms]
    return out


def choose_color_scheme(n_rooms: int):
    """(wall_colors, floor_colors) for ``n_rooms``; floor_colors None = solid.

    Up to 8 rooms: one solid colour per room.  Beyond: two-tone wall x floor.
    Among all W x F grids (W, F <= 8) that hold ``n_rooms`` identities the one
    with the fewest unused combinations wins; ties go to 4 floor colours (the
    choice of the 16 and 32 room presets), then to the most square grid, then
    to more wall than floor colours.
    """
    n_col = len(COLORS)
    if n_rooms <= n_col:
        return n_rooms, None
    best = None
    for f in range(2, n_col + 1):
        w = -(-n_rooms // f)
        if w > n_col or w < 2:
            continue
        key = (w * f - n_rooms, f != 4, abs(w - f), w < f)
        if best is None or key < best[0]:
            best = (key, w, f)
    if best is None:
        raise ValueError(f"n_rooms must be at most {n_col * n_col}, got {n_rooms}")
    return best[1], best[2]


# --------------------------------------------------------------------------
# parameters for an arbitrary number of rooms
# --------------------------------------------------------------------------
PARAM_KEYS = ("n_rooms", "wall_colors", "floor_colors", "n_corridor", "lead_in", "branch_prob",
              "n_dead_ends", "n_distractors", "max_episode_steps", "train_seeds", "eval_seeds",
              "heldout_targets")


# Beyond the largest preset (see derive_params).
LARGE_CORRIDOR_PER_ROOM = 1.5
LARGE_BRANCH_PROB = 0.25


def _interp_presets(n_rooms: int, key: str) -> float:
    """Piecewise linear in n_rooms through the five presets (last slope beyond 32)."""
    xs = sorted(SCALES)
    for lo, hi in zip(xs[:-1], xs[1:]):
        if n_rooms <= hi or hi == xs[-1]:
            a, b = SCALES[lo][key], SCALES[hi][key]
            return a + (b - a) * (n_rooms - lo) / float(hi - lo)
    raise AssertionError


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def default_seeds(n_rooms: int, n_train: int = 4, n_eval: int = 2, seed_base: int | None = None):
    """(train_seeds, eval_seeds): ``base+1..`` and ``base+901..``, base = 1000 * n_rooms."""
    base = 1000 * n_rooms if seed_base is None else int(seed_base)
    if not 1 <= n_train <= 900:
        raise ValueError(f"need between 1 and 900 train layouts, got {n_train}")
    if n_eval < 0:
        raise ValueError(f"the number of eval layouts cannot be negative, got {n_eval}")
    return ([base + 1 + i for i in range(n_train)], [base + 901 + i for i in range(n_eval)])


def scaled_step_limit(base_steps: int, geom: "Geometry") -> int:
    """Step limit for a non-default cell size.

    Walking time grows with the cell size in map units, turning time does not,
    so the limit of the default geometry (cell = 320 units) is scaled by
    ``0.5 + 0.5 * cell / 320`` and rounded up to a multiple of 10.  With this
    rule the scripted oracle needs roughly 25-50 % of the limit for every
    verified size.
    """
    if geom.cell == CELL:
        return int(base_steps)
    factor = 0.5 + 0.5 * geom.cell / float(CELL)
    return int(math.ceil(base_steps * factor / 10.0 - 1e-9)) * 10


def derive_params(n_rooms: int, geom: "Geometry | None" = None, n_train_layouts: int = 4,
                  n_eval_layouts: int = 2, seed_base: int | None = None, **overrides) -> dict:
    """Generator parameters for ``n_rooms`` candidate rooms (2 .. 64).

    Up to 32 rooms the values follow the trend of the five presets (piecewise
    linear interpolation in ``n_rooms``, rounded).  Beyond 32 rooms the last
    slope is continued, except for two values: the denser and more branched
    mazes that the trend asks for leave too few free cells next to the
    corridors to attach all rooms, so ``n_corridor`` becomes ``1.5 * n_rooms``
    and ``branch_prob`` 0.25 (:data:`LARGE_CORRIDOR_PER_ROOM`,
    :data:`LARGE_BRANCH_PROB`).

    ==================  =====================================================
    wall_colors /       solid colours up to 8 rooms, two-tone wall x floor
    floor_colors        beyond (see :func:`choose_color_scheme`)
    n_corridor          corridor cells after the airlock (path budget)
    lead_in             corridor cells before the first cell a room may hang
                        on - the minimum memory distance
    branch_prob         probability that a new corridor cell starts a fork
    n_dead_ends         empty corridor stubs
    n_distractors       neutral (grey) leaf rooms
    max_episode_steps   step limit; scaled with the cell size, see
                        :func:`scaled_step_limit`
    heldout_targets     identities kept out of the ``train`` split
    train_seeds /       layout seeds, see :func:`default_seeds`
    eval_seeds
    ==================  =====================================================

    For 2, 4, 8, 16 and 32 rooms with default arguments the result equals the
    preset in :data:`SCALES` exactly.  Every key can be overridden through a
    keyword argument, e.g. ``derive_params(6, lead_in=5, n_distractors=3)``.
    """
    if isinstance(n_rooms, bool) or not isinstance(n_rooms, int):
        raise ValueError(f"n_rooms must be an integer, got {n_rooms!r}")
    if not 2 <= n_rooms <= MAX_ROOMS:
        raise ValueError(
            f"n_rooms must be in [2, {MAX_ROOMS}], got {n_rooms}: with one room there is nothing "
            f"to remember and {MAX_ROOMS} = 8 wall x 8 floor colours is all the palette offers")
    unknown = sorted(set(overrides) - set(PARAM_KEYS))
    if unknown:
        raise ValueError(f"unknown generator parameter(s) {unknown}; valid: {list(PARAM_KEYS)}")
    geom = geom or DEFAULT_GEOMETRY
    wall, floor = choose_color_scheme(n_rooms)
    large = n_rooms > max(SCALES)
    train_seeds, eval_seeds = default_seeds(n_rooms, n_train_layouts, n_eval_layouts, seed_base)
    p = dict(
        n_rooms=n_rooms, wall_colors=wall, floor_colors=floor,
        n_corridor=(_round_half_up(_interp_presets(n_rooms, "n_corridor")) if not large
                    else _round_half_up(LARGE_CORRIDOR_PER_ROOM * n_rooms)),
        lead_in=_round_half_up(_interp_presets(n_rooms, "lead_in")),
        branch_prob=(round(_interp_presets(n_rooms, "branch_prob"), 3) if not large
                     else LARGE_BRANCH_PROB),
        n_dead_ends=_round_half_up(_interp_presets(n_rooms, "n_dead_ends")),
        n_distractors=_round_half_up(_interp_presets(n_rooms, "n_distractors")),
        max_episode_steps=scaled_step_limit(
            int(math.ceil(_interp_presets(n_rooms, "max_episode_steps") / 10.0 - 1e-9)) * 10, geom),
        train_seeds=train_seeds, eval_seeds=eval_seeds,
        heldout_targets=_round_half_up(_interp_presets(n_rooms, "heldout_targets")),
    )
    if "lead_in" in overrides and "n_corridor" not in overrides:
        # a longer lead-in needs corridor cells of its own
        p["n_corridor"] += max(0, int(overrides["lead_in"]) - p["lead_in"])
    p.update(overrides)
    validate_params(p)
    return p


def validate_params(p: dict) -> None:
    """Raise ValueError with an explanation if a parameter set cannot work."""
    missing = [k for k in PARAM_KEYS if k not in p]
    if missing:
        raise ValueError(f"generator parameters are missing {missing}")

    def _int(key, lo, hi=None):
        v = p[key]
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{key} must be an integer, got {v!r}")
        if v < lo or (hi is not None and v > hi):
            rng = f">= {lo}" if hi is None else f"in [{lo}, {hi}]"
            raise ValueError(f"{key} must be {rng}, got {v}")
        return v

    n = _int("n_rooms", 2, MAX_ROOMS)
    build_identities(_int("wall_colors", 1, len(COLORS)), p["floor_colors"], n)
    lead = _int("lead_in", 2)
    n_corr = _int("n_corridor", 2)
    if n_corr < lead:
        raise ValueError(f"n_corridor ({n_corr}) must be at least lead_in ({lead}): the lead-in "
                         f"cells are part of the corridor budget")
    if not 0.0 <= float(p["branch_prob"]) <= 1.0:
        raise ValueError(f"branch_prob must be in [0, 1], got {p['branch_prob']}")
    _int("n_dead_ends", 0)
    _int("n_distractors", 0)
    _int("max_episode_steps", 1)
    held = _int("heldout_targets", 0)
    if held >= n:
        raise ValueError(f"heldout_targets ({held}) must be smaller than n_rooms ({n}), otherwise "
                         f"the train split has no target")
    for key in ("train_seeds", "eval_seeds"):
        seeds = p[key]
        if not isinstance(seeds, (list, tuple)) or any(
                isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
            raise ValueError(f"{key} must be a list of non-negative integers, got {seeds!r}")
    if not p["train_seeds"]:
        raise ValueError("train_seeds must contain at least one seed")
    all_seeds = list(p["train_seeds"]) + list(p["eval_seeds"])
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError(f"layout seeds must be distinct (train and eval together): {all_seeds}")
    if len(all_seeds) > MAX_LAYOUTS:
        raise ValueError(f"at most {MAX_LAYOUTS} layouts per scenario, got {len(all_seeds)}")


# --------------------------------------------------------------------------
# layout generation
# --------------------------------------------------------------------------
def generate_layout(scale, map_seed: int, params: dict | None = None,
                    geom: Geometry | None = None) -> dict:
    """Generate one layout (cells + links + colour assignment) from a seed.

    ``scale`` is a preset key of :data:`SCALES` (optionally patched by
    ``params``) or a complete parameter dict such as the result of
    :func:`derive_params`.  The same (parameters, seed, geometry) always give
    the same layout.
    """
    if isinstance(scale, dict):
        p = dict(scale)
        scale = p["n_rooms"]
    else:
        if scale not in SCALES:
            raise ValueError(f"no preset for {scale} rooms (presets: {sorted(SCALES)}); "
                             f"pass derive_params({scale}) instead")
        p = dict(SCALES[scale])
    if params:
        p.update(params)
    validate_params(p)
    geom = geom or DEFAULT_GEOMETRY
    identities = build_identities(p["wall_colors"], p["floor_colors"], p["n_rooms"])
    n_total = 2 + p["n_corridor"] + p["n_rooms"] + p["n_dead_ends"] + p["n_distractors"]
    side = int(math.ceil(math.sqrt(n_total * 3.0))) + 2
    extent = (side + 2) * geom.cell
    if extent > MAX_MAP_UNITS:
        raise ValueError(
            f"map too large: {side + 2} cells x {geom.cell} units = {extent} map units, the "
            f"engine limit used here is {MAX_MAP_UNITS}; reduce tile, cell_tiles or the number "
            f"of rooms / corridor cells")
    for attempt in range(200):
        rng = random.Random(map_seed * 1000 + attempt)
        layout = _try_layout(p, identities, rng)
        if layout is None:
            continue
        layout["map_seed"] = map_seed
        layout["attempt"] = attempt
        layout["scale"] = scale
        layout["geom"] = geom
        _finalise_layout(layout)
        if layout["cue_los_violations"] != 0:
            continue
        # second, strict test (thin sight lines through two openings that the
        # point samples above can miss, mostly with large cells)
        layout["cue_portal_los_violations"] = portal_los_violations(layout)
        if layout["cue_portal_los_violations"] == 0:
            return layout
    raise RuntimeError(
        f"could not generate a layout for {p['n_rooms']} rooms, seed {map_seed}, {geom!r} in 200 "
        f"attempts (no room placement without a line of sight from the cue zone, or not enough "
        f"corridor cells to attach all rooms); try another seed, a larger n_corridor or fewer "
        f"dead ends / distractors")


def _try_layout(p, identities, rng):
    n_total = 2 + p["n_corridor"] + p["n_rooms"] + p["n_dead_ends"] + p["n_distractors"]
    side = int(math.ceil(math.sqrt(n_total * 3.0))) + 2
    W, H = side + 2, side
    cy0 = H // 2
    cells = {}       # (cx,cy) -> dict(kind, depth, parent)
    links = []       # (cell_a, cell_b)

    def free(c):
        return 0 <= c[0] < W and 0 <= c[1] < H and c not in cells

    def add(c, kind, parent):
        depth = 0 if parent is None else cells[parent]["depth"] + 1
        cells[c] = {"kind": kind, "depth": depth, "parent": parent}
        if parent is not None:
            links.append((parent, c))

    cue = (0, cy0)
    air = (1, cy0)
    add(cue, "cue", None)
    cells[cue]["depth"] = -1
    cells[air] = {"kind": "airlock", "depth": 0, "parent": cue}
    links.append((cue, air))

    # mandatory lead-in: one step east, then a bend north or south
    tip = (2, cy0)
    add(tip, "corridor", air)
    n_corr = 1
    bend = rng.choice([(0, 1), (0, -1)])
    for i in range(1, max(2, p["lead_in"])):
        if n_corr >= p["n_corridor"]:
            break
        d = bend if i == 1 else rng.choice([bend, (1, 0), (1, 0)])
        nxt = (tip[0] + d[0], tip[1] + d[1])
        if not free(nxt):
            return None
        add(nxt, "corridor", tip)
        tip = nxt
        n_corr += 1
    min_depth = cells[tip]["depth"]

    def neighbours(c):
        ds = list(DIRS.values())
        rng.shuffle(ds)
        return [(c[0] + d[0], c[1] + d[1]) for d in ds]

    guard = 0
    while n_corr < p["n_corridor"]:
        guard += 1
        if guard > 5000:
            return None
        if rng.random() < p["branch_prob"]:
            base_pool = [c for c, v in cells.items()
                         if v["kind"] == "corridor" and v["depth"] >= min_depth]
            base = rng.choice(base_pool)
        else:
            base = tip
        opts = [n for n in neighbours(base) if free(n)]
        if not opts:
            pool = [c for c, v in cells.items()
                    if v["kind"] == "corridor" and v["depth"] >= min_depth
                    and any(free(n) for n in neighbours(c))]
            if not pool:
                return None
            tip = rng.choice(pool)
            continue
        # spread out: prefer the option with most free surroundings so that
        # enough free neighbour cells remain for rooms (ties broken by seed)
        opts.sort(key=lambda n: (-sum(1 for m in neighbours(n) if free(m)), rng.random()))
        nxt = opts[0] if rng.random() < 0.8 else rng.choice(opts)
        add(nxt, "corridor", base)
        tip = nxt
        n_corr += 1

    def attach(kind, count):
        for _ in range(count):
            cands = []
            for c, v in cells.items():
                if v["kind"] != "corridor" or v["depth"] < min_depth:
                    continue
                for n in neighbours(c):
                    if free(n):
                        cands.append((c, n))
            if not cands:
                return False
            # prefer corridor cells with few children so rooms spread out
            def load(pc):
                return sum(1 for a, _b in links if a == pc[0])
            cands.sort(key=lambda pc: (load(pc), rng.random()))
            parent, c = cands[0]
            add(c, kind, parent)
        return True

    if not attach("room", p["n_rooms"]):
        return None
    if not attach("deadend", p["n_dead_ends"]):
        return None
    if not attach("distractor", p["n_distractors"]):
        return None

    room_cells = [c for c, v in cells.items() if v["kind"] == "room"]
    perm = list(range(len(identities)))
    rng.shuffle(perm)
    for c, ident in zip(room_cells, perm):
        cells[c]["room_id"] = len([1 for cc in room_cells[:room_cells.index(c)]])
        cells[c]["identity"] = ident
    return {"W": W, "H": H, "cells": cells, "links": links, "cue": cue, "airlock": air,
            "identities": identities, "params": p}


def cell_center(c, geom: Geometry | None = None):
    """Centre of cell ``c`` in map units (default geometry unless ``geom`` is given)."""
    return (geom or DEFAULT_GEOMETRY).cell_center(c)


def link_center(a, b, geom: Geometry | None = None):
    """Centre of the opening between two adjacent cells (map units)."""
    return (geom or DEFAULT_GEOMETRY).link_center(a, b)


def build_tiles(layout):
    """Rasterise a layout to a tile grid of sector keys (None = solid)."""
    geom = layout["geom"]
    ct = geom.cell_tiles
    TW, TH = layout["W"] * ct + 1, layout["H"] * ct + 1
    tiles = [[None] * TW for _ in range(TH)]
    for c, v in layout["cells"].items():
        kind = v["kind"]
        for iy in range(1, ct):
            for ix in range(1, ct):
                x, y = c[0] * ct + ix, c[1] * ct + iy
                if kind == "cue":
                    key = "cue"
                elif kind == "airlock":
                    key = "airA" if ix <= AIRLOCK_A_TILES else "airB"
                elif kind == "room":
                    key = "room%d" % v["room_id"]
                else:
                    key = "N"
                tiles[y][x] = key
    for a, b in layout["links"]:
        is_door = layout["cells"][a]["kind"] == "cue" or layout["cells"][b]["kind"] == "cue"
        for (x, y) in geom.link_tiles(a, b):
            tiles[y][x] = "door" if is_door else "N"
    return tiles


def line_of_sight(open_grid, x0, y0, x1, y1, step=8.0, tile=TILE):
    """2-D visibility test on the tile grid (open_grid[y][x] truthy = walkable)."""
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    n = max(1, int(dist / step))
    for i in range(n + 1):
        t = i / n
        tx = int((x0 + dx * t) // tile)
        ty = int((y0 + dy * t) // tile)
        if ty < 0 or tx < 0 or ty >= len(open_grid) or tx >= len(open_grid[0]):
            return False
        if not open_grid[ty][tx]:
            return False
    return True


def room_sample_points(c, geom: Geometry | None = None):
    """Points inside a room cell used for the visibility bookkeeping."""
    geom = geom or DEFAULT_GEOMETRY
    x0 = (c[0] * geom.cell_tiles + 1) * geom.tile
    y0 = (c[1] * geom.cell_tiles + 1) * geom.tile
    s = geom.content * geom.tile
    pts = []
    for fx in (0.06, 0.5, 0.94):
        for fy in (0.06, 0.5, 0.94):
            pts.append((x0 + fx * s, y0 + fy * s))
    return pts


def _link_face(geom, a, b, side):
    """End points of the face of the opening a-b that borders cell ``side`` (a or b)."""
    tiles = geom.link_tiles(a, b)
    t = geom.tile
    lo = a if (a[0] < b[0] or a[1] < b[1]) else b
    far = 0 if side == lo else 1          # the opening lies in the border of the high cell
    if tiles[0][0] == tiles[-1][0]:       # east-west link: a column of tiles
        x = (tiles[0][0] + far) * t
        return (x, tiles[0][1] * t), (x, (tiles[-1][1] + 1) * t)
    y = (tiles[0][1] + far) * t
    return (tiles[0][0] * t, y), ((tiles[-1][0] + 1) * t, y)


def portal_los_violations(layout) -> int:
    """Strict check: number of candidate rooms that can be seen from the cue zone.

    The cue zone (cue room, door, airlock) has a single exit, the opening
    between the airlock and the first corridor cell, and a candidate room is a
    leaf with a single opening.  Any sight line from the cue zone into a room
    therefore crosses both openings, so it is enough to test the sight lines
    between the airlock-side face of the exit and the room-side face of the
    room's opening.  Both faces are sampled every 1/16 tile and a ray counts as
    blocked only where it is more than 1/32 tile deep inside a wall; with this
    margin a real sight line cannot slip between the samples (the test errs on
    the side of rejecting a layout).
    """
    geom = layout["geom"]
    tile = float(geom.tile)
    tiles = layout["tiles"]
    th, tw = len(tiles), len(tiles[0])
    cells = layout["cells"]
    air = layout["airlock"]
    exits = [b for a, b in layout["links"] if a == air]
    assert len(exits) == 1, "the airlock must have exactly one exit"
    spacing, margin, step = tile / 16.0, tile / 32.0, tile / 8.0

    def is_open(x, y):
        tx, ty = int(x // tile), int(y // tile)
        return 0 <= tx < tw and 0 <= ty < th and tiles[ty][tx] is not None

    def passable(x, y):
        return (is_open(x, y) or is_open(x - margin, y) or is_open(x + margin, y)
                or is_open(x, y - margin) or is_open(x, y + margin))

    def samples(p, q):
        n = int(round(math.hypot(q[0] - p[0], q[1] - p[1]) / spacing))
        return [(p[0] + (q[0] - p[0]) * i / n, p[1] + (q[1] - p[1]) * i / n) for i in range(n + 1)]

    def clear(p, q):
        dx, dy = q[0] - p[0], q[1] - p[1]
        n = max(1, int(math.hypot(dx, dy) / step))
        for i in range(n + 1):
            if not passable(p[0] + dx * i / n, p[1] + dy * i / n):
                return False
        return True

    src = samples(*_link_face(geom, air, exits[0], air))
    seen = 0
    for c, v in cells.items():
        if v["kind"] != "room":
            continue
        dst = samples(*_link_face(geom, v["parent"], c, c))
        if any(clear(p, q) for p in src for q in dst):
            seen += 1
    return seen


def _finalise_layout(layout):
    cells = layout["cells"]
    geom = layout["geom"]
    TILE = geom.tile
    los_step = TILE / 8.0      # 8 map units for the default tile, 1/8 tile in general
    tiles = build_tiles(layout)
    layout["tiles"] = tiles
    open_grid = [[t is not None for t in row] for row in tiles]
    # adjacency over cells (tree)
    adj = {c: [] for c in cells}
    for a, b in layout["links"]:
        adj[a].append(b)
        adj[b].append(a)
    layout["adj"] = adj
    # geometric proof obligation: no room is visible from the cue zone
    viol = 0
    cue_zone_tiles = [(x, y) for y, row in enumerate(tiles) for x, t in enumerate(row)
                      if t in ("cue", "door", "airA", "airB")]
    rooms = [c for c, v in cells.items() if v["kind"] == "room"]
    for (tx, ty) in cue_zone_tiles:
        for ox, oy in ((0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9), (0.5, 0.5)):
            px, py = (tx + ox) * TILE, (ty + oy) * TILE
            for rc in rooms:
                for (qx, qy) in room_sample_points(rc, geom):
                    if line_of_sight(open_grid, px, py, qx, qy, los_step, TILE):
                        viol += 1
    layout["cue_los_violations"] = viol
    # statistics
    corr = [c for c, v in cells.items() if v["kind"] in ("corridor", "airlock")]
    forks = 0
    for c in corr:
        children = [b for a, b in layout["links"] if a == c and cells[b]["kind"] == "corridor"]
        if len(children) >= 2:
            forks += 1
    dead = [c for c, v in cells.items()
            if v["kind"] == "deadend"
            or (v["kind"] == "corridor" and len(adj[c]) == 1)]
    room_depths = [cells[c]["depth"] for c in rooms]
    layout["stats"] = {
        "n_rooms": len(rooms),
        "n_corridor_cells": len([c for c, v in cells.items() if v["kind"] == "corridor"]),
        "n_forks": forks,
        "n_dead_ends": len(dead),
        "n_distractors": len([c for c, v in cells.items() if v["kind"] == "distractor"]),
        "path_cells_min": min(room_depths),
        "path_cells_mean": sum(room_depths) / len(room_depths),
        "path_cells_max": max(room_depths),
        "grid_cells": [layout["W"], layout["H"]],
        "map_units": [layout["W"] * geom.cell, layout["H"] * geom.cell],
    }


# --------------------------------------------------------------------------
# UDMF writer
# --------------------------------------------------------------------------
def _sector_props(key, layout, target_identity):
    idents = layout["identities"]
    if key == "cue":
        ident = idents[target_identity]
        return dict(floor=CUE_RAISE, ceil=CEIL_LOW + CUE_RAISE,
                    tfloor=color_flat(ident["floor"]), tceil=color_flat(ident["floor"]),
                    wall=color_flat(ident["wall"]), tag=0)
    if key == "door":
        return dict(floor=CUE_RAISE, ceil=CEIL_LOW + CUE_RAISE, tfloor=FLAT_FLOOR,
                    tceil=FLAT_CEIL, wall=FLAT_WALL, tag=DOOR_TAG)
    if key.startswith("room"):
        rid = int(key[4:])
        cell = [v for v in layout["cells"].values() if v.get("room_id") == rid][0]
        ident = idents[cell["identity"]]
        return dict(floor=FLOOR_LOW, ceil=CEIL_LOW, tfloor=color_flat(ident["floor"]),
                    tceil=color_flat(ident["floor"]), wall=color_flat(ident["wall"]), tag=0)
    return dict(floor=FLOOR_LOW, ceil=CEIL_LOW, tfloor=FLAT_FLOOR, tceil=FLAT_CEIL,
                wall=FLAT_WALL, tag=0)


def build_textmap(layout, target_identity: int, start_angle: int = 90) -> str:
    tiles = layout["tiles"]
    geom = layout["geom"]
    TILE = geom.tile
    TH, TW = len(tiles), len(tiles[0])
    keys = sorted({t for row in tiles for t in row if t is not None},
                  key=lambda k: (len(k), k))
    sec_index = {k: i for i, k in enumerate(keys)}
    props = {k: _sector_props(k, layout, target_identity) for k in keys}

    def T(x, y):
        if 0 <= x < TW and 0 <= y < TH:
            return tiles[y][x]
        return None

    # collect unit edges: (orientation, fixed coord, start, front, back, special)
    runs = {}
    for y in range(TH):
        for x in range(-1, TW):
            w, e = T(x, y), T(x + 1, y)          # vertical edge at X = x+1
            if w == e:
                continue
            if e is not None and w is not None:
                front, back, rev = e, w, False
            elif e is not None:
                front, back, rev = e, None, False     # line goes north, front east
            else:
                front, back, rev = w, None, True      # line goes south, front west
            special = (front, back) in (("airB", "airA"), ("airA", "airB"))
            runs.setdefault(("V", x + 1, front, back, rev, special), []).append(y)
    for x in range(TW):
        for y in range(-1, TH):
            s, n = T(x, y), T(x, y + 1)          # horizontal edge at Y = y+1
            if s == n:
                continue
            if s is not None and n is not None:
                front, back, rev = s, n, False        # line goes east, front south
            elif s is not None:
                front, back, rev = s, None, False
            else:
                front, back, rev = n, None, True      # line goes west, front north
            runs.setdefault(("H", y + 1, front, back, rev, False), []).append(x)

    verts, vindex = [], {}

    def V(x, y):
        k = (x, y)
        if k not in vindex:
            vindex[k] = len(verts)
            verts.append(k)
        return vindex[k]

    lines, sides = [], []

    side_index = {}

    def side(seckey, one_sided):
        # identical sidedefs are shared between linedefs (keeps TEXTMAP small)
        k = (seckey, one_sided)
        if k not in side_index:
            side_index[k] = len(sides)
            sides.append((sec_index[seckey], props[seckey]["wall"], one_sided))
        return side_index[k]

    for key in sorted(runs.keys(), key=lambda k: (k[0], k[1], k[2], k[3] or "", k[4])):
        orient, fixed, front, back, rev, special = key
        vals = sorted(runs[key])
        # merge consecutive unit edges
        segs = []
        start = prev = vals[0]
        for v in vals[1:]:
            if v == prev + 1:
                prev = v
                continue
            segs.append((start, prev + 1))
            start = prev = v
        segs.append((start, prev + 1))
        for a, b in segs:
            if orient == "V":
                p0, p1 = (fixed * TILE, a * TILE), (fixed * TILE, b * TILE)
            else:
                p0, p1 = (a * TILE, fixed * TILE), (b * TILE, fixed * TILE)
            if rev:
                p0, p1 = p1, p0
            sf = side(front, back is None)
            sb = side(back, False) if back is not None else -1
            lines.append((V(*p0), V(*p1), sf, sb, special))

    out = ['namespace="zdoom";']
    sx, sy = geom.cell_center(layout["cue"])
    out.append('thing{x=%d.0;y=%d.0;angle=%d;type=1;skill1=true;skill2=true;skill3=true;'
               'skill4=true;skill5=true;single=true;coop=true;dm=true;class1=true;'
               'class2=true;class3=true;}' % (sx, sy, start_angle))
    for (x, y) in verts:
        out.append('vertex{x=%d.0;y=%d.0;}' % (x, y))
    for v1, v2, sf, sb, special in lines:
        s = 'linedef{v1=%d;v2=%d;sidefront=%d;' % (v1, v2, sf)
        if sb >= 0:
            s += 'sideback=%d;twosided=true;' % sb
        else:
            s += 'blocking=true;'
        if special:
            s += 'special=10;arg0=%d;arg1=%d;playercross=true;repeatspecial=true;' % (
                DOOR_TAG, DOOR_SPEED)
        out.append(s + '}')
    for sec, wall, one_sided in sides:
        if one_sided:
            out.append('sidedef{sector=%d;texturemiddle="%s";}' % (sec, wall))
        else:
            out.append('sidedef{sector=%d;texturetop="%s";texturebottom="%s";}' % (sec, wall, wall))
    for k in keys:
        pr = props[k]
        s = ('sector{heightfloor=%d;heightceiling=%d;texturefloor="%s";textureceiling="%s";'
             'lightlevel=255;' % (pr["floor"], pr["ceil"], pr["tfloor"], pr["tceil"]))
        if pr["tag"]:
            s += 'id=%d;' % pr["tag"]
        out.append(s + '}')
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# WAD writer
# --------------------------------------------------------------------------
def _solid_flat(idx: int) -> bytes:
    return bytes([idx]) * 4096


def _brick_flat(base: int, joint: int) -> bytes:
    px = bytearray([base]) * 4096
    for y in range(64):
        for x in range(64):
            row = y // 16
            if y % 16 == 0 or (x + (row % 2) * 16) % 32 == 0:
                px[y * 64 + x] = joint
    return bytes(px)


def _tile_flat(base: int, joint: int) -> bytes:
    px = bytearray([base]) * 4096
    for y in range(64):
        for x in range(64):
            if y % 32 == 0 or x % 32 == 0:
                px[y * 64 + x] = joint
    return bytes(px)


def build_flats():
    flats = [(FLAT_WALL, _brick_flat(GREY_WALL, GREY_WALL_DARK)),
             (FLAT_FLOOR, _tile_flat(GREY_FLOOR, GREY_DOOR)),
             (FLAT_CEIL, _solid_flat(GREY_CEIL))]
    for name, idx, _rgb in COLORS:
        flats.append((color_flat(name), _solid_flat(idx)))
    return flats


def write_wad(path: str, lumps) -> None:
    """lumps: list of (name, bytes)."""
    data = bytearray()
    directory = []
    offset = 12
    for name, payload in lumps:
        directory.append((offset, len(payload), name))
        data += payload
        offset += len(payload)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sii", b"PWAD", len(lumps), offset))
        f.write(bytes(data))
        for off, size, name in directory:
            f.write(struct.pack("<ii8s", off, size, name.encode("ascii")))


def map_lump_name(layout_index: int, target_identity: int) -> str:
    return "L%02dT%02d" % (layout_index, target_identity)


# --------------------------------------------------------------------------
# scale builder
# --------------------------------------------------------------------------
def layout_metadata(layout, layout_index, split):
    cells = layout["cells"]
    geom = layout["geom"]
    TILE, CELL_TILES = geom.tile, geom.cell_tiles
    rooms = []
    for c, v in cells.items():
        if v["kind"] != "room":
            continue
        ident = layout["identities"][v["identity"]]
        rooms.append({
            "room_id": v["room_id"], "cell": list(c), "parent": list(v["parent"]),
            "identity": v["identity"], "color": ident["name"], "wall": ident["wall"],
            "floor": ident["floor"], "depth_cells": v["depth"],
            "center": list(geom.cell_center(c)),
            "door": list(geom.link_center(v["parent"], c)),
        })
    rooms.sort(key=lambda r: r["room_id"])
    return {
        "layout_id": layout_index,
        "map_seed": layout["map_seed"],
        "seed_attempt": layout["attempt"],
        "split": split,
        "grid": [layout["W"], layout["H"]],
        "cue_cell": list(layout["cue"]),
        "airlock_cell": list(layout["airlock"]),
        "door_x": layout["airlock"][0] * CELL_TILES * TILE,
        "ledge_x": (layout["airlock"][0] * CELL_TILES + 1) * TILE,
        "trigger_x": (layout["airlock"][0] * CELL_TILES + 1 + AIRLOCK_A_TILES) * TILE,
        "cells": [{"cell": list(c), "kind": v["kind"], "depth": v["depth"],
                   "parent": (list(v["parent"]) if v["parent"] is not None else None)}
                  for c, v in sorted(cells.items())],
        "links": [[list(a), list(b)] for a, b in layout["links"]],
        "rooms": rooms,
        "stats": layout["stats"],
        "cue_los_violations": layout["cue_los_violations"],
        "cue_portal_los_violations": layout["cue_portal_los_violations"],
        "tiles": ["".join(_tile_char(t) for t in row) for row in layout["tiles"]],
    }


def _tile_char(t):
    if t is None:
        return "#"
    if t == "N":
        return "."
    if t == "cue":
        return "c"
    if t == "door":
        return "d"
    if t == "airA":
        return "a"
    if t == "airB":
        return "b"
    return "r"


_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def scenario_file_stem(name: str) -> str:
    """``r6_big`` -> ``multiroom_color_goal_r6_big`` (an existing prefix is kept)."""
    name = str(name)
    if name.startswith(SCENARIO_PREFIX):
        name = name[len(SCENARIO_PREFIX):]
    if not _NAME_RE.match(name):
        raise ValueError(f"invalid scenario name {name!r}: use 1-64 letters, digits, '_', '-' or "
                         f"'.', starting with a letter or digit")
    return SCENARIO_PREFIX + name


def default_scenario_name(n_rooms: int, geom: Geometry | None = None) -> str:
    """``r<rooms>`` for the default geometry, else ``r<rooms>_c<cell_tiles>_t<tile>``."""
    geom = geom or DEFAULT_GEOMETRY
    if geom == DEFAULT_GEOMETRY:
        return "r%d" % n_rooms
    name = "r%d_c%d_t%d" % (n_rooms, geom.cell_tiles, geom.tile)
    return name if geom.link_w == LINK_W else name + "_w%d" % geom.link_w


def build_scenario(params: dict, out_dir: str, name: str | None = None,
                   geom: Geometry | None = None, verbose: bool = True) -> dict:
    """Generate one scenario and write ``multiroom_color_goal_<name>.{wad,json,cfg}``.

    ``params`` is a complete parameter dict (a preset from :data:`SCALES` or the
    result of :func:`derive_params`), ``geom`` the room size.  Returns the
    metadata that was written to the .json.  The output is a pure function of
    (params, geom): the same call always writes the same bytes.
    """
    p = dict(params)
    validate_params(p)
    geom = geom or DEFAULT_GEOMETRY
    scale = p["n_rooms"]
    identities = build_identities(p["wall_colors"], p["floor_colors"], scale)
    stem = scenario_file_stem(name if name is not None else default_scenario_name(scale, geom))
    lumps = []
    layouts_meta = []
    seeds = [(s, "train") for s in p["train_seeds"]] + [(s, "eval") for s in p["eval_seeds"]]
    for li, (seed, split) in enumerate(seeds):
        layout = generate_layout(p, seed, geom=geom)
        for ident in identities:
            lumps.append((map_lump_name(li, ident["id"]), b""))
            lumps.append(("TEXTMAP", build_textmap(layout, ident["id"]).encode("ascii")))
            lumps.append(("ENDMAP", b""))
        layouts_meta.append(layout_metadata(layout, li, split))
    lumps.append(("FF_START", b""))
    lumps.extend(build_flats())
    lumps.append(("FF_END", b""))

    os.makedirs(out_dir, exist_ok=True)
    wad_path = os.path.join(out_dir, stem + ".wad")
    write_wad(wad_path, lumps)

    n_id = len(identities)
    n_held = p["heldout_targets"]
    # held-out target identities: deterministic, spread over the identity list
    held = sorted(random.Random(scale * 7919).sample(range(n_id), n_held)) if n_held else []
    train_targets = [i for i in range(n_id) if i not in held]
    train_layouts = [m["layout_id"] for m in layouts_meta if m["split"] == "train"]
    eval_layouts = [m["layout_id"] for m in layouts_meta if m["split"] == "eval"]
    meta = {
        "family": "MultiRoomColorGoal",
        "generator_version": GENERATOR_VERSION,
        "name": stem[len(SCENARIO_PREFIX):],
        "scale": scale,
        "wad": stem + ".wad",
        "cfg": stem + ".cfg",
        "tile": geom.tile, "cell_tiles": geom.cell_tiles, "content_tiles": geom.content,
        "link_lo": geom.link_lo, "link_w": geom.link_w,
        "cell_units": geom.cell, "room_units": geom.content * geom.tile,
        "cue_raise": CUE_RAISE, "door_tag": DOOR_TAG, "door_speed": DOOR_SPEED,
        "door_close_tics": DOOR_CLOSE_TICS,
        "colors": {n: {"palette_index": i, "rgb": list(rgb)} for n, i, rgb in COLORS},
        "color_scheme": "solid" if p["floor_colors"] is None else "two_tone",
        "identities": identities,
        "params": {k: v for k, v in p.items()},
        "map_lump_pattern": "L{layout:02d}T{target:02d}",
        "layouts": layouts_meta,
        "splits": {
            "train": {"layouts": train_layouts, "targets": train_targets},
            "eval_maps": {"layouts": eval_layouts, "targets": train_targets},
            "eval_colors": {"layouts": train_layouts, "targets": held},
            "eval_maps_colors": {"layouts": eval_layouts, "targets": held},
            "eval_all": {"layouts": eval_layouts, "targets": list(range(n_id))},
            "all": {"layouts": train_layouts + eval_layouts, "targets": list(range(n_id))},
        },
    }
    with open(os.path.join(out_dir, stem + ".json"), "w", newline="\n") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
        f.write("\n")
    with open(os.path.join(out_dir, stem + ".cfg"), "w", newline="\n") as f:
        f.write(CFG_TEMPLATE.format(wad=stem + ".wad", scale=scale,
                                    first_map=map_lump_name(0, 0)))
    if verbose:
        print("built", wad_path, os.path.getsize(wad_path), "bytes;",
              len(seeds), "layouts x", n_id, "targets;", repr(geom))
        for m in layouts_meta:
            print("  layout", m["layout_id"], m["split"], "seed", m["map_seed"], m["stats"])
    return meta


def build_scale(scale: int, out_dir: str, verbose: bool = True) -> dict:
    """Rebuild one of the five presets (2, 4, 8, 16, 32 rooms, default geometry)."""
    if scale not in SCALES:
        raise ValueError(f"no preset for {scale} rooms (presets: {sorted(SCALES)}); use "
                         f"build_scenario(derive_params({scale}), ...) for other room counts")
    return build_scenario(SCALES[scale], out_dir, name="r%d" % scale, verbose=verbose)


CFG_TEMPLATE = """# VizDoom MultiRoomColorGoal, {scale} candidate rooms.
# Generated by multiroom_color_goal.map_generator - do not edit by hand.
# Episode logic (target colour, room entry, success / wrong choice / timeout) lives in
# multiroom_color_goal.multiroom_env; the WAD has no ACS.

doom_scenario_path = {wad}
doom_map = {first_map}
doom_skill = 1

living_reward = 0.0

screen_resolution = RES_160X120
screen_format = RGB24
render_hud = false
render_crosshair = false
render_weapon = false
render_decals = false
render_particles = false
render_messages = false
window_visible = false
sound_enabled = false

# the step limit is enforced by the Python wrapper (max_episode_steps); 0 = no engine timeout
episode_timeout = 0
episode_start_time = 1

available_buttons =
	{{
		TURN_LEFT
		TURN_RIGHT
		MOVE_FORWARD
	}}

available_game_variables = {{ POSITION_X POSITION_Y POSITION_Z ANGLE }}

mode = PLAYER
"""


def default_scenario_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")


def ascii_map(layout_meta) -> str:
    """Human readable picture of a layout (north is up)."""
    return "\n".join(reversed(layout_meta["tiles"]))


def main(argv=None):
    """Entry point of ``python -m multiroom_color_goal.map_generator`` (same as ``.generate``)."""
    from .generate import main as _main
    return _main(argv)


if __name__ == "__main__":
    main()
