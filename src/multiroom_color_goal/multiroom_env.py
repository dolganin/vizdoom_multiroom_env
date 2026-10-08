"""VizDoom MultiRoomColorGoal environment wrapper.

Classic (old gym style) API:

    env = MultiRoomColorGoalEnv(8)                      # shipped preset, 8 rooms
    env = MultiRoomColorGoalEnv(scenario="r6_big", scenario_dir="my_scenarios")
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    env.observation_space, env.action_space, env.max_episode_steps

The current Gymnasium API (``terminated`` / ``truncated``) is provided by
``multiroom_color_goal.gym_env.MultiRoomColorGoalGymEnv``.

Task: the agent starts in a coloured *cue room*.  The colour (for the 16 and 32
room scales: the wall colour + floor colour pair) identifies the one candidate
room it has to enter.  After leaving the cue room the agent drops 32 units
(cannot climb back) and a door closes behind it, so the cue is never visible
again.  Entering the candidate room with the target identity is a success,
entering any other candidate room is a *wrong choice*; both end the episode by
default.  The target identity is NOT part of the observation: it is only
visible as pixels of the cue room.

All privileged quantities (map id, room id, target colour, visible rooms,
distance from the cue ...) are reported exclusively through ``info``.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from collections import deque

import numpy as np
from gymnasium import spaces

from .map_generator import SCENARIO_PREFIX, Geometry

try:
    from vizdoom import DoomGame, GameVariable, ScreenFormat, ScreenResolution
except ImportError:  # pragma: no cover - handled at construction time
    DoomGame = None
    GameVariable = None
    ScreenFormat = None
    ScreenResolution = None

SCENARIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")
SUPPORTED_SCALES = (2, 4, 8, 16, 32)   # presets shipped inside the package

# Three buttons / three actions (the action set of VizDoom's "my_way_home").
ACTIONS = [
    [1, 0, 0],  # 0: TURN_LEFT
    [0, 1, 0],  # 1: TURN_RIGHT
    [0, 0, 1],  # 2: MOVE_FORWARD
]
ACTION_NAMES = ["TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"]
FOV_DEG = 90.0


def resolve_scenario(scenario, scenario_dir: str | None = None) -> tuple[str, str]:
    """Locate a scenario; returns ``(directory, file_stem)``.

    ``scenario`` may be a number of rooms (``8`` -> ``r8``), a scenario name
    (``"r6_big"`` or ``"multiroom_color_goal_r6_big"``) looked up in
    ``scenario_dir`` (default: the presets shipped with the package), or a
    path to one of the scenario's files (``"my_scenarios/r6_big"``,
    ``".../multiroom_color_goal_r6_big.json"``).
    """
    if isinstance(scenario, (int, np.integer)) and not isinstance(scenario, bool):
        scenario = "r%d" % int(scenario)
    scenario = os.fspath(scenario)
    head, tail = os.path.split(scenario)
    for ext in (".json", ".wad", ".cfg"):
        if tail.endswith(ext):
            tail = tail[:-len(ext)]
            break
    if not tail:
        raise ValueError(f"empty scenario name in {scenario!r}")
    stem = tail if tail.startswith(SCENARIO_PREFIX) else SCENARIO_PREFIX + tail
    if head:
        directory = head if scenario_dir is None or os.path.isabs(head) else os.path.join(
            scenario_dir, head)
    else:
        directory = scenario_dir or SCENARIO_DIR
    missing = [stem + ext for ext in (".json", ".wad", ".cfg")
               if not os.path.isfile(os.path.join(directory, stem + ext))]
    if missing:
        name = stem[len(SCENARIO_PREFIX):]
        try:
            found = sorted(f[len(SCENARIO_PREFIX):-5] for f in os.listdir(directory)
                           if f.startswith(SCENARIO_PREFIX) and f.endswith(".json"))
        except OSError:
            found = []
        raise FileNotFoundError(
            f"scenario {name!r} not found in {directory!r} (missing {', '.join(missing)}). "
            f"Scenarios available there: {found or 'none'}. Shipped presets: "
            f"rooms={list(SUPPORTED_SCALES)}. Generate others with "
            f"'python -m multiroom_color_goal.generate --rooms N --name NAME --out-dir DIR' and "
            f"pass scenario='NAME', scenario_dir='DIR'.")
    return directory, stem


def load_metadata(scale, scenario_dir: str | None = None) -> dict:
    """Side-car metadata of a scenario (``scale``: rooms, name or path, see resolve_scenario)."""
    directory, stem = resolve_scenario(scale, scenario_dir)
    with open(os.path.join(directory, stem + ".json")) as f:
        return json.load(f)


class _Layout:
    """Run-time view of one layout from the side-car metadata."""

    def __init__(self, meta: dict, lm: dict) -> None:
        self.meta = lm
        self.layout_id = lm["layout_id"]
        self.map_seed = lm["map_seed"]
        self.geom = Geometry.from_meta(meta)
        self.tile = meta["tile"]
        self.cell_tiles = meta["cell_tiles"]
        rows = lm["tiles"]
        self.th, self.tw = len(rows), len(rows[0])
        self.open = np.array([[ch != "#" for ch in row] for row in rows], dtype=bool)
        self.kind = np.array([[ord(ch) for ch in row] for row in rows], dtype=np.uint8)
        self.room_of_tile = -np.ones((self.th, self.tw), dtype=np.int32)
        self.rooms = lm["rooms"]
        ct = self.cell_tiles
        for r in self.rooms:
            cx, cy = r["cell"]
            self.room_of_tile[cy * ct + 1:(cy + 1) * ct, cx * ct + 1:(cx + 1) * ct] = r["room_id"]
        self.cells = {tuple(c["cell"]): c for c in lm["cells"]}
        self.adj = {c: [] for c in self.cells}
        for a, b in lm["links"]:
            self.adj[tuple(a)].append(tuple(b))
            self.adj[tuple(b)].append(tuple(a))
        self.identity_to_room = {r["identity"]: r["room_id"] for r in self.rooms}
        # visibility sample points: (n_rooms, 9, 2)
        pts = []
        s = meta["content_tiles"] * self.tile
        for r in self.rooms:
            x0 = (r["cell"][0] * ct + 1) * self.tile
            y0 = (r["cell"][1] * ct + 1) * self.tile
            pts.append([(x0 + fx * s, y0 + fy * s)
                        for fx in (0.06, 0.5, 0.94) for fy in (0.06, 0.5, 0.94)])
        self.room_points = np.asarray(pts, dtype=np.float64)
        cue = lm["cue_cell"]
        # x coordinate of the Door_Close trigger line (between airlock tile columns)
        self.trigger_x = lm["trigger_x"]
        self.door_x = lm["door_x"]
        self.ledge_x = lm["ledge_x"]
        # centre of the cue door (the opening between the cue room and the airlock)
        self.door_xy = self.geom.link_center(tuple(cue), tuple(lm["airlock_cell"]))

    def tile_of(self, x: float, y: float):
        tx = min(max(int(x // self.tile), 0), self.tw - 1)
        ty = min(max(int(y // self.tile), 0), self.th - 1)
        return tx, ty

    def cell_of(self, x: float, y: float):
        return int(x // (self.tile * self.cell_tiles)), int(y // (self.tile * self.cell_tiles))

    def visible_rooms(self, x: float, y: float, angle_deg: float, max_dist: float = 4000.0):
        """Rooms with at least one sample point inside the FOV and in line of sight."""
        d = self.room_points - np.array([x, y])
        dist = np.hypot(d[..., 0], d[..., 1])
        ang = np.degrees(np.arctan2(d[..., 1], d[..., 0]))
        diff = (ang - angle_deg + 180.0) % 360.0 - 180.0
        cand = (np.abs(diff) <= FOV_DEG / 2.0) & (dist <= max_dist) & (dist > 1.0)
        if not cand.any():
            return []
        ridx, pidx = np.nonzero(cand)
        dd = d[ridx, pidx]
        n = int(dist[ridx, pidx].max() / 12.0) + 2
        t = np.linspace(0.0, 1.0, n)[None, :]
        px = x + dd[:, 0:1] * t
        py = y + dd[:, 1:2] * t
        tx = np.clip((px // self.tile).astype(np.int64), 0, self.tw - 1)
        ty = np.clip((py // self.tile).astype(np.int64), 0, self.th - 1)
        ok = self.open[ty, tx].all(axis=1)
        return sorted(set(int(r) for r in ridx[ok]))


class MultiRoomColorGoalEnv:
    """MultiRoomColorGoal, classic API (``reset() -> obs``, ``step() -> obs, r, done, info``).

    ``scale`` selects a scenario by its number of rooms: one of the shipped
    presets (2, 4, 8, 16, 32) or, together with ``scenario_dir``, a generated
    scenario with the default name ``r<N>``.  ``scenario`` (a name or a path,
    see :func:`resolve_scenario`) selects a named custom scenario and takes
    precedence over ``scale``.
    """

    def __init__(
        self,
        scale: int = 8,
        split: str = "train",
        frame_skip: int = 4,
        resolution: str = "RES_160X120",
        grayscale: bool = False,
        resize: tuple | None = (84, 84),
        frame_stack: int = 4,
        max_episode_steps: int | None = None,
        success_reward: float = 1.0,
        wrong_choice_penalty: float = 0.0,
        timeout_penalty: float = 0.0,
        living_reward: float = -0.0001,
        exploration_reward: float = 0.0,
        terminate_on_wrong: bool = True,
        layouts: list | None = None,
        targets: list | None = None,
        seed: int | None = None,
        step_info: str = "full",
        track_visibility: bool = True,
        scenario_dir: str | None = None,
        realtime_mode: bool = False,
        scenario: str | None = None,
    ) -> None:
        if DoomGame is None:
            raise ImportError("vizdoom package is not installed")
        if scenario is None:
            if scale is None:
                raise ValueError("give the number of rooms (scale) or a scenario name")
            scenario = int(scale)
        self._scenario_dir, stem = resolve_scenario(scenario, scenario_dir)
        with open(os.path.join(self._scenario_dir, stem + ".json")) as f:
            self._meta = json.load(f)
        self._scenario = self._meta.get("name", stem[len(SCENARIO_PREFIX):])
        scale = int(self._meta["scale"])
        self._scale = scale
        self._identities = self._meta["identities"]
        if split not in self._meta["splits"]:
            raise ValueError(f"unknown split {split!r}; available: {sorted(self._meta['splits'])}")
        self._split = split
        sp = self._meta["splits"][split]
        self._layout_ids = [int(i) for i in (layouts if layouts is not None else sp["layouts"])]
        self._target_ids = [int(i) for i in (targets if targets is not None else sp["targets"])]
        if not self._layout_ids or not self._target_ids:
            raise ValueError(
                f"split {split!r} of scenario {self._scenario!r} is empty "
                f"(layouts={self._layout_ids}, targets={self._target_ids})")
        known_layouts = {lm["layout_id"] for lm in self._meta["layouts"]}
        bad = [i for i in self._layout_ids if i not in known_layouts]
        bad_t = [i for i in self._target_ids if not 0 <= i < len(self._identities)]
        if bad or bad_t:
            raise ValueError(
                f"scenario {self._scenario!r} has layouts {sorted(known_layouts)} and targets "
                f"0..{len(self._identities) - 1}; got layouts={bad or 'ok'}, "
                f"targets={bad_t or 'ok'}")
        self._layouts = {lm["layout_id"]: _Layout(self._meta, lm) for lm in self._meta["layouts"]}

        self._frame_skip = int(frame_skip)
        self._grayscale = bool(grayscale)
        self._resize = tuple(resize) if resize else None
        self._frame_stack = max(1, int(frame_stack))
        self._max_episode_steps = int(max_episode_steps if max_episode_steps is not None
                                      else self._meta["params"]["max_episode_steps"])
        self._success_reward = float(success_reward)
        self._wrong_choice_penalty = float(wrong_choice_penalty)
        self._timeout_penalty = float(timeout_penalty)
        self._living_reward = float(living_reward)
        self._exploration_reward = float(exploration_reward)
        self._terminate_on_wrong = bool(terminate_on_wrong)
        self._step_info = step_info
        self._track_visibility = bool(track_visibility)
        self._rng = np.random.default_rng(seed)
        self._door_close_steps = int(math.ceil(self._meta["door_close_tics"] / self._frame_skip))

        game = DoomGame()
        game.load_config(os.path.join(self._scenario_dir, stem + ".cfg"))
        game.set_doom_scenario_path(os.path.join(self._scenario_dir, stem + ".wad"))
        self._ini_path = os.path.join(
            tempfile.gettempdir(), f"_vizdoom_multiroom_{os.getpid()}.ini")
        game.set_doom_config_path(self._ini_path)
        game.set_screen_format(ScreenFormat.RGB24)
        game.set_screen_resolution(getattr(ScreenResolution, resolution))
        game.set_window_visible(bool(realtime_mode))
        if seed is not None:
            game.set_seed(int(seed) % (2 ** 31 - 1))
        game.init()
        self._game = game
        self._var_index = {v: i for i, v in enumerate(game.get_available_game_variables())}

        h, w = game.get_screen_height(), game.get_screen_width()
        if self._resize:
            h, w = self._resize
        self._single_c = 1 if self._grayscale else 3
        self._obs_shape = (self._single_c * self._frame_stack, h, w)
        self._observation_space = spaces.Box(low=0.0, high=1.0, shape=self._obs_shape,
                                             dtype=np.float32)
        self._action_space = spaces.Discrete(len(ACTIONS))
        self._frame_buffer = deque(maxlen=self._frame_stack)
        self._last_raw = np.zeros((game.get_screen_height(), game.get_screen_width(), 3),
                                  dtype=np.uint8)
        self._layout = None
        self._current_map = None
        self._done = True

    # ------------------------------------------------------------------ API
    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def n_actions(self):
        return len(ACTIONS)

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    @property
    def metadata_dict(self) -> dict:
        return self._meta

    @property
    def scale(self) -> int:
        """Number of candidate rooms."""
        return self._scale

    @property
    def n_rooms(self) -> int:
        return self._scale

    @property
    def scenario(self) -> str:
        """Scenario name (``r8`` for the 8 room preset)."""
        return self._scenario

    @property
    def geometry(self) -> Geometry:
        """Room size of the loaded scenario (tile, cell_tiles, link_w)."""
        return Geometry.from_meta(self._meta)

    @property
    def split(self) -> str:
        return self._split

    def seed(self, seed: int | None) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, layout_id: int | None = None, target_id: int | None = None):
        """Start a new episode.

        ``layout_id`` / ``target_id`` force a particular map / target identity
        (used only by the verification scripts and evaluation sweeps); by
        default both are drawn uniformly from the configured split.
        """
        if layout_id is None:
            layout_id = int(self._rng.choice(self._layout_ids))
        if target_id is None:
            target_id = int(self._rng.choice(self._target_ids))
        self._layout = self._layouts[int(layout_id)]
        self._target_id = int(target_id)
        self._target = self._identities[self._target_id]
        self._target_room = self._layout.identity_to_room[self._target_id]
        map_name = self._meta["map_lump_pattern"].format(layout=int(layout_id),
                                                         target=int(target_id))
        if map_name != self._current_map:
            self._game.set_doom_map(map_name)
            self._current_map = map_name
        self._game.new_episode()

        self._episode_reward = 0.0
        self._episode_length = 0
        self._done = False
        self._cue_exit_step = None      # step at which the Door_Close trigger was crossed
        self._cue_sealed_step = None    # step at which the door is fully closed
        self._wrong_rooms = []
        self._visited_rooms = []
        self._seen_rooms = []
        self._target_seen_step = None
        self._first_room_seen_step = None
        self._cell_path = []
        self._revisits = 0
        self._visited_cells = set()
        self._prev_pos = None
        self._frame_buffer.clear()
        for _ in range(self._frame_stack):
            self._frame_buffer.append(
                np.zeros((self._single_c,) + self._obs_shape[1:], dtype=np.float32))
        obs = self._get_obs()
        x, y, _z, _a = self._pose()
        self._start_pos = (x, y)
        self._prev_pos = (x, y)
        self._update_cell(x, y)
        return obs

    def step(self, action):
        if self._done:
            raise RuntimeError("step() called after the episode ended; call reset() first")
        if isinstance(action, np.ndarray):
            action = int(action.reshape(-1)[0])
        elif hasattr(action, "__len__"):
            action = int(action[0])
        action = int(action)
        self._game.make_action(ACTIONS[action], self._frame_skip)
        self._episode_length += 1
        engine_done = self._game.is_episode_finished()

        x, y, z, ang = self._pose()
        lay = self._layout
        reward = self._living_reward
        moved = True
        if self._prev_pos is not None:
            moved = math.hypot(x - self._prev_pos[0], y - self._prev_pos[1]) > 1.0
        self._prev_pos = (x, y)

        # cue bookkeeping
        if self._cue_exit_step is None and x >= lay.trigger_x:
            self._cue_exit_step = self._episode_length
            self._cue_sealed_step = self._episode_length + self._door_close_steps
        if self._update_cell(x, y) and self._exploration_reward > 0.0:
            reward += self._exploration_reward

        # room entry = decision
        tx, ty = lay.tile_of(x, y)
        room_id = int(lay.room_of_tile[ty, tx])
        success = False
        wrong_now = False
        if room_id >= 0:
            if room_id not in self._visited_rooms:
                self._visited_rooms.append(room_id)
                if room_id == self._target_room:
                    success = True
                    reward += self._success_reward
                else:
                    wrong_now = True
                    self._wrong_rooms.append(room_id)
                    reward -= self._wrong_choice_penalty
            elif room_id == self._target_room:
                success = True

        visible = []
        if self._track_visibility:
            visible = lay.visible_rooms(x, y, ang)
            for r in visible:
                if r not in self._seen_rooms:
                    self._seen_rooms.append(r)
            if visible and self._first_room_seen_step is None:
                self._first_room_seen_step = self._episode_length
            if self._target_room in visible and self._target_seen_step is None:
                self._target_seen_step = self._episode_length

        done = success or (wrong_now and self._terminate_on_wrong)
        timeout = False
        if not done and (self._episode_length >= self._max_episode_steps or engine_done):
            done = True
            timeout = True
            reward -= self._timeout_penalty

        self._episode_reward += reward
        obs = self._get_obs() if not engine_done else self._stack_only()

        cell = lay.cell_of(x, y)
        cell_meta = lay.cells.get(cell)
        depth = cell_meta["depth"] if cell_meta is not None else -1
        if self._step_info == "minimal" and not done:
            info = {"moved": moved}
        else:
            info = {
                "moved": moved,
                "map_id": lay.layout_id,
                "map_seed": lay.map_seed,
                "target_id": self._target_id,
                "target_color": self._target["name"],
                "room_id": room_id,
                "visited_rooms": tuple(self._visited_rooms),
                "visible_room_ids": tuple(visible),
                "visible_room_color": tuple(lay.rooms[r]["color"] for r in visible),
                "target_visible": self._target_room in visible,
                "cue_phase": self._in_cue_phase(),
                "steps_since_cue": self._steps_since_cue(),
                "distance_from_cue": max(depth, 0),
                "distance_from_cue_units": float(math.hypot(x - lay.door_xy[0],
                                                            y - lay.door_xy[1])),
                "wrong_choices": len(self._wrong_rooms),
                "pos": (float(x), float(y), float(ang)),
            }
        if done:
            optimal = lay.rooms[self._target_room]["depth_cells"] + 1  # cell transitions from cue
            transitions = max(len(self._cell_path) - 1, 1)
            info.update({
                "reward": float(self._episode_reward),
                "length": int(self._episode_length),
                "success": 1.0 if success else 0.0,
                "truncated": bool(timeout),
                "timeout": 1.0 if timeout else 0.0,
                "wrong_choice": 1.0 if self._wrong_rooms else 0.0,
                "entered_room_id": room_id,
                "entered_room_color": lay.rooms[room_id]["color"] if room_id >= 0 else "none",
                "target_room_id": self._target_room,
                "target_wall_color": self._target["wall"],
                "target_floor_color": self._target["floor"],
                "rooms_visited": len(self._visited_rooms),
                "rooms_seen": len(self._seen_rooms),
                "seen_rooms": tuple(self._seen_rooms),
                "cells_visited": len(self._visited_cells),
                "revisits": int(self._revisits),
                "optimal_cell_path": int(optimal),
                "path_efficiency": float(min(1.0, optimal / transitions)) if success else 0.0,
                "cue_exit_step": -1 if self._cue_exit_step is None else int(self._cue_exit_step),
                "target_seen_step": (-1 if self._target_seen_step is None
                                     else int(self._target_seen_step)),
                "first_room_seen_step": (-1 if self._first_room_seen_step is None
                                         else int(self._first_room_seen_step)),
                "split": self._split,
                "start_pos": self._start_pos,
                "final_pos": (float(x), float(y)),
            })
            if success:
                # only present for successful episodes, so that averaging the
                # final infos never produces NaN
                info["time_to_goal"] = float(self._episode_length)
            self._done = True
        return obs, float(reward), bool(done), info

    def render(self):
        """Last raw RGB frame (H, W, 3) uint8 at engine resolution."""
        return self._last_raw

    def close(self):
        self._game.close()
        try:                      # engine settings file written by VizDoom; not needed again
            os.remove(self._ini_path)
        except OSError:
            pass

    # ------------------------------------------------------- privileged API
    def privileged_state(self) -> dict:
        """Ground truth for scripted controllers and analysis.  Never an observation."""
        x, y, z, ang = self._pose()
        return {
            "layout": self._layout,
            "layout_id": self._layout.layout_id,
            "target_id": self._target_id,
            "target_room": self._target_room,
            "x": x, "y": y, "z": z, "angle": ang,
            "cue_phase": self._in_cue_phase(),
        }

    # ------------------------------------------------------------- internals
    def _in_cue_phase(self) -> bool:
        return self._cue_sealed_step is None or self._episode_length < self._cue_sealed_step

    def _steps_since_cue(self) -> int:
        if self._cue_sealed_step is None:
            return 0
        return max(0, self._episode_length - self._cue_sealed_step)

    def _update_cell(self, x, y) -> bool:
        cell = self._layout.cell_of(x, y)
        new = False
        if not self._cell_path or self._cell_path[-1] != cell:
            self._cell_path.append(cell)
            if cell in self._visited_cells:
                self._revisits += 1
            else:
                new = len(self._visited_cells) > 0
            self._visited_cells.add(cell)
        return new

    def _pose(self):
        g = self._game
        return (g.get_game_variable(GameVariable.POSITION_X),
                g.get_game_variable(GameVariable.POSITION_Y),
                g.get_game_variable(GameVariable.POSITION_Z),
                g.get_game_variable(GameVariable.ANGLE))

    def _process_frame(self, screen_hwc: np.ndarray) -> np.ndarray:
        img = screen_hwc
        if self._resize:
            from PIL import Image
            h, w = self._resize
            img = np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))
        frame = img.astype(np.float32) / 255.0
        if self._grayscale:
            frame = (0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1]
                     + 0.114 * frame[:, :, 2])[:, :, np.newaxis]
        return np.ascontiguousarray(np.transpose(frame, (2, 0, 1)))

    def _stack_only(self):
        return np.concatenate(list(self._frame_buffer), axis=0)

    def _get_obs(self):
        state = self._game.get_state()
        if state is None:
            return self._stack_only()
        self._last_raw = state.screen_buffer
        self._frame_buffer.append(self._process_frame(state.screen_buffer))
        return np.concatenate(list(self._frame_buffer), axis=0)


def create_multiroom_env(config: dict, render: bool = False) -> MultiRoomColorGoalEnv:
    """Build the environment from a plain dict (e.g. the ``environment:`` section of a YAML).

    Keys: ``rooms`` (or ``scale``) and / or ``scenario`` + ``scenario_dir``, plus
    any constructor argument of :class:`MultiRoomColorGoalEnv`.
    """
    scale = config.get("rooms", config.get("scale"))
    if scale is None and config.get("scenario") is None:
        raise ValueError("MultiRoomColorGoal config needs 'rooms' (2, 4, 8, 16 or 32 for the "
                         "shipped presets) or 'scenario' (name of a generated scenario)")
    return MultiRoomColorGoalEnv(
        scale=None if scale is None else int(scale),
        scenario=config.get("scenario"),
        split=config.get("split", "train"),
        frame_skip=config.get("frame_skip", 4),
        resolution=config.get("resolution", "RES_160X120"),
        grayscale=config.get("grayscale", False),
        resize=config.get("resize", [84, 84]),
        frame_stack=config.get("frame_stack", 4),
        max_episode_steps=config.get("max_episode_steps"),
        success_reward=config.get("success_reward", 1.0),
        wrong_choice_penalty=config.get("wrong_choice_penalty", 0.0),
        timeout_penalty=config.get("timeout_penalty", 0.0),
        living_reward=config.get("living_reward", -0.0001),
        exploration_reward=config.get("exploration_reward", 0.0),
        terminate_on_wrong=config.get("terminate_on_wrong", True),
        layouts=config.get("layouts"),
        targets=config.get("targets"),
        seed=config.get("seed"),
        step_info=config.get("step_info", "full"),
        track_visibility=config.get("track_visibility", True),
        scenario_dir=config.get("scenario_dir"),
        realtime_mode=render,
    )
