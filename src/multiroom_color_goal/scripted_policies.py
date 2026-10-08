"""Scripted (privileged) controllers used only for verification, never for training.

They read ``env.privileged_state()`` - ground truth that is never part of the
observation - and therefore are NOT policies an agent could implement.

* ``oracle``     knows the map and the target room, walks the shortest path.
* ``no_memory``  same navigation, but does not know the target: it walks to a
                 candidate room drawn uniformly at random (``mode="random"``) or
                 to the nearest candidate room (``mode="nearest"``).  This is
                 what an agent can do if it retained nothing from the cue.
"""
from __future__ import annotations

import math
from collections import deque

TURN_LEFT, TURN_RIGHT, MOVE_FORWARD = 0, 1, 2


def cell_path(layout, src, dst):
    prev = {src: None}
    q = deque([src])
    while q:
        c = q.popleft()
        if c == dst:
            break
        for n in layout.adj[c]:
            if n not in prev:
                prev[n] = c
                q.append(n)
    path = [dst]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return path[::-1]


def waypoints_for(layout, src_cell, dst_cell):
    """Opening centre, cell centre, ... along the cell path.

    ``layout`` is the run-time layout from ``env.privileged_state()["layout"]``;
    its ``geom`` is the geometry of the scenario the environment has loaded, so
    the waypoints are right for any tile / cell size.
    """
    geom = layout.geom
    path = cell_path(layout, src_cell, dst_cell)
    pts = []
    for a, b in zip(path[:-1], path[1:]):
        pts.append(geom.link_center(a, b))
        pts.append(geom.cell_center(b))
    return pts


def reach_radius_for(geom) -> float:
    """Waypoint radius: 28 map units for tiles of 64 and more, 7/16 of a tile below."""
    return min(28.0, 0.4375 * geom.tile)


class WaypointController:
    """Turn-then-walk controller over a list of (x, y) waypoints."""

    def __init__(self, waypoints, turn_tolerance_deg: float = 7.5, reach_radius: float = 28.0):
        self.waypoints = list(waypoints)
        self.i = 0
        self.tol = turn_tolerance_deg
        self.radius = reach_radius

    def done(self) -> bool:
        return self.i >= len(self.waypoints)

    def act(self, x, y, angle_deg) -> int:
        while self.i < len(self.waypoints):
            wx, wy = self.waypoints[self.i]
            if math.hypot(wx - x, wy - y) < self.radius:
                self.i += 1
            else:
                break
        if self.done():
            return MOVE_FORWARD
        wx, wy = self.waypoints[self.i]
        want = math.degrees(math.atan2(wy - y, wx - x))
        diff = (want - angle_deg + 180.0) % 360.0 - 180.0
        if diff > self.tol:
            return TURN_LEFT
        if diff < -self.tol:
            return TURN_RIGHT
        return MOVE_FORWARD


def make_room_controller(env, room_id: int) -> WaypointController:
    st = env.privileged_state()
    layout = st["layout"]
    src = layout.cell_of(st["x"], st["y"])
    dst = tuple(layout.rooms[room_id]["cell"])
    return WaypointController(waypoints_for(layout, src, dst),
                              reach_radius=reach_radius_for(layout.geom))


def run_scripted_episode(env, mode: str, rng=None, layout_id=None, target_id=None,
                         on_step=None):
    """Run one episode.  mode: 'oracle' | 'no_memory_random' | 'no_memory_nearest'.

    ``on_step(env, obs, info, step_index)`` is called after reset (info=None,
    step_index=0) and after every step.  Returns the final info dict.
    """
    obs = env.reset(layout_id=layout_id, target_id=target_id)
    st = env.privileged_state()
    layout = st["layout"]
    if mode == "oracle":
        room = st["target_room"]
    elif mode == "no_memory_random":
        room = int(rng.integers(len(layout.rooms)))
    elif mode == "no_memory_nearest":
        room = min(layout.rooms, key=lambda r: (r["depth_cells"], r["room_id"]))["room_id"]
    else:
        raise ValueError(mode)
    ctrl = make_room_controller(env, room)
    if on_step is not None:
        on_step(env, obs, None, 0)
    k = 0
    while True:
        st = env.privileged_state()
        obs, _r, done, info = env.step(ctrl.act(st["x"], st["y"], st["angle"]))
        k += 1
        if on_step is not None:
            on_step(env, obs, info, k)
        if done:
            info = dict(info)
            info["chosen_room_id"] = room
            return info
