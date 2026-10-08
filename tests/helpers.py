"""Shared helpers of the test suite (not a test module)."""
from __future__ import annotations

import hashlib
import math
import os
import shutil
import sys
import tempfile

# The tests run against the source tree they live in (no installation needed).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ.setdefault("OMP_NUM_THREADS", "2")

SAT_THRESHOLD = 40   # max(R,G,B) - min(R,G,B) above this = coloured pixel (the maze is pure grey)
OBS_SHAPE = (12, 84, 84)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TempDir:
    """``with TempDir() as d`` - a directory that is removed afterwards."""

    def __enter__(self) -> str:
        self.path = tempfile.mkdtemp(prefix="mrcg_test_")
        return self.path

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def raises(exc_type, fn, *args, match: str | None = None, **kwargs) -> str:
    """Assert that ``fn(*args, **kwargs)`` raises ``exc_type``; returns the message."""
    try:
        fn(*args, **kwargs)
    except exc_type as e:
        msg = str(e)
        if match is not None:
            assert match in msg, f"expected {match!r} in the error message, got: {msg}"
        return msg
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc_type.__name__}")


def sat_pixels(rgb_hwc) -> int:
    a = rgb_hwc.astype("int16")
    return int(((a.max(axis=2) - a.min(axis=2)) > SAT_THRESHOLD).sum())


def _face(env, angle_deg, max_steps=40):
    from multiroom_color_goal.scripted_policies import TURN_LEFT, TURN_RIGHT
    for _ in range(max_steps):
        st = env.privileged_state()
        diff = (angle_deg - st["angle"] + 180.0) % 360.0 - 180.0
        if abs(diff) <= 7.5:
            return
        env.step(TURN_LEFT if diff > 0 else TURN_RIGHT)
    raise AssertionError("could not turn to %.0f degrees" % angle_deg)


def _walk(env, waypoints, radius, stop=None, max_steps=400):
    from multiroom_color_goal.scripted_policies import WaypointController
    ctrl = WaypointController(waypoints, reach_radius=radius)
    for _ in range(max_steps):
        st = env.privileged_state()
        ctrl.act(st["x"], st["y"], st["angle"])          # advances the waypoint index
        if ctrl.done() or (stop is not None and stop(st)):
            return
        env.step(ctrl.act(st["x"], st["y"], st["angle"]))
    raise AssertionError("scripted walk did not reach its goal")


def _creep(env, waypoints, stop, max_steps=600):
    """Walk in single-step pulses: MOVE only when (almost) standing and aligned.

    A Doom player keeps sliding for ~50 map units after a run; pulsing keeps
    every advance below ~25 units, so the agent can be parked inside a window
    that is narrower than its normal stopping distance.
    """
    import math
    from multiroom_color_goal.scripted_policies import MOVE_FORWARD, TURN_LEFT, TURN_RIGHT
    i = 0
    prev = None
    for _ in range(max_steps):
        st = env.privileged_state()
        x, y = st["x"], st["y"]
        speed = 0.0 if prev is None else math.hypot(x - prev[0], y - prev[1])
        prev = (x, y)
        if stop(st) and speed < 1.0:
            return
        while i < len(waypoints) - 1 and x >= waypoints[i][0] - 4.0:
            i += 1
        wx, wy = waypoints[i]
        diff = (math.degrees(math.atan2(wy - y, wx - x)) - st["angle"] + 180.0) % 360.0 - 180.0
        if abs(diff) > 7.5:
            env.step(TURN_LEFT if diff > 0 else TURN_RIGHT)
        elif speed >= 1.0 or stop(st):
            env.step(TURN_LEFT)          # coast; the heading is corrected on the next step
        else:
            env.step(MOVE_FORWARD)
    raise AssertionError("scripted creep did not reach its goal")


def check_cue_seal(env) -> dict:
    """Engine-level check of the cue mechanism on every layout of the scenario.

    Before the trigger line the cue colour is visible when looking back; after
    it the door is closed: no coloured pixel when looking back, while walking
    back towards the cue room for 40 steps and during a full turn; the agent
    cannot climb the ledge (stays >= 15 units east of it, on the low floor).
    """
    from multiroom_color_goal.scripted_policies import MOVE_FORWARD, TURN_LEFT
    meta = env.metadata_dict
    n_id = len(meta["identities"])
    out = {"open_px_min": 10 ** 9, "closed_px_max": 0, "min_x_past_ledge": float("inf"),
           "max_z_after_drop": 0.0}
    for lm in meta["layouts"]:
        layout_id = lm["layout_id"]
        env.reset(layout_id=layout_id, target_id=(layout_id * 3 + 1) % n_id)
        lay = env.privileged_state()["layout"]
        geom = lay.geom
        door = lay.door_xy
        cy = door[1]
        first = env.render()
        assert sat_pixels(first) > 0.5 * first.shape[0] * first.shape[1], \
            "the cue colour must dominate the first frame"
        # 1) drop into the first airlock columns and stop before the trigger
        #    line (a window of 2 tiles - 16 units), look back west
        _walk(env, [door], 4.0, stop=lambda s: s["x"] >= lay.ledge_x - 110.0)
        _creep(env, [door, (lay.trigger_x + geom.tile, cy)],
               stop=lambda s: s["x"] >= lay.ledge_x + 18.0)
        _face(env, 180.0)
        st = env.privileged_state()
        assert lay.ledge_x + 16.0 <= st["x"] < lay.trigger_x and st["z"] == 0.0, \
            "could not park between the ledge and the trigger line"
        assert st["cue_phase"], "trigger fired too early"
        out["open_px_min"] = min(out["open_px_min"], sat_pixels(env.render()))
        # 2) cross the trigger line, wait for the door, look back west
        _walk(env, [(lay.trigger_x + geom.tile, cy)], 4.0,
              stop=lambda s: s["x"] >= lay.trigger_x + 2.0)
        for _ in range(4):
            env.step(TURN_LEFT)
        _face(env, 180.0)
        assert not env.privileged_state()["cue_phase"], "cue phase did not end"
        out["closed_px_max"] = max(out["closed_px_max"], sat_pixels(env.render()))
        # 3) try to walk back into the cue room
        for _ in range(40):
            env.step(MOVE_FORWARD)
            st = env.privileged_state()
            out["min_x_past_ledge"] = min(out["min_x_past_ledge"], st["x"] - lay.ledge_x)
            out["max_z_after_drop"] = max(out["max_z_after_drop"], st["z"])
            out["closed_px_max"] = max(out["closed_px_max"], sat_pixels(env.render()))
        # 4) full turn inside the airlock: neither the cue nor a candidate room in view
        for _ in range(27):
            env.step(TURN_LEFT)
            out["closed_px_max"] = max(out["closed_px_max"], sat_pixels(env.render()))
    assert out["open_px_min"] > 0, "cue not visible through the open door"
    assert out["closed_px_max"] == 0, \
        "coloured pixels visible after the door closed: %d" % out["closed_px_max"]
    assert out["min_x_past_ledge"] >= 15.0, "agent got back onto the ledge"
    assert out["max_z_after_drop"] == 0.0, "agent climbed back up"
    return out


def check_scenario(scenario_dir: str, name: str, random_episodes: int = 96) -> dict:
    """All VizDoom-level checks for one scenario; returns the measured numbers."""
    import numpy as np
    from multiroom_color_goal.multiroom_env import MultiRoomColorGoalEnv
    from multiroom_color_goal.scripted_policies import run_scripted_episode

    env = MultiRoomColorGoalEnv(scenario=name, scenario_dir=scenario_dir, split="all", seed=0)
    try:
        meta = env.metadata_dict
        n = meta["scale"]
        limit = env.max_episode_steps
        assert limit == meta["params"]["max_episode_steps"]
        layouts = [m["layout_id"] for m in meta["layouts"]]
        targets = list(range(n))
        assert len(meta["identities"]) == n
        # generator-side proof obligation
        assert all(m["cue_los_violations"] == 0 for m in meta["layouts"])
        assert all(m["cue_portal_los_violations"] == 0 for m in meta["layouts"])

        # the map loads and the first observation is as documented
        obs = env.reset()
        assert obs.shape == OBS_SHAPE == env.observation_space.shape, obs.shape
        assert obs.dtype == np.float32 and 0.0 <= obs.min() and obs.max() <= 1.0
        assert env.observation_space.contains(obs)
        assert float(obs[-3:].max()) > 0.0, "black first frame"
        obs2, reward, done, info = env.step(2)
        assert obs2.shape == OBS_SHAPE and not done and info["cue_phase"]

        seal = check_cue_seal(env)

        # oracle: every (layout, target) must succeed within the step limit; on
        # the way no candidate room may be visible from the cue zone
        lengths = []
        cue_zone = {ord(c) for c in "cdab"}

        def no_room_from_cue_zone(e, _obs, step_info, _k):
            if step_info is None:
                return
            st = e.privileged_state()
            tx, ty = st["layout"].tile_of(st["x"], st["y"])
            if int(st["layout"].kind[ty, tx]) in cue_zone:
                assert step_info["visible_room_ids"] == (), \
                    "candidate room visible from the cue zone at %r" % (step_info["pos"],)

        for li in layouts:
            for ti in targets:
                info = run_scripted_episode(env, "oracle", layout_id=li, target_id=ti,
                                            on_step=no_room_from_cue_zone)
                assert info["success"] == 1.0 and info["timeout"] == 0.0, \
                    "oracle failed: layout %d target %d: %r" % (li, ti, {
                        k: info[k] for k in ("length", "timeout", "wrong_choice", "final_pos")})
                assert info["length"] <= limit
                assert 0 < info["cue_exit_step"] < info["length"]
                lengths.append(info["length"])

        # memoryless, deterministic: always the nearest room -> exactly one
        # target per layout is right -> success exactly 1 / n_rooms
        hits = 0
        for li in layouts:
            for ti in targets:
                info = run_scripted_episode(env, "no_memory_nearest", layout_id=li, target_id=ti)
                assert info["timeout"] == 0.0
                assert info["success"] == 1.0 or info["wrong_choice"] == 1.0
                hits += int(info["success"])
        assert hits == len(layouts), "nearest-room policy: %d hits, expected %d" % (
            hits, len(layouts))

        # memoryless, random room (fixed seed): about 1 / n_rooms
        rng = np.random.default_rng(12345)
        pairs = [(li, ti) for li in layouts for ti in targets]
        reps = int(math.ceil(random_episodes / float(len(pairs))))
        rnd_hits = rnd_n = 0
        for _ in range(reps):
            for li, ti in pairs:
                info = run_scripted_episode(env, "no_memory_random", rng=rng, layout_id=li,
                                            target_id=ti)
                rnd_hits += int(info["success"])
                rnd_n += 1
        p = 1.0 / n
        tol = 4.0 * math.sqrt(p * (1.0 - p) / rnd_n)
        assert abs(rnd_hits / float(rnd_n) - p) <= tol, \
            "random-room policy success %.3f, expected %.3f +- %.3f" % (rnd_hits / rnd_n, p, tol)
        return {
            "rooms": n, "cell_tiles": meta["cell_tiles"], "tile": meta["tile"],
            "link_w": meta["link_w"], "layouts": len(layouts),
            "oracle_episodes": len(lengths), "oracle_success": 1.0,
            "oracle_len_mean": sum(lengths) / float(len(lengths)),
            "oracle_len_max": max(lengths), "step_limit": limit,
            "nearest_success": hits / float(len(pairs)),
            "random_success": rnd_hits / float(rnd_n), "random_episodes": rnd_n,
            "chance": p, "seal": seal,
        }
    finally:
        env.close()


def format_row(r: dict) -> str:
    return ("rooms %2d cell_tiles %d tile %3d | oracle %3d/%-3d ok, len mean %6.1f max %4d of "
            "limit %4d (%2.0f%%) | memoryless nearest %.3f random %.3f (n=%d) vs 1/n %.3f" % (
                r["rooms"], r["cell_tiles"], r["tile"], r["oracle_episodes"],
                r["oracle_episodes"], r["oracle_len_mean"], r["oracle_len_max"],
                r["step_limit"], 100.0 * r["oracle_len_max"] / r["step_limit"],
                r["nearest_success"], r["random_success"], r["random_episodes"], r["chance"]))
