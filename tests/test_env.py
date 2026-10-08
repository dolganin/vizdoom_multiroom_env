"""Environment tests on the shipped presets: classic API, Gymnasium API, wrappers."""
from __future__ import annotations

import os

import numpy as np

import helpers
from helpers import OBS_SHAPE, TempDir, raises

import multiroom_color_goal as mrcg
from multiroom_color_goal.map_generator import Geometry, build_scenario, derive_params
from multiroom_color_goal.multiroom_env import (
    ACTION_NAMES, SUPPORTED_SCALES, MultiRoomColorGoalEnv, create_multiroom_env, load_metadata,
    resolve_scenario)
from multiroom_color_goal.scripted_policies import run_scripted_episode
from multiroom_color_goal.wrappers import MultiRoomTrainingEnv, unwrap_multiroom, wrap_training_env

PRESET_STEP_LIMITS = {2: 300, 4: 500, 8: 800, 16: 1400, 32: 2400}


def test_classic_api_on_all_presets():
    assert SUPPORTED_SCALES == (2, 4, 8, 16, 32)
    assert ACTION_NAMES == ["TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"]
    for rooms in SUPPORTED_SCALES:
        env = MultiRoomColorGoalEnv(rooms, seed=0)
        try:
            assert env.scale == env.n_rooms == rooms and env.scenario == "r%d" % rooms
            assert env.geometry == Geometry(64, 5, 2)
            assert env.max_episode_steps == PRESET_STEP_LIMITS[rooms]
            assert env.observation_space.shape == OBS_SHAPE
            assert env.action_space.n == 3 == env.n_actions
            obs = env.reset()                      # classic API: reset returns only obs
            assert isinstance(obs, np.ndarray) and obs.shape == OBS_SHAPE
            assert obs.dtype == np.float32 and 0.0 <= obs.min() and obs.max() <= 1.0
            # frame stack: the three older frames are zero after reset, newest frame last
            assert float(np.abs(obs[:9]).max()) == 0.0 and float(obs[9:].max()) > 0.0
            out = env.step(2)
            assert len(out) == 4                   # obs, reward, done, info
            obs, reward, done, info = out
            assert isinstance(reward, float) and done is False
            assert abs(reward - (-0.0001)) < 1e-12       # living reward
            for key in ("moved", "map_id", "map_seed", "target_id", "target_color", "room_id",
                        "visited_rooms", "visible_room_ids", "visible_room_color",
                        "target_visible", "cue_phase", "steps_since_cue", "distance_from_cue",
                        "distance_from_cue_units", "wrong_choices", "pos"):
                assert key in info, key
            assert env.render().shape == (120, 160, 3) and env.render().dtype == np.uint8
        finally:
            env.close()


def test_episode_outcomes_and_rewards():
    env = MultiRoomColorGoalEnv(4, split="all", seed=1, exploration_reward=0.05,
                                wrong_choice_penalty=0.25, timeout_penalty=0.5,
                                max_episode_steps=60)
    try:
        # timeout: turning on the spot never ends the episode before the limit
        env.reset(layout_id=0, target_id=0)
        for k in range(60):
            obs, reward, done, info = env.step(0)
            assert done == (k == 59)
        assert info["timeout"] == 1.0 and info["truncated"] is True and info["success"] == 0.0
        assert info["length"] == 60 and abs(reward - (-0.0001 - 0.5)) < 1e-9
        raises(RuntimeError, env.step, 0, match="call reset() first")
    finally:
        env.close()
    env = MultiRoomColorGoalEnv(4, split="all", seed=1, exploration_reward=0.05,
                                wrong_choice_penalty=0.25)
    try:
        # success
        rewards = []
        info = run_scripted_episode(env, "oracle", layout_id=1, target_id=2,
                                    on_step=lambda e, o, i, k: rewards.append(i))
        assert info["success"] == 1.0 and info["wrong_choice"] == 0.0 and info["timeout"] == 0.0
        assert info["entered_room_id"] == info["target_room_id"]
        assert info["time_to_goal"] == info["length"] and 0.0 < info["path_efficiency"] <= 1.0
        # reward = success + living reward per step + exploration bonus per new cell
        expect = 1.0 - 0.0001 * info["length"] + 0.05 * (info["cells_visited"] - 1)
        assert abs(info["reward"] - expect) < 1e-6, (info["reward"], expect)
        assert info["cue_exit_step"] > 0 and info["split"] == "all"
        # wrong choice: walk into a room that is not the target
        env.reset(layout_id=1, target_id=2)
        lay = env.privileged_state()["layout"]
        wrong = [r["room_id"] for r in lay.rooms if r["identity"] != 2][0]
        from multiroom_color_goal.scripted_policies import make_room_controller
        ctrl = make_room_controller(env, wrong)
        done = False
        while not done:
            st = env.privileged_state()
            obs, reward, done, info = env.step(ctrl.act(st["x"], st["y"], st["angle"]))
        assert info["wrong_choice"] == 1.0 and info["success"] == 0.0 and info["timeout"] == 0.0
        assert info["entered_room_id"] == wrong and "time_to_goal" not in info
        # living reward + penalty; the bonus for the new map cell falls on the same step
        # or on the one before (the cell border is one tile in front of the room)
        assert min(abs(reward - (-0.0001 - 0.25 + bonus)) for bonus in (0.0, 0.05)) < 1e-9
    finally:
        env.close()


def test_splits_and_seeding():
    meta = load_metadata(8)
    sp = meta["splits"]
    assert sp["train"]["layouts"] == [0, 1, 2, 3] and sp["eval_maps"]["layouts"] == [4, 5]
    held = sp["eval_colors"]["targets"]
    assert len(held) == 2 and not set(held) & set(sp["train"]["targets"])
    for split in ("train", "eval_maps", "eval_colors", "eval_maps_colors", "eval_all", "all"):
        env = MultiRoomColorGoalEnv(8, split=split, seed=3)
        try:
            seen = set()
            for _ in range(12):
                env.reset()
                st = env.privileged_state()
                seen.add((st["layout_id"], st["target_id"]))
                assert st["layout_id"] in sp[split]["layouts"]
                assert st["target_id"] in sp[split]["targets"]
            assert len(seen) > 1
        finally:
            env.close()

    def draws(seed):
        env = MultiRoomColorGoalEnv(8, seed=seed)
        try:
            out = []
            for _ in range(8):
                env.reset()
                st = env.privileged_state()
                out.append((st["layout_id"], st["target_id"]))
            return out
        finally:
            env.close()
    assert draws(5) == draws(5) and draws(5) != draws(6)
    # the two eval-colour splits are empty for scenarios without held-out targets
    raises(ValueError, MultiRoomColorGoalEnv, 2, split="eval_colors", match="is empty")
    raises(ValueError, MultiRoomColorGoalEnv, 2, split="nope", match="unknown split")
    raises(ValueError, MultiRoomColorGoalEnv, 2, layouts=[9], match="has layouts")


def test_scenario_lookup_and_errors():
    assert resolve_scenario(8)[1] == "multiroom_color_goal_r8"
    assert resolve_scenario("r8")[1] == "multiroom_color_goal_r8"
    assert resolve_scenario("multiroom_color_goal_r8.json")[1] == "multiroom_color_goal_r8"
    msg = raises(FileNotFoundError, MultiRoomColorGoalEnv, 6, match="multiroom_color_goal.generate")
    assert "r16" in msg and "scenario 'r6' not found" in msg
    raises(FileNotFoundError, MultiRoomColorGoalEnv, scenario="nope")
    raises(ValueError, create_multiroom_env, {}, match="needs 'rooms'")
    with TempDir() as d:
        geom = Geometry(48, 6)
        params = derive_params(3, geom, n_train_layouts=1, n_eval_layouts=1)
        build_scenario(params, d, "r3_small", geom, verbose=False)
        build_scenario(derive_params(3, n_train_layouts=1, n_eval_layouts=1), d, verbose=False)
        ways = [
            dict(scenario="r3_small", scenario_dir=d),
            dict(scenario="multiroom_color_goal_r3_small", scenario_dir=d),
            dict(scenario=os.path.join(d, "r3_small")),
            dict(scenario=os.path.join(d, "multiroom_color_goal_r3_small.json")),
            dict(scenario=os.path.join(d, "multiroom_color_goal_r3_small.wad")),
        ]
        for kw in ways:
            env = MultiRoomColorGoalEnv(**kw)
            try:
                assert env.scenario == "r3_small" and env.n_rooms == 3
                assert env.geometry == geom
                assert env.max_episode_steps == params["max_episode_steps"]
                assert env.reset().shape == OBS_SHAPE
            finally:
                env.close()
        # rooms=N finds a generated scenario with the default name in scenario_dir
        env = create_multiroom_env({"rooms": 3, "scenario_dir": d, "split": "eval_maps"})
        try:
            assert env.scenario == "r3" and env.split == "eval_maps"
            info = run_scripted_episode(env, "oracle")
            assert info["success"] == 1.0 and info["map_id"] == 1
        finally:
            env.close()
        env = create_multiroom_env({"scenario": "r3_small", "scenario_dir": d, "frame_stack": 1,
                                    "grayscale": True, "resize": None})
        try:
            assert env.reset().shape == (1, 120, 160)
        finally:
            env.close()


def test_gymnasium_api():
    import gymnasium as gym
    ids = mrcg.register_envs()
    assert ids == ["MultiRoomColorGoal-v0"] + ["MultiRoomColorGoal-R%d-v0" % n
                                               for n in (2, 4, 8, 16, 32)]
    for env_id in ids:
        assert env_id in gym.registry
    env = gym.make("MultiRoomColorGoal-R4-v0", split="all")
    try:
        core = env.unwrapped
        assert isinstance(core, mrcg.MultiRoomColorGoalGymEnv) and isinstance(core, gym.Env)
        assert core.classic_env.n_rooms == 4 and core.max_episode_steps == 500
        assert env.observation_space.shape == OBS_SHAPE and env.action_space.n == 3
        obs, info = env.reset(seed=11)
        assert env.observation_space.contains(obs)
        assert {"map_id", "map_seed", "target_id", "target_color", "cue_phase"} <= set(info)
        first = (info["map_id"], info["target_id"])
        out = env.step(env.action_space.sample())
        assert len(out) == 5
        obs, reward, terminated, truncated, info = out
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float) and terminated is False and truncated is False
        # reset(seed=...) reproduces the draw; forced map / target through options
        obs, info = env.reset(seed=11)
        assert (info["map_id"], info["target_id"]) == first
        obs, info = env.reset(options={"layout_id": 5, "target_id": 3})
        assert (info["map_id"], info["target_id"], info["target_color"]) == (5, 3, "yellow")
        raises(ValueError, env.reset, options={"level": 1}, match="unknown reset options")
    finally:
        env.close()

    # success and wrong choice terminate, the step limit truncates
    env = mrcg.MultiRoomColorGoalGymEnv(rooms=2, split="all", max_episode_steps=40)
    try:
        env.reset(seed=0)
        for k in range(40):
            obs, reward, terminated, truncated, info = env.step(1)
            assert terminated is False and truncated == (k == 39)
        assert info["timeout"] == 1.0 and info["success"] == 0.0 and info["length"] == 40
    finally:
        env.close()
    # gym.make(max_episode_steps=N) is Gymnasium's own TimeLimit wrapper: it truncates too,
    # but the environment's limit (and its end-of-episode info) is untouched
    env = gym.make("MultiRoomColorGoal-v0", rooms=2, split="all", max_episode_steps=25)
    try:
        env.reset(seed=0)
        for k in range(25):
            obs, reward, terminated, truncated, info = env.step(1)
            assert terminated is False and truncated == (k == 24)
        assert env.unwrapped.max_episode_steps == 300 and "timeout" not in info
    finally:
        env.close()
    env = mrcg.MultiRoomColorGoalGymEnv(rooms=2, split="all", render_mode="rgb_array")
    try:
        from multiroom_color_goal.scripted_policies import make_room_controller
        for goal, key in ((0, "success"), (1, "wrong_choice")):
            env.reset(options={"layout_id": 0, "target_id": 0})
            lay = env.classic_env.privileged_state()["layout"]
            room = lay.identity_to_room[goal]
            ctrl = make_room_controller(env.classic_env, room)
            terminated = truncated = False
            while not (terminated or truncated):
                st = env.classic_env.privileged_state()
                obs, reward, terminated, truncated, info = env.step(
                    ctrl.act(st["x"], st["y"], st["angle"]))
            assert terminated is True and truncated is False and info[key] == 1.0
            assert (reward > 0.9) == (key == "success")
        frame = env.render()
        assert frame.shape == (120, 160, 3) and frame.dtype == np.uint8
    finally:
        env.close()
    raises(ValueError, mrcg.MultiRoomColorGoalGymEnv, rooms=2, render_mode="human")

    # Gymnasium's own conformance checker
    from gymnasium.utils.env_checker import check_env
    import warnings
    env = mrcg.MultiRoomColorGoalGymEnv(rooms=2)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_gymnasium_custom_scenario():
    import gymnasium as gym
    with TempDir() as d:
        geom = Geometry(64, 7)
        build_scenario(derive_params(5, geom, n_train_layouts=1, n_eval_layouts=1), d, "r5_big",
                       geom, verbose=False)
        env = gym.make("MultiRoomColorGoal-v0", scenario="r5_big", scenario_dir=d)
        try:
            obs, info = env.reset(seed=0)
            assert obs.shape == OBS_SHAPE and env.unwrapped.classic_env.n_rooms == 5
            assert env.unwrapped.max_episode_steps == 700   # 580 * (0.5 + 0.5 * 448 / 320)
        finally:
            env.close()


def test_baseline_wrappers():
    meta = load_metadata(16)
    base = MultiRoomColorGoalEnv(16, split="all", seed=0)
    try:
        assert wrap_training_env(base, {}) is base
        # oracle patch: target wall | floor colour in the top-left corner of every frame
        env = wrap_training_env(base, {"oracle_cue": "patch"})
        assert isinstance(env, MultiRoomTrainingEnv) and unwrap_multiroom(env) is base
        assert env.observation_space.shape == OBS_SHAPE and env.max_episode_steps == 1400
        obs = env.reset(layout_id=0, target_id=6)       # identity 6 = green wall / blue floor
        ident = meta["identities"][6]
        wall = np.asarray(meta["colors"][ident["wall"]]["rgb"], np.float32) / 255.0
        floor = np.asarray(meta["colors"][ident["floor"]]["rgb"], np.float32) / 255.0
        assert ident["name"] == "green/blue"
        for f in range(4):
            assert np.allclose(obs[3 * f:3 * f + 3, :8, :8], wall[:, None, None])
            assert np.allclose(obs[3 * f:3 * f + 3, :8, 8:16], floor[:, None, None])
        obs, reward, done, info = env.step(2)
        assert np.allclose(obs[9:12, :8, :8], wall[:, None, None])
        # oracle planes: six constant channels appended
        env = MultiRoomTrainingEnv(base, oracle_cue="planes")
        assert env.observation_space.shape == (18, 84, 84)
        obs = env.reset(layout_id=0, target_id=6)
        assert obs.shape == (18, 84, 84)
        assert np.allclose(obs[12:15, 0, 0], wall) and np.allclose(obs[15:18, -1, -1], floor)
        raises(ValueError, MultiRoomTrainingEnv, base, oracle_cue="bogus")

        # memory-reset signal: exactly once, `delay` steps after the door has closed
        env = MultiRoomTrainingEnv(base, cue_seal_signal=True, cue_seal_reset_delay=4)
        flags = []
        info = run_scripted_episode(
            env, "oracle", layout_id=2, target_id=1,
            on_step=lambda e, o, i, k: flags.append((k, bool(i and i.get("memory_reset")),
                                                     None if i is None else i["cue_phase"])))
        assert info["success"] == 1.0
        resets = [k for k, flag, _ in flags if flag]
        sealed = min(k for k, _, cue in flags if cue is False)
        assert resets == [sealed + 4] and info["memory_reset_step"] == sealed + 4
    finally:
        base.close()

    # the same options through the Gymnasium adapter
    import gymnasium as gym
    env = gym.make("MultiRoomColorGoal-R16-v0", oracle_cue="patch", cue_seal_signal=True)
    try:
        obs, info = env.reset(seed=0, options={"layout_id": 0, "target_id": 6})
        assert np.allclose(obs[9:12, :8, :8], wall[:, None, None])
    finally:
        env.close()
