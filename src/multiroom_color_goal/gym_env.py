"""Gymnasium adapter for MultiRoomColorGoal.

    import gymnasium as gym
    import multiroom_color_goal                      # registers the ids

    env = gym.make("MultiRoomColorGoal-R8-v0")                       # preset
    env = gym.make("MultiRoomColorGoal-v0", rooms=4, split="eval_maps")
    env = gym.make("MultiRoomColorGoal-v0", scenario="r6_big", scenario_dir="my_scenarios")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

``terminated`` is set when the agent enters a candidate room (success, or a
wrong choice with ``terminate_on_wrong=True``); ``truncated`` when the step
limit is reached.  The adapter is a thin shell around the classic
:class:`~multiroom_color_goal.multiroom_env.MultiRoomColorGoalEnv`, which is
available as ``env.unwrapped.classic_env``; observations, rewards and ``info``
are exactly those of the classic environment.
"""
from __future__ import annotations

import gymnasium as gym

from .multiroom_env import SUPPORTED_SCALES, MultiRoomColorGoalEnv
from .wrappers import MultiRoomTrainingEnv

GENERIC_ID = "MultiRoomColorGoal-v0"
PRESET_ID = "MultiRoomColorGoal-R{rooms}-v0"


class MultiRoomColorGoalGymEnv(gym.Env):
    """MultiRoomColorGoal with the Gymnasium API.

    Parameters
    ----------
    rooms : int, optional
        Number of candidate rooms: a shipped preset (2, 4, 8, 16, 32) or, with
        ``scenario_dir``, a generated scenario named ``r<rooms>``.  Default 8
        if ``scenario`` is not given either.
    scenario : str, optional
        Name or path of a generated scenario (takes precedence over ``rooms``).
    scenario_dir : str, optional
        Directory with the scenario files (default: the packaged presets).
    render_mode : None or "rgb_array"
        ``render()`` returns the last raw engine frame (H, W, 3) uint8.
    oracle_cue, cue_seal_signal, cue_seal_reset_delay, oracle_patch_size
        Optional baselines, see :mod:`multiroom_color_goal.wrappers`.
    **env_kwargs
        Passed to :class:`MultiRoomColorGoalEnv` (split, frame_skip, resize,
        frame_stack, max_episode_steps, rewards, layouts, targets, seed, ...).

    ``reset(options={"layout_id": i, "target_id": j})`` forces a particular
    map / target (for evaluation sweeps).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 9}

    def __init__(self, rooms: int | None = None, scenario: str | None = None,
                 scenario_dir: str | None = None, render_mode: str | None = None,
                 oracle_cue: str | None = None, cue_seal_signal: bool = False,
                 cue_seal_reset_delay: int | None = None, oracle_patch_size: tuple = (8, 8),
                 **env_kwargs) -> None:
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"render_mode must be None or 'rgb_array', got {render_mode!r}")
        if "scale" in env_kwargs:
            if rooms is not None:
                raise ValueError("give either rooms or scale, not both")
            rooms = env_kwargs.pop("scale")
        if rooms is None and scenario is None:
            rooms = 8
        self.render_mode = render_mode
        self.classic_env = MultiRoomColorGoalEnv(
            scale=rooms, scenario=scenario, scenario_dir=scenario_dir, **env_kwargs)
        self._env = self.classic_env
        if oracle_cue or cue_seal_signal:
            if cue_seal_reset_delay is None:
                cue_seal_reset_delay = int(env_kwargs.get("frame_stack", 4))
            self._env = MultiRoomTrainingEnv(
                self.classic_env, oracle_cue=oracle_cue, cue_seal_signal=cue_seal_signal,
                cue_seal_reset_delay=cue_seal_reset_delay, oracle_patch_size=oracle_patch_size)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    @property
    def max_episode_steps(self) -> int:
        return self.classic_env.max_episode_steps

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.classic_env.seed(seed)
        options = options or {}
        unknown = sorted(set(options) - {"layout_id", "target_id"})
        if unknown:
            raise ValueError(f"unknown reset options {unknown}; valid: layout_id, target_id")
        obs = self._env.reset(layout_id=options.get("layout_id"),
                              target_id=options.get("target_id"))
        st = self.classic_env.privileged_state()
        info = {
            "map_id": st["layout_id"],
            "map_seed": st["layout"].map_seed,
            "target_id": st["target_id"],
            "target_color": self.classic_env.metadata_dict["identities"][st["target_id"]]["name"],
            "cue_phase": True,
        }
        return obs, info

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        truncated = bool(done and info.get("timeout", 0.0))
        terminated = bool(done and not truncated)
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.classic_env.render().copy()
        return None

    def close(self):
        if getattr(self, "classic_env", None) is not None:
            self.classic_env.close()
            self.classic_env = None


def register_envs() -> list:
    """Register the Gymnasium ids (idempotent; also done on package import); returns the ids."""
    from . import _register_gymnasium_ids
    _register_gymnasium_ids()
    return [GENERIC_ID] + [PRESET_ID.format(rooms=r) for r in SUPPORTED_SCALES]
