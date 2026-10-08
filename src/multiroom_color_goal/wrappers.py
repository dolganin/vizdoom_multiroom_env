"""Optional baseline wrapper around MultiRoomColorGoalEnv.

The environment itself is not changed.  The wrapper adds the two things that the
reference baselines (an upper bound and a "memory wiped" control) need from the
environment and nothing else:

1. ``oracle_cue``  - the ORACLE baseline (upper bound).  The target identity is made
   visible in every observation, so no memory is needed:
     "patch"  (default for the oracle): a small rectangle in the top-left corner of
              every stacked RGB frame is painted with the target wall colour (left
              half) and the target floor colour (right half).  Observation shape,
              model and parameter count stay identical to the memory agents.
     "planes": six constant channels (wall RGB, floor RGB) are appended instead;
              the observation becomes (3 * frame_stack + 6, H, W).
   The colours are the generator's RGB values from the scenario .json.

2. ``cue_seal_signal`` - the harder RESET-MEMORY baseline.  ``info["memory_reset"] =
   True`` is emitted exactly once per episode, ``cue_seal_reset_delay`` agent steps
   after the cue door has fully closed (default: ``frame_stack``, so that at the
   reset every frame of the stack was rendered behind the closed door).  A trainer
   that honours the flag zeroes the memory of that worker before its next forward
   pass.  The flag is present even with ``step_info: minimal``; the final info of the
   episode also carries ``memory_reset_step`` (-1 if the door never sealed).

Usage::

    env = MultiRoomTrainingEnv(MultiRoomColorGoalEnv(4), oracle_cue="patch")
    env = wrap_training_env(env, {"cue_seal_signal": True, "frame_stack": 4})

``wrap_training_env`` reads the options from a plain config dict and returns the
environment unchanged if neither key is set.  The Gymnasium adapter accepts the
same options as keyword arguments (``gymnasium.make(..., oracle_cue="patch")``).

The wrapper never changes rewards, termination or the random draws of the wrapped
environment.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

ORACLE_MODES = ("patch", "planes")


def needs_training_wrapper(config: dict) -> bool:
    return bool(config.get("oracle_cue")) or bool(config.get("cue_seal_signal"))


def target_rgb(meta: dict, target_id: int) -> tuple[np.ndarray, np.ndarray]:
    """(wall_rgb, floor_rgb) of a target identity, float32 in [0, 1]."""
    ident = meta["identities"][int(target_id)]
    colors = meta["colors"]
    wall = np.asarray(colors[ident["wall"]]["rgb"], dtype=np.float32) / 255.0
    floor = np.asarray(colors[ident["floor"]]["rgb"], dtype=np.float32) / 255.0
    return wall, floor


def paint_oracle_patch(obs: np.ndarray, wall: np.ndarray, floor: np.ndarray,
                       patch_h: int, patch_w: int) -> np.ndarray:
    """Paint wall | floor colour into the top-left corner of every stacked RGB frame."""
    out = np.array(obs, dtype=np.float32, copy=True)
    n_frames = out.shape[0] // 3
    for f in range(n_frames):
        c = 3 * f
        out[c:c + 3, :patch_h, :patch_w] = wall[:, None, None]
        out[c:c + 3, :patch_h, patch_w:2 * patch_w] = floor[:, None, None]
    return out


def append_oracle_planes(obs: np.ndarray, wall: np.ndarray, floor: np.ndarray) -> np.ndarray:
    h, w = obs.shape[1:]
    planes = np.concatenate([wall, floor]).astype(np.float32)[:, None, None]
    planes = np.broadcast_to(planes, (6, h, w))
    return np.concatenate([np.asarray(obs, dtype=np.float32), planes], axis=0)


class MultiRoomTrainingEnv:
    """Same API as MultiRoomColorGoalEnv (reset / step / spaces / max_episode_steps)."""

    def __init__(self, env, oracle_cue: str | None = None, cue_seal_signal: bool = False,
                 cue_seal_reset_delay: int = 4, oracle_patch_size: tuple = (8, 8)) -> None:
        if oracle_cue in (None, False, "", "none"):
            oracle_cue = None
        elif oracle_cue is True:
            oracle_cue = "patch"
        if oracle_cue is not None and oracle_cue not in ORACLE_MODES:
            raise ValueError(f"oracle_cue must be one of {ORACLE_MODES}, got {oracle_cue!r}")
        self._env = env
        self._oracle = oracle_cue
        self._signal = bool(cue_seal_signal)
        self._delay = max(0, int(cue_seal_reset_delay))
        self._patch_h, self._patch_w = int(oracle_patch_size[0]), int(oracle_patch_size[1])
        base = env.observation_space
        if self._oracle is not None and base.shape[0] % 3 != 0:
            raise ValueError("oracle_cue needs RGB observations (grayscale: false)")
        if self._oracle == "patch" and (self._patch_h > base.shape[1] or 2 * self._patch_w > base.shape[2]):
            raise ValueError("oracle patch does not fit into the observation")
        if self._oracle == "planes":
            self._observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(base.shape[0] + 6,) + tuple(base.shape[1:]), dtype=np.float32)
        else:
            self._observation_space = base
        self._wall = self._floor = None
        self._post_seal_steps = -1
        self._reset_step = -1
        self._t = 0

    # ------------------------------------------------------------------ API
    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def max_episode_steps(self):
        return self._env.max_episode_steps

    @property
    def unwrapped_multiroom(self):
        """The wrapped MultiRoomColorGoalEnv (analysis code needs its privileged API)."""
        return self._env

    def __getattr__(self, name):
        # only reached for attributes not defined here: delegate (n_actions, scale,
        # metadata_dict, privileged_state, seed, render, close ...)
        if name.startswith("__") or name == "_env":
            raise AttributeError(name)
        return getattr(self._env, name)

    def reset(self, layout_id: int | None = None, target_id: int | None = None):
        if layout_id is None and target_id is None:
            obs = self._env.reset()
        else:
            obs = self._env.reset(layout_id=layout_id, target_id=target_id)
        self._post_seal_steps = -1
        self._reset_step = -1
        self._t = 0
        if self._oracle is not None:
            tid = self._env.privileged_state()["target_id"]
            self._wall, self._floor = target_rgb(self._env.metadata_dict, tid)
        return self._decorate(obs)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._t += 1
        if self._signal:
            if info is None:
                info = {}
            cue_phase = info["cue_phase"] if "cue_phase" in info else self._env.privileged_state()["cue_phase"]
            if not cue_phase:
                self._post_seal_steps += 1
                if self._post_seal_steps == self._delay and not done:
                    info["memory_reset"] = True
                    self._reset_step = self._t
            if done:
                info["memory_reset_step"] = int(self._reset_step)
        return self._decorate(obs), reward, done, info

    # ------------------------------------------------------------- internals
    def _decorate(self, obs):
        if self._oracle == "patch":
            return paint_oracle_patch(obs, self._wall, self._floor, self._patch_h, self._patch_w)
        if self._oracle == "planes":
            return append_oracle_planes(obs, self._wall, self._floor)
        return obs


def wrap_training_env(env, config: dict):
    """Wrap ``env`` according to the ``environment:`` section; no-op without the keys."""
    if not needs_training_wrapper(config):
        return env
    return MultiRoomTrainingEnv(
        env,
        oracle_cue=config.get("oracle_cue"),
        cue_seal_signal=config.get("cue_seal_signal", False),
        cue_seal_reset_delay=config.get("cue_seal_reset_delay", config.get("frame_stack", 4)),
        oracle_patch_size=tuple(config.get("oracle_patch_size", (8, 8))),
    )


def unwrap_multiroom(env):
    """MultiRoomColorGoalEnv behind an optional MultiRoomTrainingEnv."""
    return getattr(env, "unwrapped_multiroom", env)
