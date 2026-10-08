"""VizDoom MultiRoomColorGoal: a visual memory task with a scalable number and size of rooms.

    from multiroom_color_goal import MultiRoomColorGoalEnv       # classic API
    env = MultiRoomColorGoalEnv(8)

    import gymnasium, multiroom_color_goal                       # Gymnasium API
    env = gymnasium.make("MultiRoomColorGoal-R8-v0")

    python -m multiroom_color_goal.generate --rooms 6 --cell-tiles 7 --name r6_big --out-dir DIR
    python -m multiroom_color_goal.inspect r6_big --scenario-dir DIR

The map generator needs only the standard library; the environment needs
vizdoom, gymnasium, numpy and Pillow.  Environment classes are therefore
imported lazily on first access.
"""
from .map_generator import (
    DEFAULT_GEOMETRY,
    SCALES,
    Geometry,
    ascii_map,
    build_scale,
    build_scenario,
    choose_color_scheme,
    derive_params,
    scaled_step_limit,
)

__version__ = "1.0.0"

_LAZY = {
    "ACTIONS": "multiroom_env", "ACTION_NAMES": "multiroom_env",
    "SUPPORTED_SCALES": "multiroom_env", "SCENARIO_DIR": "multiroom_env",
    "MultiRoomColorGoalEnv": "multiroom_env", "create_multiroom_env": "multiroom_env",
    "load_metadata": "multiroom_env", "resolve_scenario": "multiroom_env",
    "MultiRoomColorGoalGymEnv": "gym_env", "register_envs": "gym_env",
    "MultiRoomTrainingEnv": "wrappers", "wrap_training_env": "wrappers",
    "unwrap_multiroom": "wrappers",
    "run_scripted_episode": "scripted_policies", "WaypointController": "scripted_policies",
}

__all__ = sorted([
    "DEFAULT_GEOMETRY", "SCALES", "Geometry", "ascii_map", "build_scale", "build_scenario",
    "choose_color_scheme", "derive_params", "scaled_step_limit", "__version__", *_LAZY])


def __getattr__(name):
    if name in _LAZY:
        import importlib
        value = getattr(importlib.import_module("." + _LAZY[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__


def _register_gymnasium_ids() -> None:
    # Only the ids are registered here (entry points are strings), so importing
    # the package stays cheap and works without vizdoom installed.
    try:
        import gymnasium as gym
    except ImportError:      # generator-only installation
        return
    entry = "multiroom_color_goal.gym_env:MultiRoomColorGoalGymEnv"
    if "MultiRoomColorGoal-v0" not in gym.registry:
        gym.register(id="MultiRoomColorGoal-v0", entry_point=entry)
    for rooms in sorted(SCALES):
        env_id = "MultiRoomColorGoal-R%d-v0" % rooms
        if env_id not in gym.registry:
            gym.register(id=env_id, entry_point=entry, kwargs={"rooms": rooms})


_register_gymnasium_ids()
