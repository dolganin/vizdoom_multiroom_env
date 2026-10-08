"""Generator tests: need only the Python standard library (no VizDoom)."""
from __future__ import annotations

import contextlib
import io
import json
import os

import helpers
from helpers import TempDir, raises, sha256

from multiroom_color_goal import generate
from multiroom_color_goal.map_generator import (
    DEFAULT_GEOMETRY, SCALES, Geometry, build_identities, build_scale, build_scenario,
    choose_color_scheme, derive_params, generate_layout, scaled_step_limit, validate_params,
)

SHIPPED = os.path.join(helpers.SRC, "multiroom_color_goal", "scenarios")

# sha256 of the WAD files of the original project.  Never update these values:
# they are the backward compatibility contract of the generator.
PRESET_WAD_SHA256 = {
    2: "067913354d6fe3d5cf29eed2d664e2d78ea095269844bed2552b048c79baf5b0",
    4: "3ed266baf80cbb9edfed334e99d860010d8e8bb133093020288149a8c182417f",
    8: "48da51a7f026c198df66cc4a4fac119a155859c81ec62dfa6e5b507fb9593f9f",
    16: "e27897586e38b25e0a5296232e00b34ca57486f44ed6aaee59ebac3d0422d764",
    32: "e13fbd69c560185362c23004321f59494eaaca1f7b754a5cc79474f46b27f1c5",
}
# Keys the side-car .json of the original project had; their values must not change.
ORIGINAL_JSON_KEYS = {
    "family", "generator_version", "scale", "wad", "cfg", "tile", "cell_tiles", "content_tiles",
    "link_lo", "link_w", "cue_raise", "door_tag", "door_speed", "door_close_tics", "colors",
    "color_scheme", "identities", "params", "map_lump_pattern", "layouts", "splits"}


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


def test_shipped_preset_wads_have_the_original_hashes():
    for n, digest in PRESET_WAD_SHA256.items():
        assert sha256(os.path.join(SHIPPED, "multiroom_color_goal_r%d.wad" % n)) == digest, n


def test_presets_regenerate_byte_identical():
    with TempDir() as d:
        for n, digest in PRESET_WAD_SHA256.items():
            meta = build_scale(n, d, verbose=False)
            stem = "multiroom_color_goal_r%d" % n
            assert sha256(os.path.join(d, stem + ".wad")) == digest, "WAD of r%d changed" % n
            # .json and .cfg are regenerated identically as well
            for ext in (".json", ".cfg"):
                with open(os.path.join(d, stem + ext), "rb") as a, \
                        open(os.path.join(SHIPPED, stem + ext), "rb") as b:
                    assert a.read() == b.read(), stem + ext
            assert ORIGINAL_JSON_KEYS <= set(meta)
            assert meta["params"] == SCALES[n]
            assert (meta["tile"], meta["cell_tiles"], meta["content_tiles"], meta["link_lo"],
                    meta["link_w"]) == (64, 5, 4, 2, 2)


def test_presets_are_frozen():
    expect = {
        2: (2, None, 3, 2, 0.0, 0, 0, 300, 0),
        4: (4, None, 6, 2, 0.3, 1, 0, 500, 0),
        8: (8, None, 12, 3, 0.35, 2, 1, 800, 2),
        16: (4, 4, 22, 3, 0.4, 3, 2, 1400, 4),
        32: (8, 4, 40, 4, 0.45, 4, 4, 2400, 8),
    }
    assert sorted(SCALES) == sorted(expect)
    for n, e in expect.items():
        p = SCALES[n]
        assert (p["wall_colors"], p["floor_colors"], p["n_corridor"], p["lead_in"],
                p["branch_prob"], p["n_dead_ends"], p["n_distractors"], p["max_episode_steps"],
                p["heldout_targets"]) == e, n
        assert p["train_seeds"] == [n * 1000 + i for i in (1, 2, 3, 4)]
        assert p["eval_seeds"] == [n * 1000 + 901, n * 1000 + 902]


def test_derive_params_reproduces_presets():
    for n in SCALES:
        assert derive_params(n) == SCALES[n], n


def test_derive_params_trend_and_overrides():
    prev = None
    for n in range(2, 65):
        p = derive_params(n)
        validate_params(p)
        assert p["n_rooms"] == n
        assert len(build_identities(p["wall_colors"], p["floor_colors"], n)) == n
        assert (p["floor_colors"] is None) == (n <= 8), n
        if prev is not None:   # monotone in the number of rooms
            for k in ("n_corridor", "lead_in", "n_dead_ends", "n_distractors",
                      "max_episode_steps", "heldout_targets"):
                assert p[k] >= prev[k], (n, k)
            # more forks with more rooms up to the largest preset; fixed 0.25 beyond
            assert p["branch_prob"] >= prev["branch_prob"] or n == 33, n
        assert p["n_corridor"] >= n and 0 <= p["heldout_targets"] < n
        prev = p
    p = derive_params(6, lead_in=5, n_distractors=3, max_episode_steps=999,
                      train_seeds=[7, 8], eval_seeds=[9])
    assert (p["lead_in"], p["n_distractors"], p["max_episode_steps"]) == (5, 3, 999)
    assert p["n_corridor"] == derive_params(6)["n_corridor"] + 2   # room for the longer lead-in
    assert p["train_seeds"] == [7, 8] and p["eval_seeds"] == [9]
    p = derive_params(6, n_train_layouts=3, n_eval_layouts=1, seed_base=500)
    assert p["train_seeds"] == [501, 502, 503] and p["eval_seeds"] == [1401]


def test_color_scheme():
    for n in range(2, 9):
        assert choose_color_scheme(n) == (n, None)
    assert choose_color_scheme(16) == (4, 4) and choose_color_scheme(32) == (8, 4)
    assert choose_color_scheme(12) == (3, 4) and choose_color_scheme(24) == (6, 4)
    assert choose_color_scheme(9) == (3, 3) and choose_color_scheme(64) == (8, 8)
    for n in range(9, 65):
        w, f = choose_color_scheme(n)
        ids = build_identities(w, f, n)
        assert len({(i["wall"], i["floor"]) for i in ids}) == n     # all identities distinct
        assert [i["id"] for i in ids] == list(range(n))
        assert w * f - n < f, n                                     # no unused wall colour


def test_geometry_link_is_centred():
    g = DEFAULT_GEOMETRY
    assert (g.tile, g.cell_tiles, g.content, g.link_w, g.link_lo, g.cell) == (64, 5, 4, 2, 2, 320)
    for ct in range(4, 10):
        for lw in range(1, ct):
            g = Geometry(64, ct, lw)
            lo, hi = g.link_lo, g.link_lo + lw - 1          # first / last opening tile
            assert 1 <= lo and hi <= g.content              # inside the content tiles 1..content
            free_low, free_high = lo - 1, g.content - hi
            assert free_high - free_low in (0, 1), (ct, lw)  # centred up to half a tile
    # coordinates of the default geometry (the values the presets were built with)
    assert DEFAULT_GEOMETRY.cell_center((0, 0)) == (192.0, 192.0)
    assert DEFAULT_GEOMETRY.link_center((0, 0), (1, 0)) == (352.0, 192.0)
    assert DEFAULT_GEOMETRY.link_center((1, 1), (1, 0)) == (512.0, 352.0)
    g = Geometry(48, 7)
    assert g.cell_center((1, 2)) == ((7 + 1 + 3) * 48.0, (14 + 1 + 3) * 48.0)
    assert g.link_tiles((0, 0), (1, 0)) == [(7, 3), (7, 4)]


def test_geometry_validation_errors():
    raises(ValueError, Geometry, cell_tiles=3, match="cell_tiles must be in [4, 9]")
    raises(ValueError, Geometry, cell_tiles=10, match="cell_tiles must be in [4, 9]")
    raises(ValueError, Geometry, tile=31, match="tile must be in [32, 128]")
    raises(ValueError, Geometry, tile=129, match="tile must be in [32, 128]")
    raises(ValueError, Geometry, tile=64.0, match="must be an integer")
    raises(ValueError, Geometry, cell_tiles=5, link_w=5, match="link_w must be in [1, 4]")
    raises(ValueError, Geometry, link_w=0, match="link_w must be in [1, 4]")
    raises(ValueError, Geometry, tile=40, link_w=1, match="opening too narrow")
    Geometry(tile=48, link_w=1)       # 48 units: the narrowest allowed opening
    Geometry(tile=32, cell_tiles=4)
    Geometry(tile=128, cell_tiles=9, link_w=8)


def test_parameter_validation_errors():
    raises(ValueError, derive_params, 1, match="n_rooms must be in [2, 64]")
    raises(ValueError, derive_params, 65, match="n_rooms must be in [2, 64]")
    raises(ValueError, derive_params, 6.0, match="must be an integer")
    raises(ValueError, derive_params, 6, corridors=3, match="unknown generator parameter")
    raises(ValueError, derive_params, 6, lead_in=1, match="lead_in must be >= 2")
    raises(ValueError, derive_params, 6, n_corridor=2, match="must be at least lead_in")
    raises(ValueError, derive_params, 6, branch_prob=1.5, match="branch_prob must be in [0, 1]")
    raises(ValueError, derive_params, 6, heldout_targets=6, match="must be smaller than n_rooms")
    raises(ValueError, derive_params, 6, n_dead_ends=-1, match="n_dead_ends must be >= 0")
    raises(ValueError, derive_params, 6, max_episode_steps=0, match="max_episode_steps")
    raises(ValueError, derive_params, 6, wall_colors=3, match="only 3 distinct room identities")
    raises(ValueError, derive_params, 12, wall_colors=2, floor_colors=4, match="only 8 distinct")
    raises(ValueError, derive_params, 6, train_seeds=[], match="at least one seed")
    raises(ValueError, derive_params, 6, train_seeds=[1, 2], eval_seeds=[2], match="distinct")
    raises(ValueError, derive_params, 6, n_train_layouts=0, match="train layouts")
    raises(ValueError, derive_params, 6, n_train_layouts=80, n_eval_layouts=30,
           match="at most 100 layouts")
    raises(ValueError, generate_layout, 6, 1, match="no preset for 6 rooms")
    raises(ValueError, build_scale, 6, ".", match="no preset for 6 rooms")
    with TempDir() as d:
        raises(ValueError, build_scenario, derive_params(4), d, name="bad name!",
               match="invalid scenario name")
        # map larger than the engine's coordinate range
        raises(ValueError, build_scenario,
               derive_params(64, n_corridor=200), d, geom=Geometry(128, 9), verbose=False,
               match="map too large")
        # a parameter set that cannot be laid out: far too few corridor cells
        raises(RuntimeError, build_scenario,
               derive_params(20, n_corridor=3, lead_in=3, n_train_layouts=1, n_eval_layouts=0),
               d, verbose=False, match="could not generate a layout")
        assert os.listdir(d) == [], "a failed build must not leave files behind"


def test_step_limit_scales_with_cell_size():
    assert scaled_step_limit(500, DEFAULT_GEOMETRY) == 500
    assert scaled_step_limit(500, Geometry(64, 4)) == 450            # 256 / 320 units
    assert scaled_step_limit(500, Geometry(96, 7)) == 780            # 672 / 320 units
    for n in (3, 6, 12):
        steps = [derive_params(n, Geometry(t, c))["max_episode_steps"]
                 for c, t in ((4, 32), (4, 64), (5, 64), (7, 64), (7, 96), (9, 128))]
        assert steps == sorted(steps) and len(set(steps)) == len(steps), steps
        assert steps[2] == derive_params(n)["max_episode_steps"]
    # an explicit step limit is taken as is
    assert derive_params(6, Geometry(96, 7), max_episode_steps=700)["max_episode_steps"] == 700


def test_custom_generation_is_deterministic():
    cases = [(3, 4, 48, 2), (6, 7, 64, 2), (5, 5, 96, 1), (12, 6, 40, 3)]
    with TempDir() as a, TempDir() as b:
        for n, ct, tile, lw in cases:
            geom = Geometry(tile, ct, lw)
            params = derive_params(n, geom, n_train_layouts=2, n_eval_layouts=1, lead_in=4)
            m1 = build_scenario(params, a, "case", geom, verbose=False)
            m2 = build_scenario(dict(params), b, "case", Geometry(tile, ct, lw), verbose=False)
            assert m1 == m2
            for ext in (".wad", ".json", ".cfg"):
                f = "multiroom_color_goal_case" + ext
                assert sha256(os.path.join(a, f)) == sha256(os.path.join(b, f)), (n, ct, tile, ext)
            with open(os.path.join(a, "multiroom_color_goal_case.json")) as f:
                assert json.load(f) == m1
            # other seeds give other maps
            other = derive_params(n, geom, n_train_layouts=2, n_eval_layouts=1, lead_in=4,
                                  seed_base=777000)
            m3 = build_scenario(other, b, "other", geom, verbose=False)
            assert [l["tiles"] for l in m3["layouts"]] != [l["tiles"] for l in m1["layouts"]]


def test_metadata_of_a_custom_scenario():
    with TempDir() as d:
        geom = Geometry(48, 7)
        params = derive_params(10, geom, n_train_layouts=3, n_eval_layouts=2)
        meta = build_scenario(params, d, "r10_test", geom, verbose=False)
        assert meta["name"] == "r10_test" and meta["scale"] == 10
        assert meta["wad"] == "multiroom_color_goal_r10_test.wad"
        assert (meta["tile"], meta["cell_tiles"], meta["content_tiles"], meta["link_lo"],
                meta["link_w"], meta["cell_units"], meta["room_units"]) == (48, 7, 6, 3, 2, 336, 288)
        assert meta["color_scheme"] == "two_tone" and len(meta["identities"]) == 10
        assert len(meta["layouts"]) == 5
        sp = meta["splits"]
        held = sp["eval_colors"]["targets"]
        assert len(held) == params["heldout_targets"] == 3
        assert sorted(held + sp["train"]["targets"]) == list(range(10))
        assert sp["train"]["layouts"] == [0, 1, 2] and sp["eval_maps"]["layouts"] == [3, 4]
        for lm in meta["layouts"]:
            assert lm["cue_los_violations"] == 0
            assert len(lm["rooms"]) == 10
            assert sorted(r["identity"] for r in lm["rooms"]) == list(range(10))
            st = lm["stats"]
            assert st["n_corridor_cells"] == params["n_corridor"]
            assert st["n_distractors"] == params["n_distractors"]
            assert st["path_cells_min"] > params["lead_in"]     # memory distance in cells
            assert len(lm["tiles"]) == lm["grid"][1] * 7 + 1
            assert lm["trigger_x"] - lm["ledge_x"] == 2 * 48
            # cells and links form a tree rooted in the cue room
            assert len(lm["links"]) == len(lm["cells"]) - 1
            # every room is a leaf
            for r in lm["rooms"]:
                assert sum(1 for a, b in lm["links"] if r["cell"] in (a, b)) == 1


def test_layout_grid_for_many_room_counts():
    """Every room count 2..64 can be laid out with the derived parameters (one layout each)."""
    attempts = {}
    for n in range(2, 65):
        layout = generate_layout(derive_params(n), 1000 * n + 1)
        assert layout["cue_los_violations"] == 0 == layout["cue_portal_los_violations"]
        assert layout["stats"]["n_rooms"] == n
        attempts[n] = layout["attempt"]
    print("seed attempts needed (of 200), worst five:",
          sorted(attempts.items(), key=lambda kv: -kv[1])[:5])


def test_strict_line_of_sight_check():
    """A layout the point-sample check accepts although a room can be seen from the airlock.

    Found by the VizDoom grid test (3 rooms, 7 x 7 tiles of 48 units, seed 3002):
    the first attempt of the seed has a thin sight line through two openings.
    The strict check must reject that attempt.
    """
    import random
    from multiroom_color_goal import map_generator as G
    geom = Geometry(48, 7)
    params = derive_params(3, geom)
    identities = build_identities(3, None, 3)
    first = G._try_layout(params, identities, random.Random(3002 * 1000 + 0))
    first["geom"] = geom
    G._finalise_layout(first)
    assert first["cue_los_violations"] == 0            # the samples miss it
    assert G.portal_los_violations(first) == 1         # the strict check does not
    layout = generate_layout(params, 3002, geom=geom)
    assert layout["attempt"] > 0 and layout["cue_portal_los_violations"] == 0


def test_cli():
    with TempDir() as d:
        rc = _quiet(generate.main, ["--rooms", "6", "--cell-tiles", "7", "--tile", "64",
                                    "--name", "r6_big", "--out-dir", d, "--train-layouts", "2",
                                    "--eval-layouts", "1", "--seed-base", "4242", "--print-maps"])
        assert rc == 0
        assert sorted(os.listdir(d)) == ["multiroom_color_goal_r6_big" + e
                                         for e in (".cfg", ".json", ".wad")]
        with open(os.path.join(d, "multiroom_color_goal_r6_big.json")) as f:
            meta = json.load(f)
        assert meta["cell_tiles"] == 7 and meta["scale"] == 6
        assert meta["params"]["train_seeds"] == [4243, 4244]
        assert meta["params"]["eval_seeds"] == [5143]
        # default name encodes a non-default size
        assert _quiet(generate.main, ["--rooms", "3", "--tile", "48", "--out-dir", d,
                                      "--train-layouts", "1", "--eval-layouts", "0"]) == 0
        assert os.path.isfile(os.path.join(d, "multiroom_color_goal_r3_c5_t48.wad"))
        # presets through the CLI are byte-identical
        assert _quiet(generate.main, ["--preset", "2", "4", "--out-dir", d]) == 0
        for n in (2, 4):
            assert sha256(os.path.join(d, "multiroom_color_goal_r%d.wad" % n)) == \
                PRESET_WAD_SHA256[n]
        # errors are reported with exit status 2 and no traceback
        assert _quiet(generate.main, ["--rooms", "6", "--tile", "8", "--out-dir", d]) == 2
        assert _quiet(generate.main, ["--rooms", "99", "--out-dir", d]) == 2
        assert _quiet(generate.main, ["--rooms", "6", "--show-params"]) == 0
