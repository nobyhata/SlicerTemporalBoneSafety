"""Unit tests for TemporalBoneSafetyLib.core (run outside Slicer with pytest)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from TemporalBoneSafetyLib import core  # noqa: E402


SP = (0.5, 0.5, 1.0)  # anisotropic on purpose


def ball(shape, c_ijk, r_mm, sp):
    kk, jj, ii = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    d2 = ((ii - c_ijk[0]) * sp[0]) ** 2 + ((jj - c_ijk[1]) * sp[1]) ** 2 + \
         ((kk - c_ijk[2]) * sp[2]) ** 2
    return (d2 <= r_mm ** 2).astype(np.uint8)


def test_signed_distance_sign_and_scale():
    m = np.zeros((20, 40, 40), np.uint8)
    m[:, :, :20] = 1  # half-space along i, boundary between i=19 and i=20
    d = core.signed_distance_mm(m, SP)
    assert d[10, 20, 10] < 0
    assert d[10, 20, 30] > 0
    # i=30 is 10..11 voxels from the boundary -> ~5 mm at 0.5 mm spacing
    assert 4.5 <= d[10, 20, 30] <= 5.5


def test_signed_distance_empty_is_inf():
    d = core.signed_distance_mm(np.zeros((5, 5, 5)), SP)
    assert np.isinf(d).all()


def test_combined_distance_nearest_index():
    shape = (20, 40, 40)
    a = ball(shape, (5, 20, 10), 1.5, SP)
    b = ball(shape, (35, 20, 10), 1.5, SP)
    d, idx = core.combined_distance([a, b], SP)
    assert idx[10, 20, 6] == 0
    assert idx[10, 20, 33] == 1
    assert d.min() < 0


def test_trilinear_matches_linear_field():
    k, j, i = np.mgrid[0:10, 0:10, 0:10]
    f = (2 * i + 3 * j + 5 * k).astype(float)
    v = core.trilinear_sample(f, [[1.5, 2.25, 3.75]])[0]
    assert v == pytest.approx(2 * 1.5 + 3 * 2.25 + 5 * 3.75)
    assert np.isinf(core.trilinear_sample(f, [[-1, 0, 0]])[0])


def test_carve_sphere_volume():
    m = np.ones((40, 60, 60), np.uint8)
    n = core.carve_sphere(m, (30, 30, 20), 3.0, SP)
    expected = 4 / 3 * np.pi * 3.0 ** 3 / np.prod(SP)
    assert abs(n - expected) / expected < 0.15
    assert m[20, 30, 30] == 0 and m[0, 0, 0] == 1
    # carving the same place again removes nothing
    assert core.carve_sphere(m, (30, 30, 20), 3.0, SP) == 0


def test_carve_sphere_outside_volume():
    m = np.ones((10, 10, 10), np.uint8)
    assert core.carve_sphere(m, (100, 100, 100), 1.0, SP) == 0


def test_carve_path_is_continuous():
    m = np.ones((20, 20, 100), np.uint8)
    core.carve_path(m, (5, 10, 10), (95, 10, 10), 1.0, SP)
    assert (m[10, 10, 5:96] == 0).all()


def test_burr_clearance():
    shape = (30, 60, 60)
    nerve = ball(shape, (30, 30, 15), 1.0, SP)
    d = core.signed_distance_mm(nerve, SP)
    # burr centre 4 mm away along i (8 voxels), radius 1 -> ~2 mm clearance
    c = core.burr_clearance(d, (38, 30, 15), 1.0)
    assert 1.5 < c < 2.5
    assert core.burr_clearance(d, (30, 30, 15), 1.0) < 0


def test_drilling_report_closest_approach():
    shape = (30, 60, 60)
    nerve = ball(shape, (30, 30, 15), 1.0, SP)
    bone = np.ones(shape, np.uint8)
    bone[nerve > 0] = 0
    cur = bone.copy()
    core.carve_sphere(cur, (40, 30, 15), 2.0, SP)  # centre 5 mm, r 2 -> ~2 mm left
    r = core.drilling_report(bone, cur, {"Facial nerve": nerve}, SP)
    s = r["structures"]["Facial nerve"]
    assert r["removed_volume_mm3"] > 0
    assert 1.2 < s["closest_approach_mm"] < 2.6
    assert s["breach"] is False
    core.carve_sphere(cur, (32, 30, 15), 1.5, SP)  # now hit the nerve region
    r2 = core.drilling_report(bone, cur, {"Facial nerve": nerve}, SP)
    assert r2["structures"]["Facial nerve"]["closest_approach_mm"] < 0.5


def test_drilling_report_nothing_removed():
    shape = (10, 10, 10)
    bone = np.ones(shape, np.uint8)
    r = core.drilling_report(bone, bone, {"X": ball(shape, (5, 5, 5), 1, SP)}, SP)
    assert r["removed_voxels"] == 0
    assert r["structures"]["X"]["closest_approach_mm"] is None


def test_motion_stats_counts_pauses():
    t = np.arange(0, 10.01, 0.1)
    p = np.zeros((len(t), 3))
    # move 0-2 s, stop 2-4 s, move 4-6 s, stop 6-6.5 s (too short), move to end
    x = 0.0
    for n, tt in enumerate(t):
        if not (2 <= tt < 4 or 6 <= tt < 6.5):
            x += 0.5
        p[n, 0] = x
    s = core.motion_stats(t, p, still_mm_per_s=0.5, pause_s=1.0)
    assert s["pause_count"] == 1
    assert 1.8 <= s["pause_time_s"] <= 2.2
    assert s["duration_s"] == pytest.approx(10.0)


def test_zone_times():
    t = [0, 1, 2, 3, 4]
    d = [5, 1.5, 0.5, 0.5, 5]
    z = core.zone_times(t, d, danger_mm=1.0, warning_mm=2.0)
    assert z["danger_s"] == pytest.approx(2.0)
    assert z["warning_s"] == pytest.approx(1.0)
    assert z["min_distance_mm"] == 0.5


def test_zone_label():
    assert core.zone_label(-0.1, 1, 2) == "breach"
    assert core.zone_label(0.5, 1, 2) == "danger"
    assert core.zone_label(1.5, 1, 2) == "warning"
    assert core.zone_label(3, 1, 2) == "safe"
    assert core.zone_label(np.inf, 1, 2) == "none"


def test_phantom_structures_are_disjoint_and_nonempty():
    ct, masks = core.make_phantom()
    assert ct.dtype == np.int16
    for name, m in masks.items():
        assert m.any(), name
    assert not (masks["Bone"] & masks["Facial nerve"]).any()
    # facial nerve lies inside the bone block -> drilling toward it is possible
    d, _ = core.combined_distance(
        [masks[n] for n in masks if n != "Bone"], (0.3, 0.3, 0.3))
    assert d[masks["Bone"] > 0].min() < 0.5
