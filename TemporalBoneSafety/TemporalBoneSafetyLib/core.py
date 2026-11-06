"""
Pure numpy (+ optional SimpleITK) computations for the TemporalBoneSafety module.

Nothing in this file imports Slicer, so it can be unit-tested outside Slicer.

Conventions
-----------
* Volumes are numpy arrays indexed [k, j, i] (slice, row, column), exactly as
  returned by slicer.util.arrayFromVolume / arrayFromSegmentBinaryLabelmap.
* Spacing is given as (sx, sy, sz) in mm, i.e. along i, j, k.
* Points in voxel space are given as (i, j, k) floats.
* Distances are in mm. Signed distances are negative inside a structure.
"""

import math
import numpy as np

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - SimpleITK ships with Slicer
    sitk = None


# ---------------------------------------------------------------------------
# Distance maps
# ---------------------------------------------------------------------------

def signed_distance_mm(mask_kji, spacing_xyz):
    """Signed Euclidean distance (mm) to the surface of a binary mask.

    Negative inside the mask, positive outside. Returns float32.
    An empty mask returns +inf everywhere.
    """
    mask = np.asarray(mask_kji) > 0
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    if sitk is not None:
        img = sitk.GetImageFromArray(mask.astype(np.uint8))
        img.SetSpacing(tuple(float(s) for s in spacing_xyz))
        dist = sitk.SignedMaurerDistanceMap(
            img, insideIsPositive=False, squaredDistance=False,
            useImageSpacing=True)
        return sitk.GetArrayFromImage(dist).astype(np.float32)
    # Fallback: scipy
    from scipy import ndimage
    sampling = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    outside = ndimage.distance_transform_edt(~mask, sampling=sampling)
    inside = ndimage.distance_transform_edt(mask, sampling=sampling)
    return (outside - inside).astype(np.float32)


def combined_distance(masks, spacing_xyz):
    """Distance to the nearest of several structures.

    masks : list of binary arrays (same shape)
    Returns (min_dist float32, nearest_index int16); nearest_index is -1 where
    no structure exists.
    """
    if not masks:
        raise ValueError("At least one critical structure is required")
    min_dist = None
    nearest = None
    for idx, m in enumerate(masks):
        d = signed_distance_mm(m, spacing_xyz)
        if min_dist is None:
            min_dist = d
            nearest = np.where(np.isfinite(d), idx, -1).astype(np.int16)
        else:
            closer = d < min_dist
            min_dist = np.where(closer, d, min_dist)
            nearest[closer] = idx
    return min_dist.astype(np.float32), nearest


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def trilinear_sample(arr_kji, points_ijk, outside_value=np.inf):
    """Trilinear interpolation of arr at (i, j, k) points. Nx3 -> N."""
    arr = np.asarray(arr_kji, dtype=np.float64)
    pts = np.atleast_2d(np.asarray(points_ijk, dtype=np.float64))
    nk, nj, ni = arr.shape
    i, j, k = pts[:, 0], pts[:, 1], pts[:, 2]
    inside = (i >= 0) & (j >= 0) & (k >= 0) & \
             (i <= ni - 1) & (j <= nj - 1) & (k <= nk - 1)
    out = np.full(len(pts), outside_value, dtype=np.float64)
    if not inside.any():
        return out
    i, j, k = i[inside], j[inside], k[inside]
    i0 = np.clip(np.floor(i).astype(int), 0, max(ni - 2, 0))
    j0 = np.clip(np.floor(j).astype(int), 0, max(nj - 2, 0))
    k0 = np.clip(np.floor(k).astype(int), 0, max(nk - 2, 0))
    i1 = np.minimum(i0 + 1, ni - 1)
    j1 = np.minimum(j0 + 1, nj - 1)
    k1 = np.minimum(k0 + 1, nk - 1)
    fi, fj, fk = i - i0, j - j0, k - k0

    def v(kk, jj, ii):
        return arr[kk, jj, ii]

    c00 = v(k0, j0, i0) * (1 - fi) + v(k0, j0, i1) * fi
    c01 = v(k0, j1, i0) * (1 - fi) + v(k0, j1, i1) * fi
    c10 = v(k1, j0, i0) * (1 - fi) + v(k1, j0, i1) * fi
    c11 = v(k1, j1, i0) * (1 - fi) + v(k1, j1, i1) * fi
    c0 = c00 * (1 - fj) + c01 * fj
    c1 = c10 * (1 - fj) + c11 * fj
    vals = c0 * (1 - fk) + c1 * fk
    # Infinite values (empty structure) poison interpolation -> keep inf
    out[inside] = np.where(np.isfinite(vals), vals, outside_value)
    return out


def nearest_sample(arr_kji, point_ijk, outside_value=-1):
    """Nearest-neighbour lookup at one (i, j, k) point."""
    arr = np.asarray(arr_kji)
    i, j, k = (int(round(c)) for c in point_ijk)
    nk, nj, ni = arr.shape
    if 0 <= i < ni and 0 <= j < nj and 0 <= k < nk:
        return arr[k, j, i]
    return outside_value


# ---------------------------------------------------------------------------
# Virtual drilling
# ---------------------------------------------------------------------------

def _sphere_box(shape, center_ijk, radius_mm, spacing_xyz):
    nk, nj, ni = shape
    ci, cj, ck = center_ijk
    ri = radius_mm / spacing_xyz[0]
    rj = radius_mm / spacing_xyz[1]
    rk = radius_mm / spacing_xyz[2]
    i0, i1 = max(int(math.floor(ci - ri)), 0), min(int(math.ceil(ci + ri)) + 1, ni)
    j0, j1 = max(int(math.floor(cj - rj)), 0), min(int(math.ceil(cj + rj)) + 1, nj)
    k0, k1 = max(int(math.floor(ck - rk)), 0), min(int(math.ceil(ck + rk)) + 1, nk)
    if i0 >= i1 or j0 >= j1 or k0 >= k1:
        return None
    kk, jj, ii = np.ogrid[k0:k1, j0:j1, i0:i1]
    d2 = ((ii - ci) * spacing_xyz[0]) ** 2 + \
         ((jj - cj) * spacing_xyz[1]) ** 2 + \
         ((kk - ck) * spacing_xyz[2]) ** 2
    return (slice(k0, k1), slice(j0, j1), slice(i0, i1)), d2 <= radius_mm ** 2


def carve_sphere(mask_kji, center_ijk, radius_mm, spacing_xyz):
    """Remove a spherical burr from mask in place. Returns voxels removed."""
    box = _sphere_box(mask_kji.shape, center_ijk, radius_mm, spacing_xyz)
    if box is None:
        return 0
    sl, ball = box
    sub = mask_kji[sl]
    hit = ball & (sub > 0)
    n = int(hit.sum())
    if n:
        sub[hit] = 0
    return n


def carve_path(mask_kji, p0_ijk, p1_ijk, radius_mm, spacing_xyz):
    """Carve a swept sphere from p0 to p1 (both ijk). Returns voxels removed.

    Samples every radius/2 so fast drags still remove a continuous track.
    """
    p0 = np.asarray(p0_ijk, dtype=float)
    p1 = np.asarray(p1_ijk, dtype=float)
    sp = np.asarray(spacing_xyz, dtype=float)
    length_mm = float(np.linalg.norm((p1 - p0) * sp))
    step = max(radius_mm / 2.0, 1e-3)
    n = max(int(math.ceil(length_mm / step)), 1)
    removed = 0
    for t in np.linspace(0.0, 1.0, n + 1):
        removed += carve_sphere(mask_kji, p0 + t * (p1 - p0), radius_mm, sp)
    return removed


def burr_clearance(dist_kji, center_ijk, radius_mm):
    """Clearance (mm) between burr surface and nearest critical structure.

    Negative means the burr overlaps the structure (breach).
    """
    d = trilinear_sample(dist_kji, [center_ijk])[0]
    return float(d - radius_mm)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def drilling_report(original_bone, current_bone, structures, spacing_xyz,
                    breach_tolerance_mm=0.0):
    """Compare bone before/after drilling.

    structures : dict name -> binary mask
    Returns dict with removed volume and, per structure, the closest approach
    of the drilled cavity (mm, = thinnest bone left over that structure) and
    a breach flag.
    """
    orig = np.asarray(original_bone) > 0
    cur = np.asarray(current_bone) > 0
    removed = orig & ~cur
    voxel_mm3 = float(np.prod(spacing_xyz))
    report = {
        "removed_voxels": int(removed.sum()),
        "removed_volume_mm3": float(removed.sum() * voxel_mm3),
        "structures": {},
    }
    for name, m in structures.items():
        entry = {"closest_approach_mm": None, "breach": False,
                 "volume_within_1mm_removed_mm3": 0.0}
        if removed.any() and np.asarray(m).any():
            d = signed_distance_mm(m, spacing_xyz)
            vals = d[removed]
            # Distance from a removed voxel *centre*; subtract half a voxel so
            # the number approximates the thickness of bone left.
            half_voxel = 0.5 * float(min(spacing_xyz))
            ca = float(vals.min()) - half_voxel
            entry["closest_approach_mm"] = ca
            entry["breach"] = bool(ca <= breach_tolerance_mm)
            entry["volume_within_1mm_removed_mm3"] = float(
                (vals <= 1.0 + half_voxel).sum() * voxel_mm3)
        report["structures"][name] = entry
    return report


def motion_stats(timestamps_s, positions_mm, still_mm_per_s=0.5,
                 pause_s=1.0):
    """Count pauses ("hand stops") in a tool/drill trajectory.

    A pause is a run of samples moving slower than still_mm_per_s that lasts
    at least pause_s seconds.
    Returns dict: duration_s, path_length_mm, pause_count, pause_time_s.
    """
    t = np.asarray(timestamps_s, dtype=float)
    p = np.asarray(positions_mm, dtype=float)
    out = {"duration_s": 0.0, "path_length_mm": 0.0,
           "pause_count": 0, "pause_time_s": 0.0}
    if len(t) < 2:
        return out
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    out["duration_s"] = float(t[-1] - t[0])
    out["path_length_mm"] = float(step.sum())
    # Gaps between samples with no movement events also count as still.
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt, 0.0)
    still = speed < still_mm_per_s
    run = 0.0
    for is_still, d in zip(still, dt):
        if is_still:
            run += d
        else:
            if run >= pause_s:
                out["pause_count"] += 1
                out["pause_time_s"] += run
            run = 0.0
    if run >= pause_s:
        out["pause_count"] += 1
        out["pause_time_s"] += run
    return out


def zone_times(timestamps_s, distances_mm, danger_mm, warning_mm):
    """Time spent in danger (< danger) and warning (< warning) zones."""
    t = np.asarray(timestamps_s, dtype=float)
    d = np.asarray(distances_mm, dtype=float)
    if len(t) < 2:
        return {"danger_s": 0.0, "warning_s": 0.0,
                "min_distance_mm": float(d.min()) if len(d) else None}
    dt = np.diff(t)
    dd = d[:-1]
    return {"danger_s": float(dt[dd < danger_mm].sum()),
            "warning_s": float(dt[(dd >= danger_mm) & (dd < warning_mm)].sum()),
            "min_distance_mm": float(d.min())}


def zone_label(distance_mm, danger_mm, warning_mm):
    if distance_mm is None or not np.isfinite(distance_mm):
        return "none"
    if distance_mm <= 0:
        return "breach"
    if distance_mm < danger_mm:
        return "danger"
    if distance_mm < warning_mm:
        return "warning"
    return "safe"


# ---------------------------------------------------------------------------
# Synthetic phantom
# ---------------------------------------------------------------------------

def make_phantom(shape_kji=(80, 100, 100), spacing_xyz=(0.3, 0.3, 0.3)):
    """Crude temporal-bone-like phantom for demos and tests.

    Returns (ct_hu int16, masks dict). Structures are embedded in a block of
    bone: a curved facial-nerve canal, a sigmoid sinus, a dura plate on top,
    a labyrinth blob, and an air-filled antrum near the surface.
    """
    nk, nj, ni = shape_kji
    sx, sy, sz = spacing_xyz
    kk, jj, ii = np.mgrid[0:nk, 0:nj, 0:ni].astype(np.float32)
    x, y, z = ii * sx, jj * sy, kk * sz          # mm
    X, Y, Z = ni * sx, nj * sy, nk * sz

    bone = (z > 0.1 * Z) & (z < 0.85 * Z) & (y < 0.9 * Y + 1.0)

    # Dura (middle fossa / tegmen side): 1 mm plate along the superior wall.
    # Drilling enters from the lateral surface (high z), so the dura is
    # a side wall to keep away from, not the entry surface.
    dura = (y >= 0.9 * Y) & (y < 0.9 * Y + 1.0) & (z > 0.1 * Z) & (z < 0.85 * Z)

    # Facial nerve: curved tube (~1.2 mm diameter) running along y
    cx = 0.45 * X + 0.12 * X * np.sin(np.pi * y / Y)
    cz = 0.45 * Z
    facial = ((x - cx) ** 2 + (z - cz) ** 2) <= 0.6 ** 2
    facial &= (y > 0.1 * Y) & (y < 0.9 * Y)

    # Sigmoid sinus: thick tube (~5 mm) along z at the back
    sinus = ((x - 0.85 * X) ** 2 + (y - 0.75 * Y) ** 2) <= 2.5 ** 2
    sinus &= (z > 0.2 * Z) & (z < 0.85 * Z)

    # Labyrinth: blob medial to the facial nerve
    lab = ((x - 0.2 * X) ** 2 + (y - 0.5 * Y) ** 2 + (z - 0.4 * Z) ** 2) <= 2.5 ** 2

    # Antrum: air cell near the drilling surface
    antrum = ((x - 0.55 * X) ** 2 + (y - 0.5 * Y) ** 2 + (z - 0.7 * Z) ** 2) <= 2.0 ** 2

    soft = facial | sinus | lab | dura
    bone = bone & ~soft & ~antrum

    ct = np.full(shape_kji, -1000, dtype=np.int16)
    ct[z > 0.05 * Z] = 40
    ct[bone] = 1500
    ct[facial | sinus | dura] = 60
    ct[lab] = 10
    ct[antrum] = -900

    masks = {
        "Bone": bone.astype(np.uint8),
        "Facial nerve": facial.astype(np.uint8),
        "Sigmoid sinus": sinus.astype(np.uint8),
        "Dura": dura.astype(np.uint8),
        "Labyrinth": lab.astype(np.uint8),
    }
    return ct, masks
