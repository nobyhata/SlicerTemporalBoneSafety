"""
Run TemporalBoneSafety on a public OpenEar temporal bone (CC BY 4.0,
Sieber et al., Sci Data 2019; Zenodo 10.5281/zenodo.1473724).

Usage (inside Slicer, or headless):
  OPENEAR_DIR=/path/with/CBCT_Unembedded.nrrd+Segmentation.seg.nrrd \
  TBS_OUT=/path/for/results \
  Slicer --no-splash --additional-module-paths <repo>/TemporalBoneSafety \
         --python-script openear_demo.py

Get the files with fetch_openear.py (downloads only the 3 needed files of a
specimen, ~290 MB, instead of the whole 4-14 GB zip).

What it does
  1. Loads the registered CBCT and the OpenEar segmentation.
  2. Crops around the middle/inner ear (+15 mm) and resamples to 0.25 mm.
  3. Computes the safety map with the module (critical structures picked by
     the module's own name rules).
  4. Scripted mastoidectomy rehearsal through the module's Drill point:
     entry on the mastoid cortex chosen automatically (straight path toward
     the antrum next to the incus that gets deepest without crossing the
     danger line and without entering the external auditory canal);
     4 mm cutting burr from ~40 entry points around it (saucerisation),
     then a 2 mm burr along the main axis. Every stroke stops when the burr
     would come within the danger threshold of a critical structure.
  5. Drilling report (CSV) + screenshots.
"""
import os
import sys
import json
import tempfile
import time
import traceback

import numpy as np
import slicer
import vtk
import ctk
import qt
import SimpleITK as sitk

DATA = os.environ.get("OPENEAR_DIR", os.getcwd())
OUT = os.environ.get("TBS_OUT", tempfile.gettempdir())
SPECIMEN = os.environ.get("OPENEAR_SPECIMEN", "OpenEar")
os.makedirs(OUT, exist_ok=True)

CORE_FOR_ROI = ["Facial Nerve", "Chorda Tympani", "Malleus", "Incus", "Stapes",
                "Scala Tympani", "Scala Vestibuli", "External Auditory Canal"]
ROI_MARGIN_MM = 15.0
DANGER, WARNING = 1.0, 2.0


ALIASES = {"External Auditory Canal": ["external auditory canal", "eac", "ear canal"],
           "Incus": ["incus"]}


def find_segment(segNode, name):
    """Segment id by name, tolerant to OpenEar naming differences between specimens."""
    segs = segNode.GetSegmentation()
    keys = [k.lower() for k in ALIASES.get(name, [name])]
    for n in range(segs.GetNumberOfSegments()):
        sid = segs.GetNthSegmentID(n)
        nm = segs.GetSegment(sid).GetName().lower().strip()
        if nm in keys or any(nm.startswith(k) for k in keys):
            return sid
    return None


def log(*a):
    print("[openear]", *a)
    sys.stdout.flush()


def segment_bounds(segNode, names):
    segs = segNode.GetSegmentation()
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for n in names:
        sid = segs.GetSegmentIdBySegmentName(n) or find_segment(segNode, n)
        if not sid:
            continue
        b = [0.0] * 6
        segs.GetSegment(sid).GetBounds(b)
        lo = np.minimum(lo, b[0::2])
        hi = np.maximum(hi, b[1::2])
    return lo, hi


def crop(ctNode, lo, hi):
    roi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "TBS crop ROI")
    roi.SetCenter(list((lo + hi) / 2.0))
    roi.SetSize(list(hi - lo))
    roi.SetDisplayVisibility(False)
    p = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLCropVolumeParametersNode")
    p.SetInputVolumeNodeID(ctNode.GetID())
    p.SetROINodeID(roi.GetID())
    p.SetVoxelBased(False)
    p.SetIsotropicResampling(True)
    p.SetSpacingScalingConst(2.0)      # 0.125 mm -> 0.25 mm
    p.SetInterpolationMode(p.InterpolationLinear)
    slicer.modules.cropvolume.logic().Apply(p)
    out = slicer.mrmlScene.GetNodeByID(p.GetOutputVolumeNodeID())
    out.SetName(ctNode.GetName() + " cropped 0.25mm")
    return out


def main():
    t0 = time.time()
    ct = slicer.util.loadVolume(os.path.join(DATA, "CBCT_Unembedded.nrrd"))
    segNode = slicer.util.loadSegmentation(os.path.join(DATA, "Segmentation.seg.nrrd"))
    lo, hi = segment_bounds(segNode, CORE_FOR_ROI)
    lo -= ROI_MARGIN_MM
    hi += ROI_MARGIN_MM
    vb = [0.0] * 6
    ct.GetRASBounds(vb)
    lo = np.maximum(lo, vb[0::2])
    hi = np.minimum(hi, vb[1::2])
    cropped = crop(ct, lo, hi)
    log("cropped CT", cropped.GetImageData().GetDimensions(), "spacing", cropped.GetSpacing(),
        "size mm", np.round(hi - lo, 1))
    slicer.mrmlScene.RemoveNode(ct)
    segNode.SetReferenceImageGeometryParameterFromVolumeNode(cropped)

    # ---------------- module, through its widget (same path as a user)
    slicer.util.selectModule("TemporalBoneSafety")
    w = slicer.modules.TemporalBoneSafetyWidget
    logic = w.logic
    w.volumeSelector.setCurrentNode(cropped)
    w.segSelector.setCurrentNode(segNode)
    w.dangerSpin.value = DANGER
    w.warningSpin.value = WARNING
    critical = [w.criticalList.item(r).text() for r in range(w.criticalList.count)
                if w.criticalList.item(r).checkState() == qt.Qt.Checked]
    log("bone segment:", w.boneCombo.currentText)
    log("critical (auto-checked):", critical)
    t = time.time()
    w.onCompute()
    log("safety map %.1f s:" % (time.time() - t), w.statusLabel.text)
    assert logic.hasSafetyMap()

    sp = np.array(logic.spacing)
    dist = logic.distance
    bone = logic.boneArray > 0
    segs = segNode.GetSegmentation()
    incusId = find_segment(segNode, "Incus")
    eacId = find_segment(segNode, "External Auditory Canal")
    if not incusId:
        raise RuntimeError("No incus segment found")
    incus = logic._segmentArray(incusId) > 0
    eac = (logic._segmentArray(eacId) > 0) if eacId else np.zeros_like(bone)

    # ---------------- plan: target next to incus, entry on outer cortex
    kji = np.argwhere(incus)
    T = kji.mean(axis=0)[::-1]                       # (i, j, k)
    free = (~bone) & (dist > 0) & (~eac)
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(
        sitk.GetImageFromArray(free.astype(np.uint8))), sortByObjectSize=True)
    outside = sitk.GetArrayFromImage(cc) == 1
    outerShell = sitk.GetArrayFromImage(sitk.BinaryDilate(
        sitk.GetImageFromArray(outside.astype(np.uint8)), [1, 1, 1])) > 0
    surface = bone & outerShell
    cand = np.argwhere(surface)[:, ::-1].astype(float)   # (i, j, k)
    rng = np.random.default_rng(0)
    if len(cand) > 3000:
        cand = cand[rng.choice(len(cand), 3000, replace=False)]
    log("outer cortex voxels:", int(surface.sum()), "candidates:", len(cand))

    from TemporalBoneSafetyLib import core

    def move(p):
        # Same code path as dragging the Drill point (PointModifiedEvent ->
        # drillAtWorld), without per-move rendering, for batch speed.
        logic.drillAtWorld(logic._ijkToWorld(list(p)), log=logic.drillLog,
                           prevAttr="_prevDrillIjk", callback=logic.drillCallback)

    def stroke(S, r, stop_mm=2.0):
        """Points (ijk) from 3 mm outside S toward T, stopping at the danger line."""
        v_mm = (T - S) * sp
        L = np.linalg.norm(v_mm)
        u = v_mm / L
        ts = np.arange(-3.0, L - stop_mm, 0.4)
        pts = S + np.outer(ts, u) / sp
        d = core.trilinear_sample(dist, pts)
        e = core.trilinear_sample(eac.astype(np.float32), pts, outside_value=0.0)
        ok = (d - r >= DANGER) & (e < 0.5)
        n = len(pts) if ok.all() else int(np.argmin(ok))
        reach = np.linalg.norm((T - pts[max(n - 1, 0)]) * sp) if n else np.inf
        return pts[:n], reach

    R4 = 2.0
    t = time.time()
    reaches = np.array([stroke(S, R4)[1] for S in cand])
    log("planned %d candidate strokes in %.1f s" % (len(cand), time.time() - t))
    best = int(np.argmin(reaches))
    E = cand[best]
    log("best entry reach to incus centroid: %.1f mm" % reaches[best])
    near = np.linalg.norm((cand - E) * sp, axis=1)
    pool = np.where((near <= 7.0) & np.isfinite(reaches))[0]
    entries = [best] + list(rng.choice(pool[pool != best], min(40, len(pool) - 1), replace=False))

    # ---------------- drill through the module
    w.onStartRehearsal()
    w.burrSpin.value = 4.0
    w.drillButton.checked = True
    dn = logic.drillNode
    nmoves = 0
    t = time.time()
    for idx in entries:
        pts, _ = stroke(cand[idx], R4)
        logic.liftDrill()
        for p in pts:
            move(p)
            nmoves += 1
    log("4 mm burr: %d strokes, %d moves" % (len(entries), nmoves))
    w.burrSpin.value = 2.0
    pts, reach2 = stroke(E, 1.0, stop_mm=0.5)
    logic.liftDrill()
    for p in pts:
        move(p)
    w.drillButton.checked = False
    dn.SetNthControlPointPositionWorld(0, logic._ijkToWorld(list(pts[-1])))
    stopRas = logic._ijkToWorld(list(pts[-1]))
    log("2 mm burr main axis: stopped %.1f mm from incus centroid; %.1f s drilling"
        % (reach2, time.time() - t))
    log("live label:", w.drillLabel.text)

    w.onReport()
    tb = logic.reportTable
    rows = [[tb.GetCellText(r, c) for c in range(tb.GetNumberOfColumns())]
            for r in range(tb.GetNumberOfRows())]
    for r in rows:
        log("  ", r)
    csv = os.path.join(OUT, "%s_drilling_report.csv" % SPECIMEN)
    slicer.util.saveNode(tb, csv)
    with open(os.path.join(OUT, "%s_run.json" % SPECIMEN), "w") as f:
        json.dump({"specimen": SPECIMEN, "critical": critical, "spacing_mm": list(sp),
                   "crop_dims": list(cropped.GetImageData().GetDimensions()),
                   "entries": len(entries), "rows": rows,
                   "seconds_total": time.time() - t0}, f, indent=1)

    # ---------------- screenshots
    mw = slicer.util.mainWindow()
    mw.resize(1700, 1150)
    lm = slicer.app.layoutManager()
    disp = segNode.GetDisplayNode()
    for n in range(segs.GetNumberOfSegments()):
        sid = segs.GetNthSegmentID(n)
        name = segs.GetSegment(sid).GetName()
        disp.SetSegmentVisibility3D(sid, name in critical)
    segNode.CreateClosedSurfaceRepresentation()
    model = logic.surfaceModel
    E_ras = np.array(logic._ijkToWorld(list(E)))
    T_ras = np.array(logic._ijkToWorld(list(T)))
    axis = (T_ras - E_ras) / np.linalg.norm(T_ras - E_ras)

    lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
    v = lm.threeDWidget(0).threeDView()
    cam = v.cameraNode()
    vn = lm.threeDWidget(0).mrmlViewNode()
    vn.SetBoxVisible(False)
    vn.SetAxisLabelsVisible(False)

    def shot(name, opacity, back_mm, up=(0, 0, 1)):
        model.GetDisplayNode().SetOpacity(opacity)
        cam.SetFocalPoint(*T_ras)
        cam.SetPosition(*(T_ras - axis * back_mm))
        cam.SetViewUp(*up)
        cam.GetCamera().SetViewAngle(30)
        cam.ResetClippingRange()
        slicer.app.processEvents()
        v.forceRender()
        path = os.path.join(OUT, "%s_%s.png" % (SPECIMEN, name))
        v.grab().save(path)
        log("saved", path)

    up = np.cross(axis, [1, 0, 0])
    up = tuple(up / np.linalg.norm(up))
    shot("3d_surgeon_view", 1.0, 60.0, up)
    lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalView)
    slicer.modules.markups.logic().JumpSlicesToLocation(*stopRas, True)
    for c in ("Red", "Yellow", "Green"):
        sl = lm.sliceWidget(c).sliceLogic()
        sl.GetSliceNode().SetFieldOfView(40, 40, 1)
    model.GetDisplayNode().SetOpacity(1.0)
    cam.SetPosition(*(T_ras - axis * 60))
    cam.SetFocalPoint(*T_ras)
    cam.ResetClippingRange()
    w.cursorLabel.text = "Cursor: %s %.1f mm" % logic.queryWorld(list(stopRas))
    slicer.app.processEvents()
    mw.grab().save(os.path.join(OUT, "%s_slicer_window.png" % SPECIMEN))
    log("done in %.0f s" % (time.time() - t0))


try:
    main()
    code = 0
except Exception:
    traceback.print_exc()
    code = 1
sys.stdout.flush()
if os.environ.get("TBS_EXIT", "1") == "1":
    slicer.util.exit(code)
