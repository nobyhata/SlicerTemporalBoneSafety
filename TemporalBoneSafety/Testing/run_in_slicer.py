"""Run inside Slicer:
  Slicer --no-splash --additional-module-paths <path>/TemporalBoneSafety --python-script run_in_slicer.py
Runs the self-test and the full widget workflow on the phantom; writes report.csv and screenshot.png to the temp folder."""
import os, sys, tempfile, traceback, slicer
OUT = tempfile.gettempdir()
ok = True
try:
    slicer.util.selectModule("TemporalBoneSafety")   # builds the widget (UI setup)
    w = slicer.modules.TemporalBoneSafetyWidget
    print("WIDGET_OK", w.computeButton.text)
    import TemporalBoneSafety
    t = TemporalBoneSafety.TemporalBoneSafetyTest()
    t.runTest()
    print("SELFTEST_OK")
    # exercise the widget path end-to-end on the phantom
    slicer.mrmlScene.Clear()
    w.onPhantom()
    print("critical checked:", w._criticalIds(), "bone:", w.boneCombo.currentText)
    w.onCompute()
    print("map:", w.logic.hasSafetyMap(), w.statusLabel.text)
    w.drillButton.checked = True
    d = w.logic.drillNode
    p=[0,0,0]; d.GetNthControlPointPositionWorld(0,p)
    for z in range(0, 30):
        d.SetNthControlPointPositionWorld(0, [p[0]+z*0.15, p[1], p[2]-z*0.4])
    w.drillButton.checked = False
    print("drill label:", w.drillLabel.text)
    w.onReport()
    tb = w.logic.reportTable
    for r in range(tb.GetNumberOfRows()):
        print("  ", [tb.GetCellText(r,c) for c in range(tb.GetNumberOfColumns())])
    slicer.util.saveNode(tb, os.path.join(OUT, "TBS_report.csv"))
    slicer.util.mainWindow().resize(1600, 1150)
    lm = slicer.app.layoutManager()
    lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalView)
    segNode = w.segSelector.currentNode()
    segNode.GetDisplayNode().SetOpacity3D(0.6)
    for vn in ("Red",):
        sw = lm.sliceWidget(vn); sw.sliceController().fitSliceToBackground()
    tw = lm.threeDWidget(0); v = tw.threeDView()
    import ctk; v.resetFocalPoint(); v.lookFromViewAxis(ctk.ctkAxesWidget.Superior)
    cam = slicer.app.layoutManager().threeDWidget(0).threeDView().cameraNode().GetCamera()
    cam.Elevation(-40); cam.Azimuth(30); v.resetCamera(); cam.Zoom(1.3)
    slicer.app.processEvents(); v.forceRender()
    slicer.util.mainWindow().grab().save(os.path.join(OUT, "TBS_screenshot.png"))
    print("WIDGETFLOW_OK")
except Exception:
    traceback.print_exc(); ok = False
sys.stdout.flush()
slicer.util.exit(0 if ok else 1)
