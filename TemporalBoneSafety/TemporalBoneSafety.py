"""
TemporalBoneSafety - 3D Slicer scripted module

Supports the otologic-drilling need "how far and how deep can I drill?":

  1. Segmentation      - optional automatic segmentation with an nnU-Net v2 model
  2. Safety-margin map - bone surface coloured by clearance to critical
                         structures, danger/warning bone zones, live cursor
                         readout of the nearest structure and distance
  3. Virtual drilling  - a spherical burr removes bone from the patient's own
                         segmentation, with live clearance and a drilling
                         report (removed volume, thinnest bone left over each
                         structure, breaches, pauses)
  4. Tracking          - a tracked drill tip (e.g. via OpenIGTLink) gives a
                         live distance with colour/sound warning, optional
                         virtual removal, and a session log

Research use only. Not a medical device.
"""

import json
import os
import subprocess
import tempfile
import time

import numpy as np
import qt
import ctk
import vtk
import slicer
from vtk.util import numpy_support
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule, ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic, ScriptedLoadableModuleTest)
from slicer.util import VTKObservationMixin

from TemporalBoneSafetyLib import core

GENERATED_PREFIX = "TBS: "
CRITICAL_KEYWORDS = ("nerve", "facial", "chorda", "sinus", "dura", "labyrinth",
                     "cochlea", "vestib", "semicircular", "scala", "carotid", "carotis",
                     "ossic", "malleus", "incus", "stapes", "jugular", "brain")
BONE_KEYWORDS = ("bone", "temporal", "mastoid")


# =============================================================================
# Module
# =============================================================================

class TemporalBoneSafety(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Temporal Bone Safety"
        self.parent.categories = ["IGT"]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Surgical Robotics and Navigation Laboratory, Brigham and Women's Hospital"]
        self.parent.helpText = (
            "Safety-margin map, virtual drilling and drill-tip distance monitoring "
            "for otologic (temporal bone) surgery. Shows how much bone is left "
            "between the drill and the facial nerve, sigmoid sinus, dura and "
            "labyrinth. Research use only.")
        self.parent.acknowledgementText = (
            "Developed for the SPL Academy (日本版SPLプログラム) Red Sox group project.")


# =============================================================================
# Widget
# =============================================================================

class TemporalBoneSafetyWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._observedSegmentation = None
        self._crosshair = None
        self._crosshairTag = None

    # ------------------------------------------------------------------ UI
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = TemporalBoneSafetyLogic()
        self.logic.statusCallback = self._onLogicStatus

        # ---------------- Inputs
        inputs = ctk.ctkCollapsibleButton()
        inputs.text = "Inputs"
        self.layout.addWidget(inputs)
        form = qt.QFormLayout(inputs)

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.noneEnabled = True
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self.volumeSelector.setToolTip("Temporal bone CT / CBCT (crop to the ear for speed)")
        form.addRow("CT volume:", self.volumeSelector)

        self.segSelector = slicer.qMRMLNodeComboBox()
        self.segSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.segSelector.noneEnabled = True
        self.segSelector.addEnabled = True
        self.segSelector.removeEnabled = False
        self.segSelector.renameEnabled = True
        self.segSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Segmentation:", self.segSelector)

        self.boneCombo = qt.QComboBox()
        self.boneCombo.setToolTip("Segment that represents drillable bone")
        form.addRow("Bone segment:", self.boneCombo)

        self.criticalList = qt.QListWidget()
        self.criticalList.setMaximumHeight(110)
        self.criticalList.setToolTip("Structures to keep away from")
        form.addRow("Critical structures:", self.criticalList)

        self.dangerSpin = qt.QDoubleSpinBox()
        self.dangerSpin.setRange(0.1, 10.0)
        self.dangerSpin.setSingleStep(0.1)
        self.dangerSpin.setValue(1.0)
        self.dangerSpin.setSuffix(" mm")
        self.warningSpin = qt.QDoubleSpinBox()
        self.warningSpin.setRange(0.2, 20.0)
        self.warningSpin.setSingleStep(0.1)
        self.warningSpin.setValue(2.0)
        self.warningSpin.setSuffix(" mm")
        thr = qt.QHBoxLayout()
        thr.addWidget(qt.QLabel("danger <"))
        thr.addWidget(self.dangerSpin)
        thr.addWidget(qt.QLabel("warning <"))
        thr.addWidget(self.warningSpin)
        form.addRow("Thresholds:", thr)

        self.phantomButton = qt.QPushButton("Load demo phantom")
        self.phantomButton.setToolTip("Synthetic temporal-bone-like block to try the module")
        form.addRow(self.phantomButton)

        # ---------------- 1. Segmentation
        segBox = ctk.ctkCollapsibleButton()
        segBox.text = "1. Automatic segmentation (nnU-Net, optional)"
        segBox.collapsed = True
        self.layout.addWidget(segBox)
        sform = qt.QFormLayout(segBox)
        self.modelPath = ctk.ctkPathLineEdit()
        self.modelPath.filters = ctk.ctkPathLineEdit.Dirs
        self.modelPath.settingKey = "TemporalBoneSafety/nnUNetModelFolder"
        self.modelPath.setToolTip("Trained nnU-Net v2 model folder (contains dataset.json and fold_*)")
        sform.addRow("Model folder:", self.modelPath)
        self.nnunetCmd = qt.QLineEdit("nnUNetv2_predict_from_modelfolder")
        self.nnunetCmd.setToolTip("Command or full path; can live in a separate Python environment")
        sform.addRow("Command:", self.nnunetCmd)
        self.foldsEdit = qt.QLineEdit("all")
        sform.addRow("Folds:", self.foldsEdit)
        self.autoSegButton = qt.QPushButton("Run automatic segmentation")
        sform.addRow(self.autoSegButton)

        # ---------------- 2. Safety-margin map
        mapBox = ctk.ctkCollapsibleButton()
        mapBox.text = "2. Safety-margin map"
        self.layout.addWidget(mapBox)
        mform = qt.QFormLayout(mapBox)
        self.zonesCheck = qt.QCheckBox("Add danger / warning bone zones as segments")
        self.zonesCheck.checked = True
        mform.addRow(self.zonesCheck)
        self.smoothCheck = qt.QCheckBox("Smooth colored surface")
        self.smoothCheck.checked = True
        mform.addRow(self.smoothCheck)
        self.computeButton = qt.QPushButton("Compute safety map")
        self.computeButton.toolTip = "Distance maps to all checked structures; colors the bone surface"
        mform.addRow(self.computeButton)
        self.cursorLabel = qt.QLabel("Cursor: -")
        self.cursorLabel.setWordWrap(True)
        mform.addRow(self.cursorLabel)

        # ---------------- 3. Virtual drilling
        drillBox = ctk.ctkCollapsibleButton()
        drillBox.text = "3. Virtual drilling (rehearsal)"
        self.layout.addWidget(drillBox)
        dform = qt.QFormLayout(drillBox)
        self.burrSpin = qt.QDoubleSpinBox()
        self.burrSpin.setRange(0.5, 8.0)
        self.burrSpin.setSingleStep(0.5)
        self.burrSpin.setValue(3.0)
        self.burrSpin.setSuffix(" mm")
        dform.addRow("Burr diameter:", self.burrSpin)
        self.startRehearsalButton = qt.QPushButton("Start rehearsal (snapshot bone)")
        dform.addRow(self.startRehearsalButton)
        self.drillButton = qt.QPushButton("Drill mode")
        self.drillButton.checkable = True
        self.drillButton.toolTip = ("Drag the 'Drill' point in 2D/3D views; bone inside "
                                    "the burr is removed. Segment Editor erasing also counts.")
        dform.addRow(self.drillButton)
        self.drillLabel = qt.QLabel("Burr: -")
        self.drillLabel.setWordWrap(True)
        dform.addRow(self.drillLabel)
        rowB = qt.QHBoxLayout()
        self.reportButton = qt.QPushButton("Drilling report")
        self.exportButton = qt.QPushButton("Export CSV...")
        self.restoreButton = qt.QPushButton("Restore bone")
        rowB.addWidget(self.reportButton)
        rowB.addWidget(self.exportButton)
        rowB.addWidget(self.restoreButton)
        dform.addRow(rowB)

        # ---------------- 4. Tracking
        trackBox = ctk.ctkCollapsibleButton()
        trackBox.text = "4. Drill-tip tracking (intraoperative)"
        trackBox.collapsed = True
        self.layout.addWidget(trackBox)
        tform = qt.QFormLayout(trackBox)
        self.toolSelector = slicer.qMRMLNodeComboBox()
        self.toolSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
        self.toolSelector.noneEnabled = True
        self.toolSelector.addEnabled = False
        self.toolSelector.removeEnabled = False
        self.toolSelector.setMRMLScene(slicer.mrmlScene)
        self.toolSelector.setToolTip("Drill tip transform in image (RAS) space, "
                                     "e.g. DrillTipToRAS from OpenIGTLink + pivot calibration")
        tform.addRow("Tool tip transform:", self.toolSelector)
        self.toolDrillsCheck = qt.QCheckBox("Tool removes virtual bone")
        tform.addRow(self.toolDrillsCheck)
        self.soundCheck = qt.QCheckBox("Beep in danger zone")
        self.soundCheck.checked = True
        tform.addRow(self.soundCheck)
        self.trackButton = qt.QPushButton("Start tracking")
        self.trackButton.checkable = True
        tform.addRow(self.trackButton)
        self.trackLabel = qt.QLabel("-")
        f = self.trackLabel.font
        f.setPointSize(16)
        f.setBold(True)
        self.trackLabel.setFont(f)
        self.trackLabel.setAlignment(qt.Qt.AlignCenter)
        self.trackLabel.setMinimumHeight(48)
        tform.addRow(self.trackLabel)
        self.trackReportButton = qt.QPushButton("Tracking session report")
        tform.addRow(self.trackReportButton)

        self.statusLabel = qt.QLabel("")
        self.statusLabel.setWordWrap(True)
        self.layout.addWidget(self.statusLabel)
        self.layout.addStretch(1)

        # ---------------- connections
        self.segSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSegmentationChanged)
        self.volumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._updateButtons)
        self.phantomButton.connect("clicked()", self.onPhantom)
        self.autoSegButton.connect("clicked()", self.onAutoSeg)
        self.computeButton.connect("clicked()", self.onCompute)
        self.startRehearsalButton.connect("clicked()", self.onStartRehearsal)
        self.drillButton.connect("toggled(bool)", self.onDrillToggled)
        self.reportButton.connect("clicked()", self.onReport)
        self.exportButton.connect("clicked()", self.onExport)
        self.restoreButton.connect("clicked()", self.onRestore)
        self.trackButton.connect("toggled(bool)", self.onTrackToggled)
        self.trackReportButton.connect("clicked()", self.onTrackReport)
        self.burrSpin.connect("valueChanged(double)", self.logic.setBurrDiameter)

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self._installCrosshairObserver()
        self.logic.drillCallback = self._onDrillUpdate
        self.logic.trackCallback = self._onTrackUpdate
        self._updateButtons()

    def cleanup(self):
        self.removeObservers()
        if self._crosshair and self._crosshairTag:
            self._crosshair.RemoveObserver(self._crosshairTag)
        if self.logic:
            self.logic.stopDrilling()
            self.logic.stopTracking()

    def onSceneStartClose(self, caller, event):
        self.drillButton.checked = False
        self.trackButton.checked = False
        self.logic.reset()

    # ------------------------------------------------------------ helpers
    def _installCrosshairObserver(self):
        self._crosshair = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLCrosshairNode")
        if self._crosshair:
            self._crosshairTag = self._crosshair.AddObserver(
                slicer.vtkMRMLCrosshairNode.CursorPositionModifiedEvent, self._onCursorMoved)

    def _onCursorMoved(self, caller, event):
        if not self.logic.hasSafetyMap():
            return
        ras = [0.0, 0.0, 0.0]
        if not caller.GetCursorPositionRAS(ras):
            return
        name, dist = self.logic.queryWorld(ras)
        if name is None:
            self.cursorLabel.text = "Cursor: outside volume"
            return
        zone = core.zone_label(dist, self.dangerSpin.value, self.warningSpin.value)
        self.cursorLabel.text = "Cursor: %s %.1f mm (%s)" % (name, dist, zone)

    def _onLogicStatus(self, text):
        self.statusLabel.text = text
        slicer.app.processEvents()

    def _zoneStyle(self, zone):
        return {"breach": "background-color:#7f0000;color:white;",
                "danger": "background-color:#d62728;color:white;",
                "warning": "background-color:#ffbf00;color:black;",
                "safe": "background-color:#2ca02c;color:white;"}.get(zone, "")

    def _onDrillUpdate(self, name, clearance):
        zone = core.zone_label(clearance, self.dangerSpin.value, self.warningSpin.value)
        self.drillLabel.text = "Burr: %s %.1f mm clearance (%s) | breaches: %d" % (
            name, clearance, zone, self.logic.breachCount)
        self.drillLabel.setStyleSheet(self._zoneStyle(zone))

    def _onTrackUpdate(self, name, dist):
        zone = core.zone_label(dist, self.dangerSpin.value, self.warningSpin.value)
        if name is None:
            self.trackLabel.text = "outside volume"
            self.trackLabel.setStyleSheet("")
            return
        self.trackLabel.text = "%s  %.1f mm" % (name, dist)
        self.trackLabel.setStyleSheet(self._zoneStyle(zone))
        if self.soundCheck.checked and zone in ("danger", "breach"):
            self.logic.beepThrottled()

    def onSegmentationChanged(self, node):
        if self._observedSegmentation:
            self.removeObserver(self._observedSegmentation,
                                slicer.vtkSegmentation.SegmentAdded, self._refreshSegments)
            self.removeObserver(self._observedSegmentation,
                                slicer.vtkSegmentation.SegmentRemoved, self._refreshSegments)
            self.removeObserver(self._observedSegmentation,
                                slicer.vtkSegmentation.SegmentModified, self._refreshSegments)
            self._observedSegmentation = None
        if node:
            seg = node.GetSegmentation()
            self.addObserver(seg, slicer.vtkSegmentation.SegmentAdded, self._refreshSegments)
            self.addObserver(seg, slicer.vtkSegmentation.SegmentRemoved, self._refreshSegments)
            self.addObserver(seg, slicer.vtkSegmentation.SegmentModified, self._refreshSegments)
            self._observedSegmentation = seg
        self._refreshSegments()

    def _refreshSegments(self, *args):
        segNode = self.segSelector.currentNode()
        prevBone = self.boneCombo.currentData
        prevChecked = {self.criticalList.item(r).data(qt.Qt.UserRole)
                       for r in range(self.criticalList.count)
                       if self.criticalList.item(r).checkState() == qt.Qt.Checked}
        hadItems = self.criticalList.count > 0
        self.boneCombo.blockSignals(True)
        self.boneCombo.clear()
        self.criticalList.clear()
        if segNode:
            segs = segNode.GetSegmentation()
            for n in range(segs.GetNumberOfSegments()):
                sid = segs.GetNthSegmentID(n)
                name = segs.GetSegment(sid).GetName()
                if name.startswith(GENERATED_PREFIX):
                    continue
                self.boneCombo.addItem(name, sid)
                item = qt.QListWidgetItem(name)
                item.setData(qt.Qt.UserRole, sid)
                item.setFlags(item.flags() | qt.Qt.ItemIsUserCheckable)
                low = name.lower()
                if hadItems and sid in prevChecked:
                    checked = True
                elif hadItems:
                    checked = False
                else:
                    checked = any(k in low for k in CRITICAL_KEYWORDS) and \
                        not any(k in low for k in BONE_KEYWORDS)
                item.setCheckState(qt.Qt.Checked if checked else qt.Qt.Unchecked)
                self.criticalList.addItem(item)
            idx = self.boneCombo.findData(prevBone) if prevBone else -1
            if idx < 0:
                for n in range(self.boneCombo.count):
                    if any(k in self.boneCombo.itemText(n).lower() for k in BONE_KEYWORDS):
                        idx = n
                        break
            if idx >= 0:
                self.boneCombo.setCurrentIndex(idx)
        self.boneCombo.blockSignals(False)
        self._updateButtons()

    def _criticalIds(self):
        return [self.criticalList.item(r).data(qt.Qt.UserRole)
                for r in range(self.criticalList.count)
                if self.criticalList.item(r).checkState() == qt.Qt.Checked]

    def _updateButtons(self, *args):
        haveInputs = bool(self.volumeSelector.currentNode() and self.segSelector.currentNode())
        self.computeButton.enabled = haveInputs and self.boneCombo.count > 0
        self.autoSegButton.enabled = bool(self.volumeSelector.currentNode())
        haveMap = self.logic.hasSafetyMap() if self.logic else False
        for b in (self.startRehearsalButton, self.drillButton, self.reportButton,
                  self.restoreButton, self.trackButton):
            b.enabled = haveMap
        self.exportButton.enabled = self.logic is not None and self.logic.reportTable is not None

    def _configureLogic(self):
        self.logic.configure(
            self.volumeSelector.currentNode(), self.segSelector.currentNode(),
            self.boneCombo.currentData, self._criticalIds(),
            self.dangerSpin.value, self.warningSpin.value)

    # ------------------------------------------------------------ actions
    def onPhantom(self):
        with slicer.util.tryWithErrorDisplay("Failed to create phantom.", waitCursor=True):
            vol, seg = self.logic.createPhantom()
            self.volumeSelector.setCurrentNode(vol)
            self.segSelector.setCurrentNode(seg)

    def onAutoSeg(self):
        vol = self.volumeSelector.currentNode()
        segNode = self.segSelector.currentNode()
        with slicer.util.tryWithErrorDisplay("Automatic segmentation failed.", waitCursor=True):
            segNode = self.logic.runNnUNet(
                vol, segNode, self.modelPath.currentPath,
                self.nnunetCmd.text.strip(), self.foldsEdit.text.split())
            self.segSelector.setCurrentNode(segNode)
            self._refreshSegments()

    def onCompute(self):
        if self.drillButton.checked:
            self.drillButton.checked = False
        with slicer.util.tryWithErrorDisplay("Failed to compute safety map.", waitCursor=True):
            if not self._criticalIds():
                raise ValueError("Check at least one critical structure.")
            self._configureLogic()
            self.logic.computeSafetyMap(addZones=self.zonesCheck.checked,
                                        smooth=self.smoothCheck.checked)
        self._updateButtons()

    def onStartRehearsal(self):
        with slicer.util.tryWithErrorDisplay("Failed to start rehearsal.", waitCursor=True):
            self.logic.startRehearsal()
        self._updateButtons()

    def onDrillToggled(self, on):
        if on:
            try:
                self.logic.setBurrDiameter(self.burrSpin.value)
                self.logic.startDrilling()
                self.drillButton.text = "Drill mode (ON) - drag the Drill point"
            except Exception as e:
                slicer.util.errorDisplay("Cannot start drilling: %s" % e)
                self.drillButton.checked = False
        else:
            self.logic.stopDrilling()
            self.drillButton.text = "Drill mode"

    def onReport(self):
        with slicer.util.tryWithErrorDisplay("Failed to build report.", waitCursor=True):
            self.logic.buildDrillingReport()
            self.logic.showTable(self.logic.reportTable)
        self._updateButtons()

    def onExport(self):
        table = self.logic.reportTable
        if table is None:
            return
        path = qt.QFileDialog.getSaveFileName(
            None, "Export report", table.GetName() + ".csv", "CSV (*.csv)")
        if path:
            slicer.util.saveNode(table, path)
            self.statusLabel.text = "Saved %s" % path

    def onRestore(self):
        if not slicer.util.confirmOkCancelDisplay(
                "Restore the bone segment to the rehearsal snapshot?"):
            return
        with slicer.util.tryWithErrorDisplay("Failed to restore.", waitCursor=True):
            self.logic.restoreBone()

    def onTrackToggled(self, on):
        if on:
            tool = self.toolSelector.currentNode()
            if not tool:
                slicer.util.errorDisplay("Select a tool tip transform first.")
                self.trackButton.checked = False
                return
            try:
                self.logic.setBurrDiameter(self.burrSpin.value)
                self.logic.startTracking(tool, removeBone=self.toolDrillsCheck.checked)
            except Exception as e:
                slicer.util.errorDisplay("Cannot start tracking: %s" % e)
                self.trackButton.checked = False
                return
            self.trackButton.text = "Stop tracking"
        else:
            self.logic.stopTracking()
            self.trackButton.text = "Start tracking"

    def onTrackReport(self):
        with slicer.util.tryWithErrorDisplay("Failed to build tracking report.", waitCursor=True):
            self.logic.buildTrackingReport()
            self.logic.showTable(self.logic.reportTable)
        self._updateButtons()


# =============================================================================
# Logic
# =============================================================================

class TemporalBoneSafetyLogic(ScriptedLoadableModuleLogic):

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.statusCallback = None
        self.drillCallback = None
        self.trackCallback = None
        self.reset()

    def reset(self):
        self.volumeNode = None
        self.segNode = None
        self.boneId = None
        self.criticalIds = []
        self.criticalNames = []
        self.dangerMm = 1.0
        self.warningMm = 2.0
        self.spacing = None
        self.distance = None      # float32 kji, signed distance to nearest structure
        self.nearest = None       # int16 kji, index into criticalNames
        self.boneArray = None     # current bone mask (uint8 kji)
        self.originalBone = None  # rehearsal snapshot
        self.surfaceModel = None
        self.colorNode = None
        self.drillNode = None
        self._drillTag = None
        self._prevDrillIjk = None
        self.drillLog = []        # (t, x, y, z, clearance, name)
        self.trackLog = []
        self.breachCount = 0
        self.burrRadius = 1.5
        self.toolNode = None
        self._toolTag = None
        self._toolRemovesBone = False
        self._prevToolIjk = None
        self._lastBeep = 0.0
        self.reportTable = None
        self._flushTimer = None
        self._surfaceTimer = None
        self._smooth = True

    # ------------------------------------------------------------ status
    def _status(self, text):
        if self.statusCallback:
            self.statusCallback(text)
        else:
            print(text)

    # ------------------------------------------------------------ config
    def configure(self, volumeNode, segNode, boneId, criticalIds, dangerMm, warningMm):
        if not volumeNode or not segNode:
            raise ValueError("Select a CT volume and a segmentation.")
        if not boneId:
            raise ValueError("Select the bone segment.")
        if boneId in criticalIds:
            raise ValueError("The bone segment cannot also be a critical structure.")
        if warningMm <= dangerMm:
            raise ValueError("Warning threshold must be larger than danger threshold.")
        self.volumeNode = volumeNode
        self.segNode = segNode
        self.boneId = boneId
        self.criticalIds = list(criticalIds)
        seg = segNode.GetSegmentation()
        self.criticalNames = [seg.GetSegment(s).GetName() for s in self.criticalIds]
        self.dangerMm = float(dangerMm)
        self.warningMm = float(warningMm)
        self.spacing = tuple(volumeNode.GetSpacing())

    def setBurrDiameter(self, diameterMm):
        self.burrRadius = float(diameterMm) / 2.0
        if self.drillNode and self.drillNode.GetDisplayNode():
            self.drillNode.GetDisplayNode().SetGlyphSize(float(diameterMm))

    def hasSafetyMap(self):
        return self.distance is not None

    # ------------------------------------------------------------ geometry
    def _segmentArray(self, segmentId):
        return slicer.util.arrayFromSegmentBinaryLabelmap(
            self.segNode, segmentId, self.volumeNode).astype(np.uint8)

    def _writeSegmentArray(self, segmentId, arr):
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            arr.astype(np.uint8), self.segNode, segmentId, self.volumeNode)

    def _worldToIjkTransform(self):
        t = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            None, self.volumeNode.GetParentTransformNode(), t)
        m = vtk.vtkMatrix4x4()
        self.volumeNode.GetRASToIJKMatrix(m)
        t.PostMultiply()
        t.Concatenate(m)
        return t

    def worldToIjk(self, ras):
        return list(self._worldToIjkTransform().TransformPoint(ras[:3]))

    def queryWorld(self, ras):
        """Nearest structure name and signed distance (mm) at a world point."""
        if self.distance is None:
            return None, None
        ijk = self.worldToIjk(ras)
        d = core.trilinear_sample(self.distance, [ijk])[0]
        if not np.isfinite(d):
            return None, None
        idx = core.nearest_sample(self.nearest, ijk)
        name = self.criticalNames[idx] if 0 <= idx < len(self.criticalNames) else "?"
        return name, float(d)

    # ------------------------------------------------------------ phantom
    def createPhantom(self):
        spacing = (0.3, 0.3, 0.3)
        ct, masks = core.make_phantom(spacing_xyz=spacing)
        ijkToRas = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
        vol = slicer.util.addVolumeFromArray(ct, ijkToRas, "TBS phantom CT")
        vol.CreateDefaultDisplayNodes()
        vol.GetDisplayNode().SetAutoWindowLevel(False)
        vol.GetDisplayNode().SetWindowLevel(4000, 400)
        segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "TBS phantom segmentation")
        segNode.CreateDefaultDisplayNodes()
        segNode.SetReferenceImageGeometryParameterFromVolumeNode(vol)
        colors = {"Bone": (0.95, 0.92, 0.80), "Facial nerve": (1.0, 0.85, 0.0),
                  "Sigmoid sinus": (0.2, 0.3, 0.9), "Dura": (0.8, 0.4, 0.6),
                  "Labyrinth": (0.9, 0.2, 0.2)}
        for name, m in masks.items():
            sid = segNode.GetSegmentation().AddEmptySegment(name, name, colors.get(name, (0.5, 0.5, 0.5)))
            slicer.util.updateSegmentBinaryLabelmapFromArray(m, segNode, sid, vol)
        segNode.CreateClosedSurfaceRepresentation()
        slicer.util.setSliceViewerLayers(background=vol, fit=True)
        self._status("Demo phantom loaded.")
        return vol, segNode

    # ------------------------------------------------------------ nnU-Net
    def runNnUNet(self, volumeNode, segNode, modelFolder, command, folds):
        if not volumeNode:
            raise ValueError("Select a CT volume.")
        if not modelFolder or not os.path.isdir(modelFolder):
            raise ValueError("Select a valid nnU-Net model folder.")
        dsJson = os.path.join(modelFolder, "dataset.json")
        if not os.path.exists(dsJson):
            raise ValueError("dataset.json not found in model folder.")
        with open(dsJson) as f:
            labels = json.load(f).get("labels", {})
        labelItems = [(name, int(v)) for name, v in labels.items()
                      if not isinstance(v, (list, tuple)) and int(v) != 0]
        if not labelItems:
            raise ValueError("No foreground labels in dataset.json.")

        tmp = tempfile.mkdtemp(prefix="TBS_nnunet_")
        inDir, outDir = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        os.makedirs(inDir)
        os.makedirs(outDir)
        inPath = os.path.join(inDir, "case_0000.nii.gz")
        if not slicer.util.saveNode(volumeNode, inPath):
            raise RuntimeError("Could not export CT to %s" % inPath)
        cmd = [command, "-i", inDir, "-o", outDir, "-m", modelFolder, "-f"] + (folds or ["all"])
        self._status("Running: %s" % " ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True)
        lines = []
        while True:
            line = proc.stdout.readline()
            if line:
                lines.append(line.rstrip())
                self._status("nnU-Net: " + line.strip()[-120:])
            elif proc.poll() is not None:
                break
            slicer.app.processEvents()
        if proc.returncode != 0:
            raise RuntimeError("nnU-Net failed (code %d):\n%s" % (proc.returncode, "\n".join(lines[-20:])))
        outPath = os.path.join(outDir, "case.nii.gz")
        if not os.path.exists(outPath):
            raise RuntimeError("nnU-Net produced no output file.")

        labelNode = slicer.util.loadLabelVolume(outPath)
        try:
            arr = slicer.util.arrayFromVolume(labelNode)
            if arr.shape != slicer.util.arrayFromVolume(volumeNode).shape:
                raise RuntimeError("Output grid does not match the CT grid.")
            if segNode is None:
                segNode = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLSegmentationNode", volumeNode.GetName() + " auto segmentation")
                segNode.CreateDefaultDisplayNodes()
            segNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
            seg = segNode.GetSegmentation()
            for name, value in labelItems:
                sid = seg.GetSegmentIdBySegmentName(name)
                if not sid:
                    sid = seg.AddEmptySegment("", name)
                slicer.util.updateSegmentBinaryLabelmapFromArray(
                    (arr == value).astype(np.uint8), segNode, sid, volumeNode)
            segNode.CreateClosedSurfaceRepresentation()
        finally:
            slicer.mrmlScene.RemoveNode(labelNode)
        self._status("Automatic segmentation done: %d labels." % len(labelItems))
        return segNode

    # ------------------------------------------------------------ safety map
    def computeSafetyMap(self, addZones=True, smooth=True):
        self._smooth = smooth
        self._status("Reading segments...")
        masks = []
        for sid, name in zip(self.criticalIds, self.criticalNames):
            m = self._segmentArray(sid)
            if not m.any():
                self._status("Warning: '%s' is empty and is ignored." % name)
            masks.append(m)
        if not any(m.any() for m in masks):
            raise ValueError("All selected critical structures are empty.")
        self._status("Computing distance maps (%d structures)..." % len(masks))
        self.distance, self.nearest = core.combined_distance(masks, self.spacing)
        self.boneArray = self._segmentArray(self.boneId)
        if self.originalBone is None or self.originalBone.shape != self.boneArray.shape:
            self.originalBone = self.boneArray.copy()
        if addZones:
            self._updateZoneSegments()
        self._status("Building colored bone surface...")
        self.updateSurface()
        self._status("Safety map ready. Hover in any view to read the clearance.")

    def _zoneSegmentId(self, label, color):
        seg = self.segNode.GetSegmentation()
        name = GENERATED_PREFIX + label
        sid = seg.GetSegmentIdBySegmentName(name)
        if not sid:
            sid = seg.AddEmptySegment("", name, color)
        return sid

    def _updateZoneSegments(self):
        bone = self.boneArray > 0
        d = self.distance
        danger = bone & (d < self.dangerMm)
        warning = bone & (d >= self.dangerMm) & (d < self.warningMm)
        sidD = self._zoneSegmentId("danger bone < %.1f mm" % self.dangerMm, (0.84, 0.15, 0.16))
        sidW = self._zoneSegmentId("warning bone < %.1f mm" % self.warningMm, (1.0, 0.75, 0.0))
        # remove stale zone segments with other thresholds
        seg = self.segNode.GetSegmentation()
        for n in reversed(range(seg.GetNumberOfSegments())):
            sid = seg.GetNthSegmentID(n)
            if seg.GetSegment(sid).GetName().startswith(GENERATED_PREFIX) and sid not in (sidD, sidW):
                seg.RemoveSegment(sid)
        self._writeSegmentArray(sidD, danger)
        self._writeSegmentArray(sidW, warning)
        disp = self.segNode.GetDisplayNode()
        if disp:
            for sid in (sidD, sidW):
                disp.SetSegmentOpacity3D(sid, 0.0)  # shown in 2D; 3D uses the colored surface
                disp.SetSegmentOpacity2DFill(sid, 0.35)

    def _ensureColorNode(self):
        if self.colorNode is None or self.colorNode.GetScene() is None:
            self.colorNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLProceduralColorNode", "TBS clearance colors")
            self.colorNode.SetAttribute("Category", "TemporalBoneSafety")
        ctf = self.colorNode.GetColorTransferFunction()
        ctf.RemoveAllPoints()
        ctf.AddRGBPoint(0.0, 0.50, 0.0, 0.0)
        ctf.AddRGBPoint(self.dangerMm, 0.84, 0.15, 0.16)
        ctf.AddRGBPoint(self.warningMm, 1.0, 0.75, 0.0)
        ctf.AddRGBPoint(2.0 * self.warningMm, 0.17, 0.63, 0.17)
        ctf.AddRGBPoint(4.0 * self.warningMm, 0.85, 0.85, 0.80)
        return self.colorNode

    def updateSurface(self):
        """Marching-cubes surface of the current bone, colored by clearance."""
        if self.boneArray is None:
            return
        nk, nj, ni = self.boneArray.shape
        img = vtk.vtkImageData()
        img.SetDimensions(ni, nj, nk)
        img.SetSpacing(1, 1, 1)
        img.SetOrigin(0, 0, 0)
        vals = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(self.boneArray, dtype=np.uint8).ravel(),
            deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        img.GetPointData().SetScalars(vals)

        mc = vtk.vtkFlyingEdges3D()
        mc.SetInputData(img)
        mc.SetValue(0, 0.5)
        mc.ComputeNormalsOff()
        last = mc
        if self._smooth:
            sm = vtk.vtkWindowedSincPolyDataFilter()
            sm.SetInputConnection(mc.GetOutputPort())
            sm.SetNumberOfIterations(15)
            sm.SetPassBand(0.1)
            sm.NonManifoldSmoothingOn()
            sm.NormalizeCoordinatesOn()
            last = sm
        last.Update()
        polyIjk = vtk.vtkPolyData()
        polyIjk.DeepCopy(last.GetOutput())

        if polyIjk.GetNumberOfPoints() > 0:
            pts = numpy_support.vtk_to_numpy(polyIjk.GetPoints().GetData())
            clearance = core.trilinear_sample(self.distance, pts).astype(np.float32)
            clearance[~np.isfinite(clearance)] = 1e3
            arr = numpy_support.numpy_to_vtk(clearance, deep=True)
            arr.SetName("ClearanceMm")
            polyIjk.GetPointData().AddArray(arr)
            polyIjk.GetPointData().SetActiveScalars("ClearanceMm")

        ijkToRas = vtk.vtkMatrix4x4()
        self.volumeNode.GetIJKToRASMatrix(ijkToRas)
        tf = vtk.vtkTransform()
        tf.SetMatrix(ijkToRas)
        tpd = vtk.vtkTransformPolyDataFilter()
        tpd.SetInputData(polyIjk)
        tpd.SetTransform(tf)
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(tpd.GetOutputPort())
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOff()
        normals.Update()
        poly = normals.GetOutput()

        if self.surfaceModel is None or self.surfaceModel.GetScene() is None:
            self.surfaceModel = slicer.modules.models.logic().AddModel(poly)
            self.surfaceModel.SetName("TBS bone clearance surface")
        else:
            self.surfaceModel.SetAndObservePolyData(poly)
        self.surfaceModel.SetAndObserveTransformNodeID(self.volumeNode.GetTransformNodeID())
        disp = self.surfaceModel.GetDisplayNode()
        colorNode = self._ensureColorNode()
        disp.SetAndObserveColorNodeID(colorNode.GetID())
        try:
            disp.SetActiveScalar("ClearanceMm", vtk.vtkAssignAttribute.POINT_DATA)
        except AttributeError:
            disp.SetActiveScalarName("ClearanceMm")
        disp.SetScalarVisibility(True)
        disp.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
        disp.SetScalarRange(0.0, 4.0 * self.warningMm)
        disp.SetVisibility2D(False)
        segDisp = self.segNode.GetDisplayNode()
        if segDisp:
            segDisp.SetSegmentVisibility3D(self.boneId, False)

    # ------------------------------------------------------------ rehearsal
    def startRehearsal(self):
        self._requireMap()
        self.boneArray = self._segmentArray(self.boneId)
        self.originalBone = self.boneArray.copy()
        self.drillLog = []
        self.breachCount = 0
        self._status("Rehearsal started: bone snapshot taken (%d voxels)." % int(self.originalBone.sum()))

    def restoreBone(self):
        self._requireMap()
        if self.originalBone is None:
            raise ValueError("No snapshot. Start a rehearsal first.")
        self.boneArray = self.originalBone.copy()
        self._writeSegmentArray(self.boneId, self.boneArray)
        self.drillLog = []
        self.breachCount = 0
        self.updateSurface()
        self._status("Bone restored to snapshot.")

    def _requireMap(self):
        if self.distance is None:
            raise ValueError("Compute the safety map first.")

    def _timers(self):
        if self._flushTimer is None:
            self._flushTimer = qt.QTimer()
            self._flushTimer.setSingleShot(True)
            self._flushTimer.setInterval(150)
            self._flushTimer.connect("timeout()", self.flushBone)
            self._surfaceTimer = qt.QTimer()
            self._surfaceTimer.setSingleShot(True)
            self._surfaceTimer.setInterval(800)
            self._surfaceTimer.connect("timeout()", self.updateSurface)

    def _scheduleFlush(self):
        self._timers()
        self._flushTimer.start()
        self._surfaceTimer.start()

    def flushBone(self):
        if self.boneArray is not None and self.segNode is not None:
            self._writeSegmentArray(self.boneId, self.boneArray)

    def startDrilling(self):
        self._requireMap()
        if self.originalBone is None:
            self.startRehearsal()
        # pick up edits made with Segment Editor since the last read
        self.boneArray = self._segmentArray(self.boneId)
        if self.drillNode is None or self.drillNode.GetScene() is None:
            self.drillNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Drill")
            self.drillNode.CreateDefaultDisplayNodes()
            d = self.drillNode.GetDisplayNode()
            d.SetGlyphType(slicer.vtkMRMLMarkupsDisplayNode.Sphere3D)
            d.SetUseGlyphScale(False)
            d.SetSelectedColor(0.2, 0.8, 1.0)
            d.SetPointLabelsVisibility(False)
            self.drillNode.SetMaximumNumberOfControlPoints(1)
        if self.drillNode.GetNumberOfControlPoints() == 0:
            # start above the bone, at the centre of the top surface
            ijk = self._defaultDrillStartIjk()
            ras = self._ijkToWorld(ijk)
            self.drillNode.AddControlPointWorld(vtk.vtkVector3d(ras[0], ras[1], ras[2]))
        self.drillNode.GetDisplayNode().SetGlyphSize(2.0 * self.burrRadius)
        self.drillNode.SetLocked(False)
        self._prevDrillIjk = None
        self._drillTag = self.drillNode.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointModifiedEvent, self._onDrillMoved)
        self._status("Drill mode on: drag the Drill point. Bone inside the burr is removed.")

    def stopDrilling(self):
        if self.drillNode and self._drillTag:
            self.drillNode.RemoveObserver(self._drillTag)
        self._drillTag = None
        self._prevDrillIjk = None
        if self._flushTimer is not None:
            self._flushTimer.stop()
            self.flushBone()

    def _defaultDrillStartIjk(self):
        nk, nj, ni = self.boneArray.shape
        ks = np.where(self.boneArray.any(axis=(1, 2)))[0]
        k = (ks.max() + 3) if len(ks) else nk // 2
        return [ni / 2.0, nj / 2.0, float(min(k, nk - 1))]

    def _ijkToWorld(self, ijk):
        t = vtk.vtkGeneralTransform()
        t.DeepCopy(self._worldToIjkTransform())
        t.Inverse()
        return list(t.TransformPoint(ijk[:3]))

    def liftDrill(self):
        """Lift the burr: the next drill position starts a new stroke instead of
        carving a straight tunnel from the previous position."""
        self._prevDrillIjk = None

    def _onDrillMoved(self, caller, event):
        pos = [0.0, 0.0, 0.0]
        caller.GetNthControlPointPositionWorld(0, pos)
        self.drillAtWorld(pos, log=self.drillLog, prevAttr="_prevDrillIjk",
                          callback=self.drillCallback)

    def drillAtWorld(self, ras, log, prevAttr, callback=None, remove=True):
        """Remove bone along the path from the previous position to ras."""
        ijk = self.worldToIjk(ras)
        prev = getattr(self, prevAttr)
        if remove:
            if prev is None:
                removed = core.carve_sphere(self.boneArray, ijk, self.burrRadius, self.spacing)
            else:
                removed = core.carve_path(self.boneArray, prev, ijk, self.burrRadius, self.spacing)
            if removed:
                self._scheduleFlush()
        setattr(self, prevAttr, ijk)
        name, d = self.queryWorld(ras)
        if name is None:
            clearance = float("inf")
        else:
            clearance = d - (self.burrRadius if remove else 0.0)
        if remove and clearance <= 0 and (not log or log[-1][4] > 0):
            self.breachCount += 1
        log.append((time.time(), ras[0], ras[1], ras[2], clearance, name or ""))
        if callback:
            callback(name, clearance)
        return name, clearance

    # ------------------------------------------------------------ tracking
    def startTracking(self, toolNode, removeBone=False):
        self._requireMap()
        self.stopTracking()
        self.toolNode = toolNode
        self._toolRemovesBone = bool(removeBone)
        if removeBone:
            if self.originalBone is None:
                self.startRehearsal()
            self.boneArray = self._segmentArray(self.boneId)
        self.trackLog = []
        self._prevToolIjk = None
        self._toolTag = toolNode.AddObserver(
            slicer.vtkMRMLTransformNode.TransformModifiedEvent, self._onToolMoved)
        self._onToolMoved(toolNode, None)
        self._status("Tracking %s." % toolNode.GetName())

    def stopTracking(self):
        if self.toolNode and self._toolTag:
            self.toolNode.RemoveObserver(self._toolTag)
        self._toolTag = None
        self.toolNode = None
        if self._flushTimer is not None and self._toolRemovesBone:
            self._flushTimer.stop()
            self.flushBone()

    def toolTipWorld(self, toolNode):
        m = vtk.vtkMatrix4x4()
        toolNode.GetMatrixTransformToWorld(m)
        p = m.MultiplyPoint([0.0, 0.0, 0.0, 1.0])
        return [p[0], p[1], p[2]]

    def _onToolMoved(self, caller, event):
        ras = self.toolTipWorld(caller)
        self.drillAtWorld(ras, log=self.trackLog, prevAttr="_prevToolIjk",
                          callback=self.trackCallback, remove=self._toolRemovesBone)

    def beepThrottled(self, minInterval=0.4):
        now = time.time()
        if now - self._lastBeep >= minInterval:
            self._lastBeep = now
            qt.QApplication.beep()

    # ------------------------------------------------------------ reports
    def _newTable(self, name):
        table = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", slicer.mrmlScene.GenerateUniqueName(name))
        table.SetAttribute("Category", "TemporalBoneSafety")
        return table

    def _fillTable(self, table, header, rows):
        for h in header:
            col = table.AddColumn()
            col.SetName(h)
        for r in rows:
            n = table.AddEmptyRow()
            for c, v in enumerate(r):
                table.SetCellText(n, c, v if isinstance(v, str) else
                                  ("" if v is None else ("%.2f" % v if isinstance(v, float) else str(v))))
        table.Modified()

    def buildDrillingReport(self):
        self._requireMap()
        if self.originalBone is None:
            raise ValueError("No rehearsal snapshot.")
        self.flushBone()
        current = self._segmentArray(self.boneId)  # includes Segment Editor edits
        self.boneArray = current.copy()
        structures = {name: self._segmentArray(sid)
                      for sid, name in zip(self.criticalIds, self.criticalNames)}
        rep = core.drilling_report(self.originalBone, current, structures, self.spacing)
        log = self.drillLog
        motion = core.motion_stats([r[0] for r in log], [r[1:4] for r in log]) if log else \
            core.motion_stats([], [])
        rows = [
            ("Removed bone volume (mm3)", "", rep["removed_volume_mm3"], ""),
            ("Burr diameter (mm)", "", 2.0 * self.burrRadius, ""),
            ("Breach events (burr overlapped a structure)", "", self.breachCount, ""),
            ("Drilling duration (s)", "", motion["duration_s"], ""),
            ("Pauses >= 1 s", "", motion["pause_count"], ""),
            ("Pause time (s)", "", motion["pause_time_s"], ""),
            ("Burr path length (mm)", "", motion["path_length_mm"], ""),
        ]
        for name in self.criticalNames:
            s = rep["structures"][name]
            ca = s["closest_approach_mm"]
            zone = core.zone_label(ca, self.dangerMm, self.warningMm) if ca is not None else "not reached"
            rows.append(("Thinnest bone left (mm)", name, ca, zone))
            rows.append(("Bone removed within 1 mm (mm3)", name,
                         s["volume_within_1mm_removed_mm3"], ""))
        self.reportTable = self._newTable("TBS drilling report")
        self._fillTable(self.reportTable, ["Metric", "Structure", "Value", "Zone"], rows)
        self.updateSurface()
        self._status("Drilling report created.")
        return rep, motion

    def buildTrackingReport(self):
        log = self.trackLog
        if len(log) < 2:
            raise ValueError("Not enough tracking samples.")
        t = [r[0] for r in log]
        d = [r[4] for r in log]
        zones = core.zone_times(t, d, self.dangerMm, self.warningMm)
        motion = core.motion_stats(t, [r[1:4] for r in log])
        rows = [
            ("Samples", len(log)),
            ("Duration (s)", motion["duration_s"]),
            ("Minimum distance (mm)", zones["min_distance_mm"]),
            ("Time in danger zone (s)", zones["danger_s"]),
            ("Time in warning zone (s)", zones["warning_s"]),
            ("Pauses >= 1 s", motion["pause_count"]),
            ("Pause time (s)", motion["pause_time_s"]),
            ("Tip path length (mm)", motion["path_length_mm"]),
        ]
        self.reportTable = self._newTable("TBS tracking report")
        self._fillTable(self.reportTable, ["Metric", "Value"], rows)
        # raw log as a second table for analysis
        raw = self._newTable("TBS tracking log")
        t0 = log[0][0]
        self._fillTable(raw, ["t_s", "R", "A", "S", "distance_mm", "nearest"],
                        [(r[0] - t0, r[1], r[2], r[3], r[4], r[5]) for r in log])
        self._status("Tracking report created.")
        return zones, motion

    def showTable(self, table):
        lm = slicer.app.layoutManager()
        if lm:
            lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpTableView)
        sel = slicer.app.applicationLogic().GetSelectionNode()
        sel.SetActiveTableID(table.GetID())
        slicer.app.applicationLogic().PropagateTableSelection()


# =============================================================================
# Test
# =============================================================================

class TemporalBoneSafetyTest(ScriptedLoadableModuleTest):

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_PhantomWorkflow()

    def test_PhantomWorkflow(self):
        self.delayDisplay("Phantom workflow")
        logic = TemporalBoneSafetyLogic()
        vol, segNode = logic.createPhantom()
        seg = segNode.GetSegmentation()
        boneId = seg.GetSegmentIdBySegmentName("Bone")
        crit = [seg.GetSegmentIdBySegmentName(n)
                for n in ("Facial nerve", "Sigmoid sinus", "Dura", "Labyrinth")]
        logic.configure(vol, segNode, boneId, crit, 1.0, 2.0)
        logic.computeSafetyMap()
        self.assertTrue(logic.hasSafetyMap())
        self.assertGreater(logic.surfaceModel.GetPolyData().GetNumberOfPoints(), 0)

        # cursor query on the facial nerve centre line is negative/near zero
        _, masks = core.make_phantom()
        kji = np.argwhere(masks["Facial nerve"] > 0)[len(np.argwhere(masks["Facial nerve"] > 0)) // 2]
        ras = logic._ijkToWorld([float(kji[2]), float(kji[1]), float(kji[0])])
        name, d = logic.queryWorld(ras)
        self.assertEqual(name, "Facial nerve")
        self.assertLess(d, 0.5)

        # drill straight down toward the nerve
        logic.setBurrDiameter(2.0)
        logic.startRehearsal()
        logic.startDrilling()
        top = logic._defaultDrillStartIjk()
        target = [float(kji[2]), float(kji[1]), float(kji[0])]
        for t in np.linspace(0.0, 1.0, 25):
            ijk = [top[0] + t * (target[0] - top[0]), top[1] + t * (target[1] - top[1]),
                   top[2] + t * (target[2] - top[2])]
            logic.drillNode.SetNthControlPointPositionWorld(0, logic._ijkToWorld(ijk))
        logic.stopDrilling()
        self.assertGreater(logic.breachCount, 0)
        rep, motion = logic.buildDrillingReport()
        self.assertGreater(rep["removed_volume_mm3"], 0)
        self.assertLess(rep["structures"]["Facial nerve"]["closest_approach_mm"], 0.5)

        logic.restoreBone()
        self.assertEqual(int(logic._segmentArray(boneId).sum()), int(logic.originalBone.sum()))

        # tracking with a moving transform
        tool = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "DrillTipToRAS")
        logic.startTracking(tool)
        for t in np.linspace(0.0, 1.0, 10):
            p = logic._ijkToWorld([top[0], top[1], top[2] - t * 20])
            m = vtk.vtkMatrix4x4()
            m.SetElement(0, 3, p[0])
            m.SetElement(1, 3, p[1])
            m.SetElement(2, 3, p[2])
            tool.SetMatrixTransformToParent(m)
        logic.stopTracking()
        zones, _ = logic.buildTrackingReport()
        self.assertIsNotNone(zones["min_distance_mm"])
        self.delayDisplay("Test passed")
