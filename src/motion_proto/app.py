from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph.opengl as gl

from .bvh.loader import load_bvh
from .bvh.kinematics import eval_pose_world, get_skeleton_world
from .preprocess import (
    PreprocessConfig,
    apply_world_transform,
    load_axis_presets,
    preprocess,
)


class MotionViewer(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Motion Understanding Proto - BVH Viewer")

        self.motion_raw = None
        self.state = None
        self.skel = None
        self.frame = 0
        self.edge_items = []

        # Load presets once
        self.axis_presets = load_axis_presets()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        layout.addLayout(top)

        self.btn_load = QtWidgets.QPushButton("Load BVH")
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.lbl = QtWidgets.QLabel("No file loaded")

        top.addWidget(self.btn_load)
        top.addWidget(self.btn_play)
        top.addWidget(self.btn_pause)
        top.addWidget(self.lbl, 1)

        # Preprocess controls (M2)
        prep = QtWidgets.QHBoxLayout()
        layout.addLayout(prep)

        self.cmb_axis = QtWidgets.QComboBox()
        for pid, p in self.axis_presets.items():
            label = str(p.get("label") or pid)
            self.cmb_axis.addItem(label, pid)
        # Prefer Axis template by default if available
        if "axis_template" in self.axis_presets:
            idx = self.cmb_axis.findData("axis_template")
            if idx >= 0:
                self.cmb_axis.setCurrentIndex(idx)

        self.chk_fx = QtWidgets.QCheckBox("Flip X")
        self.chk_fy = QtWidgets.QCheckBox("Flip Y")
        self.chk_fz = QtWidgets.QCheckBox("Flip Z")

        self.cmb_fps = QtWidgets.QComboBox()
        self.cmb_fps.addItem("Keep FPS", None)
        self.cmb_fps.addItem("30 FPS", 30.0)
        self.cmb_fps.addItem("60 FPS", 60.0)

        self.cmb_scale = QtWidgets.QComboBox()
        self.cmb_scale.addItem("Scale: Auto", "auto")
        self.cmb_scale.addItem("Scale: None", "none")
        self.cmb_scale.addItem("Scale: Factor", "factor")
        self.spn_factor = QtWidgets.QDoubleSpinBox()
        self.spn_factor.setRange(1e-6, 1e6)
        self.spn_factor.setDecimals(6)
        self.spn_factor.setSingleStep(0.01)
        self.spn_factor.setValue(1.0)

        self.btn_apply = QtWidgets.QPushButton("Apply Preprocess")
        self.lbl_prep = QtWidgets.QLabel("Preprocess: (not applied)")

        prep.addWidget(QtWidgets.QLabel("Axis:"))
        prep.addWidget(self.cmb_axis)
        prep.addWidget(self.chk_fx)
        prep.addWidget(self.chk_fy)
        prep.addWidget(self.chk_fz)
        prep.addSpacing(12)
        prep.addWidget(QtWidgets.QLabel("FPS:"))
        prep.addWidget(self.cmb_fps)
        prep.addSpacing(12)
        prep.addWidget(self.cmb_scale)
        prep.addWidget(self.spn_factor)
        prep.addWidget(self.btn_apply)
        prep.addWidget(self.lbl_prep, 1)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(self.slider)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=300)
        layout.addWidget(self.view, 1)

        grid = gl.GLGridItem()
        grid.scale(10, 10, 1)
        self.view.addItem(grid)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        self.btn_load.clicked.connect(self._on_load)
        self.btn_play.clicked.connect(self._on_play)
        self.btn_pause.clicked.connect(self._on_pause)
        self.slider.valueChanged.connect(self._on_slider)
        self.btn_apply.clicked.connect(self._apply_preprocess)

        self._set_enabled(False)

    def _set_enabled(self, enabled: bool) -> None:
        self.btn_play.setEnabled(enabled)
        self.btn_pause.setEnabled(enabled)
        self.slider.setEnabled(enabled)
        self.btn_apply.setEnabled(enabled)
        self.cmb_axis.setEnabled(enabled)
        self.chk_fx.setEnabled(enabled)
        self.chk_fy.setEnabled(enabled)
        self.chk_fz.setEnabled(enabled)
        self.cmb_fps.setEnabled(enabled)
        self.cmb_scale.setEnabled(enabled)
        self.spn_factor.setEnabled(enabled)

    def _on_load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open BVH", str(Path.cwd()), "BVH files (*.bvh);;All files (*.*)"
        )
        if not path:
            return

        self.motion_raw = load_bvh(path)
        self.skel = get_skeleton_world(self.motion_raw, include_end_sites=True)

        # Apply preprocess immediately with current UI config
        self._apply_preprocess(reset_slider=True)

        # label is set in _apply_preprocess
        self._set_enabled(True)

        for it in self.edge_items:
            self.view.removeItem(it)
        self.edge_items = []

        for _ in self.skel.edges:
            item = gl.GLLinePlotItem(pos=np.zeros((2, 3)), width=2, antialias=True, mode="lines")
            self.edge_items.append(item)
            self.view.addItem(item)

        self._render_frame(0)

    def _current_preprocess_config(self) -> PreprocessConfig:
        fps = self.cmb_fps.currentData()
        scale_mode = str(self.cmb_scale.currentData())
        return PreprocessConfig(
            target_fps=float(fps) if fps is not None else None,
            scale_mode=scale_mode,
            scale_factor=float(self.spn_factor.value()),
            axis_preset_id=str(self.cmb_axis.currentData() or "none"),
            flip_x=bool(self.chk_fx.isChecked()),
            flip_y=bool(self.chk_fy.isChecked()),
            flip_z=bool(self.chk_fz.isChecked()),
        )

    def _apply_preprocess(self, reset_slider: bool = False) -> None:
        if self.motion_raw is None:
            return

        cfg = self._current_preprocess_config()
        self.state = preprocess(self.motion_raw, cfg, self.axis_presets)

        m = self.state.motion
        if reset_slider:
            self.slider.setMinimum(0)
            self.slider.setMaximum(m.frames - 1)
            self.slider.setValue(0)
            self.frame = 0
        else:
            # clamp frame
            self.frame = max(0, min(self.frame, m.frames - 1))
            self.slider.blockSignals(True)
            self.slider.setMaximum(m.frames - 1)
            self.slider.setValue(self.frame)
            self.slider.blockSignals(False)

        axis_id = cfg.axis_preset_id
        axis_label = str(self.axis_presets.get(axis_id, {}).get("label") or axis_id)
        self.lbl_prep.setText(
            f"Preprocess: axis={axis_label}, scale={self.state.scale_factor:.6g}, fps={1.0/m.frame_time:.2f}"
        )

        self.lbl.setText(
            f"{m.frames} frames | dt={m.frame_time:.4f}s | axis={axis_id} | scale={self.state.scale_factor:.6g}"
        )

        # Adjust camera distance based on current pose size
        try:
            pose0 = eval_pose_world(m, 0, include_end_sites=True)
            pts0 = np.asarray(pose0.positions, dtype=np.float64)
            pts0 = apply_world_transform(pts0, self.state.axis_matrix, self.state.scale_factor)
            rng = pts0.max(axis=0) - pts0.min(axis=0)
            size = float(np.max(np.abs(rng))) if pts0.size else 200.0
            self.view.setCameraPosition(distance=max(200.0, size * 2.0))
        except Exception:
            pass

        self._render_frame(self.frame)

    def _render_frame(self, frame_idx: int) -> None:
        if self.state is None or self.skel is None:
            return
        pose = eval_pose_world(self.state.motion, frame_idx, include_end_sites=True)
        pts = np.array(pose.positions, dtype=np.float64)
        pts = apply_world_transform(pts, self.state.axis_matrix, self.state.scale_factor)

        for eidx, (a, b) in enumerate(self.skel.edges):
            seg = np.vstack([pts[a], pts[b]])
            self.edge_items[eidx].setData(pos=seg)

    def _on_play(self) -> None:
        if self.state is None:
            return
        interval_ms = max(1, int(self.state.motion.frame_time * 1000))
        self.timer.start(interval_ms)

    def _on_pause(self) -> None:
        self.timer.stop()

    def _on_tick(self) -> None:
        if self.state is None:
            return
        self.frame = (self.frame + 1) % self.state.motion.frames
        self.slider.blockSignals(True)
        self.slider.setValue(self.frame)
        self.slider.blockSignals(False)
        self._render_frame(self.frame)

    def _on_slider(self, v: int) -> None:
        self.frame = v
        self._render_frame(self.frame)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    w = MotionViewer()
    w.resize(1100, 800)
    w.show()
    sys.exit(app.exec())
