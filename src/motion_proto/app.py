from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph.opengl as gl

from .bvh.loader import load_bvh
from .bvh.kinematics import eval_pose_world, get_skeleton_world


class MotionViewer(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Motion Understanding Proto - BVH Viewer")

        self.motion = None
        self.skel = None
        self.frame = 0
        self.edge_items = []

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

        self._set_enabled(False)

    def _set_enabled(self, enabled: bool) -> None:
        self.btn_play.setEnabled(enabled)
        self.btn_pause.setEnabled(enabled)
        self.slider.setEnabled(enabled)

    def _on_load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open BVH", str(Path.cwd()), "BVH files (*.bvh);;All files (*.*)"
        )
        if not path:
            return

        self.motion = load_bvh(path)
        self.skel = get_skeleton_world(self.motion, include_end_sites=True)

        self.slider.setMinimum(0)
        self.slider.setMaximum(self.motion.frames - 1)
        self.slider.setValue(0)
        self.frame = 0

        self.lbl.setText(
            f"{Path(path).name} | frames={self.motion.frames} | dt={self.motion.frame_time:.4f}s | channels={self.motion.total_channels}"
        )
        self._set_enabled(True)

        for it in self.edge_items:
            self.view.removeItem(it)
        self.edge_items = []

        for _ in self.skel.edges:
            item = gl.GLLinePlotItem(pos=np.zeros((2, 3)), width=2, antialias=True, mode="lines")
            self.edge_items.append(item)
            self.view.addItem(item)

        self._render_frame(0)

    def _render_frame(self, frame_idx: int) -> None:
        if self.motion is None or self.skel is None:
            return
        pose = eval_pose_world(self.motion, frame_idx, include_end_sites=True)
        pts = np.array(pose.positions, dtype=np.float64)

        for eidx, (a, b) in enumerate(self.skel.edges):
            seg = np.vstack([pts[a], pts[b]])
            self.edge_items[eidx].setData(pos=seg)

    def _on_play(self) -> None:
        if self.motion is None:
            return
        interval_ms = max(1, int(self.motion.frame_time * 1000))
        self.timer.start(interval_ms)

    def _on_pause(self) -> None:
        self.timer.stop()

    def _on_tick(self) -> None:
        if self.motion is None:
            return
        self.frame = (self.frame + 1) % self.motion.frames
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
