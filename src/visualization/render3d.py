"""
./src/visualization/render3d.py

Purpose:
- Real-time 3D stickman viewer using pyqtgraph.
- Supports live updates (latest frame) and offline playback (sequence).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple
import time

import numpy as np

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui


POSE_CONNECTIONS = [
	(0, 1), (1, 2), (2, 3), (3, 7),
	(0, 4), (4, 5), (5, 6), (6, 8),
	(9, 10),
	(11, 12),
	(11, 13), (13, 15),
	(12, 14), (14, 16),
	(15, 17), (16, 18),
	(17, 19), (18, 20),
	(19, 21), (20, 22),
	(11, 23), (12, 24),
	(23, 24),
	(23, 25), (24, 26),
	(25, 27), (26, 28),
	(27, 29), (28, 30),
	(29, 31), (30, 32),
]


@dataclass
class RenderConfig:
	point_size: float = 6.0
	line_width: float = 2.0
	axis_size: float = 0.25
	grid_size: float = 2.0
	grid_spacing: float = 0.1
	background: Tuple[int, int, int] = (20, 20, 20)
	point_color: Tuple[float, float, float, float] = (0.9, 0.8, 0.2, 1.0)
	line_color: Tuple[float, float, float, float] = (0.2, 0.9, 0.5, 1.0)
	flip_y: bool = True
	flip_z: bool = True


class Pose3DWindow:
	"""Interactive 3D window that updates a stickman at a fixed FPS."""

	def __init__(
		self,
		get_latest_landmarks: Callable[[], Optional[np.ndarray]],
		fps: int = 30,
		connections: Sequence[Tuple[int, int]] = POSE_CONNECTIONS,
		config: Optional[RenderConfig] = None,
		window_title: str = "Pose 3D Viewer",
	):
		self.get_latest_landmarks = get_latest_landmarks
		self.fps = max(1, int(fps))
		self.connections = list(connections)
		self.config = config or RenderConfig()

		self._app = pg.mkQApp(window_title)
		self._view = gl.GLViewWidget()
		self._view.setWindowTitle(window_title)
		self._view.setBackgroundColor(self.config.background)
		self._view.setCameraPosition(distance=1.5, elevation=15, azimuth=30)

		self._grid = gl.GLGridItem()
		self._grid.setSize(self.config.grid_size, self.config.grid_size, self.config.grid_size)
		self._grid.setSpacing(self.config.grid_spacing, self.config.grid_spacing, self.config.grid_spacing)
		self._view.addItem(self._grid)

		self._axis = gl.GLAxisItem()
		self._axis.setSize(
			self.config.axis_size,
			self.config.axis_size,
			self.config.axis_size,
		)
		self._view.addItem(self._axis)

		self._scatter = gl.GLScatterPlotItem(
			pos=np.zeros((33, 3), dtype=np.float32),
			size=self.config.point_size,
			color=self.config.point_color,
			pxMode=True,
		)
		self._view.addItem(self._scatter)

		self._lines = []
		for _ in self.connections:
			line = gl.GLLinePlotItem(
				pos=np.zeros((2, 3), dtype=np.float32),
				color=self.config.line_color,
				width=self.config.line_width,
				antialias=True,
			)
			self._lines.append(line)
			self._view.addItem(line)

		self._timer = QtCore.QTimer()
		self._timer.timeout.connect(self._on_timer)
		self._timer.start(int(1000 / self.fps))

		self._view.show()

	def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
		if landmarks.ndim != 2 or landmarks.shape[1] < 3:
			raise ValueError("Expected landmarks shape (N, 3) or (N, >=3).")
		coords = landmarks[:, :3].astype(np.float32, copy=False)
		if self.config.flip_y:
			coords[:, 1] *= -1.0
		if self.config.flip_z:
			coords[:, 2] *= -1.0
		return coords

	def _on_timer(self):
		landmarks = self.get_latest_landmarks()
		if landmarks is None:
			return
		coords = self._normalize_landmarks(landmarks)
		if coords.shape[0] < 33:
			return
		self._scatter.setData(pos=coords)
		for line, (a, b) in zip(self._lines, self.connections):
			line.setData(pos=np.vstack([coords[a], coords[b]]))

	def run(self):
		self._app.exec()


def run_live_viewer(
	get_latest_landmarks: Callable[[], Optional[np.ndarray]],
	fps: int = 30,
	window_title: str = "Pose 3D Viewer (Live)",
) -> None:
	"""Start a live viewer that always renders the latest pose."""
	viewer = Pose3DWindow(get_latest_landmarks=get_latest_landmarks, fps=fps, window_title=window_title)
	viewer.run()


def run_playback_viewer(
	frames: Sequence[np.ndarray],
	fps: int = 30,
	window_title: str = "Pose 3D Viewer (Playback)",
) -> None:
	"""Play back a sequence of poses at a fixed FPS."""
	if not frames:
		raise ValueError("Playback frames cannot be empty.")

	start_time = time.monotonic()
	index = 0

	def _latest() -> Optional[np.ndarray]:
		nonlocal index
		elapsed = time.monotonic() - start_time
		target_index = int(elapsed * fps)
		if target_index >= len(frames):
			return frames[-1]
		index = max(index, target_index)
		return frames[index]

	viewer = Pose3DWindow(get_latest_landmarks=_latest, fps=fps, window_title=window_title)
	viewer.run()