from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

def _install_mediapipe_stub() -> None:
	mp = types.ModuleType("mediapipe")
	tasks = types.ModuleType("mediapipe.tasks")
	tasks_python = types.ModuleType("mediapipe.tasks.python")
	tasks_python_vision = types.ModuleType("mediapipe.tasks.python.vision")
	tasks_vision = types.ModuleType("mediapipe.tasks.vision")

	class _BaseOptions:  # pragma: no cover - stub
		def __init__(self, model_asset_path: str):
			self.model_asset_path = model_asset_path

	class _PoseLandmarkerOptions:  # pragma: no cover - stub
		def __init__(self, **kwargs):
			self.kwargs = kwargs

	class _PoseLandmarker:  # pragma: no cover - stub
		@classmethod
		def create_from_options(cls, options):
			return object()

	class _RunningMode:  # pragma: no cover - stub
		IMAGE = "IMAGE"
		VIDEO = "VIDEO"
		LIVE_STREAM = "LIVE_STREAM"

	class _PoseLandmarkerResult:  # pragma: no cover - stub
		pass

	tasks.BaseOptions = _BaseOptions
	tasks.vision = tasks_vision
	tasks_vision.PoseLandmarker = _PoseLandmarker
	tasks_vision.PoseLandmarkerOptions = _PoseLandmarkerOptions
	tasks_vision.RunningMode = _RunningMode
	tasks_vision.PoseLandmarkerResult = _PoseLandmarkerResult

	sys.modules.setdefault("mediapipe", mp)
	sys.modules.setdefault("mediapipe.tasks", tasks)
	sys.modules.setdefault("mediapipe.tasks.python", tasks_python)
	sys.modules.setdefault("mediapipe.tasks.python.vision", tasks_python_vision)
	sys.modules.setdefault("mediapipe.tasks.vision", tasks_vision)
	mp.tasks = tasks
	tasks.python = tasks_python
	tasks_python.vision = tasks_python_vision


_install_mediapipe_stub()

from bouldering_cv_analysis.models.landmarker import PoseLandmarkerModel


class TestPoseLandmarkerModel(TestCase):
	def test_create_landmarker_invalid_input_mode_raises(self) -> None:
		model = PoseLandmarkerModel.__new__(PoseLandmarkerModel)
		model.input_mode = "bad_mode"

		with self.assertRaises(ValueError):
			PoseLandmarkerModel._create_landmarker(model)

	def test_detect_raises_when_image_is_none(self) -> None:
		fake_landmarker = MagicMock()

		with patch.object(PoseLandmarkerModel, "_create_landmarker", return_value=fake_landmarker):
			model = PoseLandmarkerModel("dummy.task", input_mode="image")

		with self.assertRaises(ValueError):
			model.detect(None)

	def test_detect_requires_timestamp_for_video(self) -> None:
		fake_landmarker = MagicMock()

		with patch.object(PoseLandmarkerModel, "_create_landmarker", return_value=fake_landmarker):
			model = PoseLandmarkerModel("dummy.task", input_mode="video")

		with self.assertRaises(ValueError):
			model.detect(object())

	def test_detect_image_mode_calls_detect(self) -> None:
		fake_landmarker = MagicMock()
		fake_landmarker.detect.return_value = "image_result"

		with patch.object(PoseLandmarkerModel, "_create_landmarker", return_value=fake_landmarker):
			model = PoseLandmarkerModel("dummy.task", input_mode="image")

		result = model.detect(object())

		self.assertEqual(result, "image_result")
		fake_landmarker.detect.assert_called_once()

	def test_detect_video_mode_calls_detect_for_video(self) -> None:
		fake_landmarker = MagicMock()
		fake_landmarker.detect_for_video.return_value = "video_result"

		with patch.object(PoseLandmarkerModel, "_create_landmarker", return_value=fake_landmarker):
			model = PoseLandmarkerModel("dummy.task", input_mode="video")

		image = object()
		result = model.detect(image, frame_timestamp=123)

		self.assertEqual(result, "video_result")
		fake_landmarker.detect_for_video.assert_called_once_with(image, 123)

	def test_detect_live_stream_mode_calls_detect_async(self) -> None:
		fake_landmarker = MagicMock()
		fake_landmarker.detect_async.return_value = "async_result"

		with patch.object(PoseLandmarkerModel, "_create_landmarker", return_value=fake_landmarker):
			model = PoseLandmarkerModel("dummy.task", input_mode="live_stream")

		image = object()
		result = model.detect(image, frame_timestamp=456)

		self.assertEqual(result, "async_result")
		fake_landmarker.detect_async.assert_called_once_with(image, 456)
