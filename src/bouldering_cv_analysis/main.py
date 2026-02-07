import cv2
import numpy as np
from pathlib import Path

from bouldering_cv_analysis.config import get_settings, Settings

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



# Pose landmark connections (MediaPipe pose topology)
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


def test_image(
	IMG_PATH: str = ".\\assets\\image.jpg",
	settings: Settings = get_settings(),
	enable_3d: bool = False,
):
	import visualization.render2d as render2d_module
	import visualization.render3d as render3d_module
	import bouldering_cv_analysis.models.landmarker as landmarker_module
	# test with image input
	model = landmarker_module.PoseLandmarkerModel(
		model_path=settings.model_bundle_path, input_mode="image", output_segmentation_masks=True,
		**settings.detection_confidence
	)
	results = model.detect(IMG_PATH)
	annotated = render2d_module.draw_landmarks_on_image_mediapipe(IMG_PATH, results)
	cv2.imshow("annotated", annotated)
	output_path = Path(IMG_PATH)
	output_path = str(output_path.with_name(f"annotated_{output_path.name}"))
	cv2.imwrite(output_path, annotated)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	from processing.PoseLandmarkerResult_parser import Pose_Parser
	parser = Pose_Parser(results)
	parser.print_results()
	if enable_3d:
		pose = parser.to_ndarray(target="pose_world_landmark")
		if pose is not None:
			render3d_module.run_playback_viewer([pose[:, :3]], fps=30, window_title="Pose 3D Viewer (Image)")


def test_video(
	VIDEO_PATH: str = ".\\assets\\video2.mp4",
	settings: Settings = get_settings(),
	enable_3d: bool = False,
):
	import visualization.render2d as render2d_module
	import visualization.render3d as render3d_module
	import bouldering_cv_analysis.models.landmarker as landmarker_module
	import processing.PoseLandmarkerResult_parser as parser_module
	
	cap = cv2.VideoCapture(VIDEO_PATH)
	if not cap.isOpened():
		raise ValueError(f"Failed to open video: {VIDEO_PATH}")

	video_model = landmarker_module.PoseLandmarkerModel(
		model_path=settings.model_bundle_path, input_mode="video", output_segmentation_masks=True,
		**settings.detection_confidence
	)

	fps = cap.get(cv2.CAP_PROP_FPS)
	if fps <= 0:
		fps = 30.0
		print("Warning: Failed to read FPS from video, defaulting to 30 FPS.")
		
	frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	if frame_width <= 0 or frame_height <= 0:
		raise ValueError("Failed to read video frame size.")

	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	OUTPUT_VIDEO_PATH = Path(VIDEO_PATH)
	OUTPUT_VIDEO_PATH = str(OUTPUT_VIDEO_PATH.with_name(f"annotated_{OUTPUT_VIDEO_PATH.name}"))
	writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
	if not writer.isOpened():
		raise ValueError(f"Failed to open video writer: {OUTPUT_VIDEO_PATH}")

	frame_index = 0
	frames_3d = []
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		frame_timestamp = int((frame_index / fps) * 1000)
		results = video_model.detect(frame, frame_timestamp=frame_timestamp, image_mode="BGR")
		annotated_frame = render2d_module.draw_landmarks_on_image_mediapipe(frame, results, image_format="BGR")
		writer.write(annotated_frame)
		cv2.imshow("annotated_video", annotated_frame)
		parser = parser_module.Pose_Parser(results)
		parser.print_results()
		if enable_3d:
			pose = parser.to_ndarray(target="pose_world_landmark")
			if pose is not None:
				frames_3d.append(pose[:, :3])

		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

		frame_index += 1

	cap.release()
	writer.release()
	cv2.destroyAllWindows()
	if enable_3d and frames_3d:
		render3d_module.run_playback_viewer(frames_3d, fps=int(fps), window_title="Pose 3D Viewer (Video)")


def test_live_stream(
	settings: Settings = get_settings(),
	enable_3d: bool = False,
):
	import visualization.render2d as render2d_module
	import visualization.render3d as render3d_module
	import bouldering_cv_analysis.models.landmarker as landmarker_module
	import time
	import threading

	latest_result = {"result": None, "pose": None, "timestamp_ms": None}

	def _handle_result(result, output_image, timestamp_ms):
		latest_result["result"] = result
		latest_result["timestamp_ms"] = timestamp_ms
		if result is None:
			return
		import processing.PoseLandmarkerResult_parser as parser_module
		parser = parser_module.Pose_Parser(result)
		pose = parser.to_ndarray(target="pose_world_landmark")
		if pose is not None:
			latest_result["pose"] = pose[:, :3]

	video_model = landmarker_module.PoseLandmarkerModel(
		model_path=settings.model_bundle_path,
		input_mode="live_stream",
		output_segmentation_masks=True,
		result_callback=_handle_result,
		**settings.detection_confidence
	)

	cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
	# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
	# cap.set(cv2.CAP_PROP_FPS, 30)
	# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

	stop_event = threading.Event()

	def _capture_loop():
		while not stop_event.is_set():
			ret, frame = cap.read()
			if not ret:
				break
			if frame is None or frame.mean() == 0:
				continue

			frame_timestamp = int(time.monotonic() * 1000)
			video_model.detect(frame, frame_timestamp=frame_timestamp, image_mode="BGR")

			result = latest_result["result"]
			if result is not None:
				annotated_frame = render2d_module.draw_landmarks_on_image_mediapipe(frame, result, image_format="BGR")
			else:
				annotated_frame = frame

			cv2.imshow("live_stream (press q to quit)", annotated_frame)
			if cv2.waitKey(1) & 0xFF == ord("q"):
				stop_event.set()
				break

		cap.release()
		cv2.destroyAllWindows()

	if enable_3d:
		thread = threading.Thread(target=_capture_loop, daemon=True)
		thread.start()
		render3d_module.run_live_viewer(lambda: latest_result["pose"], fps=30)
		stop_event.set()
		thread.join(timeout=1.0)
	else:
		_capture_loop()
	
if __name__ == "__main__":
	settings = get_settings()
	# test_image(".\\assets\\image3.jpg", settings=settings, enable_3d=True)
	test_video(settings=settings, enable_3d=True)
	# test_live_stream(settings=settings)

