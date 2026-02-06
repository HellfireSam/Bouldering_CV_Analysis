"""
./src/visualization/render2d.py

Purpose:
- Visualize pose landmarks on RGB images using MediaPipe drawing utilities.
- Produce annotated images with landmarks and connections drawn.
"""

from typing import Literal, Union
import cv2
import numpy as np
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from mediapipe.tasks.python import vision
import mediapipe as mp


def draw_landmarks_on_image_mediapipe(
	image: Union[np.ndarray, str, mp.Image],
	detection_result: PoseLandmarkerResult,
	image_format: Literal["RGB", "BGR"] = "BGR",
) -> np.ndarray:
	"""
	Draw pose landmarks and connections on an image using MediaPipe default methods.

	Args:
		image: Input image in RGB or BGR format, or a file path to an image (image_format will be set to "BGR" if a file path is provided).
		detection_result: Output from PoseLandmarker.detect().
		image_format: Color format for the input image.

	Returns:
		Annotated image in BGR format for OpenCV compatibility.
	"""
	# convert everything to a numpy array for drawing
	if isinstance(image, str):
		image = cv2.imread(image)
		image_format = "BGR"
	elif isinstance(image, mp.Image):
		# Convert mp.Image to numpy array
		image = np.array(image.data)
	
	# Convert to RGB for drawing if input is in BGR format
	if image_format == "BGR":
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	pose_landmarks_list = detection_result.pose_landmarks
	annotated_image = np.copy(image)

	pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
	pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

	for pose_landmarks in pose_landmarks_list:
		drawing_utils.draw_landmarks(
			image=annotated_image,
			landmark_list=pose_landmarks,
			connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
			landmark_drawing_spec=pose_landmark_style,
			connection_drawing_spec=pose_connection_style)
	
	return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR) # convert back to BGR for OpenCV display and saving

if __name__ == "__main__":
	import cv2
	import bouldering_cv_analysis.models.landmarker as landmarker_module

	# # test with image input
	# IMG_PATH = ".\\assets\\image6.jpg"
	# model = landmarker_module.PoseLandmarkerModel(
	# 	model_path=".\\models\\pose_landmarker_lite.task", input_mode="image", output_segmentation_masks=True
	# )
	# results = model.detect(IMG_PATH)
	# annotated = draw_landmarks_on_image_mediapipe(IMG_PATH, results)
	# cv2.imshow("annotated", annotated)
	# cv2.imwrite(".\\assets\\annotated_image6.jpg", annotated)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# from processing.PoseLandmarkerResult_parser import Pose_Parser
	# parser = Pose_Parser(results)
	# parser.print_results()
	# print(parser.to_dict_all())
	# input()
	# print(parser.to_ndarray_all())
	# input()


	# TODO: test with video input.
	VIDEO_PATH = ".\\assets\\video2.mp4"
	OUTPUT_VIDEO_PATH = ".\\assets\\annotated_video2.mp4"
	cap = cv2.VideoCapture(VIDEO_PATH)
	if not cap.isOpened():
		raise ValueError(f"Failed to open video: {VIDEO_PATH}")

	video_model = landmarker_module.PoseLandmarkerModel(
		model_path=".\\models\\pose_landmarker_heavy.task", input_mode="video", output_segmentation_masks=False,
		min_pose_detection_confidence=0, min_tracking_confidence=0.3
	)

	fps = cap.get(cv2.CAP_PROP_FPS)
	if fps <= 0:
		fps = 30.0
	frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	if frame_width <= 0 or frame_height <= 0:
		raise ValueError("Failed to read video frame size.")

	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
	if not writer.isOpened():
		raise ValueError(f"Failed to open video writer: {OUTPUT_VIDEO_PATH}")

	frame_index = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		frame_timestamp = int((frame_index / fps) * 1000)
		results = video_model.detect(frame, frame_timestamp=frame_timestamp, image_mode="BGR")
		annotated_frame = draw_landmarks_on_image_mediapipe(frame, results, image_format="BGR")
		writer.write(annotated_frame)
		cv2.imshow("annotated_video", annotated_frame)

		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

		frame_index += 1

	cap.release()
	writer.release()
	cv2.destroyAllWindows()
	
