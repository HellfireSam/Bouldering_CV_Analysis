"""
./src/bouldering_cv_analysis/models/landmarker.py

Purpose: 
- MediaPipe Landmarker model wrapper
- For loading and initializing the MediaPipe Pose Landmarker model. 
- Used to configure and manipulate the model
- adapted from https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python

"""

from typing import Callable, Literal, Optional, Union
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerResult

class PoseLandmarkerModel:
	"""
	Wrapper for the MediaPipe Pose Landmarker model.

	Handles initialization, configuration, and creation of a MediaPipe Pose
	Landmarker instance for detecting human poses in images, videos, or live
	streams.
	"""

	def __init__(self, model_path: str, input_mode: str="video", 
				min_pose_detection_confidence: float=0.5,
				min_pose_presence_confidence: float=0.5,
				min_tracking_confidence: float=0.5,
				output_segmentation_masks: bool=False,
				result_callback: Optional[Callable[[PoseLandmarkerResult, mp.Image, int], None]] = None):
		"""
		Initialize the PoseLandmarkerModel with configuration parameters.
		
		Args:
			model_path: Path to the .task pose landmarker model file.
			input_mode: Mode of operation: "image", "video", or "live_stream".
				Defaults to "video".
			min_pose_detection_confidence: Minimum confidence score for pose
				detection [0.0-1.0]. Defaults to 0.5.
			min_pose_presence_confidence: Minimum confidence score for pose
				presence [0.0-1.0]. Defaults to 0.5.
			min_pose_tracking_confidence: Minimum confidence score for pose
				tracking [0.0-1.0]. Defaults to 0.5.
			output_segmentation_masks: Whether to output segmentation masks. Defaults to False.
		"""
		self.model_path = model_path
		self.input_mode = input_mode
		self.min_pose_detection_confidence = min_pose_detection_confidence
		self.min_pose_presence_confidence = min_pose_presence_confidence
		self.min_tracking_confidence = min_tracking_confidence
		self.output_segmentation_masks = output_segmentation_masks
		self.result_callback = result_callback
		self.landmarker = self._create_landmarker()


	def _create_landmarker(self) -> PoseLandmarker:
		"""
		Create and return a configured MediaPipe Pose Landmarker instance.

		Validates the input mode and constructs a PoseLandmarker with the instance's
		configuration parameters including confidence thresholds and running mode.

		Returns:
			PoseLandmarker: A configured MediaPipe Pose Landmarker instance.

		Raises:
			ValueError: If input_mode is not one of "image", "video", or "live_stream".
		"""
		if self.input_mode not in ["image", "video", "live_stream"]:
			raise ValueError(f"Invalid input_mode: {self.input_mode}. Must be 'image', 'video', or 'live_stream'.")
		
		BaseOptions = mp.tasks.BaseOptions
		PoseLandmarker = vision.PoseLandmarker
		PoseLandmarkerOptions = vision.PoseLandmarkerOptions
		VisionRunningMode = vision.RunningMode

		if self.input_mode == "live_stream":
			def _default_result_callback(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
				print(f"pose landmarker result: {result}")

		running_mode_map = {"image": VisionRunningMode.IMAGE, 
							"video": VisionRunningMode.VIDEO, 
							"live_stream": VisionRunningMode.LIVE_STREAM}

		options = PoseLandmarkerOptions(
			base_options=BaseOptions(model_asset_path=self.model_path),
			running_mode=running_mode_map[self.input_mode],
			num_poses=1,
			min_pose_detection_confidence=self.min_pose_detection_confidence,
			min_pose_presence_confidence=self.min_pose_presence_confidence,
			min_tracking_confidence=self.min_tracking_confidence,
			output_segmentation_masks=self.output_segmentation_masks,
			result_callback=(self.result_callback or _default_result_callback) if self.input_mode == "live_stream" else None)

		return PoseLandmarker.create_from_options(options) 

	def detect(
		self,
		image: Union[mp.Image, str, np.ndarray],
		frame_timestamp: int=None,
		image_mode: Literal["RGB", "BGR"]="BGR",
	) -> PoseLandmarkerResult | None:
		"""
		Run pose detection on an image and return the raw MediaPipe result object.

		The result is typically parsed to analyzer.py to extract relevant
		pose landmark data.

		Args:
			image (mp.Image | str | np.ndarray): The input image for pose detection. If a string
				is provided, it is treated as a file path and loaded via OpenCV. If a numpy array
				is provided, it is treated as an image in BGR format, unless image_mode is set to "RGB".
			frame_timestamp (int, optional): Frame timestamp in milliseconds. Required
				for "video" and "live_stream" modes. Defaults to None.
			image_mode ("RGB" | "BGR"): Color format for numpy arrays and mp.Image inputs.

		Raises:
			ValueError: If image is None.
			ValueError: If frame_timestamp is not provided in "video" or
				"live_stream" modes.
		Returns:
			PoseLandmarkerResult: Pose landmarker result object.
			None: In live_stream mode, results are returned via the result_callback.
		"""
		if image is None:
			raise ValueError("Input image parsed for detection cannot be None.")
			
		if self.input_mode != "image" and frame_timestamp is None:
			raise ValueError("frame_timestamp must be provided for video or live_stream mode.")
		
		# ilay: unsure if we need to resize the image for performance
		# def resize_with_aspect(image: np.ndarray, max_size=640):
		# 	h, w = image.shape[:2]
		# 	scale = max_size / max(h, w)
		# 	new_w = int(w * scale)
		# 	new_h = int(h * scale)
		# 	return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

		if isinstance(image, str): # file path
			bgr = cv2.imread(image)
			if bgr is None:
				raise ValueError(f"Failed to load image from path: {image}")
			rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
			image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
		
		elif isinstance(image, np.ndarray): # numpy array
			if image_mode == "BGR":
				rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			else:
				rgb = image
			image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
		
		elif isinstance(image, mp.Image): # mp.Image
			if image_mode == "BGR":
				rgb = cv2.cvtColor(image.numpy_view(), cv2.COLOR_BGR2RGB)
				image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
		else:
			raise ValueError("Unsupported image type. Expected mp.Image, str, or np.ndarray.")
			
		# returning the prediction results
		if self.input_mode == "live_stream":
			return self.landmarker.detect_async(image, frame_timestamp)
		elif self.input_mode == "video":
			return self.landmarker.detect_for_video(image, frame_timestamp)
		else:  # image mode
			return self.landmarker.detect(image)