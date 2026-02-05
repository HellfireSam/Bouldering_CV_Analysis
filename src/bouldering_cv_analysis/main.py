import os
import cv2
import numpy as np
from dotenv import load_dotenv

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


def resize_with_aspect(image, max_size=640):
	h, w = image.shape[:2]
	scale = max_size / max(h, w)
	new_w = int(w * scale)
	new_h = int(h * scale)
	return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def draw_landmarks_on_image(rgb_image, detection_result):
	annotated_image = rgb_image.copy()
	h, w, _ = annotated_image.shape

	for pose_landmarks in detection_result.pose_landmarks:
		# Draw landmarks
		for lm in pose_landmarks:
			x = int(lm.x * w)
			y = int(lm.y * h)
			cv2.circle(annotated_image, (x, y), 4, (0, 255, 0), -1)

		# Draw connections
		for start_idx, end_idx in POSE_CONNECTIONS:
			lm_start = pose_landmarks[start_idx]
			lm_end = pose_landmarks[end_idx]

			x1, y1 = int(lm_start.x * w), int(lm_start.y * h)
			x2, y2 = int(lm_end.x * w), int(lm_end.y * h)

			cv2.line(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

	return annotated_image

# def 

if __name__ == "__main__":
	load_dotenv()
	MODEL_BUNDLE_PATH = os.getenv("MODEL_BUNDLE_PATH")
	IMG_PATH = os.getenv("IMG_PATH")

	# Create PoseLandmarker
	base_options = python.BaseOptions(model_asset_path=MODEL_BUNDLE_PATH)
	options = vision.PoseLandmarkerOptions(
		base_options=base_options,
		output_segmentation_masks=True
	)
	detector = vision.PoseLandmarker.create_from_options(options)
	
	# Load image 
	bgr = cv2.imread(IMG_PATH) 
	bgr = resize_with_aspect(bgr, max_size=640)
	rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

	image = mp.Image(
		image_format=mp.ImageFormat.SRGB,
		data=rgb
	)
	
	# Detect pose
	detection_result = detector.detect(image)

	# Visualize
	annotated_image = draw_landmarks_on_image(
		image.numpy_view(),
		detection_result
	)

	bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
	cv2.imshow("Pose Landmarks", bgr)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
