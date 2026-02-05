"""
./src/visualization/render.py

Purpose:
- visualize the pose landmarks on the media provided
- produces annotated images with landmarks and connections drawn
"""

import numpy as np
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
import mediapipe as mp

def resize_with_aspect(image, max_size=640):
	h, w = image.shape[:2]
	scale = max_size / max(h, w)
	new_w = int(w * scale)
	new_h = int(h * scale)
	return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

  for pose_landmarks in pose_landmarks_list:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)

  return annotated_image

if __name__ == "__main__":
    import sys
    import cv2
    import bouldering_cv_analysis.models.landmarker as landmarker_module

    bgr = cv2.imread(".\\assets\\image4.jpg") 
    bgr = resize_with_aspect(bgr, max_size=640)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(
		image_format=mp.ImageFormat.SRGB,
		data=rgb
	)

    model = landmarker_module.PoseLandmarkerModel(
        model_path=".\\models\\pose_landmarker_lite.task", input_mode="image"
    )
    results = model.detect(image)
    annotated = draw_landmarks_on_image(image.numpy_view(), results)
    bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imshow("Annotated Image", bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()