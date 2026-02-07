"""
Parse the raw MediaPipe PoseLandmarkerResult into convenient Python structures.
"""

from typing import Any, Dict, Iterable, List, Literal, Optional
import numpy as np

from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult

class Pose_Parser:
    """
    A utility class to parse MediaPipe PoseLandmarkerResult into plain Python dictionaries or NumPy arrays for easier processing and analysis.
    Stores one PoseLandmarkerResult (1 frame) at a time, and provides methods to convert it into different formats.
    """
    def __init__(self, detection_results: PoseLandmarkerResult):
        self.detection_results = detection_results

    def _landmark_to_dict(self, landmark: Any) -> Dict[str, Any]:
        data = {
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
        }
        if hasattr(landmark, "visibility"):
            data["visibility"] = landmark.visibility
        if hasattr(landmark, "presence"):
            data["presence"] = landmark.presence
        return data

    def _is_empty(self) -> bool:
        return not self.detection_results.pose_landmarks

    def to_dict(self, target: Literal["pose_landmark", "pose_world_landmark"] = "pose_world_landmark") -> Optional[List[Dict[str, Any]]]:
        """
        Convert the detection result to a plain-Python list of dictionaries. 

        Args:
            target: Specify which landmarks to convert. Options are "pose_landmark" or "pose_world_landmark".

        Returns:
            List of dictionaries representing the specified landmarks, or None if no landmarks are present.

            Structure:
            [
                {"x": ..., "y": ..., "z": ..., "visibility": ..., "presence": ...},
                ...
            ]
        """
        if self._is_empty():
            return None
        if target == "pose_landmark":
            return [self._landmark_to_dict(lm) for lm in self.detection_results.pose_landmarks[0]]
        elif target == "pose_world_landmark":
            return [self._landmark_to_dict(lm) for lm in self.detection_results.pose_world_landmarks[0]]
        else:
            raise ValueError(f"Invalid target specified: {target}. Must be 'pose_landmark' or 'pose_world_landmark'.")

    def to_dict_all(self, include_segmentation_masks: bool = False) -> Optional[Dict[str, list[Dict[str, Any]]]]:
        """
        Convert the detection result to a plain-Python dictionary.

        Args:
            include_segmentation_masks: If True, include segmentation masks as-is.

        Returns:
            Dictionary with pose landmarks and pose world landmarks, or None if no landmarks are present.

            Structure:
            {
                "pose_landmarks": [
                    [{"x": ..., "y": ..., "z": ..., "visibility": ..., "presence": ...}, ...],
                    ...
                ],
                "pose_world_landmarks": [
                    [{"x": ..., "y": ..., "z": ..., "visibility": ..., "presence": ...}, ...],
                    ...
                ],
                "segmentation_masks": [...],  # optional
            }
        """
        
        if self._is_empty():
            return None

        result: Dict[str, list[Dict[str, Any]]] = {
            "pose_landmarks": self.to_dict(target="pose_landmark"),
            "pose_world_landmarks": self.to_dict(target="pose_world_landmark"),
        }

        if include_segmentation_masks:
            result["segmentation_masks"] = self.detection_results.segmentation_masks

        return result
    
    def to_ndarray(self, target: Literal["pose_landmark", "pose_world_landmark"] = "pose_world_landmark") -> Optional[np.ndarray]:
        """Convert the specified landmarks to a NumPy array for easier numerical processing.
        Args:
            target: Specify which landmarks to convert. Options are "pose_landmark" or "pose_world_landmark".
        Returns:
            NumPy array of shape (num_landmarks, 5) containing x, y, z, visibility, and presence,
            or None if no landmarks are present.

            Structure:
                [
                    [x, y, z, visibility, presence],
                    ...
                ]
        """
        if self._is_empty():
            return None
        if target == "pose_landmark":
            return np.array([[lm.x, lm.y, lm.z, getattr(lm, "visibility"), getattr(lm, "presence")] for lm in self.detection_results.pose_landmarks[0]])
        elif target == "pose_world_landmark":
            return np.array([[lm.x, lm.y, lm.z, getattr(lm, "visibility"), getattr(lm, "presence")] for lm in self.detection_results.pose_world_landmarks[0]])
        else:
            raise ValueError(f"Invalid target specified: {target}. Must be 'pose_landmark' or 'pose_world_landmark'.")

    def to_ndarray_all(self) -> Optional[Dict[str, np.ndarray]]:
        """Convert all landmarks to NumPy arrays for easier numerical processing.
        Returns:
            Dictionary with pose landmarks and pose world landmarks as NumPy arrays,
            or None if no landmarks are present.

            Structure:
            {
                "pose_landmarks": np.ndarray of shape (num_landmarks, 5),
                "pose_world_landmarks": np.ndarray of shape (num_landmarks, 5),
                "segmentation_masks": [...],  # optional
            }
        """
        if self._is_empty():
            return None

        result: Dict[str, np.ndarray] = {
            "pose_landmarks": self.to_ndarray(target="pose_landmark"),
            "pose_world_landmarks": self.to_ndarray(target="pose_world_landmark"),
        }

        if hasattr(self.detection_results, "segmentation_masks"):
            result["segmentation_masks"] = self.detection_results.segmentation_masks

        return result

    def print_results(self):
        """
        Prints the raw detection results to the console for debugging purposes. Prints "None" if empty.
        """
        print("\n")
        print("============================================")
        print("|    printing raw detection results:       |")
        print("============================================")
        print("Pose Landmarks:")
        if self._is_empty():
            print("None")
        else:
            counter = 0
            for i in self.detection_results.pose_landmarks[0]:
                print(f"{counter}: {self._landmark_to_dict(i)}")
                counter += 1
        print("-------------------------------------------")
        print("Pose World Landmarks:")
        if self._is_empty():
            print("None")
        else:
            counter = 0
            for i in self.detection_results.pose_world_landmarks[0]:
                print(f"{counter}: {self._landmark_to_dict(i)}")
                counter += 1
        print("-------------------------------------------")
