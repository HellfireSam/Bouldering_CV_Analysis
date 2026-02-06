"""
Parse the raw MediaPipe PoseLandmarkerResult into convenient Python structures.
"""

from typing import Any, Dict, Iterable, List, Literal
import numpy as np

from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult

class Pose_Parser:
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

    def get_timestamp(self) -> int:
        """Get the timestamp (ms) of the detection result, if available."""
        if hasattr(self.detection_results, "timestamp_ms"):
            return self.detection_results.timestamp_ms
        else:
            raise AttributeError("The detection results do not contain a timestamp.")
        
    def to_dict(self, target: Literal["pose_landmark", "pose_world_landmark"] = "pose_world_landmark") -> List[Dict[str, Any]]:
        """
        Convert the detection result to a plain-Python list of dictionaries. 

        Args:
            target: Specify which landmarks to convert. Options are "pose_landmark" or "pose_world_landmark".

        Returns:
            List of dictionaries representing the specified landmarks.

            Structure:
            [
                {"x": ..., "y": ..., "z": ..., "visibility": ..., "presence": ...},
                ...
            ]
        """
        if target == "pose_landmark":
            return [self._landmark_to_dict(lm) for lm in self.detection_results.pose_landmarks[0]]
        elif target == "pose_world_landmark":
            return [self._landmark_to_dict(lm) for lm in self.detection_results.pose_world_landmarks[0]]
        else:
            raise ValueError(f"Invalid target specified: {target}. Must be 'pose_landmark' or 'pose_world_landmark'.")

    def to_dict_all(self, include_segmentation_masks: bool = False) -> Dict[str, list[Dict[str, Any]]]:
        """
        Convert the detection result to a plain-Python dictionary.

        Args:
            include_segmentation_masks: If True, include segmentation masks as-is.

        Returns:
            Dictionary with pose landmarks and pose world landmarks.

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
                "timestamp_ms": ...           # optional
            }
        """
        
        result: Dict[str, list[Dict[str, Any]]] = {
            "pose_landmarks": self.to_dict(target="pose_landmark"),
            "pose_world_landmarks": self.to_dict(target="pose_world_landmark"),
        }

        if include_segmentation_masks:
            result["segmentation_masks"] = self.detection_results.segmentation_masks

        if hasattr(self.detection_results, "timestamp_ms"):
            result["timestamp_ms"] = self.detection_results.timestamp_ms

        return result
    
    def to_ndarray(self, target: Literal["pose_landmark", "pose_world_landmark"] = "pose_world_landmark") -> np.ndarray:
        """Convert the specified landmarks to a NumPy array for easier numerical processing.
        Args:
            target: Specify which landmarks to convert. Options are "pose_landmark" or "pose_world_landmark".
        Returns:
            NumPy array of shape (num_landmarks, 5) containing x, y, z, visibility, and presence.

            Structure:
                [
                    [x, y, z, visibility, presence],
                    ...
                ]
        """
        if target == "pose_landmark":
            return np.array([[lm.x, lm.y, lm.z, getattr(lm, "visibility"), getattr(lm, "presence")] for lm in self.detection_results.pose_landmarks[0]])
        elif target == "pose_world_landmark":
            return np.array([[lm.x, lm.y, lm.z, getattr(lm, "visibility"), getattr(lm, "presence")] for lm in self.detection_results.pose_world_landmarks[0]])
        else:
            raise ValueError(f"Invalid target specified: {target}. Must be 'pose_landmark' or 'pose_world_landmark'.")

    def to_ndarray_all(self) -> Dict[str, np.ndarray]:
        """Convert all landmarks to NumPy arrays for easier numerical processing.
        Returns:
            Dictionary with pose landmarks and pose world landmarks as NumPy arrays.

            Structure:
            {
                "pose_landmarks": np.ndarray of shape (num_landmarks, 5),
                "pose_world_landmarks": np.ndarray of shape (num_landmarks, 5),
                "segmentation_masks": [...],  # optional
                "timestamp_ms": ...           # optional
            }
        """
        result: Dict[str, np.ndarray] = {
            "pose_landmarks": self.to_ndarray(target="pose_landmark"),
            "pose_world_landmarks": self.to_ndarray(target="pose_world_landmark"),
        }

        if hasattr(self.detection_results, "segmentation_masks"):
            result["segmentation_masks"] = self.detection_results.segmentation_masks

        if hasattr(self.detection_results, "timestamp_ms"):
            result["timestamp_ms"] = self.detection_results.timestamp_ms

        return result

    def print_results(self):
        """
        Prints the raw detection results to the console for debugging purposes.
        """
        print("\n")
        print("============================================")
        print("|    printing raw detection results:       |")
        print("============================================")
        print("Pose Landmarks:")
        counter = 0
        for i in self.detection_results.pose_landmarks[0]:
            print(f"{counter}: {self._landmark_to_dict(i)}")
            counter += 1
        print("-------------------------------------------")
        print("Pose World Landmarks:")
        counter = 0
        for i in self.detection_results.pose_world_landmarks[0]:
            print(f"{counter}: {self._landmark_to_dict(i)}")
            counter += 1
        print("-------------------------------------------")
        if hasattr(self.detection_results, "timestamp_ms"):
            print(f"Timestamp (ms): {self.detection_results.timestamp_ms}")
            print("-------------------------------------------")
