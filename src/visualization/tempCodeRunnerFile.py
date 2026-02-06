# test with image input
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