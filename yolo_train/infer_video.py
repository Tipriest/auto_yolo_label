from ultralytics import YOLO

from ultralytics import YOLO, ASSETS

model = YOLO(
    "runs/detect/train4_0401/weights/best.pt",
    task="detect",
)

# reference https://docs.ultralytics.com/modes/predict/ for more information.
# model.predict(
#     source="/home/tipriest/Documents/MasterDegree/ros_tools_ws/src/ros_tools/rostopic_to_video/videos/采集待分割视频.mp4",  # (str, optional) source directory for images or videos
#     imgsz=640,  # (int | list) input images size as int or list[w,h] for predict
#     conf=0.25,  # (float) minimum confidence threshold
#     iou=0.7,  # (float) intersection over union (IoU) threshold for NMS
#     # device="cuda:0",  # (int | str | list, optional) device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
#     batch=1,  # (int) batch size
#     half=False,  # (bool) use FP16 half-precision inference
#     max_det=300,  # (int) Limits the maximum number of detections per image. Useful in dense scenes to prevent excessive detections.
#     vid_stride=1,  # (int) video frame-rate stride
#     stream_buffer=False,  # (bool) buffer all streaming frames (True) or return the most recent frame (False)
#     visualize=False,  # (bool) visualize model features
#     augment=False,  # (bool) apply image augmentation to prediction sources
#     agnostic_nms=False,  # (bool) class-agnostic NMS
#     classes=None,  # (int | list[int], optional) filter results by class, i.e. classes=0, or classes=[0,2,3]
#     retina_masks=False,  # (bool) use high-resolution segmentation masks
#     embed=None,  # (list[int], optional) return feature vectors/embeddings from given layers
#     show=False,  # (bool) show predicted images and videos if environment allows
#     save=True,  # (bool) save prediction results
#     save_frames=False,  # (bool) save predicted individual video frames
#     save_txt=False,  # (bool) save results as .txt file
#     save_conf=False,  # (bool) save results with confidence scores
#     save_crop=False,  # (bool) save cropped images with results
#     stream=False,  # (bool) for processing long videos or numerous images with reduced memory usage by returning a generator
#     verbose=True,  # (bool) enable/disable verbose inference logging in the terminal
# )

model.predict(
    source="/home/tipriest/Documents/MasterDegree/ros_tools_ws/src/ros_tools/rostopic_to_video/videos/采集待分割视频.mp4",
    save = True
)
