from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolov8s.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8s.pt")
# model = YOLO("runs/detect/train3/weights/last.pt")


# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(
    data="/data/self_make/water_world2026-04-04_19-25-25/p04_CVat-Finetune-Dataset/water_world2026-04-04_21-29-49/dataset.yaml",
    epochs=1000,
)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
# results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
# success = model.export(format="onnx")
