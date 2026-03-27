from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolo26n.yaml")

# Load a pretrained YOLO model (recommended for training)
# model = YOLO("yolo26n.pt")
model = YOLO("runs/detect/train3/weights/last.pt")


# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(
    data="/data/self_make/masterDegree/outdoor8_gazebo_partial2/dataset.yaml",
    epochs=300,
)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
# results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
success = model.export(format="onnx")
