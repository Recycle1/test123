from ultralytics import YOLO

# Load the model and run the tracker with a custom configuration file
model = YOLO('yolov8n.pt')
results = model.track(source="single_video.avi", conf=0.3, iou=0.5, show=True)
