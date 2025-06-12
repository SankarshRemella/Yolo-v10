from ultralytics import YOLO
 
# Load model
model = YOLO("yolov10m.pt")  # Load a pretrained model
 
# Train model
model.train(
    data="data\\road_sign.yaml",  # Path to your data file
    imgsz=640,
    device='cpu',
    batch=1,
    epochs=30,
    patience=50,
    project="results",
    name="yolov10_road_sign",
    amp=False
)
 