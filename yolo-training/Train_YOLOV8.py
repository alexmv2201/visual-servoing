from ultralytics import YOLO
import cv2

# Load Model
model = YOLO("yolo11x.pt")
# Train model - adjust path
results = model.train(
    data="/home/alex/Images_for_yolo/YOLO_all/YOLO_For_Labeling_Task_Board/config.yaml", 
    epochs=200, 
    batch=20,  # Set the batch size to 1 to save memory
    project="/home/alex/Images_for_yolo/YOLO_all/YOLO_For_Labeling_Task_Board/runs/detect",
    name="train"  
)
