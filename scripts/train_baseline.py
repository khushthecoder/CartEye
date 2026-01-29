from ultralytics import YOLO
import os

def train_baseline():
    data_yaml = "/Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/data/data_subset.yaml"
    
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data=data_yaml,
        epochs=10,
        imgsz=320,
        batch=16,
        name='baseline_v8n',
        project='/Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/runs/detect',
        mosaic=1.0,
        device='mps',
        workers=1,
        amp=False,
        verbose=True
    )
    
    print("Training completed. Results saved in runs/detect/baseline_v8n")

if __name__ == "__main__":
    train_baseline()
