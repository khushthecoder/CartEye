from ultralytics import YOLO
import os

def train_augmented():
    # Path to data_subset.yaml (using the 10% subset for fast iterations)
    data_yaml = "/Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/data/data_subset.yaml"
    
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    # Training parameters with Advanced Augmentations
    # - mixup: 0.1 (simulates overlapping objects)
    # - blur: 0.01 (simulates low-quality camera feed)
    # - hsv_s: 0.7 (color saturation variation)
    # - hsv_v: 0.4 (brightness variation)
    # - degrees: 10.0 (rotation variation)
    results = model.train(
        data=data_yaml,
        epochs=10,
        imgsz=320,
        batch=16,
        name='augmented_v8n',
        project='/Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/runs/detect',
        mosaic=1.0,         # Enabled by default, good for crowded scenes
        mixup=0.1,          # Handle overlaps
        blur=0.01,          # Handle low quality
        degrees=10.0,       # Handle rotation
        device='mps',
        workers=1,
        amp=False,
        verbose=True
    )
    
    print("Training completed. Results saved in runs/detect/augmented_v8n")

if __name__ == "__main__":
    train_augmented()
