from ultralytics import YOLO
import os

def train_v10():
    data_yaml = "/Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/data/data_subset.yaml"

    model = YOLO('yolov10n.pt')

    results = model.train(
        data=data_yaml,
        epochs=10,
        imgsz=320,
        batch=16,
        name='v10n_baseline',
        project='/Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/runs/detect',
        mosaic=1.0,
        mixup=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_s=0.7,
        hsv_v=0.4,
        device='mps',
        workers=1,
        amp=False,
        verbose=True
    )
    
    print("Training completed. Results saved in runs/detect/v10n_baseline")

if __name__ == "__main__":
    train_v10()
