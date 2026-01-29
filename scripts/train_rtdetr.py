from ultralytics import RTDETR
import os

def train_rtdetr():
    data_yaml = "/Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/data/data_subset.yaml"

    model = RTDETR('rtdetr-l.pt')

    results = model.train(
        data=data_yaml,
        epochs=10,
        imgsz=320,
        batch=8, 
        name='rtdetr_baseline',
        project='/Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/runs/detect',
        mosaic=1.0,
        mixup=0.1,
        degrees=10.0,
        device='mps',
        workers=1,
        amp=False,
        verbose=True
    )
    
    print("Training completed. Results saved in runs/detect/rtdetr_baseline")

if __name__ == "__main__":
    train_rtdetr()
