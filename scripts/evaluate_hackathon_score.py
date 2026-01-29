import os
import yaml
import json
import torch
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
from collections import Counter

def load_gt_labels(label_dir):
    """Loads ground truth category lists from YOLO .txt files."""
    gt_data = {}
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            path = os.path.join(label_dir, label_file)
            img_id = Path(label_file).stem
            categories = []
            with open(path, 'r') as f:
                for line in f:
                    # YOLO format: <class_id> <x_center> <y_center> <width> <height>
                    cls_id = int(line.split()[0])
                    categories.append(cls_id)
            gt_data[img_id] = categories
    return gt_data

def evaluate_hackathon_score(model_path, data_yaml_path, conf_threshold=0.25, iou_threshold=0.7):
    # Load configuration
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    root_dir = data_config['path']
    val_img_dir = os.path.join(root_dir, data_config['val'])
    val_label_dir = os.path.join(root_dir, data_config['val'].replace('images', 'labels'))
    
    # Load Ground Truth
    print(f"Loading ground truth from {val_label_dir}...")
    gt_labels = load_gt_labels(val_label_dir)
    
    # Load Model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Run Inference
    print(f"Running inference on {len(gt_labels)} images...")
    results = model.predict(
        source=val_img_dir,
        conf=conf_threshold,
        iou=iou_threshold,
        save=False,
        verbose=False,
        stream=True,
        device='mps'
    )
    
    total_images = 0
    exact_matches = 0
    count_errors = 0
    category_errors = 0 # Count matches, but IDs don't
    
    for res in tqdm(results, total=len(gt_labels)):
        img_id = Path(res.path).stem
        if img_id not in gt_labels:
            continue
            
        total_images += 1
        gt_cats = gt_labels[img_id]
        
        # Extract predicted categories
        if len(res.boxes) > 0:
            pred_cats = res.boxes.cls.cpu().numpy().astype(int).tolist()
        else:
            pred_cats = []
            
        # Comparison logic: Exact Match (order-agnostic)
        # We use Counter to handle multiple instances of the same category
        if Counter(pred_cats) == Counter(gt_cats):
            exact_matches += 1
        else:
            # Analyze error type
            if len(pred_cats) != len(gt_cats):
                count_errors += 1
            else:
                category_errors += 1
                
    # Calculate Score
    accuracy = (exact_matches / total_images) * 100 if total_images > 0 else 0
    
    print("\n" + "="*30)
    print("HACKATHON EVALUATION REPORT")
    print("="*30)
    print(f"Total Images Evaluated: {total_images}")
    print(f"Exact Matches:         {exact_matches}")
    print(f"Count Discrepancies:   {count_errors}")
    print(f"Category Mis-matches:  {category_errors}")
    print("-"*30)
    print(f"FINAL SCORE:           {accuracy:.2f}%")
    print("="*30)
    print("Note: Goal is to maximize Exact Matches (100% is perfect).")

if __name__ == "__main__":
    # Example usage with latest augmented weights
    MODEL = "/Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/runs/detect/augmented_v8n2/weights/best.pt"
    DATA_YAML = "/Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/data/data_subset.yaml"
    
    if os.path.exists(MODEL):
        evaluate_hackathon_score(MODEL, DATA_YAML)
    else:
        print(f"Error: Model not found at {MODEL}")
