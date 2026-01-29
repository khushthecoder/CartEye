import os
import yaml
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

def get_ground_truth_counts(label_dir):
    """Counts the number of objects in each label file."""
    counts = {}
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            path = os.path.join(label_dir, label_file)
            with open(path, 'r') as f:
                lines = f.readlines()
                counts[Path(label_file).stem] = len(lines)
    return counts

def evaluate_thresholds(model_path, data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    root_dir = data_config['path']
    val_img_dir = os.path.join(root_dir, data_config['val'])
    val_label_dir = os.path.join(root_dir, data_config['val'].replace('images', 'labels'))
    
    gt_counts = get_ground_truth_counts(val_label_dir)
    image_files = [f for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # High confidence range to filter noise from 1-epoch model
    conf_range = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    iou_range = [0.45, 0.6, 0.7]
    
    results = []
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    pbar = tqdm(total=len(conf_range) * len(iou_range), desc="Optimizing Thresholds")
    for conf in conf_range:
        for iou in iou_range:
            pbar.set_postfix({"conf": conf, "iou": iou})
            print(f"Evaluating conf={conf}, iou={iou}...")
            total_abs_error = 0
            exact_matches = 0
            total_predicted = 0
            total_actual = 0
            
            preds = model.predict(
                source=val_img_dir,
                conf=conf,
                iou=iou,
                save=False,
                verbose=False,
                device='mps',
                stream=True
            )
            
            count_processed = 0
            for pred in preds:
                img_stem = Path(pred.path).stem
                pred_count = len(pred.boxes)
                actual_count = gt_counts.get(img_stem, 0)
                
                error = abs(pred_count - actual_count)
                total_abs_error += error
                if error == 0:
                    exact_matches += 1
                
                total_predicted += pred_count
                total_actual += actual_count
                count_processed += 1
            
            mae = total_abs_error / count_processed
            accuracy = (exact_matches / count_processed) * 100
            
            results.append({
                'conf': conf,
                'iou': iou,
                'mae': mae,
                'accuracy': accuracy,
                'total_pred': total_predicted,
                'total_actual': total_actual
            })
            
            pbar.update(1)
            print(f"  MAE: {mae:.4f} | Accuracy: {accuracy:.2f}%")
    
    pbar.close()
            
    df = pd.DataFrame(results)
    output_path = "reports/counting_optimization_results.csv"
    os.makedirs("reports", exist_ok=True)
    df.to_csv(output_path, index=False)
    
    best_by_mae = df.loc[df['mae'].idxmin()]
    best_by_acc = df.loc[df['accuracy'].idxmax()]
    
    print("\n=== OPTIMIZATION SUMMARY ===")
    print(f"Best by MAE: conf={best_by_mae['conf']}, iou={best_by_mae['iou']} (MAE: {best_by_mae['mae']:.4f})")
    print(f"Best by Accuracy: conf={best_by_acc['conf']}, iou={best_by_acc['iou']} (Accuracy: {best_by_acc['accuracy']:.2f}%)")
    print(f"Detailed results saved to {output_path}")

if __name__ == "__main__":
    MODEL_PATH = "/Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/runs/detect/augmented_v8n2/weights/best.pt"
    DATA_YAML = "/Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/data/data_subset.yaml"
    
    if os.path.exists(MODEL_PATH):
        evaluate_thresholds(MODEL_PATH, DATA_YAML)
    else:
        print(f"Error: Weights not found at {MODEL_PATH}")
        runs_dir = "/Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/runs/detect"
        print(f"Available runs: {os.listdir(runs_dir)}")
