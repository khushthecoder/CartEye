import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from tqdm import tqdm
import numpy as np

def calculate_iou(box1, box2):
    """
    Calculate IOU between two boxes in YOLO format [cls, x_c, y_c, w, h]
    """
    b1_x1, b1_y1 = box1[1] - box1[3]/2, box1[2] - box1[4]/2
    b1_x2, b1_y2 = box1[1] + box1[3]/2, box1[2] + box1[4]/2
    b2_x1, b2_y1 = box2[1] - box2[3]/2, box2[2] - box2[4]/2
    b2_x2, b2_y2 = box2[1] + box2[3]/2, box2[2] + box2[4]/2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y1_bottom = max(b1_y1, b2_y1) 
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = box1[3] * box1[4]
    b2_area = box2[3] * box2[4]
    
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-6)
    return iou

def run_eda(data_yaml_path, labels_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    cat_names = data_config['names']
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    all_data = []
    image_stats = []
    
    for label_file in tqdm(label_files, desc="Parsing labels"):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        num_objects = len(lines)
        image_stats.append({'file': os.path.basename(label_file), 'count': num_objects})
        
        boxes = []
        for line in lines:
            parts = list(map(float, line.strip().split()))
            if not parts: continue
            cls = int(parts[0])
            all_data.append({
                'class_id': cls,
                'class_name': cat_names[cls],
                'x_center': parts[1],
                'y_center': parts[2],
                'width': parts[3],
                'height': parts[4],
                'area': parts[3] * parts[4],
                'aspect_ratio': parts[3] / (parts[4] + 1e-6)
            })
            boxes.append(parts)
        
        if len(boxes) > 1:
            max_iou = 0
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    iou = calculate_iou(boxes[i], boxes[j])
                    if iou > max_iou:
                        max_iou = iou
            image_stats[-1]['max_iou'] = max_iou
        else:
            image_stats[-1]['max_iou'] = 0

    df = pd.DataFrame(all_data)
    img_df = pd.DataFrame(image_stats)
    
    plt.figure(figsize=(20, 10))
    class_counts = df['class_name'].value_counts()
    sns.barplot(x=class_counts.index[:50], y=class_counts.values[:50]) 
    plt.xticks(rotation=90)
    plt.title("Top 50 Categories by Frequency")
    plt.savefig(os.path.join(output_dir, "class_distribution_top50.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(img_df['count'], bins=30, kde=True)
    plt.title("Distribution of Object Counts per Image")
    plt.xlabel("Number of Objects")
    plt.savefig(os.path.join(output_dir, "objects_per_image.png"))
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.hexbin(df['width'], df['height'], gridsize=30, cmap='YlGnBu')
    plt.colorbar(label='Count')
    plt.xlabel('Normalized Width')
    plt.ylabel('Normalized Height')
    plt.title('BBox Dimensions Density')
    plt.savefig(os.path.join(output_dir, "bbox_dimensions.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(img_df[img_df['count'] > 1]['max_iou'], bins=30, kde=True)
    plt.title("Distribution of Max IOU per Image (Crowdedness)")
    plt.xlabel("Max IOU")
    plt.savefig(os.path.join(output_dir, "max_iou_distribution.png"))
    plt.close()

    summary = {
        "Total Images": len(img_df),
        "Total Objects": len(df),
        "Avg Objects/Image": img_df['count'].mean(),
        "Max Objects/Image": img_df['count'].max(),
        "Median Objects/Image": img_df['count'].median(),
        "Images with Overlap (IOU > 0.5)": len(img_df[img_df['max_iou'] > 0.5]),
        "Smallest Object Area": df['area'].min(),
        "Largest Object Area": df['area'].max(),
        "Avg Aspect Ratio": df['aspect_ratio'].mean()
    }
    
    print("\n=== EDA SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    
    with open(os.path.join(output_dir, 'summary.yaml'), 'w') as f:
        yaml.dump(summary, f)

if __name__ == "__main__":
    run_eda(
        "/Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/data/data.yaml",
        "/Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/data/processed/train/labels",
        "/Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/reports/eda"
    )