import json
import os
from pathlib import Path
from tqdm import tqdm

def convert_coco_to_yolo(json_path, cat_path, output_dir, img_dir, target_img_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    with open(cat_path, 'r') as f:
        cat_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(target_img_dir, exist_ok=True)

    images = {img['id']: img for img in data['images']}
    
    cat_id_map = {cat['id']: i for i, cat in enumerate(cat_data['categories'])}
    cat_names = [cat['name'] for cat in cat_data['categories']]

    labels_dict = {}
    for ann in tqdm(data['annotations'], desc=f"Converting {os.path.basename(json_path)}"):
        img_id = ann['image_id']
        if img_id not in images:
            continue
            
        img_info = images[img_id]
        file_name = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        x_min, y_min, width, height = ann['bbox']

        width = min(width, img_width - x_min)
        height = min(height, img_height - y_min)
        
        x_center = (x_min + width / 2.0) / img_width
        y_center = (y_min + height / 2.0) / img_height
        w_norm = width / img_width
        h_norm = height / img_height
        
        cat_id = cat_id_map[ann['category_id']]
        
        line = f"{cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
        
        label_file = os.path.join(output_dir, Path(file_name).stem + '.txt')
        if label_file not in labels_dict:
            labels_dict[label_file] = []
        labels_dict[label_file].append(line)

    for label_file, lines in tqdm(labels_dict.items(), desc=f"Writing {os.path.basename(json_path)} labels"):
        with open(label_file, 'w') as f:
            f.write("\n".join(lines) + "\n")

    for img_id, img_info in tqdm(images.items(), desc=f"Symlinking {os.path.basename(json_path)}"):
        src_path = os.path.join(img_dir, img_info['file_name'])
        dst_path = os.path.join(target_img_dir, img_info['file_name'])
        if not os.path.exists(dst_path):
            os.symlink(src_path, dst_path)
            
    return cat_names

if __name__ == "__main__":
    base_path = "/Users/khushchaudhari/Downloads/vista26/Vistas Dataset Public/Vistas Dataset Public"
    root_path = "/Users/khushchaudhari/Downloads/vista26"
    output_base = "/Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/data/processed"
    cat_path = os.path.join(base_path, "Categories.json")
    
    # Train
    cat_names = convert_coco_to_yolo(
        os.path.join(base_path, "instances_train.json"),
        cat_path,
        os.path.join(output_base, "train/labels"),
        os.path.join(base_path, "train"),
        os.path.join(output_base, "train/images")
    )
    

    convert_coco_to_yolo(
        os.path.join(base_path, "instances_test.json"),
        cat_path,
        os.path.join(output_base, "val/labels"),
        os.path.join(base_path, "test"),
        os.path.join(output_base, "val/images")
    )
    
    with open(os.path.join(root_path, "instances_val.json"), 'r') as f:
        comp_test_data = json.load(f)
    
    comp_test_img_dir = os.path.join(output_base, "test/images")
    os.makedirs(comp_test_img_dir, exist_ok=True)
    src_val_dir = os.path.join(base_path, "validation")
    
    for img_info in tqdm(comp_test_data['images'], desc="Symlinking competition test images"):
        src_path = os.path.join(src_val_dir, img_info['file_name'])
        dst_path = os.path.join(comp_test_img_dir, img_info['file_name'])
        if not os.path.exists(dst_path):
            os.symlink(src_path, dst_path)

    yaml_content = {
        'path': output_base,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {i: name for i, name in enumerate(cat_names)}
    }
    
    import yaml
    with open(os.path.join(output_base, '../data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
