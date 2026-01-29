import os
import random
import yaml
import shutil

def create_subset(data_yaml_path, subset_ratio=0.1):
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    root_dir = data_config['path']
    
    for split in ['train', 'val']:
        src_img_dir = os.path.join(root_dir, f'{split}/images')
        src_label_dir = os.path.join(root_dir, f'{split}/labels')
        
        dst_dir = os.path.join(root_dir, f'{split}_subset')
        dst_img_dir = os.path.join(dst_dir, 'images')
        dst_label_dir = os.path.join(dst_dir, 'labels')
        
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_label_dir, exist_ok=True)
        
        all_images = [f for f in os.listdir(src_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        subset_count = int(len(all_images) * subset_ratio)
        subset_images = random.sample(all_images, subset_count)
        
        print(f"Creating {split} subset with {len(subset_images)} images...")
        
        for img_name in subset_images:
            src_img = os.path.join(src_img_dir, img_name)
            dst_img = os.path.join(dst_img_dir, img_name)
            if not os.path.exists(dst_img):
                os.symlink(src_img, dst_img)
                
            label_name = os.path.splitext(img_name)[0] + '.txt'
            src_label = os.path.join(src_label_dir, label_name)
            dst_label = os.path.join(dst_label_dir, label_name)
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)

    subset_config = data_config.copy()
    subset_config['train'] = 'train_subset/images'
    subset_config['val'] = 'val_subset/images'
    
    subset_yaml_path = os.path.join(os.path.dirname(data_yaml_path), 'data_subset.yaml')
    with open(subset_yaml_path, 'w') as f:
        yaml.dump(subset_config, f, sort_keys=False)
    
    print(f"New config created: {subset_yaml_path}")
    return subset_yaml_path

if __name__ == "__main__":
    create_subset("/Users/khushchaudhari/Documents/IITB_HACKATHAON/CartEye/data/data.yaml")
