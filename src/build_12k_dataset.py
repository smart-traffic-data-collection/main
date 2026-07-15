import os
import shutil
import yaml

# --- Path Configurations ---
BASE_DIR = '/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/UrbanIng-V2X'
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
LABELS_BASE = os.path.join(BASE_DIR, 'yolo_labels')

# The final master directory
OUT_DIR = '/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/urbaning_12k'

# --- Initialization ---
print("Initializing UrbanIng 12k Dataset Compiler...")

# Create the strict YOLO folder structure
dirs_to_make = [
    os.path.join(OUT_DIR, 'images', 'train'),
    os.path.join(OUT_DIR, 'images', 'val'),
    os.path.join(OUT_DIR, 'labels', 'train'),
    os.path.join(OUT_DIR, 'labels', 'val')
]

for d in dirs_to_make:
    os.makedirs(d, exist_ok=True)

# --- Automated Sorting Engine ---
total_train = 0
total_val = 0

for seq_folder in sorted(os.listdir(DATASET_DIR)):
    if not seq_folder.startswith('20241126_'):
        continue

    # Extract sequence info (e.g., '0004', '0008', '00')
    parts = seq_folder.split('_')
    base_seq = parts[1]
    sub_seq = parts[3]

    # Map sequences to their strict train/val splits
    if base_seq == '0008':
        split = 'val'
        label_folder_name = f"labels_0008_{sub_seq}"
    elif base_seq in ['0001', '0004', '0010', '0013']:
        split = 'train'
        label_folder_name = f"labels_{base_seq}"
    else:
        continue  # Ignore sequences not in your targeted 12k list

    dataset_seq_path = os.path.join(DATASET_DIR, seq_folder)
    labels_seq_path = os.path.join(LABELS_BASE, label_folder_name)

    print(f"➔ Compiling {seq_folder} into {split.upper()}...")

    for cam_folder in os.listdir(dataset_seq_path):
        if 'thermal_camera' not in cam_folder:
            continue

        dataset_cam_path = os.path.join(dataset_seq_path, cam_folder)
        labels_cam_path = os.path.join(labels_seq_path, f"yolo_{cam_folder}")

        if not os.path.exists(labels_cam_path):
            continue

        moved_for_cam = 0
        for txt_file in os.listdir(labels_cam_path):
            if not txt_file.endswith('.txt'):
                continue

            base_name = os.path.splitext(txt_file)[0]
            txt_path = os.path.join(labels_cam_path, txt_file)

            # Check for corresponding image (.png or .jpg)
            img_path_png = os.path.join(dataset_cam_path, f"{base_name}.png")
            img_path_jpg = os.path.join(dataset_cam_path, f"{base_name}.jpg")
            
            img_path = img_path_png if os.path.exists(img_path_png) else img_path_jpg if os.path.exists(img_path_jpg) else None

            if img_path:
                # Appending the camera name prevents timestamp collisions across different camera feeds
                new_base_name = f"{base_name}_{cam_folder}"
                
                img_dest = os.path.join(OUT_DIR, 'images', split, f"{new_base_name}{os.path.splitext(img_path)[1]}")
                txt_dest = os.path.join(OUT_DIR, 'labels', split, f"{new_base_name}.txt")

                shutil.copy2(img_path, img_dest)
                shutil.copy2(txt_path, txt_dest)
                
                moved_for_cam += 1
                if split == 'train': total_train += 1
                if split == 'val': total_val += 1

        print(f"    - {cam_folder}: Copied {moved_for_cam} valid pairs.")

print(f"\n[DATASET BUILT] Total Train Images: {total_train} | Total Validation Images: {total_val}")

# --- Generate the 7-Class data.yaml ---
yaml_path = os.path.join(OUT_DIR, 'data.yaml')
yaml_content = {
    'path': OUT_DIR,
    'train': 'images/train',
    'val': 'images/val',
    'nc': 7,
    'names': {
        0: 'car',
        1: 'truck',
        2: 'bus',
        3: 'van',
        4: 'pedestrian',
        5: 'bicycle',
        6: 'motorcycle'
    }
}

with open(yaml_path, 'w') as f:
    yaml.dump(yaml_content, f, sort_keys=False)

print(f"[YAML GENERATED] Exact 7-class configuration saved to {yaml_path}")
print("Ready for YOLO Master Training.")