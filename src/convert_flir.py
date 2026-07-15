import json
import os

def convert_coco_to_yolo(json_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Map COCO category IDs to YOLO IDs (0-indexed)
    # FLIR v2 usually: 1: person, 2: bicycle, 3: car
    cat_map = {1: 0, 2: 1, 3: 2} 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a dictionary for images
    images = {img['id']: img for img in data['images']}

    for ann in data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        
        if category_id not in cat_map:
            continue

        img = images[image_id]
        w_img, h_img = img['width'], img['height']
        
        # COCO: [x_min, y_min, width, height]
        x_min, y_min, w_box, h_box = ann['bbox']
        
        # YOLO: [class, x_center, y_center, width, height] normalized
        x_center = (x_min + w_box / 2) / w_img
        y_center = (y_min + h_box / 2) / h_img
        w_norm = w_box / w_img
        h_norm = h_box / h_img

        # Prepare filename (e.g., image_name.txt)
        file_name = os.path.splitext(img['file_name'].split('/')[-1])[0] + ".txt"
        
        with open(os.path.join(output_dir, file_name), 'a') as out_f:
            out_f.write(f"{cat_map[category_id]} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

# Run for Training set
print("Converting Training Labels...")
convert_coco_to_yolo(
    '/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/flir_dataset/FLIR_ADAS_v2/images_thermal_train/coco.json',
    '/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/flir_dataset/FLIR_ADAS_v2/images_thermal_train/labels'
)

# Run for Validation set
print("Converting Validation Labels...")
convert_coco_to_yolo(
    '/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/flir_dataset/FLIR_ADAS_v2/images_thermal_val/coco.json',
    '/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/flir_dataset/FLIR_ADAS_v2/images_thermal_val/labels'
)
print("Finished!")
