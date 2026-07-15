import cv2
import os

IMAGE_DIR = '/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/UrbanIng-V2X/dataset/20241126_0017_crossing1_00/crossing1_13_thermal_camera'
LABEL_DIR = '/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/UrbanIng-V2X/yolo_labels/labels_0017/yolo_crossing1_13_thermal_camera'
OUTPUT_DIR = '/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/UrbanIng-V2X/yolo_labels/check_proj_0017_00/check_projections'

os.makedirs(OUTPUT_DIR, exist_ok=True)
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg'))]

# PRE-FILTER: Find only the images that have cars in them
images_with_cars = []
for img_name in image_files:
    base_name = os.path.splitext(img_name)[0]
    label_path = os.path.join(LABEL_DIR, f"{base_name}.txt")
    # If the text file exists and is larger than 0 bytes (meaning it has YOLO coordinates)
    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
        images_with_cars.append(img_name)

print(f"Found {len(images_with_cars)} images containing vehicles. Drawing the first {len(images_with_cars)} ...")

for img_name in images_with_cars[:len(images_with_cars)]:
    img_path = os.path.join(IMAGE_DIR, img_name)
    base_name = os.path.splitext(img_name)[0]
    label_path = os.path.join(LABEL_DIR, f"{base_name}.txt")
    
    img = cv2.imread(img_path)
    if img is None: continue
    img_h, img_w = img.shape[:2]
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, cx, cy, w, h = map(float, parts)
                
                center_x, center_y = int(cx * img_w), int(cy * img_h)
                box_w, box_h = int(w * img_w), int(h * img_h)
                
                x1, y1 = int(center_x - box_w / 2), int(center_y - box_h / 2)
                x2, y2 = int(center_x + box_w / 2), int(center_y + box_h / 2)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"Class: {int(class_id)}", (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), img)

print("Done! Check the 'check_projections' folder now.")