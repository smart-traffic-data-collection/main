'''
Author - Ansh Tuli
'''

import cv2
import os
import glob

input_dir = "/Users/anshtuli/PycharmProjects/SmartTraffic_Vision/datasets/UrbanIng-V2X/dataset/raw_img"
output_dir = "/Users/anshtuli/PycharmProjects/SmartTraffic_Vision/datasets/UrbanIng-V2X/dataset/processed_img"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

for img_path in glob.glob(os.path.join(input_dir, "*.jpg")):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    enhanced = clahe.apply(img)

    # Save as PNG to avoid any quality loss
    save_path = os.path.join(output_dir, os.path.basename(img_path).replace('.jpg', '.png'))
    cv2.imwrite(save_path, enhanced)

print("Pre-processing completed.")