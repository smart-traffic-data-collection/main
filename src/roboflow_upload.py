from roboflow import Roboflow
import os
import glob

# --- CONFIGURATION ---
# Replace with your actual Private API Key
API_KEY = "hCcjiviexVF2hjjHhPkj" 

# Replace with your workspace name and project name. 
# (e.g., if your url is app.roboflow.com/karels-workspace/d_500)
WORKSPACE_NAME = "karels-workspace-qdwtl" 
PROJECT_NAME = "d_500" 

# The folder where your 500 images are sitting
IMAGE_DIR = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/roboflow_upload_500"
# ----------------------

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)

image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
total_images = len(image_paths)

print(f"Found {total_images} images. Starting API upload...")

uploaded_count = 0
for img_path in image_paths:
    try:
        # Uploads the image without any labels
        project.upload(img_path)
        uploaded_count += 1
        if uploaded_count % 50 == 0:
            print(f"Progress: {uploaded_count} / {total_images} uploaded...")
    except Exception as e:
        print(f"Failed to upload {os.path.basename(img_path)}: {e}")

print(f"Upload complete! Successfully transferred {uploaded_count} images.")
