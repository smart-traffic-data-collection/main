import os
import glob
import shutil

# Paths based on your workstation structure
dataset_dir = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/UrbanIng-V2X/dataset"
export_dir = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/roboflow_upload_500"

# Target dataset size
TARGET_FRAMES = 500

# The exact 5 crossing folders you specified
target_cameras = [
    "crossing2_13_thermal_camera",
    "crossing2_14_thermal_camera",
    "crossing2_15_thermal_camera",
    "crossing2_33_thermal_camera",
    "crossing2_34_thermal_camera"
]

# Create the fresh export directory
os.makedirs(export_dir, exist_ok=True)
print("Scanning sequences 0001 and 0004 for the 5 target cameras...")

all_images = []

# Loop through each specific camera folder to gather images
for camera in target_cameras:
    search_pattern = os.path.join(dataset_dir, "*000[14]*", camera, "*.jpg")
    all_images.extend(glob.glob(search_pattern))

# Sort to maintain chronological order for proper time-distributed slicing
all_images = sorted(all_images)
total_found = len(all_images)

print(f"Found {total_found} total thermal images matching the criteria.")

if total_found == 0:
    print("Error: No images found. Double check the file paths.")
else:
    # Calculate the step size to evenly distribute the 500 frames
    step = max(1, total_found // TARGET_FRAMES)
    
    # Slice the array using the step
    selected_images = all_images[::step][:TARGET_FRAMES]
    
    print(f"Skipping every {step} frames to extract exactly {len(selected_images)} highly diverse images...")
    
    # Copy files to the export folder
    for img_path in selected_images:
        file_name = os.path.basename(img_path)
        
        # Grab the camera name and sequence name to prevent file overwrites
        camera_folder = os.path.basename(os.path.dirname(img_path))
        seq_folder = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        
        # Create a clean, traceable filename (e.g., 20241126_0001..._crossing2_13..._frameX.jpg)
        new_name = f"{seq_folder}_{camera_folder}_{file_name}"
        dest_path = os.path.join(export_dir, new_name)
        
        shutil.copy2(img_path, dest_path)

    print(f"Success! {len(selected_images)} frames are waiting in: {export_dir}")
