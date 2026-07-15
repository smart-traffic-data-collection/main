import cv2
import glob
import os

# 1. Point to your latest inference folder
image_folder = '/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/models/finals/visual_test_flir_freeze_0008_01'
video_name = '/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/models/finals/visual_test_flir_freeze_0008_01/visual_test_flir_freeze_0008_01.mp4'
fps = 30

# 2. Grab all JPGs and sort them chronologically
images = glob.glob(os.path.join(image_folder, '*.jpg'))
images.sort()

if not images:
    print("No images found! Check your folder path.")
    exit()

# 3. Get the dimensions from the first image
frame = cv2.imread(images[0])
height, width, layers = frame.shape

# 4. Initialize the Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

print(f"Stitching {len(images)} frames into a video...")

# 5. Loop through and compile the video
for i, image in enumerate(images):
    video.write(cv2.imread(image))
    if i % 100 == 0:
        print(f"Processed {i}/{len(images)} frames...")

# NO destroyAllWindows() here anymore! Just release the video safely.
video.release()
print(f"Success! Video saved to: {video_name}")
