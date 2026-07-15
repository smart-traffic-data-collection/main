from ultralytics import YOLO
import os

# -----------------------
# CONFIG (CHANGE ONLY THIS)
# -----------------------
MODEL_PATH = "/home/nawab/traffic_project/yolo/best.pt"

IMAGE_FOLDER = "/home/nawab/traffic_project/datasets/UrbanIng-V2X/dataset/20241126_0017_crossing1_00/crossing1_13_thermal_camera"

OUTPUT_DIR = "/home/nawab/traffic_project/yolo_predictions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# LOAD MODEL
# -----------------------
model = YOLO(MODEL_PATH)

# -----------------------
# RUN ON FULL FOLDER
# -----------------------
results = model.predict(
    source=IMAGE_FOLDER,
    save=True,        # saves images with boxes
    save_txt=True,    # saves YOLO txt labels
    project=OUTPUT_DIR,
    name="predictions",
    conf=0.25
)

print("DONE ✔ YOLO inference completed")
print("Saved at:", OUTPUT_DIR)

