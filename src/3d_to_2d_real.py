import os
import json
import numpy as np
import cv2

# --- Configuration ---
video_writer = None
BASE = "/home/nawab/traffic_project/datasets/UrbanIng-V2X"

sequence_id = "20241126_0017_crossing1_00"
camera_name = "crossing1_13_thermal_camera"

base_path = os.path.join(BASE, "dataset", sequence_id)

CALIB_FILE = os.path.join(base_path, "calibration.json")
TRACKS_FILE = os.path.join(BASE, "labels", f"{sequence_id}.json")

TARGET_CAMERA_PATH = os.path.join(base_path, camera_name)

OUTPUT_DIR = os.path.join(BASE, "yolo_labels", camera_name)

CLASS_MAP = {
    "car": 0, "truck": 1, "bus": 2, "van": 3,
    "pedestrian": 4, "bicycle": 5, "motorcycle": 6,
    "cyclist": 5, "bike": 5, "person": 4
}

IMG_WIDTH, IMG_HEIGHT = 640, 480

def load_camera_calibration(calibration_data, camera_name):
    cam_data = calibration_data[camera_name]
    K = np.array(cam_data["intrinsics"]["IntrinsicMatrix"]).T 
    radial = cam_data["intrinsics"]["RadialDistortion"]
    tangential = cam_data["intrinsics"]["TangentialDistortion"]
    D = np.array([radial[0], radial[1], tangential[0], tangential[1], radial[2]])
    cTg = np.array(cam_data["extrinsics"]["cTg"])
    rvec, _ = cv2.Rodrigues(cTg[:3, :3])
    tvec = cTg[:3, 3]
    return K, D, rvec, tvec, cTg

def compute_3d_box_corners(position, dimensions, yaw):
    x, y, z = position
    l, w, h = dimensions 
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2] 
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    corners_local = np.vstack([x_corners, y_corners, z_corners])
    return np.dot(R_z, corners_local).T + np.array([x, y, z])

def get_yolo_bounding_box(corners_2d):
    pts = corners_2d.reshape(-1, 2)
    min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
    min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
    
    # Clamp safely
    min_x, max_x = max(0, min_x), min(IMG_WIDTH, max_x)
    min_y, max_y = max(0, min_y), min(IMG_HEIGHT, max_y)
    
    if min_x >= IMG_WIDTH or max_x <= 0 or min_y >= IMG_HEIGHT or max_y <= 0: 
        return None
        
    box_width = (max_x - min_x) / IMG_WIDTH
    box_height = (max_y - min_y) / IMG_HEIGHT
    
    pixel_w = box_width * IMG_WIDTH
    pixel_h = box_height * IMG_HEIGHT
    
    # Drop boxes that are too small in total area (e.g. 15x15 = 225 pixels)
    if pixel_w * pixel_h < 225 or pixel_w < 10 or pixel_h < 10: 
        return None
        
    return (((max_x + min_x) / 2) / IMG_WIDTH, ((max_y + min_y) / 2) / IMG_HEIGHT, box_width, box_height)

# FINAL BOSS FIX 2: Intersection over Smaller Area (IoS)
def calculate_ios(box1, box2):
    x1_min, y1_min = box1[1] - box1[3]/2, box1[2] - box1[4]/2
    x1_max, y1_max = box1[1] + box1[3]/2, box1[2] + box1[4]/2
    x2_min, y2_min = box2[1] - box2[3]/2, box2[2] - box2[4]/2
    x2_max, y2_max = box2[1] + box2[3]/2, box2[2] + box2[4]/2
    
    inter_x_min, inter_y_min = max(x1_min, x2_min), max(y1_min, y2_min)
    inter_x_max, inter_y_max = min(x1_max, x2_max), min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min: return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = box1[3] * box1[4]
    area2 = box2[3] * box2[4]
    
    # Divide by the smaller box's area. If the small box is totally inside the big box, this equals 1.0!
    return inter_area / min(area1, area2)

def apply_nms(yolo_boxes, ios_threshold=0.60):
    keep = []
    # Sort boxes by area (Largest first). This ensures the big, correct box eats the tiny glitch box.
    yolo_boxes = sorted(yolo_boxes, key=lambda b: b[3]*b[4], reverse=True)
    
    for box in yolo_boxes:
        discard = False
        for kept_box in keep:
            if calculate_ios(box, kept_box) > ios_threshold:
                discard = True
                break
        if not discard: keep.append(box)
    return keep

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CALIB_FILE, 'r') as f: calib_data = json.load(f)
    with open(TRACKS_FILE, 'r') as f: tracks_data = json.load(f)
        
    K, D, rvec, tvec, cTg = load_camera_calibration(calib_data, camera_name)
    
    # Calculate mathematically when the camera's distortion polynomial will break
    k1 = D[0]
    if k1 < 0:
        # Prevent the equation from crossing zero (with a 20% safety margin)
        max_r_squared = abs(1.0 / k1) * 0.8 
    else:
        max_r_squared = 4.0

    tracker_frames = {}
    for track in tracks_data["tracks"]:
        obj_type = track.get("object_type", "").lower()
        class_id = CLASS_MAP.get(obj_type, -1)
        if class_id == -1: continue
            
        dims = track["dimensions"][0] 
        
        for i, ts in enumerate(track["timestamps"]):
            pos = track["positions"][i]
            ori = track["orientations"][i] if "orientations" in track else 0.0
            yaw = ori[2] if isinstance(ori, (list, tuple)) else float(ori)
            
            corners_3d = compute_3d_box_corners(pos, dims, yaw)
            corners_cam = np.dot(cTg[:3, :3], corners_3d.T).T + cTg[:3, 3]
            
            if np.any(corners_cam[:, 2] <= 0.5):
                continue
                
            # FINAL BOSS FIX 1: The Wraparound Check
            # Calculate r^2 for all points. If any point exceeds the polynomial limit, drop the car.
            r_squared = (corners_cam[:, 0] / corners_cam[:, 2])**2 + (corners_cam[:, 1] / corners_cam[:, 2])**2
            if np.any(r_squared > max_r_squared):
                continue 
            
            corners_2d, _ = cv2.projectPoints(corners_3d, rvec, tvec, K, D)
            yolo_box = get_yolo_bounding_box(corners_2d)
            
            if yolo_box:
                cx, cy, bw, bh = yolo_box
                ts_ms = int(ts * 1000)
                if ts_ms not in tracker_frames: tracker_frames[ts_ms] = []
                tracker_frames[ts_ms].append((class_id, cx, cy, bw, bh))

    image_files = [f for f in os.listdir(TARGET_CAMERA_PATH) if f.endswith(('.png', '.jpg'))]
    tracker_timestamps = list(tracker_frames.keys())
    
    success_count = 0
    for img_name in image_files:
        img_base = os.path.splitext(img_name)[0]
        try: img_ts_ms = int(img_base)
        except ValueError: continue
            
        if tracker_timestamps:
            closest_track_ts = min(tracker_timestamps, key=lambda k: abs(k - img_ts_ms))
            time_diff = abs(closest_track_ts - img_ts_ms)
        else:
            time_diff = 999999
        
        label_filename = os.path.join(OUTPUT_DIR, f"{img_base}.txt")
        
        if time_diff <= 60 and closest_track_ts in tracker_frames:
            raw_boxes = tracker_frames[closest_track_ts]
            
            # Apply the new IoS NMS logic
            clean_boxes = apply_nms(raw_boxes, ios_threshold=0.60)
            
            with open(label_filename, 'w') as f:
                for box in clean_boxes: 
                    f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                img_path = os.path.join(TARGET_CAMERA_PATH, img_name)
                img = cv2.imread(img_path)    

                if img is None:
                     continue

                h, w = img.shape[:2]

# init video writer once
                if video_writer is None:
                    video_writer = cv2.VideoWriter(
                     "output_2d_boxes_real.mp4",
                     cv2.VideoWriter_fourcc(*"mp4v"),
                    3,
                    (w, h)
                )

# draw boxes on image (optional but needed for video)
                for box in clean_boxes:
                    cx, cy, bw, bh = box[1], box[2], box[3], box[4]

                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

                video_writer.write(img)
            success_count += 1
        else:
            open(label_filename, 'w').close()
            
    print(f"Success! Wraparound distortion and NMS overlaps have been eradicated.")
    if video_writer is not None:
        video_writer.release()
        print("video saved")