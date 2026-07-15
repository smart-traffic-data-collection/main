import os
import json
import numpy as np
import cv2

# --- Base Directory Configurations ---
BASE_DIR = '/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/UrbanIng-V2X'
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
LABELS_DIR = os.path.join(BASE_DIR, 'labels')
OUTPUT_BASE = os.path.join(BASE_DIR, 'yolo_labels')

CLASS_MAP = {
    "car": 0, "truck": 1, "bus": 2, "van": 3,
    "pedestrian": 4, "bicycle": 5, "motorcycle": 6,
    "cyclist": 5, "bike": 5, "person": 4
}

IMG_WIDTH, IMG_HEIGHT = 640, 480

# --- Core Mathematical Functions ---
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
    
    min_x, max_x = max(0, min_x), min(IMG_WIDTH, max_x)
    min_y, max_y = max(0, min_y), min(IMG_HEIGHT, max_y)
    
    if min_x >= IMG_WIDTH or max_x <= 0 or min_y >= IMG_HEIGHT or max_y <= 0: 
        return None
        
    box_width = (max_x - min_x) / IMG_WIDTH
    box_height = (max_y - min_y) / IMG_HEIGHT
    
    pixel_w = box_width * IMG_WIDTH
    pixel_h = box_height * IMG_HEIGHT
    
    if pixel_w * pixel_h < 225 or pixel_w < 10 or pixel_h < 10: 
        return None
        
    return (((max_x + min_x) / 2) / IMG_WIDTH, ((max_y + min_y) / 2) / IMG_HEIGHT, box_width, box_height)

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
    
    return inter_area / min(area1, area2)

def apply_nms(yolo_boxes, ios_threshold=0.60):
    keep = []
    yolo_boxes = sorted(yolo_boxes, key=lambda b: b[3]*b[4], reverse=True)
    
    for box in yolo_boxes:
        discard = False
        for kept_box in keep:
            if calculate_ios(box, kept_box) > ios_threshold:
                discard = True
                break
        if not discard: keep.append(box)
    return keep

# --- Sequence Processing Engine ---
def process_camera(calib_file, tracks_file, target_camera_path, output_dir, camera_name):
    os.makedirs(output_dir, exist_ok=True)
    with open(calib_file, 'r') as f: calib_data = json.load(f)
    with open(tracks_file, 'r') as f: tracks_data = json.load(f)
        
    try:
        K, D, rvec, tvec, cTg = load_camera_calibration(calib_data, camera_name)
    except KeyError:
        print(f"    [WARNING] Camera {camera_name} not found in calibration. Skipping.")
        return

    k1 = D[0]
    max_r_squared = abs(1.0 / k1) * 0.8 if k1 < 0 else 4.0

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
            
            if np.any(corners_cam[:, 2] <= 0.5): continue
                
            r_squared = (corners_cam[:, 0] / corners_cam[:, 2])**2 + (corners_cam[:, 1] / corners_cam[:, 2])**2
            if np.any(r_squared > max_r_squared): continue 
            
            corners_2d, _ = cv2.projectPoints(corners_3d, rvec, tvec, K, D)
            yolo_box = get_yolo_bounding_box(corners_2d)
            
            if yolo_box:
                cx, cy, bw, bh = yolo_box
                ts_ms = int(ts * 1000)
                if ts_ms not in tracker_frames: tracker_frames[ts_ms] = []
                tracker_frames[ts_ms].append((class_id, cx, cy, bw, bh))

    image_files = [f for f in os.listdir(target_camera_path) if f.endswith(('.png', '.jpg'))]
    tracker_timestamps = list(tracker_frames.keys())
    
    success_count = 0
    for img_name in image_files:
        img_base = os.path.splitext(img_name)[0]
        try: img_ts_ms = int(img_base)
        except ValueError: continue
            
        time_diff = 999999
        if tracker_timestamps:
            closest_track_ts = min(tracker_timestamps, key=lambda k: abs(k - img_ts_ms))
            time_diff = abs(closest_track_ts - img_ts_ms)
        
        label_filename = os.path.join(output_dir, f"{img_base}.txt")
        
        if time_diff <= 60 and closest_track_ts in tracker_frames:
            raw_boxes = tracker_frames[closest_track_ts]
            clean_boxes = apply_nms(raw_boxes, ios_threshold=0.60)
            
            with open(label_filename, 'w') as f:
                for box in clean_boxes: 
                    f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
            success_count += 1
        else:
            open(label_filename, 'w').close()
            
    print(f"    [OK] Processed {camera_name}: generated {success_count} valid label files.")

# --- Targeted Sequence Loop ---
if __name__ == "__main__":
    print("Running 3D-to-2D Projection strictly for sequence 0008...\n")
    
    # Explicitly target the two 0008 folders
    target_folders = [
        "20241126_0008_crossing1_00",
        "20241126_0008_crossing1_01"
    ]
    
    for seq_folder in target_folders:
        seq_path = os.path.join(DATASET_DIR, seq_folder)
        
        if not os.path.exists(seq_path):
            print(f"  [ERROR] Folder {seq_folder} not found in dataset directory!")
            continue
            
        print(f"➔ Processing Sequence: {seq_folder}")
        
        calib_file = os.path.join(seq_path, 'calibration.json')
        tracks_file = os.path.join(LABELS_DIR, f"{seq_folder}.json")
        
        if not os.path.exists(calib_file) or not os.path.exists(tracks_file):
            print(f"  [SKIPPED] Missing calibration or track data for {seq_folder}")
            continue
            
        # Extract the exact identifier to prevent merging (e.g. '0008_00')
        parts = seq_folder.split('_')
        seq_num_exact = f"{parts[1]}_{parts[3]}" 
        
        for cam_folder in os.listdir(seq_path):
            if 'thermal_camera' in cam_folder:
                cam_path = os.path.join(seq_path, cam_folder)
                
                # Output path directly mapped: yolo_labels/labels_0008_00/yolo_crossing1_...
                out_dir = os.path.join(OUTPUT_BASE, f"labels_{seq_num_exact}", f"yolo_{cam_folder}")
                
                print(f"  Processing Camera: {cam_folder}")
                process_camera(
                    calib_file=calib_file, 
                    tracks_file=tracks_file, 
                    target_camera_path=cam_path, 
                    output_dir=out_dir, 
                    camera_name=cam_folder
                )
                
    print("\nProjection complete! Sequences 0008_00 and 0008_01 are fully distinct.")