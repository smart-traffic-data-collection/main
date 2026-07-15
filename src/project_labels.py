import json
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

DATA_ROOT = '/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/UrbanIng-V2X'

# The exact sequences required by your Master Plan
TARGET_SEQUENCES = [
    '20241126_0001_crossing2_00',
    '20241126_0004_crossing2_00',
    '20241126_0008_crossing1_00'
]

def build_3d_box(pos, dim, ori):
    flat_dim = np.array(dim).flatten()
    if len(flat_dim) < 3: return None
    l, w, h = flat_dim[:3] 
    
    corners_local = np.array([
        [ l/2,  w/2,  h/2], [ l/2, -w/2,  h/2],
        [-l/2, -w/2,  h/2], [-l/2,  w/2,  h/2],
        [ l/2,  w/2, -h/2], [ l/2, -w/2, -h/2],
        [-l/2, -w/2, -h/2], [-l/2,  w/2, -h/2]
    ])
    
    if isinstance(ori, (float, int)): rot = R.from_euler('z', ori)
    elif isinstance(ori, list) and len(ori) == 1: rot = R.from_euler('z', ori[0])
    elif len(ori) == 4: rot = R.from_quat(ori) 
    elif len(ori) == 3: rot = R.from_euler('xyz', ori) 
    else: rot = R.from_matrix(np.eye(3))
        
    return rot.apply(corners_local) + np.array(pos)

def get_yolo_class(obj_type):
    obj_type = obj_type.lower()
    if 'pedestrian' in obj_type or 'person' in obj_type: return 0
    if 'cyclist' in obj_type or 'bicycle' in obj_type or 'motorcycle' in obj_type: return 1
    if 'car' in obj_type or 'truck' in obj_type or 'bus' in obj_type or 'vehicle' in obj_type or 'van' in obj_type: return 2
    return None 

def process_sensor(seq_name, cam_name):
    print(f"\n---> Processing {seq_name} / {cam_name} <---")
    
    calib_path = os.path.join(DATA_ROOT, 'dataset', seq_name, 'calibration.json')
    label_path = os.path.join(DATA_ROOT, 'labels', f'{seq_name}.json')
    output_dir = os.path.join(DATA_ROOT, 'dataset', seq_name, cam_name, 'labels_yolo')
    
    if not os.path.exists(calib_path) or not os.path.exists(label_path):
        print("Missing calibration or labels. Skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(calib_path, 'r') as f: calib = json.load(f)
    with open(label_path, 'r') as f: labels_data = json.load(f)

    if cam_name not in calib:
        print(f"Camera {cam_name} not found in calibration. Skipping.")
        return

    K = np.array(calib[cam_name]['intrinsics']['IntrinsicMatrix'])
    if K.shape == (3, 3) and K[2, 2] != 1.0: K = K.T 

    cTg = np.array(calib[cam_name]['extrinsics']['cTg'])
    gTc = np.linalg.inv(cTg)
    img_width, img_height = calib[cam_name]['intrinsics']['ImageSize']

    def project_3d_to_2d(corners_3d_global):
        points_2d = []
        for corner in corners_3d_global:
            point_3d_hom = np.array([corner[0], corner[1], corner[2], 1.0])
            point_cam = np.dot(gTc, point_3d_hom) 
            if point_cam[2] <= 0: return None 
            point_img = np.dot(K, point_cam[:3])
            points_2d.append([point_img[0]/point_img[2], point_img[1]/point_img[2]])
            
        points_2d = np.array(points_2d)
        u_min, v_min = np.min(points_2d, axis=0)
        u_max, v_max = np.max(points_2d, axis=0)
        u_min, u_max = max(0, min(u_min, img_width)), max(0, min(u_max, img_width))
        v_min, v_max = max(0, min(v_min, img_height)), max(0, min(v_max, img_height))
        
        if u_min >= img_width or u_max <= 0 or v_min >= img_height or v_max <= 0: return None
        return [u_min, v_min, u_max, v_max]

    frames_data = {}
    for track in labels_data.get('tracks', []):
        yolo_class_id = get_yolo_class(track.get('object_type', ''))
        if yolo_class_id is None: continue 
            
        dim = track['dimensions']
        for t, p, o in zip(track['timestamps'], track['positions'], track['orientations']):
            if t not in frames_data: frames_data[t] = []
            corners_3d = build_3d_box(p, dim, o)
            if corners_3d is not None: frames_data[t].append((yolo_class_id, corners_3d))

    boxes_drawn = 0
    for timestamp, objects in frames_data.items():
        ts_str = str(int(timestamp)) if timestamp.is_integer() else str(timestamp).replace('.', '')
        if len(ts_str) < 13: ts_str = str(int(timestamp * 1000))
            
        yolo_lines = []
        for yolo_class_id, corners in objects:
            box_2d = project_3d_to_2d(corners)
            if box_2d is not None:
                u_min, v_min, u_max, v_max = box_2d
                w, h = u_max - u_min, v_max - v_min
                if (w / img_width) > 0.001 and (h / img_height) > 0.001:
                    yolo_lines.append(f"{yolo_class_id} {(u_min + w/2)/img_width:.6f} {(v_min + h/2)/img_height:.6f} {w/img_width:.6f} {h/img_height:.6f}")
                    boxes_drawn += 1
                    
        with open(os.path.join(output_dir, f"{ts_str}.txt"), 'w') as f: f.write("\n".join(yolo_lines))

    print(f"Success! Generated {boxes_drawn} boxes.")

# --- The Main Runner ---
for seq in TARGET_SEQUENCES:
    seq_path = os.path.join(DATA_ROOT, 'dataset', seq)
    if os.path.exists(seq_path):
        # Dynamically find any thermal cameras in this sequence folder
        thermal_cameras = [d for d in os.listdir(seq_path) if 'thermal' in d and os.path.isdir(os.path.join(seq_path, d))]
        for cam in thermal_cameras:
            process_sensor(seq, cam)
    else:
        print(f"Sequence folder not found: {seq}")

print("\n--- ALL BATCH PROCESSING COMPLETE ---")