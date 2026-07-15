import cv2
import math
from collections import defaultdict
from ultralytics import YOLO

# --- CONFIGURATION ---
CUSTOM_YAML = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/scripts/custom_thermal_tracker.yaml"
TARGET_VIDEO = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/UrbanIng-V2X/dataset/20241126_0008_crossing1_00/crossing1_13_thermal_camera/13.mp4"
BASE_DIR = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/models/finals/"

# Your 4 models to compare
MODELS_TO_TEST = {
    "Fresh_12k_X": f"{BASE_DIR}urbaning_fresh_12k_X/weights/best.pt",
    "Master_Pipeline": f"{BASE_DIR}urbaning_master_pipeline/weights/best.pt",
    "Master_Pipeline_DA": f"{BASE_DIR}urbaning_master_pipeline_DA/weights/best.pt",
    "Retrain_Aug_v2": f"{BASE_DIR}urbaning_retrain_augmented_v2/weights/best.pt"
}

def evaluate_model(model_name, weights_path):
    print(f"[{model_name}] Running inference...")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        return {"error": "Weights not found"}

    cap = cv2.VideoCapture(TARGET_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    track_history = defaultdict(list)
    track_centroids = defaultdict(dict)
    frame_counter = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame_counter += 1
        
        results = model.track(frame, persist=True, tracker=CUSTOM_YAML, conf=0.10, imgsz=1024, verbose=False)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            boxes = results[0].boxes.xywh.cpu().tolist()
            for i, track_id in enumerate(track_ids):
                track_history[track_id].append(frame_counter)
                track_centroids[track_id][frame_counter] = (boxes[i][0], boxes[i][1])
    cap.release()

    # Calculate Metrics
    EDGE_MARGIN = 50
    NOISE_FRAME_THRESHOLD = 5
    
    total_unique_tracks = len(track_history)
    short_lived_tracks = 0
    total_missing_frames = 0
    total_tracked_frames = 0
    valid_ids = 0

    def is_near_edge(cx, cy):
        return cx < EDGE_MARGIN or cx > (width - EDGE_MARGIN) or cy < EDGE_MARGIN or cy > (height - EDGE_MARGIN)

    for tid, frames in track_history.items():
        if len(frames) < NOISE_FRAME_THRESHOLD:
            short_lived_tracks += 1
            continue
            
        valid_ids += 1
        total_tracked_frames += len(frames)
        
        # Count internal dropouts
        for i in range(len(frames) - 1):
            gap = frames[i + 1] - frames[i]
            if gap > 1:
                total_missing_frames += (gap - 1)

    # Reliability Math
    if total_tracked_frames > 0:
        dropout_rate = total_missing_frames / (total_tracked_frames + total_missing_frames)
        noise_penalty = (short_lived_tracks / total_unique_tracks) * 100 if total_unique_tracks > 0 else 0
        stability_index = max(0.0, 100.0 - (dropout_rate * 100) - (noise_penalty * 0.5))
    else:
        stability_index = 0.0

    return {
        "valid_vehicles": valid_ids,
        "noise_tracks": short_lived_tracks,
        "missing_frames": total_missing_frames,
        "stability": stability_index
    }

def main():
    print("\nStarting Automated Benchmark across 4 models...\n")
    results_table = {}
    
    for name, path in MODELS_TO_TEST.items():
        results_table[name] = evaluate_model(name, path)
        
    print("\n" + "="*85)
    print(f"{'MODEL NAME':<25} | {'VEHICLE COUNT':<15} | {'NOISE TRACKS':<15} | {'DROPOUTS':<10} | {'STABILITY':<10}")
    print("="*85)
    
    for name, metrics in results_table.items():
        if "error" in metrics:
            print(f"{name:<25} | {'ERROR: WEIGHTS NOT FOUND':<55}")
        else:
            print(f"{name:<25} | {metrics['valid_vehicles']:<15} | {metrics['noise_tracks']:<15} | {metrics['missing_frames']:<10} | {metrics['stability']:.2f}%")
    print("="*85)

if __name__ == "__main__":
    main()
