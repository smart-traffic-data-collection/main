import cv2
from collections import defaultdict
from ultralytics import YOLO

CUSTOM_YAML = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/scripts/custom_thermal_tracker.yaml"
TARGET_VIDEO = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/UrbanIng-V2X/dataset/20241126_0008_crossing1_00/crossing1_13_thermal_camera/13.mp4"
CUSTOM_WEIGHTS = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/models/finals/urbaning_retrain_augmented_v2/weights/best.pt"

def main():
    print(f"Loading custom model: {CUSTOM_WEIGHTS}")
    model = YOLO(CUSTOM_WEIGHTS)
    cap = cv2.VideoCapture(TARGET_VIDEO)
    track_history = defaultdict(list)
    frame_counter = 0

    print(f"Processing... (Tracker: custom_thermal_tracker.yaml)")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame_counter += 1
        
        results = model.track(frame, persist=True, tracker=CUSTOM_YAML, conf=0.10, imgsz=1024, verbose=False)
        
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            for track_id in track_ids:
                track_history[track_id].append(frame_counter)

    cap.release()

    total_unique_tracks = len(track_history)
    short_lived_tracks = 0
    total_missing_frames = 0
    total_tracked_frames = 0
    
    NOISE_FRAME_THRESHOLD = 5

    for track_id, appearance_frames in track_history.items():
        total_tracked_frames += len(appearance_frames)
        
        if len(appearance_frames) < NOISE_FRAME_THRESHOLD:
            short_lived_tracks += 1
            
        # Accurately count missing frames within a continuous track
        for i in range(len(appearance_frames) - 1):
            gap = appearance_frames[i + 1] - appearance_frames[i]
            if gap > 1:
                total_missing_frames += (gap - 1)

    # NEW METRIC: Based on volume of successful tracking vs dropouts
    if total_tracked_frames > 0:
        dropout_rate = total_missing_frames / (total_tracked_frames + total_missing_frames)
        noise_penalty = (short_lived_tracks / total_unique_tracks) * 100 if total_unique_tracks > 0 else 0
        stability_index = max(0.0, 100.0 - (dropout_rate * 100) - (noise_penalty * 0.5))
    else:
        stability_index = 0.0

    print("\n" + "="*50)
    print("   SMART TRAFFIC DATA: TRACKING STABILITY REPORT   ")
    print("="*50)
    print(f"Total Unique Track IDs       : {total_unique_tracks} (Realistic Vehicle Count)")
    print(f"Short-Lived Noise Tracks     : {short_lived_tracks}")
    print(f"Internal Track Dropouts      : {total_missing_frames} frames missing within tracks")
    print("-" * 50)
    print(f"True Track Reliability Score : {stability_index:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
