import cv2
import math
from collections import defaultdict
from ultralytics import YOLO

CUSTOM_YAML = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/scripts/custom_thermal_tracker.yaml"
TARGET_VIDEO = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/UrbanIng-V2X/dataset/20241126_0008_crossing1_00/crossing1_13_thermal_camera/13.mp4"
CUSTOM_WEIGHTS = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/models/finals/urbaning_flir_freeze/weights/best.pt"
OUTPUT_VIDEO = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/audit_output_flir_freeze.mp4"

def main():
    print(f"Loading custom model for Visual Audit...")
    model = YOLO(CUSTOM_WEIGHTS)
    cap = cv2.VideoCapture(TARGET_VIDEO)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    track_history = defaultdict(list)
    track_centroids = defaultdict(dict)
    frame_counter = 0

    print("Processing video and generating annotated output...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame_counter += 1
        
        results = model.track(frame, persist=True, tracker=CUSTOM_YAML, conf=0.10, imgsz=1024, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            boxes = results[0].boxes.xywh.cpu().tolist()
            
            for i, track_id in enumerate(track_ids):
                track_history[track_id].append(frame_counter)
                track_centroids[track_id][frame_counter] = (boxes[i][0], boxes[i][1])

    cap.release()
    out.release()
    print(f"\nVideo saved to: {OUTPUT_VIDEO}")

    # --- ADVANCED Spatial ID Switch Logic ---
    NOISE_FRAME_THRESHOLD = 5
    MAX_IDSW_FRAME_GAP = 60
    MAX_IDSW_PIXEL_DIST = 150
    EDGE_MARGIN = 50  # Pixels from the edge to ignore exits/entries

    track_starts = {} 
    track_ends = {}   

    # Helper function to check if a car is at the edge of the camera view
    def is_near_edge(cx, cy):
        return cx < EDGE_MARGIN or cx > (width - EDGE_MARGIN) or cy < EDGE_MARGIN or cy > (height - EDGE_MARGIN)

    for tid, frames in track_history.items():
        if len(frames) >= NOISE_FRAME_THRESHOLD:
            start_coord = track_centroids[tid][frames[0]]
            end_coord = track_centroids[tid][frames[-1]]
            
            # Only record starts/ends if they happen INSIDE the intersection, not at the edge
            if not is_near_edge(start_coord[0], start_coord[1]):
                track_starts[tid] = (frames[0], start_coord)
            if not is_near_edge(end_coord[0], end_coord[1]):
                track_ends[tid] = (frames[-1], end_coord)

    switch_events = []

    for start_tid, (start_frame, start_centroid) in track_starts.items():
        for end_tid, (end_frame, end_centroid) in track_ends.items():
            if start_tid == end_tid: continue
            
            frame_diff = start_frame - end_frame
            if 1 <= frame_diff <= MAX_IDSW_FRAME_GAP:
                dist = math.hypot(start_centroid[0] - end_centroid[0], start_centroid[1] - end_centroid[1])
                if dist < MAX_IDSW_PIXEL_DIST:
                    switch_events.append({
                        'frame': start_frame, 
                        'old_id': end_tid, 
                        'new_id': start_tid,
                        'gap': frame_diff
                    })
                    break 

    switch_events = sorted(switch_events, key=lambda x: x['frame'])

    print("\n" + "="*55)
    print("   FILTERED SINGLE-CAR ID SWITCH TIMELINE LOG   ")
    print("="*55)
    if len(switch_events) == 0:
        print("No internal ID switches detected (Edge Exits Ignored).")
    else:
        for event in switch_events:
            print(f"Frame {event['frame']:<4} | Track ID {event['old_id']:<3} died -> Reassigned as ID {event['new_id']:<3} (Gap: {event['gap']} frames)")
    print("="*55)

if __name__ == "__main__":
    main()
