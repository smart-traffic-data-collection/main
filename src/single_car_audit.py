import cv2
import math
from ultralytics import YOLO

CUSTOM_YAML = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/scripts/custom_thermal_tracker.yaml"
TARGET_VIDEO = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/UrbanIng-V2X/dataset/20241126_0008_crossing1_00/crossing1_13_thermal_camera/13.mp4"
CUSTOM_WEIGHTS = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/models/finals/urbaning_fresh/weights/best.pt"

# Output Files
OUTPUT_VIDEO = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/single_car_lineage_fresh_3.mp4"
OUTPUT_IMAGE = "/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/target_first_frame_fresh_3.jpg"

# ---------------------------------------------------------
TARGET_CAR_ID = 25
# ---------------------------------------------------------

def get_iou(box1, box2):
    """Calculates the exact overlap percentage between two boxes (0.0 to 1.0)"""
    x1_min, y1_min = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x1_max, y1_max = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    x2_min, y2_min = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_max, y2_max = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    xA, yA = max(x1_min, x2_min), max(y1_min, y2_min)
    xB, yB = min(x1_max, x2_max), min(y1_max, y2_max)

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (x1_max - x1_min) * (y1_max - y1_min)
    box2Area = (x2_max - x2_min) * (y2_max - y2_min)

    return interArea / float(box1Area + box2Area - interArea + 1e-5)

def main():
    print(f"\n[INIT] Loading custom model to audit STARTING ID: {TARGET_CAR_ID}...")
    model = YOLO(CUSTOM_WEIGHTS)
    cap = cv2.VideoCapture(TARGET_VIDEO)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    current_target_id = TARGET_CAR_ID
    target_class_id = None  
    is_tracking_active = False
    last_known_box = None
    frames_missing = 0
    total_switches = 0
    total_lost_frames = 0
    
    target_aliases = {TARGET_CAR_ID}
    known_other_cars = set() 
    
    EDGE_MARGIN = 30
    track_permanently_lost = False

    print(f"[PROCESS] Scanning video for ID {TARGET_CAR_ID}...")
    frame_counter = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame_counter += 1
        
        results = model.track(frame, persist=True, tracker=CUSTOM_YAML, conf=0.10, imgsz=1024, verbose=False)
        
        current_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes is not None and results[0].boxes.id is not None else []
        current_boxes = results[0].boxes.xywh.cpu().tolist() if results[0].boxes is not None else []
        current_classes = results[0].boxes.cls.int().cpu().tolist() if results[0].boxes is not None and results[0].boxes.cls is not None else []

        # 1. Wait for target to appear
        if not is_tracking_active and not track_permanently_lost:
            if current_target_id in current_ids:
                is_tracking_active = True
                idx = current_ids.index(current_target_id)
                target_class_id = current_classes[idx] 
                
                print(f" -> Frame {frame_counter}: Target ID {current_target_id} entered (Class ID: {target_class_id}).")
                
                cx, cy, w, h = current_boxes[idx]
                snapshot = frame.copy()
                cv2.rectangle(snapshot, (int(cx - w/2), int(cy - h/2)), (int(cx + w/2), int(cy + h/2)), (0, 255, 0), 3)
                cv2.putText(snapshot, f"INITIAL TARGET: ID {current_target_id}", (int(cx - w/2), int(cy - h/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imwrite(OUTPUT_IMAGE, snapshot)
            else:
                out.write(frame)
                continue 

        # 2. Track Target Lineage
        found_any_alias = False
        active_id_this_frame = None

        # Check A: Is our Primary ID alive?
        if current_target_id in current_ids and not track_permanently_lost:
            found_any_alias = True
            active_id_this_frame = current_target_id
            
        # Check B: Did it flicker to an already known alias?
        elif not track_permanently_lost:
            for alias_id in target_aliases:
                if alias_id in current_ids:
                    idx = current_ids.index(alias_id)
                    candidate_box = current_boxes[idx]
                    iou = get_iou(last_known_box, candidate_box)
                    
                    # STRICT IoU for aliases: Must have at least 20% overlap
                    if iou > 0.20:
                        found_any_alias = True
                        active_id_this_frame = alias_id
                        current_target_id = alias_id
                        break

        # If we successfully found our car this frame
        if found_any_alias:
            if frames_missing > 0:
                print(f" -> Frame {frame_counter}: Target tracking resumed on ID {active_id_this_frame} (Lost for {frames_missing} frames)")
                total_lost_frames += frames_missing
                frames_missing = 0
            
            idx = current_ids.index(active_id_this_frame)
            last_known_box = current_boxes[idx]
            cx, cy, w, h = last_known_box
            
            cv2.rectangle(frame, (int(cx - w/2), int(cy - h/2)), (int(cx + w/2), int(cy + h/2)), (0, 255, 0), 3)
            cv2.putText(frame, f"AUDIT ID: {current_target_id}", (int(cx - w/2), int(cy - h/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # --- SPATIAL BLACKLISTING ---
            for i, cid in enumerate(current_ids):
                if cid in target_aliases: continue 
                cand_box = current_boxes[i]
                iou = get_iou(last_known_box, cand_box)
                
                # If a car is on screen and barely overlaps us (IoU < 10%), it is definitively a different vehicle.
                if iou < 0.10:
                    known_other_cars.add(cid)

        # Check C: Target completely dropped. Do we switch?
        elif not track_permanently_lost:
            frames_missing += 1
            
            # Wait 3 frames to debounce
            if frames_missing >= 3:
                found_replacement = False
                
                for i, cid in enumerate(current_ids):
                    if cid in target_aliases: continue 
                    if cid in known_other_cars: continue 
                    if current_classes[i] != target_class_id: continue
                    
                    cand_box = current_boxes[i]
                    iou = get_iou(last_known_box, cand_box)
                    dist = math.hypot(cand_box[0] - last_known_box[0], cand_box[1] - last_known_box[1])
                    
                    # --- STRICT IoU REPLACEMENT LOGIC ---
                    # To steal the ID, the new box MUST overlap the old box significantly (IoU > 40%)
                    # OR, if it's moving very fast and the overlap is smaller, distance must be extremely tight.
                    if iou > 0.80 or (iou > 0.80 and dist < 40):
                        print(f" -> Frame {frame_counter}: STRICT IoU SWITCH DETECTED! ID {current_target_id} -> ID {cid} (Lost for {frames_missing} frames, Overlap: {iou:.2f})")
                        current_target_id = cid
                        target_aliases.add(cid) 
                        last_known_box = cand_box
                        total_switches += 1
                        total_lost_frames += frames_missing
                        frames_missing = 0
                        found_replacement = True
                        break

                if not found_replacement:
                    is_near_edge = last_known_box[0] < EDGE_MARGIN or last_known_box[0] > (width - EDGE_MARGIN) or last_known_box[1] < EDGE_MARGIN or last_known_box[1] > (height - EDGE_MARGIN)
                    
                    if is_near_edge and frames_missing > 20: 
                        print(f" -> Frame {frame_counter}: Target safely exited the camera view.")
                        track_permanently_lost = True
                    elif frames_missing > 300: 
                        print(f" -> Frame {frame_counter}: Track permanently lost.")
                        total_lost_frames += frames_missing 
                        track_permanently_lost = True
                
            if frames_missing > 0 and not track_permanently_lost:
                 cv2.putText(frame, f"TARGET LOST (Missing: {frames_missing})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        out.write(frame)

    if frames_missing > 0 and not track_permanently_lost:
        total_lost_frames += frames_missing

    cap.release()
    out.release()
    
    print("\n" + "="*50)
    print("   SINGLE VEHICLE LINEAGE AUDIT RESULTS   ")
    print("="*50)
    print(f"Original Starting ID : {TARGET_CAR_ID}")
    print(f"Final Ending ID      : {current_target_id if not track_permanently_lost else 'PERMANENTLY LOST'}")
    print(f"All Aliases Owned    : {target_aliases}")
    print("-" * 50)
    print(f"Total True Switches  : {total_switches}")
    print(f"Total Missing Frames : {total_lost_frames}")
    print(f"First Frame Snapshot : {OUTPUT_IMAGE}")
    print("="*50)

if __name__ == "__main__":
    main()