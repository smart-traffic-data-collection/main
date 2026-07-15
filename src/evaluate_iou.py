import os
import numpy as np

# =========================
# PATHS
# =========================
BASE = "/home/nawab/traffic_project/datasets/UrbanIng-V2X"

GT_PATH = os.path.join(BASE,
                       "yolo_labels",
                       "crossing1_13_thermal_camera")
PRED_PATH = os.path.join("/home/nawab/traffic_project", "yolo_predictions", "predictions", "labels")

IOU_THRESHOLD = 0.3


# =========================
# READ YOLO BOXES
# =========================
def read_boxes(file_path):
    boxes = []

    if not os.path.exists(file_path):
        return boxes

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            _, cx, cy, w, h = map(float, parts)
            boxes.append((cx, cy, w, h))

    return boxes


# =========================
# IOU FUNCTION
# =========================
def iou(b1, b2):
    x1_min = b1[0] - b1[2] / 2
    y1_min = b1[1] - b1[3] / 2
    x1_max = b1[0] + b1[2] / 2
    y1_max = b1[1] + b1[3] / 2

    x2_min = b2[0] - b2[2] / 2
    y2_min = b2[1] - b2[3] / 2
    x2_max = b2[0] + b2[2] / 2
    y2_max = b2[1] + b2[3] / 2

    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    area1 = b1[2] * b1[3]
    area2 = b2[2] * b2[3]

    union = area1 + area2 - inter_area

    return inter_area / (union + 1e-6)


# =========================
# MATCH FILES BY TIMESTAMP
# =========================
def extract_ts(filename):
    return int(filename.replace(".txt", ""))


gt_files = sorted(os.listdir(GT_PATH))
pred_files = sorted(os.listdir(PRED_PATH))

gt_map = {extract_ts(f): f for f in gt_files}
pred_map = {extract_ts(f): f for f in pred_files}

pred_used = set()

TP, FP, FN = 0, 0, 0


# =========================
# MAIN LOOP (NEAREST MATCHING)
# =========================
for gt_ts, gt_file in gt_map.items():

    gt_boxes = read_boxes(os.path.join(GT_PATH, gt_file))

    # find nearest prediction frame
    closest_pred_ts = min(pred_map.keys(), key=lambda x: abs(x - gt_ts))
    pred_file = pred_map[closest_pred_ts]

    pred_boxes = read_boxes(os.path.join(PRED_PATH, pred_file))

    matched_pred = set()

    for gt_box in gt_boxes:

        best_iou = 0
        best_j = -1

        for j, pred_box in enumerate(pred_boxes):

            if j in matched_pred:
                continue

            score = iou(gt_box, pred_box)

            if score > best_iou:
                best_iou = score
                best_j = j

        if best_iou >= IOU_THRESHOLD:
            TP += 1
            matched_pred.add(best_j)
        else:
            FN += 1

    FP += (len(pred_boxes) - len(matched_pred))


# =========================
# FINAL METRICS
# =========================
precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

print("\n===== FINAL RESULTS =====")


if len(gt_boxes) > 0:
    print("GT sample:", gt_boxes[0])
else:
    print("GT sample: EMPTY")

if len(pred_boxes) > 0:
    print("PRED sample:", pred_boxes[0])
else:
    print("PRED sample: EMPTY")

print("TP:", TP)
print("FP:", FP)
print("FN:", FN)
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))