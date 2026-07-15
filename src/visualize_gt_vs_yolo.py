import os
import cv2

# =========================
# PATHS
# =========================
BASE = "/home/nawab/traffic_project/datasets/UrbanIng-V2X"

GT_PATH = os.path.join(BASE,
                       "yolo_labels",
                       "crossing1_13_thermal_camera")
PRED_PATH = os.path.join("/home/nawab/traffic_project", "yolo_predictions", "predictions", "labels")

IMG_PATH = os.path.join(BASE,
                        "dataset",
                        "20241126_0017_crossing1_00",
                        "crossing1_13_thermal_camera")

OUTPUT_VIS = "/home/nawab/traffic_project/vis_output"
os.makedirs(OUTPUT_VIS, exist_ok=True)


# =========================
# READ YOLO FORMAT
# =========================
def read_boxes(path):
    boxes = []
    if not os.path.exists(path):
        return boxes

    with open(path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 5:
                continue
            _, cx, cy, w, h = map(float, p)
            boxes.append((cx, cy, w, h))
    return boxes


# =========================
# DRAW BOXES
# =========================
def draw_boxes(img, boxes, color):
    h, w = img.shape[:2]

    for (cx, cy, bw, bh) in boxes:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


# =========================
# MAIN VISUAL LOOP
# =========================
gt_files = sorted(os.listdir(GT_PATH))

for f in gt_files[:50]:   # limit for testing

    gt_file = os.path.join(GT_PATH, f)
    pred_file = os.path.join(PRED_PATH, f)

    img_file = os.path.join(IMG_PATH, f.replace(".txt", ".jpg"))

    img = cv2.imread(img_file)
    if img is None:
        continue

    gt_boxes = read_boxes(gt_file)
    pred_boxes = read_boxes(pred_file)

    # GREEN = GT
    draw_boxes(img, gt_boxes, (0, 255, 0))

    # RED = YOLO
    draw_boxes(img, pred_boxes, (0, 0, 255))

    out_path = os.path.join(OUTPUT_VIS, f.replace(".txt", ".jpg"))
    cv2.imwrite(out_path, img)

print("Done. Check folder:", OUTPUT_VIS)

