'''
Author - Ansh Tuli
'''

import cv2
import numpy as np
import json
import math
from ultralytics import YOLO


K = np.array([
    [624.4972715456861, 0.17392899863525, 323.6067680922108],
    [0, 624.7158398780344, 240.24654996439816],
    [0, 0, 1]
])
dist_coeffs = np.array([
    -0.5604540209427666, 0.38344124138432195, 0.001203589161761953,
    -3.605689580952435e-05, -0.1712882535330263
])
cTg = np.array([
    [0.06092409999638326, -0.9981422726956339, -0.0005074422382535526, -4.450705016228094],
    [-0.38356261613354237, -0.022942341970351943, -0.9232298567798314, 1.720216548271677],
    [0.9215031055533254, 0.0564415839865422, -0.38424780292496996, 18.707556274317948],
    [0, 0, 0, 1]
])

R = cTg[:3, :3]
T = cTg[:3, 3]


H = K @ np.column_stack([R[:, 0], R[:, 1], T])
H = H / H[2, 2]
H_inv = np.linalg.inv(H)

h_img, w_img = 480, 640
bottom_pixels = np.array([
    [0, h_img, 1], [w_img, h_img, 1], [w_img, h_img, 1], [0, h_img, 1]
], dtype=np.float64).T

gnd = H_inv @ bottom_pixels
gnd /= gnd[2]

x_min, x_max = gnd[0].min(), gnd[0].max()
y_min, y_max = gnd[1].min(), gnd[1].max()

out_w, out_h = 600, 600
scale = 21

tx = (out_w / 2) + scale * ((x_min + x_max) / 2)
ty = (out_h / 2) - scale * ((y_min + y_max) / 2)

T_offset = np.array([
    [scale, 0, tx],
    [0, scale, ty],
    [0, 0, 1]
], dtype=np.float64)

H_final = T_offset @ H_inv
H_final = H_final / H_final[2, 2]


def transform_bev_coords(x, y, size_w):
    return int((size_w - 1) - y), int((size_w - 1) - x) #For flip and rotate


def image_to_ground_raycasting(u, v, K, cTg):
    K_inv = np.linalg.inv(K)
    gTc = np.linalg.inv(cTg)
    cam_origin = gTc[:3, 3]
    ray_world = gTc[:3, :3] @ (K_inv @ np.array([u, v, 1.0]))

    if ray_world[2] == 0: return None, None
    t = -cam_origin[2] / ray_world[2]
    intersect = cam_origin + t * ray_world
    return intersect[0], intersect[1]



def get_tracks_in_my_camera(gt_json_path, query_timestamp, K, cTg, img_w=640, img_h=480):
    with open(gt_json_path) as f:
        data = json.load(f)

    visible_tracks = []
    zero_dist = np.zeros(4) #array of 0 to solve problem of dist_coeffs

    for track in data["tracks"]:
        timestamps_rounded = [round(t, 1) for t in track["timestamps"]]
        ts_key = round(query_timestamp, 1)

        if ts_key not in timestamps_rounded:
            continue
        idx = timestamps_rounded.index(ts_key)

        x, y, z = track["positions"][idx]
        p_cam = cTg @ np.array([x, y, z, 1.0])

        if p_cam[2] <= 0:
            continue

        rvec, _ = cv2.Rodrigues(cTg[:3, :3])
        pt3d = np.array([[[x, y, z]]], dtype=np.float32)
        pt2d, _ = cv2.projectPoints(pt3d, rvec, cTg[:3, 3], K, zero_dist)
        u, v = pt2d[0][0]

        if 0 <= u <= img_w and 0 <= v <= img_h:
            visible_tracks.append({
                "track_id": track["track_id"],
                "object_type": track["object_type"],
                "position": [x, y, z],
                "pixel": (int(u), int(v)),
            })
    return visible_tracks

#path="/Users/anshtuli/PycharmProjects/SmartTraffic_Vision/datasets/20241126_0001_crossing2_00.7z/crossing2_14_thermal_camera/1732632674003.jpg"
path="/Users/anshtuli/PycharmProjects/SmartTraffic_Vision/datasets/20241126_0001_crossing2_00.7z/crossing2_14_thermal_camera/1732632675834.jpg"

#timestamps= 1732632674.0
timestamps= 1732632675.8


model = YOLO('/Users/anshtuli/PycharmProjects/SmartTraffic_Vision/best_fixed.pt')
cap = cv2.VideoCapture(path)
ret, frame = cap.read()
cap.release()

frame_undist = cv2.undistort(frame, K, dist_coeffs)


bev = cv2.warpPerspective(frame_undist, H_final, (out_w, out_h))
bev_yolo = bev.copy()

# Rotate and flip BEV background
bev = cv2.flip(cv2.rotate(bev, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
bev_yolo = cv2.flip(cv2.rotate(bev_yolo, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)


visible_gt = get_tracks_in_my_camera(
    "/Users/anshtuli/PycharmProjects/SmartTraffic_Vision/datasets/20241126_0001_crossing2_00.json",
    timestamps, K, cTg
)

print("\n--- GROUND TRUTH ---")
for t in visible_gt:
    u, v = t["pixel"]
    gt_x, gt_y, _ = t["position"]
    print(f"GT Object: {t['object_type']} | ID: {t['track_id']} | World Pos: ({gt_x:.2f}, {gt_y:.2f})")

    # Map to BEV
    bev_pt = H_final @ np.array([u, v, 1.0])
    bev_pt /= bev_pt[2]

    cx, cy = transform_bev_coords(bev_pt[0], bev_pt[1], out_w)

    cv2.circle(frame_undist, (u, v), 6, (0, 255, 0), -1)
    cv2.putText(frame_undist, f"GT_{t['track_id']}", (u + 4, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if 0 <= cx < out_w and 0 <= cy < out_h:
        cv2.circle(bev, (cx, cy), 6, (0, 255, 0), -1)
        cv2.putText(bev, f"GT_{t['track_id']}", (cx + 4, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


results = model.track(frame_undist, persist=True, stream=True, verbose=False)
frame_yolo_annotated = frame_undist.copy()

print("\n--- YOLO DETECTIONS ---")
for r in results:
    frame_yolo_annotated = r.plot()

    for box in r.boxes:
        track_id = int(box.id.item()) if box.id is not None else -1
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        px = (x1 + x2) / 2.0
        py = y2


        bev_pt = H_final @ np.array([px, py, 1.0])
        bev_pt /= bev_pt[2]
        bx, by = transform_bev_coords(bev_pt[0], bev_pt[1], out_w)

        #Raycasting
        world_x, world_y = image_to_ground_raycasting(px, py, K, cTg)



        if 0 <= bx < out_w and 0 <= by < out_h:
            cv2.circle(bev_yolo, (bx, by), 5, (0, 0, 255), -1)
            cv2.putText(bev_yolo, f"ID:{track_id}", (bx + 7, by - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            if visible_gt:
                # Find closest GT object
                closest_gt = min(visible_gt,
                                 key=lambda gt: math.hypot(gt["position"][0] - world_x, gt["position"][1] - world_y))
                error = math.hypot(closest_gt["position"][0] - world_x, closest_gt["position"][1] - world_y)
                print(f"YOLO ID: {track_id} | World Pos: ({world_x:.2f}, {world_y:.2f})")
                if error < 2:
                    print(f"   -> Closest to GT {closest_gt['track_id']} | Offset Error: {error:.2f} meters")

cv2.imshow("BEV Yolo Points", bev_yolo)
cv2.imshow("Original Yolo (Undistorted)", frame_yolo_annotated)
cv2.imshow("BEV GT objects", bev)
cv2.imshow("Original GT objects (Undistorted)", frame_undist)
cv2.waitKey(0)