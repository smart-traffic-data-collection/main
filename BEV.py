'''
Author - Ansh Tuli
'''

import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO('/Users/anshtuli/PycharmProjects/SmartTraffic_Vision/best_fixed.pt')


K = np.array([
    [624.4972715456861,  0.17392899863525,   323.6067680922108],
    [0,                  624.7158398780344,  240.24654996439816],
    [0,                  0,                  1               ]
])
dist_coeffs = np.array([
    -0.5604540209427666,
     0.38344124138432195,
     0.001203589161761953,
    -3.605689580952435e-05,
    -0.1712882535330263
])
cTg = np.array([
    [ 0.06092409999638326, -0.9981422726956339, -0.0005074422382535526, -4.450705016228094],
    [-0.38356261613354237, -0.022942341970351943,-0.9232298567798314,    1.720216548271677],
    [ 0.9215031055533254,   0.0564415839865422,  -0.38424780292496996,  18.707556274317948],
    [0, 0, 0, 1]
])

R = cTg[:3, :3]
T = cTg[:3, 3]


class BEVKalmanTracker:
    def __init__(self, init_x, init_y):
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], dtype=np.float32)

        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], dtype=np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

        self.kf.statePre = np.array([[init_x], [init_y], [0], [0]], dtype=np.float32)
        self.kf.statePost = self.kf.statePre.copy()

    def update(self, meas_x, meas_y):
        self.kf.predict()

        measurement = np.array([[meas_x], [meas_y]], dtype=np.float32)

        self.kf.correct(measurement)
        return int(self.kf.statePost[0]), int(self.kf.statePost[1])

H = K @ np.column_stack([R[:, 0], R[:, 1], T])
H = H / H[2, 2]
H_inv = np.linalg.inv(H)

h_img, w_img = 480, 640
y_crop_start = int(h_img)
bottom_pixels = np.array([
    [0,     y_crop_start, 1],
    [w_img, y_crop_start, 1],
    [w_img, h_img,        1],
    [0,     h_img,        1],
], dtype=np.float64).T

gnd = H_inv @ bottom_pixels
gnd /= gnd[2]

xs, ys = gnd[0], gnd[1]
x_min, x_max = xs.min(), xs.max()
y_min, y_max = ys.min(), ys.max()

out_w, out_h = 600, 600
scale = 21

gx_center = (x_min + x_max) / 2
gy_center = (y_min + y_max) / 2

tx = (out_w / 2) + scale * gx_center
ty = (out_h / 2) - scale * gy_center

T_offset = np.array([
    [scale,  0,    tx],
    [0,      scale, ty],
    [0,      0,     1 ]
], dtype=np.float64)

H_final = T_offset @ H_inv
H_final = H_final / H_final[2, 2]

mapx, mapy = cv2.initUndistortRectifyMap(K, dist_coeffs, None, K, (w_img, h_img), cv2.CV_32FC1)

# --- Video Capture ---
cap = cv2.VideoCapture("/Users/anshtuli/PycharmProjects/SmartTraffic_Vision/datasets/frames_to_videos/crossing2_14_thermal_camera.mp4")

trackers = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue


    frame_undist = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)


    bev = cv2.warpPerspective(frame_undist, H_final, (out_w, out_h))

    results = model.track(frame_undist, persist=True, verbose=False, device="mps")

    frame_annotated = frame_undist.copy()
    for r in results:
        frame_annotated = r.plot()

        if r.boxes is None:
            continue

        for box in r.boxes:
            # 1. FIX: Correctly extract the YOLO track ID
            track_id = int(box.id.item()) if box.id is not None else -1
            if track_id == -1:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Find the bottom-center pixel
            px = (x1 + x2) / 2.0
            py = y2

            # Project raw pixel to raw BEV canvas coordinates
            pt = np.array([px, py, 1.0])
            bev_pt = H_final @ pt
            bev_pt /= bev_pt[2]

            bx, by = int(bev_pt[0]), int(bev_pt[1])

            # 2. FIX: Corrected variable naming & indentation blocks
            if track_id not in trackers:
                trackers[track_id] = BEVKalmanTracker(bx, by)
                smooth_bx, smooth_by = bx, by
            else:
                smooth_bx, smooth_by = trackers[track_id].update(bx, by)


            # if 0 <= smooth_bx < out_w and 0 <= smooth_by < out_h:
            #     cv2.circle(bev, (smooth_bx, smooth_by), 5, (0, 0, 255), -1)



            # Draw the RAW YOLO point in faint Red
            if 0 <= bx < out_w and 0 <= by < out_h:
                cv2.circle(bev, (bx, by), 4, (0, 0, 100), -1)


            if 0 <= smooth_bx < out_w and 0 <= smooth_by < out_h:
                cv2.circle(bev, (smooth_bx, smooth_by), 4, (0, 255, 0), -1)


            if 0 <= bx < out_w and 0 <= by < out_h and 0 <= smooth_bx < out_w and 0 <= smooth_by < out_h:
                cv2.line(bev, (bx, by), (smooth_bx, smooth_by), (255, 255, 255), 1)

    bev = cv2.rotate(bev, cv2.ROTATE_90_COUNTERCLOCKWISE)
    bev = cv2.flip(bev, 1)

    cv2.imshow("BEV Video", bev)
    cv2.imshow("Original Video", frame_annotated)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
