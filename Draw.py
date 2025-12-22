import cv2
import numpy as np

import Constant


def UIDrawing(h, disp, sm_ratio, is_closed, current_seq, decoded_text):
    cv2.putText(disp, f"ratio={sm_ratio:.3f}" if sm_ratio else "ratio=--",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(disp, f"THR={Constant.BLINK_RATIO_THR:.3f}  state={'CLOSED' if is_closed else 'OPEN'}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(disp, f"SEQ: {current_seq}", (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(disp, f"OUT: {decoded_text[-50:]}",
                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(disp, "Controls: C=collect OPEN, V=collect CLOSE, R=reset calib, ESC=quit",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

def draw_points(lm, w, h , disp, idx_list, color):
    for i in idx_list:
        x, y = int(lm[i].x * w), int(lm[i].y * h)
        cv2.circle(disp, (x, y), 2, color, -1)

def iris_mid_drawing(left_iris, right_iris, left_eye, right_eye, lm, w, h, disp):
    for iris_idx, eye_idx in [(left_iris, left_eye), (right_iris, right_eye)]:
        dx, dy = iris_offset(lm, w, h, eye_idx, iris_idx)
        iris_pts = [(lm[i].x * w, lm[i].y * h) for i in iris_idx]
        iris_center = np.mean(iris_pts, axis=0).astype(int)
        cv2.circle(disp, tuple(iris_center), 3, (255, 0, 0), -1)

def iris_offset(lm, w, h, eye_idx, iris_idx):
    def pt(i): return np.array([lm[i].x*w, lm[i].y*h])
    eye_pts = [pt(i) for i in eye_idx]
    iris_pts = [pt(i) for i in iris_idx]
    eye_center = np.mean(eye_pts, axis=0)
    iris_center = np.mean(iris_pts, axis=0)
    ex = max(np.linalg.norm(eye_pts[0]-eye_pts[1]), 1e-6)
    ey = max(np.linalg.norm(eye_pts[2]-eye_pts[3]), 1e-6)
    dx = (iris_center[0] - eye_center[0]) / ex
    dy = (iris_center[1] - eye_center[1]) / ey
    return dx, dy

def do_draw(disp, sm_ratio, is_closed, current_seq, decoded_text, lm, w, h, LEFT_EYE=Constant.LEFT_EYE, RIGHT_EYE=Constant.RIGHT_EYE,
            LEFT_IRIS=Constant.LEFT_IRIS, RIGHT_IRIS=Constant.RIGHT_IRIS):
    UIDrawing(h, disp, sm_ratio, is_closed, current_seq, decoded_text)

    draw_points(lm, w, h, disp, LEFT_EYE, (0, 255, 0))
    draw_points(lm, w, h, disp, RIGHT_EYE, (0, 255, 0))

    draw_points(lm, w, h, disp, LEFT_IRIS, (0, 0, 255))
    draw_points(lm, w, h, disp, RIGHT_IRIS, (0, 0, 255))

    iris_mid_drawing(LEFT_IRIS, RIGHT_IRIS, LEFT_EYE, RIGHT_EYE, lm, w, h, disp)