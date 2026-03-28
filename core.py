# core.py
import cv2
import mediapipe as mp
import numpy as np
import math
import time
from config import Config


class DoubleBlinkDetector:
    def __init__(self):
        self.blink_count = 0
        self.last_blink_time = 0
        self.eye_closed = False
        self.blink_start_time = 0

    def update(self, ear):
        current_time = time.time()
        is_blink = False
        if ear < Config.EAR_BLINK_THRESH:
            if not self.eye_closed:
                self.eye_closed = True
                self.blink_start_time = current_time
        else:
            if self.eye_closed:
                self.eye_closed = False
                duration = current_time - self.blink_start_time
                if 0.05 < duration < 0.4:
                    if current_time - self.last_blink_time > Config.DOUBLE_BLINK_WINDOW:
                        self.blink_count = 0
                    self.blink_count += 1
                    self.last_blink_time = current_time
                    if self.blink_count == 2:
                        self.blink_count = 0
                        is_blink = True
        return is_blink


class KalmanStabilizer:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
        self.first_run = True

    def reset(self):
        self.first_run = True

    def update(self, x, y):
        if self.first_run:
            self.kf.statePost = np.array([[np.float32(x)], [np.float32(y)], [0], [0]], np.float32)
            self.first_run = False
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.predict()
        estimated = self.kf.correct(measurement)
        return float(estimated[0]), float(estimated[1])


class PolynomialGazeEstimator:
    def __init__(self):
        self.X_train = []
        self.Y_train = []
        self.coeffs_x = None
        self.coeffs_y = None
        self.is_calibrated = False

    def enhance_features(self, feat):
        lx, ly, rx, ry = feat
        ax, ay = (lx + rx) / 2, (ly + ry) / 2
        return [1, ax, ay, ax ** 2, ay ** 2, ax * ay]

    def add_data(self, feat, sx, sy):
        self.X_train.append(self.enhance_features(feat))
        self.Y_train.append([sx, sy])

    def train(self):
        if not self.X_train: return False
        X = np.array(self.X_train)
        Y = np.array(self.Y_train)
        try:
            self.coeffs_x, _, _, _ = np.linalg.lstsq(X, Y[:, 0], rcond=None)
            self.coeffs_y, _, _, _ = np.linalg.lstsq(X, Y[:, 1], rcond=None)
            self.is_calibrated = True
            return True
        except:
            return False

    def predict(self, feat):
        if not self.is_calibrated: return None
        enhanced_feat = np.array(self.enhance_features(feat))
        return np.dot(enhanced_feat, self.coeffs_x), np.dot(enhanced_feat, self.coeffs_y)


class EyeTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.buffer = []
        self.kalman = KalmanStabilizer()
        self.blink_detector = DoubleBlinkDetector()

    def _get_vec(self, lm, right_eye=True):
        i, o, p = (362, 263, 473) if right_eye else (133, 33, 468)
        cx, cy = (lm[i].x + lm[o].x) / 2, (lm[i].y + lm[o].y) / 2
        return lm[p].x - cx, lm[p].y - cy

    def _get_ear(self, lm):
        lv = math.hypot(lm[159].x - lm[145].x, lm[159].y - lm[145].y)
        lh = math.hypot(lm[33].x - lm[133].x, lm[33].y - lm[133].y)
        rv = math.hypot(lm[386].x - lm[374].x, lm[386].y - lm[374].y)
        rh = math.hypot(lm[362].x - lm[263].x, lm[362].y - lm[263].y)
        return ((lv / (lh + 1e-6)) + (rv / (rh + 1e-6))) / 2

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks: return None, 0, False
        lm = res.multi_face_landmarks[0].landmark
        feat = (self._get_vec(lm, False) + self._get_vec(lm, True))

        self.buffer.append(feat)
        if len(self.buffer) > Config.STABILITY_BUFFER:
            self.buffer.pop(0)

        ear = self._get_ear(lm)
        is_double_blink = self.blink_detector.update(ear)
        return feat, ear, is_double_blink

    def is_stable(self):
        return len(self.buffer) >= 2 and np.mean(np.std(np.array(self.buffer), axis=0)) < Config.STABILITY_THRESHOLD