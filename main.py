import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

# ================= 核心參數調校 =================
# 1. 邊緣延伸係數
SCREEN_EXPANSION = 1.25

# 2. 眨眼防下墜門檻
EAR_FREEZE_THRESH = 0.325

# 3. 穩定度門檻
STABILITY_THRESHOLD = 0.006
STABILITY_BUFFER = 15

# 4. 校正採樣數
CALIB_SAMPLES = 50

# ================= 視覺配色 =================
COLOR_BG = (20, 20, 20)
COLOR_TARGET = (0, 0, 255)
COLOR_LOCKED = (0, 255, 0)


# ================= [新增] 卡爾曼濾波穩定器 (Kalman Filter) =================
class KalmanStabilizer:
    def __init__(self):
        # 初始化 OpenCV 的 Kalman Filter
        # 4: 狀態數 (x, y, dx, dy) -> 位置 + 速度
        # 2: 觀測數 (x, y) -> 我們只能看到位置
        self.kf = cv2.KalmanFilter(4, 2)

        # 1. 測量矩陣 (Measurement Matrix) - 觀測值對應到狀態的 x, y
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)

        # 2. 轉移矩陣 (Transition Matrix) - 物理模型：下個位置 = 現在位置 + 速度
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)

        # 3. 雜訊參數 (科展調校重點！)
        # Q (Process Noise): 數值越小，代表系統越平滑 (相信物理模型)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
        # R (Measurement Noise): 數值越大，代表越抗抖動 (不相信攝影機雜訊)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

        self.first_run = True

    def update(self, x, y):
        # 第一次執行時，直接將狀態設為當前位置，避免游標從 (0,0) 飛過來
        if self.first_run:
            self.kf.statePost = np.array([[np.float32(x)], [np.float32(y)], [0], [0]], np.float32)
            self.first_run = False

        # 1. 測量 (Measurement)
        measurement = np.array([[np.float32(x)], [np.float32(y)]])

        # 2. 預測 (Prediction)
        self.kf.predict()

        # 3. 修正 (Correction)
        estimated = self.kf.correct(measurement)

        # 回傳修正後的 x, y
        return float(estimated[0]), float(estimated[1])


# ================= 數學核心 (多項式迴歸) =================
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
        poly = np.array(self.enhance_features(feat))
        return np.dot(poly, self.coeffs_x), np.dot(poly, self.coeffs_y)


# ================= 特徵提取器 & 穩定度 =================
class EyeSystem:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                                         min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.sw, self.sh = pyautogui.size()
        self.buffer = []

        # [修改] 初始化卡爾曼濾波器
        self.kalman = KalmanStabilizer()

    def get_vec(self, lm, right_eye=True):
        if right_eye:
            i, o, p = 362, 263, 473
        else:
            i, o, p = 133, 33, 468
        cx, cy = (lm[i].x + lm[o].x) / 2, (lm[i].y + lm[o].y) / 2
        return lm[p].x - cx, lm[p].y - cy

    def get_ear(self, lm):
        lv = math.hypot(lm[159].x - lm[145].x, lm[159].y - lm[145].y)
        lh = math.hypot(lm[33].x - lm[133].x, lm[33].y - lm[133].y)
        rv = math.hypot(lm[386].x - lm[374].x, lm[386].y - lm[374].y)
        rh = math.hypot(lm[362].x - lm[263].x, lm[362].y - lm[263].y)
        return ((lv / (lh + 1e-6)) + (rv / (rh + 1e-6))) / 2

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks: return None, 0
        lm = res.multi_face_landmarks[0].landmark
        feat = (self.get_vec(lm, False) + self.get_vec(lm, True))
        self.buffer.append(feat)
        if len(self.buffer) > STABILITY_BUFFER: self.buffer.pop(0)
        return feat, self.get_ear(lm)

    def is_stable(self):
        if len(self.buffer) < 2: return False
        return np.mean(np.std(np.array(self.buffer), axis=0)) < STABILITY_THRESHOLD

    def smooth(self, x, y):
        # [修改] 改用卡爾曼濾波進行平滑運算
        kx, ky = self.kalman.update(x, y)
        return kx, ky


# ================= 主程式 =================
def main():
    # 關閉安全停機 (防止碰到邊緣報錯)
    pyautogui.FAILSAFE = False

    cap = cv2.VideoCapture(0)
    eyes = EyeSystem()
    model = PolynomialGazeEstimator()
    W, H = pyautogui.size()

    win_name = "Python Mouse Driver (Kalman Filter Edition)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    calib_pts = []
    # 稍微內縮一點生成校正點，這樣使用者比較好盯
    pad_x, pad_y = W * 0.15, H * 0.15
    for r in np.linspace(pad_y, H - pad_y, 3):
        for c in np.linspace(pad_x, W - pad_x, 3):
            calib_pts.append((int(c), int(r)))

    calib_idx = 0
    calib_buffer = []
    mode = "CALIB"
    print("=== 卡爾曼濾波強化版啟動 ===")
    print(f"邊緣延伸係數: {SCREEN_EXPANSION}")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        feat, ear = eyes.process(frame)

        if mode == "CALIB":
            display = np.zeros((H, W, 3), dtype=np.uint8)
            small = cv2.resize(frame, (320, 240))
            display[H - 240:H, W - 320:W] = small

            if feat:
                pt = calib_pts[calib_idx]
                is_locked = eyes.is_stable() and (ear > EAR_FREEZE_THRESH)
                color = COLOR_LOCKED if is_locked else COLOR_TARGET
                cv2.circle(display, pt, 25 if is_locked else 15, color, -1)
                cv2.circle(display, pt, 45, color, 2)
                cv2.putText(display, f"CALIB {calib_idx + 1}/9", (W // 2 - 100, H // 2 - 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (255, 255, 255), 2)

                if is_locked:
                    calib_buffer.append(feat)
                    prog = len(calib_buffer) / CALIB_SAMPLES
                    cv2.ellipse(display, pt, (55, 55), -90, 0, int(360 * prog), COLOR_LOCKED, 6)
                    if len(calib_buffer) >= CALIB_SAMPLES:
                        for f in calib_buffer: model.add_data(f, pt[0], pt[1])
                        calib_buffer = []
                        calib_idx += 1
                        eyes.buffer = []
                        time.sleep(0.5)
                        if calib_idx >= 9:
                            print("Training...");
                            model.train()
                            mode = "RUN"
                            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(win_name, 320, 240)
                else:
                    if len(calib_buffer) > 0: calib_buffer.pop(0)

            cv2.imshow(win_name, display)
            if cv2.waitKey(1) & 0xFF == 27: break

        elif mode == "RUN":
            cv2.imshow(win_name, cv2.resize(frame, (320, 240)))

            if feat and ear > EAR_FREEZE_THRESH:
                pred = model.predict(feat)
                if pred:
                    # 使用 Kalman Filter 進行平滑
                    raw_sx, raw_sy = eyes.smooth(pred[0], pred[1])

                    # === 邊緣延伸算法 (Edge Boost) ===
                    # 1. 計算相對於螢幕中心的偏移量
                    center_x, center_y = W / 2, H / 2
                    off_x = raw_sx - center_x
                    off_y = raw_sy - center_y

                    # 2. 放大偏移量
                    final_x = center_x + (off_x * SCREEN_EXPANSION)
                    final_y = center_y + (off_y * SCREEN_EXPANSION)

                    # 3. 限制在螢幕範圍內 (避免報錯)
                    final_x = np.clip(final_x, 0, W)
                    final_y = np.clip(final_y, 0, H)

                    try:
                        pyautogui.moveTo(final_x, final_y)
                    except:
                        pass

            if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()