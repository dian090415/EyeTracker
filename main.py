import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
from PIL import ImageFont, ImageDraw, Image


# ================= 參數設定 =================
class constant:
    SCREEN_EXPANSION = 1.2
    EAR_THRESH = 0.28
    STABILITY_THRESHOLD = 0.006
    STABILITY_BUFFER = 15
    CALIB_SAMPLES = 40
    DWELL_TIME = 1.0  # 注視多久後彈出鍵盤

    COLOR_BG = (30, 30, 30)
    COLOR_KEY_NORMAL = (50, 50, 50)
    COLOR_KEY_HOVER = (100, 100, 200)
    COLOR_KEY_ACTIVE = (0, 255, 0)

    # [新增] 輸入框顏色
    COLOR_INPUT_BG = (255, 255, 255)
    COLOR_INPUT_TEXT = (0, 0, 0)


# ================= 卡爾曼濾波 (維持不變) =================
class KalmanStabilizer:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
        self.first_run = True

    def update(self, x, y):
        if self.first_run:
            self.kf.statePost = np.array([[np.float32(x)], [np.float32(y)], [0], [0]], np.float32)
            self.first_run = False
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.predict()
        estimated = self.kf.correct(measurement)
        return float(estimated[0]), float(estimated[1])


# ================= EyeSystem (移除嘴巴，只留眼睛) =================
class EyeSystem:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.buffer = []
        self.kalman = KalmanStabilizer()

    def get_vec(self, lm, right_eye=True):
        i, o, p = (362, 263, 473) if right_eye else (133, 33, 468)
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
        if len(self.buffer) > constant.STABILITY_BUFFER: self.buffer.pop(0)
        return feat, self.get_ear(lm)

    def is_stable(self):
        if len(self.buffer) < 2: return False
        return np.mean(np.std(np.array(self.buffer), axis=0)) < constant.STABILITY_THRESHOLD

    def smooth(self, x, y):
        return self.kalman.update(x, y)


# ================= 數學模型 =================
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


# ================= 虛擬鍵盤 (微調關閉邏輯) =================
class VirtualKeyboard:
    def __init__(self, w, h):
        self.W, self.H = w, h
        self.keys = [
            "ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏ",
            "ㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙ",
            "ㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠ",
            "ㄡㄢㄣㄤㄥㄦˊˇˋ˙SPACE",
            "送出"  # 改成送出，更像通訊軟體
        ]
        self.buttons = []
        self.font = None
        self.setup_layout()
        self.hover_key = None
        self.hover_start_time = 0
        self.input_buffer = ""  # 暫存輸入的文字

        try:
            self.font = ImageFont.truetype("msjh.ttc", 24)
        except:
            self.font = ImageFont.load_default()

    def setup_layout(self):
        rows = len(self.keys)
        margin_top = int(self.H * 0.4)
        btn_h = (self.H - margin_top) // rows
        for r, line in enumerate(self.keys):
            if line == "送出":
                x1, y1 = 0, margin_top + r * btn_h
                x2, y2 = self.W, y1 + btn_h
                self.buttons.append({'char': 'SEND', 'rect': (x1, y1, x2, y2), 'label': '送 出 訊 息'})
                continue
            cols = len(line)
            if "SPACE" in line: line = line.replace("SPACE", " "); cols = len(line)
            btn_w = self.W // cols
            for c, char in enumerate(line):
                x1 = c * btn_w;
                y1 = margin_top + r * btn_h
                self.buttons.append({'char': char, 'rect': (x1, y1, x1 + btn_w, y1 + btn_h), 'label': char})

    def update(self, gaze_x, gaze_y, frame):
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        should_close = False
        final_text = None

        # 顯示目前已輸入的暫存文字 (在鍵盤上方)
        draw.rectangle([0, int(self.H * 0.4) - 50, self.W, int(self.H * 0.4)], fill=(255, 255, 255))
        draw.text((20, int(self.H * 0.4) - 40), f"輸入中: {self.input_buffer}", font=self.font, fill=(0, 0, 0))

        for btn in self.buttons:
            x1, y1, x2, y2 = btn['rect']
            char = btn['char']
            label = btn['label']
            is_hover = (x1 < gaze_x < x2) and (y1 < gaze_y < y2)
            if is_hover:
                if self.hover_key != char:
                    self.hover_key = char;
                    self.hover_start_time = time.time()
                elapsed = time.time() - self.hover_start_time
                progress = min(elapsed / constant.DWELL_TIME, 1.0)
                draw.rectangle([x1, y2 - 5, x1 + (x2 - x1) * progress, y2], fill=constant.COLOR_KEY_ACTIVE)

                if elapsed >= constant.DWELL_TIME:
                    if char == 'SEND':
                        should_close = True
                        final_text = self.input_buffer
                        self.input_buffer = ""  # 清空暫存
                    elif char == 'SPACE':
                        self.input_buffer += " "
                    else:
                        self.input_buffer += char

                    self.hover_key = None
            else:
                if self.hover_key == char: self.hover_key = None

            # 繪製按鈕
            border_color = (0, 0, 255) if char == 'SEND' else (200, 200, 200)
            draw.rectangle([x1, y1, x2, y2], outline=border_color, width=2)

            # 文字置中
            bbox = draw.textbbox((0, 0), label, font=self.font)
            tx = x1 + (x2 - x1 - (bbox[2] - bbox[0])) // 2
            ty = y1 + (y2 - y1 - (bbox[3] - bbox[1])) // 2
            text_color = (255, 100, 100) if char == 'SEND' else (255, 255, 255)
            draw.text((tx, ty), label, font=self.font, fill=text_color)

        draw.ellipse((gaze_x - 5, gaze_y - 5, gaze_x + 5, gaze_y + 5), fill=(0, 255, 255))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR), should_close, final_text


# ================= [核心] 模擬手機輸入框 =================
class InputField:
    def __init__(self, w, h):
        self.W, self.H = w, h
        # 輸入框位置：螢幕底部上方一點點
        self.rect = (100, h - 150, w - 100, h - 80)
        self.hover_start = 0
        self.is_hovering = False
        self.chat_history = ["系統: 歡迎使用眼控聊天室！", "系統: 請注視下方輸入框開始打字。"]  # 模擬聊天紀錄

    def update(self, gaze_x, gaze_y, frame, is_keyboard_open):
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 1. 繪製聊天紀錄 (不管有沒有開鍵盤都要顯示)
        try:
            font = ImageFont.truetype("msjh.ttc", 20)
        except:
            font = ImageFont.load_default()

        y_offset = 50
        for msg in self.chat_history[-8:]:  # 只顯示最後 8 行
            draw.text((50, y_offset), msg, font=font, fill=(200, 200, 200))
            y_offset += 40

        # 如果鍵盤已經打開，就不用畫輸入框了 (因為被鍵盤擋住，或者輸入邏輯轉移給鍵盤)
        should_open_keyboard = False

        if not is_keyboard_open:
            x1, y1, x2, y2 = self.rect

            # 碰撞偵測
            if x1 < gaze_x < x2 and y1 < gaze_y < y2:
                if not self.is_hovering:
                    self.is_hovering = True
                    self.hover_start = time.time()

                elapsed = time.time() - self.hover_start
                progress = min(elapsed / constant.DWELL_TIME, 1.0)

                # 視覺回饋：外框變色 + 進度條
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
                draw.rectangle([x1, y2 - 5, x1 + (x2 - x1) * progress, y2], fill=(0, 255, 0))

                if elapsed >= constant.DWELL_TIME:
                    should_open_keyboard = True
                    self.is_hovering = False
            else:
                self.is_hovering = False
                draw.rectangle([x1, y1, x2, y2], outline=(150, 150, 150), width=2)
                draw.rectangle([x1, y1, x2, y2], fill=(50, 50, 50))

            # 繪製 "點擊這裡輸入訊息" 的文字
            draw.text((x1 + 20, y1 + 20), "注視此處輸入訊息...", font=font, fill=(200, 200, 200))

        # 繪製游標
        draw.ellipse((gaze_x - 5, gaze_y - 5, gaze_x + 5, gaze_y + 5), fill=(0, 255, 255))

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR), should_open_keyboard

    def add_message(self, text):
        if text:
            self.chat_history.append(f"我: {text}")


# ================= 主程式 =================
def main():
    pyautogui.FAILSAFE = False
    cap = cv2.VideoCapture(0)
    eyes = EyeSystem()
    model = PolynomialGazeEstimator()
    W, H = pyautogui.size()

    keyboard = VirtualKeyboard(W, H)
    input_field = InputField(W, H)

    win_name = "Eye Chat Demo"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    calib_pts = []
    pad_x, pad_y = W * 0.15, H * 0.15
    for r in np.linspace(pad_y, H - pad_y, 3):
        for c in np.linspace(pad_x, W - pad_x, 3):
            calib_pts.append((int(c), int(r)))

    calib_idx = 0
    calib_buffer = []
    is_calibrated = False

    # [狀態] 是否顯示鍵盤
    show_keyboard = False

    print("=== 系統啟動 ===")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        feat, ear = eyes.process(frame)

        # 建立黑色背景畫布 (模擬軟體介面)
        display = np.zeros((H, W, 3), dtype=np.uint8)

        # 右上角顯示攝影機小視窗
        cam_small = cv2.resize(frame, (320, 240))
        display[0:240, W - 320:W] = cam_small

        if not is_calibrated:
            # === 校正模式 ===
            if feat:
                pt = calib_pts[calib_idx]
                is_locked = eyes.is_stable() and (ear > constant.EAR_THRESH)
                color = constant.COLOR_LOCKED if is_locked else constant.COLOR_TARGET
                cv2.circle(display, pt, 20, color, -1)

                if is_locked:
                    calib_buffer.append(feat)
                    prog = len(calib_buffer) / constant.CALIB_SAMPLES
                    cv2.ellipse(display, pt, (50, 50), -90, 0, int(360 * prog), constant.COLOR_LOCKED, 5)
                    if len(calib_buffer) >= constant.CALIB_SAMPLES:
                        for f in calib_buffer: model.add_data(f, pt[0], pt[1])
                        calib_buffer = []
                        calib_idx += 1
                        eyes.buffer = []
                        time.sleep(0.5)
                        if calib_idx >= 9:
                            model.train()
                            is_calibrated = True
                else:
                    if len(calib_buffer) > 0: calib_buffer.pop(0)
        else:
            # === 運作模式 (模擬聊天軟體) ===
            if feat and ear > constant.EAR_THRESH:
                pred = model.predict(feat)
                if pred:
                    kx, ky = eyes.smooth(pred[0], pred[1])

                    if show_keyboard:
                        # [狀態 A: 鍵盤輸入模式]
                        # 顯示鍵盤，並接收輸入結果
                        display, close_req, text_result = keyboard.update(kx, ky, display)

                        if close_req:
                            # 使用者按下 "送出"
                            input_field.add_message(text_result)  # 加到聊天紀錄
                            show_keyboard = False  # 收起鍵盤
                            print(f"訊息已送出: {text_result}")

                    else:
                        # [狀態 B: 瀏覽模式]
                        # 顯示聊天紀錄 + 底部輸入框
                        # 這裡的 cursor 控制權交給 input_field
                        display, open_req = input_field.update(kx, ky, display, False)

                        if open_req:
                            # 使用者注視輸入框 -> 彈出鍵盤
                            show_keyboard = True
                            print("開啟鍵盤")

        cv2.imshow(win_name, display)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()