import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
import json
import os
import ctypes
from ctypes import wintypes
from PIL import ImageFont, ImageDraw, Image
from collections import defaultdict


# ================= 參數設定 =================
class constant:
    # 螢幕游標移動放大倍率
    SCREEN_EXPANSION = 1.4

    # 眼睛判定參數
    EAR_THRESH = 0.28  # 張眼判定 (數值越大越難觸發，請依個人眼睛調整)
    EAR_BLINK_THRESH = 0.22  # 閉眼判定

    # 游標穩定參數 (Kalman 濾波相關)
    STABILITY_THRESHOLD = 0.008
    STABILITY_BUFFER = 10

    # 校正參數
    CALIB_SAMPLES = 40

    # 互動時間參數
    DWELL_TIME = 0.8  # 注視按鈕多久觸發 (稍微調長一點避免誤觸)
    DOUBLE_BLINK_WINDOW = 0.6

    # 介面顏色 (高對比風格)
    COLOR_BG_OPAQUE = (0, 0, 0)  # 全黑背景
    COLOR_BORDER = (255, 255, 255)  # 白色邊框
    COLOR_TEXT = (255, 255, 255)  # 白色文字


# ================= 智慧輸入法引擎 (Smart IME) =================
class SmartIME:
    def __init__(self, filename="user_habits.json"):
        self.filename = filename
        self.model = defaultdict(lambda: defaultdict(int))
        self.load_habits()

    def load_habits(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        for next_char, count in v.items():
                            self.model[k][next_char] = count
            except:
                pass

    def save_habits(self):
        data = {k: dict(v) for k, v in self.model.items()}
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def learn(self, text):
        if len(text) < 2: return
        for i in range(len(text) - 1):
            self.model[text[i]][text[i + 1]] += 1
        self.save_habits()

    def predict(self, current_char):
        if current_char not in self.model: return []
        candidates = sorted(self.model[current_char].items(), key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:6]]


# ================= Windows 游標形狀偵測 =================
class CURSORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD), ("hCursor", wintypes.HANDLE),
                ("ptScreenPos", wintypes.POINT)]


def is_text_cursor():
    try:
        user32 = ctypes.windll.user32
        h_ibeam = user32.LoadCursorW(0, 32513)
        cursor_info = CURSORINFO()
        cursor_info.cbSize = ctypes.sizeof(CURSORINFO)
        user32.GetCursorInfo(ctypes.byref(cursor_info))
        return cursor_info.hCursor == h_ibeam
    except:
        return False


# ================= 眼動追蹤系統核心 =================
class DoubleBlinkDetector:
    def __init__(self):
        self.blink_count = 0
        self.last_blink_time = 0
        self.eye_closed = False
        self.blink_start_time = 0

    def update(self, ear):
        current_time = time.time()
        is_blink = False
        if ear < constant.EAR_BLINK_THRESH:
            if not self.eye_closed:
                self.eye_closed = True
                self.blink_start_time = current_time
        else:
            if self.eye_closed:
                self.eye_closed = False
                duration = current_time - self.blink_start_time
                if 0.05 < duration < 0.4:
                    if current_time - self.last_blink_time > constant.DOUBLE_BLINK_WINDOW:
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

    def update(self, x, y):
        if self.first_run:
            self.kf.statePost = np.array([[np.float32(x)], [np.float32(y)], [0], [0]], np.float32)
            self.first_run = False
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.predict()
        estimated = self.kf.correct(measurement)
        return float(estimated[0]), float(estimated[1])


class EyeSystem:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.buffer = []
        self.kalman = KalmanStabilizer()
        self.blink_detector = DoubleBlinkDetector()

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
        if not res.multi_face_landmarks: return None, 0, False
        lm = res.multi_face_landmarks[0].landmark
        feat = (self.get_vec(lm, False) + self.get_vec(lm, True))
        self.buffer.append(feat)
        if len(self.buffer) > constant.STABILITY_BUFFER: self.buffer.pop(0)
        return feat, self.get_ear(lm), self.blink_detector.update(self.get_ear(lm))

    def is_stable(self):
        return len(self.buffer) >= 2 and np.mean(np.std(np.array(self.buffer), axis=0)) < constant.STABILITY_THRESHOLD

    def smooth(self, x, y):
        return self.kalman.update(x, y)


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
        return np.dot(np.array(self.enhance_features(feat)), self.coeffs_x), \
            np.dot(np.array(self.enhance_features(feat)), self.coeffs_y)


# ================= 手機式半版虛擬鍵盤 (修復版) =================
class VirtualKeyboard:
    def __init__(self, w, h):
        self.W, self.H = w, h
        self.ime = SmartIME()
        self.layout_rows = [
            "ㄅㄉˇˋㄓˊ˙ㄚㄞㄢㄦ",
            "ㄆㄊㄍㄐㄔㄗㄧㄛㄟㄣ",
            "ㄇㄋㄎㄑㄕㄘㄨㄜㄠㄤ",
            "ㄈㄌㄏㄒㄖㄙㄩㄝㄡㄥ",
            "SPACE,BACKSPACE,ENTER,MINIMIZE"
        ]
        self.buttons = []
        self.predict_buttons = []
        self.font = None
        self.hover_key = None
        self.hover_start_time = 0
        self.input_buffer = ""
        self.keyboard_height_ratio = 0.5

        try:
            # 嘗試載入微軟正黑體，讓中文顯示更漂亮
            self.font = ImageFont.truetype("msjh.ttc", 36)
            self.font_large = ImageFont.truetype("msjh.ttc", 48)
        except:
            self.font = ImageFont.load_default()
            self.font_large = ImageFont.load_default()

        self.setup_layout()
        self.update_predictions()

    def setup_layout(self):
        self.buttons = []
        start_y = int(self.H * (1 - self.keyboard_height_ratio))
        area_h = self.H - start_y
        prediction_bar_h = 70
        keys_area_h = area_h - prediction_bar_h
        keys_start_y = start_y + prediction_bar_h

        row_count = len(self.layout_rows)
        btn_h = keys_area_h // row_count

        for r, line in enumerate(self.layout_rows):
            keys = []
            if "," in line:
                keys = line.split(",")
            else:
                keys = list(line)
            col_count = len(keys)
            btn_w = self.W // col_count
            for c, char in enumerate(keys):
                x1 = c * btn_w
                y1 = keys_start_y + r * btn_h
                x2 = x1 + btn_w
                y2 = y1 + btn_h
                label = char
                if char == 'MINIMIZE': label = '縮小'
                if char == 'BACKSPACE': label = '←'
                if char == 'ENTER': label = '送出'
                if char == 'SPACE': label = '空白'
                self.buttons.append({'val': char, 'rect': (x1, y1, x2, y2), 'label': label, 'is_predict': False})

    def update_predictions(self):
        self.predict_buttons = []
        last_char = self.input_buffer[-1] if self.input_buffer else ""
        predictions = self.ime.predict(last_char)
        start_y = int(self.H * (1 - self.keyboard_height_ratio))
        bar_h = 70
        if not predictions: predictions = ["，", "。", "！", "？", "：", "......"]
        count = len(predictions)
        btn_w = self.W // count
        for i, char in enumerate(predictions):
            x1 = i * btn_w
            y1 = start_y
            x2 = x1 + btn_w
            y2 = start_y + bar_h
            self.predict_buttons.append({'val': char, 'rect': (x1, y1, x2, y2), 'label': char, 'is_predict': True})

    def update(self, gaze_x, gaze_y, frame):
        # ★★★ 關鍵修復：強制將攝影機畫面放大到全螢幕解析度 ★★★
        # 這樣才能確保畫布夠大，鍵盤不會畫到畫面外面去
        frame = cv2.resize(frame, (self.W, self.H))

        # 轉換為 PIL 圖片以便繪製中文
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        start_y = int(self.H * (1 - self.keyboard_height_ratio))

        # --- 1. 繪製全黑不透明背景 (鍵盤區) ---
        draw.rectangle([0, start_y, self.W, self.H], fill='black')

        # --- 2. 繪製頂部文字輸入框 (全黑底、白字) ---
        text_box_h = 100
        draw.rectangle([0, 0, self.W, text_box_h], fill='black')
        draw.rectangle([0, 0, self.W, text_box_h], outline='white', width=3)
        draw.text((30, 20), f"輸入: {self.input_buffer}", font=self.font_large, fill='white')

        should_close = False
        all_buttons = self.predict_buttons + self.buttons

        for btn in all_buttons:
            x1, y1, x2, y2 = btn['rect']
            val = btn['val']
            label = btn['label']
            is_predict = btn.get('is_predict', False)

            # 判斷眼睛注視
            is_hover = (x1 < gaze_x < x2) and (y1 < gaze_y < y2)

            # --- 3. 按鈕繪製邏輯 (黑底白框) ---
            btn_color = 'black'
            text_color = 'white'
            border_color = 'white'

            if is_predict:
                btn_color = (0, 40, 0)  # 預測區稍微深綠一點區分

            # 懸停效果
            if is_hover:
                if self.hover_key != val:
                    self.hover_key = val
                    self.hover_start_time = time.time()

                elapsed = time.time() - self.hover_start_time
                progress = min(elapsed / constant.DWELL_TIME, 1.0)

                # 注視中：灰色填充
                btn_color = (80, 80, 80)

                # 繪製進度條效果 (綠色填滿)
                fill_width = x1 + (x2 - x1) * progress
                draw.rectangle([x1, y1, x2, y2], fill=btn_color)  # 先畫灰底
                draw.rectangle([x1, y1, fill_width, y2], fill=(0, 200, 0))  # 再畫進度

                # 觸發確認
                if elapsed >= constant.DWELL_TIME:
                    self.hover_key = None
                    # 閃爍特效
                    draw.rectangle([x1, y1, x2, y2], fill='green')

                    if val == 'MINIMIZE':
                        should_close = True
                    elif val == 'BACKSPACE':
                        self.input_buffer = self.input_buffer[:-1]
                        pyautogui.press('backspace')
                        self.update_predictions()
                    elif val == 'ENTER':
                        self.ime.learn(self.input_buffer)
                        self.input_buffer = ""
                        pyautogui.press('enter')
                        self.update_predictions()
                    elif val == 'SPACE':
                        self.input_buffer += " "
                        pyautogui.press('space')
                    else:
                        self.input_buffer += val
                        self.type_key(val)
                        self.update_predictions()
            else:
                if self.hover_key == val: self.hover_key = None
                draw.rectangle([x1, y1, x2, y2], fill=btn_color)

            # 畫邊框 (確保每個按鈕都分明)
            draw.rectangle([x1, y1, x2, y2], outline=border_color, width=2)

            # 畫文字 (置中)
            bbox = draw.textbbox((0, 0), label, font=self.font)
            tx = x1 + (x2 - x1 - (bbox[2] - bbox[0])) // 2
            ty = y1 + (y2 - y1 - (bbox[3] - bbox[1])) // 2
            draw.text((tx, ty), label, font=self.font, fill=text_color)

        # --- 4. 畫出眼睛注視點 (綠色圓點 = 矯正後的座標) ---
        r = 15
        draw.ellipse((gaze_x - r, gaze_y - r, gaze_x + r, gaze_y + r), fill=(0, 255, 0), outline='white')

        return cv2.cvtColor(np.array(img_pil.convert('RGB')), cv2.COLOR_RGB2BGR), should_close

    def type_key(self, char):
        try:
            import pyperclip
            pyperclip.copy(char)
            pyautogui.hotkey('ctrl', 'v')
        except:
            pass


# ================= 主程式 =================
def main():
    pyautogui.FAILSAFE = False
    cap = cv2.VideoCapture(0)
    eyes = EyeSystem()
    model = PolynomialGazeEstimator()
    W, H = pyautogui.size()
    keyboard = VirtualKeyboard(W, H)

    win_name = "AI Eye Tracker"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # 初始全螢幕校正
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    calib_pts = []
    pad_x, pad_y = W * 0.15, H * 0.15
    for r in np.linspace(pad_y, H - pad_y, 3):
        for c in np.linspace(pad_x, W - pad_x, 3): calib_pts.append((int(c), int(r)))

    calib_idx = 0
    calib_buffer = []
    current_mode = 'CALIB'

    # 小視窗按鈕位置
    win_pos_x, win_pos_y = 50, 50
    btn_local_x1, btn_local_y1 = 110, 10
    btn_local_x2, btn_local_y2 = 210, 50

    print("=== 系統啟動：請注視螢幕上的校正點 ===")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        # 取得眼部特徵與矯正數據
        feat, ear, is_double_blink = eyes.process(frame)

        display = np.zeros((H, W, 3), dtype=np.uint8)

        if current_mode == 'CALIB':
            # 校正模式 (全螢幕黑底)
            small_cam = cv2.resize(frame, (320, 240))
            display[H - 240:H, W - 320:W] = small_cam
            if feat:
                pt = calib_pts[calib_idx]
                is_locked = eyes.is_stable() and (ear > constant.EAR_THRESH)
                color = (0, 255, 0) if is_locked else (0, 0, 255)
                cv2.circle(display, pt, 25, color, -1)
                cv2.circle(display, pt, 50, color, 2)
                if is_locked:
                    calib_buffer.append(feat)
                    prog = len(calib_buffer) / constant.CALIB_SAMPLES
                    cv2.ellipse(display, pt, (60, 60), -90, 0, int(360 * prog), (0, 255, 0), 5)
                    if len(calib_buffer) >= constant.CALIB_SAMPLES:
                        for f in calib_buffer: model.add_data(f, pt[0], pt[1])
                        calib_buffer = []
                        calib_idx += 1
                        eyes.buffer = []
                        time.sleep(0.5)
                        if calib_idx >= 9:
                            model.train()
                            current_mode = 'DESKTOP'
                            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(win_name, 320, 240)
                            cv2.moveWindow(win_name, win_pos_x, win_pos_y)
                else:
                    if len(calib_buffer) > 0: calib_buffer.pop(0)
            cv2.imshow(win_name, display)

        elif current_mode == 'DESKTOP':
            # 桌面模式 (小視窗)
            display_small = cv2.resize(frame, (320, 240))
            cv2.rectangle(display_small, (btn_local_x1, btn_local_y1), (btn_local_x2, btn_local_y2), (0, 0, 0), -1)
            cv2.rectangle(display_small, (btn_local_x1, btn_local_y1), (btn_local_x2, btn_local_y2), (255, 255, 255), 2)
            cv2.putText(display_small, "Keyboard", (125, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if feat:
                pred = model.predict(feat)
                if pred:
                    kx, ky = eyes.smooth(pred[0], pred[1])

                    center_x, center_y = W / 2, H / 2
                    off_x, off_y = kx - center_x, ky - center_y
                    final_x = np.clip(center_x + off_x * constant.SCREEN_EXPANSION, 0, W)
                    final_y = np.clip(center_y + off_y * constant.SCREEN_EXPANSION, 0, H)

                    try:
                        if ear > constant.EAR_THRESH: pyautogui.moveTo(final_x, final_y)

                        if is_double_blink:
                            pyautogui.click()
                            cv2.putText(display_small, "Input Mode...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        (0, 255, 0), 2)
                            cv2.imshow(win_name, display_small)
                            cv2.waitKey(1)

                            found_cursor = False
                            for _ in range(5):
                                time.sleep(0.1)
                                if is_text_cursor(): found_cursor = True; break

                            if found_cursor:
                                current_mode = 'KEYBOARD'
                                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                keyboard.input_buffer = ""
                                keyboard.update_predictions()

                        abs_btn_x1 = win_pos_x + btn_local_x1
                        abs_btn_x2 = win_pos_x + btn_local_x2
                        abs_btn_y1 = win_pos_y + btn_local_y1
                        abs_btn_y2 = win_pos_y + btn_local_y2

                        if abs_btn_x1 < final_x < abs_btn_x2 and abs_btn_y1 < final_y < abs_btn_y2:
                            cv2.rectangle(display_small, (btn_local_x1, btn_local_y1), (btn_local_x2, btn_local_y2),
                                          (0, 255, 0), -1)
                            current_mode = 'KEYBOARD'
                            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                            keyboard.input_buffer = ""
                            keyboard.update_predictions()
                    except:
                        pass
            cv2.imshow(win_name, display_small)

        elif current_mode == 'KEYBOARD':
            # 鍵盤模式
            if feat and ear > constant.EAR_THRESH:
                pred = model.predict(feat)
                if pred:
                    kx, ky = eyes.smooth(pred[0], pred[1])
                    # 因為 update 裡面做了 resize，所以回傳的圖片就是全螢幕尺寸
                    display_kb, close_req = keyboard.update(kx, ky, frame)

                    if close_req:
                        current_mode = 'DESKTOP'
                        eyes.kalman = KalmanStabilizer()
                        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(win_name, 320, 240)
                        cv2.moveWindow(win_name, win_pos_x, win_pos_y)

                    cv2.imshow(win_name, display_kb)
            else:
                # 沒偵測到眼睛時，顯示定格畫面
                cv2.imshow(win_name, keyboard.update(0, 0, frame)[0])

        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__": main()