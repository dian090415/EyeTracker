# main.py
import cv2
import pyautogui
import numpy as np
import time

# 匯入我們自己寫的模組
from config import Config
from utils import WindowsHelper
from core import EyeTracker, PolynomialGazeEstimator
from keyboard import VirtualKeyboard


class EyeTrackingApp:
    def __init__(self):
        pyautogui.FAILSAFE = False
        self.cap = cv2.VideoCapture(0)
        self.tracker = EyeTracker()
        self.model = PolynomialGazeEstimator()

        self.W, self.H = pyautogui.size()
        self.keyboard = VirtualKeyboard(self.W, self.H)

        self.win_name = "AI Eye Tracker"
        self.mode = 'CALIB'
        self.win_pos = (50, 50)

        self._init_calibration_points()
        self._setup_window()

    def _init_calibration_points(self):
        self.calib_pts = []
        pad_x, pad_y = self.W * 0.15, self.H * 0.15
        for r in np.linspace(pad_y, self.H - pad_y, 3):
            for c in np.linspace(pad_x, self.W - pad_x, 3):
                self.calib_pts.append((int(c), int(r)))
        self.calib_idx = 0
        self.calib_buffer = []

    def _setup_window(self):
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        self._set_fullscreen(True)

    def _set_fullscreen(self, is_full):
        if is_full:
            cv2.setWindowProperty(self.win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(self.win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.win_name, 320, 240)
            cv2.moveWindow(self.win_name, self.win_pos[0], self.win_pos[1])

    def run(self):
        print("=== 系統啟動：請注視螢幕上的校正點 ===")
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            feat, ear, is_double_blink = self.tracker.process_frame(frame)
            display = np.zeros((self.H, self.W, 3), dtype=np.uint8)

            if self.mode == 'CALIB':
                self._handle_calibration(frame, feat, ear, display)
            elif self.mode == 'DESKTOP':
                self._handle_desktop(frame, feat, ear, is_double_blink)
            elif self.mode == 'KEYBOARD':
                self._handle_keyboard(frame, feat, ear)

            # ★ 加入安檢 / 快捷鍵邏輯 ★
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break # 偵測到 ESC 鍵，離開系統
            elif key == ord('k'): # 按下鍵盤的 'k'，強制跳出全螢幕虛擬鍵盤
                if self.mode != 'KEYBOARD':
                    print("手動觸發虛擬鍵盤")
                    self._switch_to_keyboard()

        self.cap.release()
        cv2.destroyAllWindows()

    def _handle_calibration(self, frame, feat, ear, display):
        display[self.H - 240:self.H, self.W - 320:self.W] = cv2.resize(frame, (320, 240))
        if feat:
            pt = self.calib_pts[self.calib_idx]
            is_locked = self.tracker.is_stable() and (ear > Config.EAR_THRESH)
            color = (0, 255, 0) if is_locked else (0, 0, 255)

            cv2.circle(display, pt, 25, color, -1)
            cv2.circle(display, pt, 50, color, 2)

            if is_locked:
                self.calib_buffer.append(feat)
                prog = len(self.calib_buffer) / Config.CALIB_SAMPLES
                cv2.ellipse(display, pt, (60, 60), -90, 0, int(360 * prog), (0, 255, 0), 5)

                if len(self.calib_buffer) >= Config.CALIB_SAMPLES:
                    for f in self.calib_buffer: self.model.add_data(f, pt[0], pt[1])
                    self.calib_buffer.clear()
                    self.calib_idx += 1
                    self.tracker.buffer.clear()
                    time.sleep(0.5)

                    if self.calib_idx >= 9:
                        self.model.train()
                        self.mode = 'DESKTOP'
                        self._set_fullscreen(False)
            else:
                if self.calib_buffer: self.calib_buffer.pop(0)
        cv2.imshow(self.win_name, display)

    def _handle_desktop(self, frame, feat, ear, is_double_blink):
        display_small = cv2.resize(frame, (320, 240))
        btn_rect = (110, 10, 210, 50)  # (x1, y1, x2, y2)

        cv2.rectangle(display_small, btn_rect[:2], btn_rect[2:], (0, 0, 0), -1)
        cv2.rectangle(display_small, btn_rect[:2], btn_rect[2:], (255, 255, 255), 2)
        cv2.putText(display_small, "Keyboard", (125, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if feat:
            pred = self.model.predict(feat)
            if pred:
                kx, ky = self.tracker.kalman.update(pred[0], pred[1])
                cx, cy = self.W / 2, self.H / 2
                fx = np.clip(cx + (kx - cx) * Config.SCREEN_EXPANSION, 0, self.W)
                fy = np.clip(cy + (ky - cy) * Config.SCREEN_EXPANSION, 0, self.H)

                if ear > Config.EAR_THRESH:
                    pyautogui.moveTo(fx, fy)

                self._check_desktop_interactions(fx, fy, is_double_blink, display_small, btn_rect)

        cv2.imshow(self.win_name, display_small)

    def _check_desktop_interactions(self, fx, fy, is_double_blink, display, btn_rect):
        abs_x1, abs_y1 = self.win_pos[0] + btn_rect[0], self.win_pos[1] + btn_rect[1]
        abs_x2, abs_y2 = self.win_pos[0] + btn_rect[2], self.win_pos[1] + btn_rect[3]

        if abs_x1 < fx < abs_x2 and abs_y1 < fy < abs_y2:
            self._switch_to_keyboard()
            return

        if is_double_blink:
            pyautogui.click()
            cv2.putText(display, "Input Mode...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(self.win_name, display)
            cv2.waitKey(1)

            for _ in range(5):
                time.sleep(0.1)
                if WindowsHelper.is_text_cursor():
                    self._switch_to_keyboard()
                    break

    def _switch_to_keyboard(self):
        self.mode = 'KEYBOARD'
        self._set_fullscreen(True)
        self.keyboard.input_buffer = ""
        self.keyboard.update_predictions()

    def _handle_keyboard(self, frame, feat, ear):
        if feat and ear > Config.EAR_THRESH:
            pred = self.model.predict(feat)
            if pred:
                # 1. 取得濾波後的原始座標
                kx, ky = self.tracker.kalman.update(pred[0], pred[1])

                # 2. ★ 補上座標放大對應 ★
                cx, cy = self.W / 2, self.H / 2
                fx = np.clip(cx + (kx - cx) * Config.SCREEN_EXPANSION, 0, self.W)
                fy = np.clip(cy + (ky - cy) * Config.SCREEN_EXPANSION, 0, self.H)

                # 3. 把放大後的正確座標 (fx, fy) 傳給鍵盤畫綠點
                display_kb, close_req = self.keyboard.render(fx, fy, frame)

                if close_req:
                    self.mode = 'DESKTOP'
                    self.tracker.kalman.reset()
                    self._set_fullscreen(False)

                cv2.imshow(self.win_name, display_kb)
        else:
            cv2.imshow(self.win_name, self.keyboard.render(0, 0, frame)[0])


if __name__ == "__main__":
    app = EyeTrackingApp()
    app.run()