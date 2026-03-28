# keyboard.py
import cv2
import pyautogui
import numpy as np
import time
import json
import os
from PIL import ImageFont, ImageDraw, Image
from collections import defaultdict
from config import Config


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


class Button:
    def __init__(self, char, rect, is_predict=False):
        self.val = char
        self.rect = rect
        self.is_predict = is_predict
        self.label = self._set_label(char)

    def _set_label(self, char):
        labels = {'MINIMIZE': '縮小', 'BACKSPACE': '←', 'ENTER': '送出', 'SPACE': '空白'}
        return labels.get(char, char)


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
        self.hover_key = None
        self.hover_start_time = 0
        self.input_buffer = ""
        self.keyboard_height_ratio = 0.5

        self._init_fonts()
        self.setup_layout()
        self.update_predictions()

    def _init_fonts(self):
        try:
            self.font = ImageFont.truetype("msjh.ttc", 36)
            self.font_large = ImageFont.truetype("msjh.ttc", 48)
        except:
            self.font = ImageFont.load_default()
            self.font_large = ImageFont.load_default()

    def setup_layout(self):
        self.buttons.clear()
        start_y = int(self.H * (1 - self.keyboard_height_ratio))
        prediction_bar_h = 70
        keys_start_y = start_y + prediction_bar_h
        keys_area_h = self.H - keys_start_y

        row_count = len(self.layout_rows)
        btn_h = keys_area_h // row_count

        for r, line in enumerate(self.layout_rows):
            keys = line.split(",") if "," in line else list(line)
            btn_w = self.W // len(keys)
            for c, char in enumerate(keys):
                rect = (c * btn_w, keys_start_y + r * btn_h, (c + 1) * btn_w, keys_start_y + (r + 1) * btn_h)
                self.buttons.append(Button(char, rect))

    def update_predictions(self):
        self.predict_buttons.clear()
        last_char = self.input_buffer[-1] if self.input_buffer else ""
        predictions = self.ime.predict(last_char) or ["，", "。", "！", "？", "：", "......"]

        start_y = int(self.H * (1 - self.keyboard_height_ratio))
        btn_w = self.W // len(predictions)

        for i, char in enumerate(predictions):
            rect = (i * btn_w, start_y, (i + 1) * btn_w, start_y + 70)
            self.predict_buttons.append(Button(char, rect, is_predict=True))

    def process_action(self, val):
        should_close = False
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
            self._type_key(val)
            self.update_predictions()
        return should_close

    def _type_key(self, char):
        try:
            import pyperclip
            pyperclip.copy(char)
            pyautogui.hotkey('ctrl', 'v')
        except:
            pass

    def render(self, gaze_x, gaze_y, frame):
        frame = cv2.resize(frame, (self.W, self.H))
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        start_y = int(self.H * (1 - self.keyboard_height_ratio))

        draw.rectangle([0, start_y, self.W, self.H], fill='black')
        draw.rectangle([0, 0, self.W, 100], fill='black', outline='white', width=3)
        draw.text((30, 20), f"輸入: {self.input_buffer}", font=self.font_large, fill='white')

        should_close = False
        all_buttons = self.predict_buttons + self.buttons

        for btn in all_buttons:
            x1, y1, x2, y2 = btn.rect
            is_hover = (x1 < gaze_x < x2) and (y1 < gaze_y < y2)
            btn_color = (0, 40, 0) if btn.is_predict else 'black'

            if is_hover:
                if self.hover_key != btn.val:
                    self.hover_key = btn.val
                    self.hover_start_time = time.time()

                elapsed = time.time() - self.hover_start_time
                progress = min(elapsed / Config.DWELL_TIME, 1.0)

                draw.rectangle([x1, y1, x2, y2], fill=(80, 80, 80))
                draw.rectangle([x1, y1, x1 + (x2 - x1) * progress, y2], fill=(0, 200, 0))

                if elapsed >= Config.DWELL_TIME:
                    self.hover_key = None
                    draw.rectangle([x1, y1, x2, y2], fill='green')
                    should_close = self.process_action(btn.val)
            else:
                if self.hover_key == btn.val: self.hover_key = None
                draw.rectangle([x1, y1, x2, y2], fill=btn_color)

            draw.rectangle([x1, y1, x2, y2], outline='white', width=2)

            bbox = draw.textbbox((0, 0), btn.label, font=self.font)
            tx = x1 + (x2 - x1 - (bbox[2] - bbox[0])) // 2
            ty = y1 + (y2 - y1 - (bbox[3] - bbox[1])) // 2
            draw.text((tx, ty), btn.label, font=self.font, fill='white')

        draw.ellipse((gaze_x - 15, gaze_y - 15, gaze_x + 15, gaze_y + 15), fill=(0, 255, 0), outline='white')

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR), should_close