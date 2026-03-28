# config.py

class Config:
    # 螢幕游標移動放大倍率
    SCREEN_EXPANSION = 1.4

    # 眼睛判定參數
    EAR_THRESH = 0.28
    EAR_BLINK_THRESH = 0.22

    # 游標穩定參數 (Kalman 濾波)
    STABILITY_THRESHOLD = 0.008
    STABILITY_BUFFER = 10

    # 校正參數
    CALIB_SAMPLES = 40

    # 互動時間參數
    DWELL_TIME = 0.8
    DOUBLE_BLINK_WINDOW = 0.6

    # 介面顏色
    COLOR_BG_OPAQUE = (0, 0, 0)
    COLOR_BORDER = (255, 255, 255)
    COLOR_TEXT = (255, 255, 255)