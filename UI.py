import numpy as np
import cv2
import Constant


__bar_w = 200
__bar_x = 10
__bar_y = 160

def show_blink_range_by_strip(disp, sm_ratio):
    val = int(np.clip((sm_ratio / 0.5) * __bar_w, 0, __bar_w))
    thr = int(np.clip((Constant.BLINK_RATIO_THR / 0.5) * __bar_w, 0, __bar_w))
    cv2.rectangle(disp, (__bar_x, __bar_y), (__bar_x + __bar_w, __bar_y + 10), (50, 50, 50), 1)
    cv2.rectangle(disp, (__bar_x, __bar_y), (__bar_x + val, __bar_y + 10), (0, 200, 0), -1)
    cv2.line(disp, (__bar_x + thr, __bar_y - 2), (__bar_x + thr, __bar_y + 12), (0, 0, 255), 2)