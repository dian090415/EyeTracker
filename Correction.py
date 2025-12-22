import Constant
import numpy as np


def apply_calibration(open_samples, close_samples, fallback_thr):
    global BLINK_RATIO_THR
    if open_samples and close_samples:
        open_mean = np.mean(open_samples)
        close_mean = np.mean(close_samples)
        # 門檻設在兩者中間，再留一點
        return (open_mean + close_mean) / 2.0 * 0.95

    return fallback_thr