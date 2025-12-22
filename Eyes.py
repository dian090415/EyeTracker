import numpy as np

import Constant

def eye_open_ratio(lms, w, h):
    def pt(i): return np.array([lms[i].x*w, lms[i].y*h], dtype=np.float32)
    # 左眼
    l_h = np.linalg.norm(pt(Constant.L_H[0]) - pt(Constant.L_H[1])) + 1e-6
    l_v = np.linalg.norm(pt(Constant.L_V[0]) - pt(Constant.L_V[1]))
    # 右眼
    r_h = np.linalg.norm(pt(Constant.R_H[0]) - pt(Constant.R_H[1])) + 1e-6
    r_v = np.linalg.norm(pt(Constant.R_V[0]) - pt(Constant.R_V[1]))
    # 兩眼平均
    return 0.5*((l_v/l_h) + (r_v/r_h))

