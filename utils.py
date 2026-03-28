# utils.py
import ctypes
from ctypes import wintypes

class WindowsHelper:
    """處理 Windows 底層 API 的工具類別"""
    class CURSORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD),
                    ("hCursor", wintypes.HANDLE), ("ptScreenPos", wintypes.POINT)]

    @staticmethod
    def is_text_cursor():
        try:
            user32 = ctypes.windll.user32
            h_ibeam = user32.LoadCursorW(0, 32513)
            cursor_info = WindowsHelper.CURSORINFO()
            cursor_info.cbSize = ctypes.sizeof(WindowsHelper.CURSORINFO)
            user32.GetCursorInfo(ctypes.byref(cursor_info))
            return cursor_info.hCursor == h_ibeam
        except:
            return False