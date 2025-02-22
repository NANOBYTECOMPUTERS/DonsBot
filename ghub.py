#ghub.py ---
from ctypes import *
from os import path

# Define constants for mouse input
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010

class GhubMouse:
    def __init__(self):
        self.basedir = path.dirname(path.abspath(__file__))
        self.dlldir = path.join(self.basedir, 'ghub_mouse.dll')
        
        if not path.exists(self.dlldir):
            raise FileNotFoundError(f"Could not find DLL at {self.dlldir}")

        self.gm = CDLL(self.dlldir)
        self.gmok = self.gm.mouse_open() == 1  # Assuming 1 means success
        if not self.gmok:
            print("Warning: Failed to open ghub_mouse.dll. Falling back to SendInput.")

    @staticmethod
    def _send_input(*inputs):
        nInputs = len(inputs)
        LPINPUT = INPUT * nInputs
        pInputs = LPINPUT(*inputs)
        cbSize = c_int(sizeof(INPUT))
        return windll.user32.SendInput(nInputs, pInputs, cbSize)

    @staticmethod
    def _create_input(structure):
        return INPUT(0, _INPUTunion(mi=structure))

    @staticmethod
    def _create_mouse_input(flags, x, y, data):
        return MOUSEINPUT(x, y, data, flags, 0, None)

    def mouse_xy(self, x, y):
        if self.gmok:
            return self.gm.moveR(x, y) == 1
        return self._send_input(self._create_input(self._create_mouse_input(MOUSEEVENTF_MOVE, x, y, 0)))

    def mouse_down(self, key=1):
        flag = MOUSEEVENTF_LEFTDOWN if key == 1 else MOUSEEVENTF_RIGHTDOWN
        if self.gmok:
            return self.gm.press(key) == 1
        return self._send_input(self._create_input(self._create_mouse_input(flag, 0, 0, 0)))

    def mouse_up(self, key=1):
        flag = MOUSEEVENTF_LEFTUP if key == 1 else MOUSEEVENTF_RIGHTUP
        if self.gmok:
            return self.gm.release(key) == 1
        return self._send_input(self._create_input(self._create_mouse_input(flag, 0, 0, 0)))

    def mouse_close(self):
        if self.gmok:
            result = self.gm.mouse_close()
            self.gmok = False
            return result == 1
LONG = c_long
DWORD = c_ulong
ULONG_PTR = POINTER(DWORD)

class MOUSEINPUT(Structure):
    _fields_ = (
        ('dx', LONG),
        ('dy', LONG),
        ('mouseData', DWORD),
        ('dwFlags', DWORD),
        ('time', DWORD),
        ('dwExtraInfo', ULONG_PTR)
    )

class _INPUTunion(Union):
    _fields_ = (('mi', MOUSEINPUT),)

class INPUT(Structure):
    _fields_ = (
        ('type', DWORD),
        ('union', _INPUTunion)
    )

