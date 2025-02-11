import os
import threading
import cv2
import win32api
import time
from typing import List
from buttons import Buttons
from capture import capture
from config_watcher import cfg
from mouse import mouse
from shooting import shooting
from visual import visuals

# Constants
CLASS_0 = 0.0
CLASS_1 = 1.0
CLASS_5 = 5.0
CLASS_6 = 6.0
CLASS_7 = 7.0
CLASS_10 = 10.0

class HotkeysWatcher(threading.Thread):
    def __init__(self):
        super(HotkeysWatcher, self).__init__()
        self.daemon = True
        self.name = 'HotkeysWatcher'
        self.app_pause = 0
        self.clss = self.active_classes()
        self.start()

    def run(self):
        cfg_reload_prev_state = 0
        while True:
            cfg_reload_prev_state = self.process_hotkeys(cfg_reload_prev_state)

            if win32api.GetAsyncKeyState(Buttons.KEY_CODES.get(cfg.hotkey_exit)) & 0xFF:
                capture.quit()
                if cfg.show_window:
                    visuals.queue.put(None)
                    time.sleep(0.1)
                    visuals.quit()
                    cv2.destroyAllWindows()
                os._exit(0)

    def process_hotkeys(self, cfg_reload_prev_state):
        self.app_pause = win32api.GetKeyState(Buttons.KEY_CODES[cfg.hotkey_pause])
        app_reload_cfg = win32api.GetKeyState(Buttons.KEY_CODES[cfg.hotkey_reload_config])
        if win32api.GetAsyncKeyState(Buttons.KEY_CODES['F5']) & 0xFF:
           cfg.edit_config()
        if app_reload_cfg != cfg_reload_prev_state:
            if app_reload_cfg in (1, 0):
                cfg.read(verbose=False)
                capture.restart()
                mouse.update_settings()
                self.clss = self.active_classes()
                if not cfg.show_window:
                    cv2.destroyAllWindows()
        cfg_reload_prev_state = app_reload_cfg
        return cfg_reload_prev_state

    def active_classes(self) -> List[int]:
        clss = [CLASS_0, CLASS_1]
        if cfg.hideout_targets:
            clss.extend([CLASS_5, CLASS_6])
        if not cfg.disable_headshot:
            clss.append(CLASS_7)
        if cfg.third_person:
            clss.append(CLASS_10)
        self.clss = clss

hotkeys_watcher = HotkeysWatcher()