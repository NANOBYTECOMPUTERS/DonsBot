import os
import threading
import cv2
import win32api
import time
from typing import List
from buttons import Buttons
from config_watcher import cfg
from utils import log_error

# Constants
CLASS_0 = 0.0
CLASS_1 = 1.0
CLASS_5 = 5.0
CLASS_6 = 6.0
CLASS_7 = 7.0
CLASS_10 = 10.0

class HotkeysWatcher(threading.Thread):
    def __init__(self, context):
        super().__init__()
        self.context = context
        self.daemon = True
        self.name = 'HotkeysWatcher'
        self.app_pause = 0
        self.clss = self.active_classes()  # Initialize clss
        self.start()

    def run(self):
        cfg_reload_prev_state = 0
        while True:
            try:
                cfg_reload_prev_state = self.process_hotkeys(cfg_reload_prev_state)
                if win32api.GetAsyncKeyState(Buttons.KEY_CODES.get(cfg.hotkey_exit)) & 0xFF:
                    self.clean_shutdown()
                    os._exit(0)
            except Exception as e:
                log_error("Hotkeys watcher error", e)
                time.sleep(1)
            # Introduce a small delay to prevent a busy loop
            time.sleep(0.08)

    def process_hotkeys(self, cfg_reload_prev_state):
        self.app_pause = win32api.GetKeyState(Buttons.KEY_CODES[cfg.hotkey_pause])
        app_reload_cfg = win32api.GetKeyState(Buttons.KEY_CODES[cfg.hotkey_reload_config])

        # Handle F5 key: open config editor and wait for key release
        if win32api.GetAsyncKeyState(Buttons.KEY_CODES['F5']) & 0x8000:
            try:
                time.sleep(0.08)
                cfg.edit_config()
            except Exception as e:
                log_error("Error opening config editor", e)
            # Wait for the F5 key to be released (consider debouncing if necessary)
            while win32api.GetAsyncKeyState(Buttons.KEY_CODES['F5']) & 0x8000:
                time.sleep(0.08)

        # Reload configuration if state changes
        if app_reload_cfg != cfg_reload_prev_state:
            if app_reload_cfg in (1, 0):
                cfg.read(verbose=True)
                self.context.capture.restart()
                self.context.mouse.update_settings()
                self.clss = self.active_classes()  # Update clss after config reload
                if not cfg.show_window:
                    cv2.destroyAllWindows()
        cfg_reload_prev_state = app_reload_cfg

        return cfg_reload_prev_state

    def active_classes(self) -> List[float]:
        clss = [CLASS_0, CLASS_1]
        if cfg.hideout_targets:
            clss.extend([CLASS_5, CLASS_6])
        if not cfg.disable_headshot:
            clss.append(CLASS_7)
        if cfg.third_person:
            clss.append(CLASS_10)
        return clss

    def clean_shutdown(self):
        self.context.cleanup()
