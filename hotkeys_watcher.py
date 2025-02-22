import threading
import time
import win32api
import os
from buttons import Buttons
from config_watcher import cfg
from utils import log_error

class HotkeysWatcher:
    def __init__(self, context):
        self.context = context
        self.running = True
        self.app_pause = 0
        self.clss = None  # Cached active classes, if used by frame_parser
        self.hotkey_exit = Buttons.KEY_CODES.get(cfg.hotkey_exit, 113)  # Default F2
        self.hotkey_pause = Buttons.KEY_CODES.get(cfg.hotkey_pause, 114)  # Default F3
        self.hotkey_reload = Buttons.KEY_CODES.get(cfg.hotkey_reload_config, 115)  # Default F4
        self.hotkey_edit = Buttons.KEY_CODES.get(cfg.hotkey_edit_config, 116)  # Default F5
        
        log_error("HotkeysWatcher initialized")
        self.start_monitoring_hotkeys()

    def start_monitoring_hotkeys(self):
        """Start a thread to monitor hotkeys."""
        self.hotkey_thread = threading.Thread(target=self.monitor_hotkeys, daemon=True)
        self.hotkey_thread.start()

    def monitor_hotkeys(self):
        """Monitor hotkey states and trigger actions."""
        while self.running:
            try:
                # Exit hotkey (e.g., F2)
                if win32api.GetAsyncKeyState(self.hotkey_exit) & 0x8000:
                    log_error("Exit hotkey pressed, shutting down")
                    self.context.cleanup()
                    os._exit(0)  # Force immediate exit
                
                # Pause hotkey (e.g., F3)
                if win32api.GetAsyncKeyState(self.hotkey_pause) & 0x8000:
                    self.app_pause = 1 if self.app_pause == 0 else 0
                    log_error(f"App pause toggled to: {self.app_pause}")
                    time.sleep(0.2)  # Debounce
                
                # Reload config hotkey (e.g., F4)
                if win32api.GetAsyncKeyState(self.hotkey_reload) & 0x8000:
                    cfg.read(verbose=True)
                    log_error("Config reloaded")
                    time.sleep(0.2)  # Debounce
                
                # Edit config hotkey (e.g., F5)
                if win32api.GetAsyncKeyState(self.hotkey_edit) & 0x8000:
                    cfg.edit_config()
                    log_error("Config editor opened")
                    time.sleep(0.2)  # Debounce
            
            except Exception as e:
                log_error(f"Error in hotkey monitoring: {e}")
            time.sleep(0.01)  # Reduce CPU usage

    def active_classes(self):
        """Return active class IDs for targeting (e.g., for frame_parser)."""
        # Placeholder: Adjust based on your targeting logic
        # Example: Return all classes if no specific filter
        return list(range(11))  # Matches cls_model_data in frame_parser.py

    def quit(self):
        """Stop the hotkey monitoring thread."""
        self.running = False
        if hasattr(self, 'hotkey_thread'):
            self.hotkey_thread.join(timeout=1.0)
        log_error("HotkeysWatcher stopped")

# Example usage (if run standalone)
if __name__ == "__main__":
    from init import AppContext
    context = AppContext()
    context.initialize()
    watcher = HotkeysWatcher(context)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.quit()