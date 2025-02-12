import cv2
from screeninfo import get_monitors
import threading
import queue
import numpy as np
from mss import mss
from config_watcher import cfg
import time
class Capture(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.name = "Capture"
        self.print_startup_messages()
        self.screen_x_center = int(cfg.detection_window_width // 2)
        self.screen_y_center = int(cfg.detection_window_height // 2)
        self.prev_detection_window_width = cfg.detection_window_width
        self.prev_detection_window_height = cfg.detection_window_height
        self.prev_bettercam_capture_fps = cfg.bettercam_capture_fps
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        self.monitor = self.setup_mss()
        self.circle_mask = self._create_circle_mask()
        # Create the alpha mask once for use in bitwise_and
        self.alpha_channel = cv2.merge([self.circle_mask] * 4)

    def setup_mss(self):
        x, y = self.get_primary_display_resolution()
        left = int(x / 2 - cfg.detection_window_width / 2)
        top = int(y / 2 - cfg.detection_window_height / 2)
        return {
            "left": left,
            "top": top,
            "width": int(cfg.detection_window_width),
            "height": int(cfg.detection_window_height)
        }
    
    def run(self):
        with mss() as sct:
            target_fps = cfg.bettercam_capture_fps if cfg.bettercam_capture_fps > 0 else 1
            sleep_interval = 1.0 / target_fps

            while self.running:
                start_time = time.perf_counter()
                screenshot = sct.grab(self.monitor)
                frame = np.asarray(screenshot, dtype=np.uint8)
                masked_frame = cv2.bitwise_and(frame, self.alpha_channel)
                try:
                    self.frame_queue.put(masked_frame, block=False)
                except queue.Full:
                    try:
                        self.frame_queue.get(block=False)
                    except queue.Empty:
                        pass
                    self.frame_queue.put_nowait(masked_frame)
                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, sleep_interval - elapsed)
                time.sleep(sleep_time)

    def get_new_frame(self):
        try:
            return self.frame_queue.get(timeout=1)
        except queue.Empty:
            return None
        
    def get_primary_display_resolution(self):
        for m in get_monitors():
            if m.is_primary:
                return m.width, m.height
        raise ValueError("No primary monitor found")
    
    def print_startup_messages(self):
        version = "Unknown"
        try:
            with open('./version', 'r') as f:
                version = f.readline().split('=')[1].strip()
        except FileNotFoundError:
            print('(version file is not found)')
        print(f'Stream Guardian (Version {version})\n\n'
              'Hotkeys:\n'
              f'[{cfg.hotkey_targeting}] - Stream Active\n'
              f'[{cfg.hotkey_exit}] - EXIT\n'
              f'[{cfg.hotkey_pause}] - Pause Stream\n'
              f'[{cfg.hotkey_reload_config}] - Reset\n')
        
    def _create_circle_mask(self):
        height = cfg.detection_window_height
        width = cfg.detection_window_width
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (width // 2, height // 2),
                    (width // 2, height // 2), 0, 0, 360, 255, -1)
        return mask
    
    def quit(self):
        self.running = False
        cv2.destroyAllWindows()
        self.join()
        
capture = Capture()
capture.start()