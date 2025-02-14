import queue
import threading
import time
import cv2
import torch
import win32gui
import win32con
import win32api
import os
from config_watcher import cfg
from capture import capture
from overlay import overlay
from buttons import Buttons

SCREENSHOT_DIRECTORY = "screenshots"
CIRCLE_RADIUS = 5

class Visuals(threading.Thread):
    def __init__(self):
        # Validate critical configuration values
        try:
            detection_width = int(cfg.detection_window_width)
            detection_height = int(cfg.detection_window_height)
        except (ValueError, TypeError) as e:
            raise ValueError("Invalid detection window dimensions in config") from e

        overlay.show(detection_width, detection_height)
        os.makedirs(SCREENSHOT_DIRECTORY, exist_ok=True)
        self.show_window = getattr(cfg, "show_window", False)
        self.show_overlay = getattr(cfg, "show_overlay", False)

        if self.show_window or self.show_overlay:
            super().__init__()
            self.queue = queue.Queue(maxsize=1)
            self.daemon = True
            self.name = 'Visuals'
            self.image = None
            self.screenshot_taken = False
            self.screenshot_lock = threading.Lock()  # Lock for screenshot flag
            self.interpolation = cv2.INTER_NEAREST if self.show_window else None
            self.draw_line_data = None
            self.draw_predicted_position_data = None
            self.draw_speed_data = None
            self.draw_bscope_data = None
            self.draw_history_point_data = []
            self.cls_model_data = {
                0: 'player',
                1: 'bot',
                2: 'weapon',
                3: 'outline',
                4: 'dead_body',
                5: 'hideout_target_human',
                6: 'hideout_target_balls',
                7: 'head',
                8: 'smoke',
                9: 'fire',
                10: 'third_person'
            }
            self.disabled_line_classes = [2, 3, 4, 8, 9, 10]
            self.cached_resize_dims = None
            self.running = True
            self.start()

    def run(self):
        if self.show_window:
            self.spawn_debug_window()
        while self.running:
            try:
                self.image = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.image is None:
                self.running = False
                break

            # Validate image dimensions before proceeding
            if not hasattr(self.image, "shape"):
                continue
            self.handle_screenshot()
            self.display_image()

    def handle_screenshot(self):
        screenshot_key = Buttons.KEY_CODES.get(getattr(cfg, "debug_window_screenshot_key", None))
        if screenshot_key is not None:
            screenshot_key_state = win32api.GetAsyncKeyState(screenshot_key)
            with self.screenshot_lock:
                if screenshot_key_state & 0x8000 and not self.screenshot_taken:
                    self.screenshot_taken = True
                    if self.image is not None:
                        image_copy = self.image.copy()
                        threading.Thread(target=self.save_screenshot, args=(image_copy,), daemon=True).start()
                elif not (screenshot_key_state & 0x8000):
                    self.screenshot_taken = False

    def save_screenshot(self, image):
        filename = os.path.join(SCREENSHOT_DIRECTORY, f"{time.time()}.jpg")
        try:
            cv2.imwrite(filename, image)
        except Exception as e:
            print(f"Error saving screenshot: {e}")

    def display_image(self):
        if self.show_window:
            display_img = self.image
            # Resize image only if needed and configured percentage is valid
            if hasattr(cfg, "debug_window_scale_percent") and int(cfg.debug_window_scale_percent) != 100:
                display_img = self.resize_image(display_img)
            cv2.imshow(getattr(cfg, "debug_window_name", "Debug Window"), display_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                self.cleanup()

    def resize_image(self, display_img):
        # Check if cached dimensions need update
        if (self.cached_resize_dims is None or 
            (display_img.shape[0], display_img.shape[1]) != self.cached_resize_dims[2:]):
            try:
                scale_percent = int(cfg.debug_window_scale_percent)
            except (ValueError, TypeError):
                scale_percent = 100
            height = int(display_img.shape[0] * scale_percent / 100)
            width = int(display_img.shape[1] * scale_percent / 100)
            self.cached_resize_dims = (width, height, display_img.shape[0], display_img.shape[1])
        else:
            width, height, _, _ = self.cached_resize_dims
        return cv2.resize(display_img, (width, height), self.interpolation)

    def spawn_debug_window(self):
        debug_window_name = getattr(cfg, "debug_window_name", "Debug Window")
        cv2.namedWindow(debug_window_name)
        if getattr(cfg, "debug_window_always_on_top", False):
            try:
                x = max(getattr(cfg, "spawn_window_pos_x", 0), 0)
                y = max(getattr(cfg, "spawn_window_pos_y", 0), 0)
                debug_window_hwnd = win32gui.FindWindow(None, debug_window_name)
                win32gui.SetWindowPos(
                    debug_window_hwnd,
                    win32con.HWND_TOPMOST,
                    x, y,
                    int(cfg.detection_window_width),
                    int(cfg.detection_window_height),
                    0
                )
            except Exception as e:
                print(f'Error setting window to always on top: {e}')

    def cleanup(self):
        self.running = False
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def draw_aim_point(self, x, y):
        if self.show_window and self.image is not None:
            cv2.circle(self.image, (int(x), int(y)), radius=CIRCLE_RADIUS, color=(0, 0, 255), thickness=-1)
        if self.show_overlay:
            overlay.draw_circle(int(x), int(y), CIRCLE_RADIUS, 'red')

    def draw_target_line(self, target_x, target_y, target_cls):
        if target_cls not in self.disabled_line_classes:
            self.draw_line_data = (target_x, target_y)

    def draw_predicted_position(self, target_x, target_y, target_cls):
        if target_cls not in self.disabled_line_classes:
            self.draw_predicted_position_data = (target_x, target_y)

    def draw_speed(self, speed_preprocess, speed_inference, speed_postprocess):
        self.draw_speed_data = (speed_preprocess, speed_inference, speed_postprocess)

    def draw_bscope(self, x1, x2, y1, y2, bscope):
        self.draw_bscope_data = (x1, x2, y1, y2, bscope)

    def draw_history_point_add_point(self, x, y):
        self.draw_history_point_data.append([int(x), int(y)])

    def clear(self):
        self.draw_line_data = None
        self.draw_predicted_position_data = None
        self.draw_speed_data = None
        self.draw_bscope_data = None
        self.draw_history_point_data = []

visuals = Visuals()