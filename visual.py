import queue
import threading
import time
import cv2
import win32gui
import win32con
import win32api
import os
import numpy as np
from config_watcher import cfg
from buttons import Buttons
from utils import log_error
import supervision as sv

SCREENSHOT_DIRECTORY = "screenshots"
CIRCLE_RADIUS = 5

class Visuals(threading.Thread):
    def __init__(self, context):
        self.context = context
        try:
            detection_width = int(cfg.detection_window_width)
            detection_height = int(cfg.detection_window_height)
        except (ValueError, TypeError) as e:
            log_error("Invalid detection window dimensions in config", e)
            raise ValueError("Invalid detection window dimensions in config") from e

        self.context.overlay.show(detection_width, detection_height)
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
            self.screenshot_lock = threading.Lock()
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
            self.editing_mask = False
            self.mask_points = self.context.capture.mask_points.copy()
            self.selected_point = None
            self.bounding_box_annotator = sv.BoxAnnotator(
                color=sv.ColorPalette.DEFAULT,
                thickness=2,
                color_lookup=sv.ColorLookup.CLASS
            )
            self.label_annotator = sv.LabelAnnotator(
                color=sv.ColorPalette.DEFAULT,
                text_color=sv.Color.WHITE,
                text_scale=0.5,
                text_thickness=1,
                text_padding=5,
                text_position=sv.Position.TOP_LEFT,
                color_lookup=sv.ColorLookup.CLASS
            )
            self.start()

    def run(self):
        if self.show_window:
            self.spawn_debug_window()
        while self.running:
            try:
                item = self.queue.get(timeout=0.2)
                if item is None:
                    log_error("Received None in visuals queue, continuing")
                    continue
                if isinstance(item, tuple) and len(item) == 2:
                    image, detections = item
                else:
                    log_error(f"Warning: Unexpected item format in visuals queue: {type(item)} - {item}")
                    image = item if not isinstance(item, tuple) else item[0]
                    detections = sv.Detections.empty() if hasattr(sv, 'Detections') else None
                
                if image is None:
                    self.running = False
                    break

                if not hasattr(image, "shape"):
                    log_error("Invalid image received in visuals queue")
                    continue
                self.image = image
                self.handle_screenshot()
                if self.show_window or self.show_overlay:
                    self.display_image(detections if detections is not None else sv.Detections.empty())
            except queue.Empty:
                continue
            except Exception as e:
                log_error("Error in Visuals run loop", e)

    def display_image(self, detections):
        if self.show_window:
            display_img = self.image.copy()
            if hasattr(cfg, "debug_window_scale_percent") and int(cfg.debug_window_scale_percent) != 100:
                display_img = self.resize_image(display_img)

            if cfg.show_boxes:
                annotated_img = self.annotate_with_supervision(detections, display_img)
            else:
                annotated_img = display_img

    def mouse_callback(self, event, x, y, flags, param):
        if not self.show_window:
            return

        log_error(f"Mouse event: {event}, x: {x}, y: {y}")

        if event == cv2.EVENT_LBUTTONDOWN:
            self.editing_mask = True
            # Find nearest point to click
            distances = [((p[0] - x)**2 + (p[1] - y)**2) for p in self.mask_points]
            if distances:
                self.selected_point = distances.index(min(distances))
                log_error(f"Selected point: {self.selected_point}")
        elif event == cv2.EVENT_MOUSEMOVE and self.editing_mask and self.selected_point is not None:
            self.mask_points[self.selected_point] = (x, y)
            log_error(f"Updated point {self.selected_point} to {x}, {y}")
        elif event == cv2.EVENT_LBUTTONUP and self.editing_mask:
            self.editing_mask = False
            self.selected_point = None
            self.context.capture.save_mask_points(self.mask_points)
            log_error("Mask points saved")

    def display_image(self, detections):
        if self.show_window:
            display_img = self.image.copy()
            if hasattr(cfg, "debug_window_scale_percent") and int(cfg.debug_window_scale_percent) != 100:
                display_img = self.resize_image(display_img)

            # Annotate image with supervision detections
            annotated_img = self.annotate_with_supervision(detections, display_img)

            if self.editing_mask or len(self.mask_points) > 0:
                points = np.array(self.mask_points, dtype=np.int32)
                cv2.polylines(display_img, [points], True, (0, 255, 0), 2)  # Green for visibility
                cv2.fillPoly(display_img, [points], (0, 0, 0, 128))  # Semi-transparent black to show the area that will be hidden
                for i, point in enumerate(self.mask_points):
                    px, py = point
                    if isinstance(px, (int, float)) and isinstance(py, (int, float)):
                        cv2.circle(annotated_img, (int(px), int(py)), 5, (0, 0, 255), -1)
                        cv2.putText(annotated_img, str(i), (int(px) + 10, int(py) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    else:
                        log_error(f"Invalid point at index {i}: {point}")

            cv2.imshow(getattr(cfg, "debug_window_name", "Debug Window"), annotated_img)
            if self.show_window:
                cv2.setMouseCallback(getattr(cfg, "debug_window_name", "Debug Window"), self.mouse_callback)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                self.cleanup()

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
            log_error("Error saving screenshot", e)

    def spawn_debug_window(self):
        self.window_name = getattr(cfg, "debug_window_name", "Debug Window")
        cv2.namedWindow(self.window_name)
        if getattr(cfg, "debug_window_always_on_top", False):
            try:
                x = max(getattr(cfg, "spawn_window_pos_x", 0), 0)
                y = max(getattr(cfg, "spawn_window_pos_y", 0), 0)
                debug_window_hwnd = win32gui.FindWindow(None, self.window_name)
                win32gui.SetWindowPos(
                    debug_window_hwnd,
                    win32con.HWND_TOPMOST,
                    x, y,
                    int(cfg.detection_window_width),
                    int(cfg.detection_window_height),
                    0
                )
            except Exception as e:
                log_error("Error setting window to always on top", e)

    def resize_image(self, display_img):
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

    def cleanup(self):
        self.running = False
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def annotate_with_supervision(self, detections, image):
        if cfg.show_boxes:
            annotated_image = self.bounding_box_annotator.annotate(
                scene=image,
                detections=detections
            )
            if cfg.show_labels or cfg.show_conf:
                labels = [f"{self.cls_model_data.get(cls, 'Unknown')} {conf:.2f}" for cls, conf in zip(detections.class_id, detections.confidence)]
                annotated_image = self.label_annotator.annotate(
                    scene=annotated_image,
                    detections=detections,
                    labels=labels
                )
            return annotated_image
        return image

    def draw_aim_point(self, x, y):
        if self.show_window and self.image is not None:
            cv2.circle(self.image, (int(x), int(y)), radius=CIRCLE_RADIUS, color=(0, 0, 255), thickness=-1)