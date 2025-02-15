import threading
import queue
import ctypes
import time
import cv2
import numpy as np
import bettercam
from screeninfo import get_monitors
from config_watcher import cfg
from utils import log_error
import pickle
import os

class Capture(threading.Thread):
    MASK_FILE = "custom_mask.pkl"
    
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.name = "Capture"
        self.print_startup_messages()
        self._custom_region = []
        self._offset_x = None
        self._offset_y = None
        self.screen_x_center = cfg.detection_window_width // 2
        self.screen_y_center = cfg.detection_window_height // 2
        self.prev_detection_window_width = cfg.detection_window_width
        self.prev_detection_window_height = cfg.detection_window_height
        self.prev_bettercam_capture_fps = cfg.bettercam_capture_fps
        
        # Cache for OBS camera index (initialize to None)
        self.cached_obs_camera_index = None
        
        # Load or initialize custom mask points and create mask once
        self.mask_points = self._load_mask_points()
        self.custom_mask = self._create_mask_from_points(self.mask_points)
        self.frame_queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._primary_resolution = self.get_primary_display_resolution()
        self.capture_method = None
        self.bc = None  # Bettercam instance
        self.obs_camera = None  # OBS camera instance
        self._last_error_time = 0  # General error timestamp, not specific to OBS
        
        if cfg.bettercam_capture:
            self.setup_bettercam()
        elif cfg.obs_capture:
            self.setup_obs()
    
    def _load_mask_points(self):
        """Load mask points from file or create default if not exists"""
        if os.path.exists(self.MASK_FILE):
            try:
                with open(self.MASK_FILE, 'rb') as f:
                    points = pickle.load(f)
                if len(points) == 6 and all(len(p) == 2 for p in points):
                    return points
            except Exception as e:
                log_error("Error loading mask points", e)
        # Default to a centered hexagon if no valid mask exists
        w, h = cfg.detection_window_width, cfg.detection_window_height
        center_x, center_y = w // 2, h // 2
        radius = min(w, h) // 4
        return [
            (center_x, center_y - radius),  # Top
            (int(center_x + radius * np.cos(np.pi/3)), int(center_y - radius * np.sin(np.pi/3))),  # Top-right
            (center_x + radius, center_y),  # Right
            (center_x, center_y + radius),  # Bottom
            (int(center_x - radius * np.cos(np.pi/3)), int(center_y + radius * np.sin(np.pi/3))),  # Bottom-left
            (center_x - radius, center_y)  # Left
        ]
    
    def _create_mask_from_points(self, points):
        """Create a white mask with black polygon area (inverted logic)"""
        height = int(cfg.detection_window_height)
        width = int(cfg.detection_window_width)
        mask = np.ones((height, width), dtype=np.uint8) * 255  # White background
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 0)  # Black polygon area
        return mask
    
    def save_mask_points(self, points):
        """Save mask points to file and update mask in memory"""
        if len(points) != 6:
            log_error(f"Expected 6 points, got {len(points)}")
            return
        try:
            with open(self.MASK_FILE, 'wb') as f:
                pickle.dump(points, f)
            self.mask_points = points
            self.custom_mask = self._create_mask_from_points(points)
            log_error("Custom polygon mask saved and updated in memory")
        except Exception as e:
            log_error("Error saving mask points", e)
    
    def setup_bettercam(self):
        """Configure and start the bettercam instance"""
        region = self.calculate_screen_offset(
            custom_region=self._custom_region if self._custom_region else None,
            x_offset=0 if self._offset_x is None else self._offset_x,
            y_offset=0 if self._offset_y is None else self._offset_y,
        )
        self.bc = bettercam.create(
            device_idx=cfg.bettercam_monitor_id,
            output_idx=cfg.bettercam_gpu_id,
            output_color="BGR",
            max_buffer_len=16,
            region=region,
        )
        if not self.bc.is_capturing:
            self.bc.start(
                region=region,
                target_fps=cfg.bettercam_capture_fps,
            )
        self.capture_method = "bettercam"
    
    def setup_obs(self):
        """Setup the OBS Virtual Camera capture"""
        if cfg.obs_camera_id == "auto":
            if self.cached_obs_camera_index is None:
                self.cached_obs_camera_index = self.find_obs_virtual_camera()
            camera_id = self.cached_obs_camera_index
            if camera_id == -1:
                log_error("OBS Virtual Camera not found")
                raise RuntimeError("OBS Virtual Camera not found")
        elif cfg.obs_camera_id.isdigit():
            camera_id = int(cfg.obs_camera_id)
        self.obs_camera = cv2.VideoCapture(camera_id)
        self.obs_camera.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.detection_window_width)
        self.obs_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.detection_window_height)
        self.obs_camera.set(cv2.CAP_PROP_FPS, cfg.obs_capture_fps)
        self.capture_method = "obs"
    
    def capture_frame(self):
        """Capture a single frame and apply the pre-created custom polygon mask"""
        frame = None
        if self.capture_method == "bettercam":
            frame = self.bc.get_latest_frame()
        elif self.capture_method == "obs":
            ret_val, frame = self.obs_camera.read()
            if not ret_val:
                current_time = time.time()
                if current_time - self._last_error_time > 1:
                    log_error("Failed to capture frame from OBS Virtual Camera")
                    self._last_error_time = current_time
                return None

        if frame is not None and cfg.polygon_mask_enabled:
            # If the frame has an alpha channel, convert it to BGR
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if frame.shape[:2] == self.custom_mask.shape:
                # Use OpenCV bitwise operation to apply the mask efficiently
                # Invert the mask so that the polygon area (originally black, 0) becomes 255 and vice-versa.
                inv_mask = cv2.bitwise_not(self.custom_mask)
                frame = cv2.bitwise_and(frame, frame, mask=inv_mask)
            else:
                log_error(f"Frame dimensions {frame.shape[:2]} do not match mask dimensions {self.custom_mask.shape}")
        return frame
    
    def run(self):
        """Continuously capture frames and enqueue them"""
        while not self._stop_event.is_set():
            frame = self.capture_frame()
            if frame is not None:
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Instead of two queue operations, simply remove the old frame and put the new one
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put_nowait(frame)
            else:
                # Slightly longer sleep to reduce CPU usage when no frame is available.
                time.sleep(0.08)
    
    def get_new_frame(self):
        """Retrieve a new frame from the frame queue with a timeout"""
        try:
            return self.frame_queue.get(timeout=0.17)
        except queue.Empty:
            return None
    
    def restart(self):
        """Restart the capture instance if configuration parameters have changed"""
        if self.capture_method == "bettercam" and (
            self.prev_detection_window_height != cfg.detection_window_height or
            self.prev_detection_window_width != cfg.detection_window_width or
            self.prev_bettercam_capture_fps != cfg.bettercam_capture_fps
        ):
            self.bc.stop()
            self.setup_bettercam()
            self.screen_x_center = cfg.detection_window_width // 2
            self.screen_y_center = cfg.detection_window_height // 2
            self.prev_detection_window_width = cfg.detection_window_width
            self.prev_detection_window_height = cfg.detection_window_height
            log_error("Capture reloaded")
    
    def calculate_screen_offset(self, custom_region=None, x_offset=None, y_offset=None):
        """Calculate and return the screen capture region"""
        x_offset = 0 if x_offset is None else x_offset
        y_offset = 0 if y_offset is None else y_offset
        if not custom_region:
            left, top = self._primary_resolution
        else:
            left, top = custom_region
        left = left / 2 - cfg.detection_window_width / 2 + x_offset
        top = top / 2 - cfg.detection_window_height / 2 - y_offset
        width = left + cfg.detection_window_width
        height = top + cfg.detection_window_height
        return (int(left), int(top), int(width), int(height))
    
    def get_primary_display_resolution(self):
        """Retrieve the resolution of the primary monitor"""
        monitors = get_monitors()
        for monitor in monitors:
            if monitor.is_primary:
                return (monitor.width, monitor.height)
        return (1920, 1080)
    
    def find_obs_virtual_camera(self):
        """Attempt to locate the OBS Virtual Camera"""
        max_tested = 20
        obs_backend_name = "DSHOW"
        for i in range(max_tested):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                continue
            backend_name = cap.getBackendName()
            if backend_name == obs_backend_name:
                log_error(f"OBS Virtual Camera found at index {i}")
                cap.release()
                return i
            cap.release()
        return -1
    
    def print_startup_messages(self):
        """Print the application version and hotkey configuration"""
        version = 0
        try:
            with open("./version", "r") as f:
                lines = f.read().split("\n")
                version = lines[0].split("=")[1] if "=" in lines[0] else "Unknown"
        except FileNotFoundError:
            log_error("(version file is not found)")
        log_error(
            f"Activated (Version {version})\n\n"
            f"Hotkeys:\n"
            f"[{cfg.hotkey_targeting}] - Stream Effects\n"
            f"[{cfg.hotkey_exit}] - EXIT\n"
            f"[{cfg.hotkey_pause}] - PAUSE\n"
            f"[{cfg.hotkey_reload_config}] - Reload\n"
            f"[F5] - Settings\n"
        )
    
    def quit(self):
        """Stop the capture thread and release resources"""
        self._stop_event.set()
        if self.capture_method == "bettercam" and self.bc and self.bc.is_capturing:
            self.bc.stop()
        if self.capture_method == "obs" and self.obs_camera is not None:
            self.obs_camera.release()
        self.join()