# mouse.py ---
import math
import os
import time
import torch
import torch.nn as nn
import win32api
import win32con
import supervision as sv
import cupy as cp
from buttons import Buttons
from config_watcher import cfg
from utils import get_torch_device
from utils import log_error

# Dynamic module imports based on config (ensure config is trusted)
if getattr(cfg, "arduino_move", False) or getattr(cfg, "arduino_shoot", False):
    try:
        from arduino import arduino
    except ImportError as ie:
        raise ImportError("Arduino module required by cfg but not found.") from ie

if getattr(cfg, "mouse_rzr", False):
    try:
        from rzctl import RZControl
    except ImportError as ie:
        raise ImportError("RZControl module required by cfg but not found.") from ie

if getattr(cfg, "mouse_ghub", False):
    try:
        from ghub import GhubMouse

    except ImportError as ie:
        raise ImportError("gHub module required by cfg but not found.") from ie


class MouseNet(nn.Module):
    """Neural network model for processing mouse movements."""
    def __init__(self, device):
        super(MouseNet, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)
        self.device = device

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class MouseThread:
    """Thread for processing mouse movement, target prediction and shooting."""
    def __init__(self, context):
        self.context = context
        self.device = get_torch_device(cfg)
        # Validate numeric configuration values again
        try:
            self.dpi = float(cfg.mouse_dpi)
            self.mouse_sensitivity = float(cfg.mouse_sensitivity)
            self.fov_x = float(cfg.mouse_fov_width)
            self.fov_y = float(cfg.mouse_fov_height)
            self.prediction_interval = float(cfg.prediction_interval)
            self.bscope_multiplier = float(cfg.bscope_multiplier)
        except (ValueError, AttributeError) as e:
            raise ValueError("Invalid numeric configuration value.") from e

        self.disable_prediction = cfg.disable_prediction
        self.screen_width = cfg.detection_window_width
        self.screen_height = cfg.detection_window_height
        self.center_x = self.screen_width / 2.0
        self.center_y = self.screen_height / 2.0
        self.prev_position = None
        self.prev_velocity = None
        self.prev_time = None
        self.max_distance = math.hypot(self.screen_width, self.screen_height) / 2.0
        self.min_speed_multiplier = cfg.mouse_min_speed_multiplier
        self.max_speed_multiplier = cfg.mouse_max_speed_multiplier
        self.bscope = False

        # Precompute values that do not change often
        self.update_settings()

        self.model = MouseNet(self.device).to(self.device)
        self.model.eval()

        # Initialize additional modules based on cfg with error handling
        if getattr(cfg, "mouse_ghub", False):
            try:
                self.ghub = GhubMouse()
            except Exception as e:
                raise RuntimeError("Failed to initialize gHub.") from e

        if getattr(cfg, "mouse_rzr", False):
            dll_name = "rzctl.dll"
            script_directory = os.path.dirname(os.path.abspath(__file__))
            dll_path = os.path.join(script_directory, dll_name)
            if not os.path.exists(dll_path):
                raise FileNotFoundError(f"Required DLL not found: {dll_path}")
            self.rzr = RZControl(dll_path)
            if not self.rzr.init():
                raise RuntimeError("Failed to initialize RZControl (rzctl).")

        self.hotkey_codes = []
        for key in cfg.hotkey_targeting:
            code = Buttons.KEY_CODES.get(key.strip())
            if code is not None:
                self.hotkey_codes.append(code)

    def update_settings(self):
        """Update settings from the configuration file and precompute constants."""
        try:
            self.dpi = cfg.mouse_dpi
            self.mouse_sensitivity = cfg.mouse_sensitivity
            self.fov_x = cfg.mouse_fov_width
            self.fov_y = cfg.mouse_fov_height
            self.disable_prediction = cfg.disable_prediction
            self.prediction_interval = cfg.prediction_interval
            self.bscope_multiplier = cfg.bscope_multiplier
            self.screen_width = cfg.detection_window_width
            self.screen_height = cfg.detection_window_height
            self.center_x = self.screen_width / 2.0
            self.center_y = self.screen_height / 2.0

            # Precompute per-pixel degrees
            self.deg_per_pixel_x = self.fov_x / self.screen_width
            self.deg_per_pixel_y = self.fov_y / self.screen_height
        except Exception as exc:
            log_error("Error updating settings: {exc}")

    def process_data(self, data):
        try:
            if isinstance(data, sv.Detections):
                # Convert xyxy to CuPy array for GPU processing
                xyxy = cp.asarray(data.xyxy)
                target_data = cp.mean(xyxy, axis=1)  # Mean along axis 1 (x, y pairs)
                target_w = xyxy[:, 2] - xyxy[:, 0]   # Widths
                target_h = xyxy[:, 3] - xyxy[:, 1]   # Heights
                # Extract first target (or adapt for multi-target logic)
                target_x, target_y = cp.asnumpy(target_data[0]), cp.asnumpy(target_data[1])
                target_cls = data.class_id[0] if data.class_id.size > 0 else None
                target_id = data.tracker_id[0] if data.tracker_id is not None else None
            else:
                (target_x, target_y, target_w, target_h, target_cls, target_id) = data
            move_x, move_y = self.calc_movement(target_x, target_y)


            # Rest of your logic (bscope, shooting, etc.)
            self.bscope = (self.check_target_in_scope(target_x, target_y, target_w, target_h, self.bscope_multiplier)
                        if (cfg.auto_shoot or cfg.triggerbot) else False)
            self.bscope = cfg.force_click or self.bscope

            shooting_key_state = self.get_shooting_key_state()
            self.context.shooting.queue.put((self.bscope, shooting_key_state))
            self.move_mouse(move_x, move_y, shooting_key_state)
        except Exception as e:
            log_error("Error in process_data", e)

    def predict_target_position(self, target_x, target_y, current_time):
        """Predict target position with velocity and acceleration."""
        if self.prev_time is None or self.prev_position is None:
            # First call: initialize and return current position
            self.prev_position = (target_x, target_y)
            self.prev_time = current_time
            return target_x, target_y
        
        if self.prev_prev_time is None or self.prev_prev_position is None:
            # Second call: compute velocity, no acceleration yet
            delta_time = current_time - self.prev_time
            if delta_time == 0:
                return target_x, target_y
            velocity_x = (target_x - self.prev_position[0]) / delta_time
            velocity_y = (target_y - self.prev_position[1]) / delta_time
            pred_x = target_x + velocity_x * self.prediction_interval
            pred_y = target_y + velocity_y * self.prediction_interval
            self.prev_prev_position = self.prev_position
            self.prev_prev_time = self.prev_time
            self.prev_position = (target_x, target_y)
            self.prev_time = current_time
            return pred_x, pred_y

        # Third call onward: compute acceleration
        delta_time = current_time - self.prev_time
        prev_delta_time = self.prev_time - self.prev_prev_time
        if delta_time == 0 or prev_delta_time == 0:
            return target_x, target_y

        # Velocity between current and previous
        velocity_x = (target_x - self.prev_position[0]) / delta_time
        velocity_y = (target_y - self.prev_position[1]) / delta_time
        # Previous velocity
        prev_velocity_x = (self.prev_position[0] - self.prev_prev_position[0]) / prev_delta_time
        prev_velocity_y = (self.prev_position[1] - self.prev_prev_position[1]) / prev_delta_time
        # Acceleration
        accel_x = (velocity_x - prev_velocity_x) / delta_time
        accel_y = (velocity_y - prev_velocity_y) / delta_time

        # Predict using position + velocity * t + 0.5 * acceleration * t^2
        t = self.prediction_interval
        pred_x = target_x + velocity_x * t + 0.5 * accel_x * t * t
        pred_y = target_y + velocity_y * t + 0.5 * accel_y * t * t

        # Update history
        self.prev_prev_position = self.prev_position
        self.prev_prev_time = self.prev_time
        self.prev_position = (target_x, target_y)
        self.prev_time = current_time
        return pred_x, pred_y

    def calculate_speed_multiplier(self, distance):
        """Calculate a speed multiplier based on the distance to target."""
        normalized_distance = min(distance / self.max_distance, 1.0)
        return self.min_speed_multiplier + (self.max_speed_multiplier - self.min_speed_multiplier) * (1 - normalized_distance)

    def calc_movement(self, target_x, target_y):
        """Calculate movement values using CuPy for potential batch processing."""
        # Convert inputs to CuPy arrays (works with scalars or arrays)
        target_x = cp.asarray(target_x)
        target_y = cp.asarray(target_y)
        
        # Compute offsets from center
        offset_x = target_x - self.center_x
        offset_y = target_y - self.center_y
        
        # Compute distance using CuPy’s hypot function
        distance = cp.hypot(offset_x, offset_y)
        speed_multiplier = self.calculate_speed_multiplier(distance)

        # Use precomputed degrees per pixel
        mouse_move_x = offset_x * self.deg_per_pixel_x
        mouse_move_y = offset_y * self.deg_per_pixel_y

        # Calculate final movement with scaling
        move_x = (mouse_move_x / 360) * (self.dpi / self.mouse_sensitivity) * speed_multiplier
        move_y = (mouse_move_y / 360) * (self.dpi / self.mouse_sensitivity) * speed_multiplier

        # Convert back to NumPy/Python scalars for downstream use
        return cp.asnumpy(move_x), cp.asnumpy(move_y)

    def move_mouse(self, x, y, shooting_key_state):
        """Move the mouse cursor based on computed values and system settings."""
        x = x or 0
        y = y or 0
        try:
            if x != 0 or y != 0:
                if (shooting_key_state and not cfg.mouse_auto_aim and not cfg.triggerbot) or cfg.mouse_auto_aim:
                    if not cfg.mouse_ghub and not cfg.arduino_move and not cfg.mouse_rzr:
                        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(x), int(y), 0, 0)
                    elif cfg.mouse_ghub and not cfg.arduino_move and not cfg.mouse_rzr:
                        self.ghub.mouse_xy(int(x), int(y))
                    elif cfg.mouse_rzr:
                        self.rzr.mouse_move(int(x), int(y), True)
                    elif cfg.arduino_move:
                        arduino.move(int(x), int(y))
        except Exception as exc:
            log_error("Error in move_mouse: {exc}")

    def get_shooting_key_state(self):
        """Check and return whether any of the configured hotkeys are pressed."""
        try:
            for key_code in self.hotkey_codes:
                state = (win32api.GetKeyState(key_code)
                         if cfg.mouse_lock_target
                         else win32api.GetAsyncKeyState(key_code))
                if state < 0 or state == 1:
                    return True
        except Exception as exc:
            log_error("Error checking shooting key state: {exc}")
        return False

    def check_target_in_scope(self, target_x, target_y, target_w, target_h, reduction_factor):
            """
            Determine if the target is within a reduced bounding box (scope) on the screen.
            Draw the box if the configuration option is enabled.
            """
            reduced_w = target_w * reduction_factor / 2.0
            reduced_h = target_h * reduction_factor / 2.0
            x1, x2 = target_x - reduced_w, target_x + reduced_w
            y1, y2 = target_y - reduced_h, target_y + reduced_h
            bscope = (self.center_x > x1 and self.center_x < x2 and
                    self.center_y > y1 and self.center_y < y2)
            # This method no longer exists in Visuals, comment it out or remove it
            # if cfg.show_window and cfg.show_bscope_box:
            #     self.context.draw_bscope(x1, x2, y1, y2, bscope)
            return bscope