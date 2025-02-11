import os
import math
import time
import torch
import torch.nn as nn
import win32api
import win32con
import supervision as sv
from buttons import Buttons
from config_watcher import cfg
from shooting import shooting
from visual import visuals

if cfg.mouse_rzr:
    from rzctl import RZControl
if cfg.mouse_ghub:
    from ghub import gHub

class MouseNet(nn.Module):
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
    def __init__(self):
        self.device = self.get_device()
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

        self.prev_position = None
        self.prev_velocity = None
        self.prev_time = None
        self.max_distance = math.sqrt(self.screen_width**2 + self.screen_height**2) / 2.0
        self.min_speed_multiplier = cfg.mouse_min_speed_multiplier
        self.max_speed_multiplier = cfg.mouse_max_speed_multiplier
        self.bscope = False

        self.model = MouseNet(self.device).to(self.device)
        self.model.eval()

        if cfg.mouse_ghub:
            self.ghub = gHub()

        if cfg.mouse_rzr:
            dll_name = "rzctl.dll"
            script_directory = os.path.dirname(os.path.abspath(__file__))
            dll_path = os.path.join(script_directory, dll_name)
            self.rzr = RZControl(dll_path)
            if not self.rzr.init():
                print("Failed to initialize rzctl")
                
        self.hotkey_codes = []
        for key in cfg.hotkey_targeting:
            code = Buttons.KEY_CODES.get(key.strip())
            if code is not None:
                self.hotkey_codes.append(code)

    def get_device(self):
        if 'cpu' in cfg.ai_device:
            return torch.device('cpu')
        return torch.device(f'cuda:{cfg.ai_device}')

    def process_data(self, data):
        if isinstance(data, sv.Detections):
            target_data = data.xyxy.mean(axis=1)
            target_w = data.xyxy[:, 2] - data.xyxy[:, 0]
            target_h = data.xyxy[:, 3] - data.xyxy[:, 1]
            target_cls = data.class_id[0] if data.class_id.size > 0 else None
            target_id = data.tracker_id[0] if data.tracker_id is not None else None
            target_x, target_y = target_data[0], target_data[1]
        else:
            target_x, target_y, target_w, target_h, target_cls, target_id = data

        if cfg.ai_mouse_net:
            input_list = [
                target_x, target_y, 
                target_w, target_h, 
                target_cls if target_cls is not None else 0, 
                target_id if target_id is not None else 0, 
                self.center_x, self.center_y
            ]
            input_tensor = torch.tensor(input_list, device=self.device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                movement = self.model(input_tensor).squeeze(0)
            move_x, move_y = movement.tolist()
        else:
            move_x, move_y = self.calc_movement(target_x, target_y)

        if ((cfg.show_window and (cfg.show_target_line or cfg.show_target_prediction_line or cfg.show_history_points)) or
                (cfg.show_overlay and (cfg.show_target_line or cfg.show_target_prediction_line or cfg.show_history_points))):
            if cfg.show_target_line:
                visuals.draw_target_line(target_x, target_y, target_cls)
            if cfg.show_target_prediction_line and not self.disable_prediction:
                target_x, target_y = self.predict_target_position(target_x, target_y, time.time())
                visuals.draw_predicted_position(target_x, target_y, target_cls)
            if cfg.show_history_points:
                visuals.draw_history_point_add_point(target_x, target_y)

        self.bscope = (self.check_target_in_scope(target_x, target_y, target_w, target_h, self.bscope_multiplier)
                       if (cfg.auto_shoot or cfg.triggerbot) else False)
        self.bscope = cfg.force_click or self.bscope
        shooting.queue.put((self.bscope, self.get_shooting_key_state()))
        self.move_mouse(move_x, move_y)

    def predict_target_position(self, target_x, target_y, current_time):
        if self.prev_time is None or self.prev_position is None or self.prev_velocity is None:
            self.prev_position = (target_x, target_y)
            self.prev_velocity = (0.0, 0.0)
            self.prev_time = current_time
            return target_x, target_y
        delta_time = current_time - self.prev_time
        if delta_time == 0:
            return target_x, target_y
        curr_pos = (target_x, target_y)
        velocity = (
            (curr_pos[0] - self.prev_position[0]) / delta_time,
            (curr_pos[1] - self.prev_position[1]) / delta_time
        )
        acceleration = (
            (velocity[0] - self.prev_velocity[0]) / delta_time,
            (velocity[1] - self.prev_velocity[1]) / delta_time
        )
        pred_int = delta_time * self.prediction_interval
        predicted_x = curr_pos[0] + velocity[0] * pred_int + 0.5 * acceleration[0] * (pred_int ** 2)
        predicted_y = curr_pos[1] + velocity[1] * pred_int + 0.5 * acceleration[1] * (pred_int ** 2)
        self.prev_position = curr_pos
        self.prev_velocity = velocity
        self.prev_time = current_time
        return predicted_x, predicted_y

    def calculate_speed_multiplier(self, distance):
        normalized_distance = min(distance / self.max_distance, 1.0)
        return self.min_speed_multiplier + (self.max_speed_multiplier - self.min_speed_multiplier) * (1 - normalized_distance)

    def calc_movement(self, target_x, target_y):
        offset_x = target_x - self.center_x
        offset_y = target_y - self.center_y
        distance = math.hypot(offset_x, offset_y)
        speed_multiplier = self.calculate_speed_multiplier(distance)
        deg_per_pixel_x = self.fov_x / self.screen_width
        deg_per_pixel_y = self.fov_y / self.screen_height
        mouse_move_x = offset_x * deg_per_pixel_x
        mouse_move_y = offset_y * deg_per_pixel_y
        move_x = (mouse_move_x / 360) * (self.dpi / self.mouse_sensitivity) * speed_multiplier
        move_y = (mouse_move_y / 360) * (self.dpi / self.mouse_sensitivity) * speed_multiplier
        return move_x, move_y

    def move_mouse(self, x, y):
        x = x if x is not None else 0
        y = y if y is not None else 0
        if x != 0 or y != 0:
            if (self.get_shooting_key_state() and not cfg.mouse_auto_aim and not cfg.triggerbot) or cfg.mouse_auto_aim:
                if not cfg.mouse_ghub and not cfg.arduino_move and not cfg.mouse_rzr:
                    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(x), int(y), 0, 0)
                elif cfg.mouse_ghub and not cfg.arduino_move and not cfg.mouse_rzr:
                    self.ghub.mouse_xy(int(x), int(y))
                elif cfg.mouse_rzr:
                    self.rzr.mouse_move(int(x), int(y), True)

    def get_shooting_key_state(self):
        for key_code in self.hotkey_codes:
            state = (win32api.GetKeyState(key_code) if cfg.mouse_lock_target else win32api.GetAsyncKeyState(key_code))
            if state < 0 or state == 1:
                return True
        return False

    def check_target_in_scope(self, target_x, target_y, target_w, target_h, reduction_factor):
        reduced_w = target_w * reduction_factor / 2.0
        reduced_h = target_h * reduction_factor / 2.0
        x1, x2 = target_x - reduced_w, target_x + reduced_w
        y1, y2 = target_y - reduced_h, target_y + reduced_h
        bscope = (self.center_x > x1 and self.center_x < x2 and
                  self.center_y > y1 and self.center_y < y2)
        if cfg.show_window and cfg.show_bscope_box:
            visuals.draw_bscope(x1, x2, y1, y2, bscope)
        return bscope

    def update_settings(self):
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

mouse = MouseThread()