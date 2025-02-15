import configparser
import random
import os
from threading import Lock
from configeditor import ConfigEditor  # Move import here
from utils import log_error
class Config:
    CONFIG_SECTIONS = {
        "DETECTION_WINDOW": "Detection window",
        "CAPTURE_METHODS": "Capture Methods",
        "AIM": "Aim",
        "HOTKEYS": "Hotkeys",
        "MOUSE": "Mouse",
        "SHOOTING": "Shooting",
        "ARDUINO": "Arduino",
        "AI": "AI",
        "OVERLAY": "overlay",
        "DEBUG_WINDOW": "Debug window",
    }

    def __init__(self):
        self.config = configparser.ConfigParser()
        self.editor_lock = Lock()
        self.restart_callback = None
        self.read(verbose=False)

    def set_restart_callback(self, callback):
        self.restart_callback = callback

    def edit_config(self):
        with self.editor_lock:
            try:
                editor = ConfigEditor(self, self.restart_callback)
                editor.show()
            except Exception as e:
                log_error("Error opening config editor: {e}")

    def read(self, verbose=True):
        directory = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(directory, "config.ini")
        
        if not os.path.isfile(config_path):
            self.write()
        
        self.config.read(config_path)
        
        if verbose:
            print("Config file loaded")

        self.load_detection_window()
        self.load_capture_methods()
        self.load_aim()
        self.load_hotkeys()
        self.load_mouse()
        self.load_shooting()
        self.load_arduino()
        self.load_ai()
        self.load_overlay()
        self.load_debug_window()

        if verbose:
            print("Config reloaded")

    def load_detection_window(self):
        section = self.CONFIG_SECTIONS["DETECTION_WINDOW"]
        self.detection_window_width = int(self.config[section]["detection_window_width"])
        self.detection_window_height = int(self.config[section]["detection_window_height"])
        self.polygon_mask_enabled = self.config.getboolean(section, "polygon_mask_enabled", fallback=False)
        self.use_padding = self.config.getboolean(section, "use_padding")
        self.shared_memory_usage = self.config.getboolean(section, "shared_memory_usage")
        self.show_trajectory = self.config.getboolean(section, "show_trajectory")

    def load_capture_methods(self):
        section = self.CONFIG_SECTIONS["CAPTURE_METHODS"]
        self.mss_capture = self.config.getboolean(section, "mss_capture")
        self.mss_capture_fps = int(self.config[section]["mss_fps"])
        self.bettercam_capture = self.config.getboolean(section, "bettercam_capture")
        self.bettercam_capture_fps = int(self.config[section]["bettercam_capture_fps"])
        self.bettercam_monitor_id = int(self.config[section]["bettercam_monitor_id"])
        self.bettercam_gpu_id = int(self.config[section]["bettercam_gpu_id"])
        self.obs_capture = self.config.getboolean(section, "obs_capture")
        self.obs_camera_id = self.config[section]["obs_camera_id"]
        self.obs_capture_fps = int(self.config[section]["obs_capture_fps"])

    def load_aim(self):
        section = self.CONFIG_SECTIONS["AIM"]
        self.body_y_offset = float(self.config[section]["body_y_offset"])
        self.hideout_targets = self.config.getboolean(section, "hideout_targets")
        self.disable_headshot = self.config.getboolean(section, "disable_headshot")
        self.disable_prediction = self.config.getboolean(section, "disable_prediction")
        self.prediction_interval = float(self.config[section]["prediction_interval"])
        self.third_person = self.config.getboolean(section, "third_person")
        self.switch_threshold = float(self.config[section]["switch_threshold"])
        self.smoothing_factor = float(self.config[section]["smoothing_factor"])

    def load_hotkeys(self):
        section = self.CONFIG_SECTIONS["HOTKEYS"]
        self.hotkey_targeting = self.config[section]["hotkey_targeting"].split(",")
        self.hotkey_exit = self.config[section]["hotkey_exit"]
        self.hotkey_pause = self.config[section]["hotkey_pause"]
        self.hotkey_reload_config = self.config[section]["hotkey_reload_config"]

    def load_mouse(self):
        section = self.CONFIG_SECTIONS["MOUSE"]
        self.mouse_dpi = int(self.config[section]["mouse_dpi"])
        self.mouse_sensitivity = float(self.config[section]["mouse_sensitivity"])
        self.mouse_fov_width = float(self.config[section]["mouse_fov_width"])
        self.mouse_fov_height = float(self.config[section]["mouse_fov_height"])
        self.mouse_min_speed_multiplier = float(self.config[section]["mouse_min_speed_multiplier"])
        self.mouse_max_speed_multiplier = float(self.config[section]["mouse_max_speed_multiplier"])
        self.mouse_lock_target = self.config.getboolean(section, "mouse_lock_target")
        self.mouse_auto_aim = self.config.getboolean(section, "mouse_auto_aim")
        self.mouse_ghub = self.config.getboolean(section, "mouse_ghub")
        self.mouse_rzr = self.config.getboolean(section, "mouse_rzr")

    def load_shooting(self):
        section = self.CONFIG_SECTIONS["SHOOTING"]
        self.auto_shoot = self.config.getboolean(section, "auto_shoot")
        self.triggerbot = self.config.getboolean(section, "triggerbot")
        self.force_click = self.config.getboolean(section, "force_click")
        self.bscope_multiplier = float(self.config[section]["bscope_multiplier"])

    def load_arduino(self):
        section = self.CONFIG_SECTIONS["ARDUINO"]
        self.arduino_move = self.config.getboolean(section, "arduino_move")
        self.arduino_shoot = self.config.getboolean(section, "arduino_shoot")
        self.arduino_port = self.config[section]["arduino_port"]
        self.arduino_baudrate = int(self.config[section]["arduino_baudrate"])
        self.arduino_16_bit_mouse = self.config.getboolean(section, "arduino_16_bit_mouse")

    def load_ai(self):
        section = self.CONFIG_SECTIONS["AI"]
        self.ai_model_name = self.config[section]["ai_model_name"]
        self.ai_model_image_size = int(self.config[section]["ai_model_image_size"])
        self.ai_conf = float(self.config[section]["ai_conf"])
        self.ai_device = self.config[section]["ai_device"]
        self.ai_enable_amd = self.config.getboolean(section, "ai_enable_AMD")
        self.ai_mouse_net = self.config.getboolean(section, "ai_mouse_net")
        self.disable_tracker = self.config.getboolean(section, "disable_tracker")

    def load_overlay(self):
        section = self.CONFIG_SECTIONS["OVERLAY"]
        self.show_overlay = self.config.getboolean(section, "show_overlay")
        self.overlay_show_borders = self.config.getboolean(section, "overlay_show_borders")
        self.overlay_show_boxes = self.config.getboolean(section, "overlay_show_boxes")
        self.overlay_show_target_line = self.config.getboolean(section, "overlay_show_target_line")
        self.overlay_show_target_prediction_line = self.config.getboolean(section, "overlay_show_target_prediction_line")
        self.overlay_show_labels = self.config.getboolean(section, "overlay_show_labels")
        self.overlay_show_conf = self.config.getboolean(section, "overlay_show_conf")

    def load_debug_window(self):
        section = self.CONFIG_SECTIONS["DEBUG_WINDOW"]
        self.show_window = self.config.getboolean(section, "show_window")
        self.show_detection_speed = self.config.getboolean(section, "show_detection_speed")
        self.show_window_fps = self.config.getboolean(section, "show_window_fps")
        self.show_boxes = self.config.getboolean(section, "show_boxes")
        self.show_labels = self.config.getboolean(section, "show_labels")
        self.show_conf = self.config.getboolean(section, "show_conf")
        self.show_target_line = self.config.getboolean(section, "show_target_line")
        self.show_target_prediction_line = self.config.getboolean(section, "show_target_prediction_line")
        self.show_bscope_box = self.config.getboolean(section, "show_bscope_box")
        self.show_history_points = self.config.getboolean(section, "show_history_points")
        self.debug_window_always_on_top = self.config.getboolean(section, "debug_window_always_on_top")
        self.spawn_window_pos_x = int(self.config[section]["spawn_window_pos_x"])
        self.spawn_window_pos_y = int(self.config[section]["spawn_window_pos_y"])
        self.debug_window_scale_percent = int(self.config[section]["debug_window_scale_percent"])
        self.debug_window_screenshot_key = self.config[section]["debug_window_screenshot_key"]
        self.debug_window_name = 'OBS STREAM'

    def get_random_window_name(self):
        try:
            with open("window_names.txt", "r", encoding="utf-8") as file:
                window_names = file.read().splitlines()
            return random.choice(window_names) if window_names else "Calculator"
        except FileNotFoundError:
            print("window_names.txt file not found, using default window name.")
            return "Calculator"
        
    def edit_config(self):
        with self.editor_lock:
            try:
                editor = ConfigEditor(self, self.restart_callback)
                editor.show()
            except Exception as e:
                log_error("Error opening config editor: {e}")

    def write(self):
        """Write current configuration to file"""
        directory = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(directory, "config.ini")
        
        with open(config_path, 'w') as configfile:
            self.config.write(configfile)    
cfg = Config()