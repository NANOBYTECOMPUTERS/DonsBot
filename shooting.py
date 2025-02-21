#shooting.py ---

import os
import queue
import threading
import time

import win32api
import win32con

from config_watcher import cfg
if cfg.mouse_rzr:
    from rzctl import RZControl, MOUSE_CLICK

if cfg.arduino_move or cfg.arduino_shoot:
    from arduino import arduino

if cfg.mouse_ghub:
    from ghub import GhubMouse


class Shooting(threading.Thread):
    def __init__(self, context):
        super(Shooting, self).__init__()
        self.queue = queue.Queue(maxsize=1)
        self.daemon = True
        self.name = 'Shooting'
        self.button_pressed = False

        if cfg.mouse_ghub:
            self.ghub = GhubMouse()

        if cfg.mouse_rzr:
            dll_name = "rzctl.dll"
            script_directory = os.path.dirname(os.path.abspath(__file__))
            dll_path = os.path.join(script_directory, dll_name)
            self.rzr = RZControl(dll_path)
            if not self.rzr.init():
                print("Failed to initialize rzctl")

    def run(self):
        while True:
            bscope, shooting_state = self.queue.get()
            while not self.queue.empty():
                bscope, shooting_state = self.queue.get()  # Process latest state
            self.shoot(bscope, shooting_state)
            
    def shoot(self, bscope, shooting_state):
        auto_shoot_active = cfg.auto_shoot and bscope
        triggerbot_active = cfg.triggerbot
        should_shoot = auto_shoot_active or (cfg.mouse_auto_aim and bscope)
        if should_shoot and not self.button_pressed:
            self._press_button()
            time.sleep(0.01)  # Debounce
        elif (not bscope or (not shooting_state and not triggerbot_active)) and self.button_pressed:
            self._release_button()
            time.sleep(0.01)  # Debounce

    def _press_button(self):
        if cfg.mouse_rzr:
            self.rzr.mouse_click(MOUSE_CLICK.LEFT_DOWN)
        elif cfg.mouse_ghub:
            self.ghub.mouse_down()
        elif cfg.arduino_shoot:
            arduino.press()
        else:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self.button_pressed = True

    def _release_button(self):
        if cfg.mouse_rzr:
            self.rzr.mouse_click(MOUSE_CLICK.LEFT_UP)
        elif cfg.mouse_ghub:
            self.ghub.mouse_up()
        elif cfg.arduino_shoot:
            arduino.release()
        else:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        self.button_pressed = False