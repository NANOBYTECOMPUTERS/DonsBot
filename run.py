import os
import time
import sys
import subprocess
import torch
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
from config_watcher import cfg
from init import get_app_context
from checks import run_checks
from utils import log_error

# Constants
IOU_THRESHOLD = 0.7
MAX_DETECTIONS = 20
ARROW_COLOR = (0, 255, 0)
ALIGNMENT = 128

# Runtime configuration (precomputed values)
class RunConfig:
    def __init__(self):
        self.pad_height = ((cfg.detection_window_height + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
        self.pad_width = ((cfg.detection_window_width + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
        self.use_padding = cfg.use_padding
        self.show_visuals = cfg.show_window or cfg.show_overlay
        self.show_trajectory = self.show_visuals and cfg.show_trajectory
        self.frame_interval = 1 / cfg.bettercam_capture_fps  # Dynamic sleep based on target FPS

run_config = RunConfig()

# Optimized padding function
def pad_to_alignment(arr):
    if arr.shape[:2] == (run_config.pad_height, run_config.pad_width):
        return arr
    return np.pad(arr, ((0, run_config.pad_height - arr.shape[0]), 
                        (0, run_config.pad_width - arr.shape[1]), (0, 0)), mode='constant')

class Tracker:
    def __init__(self):
        self.global_tracker = sv.ByteTrack() if not cfg.disable_tracker else None

    def update(self, detections):
        if self.global_tracker is not None:
            return self.global_tracker.update_with_detections(detections)
        return detections

@torch.inference_mode()
def perform_detection(model, image, tracker):
    image_np = np.ascontiguousarray(image)
    if run_config.use_padding:
        image_np = pad_to_alignment(image_np)
    results = model.predict(
        source=[image_np],
        imgsz=cfg.ai_model_image_size,
        stream=True,
        conf=cfg.ai_conf,
        iou=IOU_THRESHOLD,
        device=cfg.ai_device,
        half=True,
        max_det=MAX_DETECTIONS,
        verbose=False
    )
    for result in results:  # Stream mode yields one result per frame
        detections = sv.Detections.from_ultralytics(result)
        return tracker.update(detections) or sv.Detections.empty()
    return sv.Detections.empty()


def run_bot(context):
    run_checks()
    model = YOLO(f"models/{cfg.ai_model_name}", task="detect")
    show_visuals = cfg.show_window or cfg.show_overlay
    
    while True:
        try:
            image = context.capture.get_new_frame()
            if image is None:
                time.sleep(1 / cfg.bettercam_capture_fps)
                continue
            if context.hotkeys_watcher.app_pause != 0:
                time.sleep(0.1)
                continue
            
            # Apply mask before detection if enabled
            masked_image = image if not cfg.polygon_mask_enabled else context.capture.custom_mask_frame(image)
            result = perform_detection(model, masked_image, context.tracker) or sv.Detections.empty()
            context.frame_parser.parse(result)
            
            if show_visuals:
                context.visuals.queue.put((image, result))  # Original image for visuals
                if cfg.show_overlay:
                    context.overlay.queue.put((image, result))
        except Exception as e:
            log_error("Error in main loop", e)
            time.sleep(0.1)

def restart_bot(context):
    """Restart the entire application cleanly"""
    print("Restarting bot...")
    try:
        context.cleanup()  # Full cleanup
        cv2.destroyAllWindows()
        # Close any lingering Tkinter roots
        if 'tkinter' in sys.modules:
            import tkinter as tk
            if tk._default_root:
                tk._default_root.destroy()
        time.sleep(0.5)  # Longer delay to ensure cleanup
        python = sys.executable
        args = sys.argv[:]
        subprocess.Popen([python] + args)
        os._exit(0)  # Force exit, stronger than sys.exit
    except Exception as e:
        log_error(f"Error during restart: {e}")
        os._exit(1)
def main():
    context = get_app_context()
    try:
        cfg.set_restart_callback(lambda: restart_bot(context))
        context.tracker = Tracker()
        run_bot(context)
    finally:
        if context:
            context.cleanup()
    os._exit(0)

if __name__ == "__main__":
    main()