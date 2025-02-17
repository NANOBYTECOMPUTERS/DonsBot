# run.py ---

import os
import time
import sys
import subprocess
import torch
from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
from config_watcher import cfg
from init import get_app_context
from checks import run_checks
from utils import log_error

IOU_THRESHOLD = 0.7
MAX_DETECTIONS = 20
ARROW_COLOR = (0, 255, 0)
ALIGNMENT = 128

# Create a global cache for padded dimensions to avoid re-computation across frames.
PADDING_CACHE = {}

class Tracker:
    def __init__(self):
        self.global_tracker = None
        if not cfg.disable_tracker:
            self.global_tracker = sv.ByteTrack()

    def update(self, detections):
        if self.global_tracker is not None:
            return self.global_tracker.update_with_detections(detections)
        return detections

tracker = Tracker()

def pad_to_alignment(arr, alignment=ALIGNMENT):
    """
    Pad an image array so that its dimensions are multiples of a given alignment.
    Uses a global cache to store the padded dimensions for images with identical shapes.
    """
    key = (arr.shape[0], arr.shape[1], arr.shape[2])
    if key in PADDING_CACHE:
        new_height, new_width = PADDING_CACHE[key]
    else:
        new_height = ((arr.shape[0] + alignment - 1) // alignment) * alignment
        new_width = ((arr.shape[1] + alignment - 1) // alignment) * alignment
        PADDING_CACHE[key] = (new_height, new_width)
    padded_arr = np.zeros((new_height, new_width, arr.shape[2]), dtype=arr.dtype)
    padded_arr[:arr.shape[0], :arr.shape[1], :] = arr
    return padded_arr

@torch.inference_mode()
def perform_detection(model, image):
    """
    Perform detection on the given image using the provided model.
    """
    # Ensure image is contiguous; if it already is contiguous, this is still a lightweight call.
    image_np = np.ascontiguousarray(image)

    if cfg.use_padding:
        image_np = pad_to_alignment(image_np)

    if image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)

    results = model.predict(
        source=image_np,
        imgsz=cfg.ai_model_image_size,
        stream=True,
        conf=cfg.ai_conf,
        iou=IOU_THRESHOLD,
        device=cfg.ai_device,
        save=False,
        show=False,
        half=True,
        max_det=MAX_DETECTIONS,
        agnostic_nms=False,
        augment=True,
        vid_stride=False,
        visualize=False,
        verbose=True,
        show_boxes=False,
        show_labels=False,
        show_conf=False
    )
    # Process only the first detection (assuming it is the desired behavior)
    for result in results:
        detections = sv.Detections.from_ultralytics(result)
        tracked_detections = tracker.update(detections)
        if tracked_detections is not None:
            return tracked_detections
    return sv.Detections.empty()

def draw_trajectory(image, detections):
    for track in detections.tracks:
        if track.trace and len(track.trace) > 1:
            points = track.trace[-2:]
            p0 = (int(points[0][0]), int(points[0][1]))
            p1 = (int(points[1][0]), int(points[1][1]))
            cv2.arrowedLine(image, p0, p1, ARROW_COLOR, 2)

def run_bot(context):
    run_checks()

    try:
        model_path = f"models/{cfg.ai_model_name}"
        model = YOLO(model_path, task="detect")
    except Exception as e:
        log_error("Error loading AI model", e)
        raise

    while True:
        try:
            image = context.capture.get_new_frame()
            if image is not None:
                if cfg.show_window or cfg.show_overlay:
                    context.visuals.queue.put((image, sv.Detections.empty()))  # Default to empty detections if needed

                if context.hotkeys_watcher.app_pause != 0:
                    time.sleep(0.5)
                    continue

                result = perform_detection(model, image)
                if result is None:
                    result = sv.Detections.empty()  # Provide a default empty detection if None

                context.frame_parser.parse(result)
                if cfg.show_window or cfg.show_overlay:
                    context.visuals.queue.put((image, result))
                    context.overlay.queue.put((image, result))

                if (cfg.show_window or cfg.show_overlay) and cfg.show_trajectory:
                    draw_trajectory(image, result)
            else:
                time.sleep(0.08)
        except Exception as e:
            log_error("Error in main loop", e)
            time.sleep(1)

def restart_bot(context):
    """Restart the entire application"""
    print("Restarting bot...")
    try:
        context.cleanup()
        python = sys.executable
        args = sys.argv[:]
        subprocess.Popen([python] + args)
        os._exit(0)
    except Exception as e:
        log_error("Error during restart", e)
        os._exit(1)

def main():
    context = None
    try:
        context = get_app_context()
        cfg.set_restart_callback(restart_bot)
        context.tracker = Tracker()
        run_bot(context)
    except KeyboardInterrupt:
        if context:
            context.cleanup()
        sys.exit(0)
    except Exception as e:
        log_error("Fatal error", e)
        if context:
            context.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()