import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv
from config_watcher import cfg
from capture import capture
from visual import visuals
from frame_parser import frame_parser
from hotkeys_watcher import hotkeys_watcher
from checks import run_checks


IOU_THRESHOLD = 0.5
MAX_DETECTIONS = 20
ARROW_COLOR = (0, 255, 0)
ALIGNMENT = 128

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

def pad_to_alignment(arr, alignment=ALIGNMENT, padded_dims_cache=None):
    if padded_dims_cache is None:
        padded_dims_cache = {}
    key = (arr.shape[0], arr.shape[1], arr.shape[2])
    if key in padded_dims_cache:
        new_height, new_width = padded_dims_cache[key]
    else:
        new_height = ((arr.shape[0] + alignment - 1) // alignment) * alignment
        new_width = ((arr.shape[1] + alignment - 1) // alignment) * alignment
        padded_dims_cache[key] = (new_height, new_width)
    padded_arr = np.zeros((new_height, new_width, arr.shape[2]), dtype=arr.dtype)
    padded_arr[:arr.shape[0], :arr.shape[1], :] = arr
    return padded_arr

@torch.inference_mode()
def perform_detection(model, image, use_engine):
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
        half=False,
        max_det=MAX_DETECTIONS,
        agnostic_nms=False,
        augment=False,
        vid_stride=False,
        visualize=False,
        verbose=True,
        show_boxes=False,
        show_labels=False,
        show_conf=False
    )
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

def main():
    run_checks()
    try:
        model_path = f"models/{cfg.ai_model_name}"
        model = YOLO(model_path, task="detect")
        use_engine = model_path.endswith('.engine')
    except Exception as e:
        print(f"An error occurred when loading the AI model: {str(e)}")
        raise

    while True:
        image = capture.get_new_frame()
        if image is not None:
            if cfg.show_window or cfg.show_overlay:
                visuals.queue.put(image)
            result = perform_detection(model, image, use_engine)
            if hotkeys_watcher.app_pause == 0:
                if result is not None:
                    frame_parser.parse(result)
                    if (cfg.show_window or cfg.show_overlay) and cfg.show_trajectory:
                        draw_trajectory(image, result)
                else:
                    frame_parser.parse(sv.Detections.empty())
        else:
            time.sleep(0.01)

if __name__ == "__main__":
    main()