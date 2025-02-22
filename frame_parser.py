import math
import numpy as np
import cupy as cp
import supervision as sv
from config_watcher import cfg
from utils import get_torch_device, log_error

HEADSHOT_CLASS = 7
HEADSHOT_WEIGHT = 0.5
NON_HEADSHOT_WEIGHT = 1.0
MIN_DETECTIONS_FOR_CUDA = 0

class Target:
    def __init__(self, x1, y1, x2, y2, cls, id=None, body_y_offset=0.0):
        self.x = x1
        self.y = y1 if cls == HEADSHOT_CLASS else y1 - body_y_offset * (y2 - y1)
        self.w = x2 - x1
        self.h = y2 - y1
        self.cls = cls
        self.id = id
        self.center_x = self.x + self.w / 2
        self.center_y = self.y + self.h / 2

class FrameParser:
    def __init__(self, context):
        self.context = context
        self.device = get_torch_device(cfg)  # Kept for compatibility, though CuPy manages its own device
        self.current_locked_target_id = None
        self.previous_target = None
        self.body_y_offset = float(cfg.body_y_offset) if isinstance(getattr(cfg, 'body_y_offset', 0.0), (int, float)) else 0.0
        self.switch_threshold = float(getattr(cfg, 'switch_threshold', 1.0))
        self.smoothing_factor = float(getattr(cfg, 'smoothing_factor', 0.999))
        self.screen_width = int(getattr(cfg, 'detection_window_width', 1920))
        self.screen_height = int(getattr(cfg, 'detection_window_height', 1080))
        self.min_detections_for_cuda = MIN_DETECTIONS_FOR_CUDA
        self.active_classes = self.context.hotkeys_watcher.active_classes()
        self.cls_model_data = {
            0: 'player', 1: 'bot', 2: 'weapon', 3: 'outline', 4: 'dead_body',
            5: 'hideout_target_human', 6: 'hideout_target_balls', 7: 'head',
            8: 'smoke', 9: 'fire', 10: 'third_person'
        }

    def sort_targets(self, frame):
        if not (hasattr(frame, "xyxy") or (hasattr(frame, "boxes") and hasattr(frame.boxes, "xywh"))):
            return None

        if isinstance(frame, sv.Detections):
            boxes_array, classes_array, ids_array = self._convert_sv_to_numpy(frame)
            confidence_array = frame.confidence
        else:
            boxes_array = frame.boxes.xywh.cpu().numpy()
            classes_array = frame.boxes.cls.cpu().numpy()
            confidence_array = frame.boxes.conf.cpu().numpy()
            ids_array = (frame.boxes.id.cpu().numpy() if frame.boxes.id is not None 
                         else np.zeros_like(classes_array))

        if classes_array.size == 0:
            return None

        weights = np.ones(boxes_array.shape[0], dtype=np.float32)
        if not cfg.disable_headshot:
            mask = classes_array == HEADSHOT_CLASS
            weights[mask] *= HEADSHOT_WEIGHT
            weights[~mask] *= NON_HEADSHOT_WEIGHT
        else:
            weights[classes_array == HEADSHOT_CLASS] *= NON_HEADSHOT_WEIGHT
        weights *= (1 + confidence_array)

        if boxes_array.shape[0] < self.min_detections_for_cuda:
            return self._find_nearest_target_cpu(boxes_array, classes_array, ids_array, weights)
        else:
            return self._find_nearest_target_gpu(boxes_array, classes_array, ids_array, weights)

    def _convert_sv_to_numpy(self, frame):
        xyxy = frame.xyxy
        xywh = np.column_stack([
            (xyxy[:, 0] + xyxy[:, 2]) / 2,
            (xyxy[:, 1] + xyxy[:, 3]) / 2,
            xyxy[:, 2] - xyxy[:, 0],
            xyxy[:, 3] - xyxy[:, 1]
        ]).astype(np.float32)
        classes_array = frame.class_id.astype(np.float32)
        ids_array = (frame.tracker_id.astype(np.int32) if frame.tracker_id is not None 
                    else np.zeros_like(classes_array))
        return xywh, classes_array, ids_array

    def _find_nearest_target_cpu(self, boxes_array, classes_array, ids_array, weights):
        center = np.array([self.screen_width / 2, self.screen_height / 2])
        cx = boxes_array[:, 0] + boxes_array[:, 2] / 2
        cy = boxes_array[:, 1] + boxes_array[:, 3] / 2
        distances = np.hypot(cx - center[0], cy - center[1]) * weights
        best_idx = np.argmin(distances)
        if distances[best_idx] == np.inf:
            return None
        return Target(
            boxes_array[best_idx, 0],
            boxes_array[best_idx, 1],
            boxes_array[best_idx, 0] + boxes_array[best_idx, 2],
            boxes_array[best_idx, 1] + boxes_array[best_idx, 3],
            classes_array[best_idx],
            ids_array[best_idx],
            body_y_offset=self.body_y_offset
        )

    def _find_nearest_target_gpu(self, boxes_array, classes_array, ids_array, weights):
        """GPU-accelerated nearest target search using CuPy (no JIT)."""
        # Move data to GPU
        boxes = cp.asarray(boxes_array)
        classes = cp.asarray(classes_array)
        ids = cp.asarray(ids_array)
        weights = cp.asarray(weights)
        center = cp.array([self.screen_width / 2, self.screen_height / 2], dtype=cp.float32)

        # Compute centers
        cx = boxes[:, 0] + boxes[:, 2] / 2
        cy = boxes[:, 1] + boxes[:, 3] / 2

        # Compute squared distances
        dx = cx - center[0]
        dy = cy - center[1]
        distance_sq = dx * dx + dy * dy

        # Apply weights based on prioritize_headshot
        if not cfg.disable_headshot:
            head_mask = classes == HEADSHOT_CLASS
            distance_sq = cp.where(head_mask, distance_sq * HEADSHOT_WEIGHT, distance_sq * NON_HEADSHOT_WEIGHT)
        else:
            area = boxes[:, 2] * boxes[:, 3]
            distance_sq = weights * (distance_sq / area.clip(min=1e-6))  # Avoid division by zero

        # Find minimum distance index
        best_idx = cp.argmin(distance_sq)
        if distance_sq[best_idx] == float('inf'):
            return self._find_nearest_target_cpu(boxes_array, classes_array, ids_array, weights)

        # Move result back to CPU
        best_idx = best_idx.get()  # Transfer scalar to host
        target_info = (
            boxes_array[best_idx, 0],
            boxes_array[best_idx, 1],
            boxes_array[best_idx, 0] + boxes_array[best_idx, 2],
            boxes_array[best_idx, 1] + boxes_array[best_idx, 3],
            classes_array[best_idx],
            ids_array[best_idx]
        )
        return Target(*target_info, body_y_offset=self.body_y_offset)

    def parse(self, result):
        if isinstance(result, sv.Detections):
            self._process_sv_detections(result)
        else:
            self._process_yolo_detections(result)

    def _process_sv_detections(self, detections):
        if detections.xyxy.size > 0:
            target = self.sort_targets(detections)
            self._handle_target(target, detections)

    def _process_yolo_detections(self, results):
        for frame in results:
            if hasattr(frame, "boxes") and frame.boxes:
                target = self.sort_targets(frame)
                self._handle_target(target, frame)

    def _handle_target(self, target, frame):
        if not target or target.cls not in self.active_classes:
            self.current_locked_target_id = None
            self.previous_target = None
            return
        
        if self.current_locked_target_id is None:
            self._switch_target(target, None)
            return
        
        locked_idx = self._get_locked_index(frame.tracker_id)
        if locked_idx == -1:
            self._switch_target(target, None)
            return
        
        box = frame.xyxy[locked_idx]
        center_x, center_y, cls, id = self._extract_target_info(frame, locked_idx, box)
        current_pos = np.array([center_x, center_y])
        prev_pos = np.array([self.previous_target.center_x, self.previous_target.center_y]) if self.previous_target else current_pos
        
        if np.linalg.norm(current_pos - prev_pos) > self.switch_threshold:
            new_target = Target(*box, cls, id, body_y_offset=self.body_y_offset)
            self._switch_target(new_target, None)
            return
        
        if cls == 0:  # Player
            mask = (frame.class_id == HEADSHOT_CLASS) & (frame.tracker_id == id)
            if np.any(mask):
                head_idx = np.where(mask)[0][0]
                head_target = Target(*frame.xyxy[head_idx], HEADSHOT_CLASS, frame.tracker_id[head_idx], 
                                   body_y_offset=self.body_y_offset)
                self.context.mouse.process_data((head_target.center_x, head_target.center_y, head_target.w, 
                                               head_target.h, HEADSHOT_CLASS, head_target.id))
                self.context.visuals.draw_aim_point(head_target.center_x, head_target.center_y)
                self.previous_target = head_target
                self.current_locked_target_id = head_target.id
                return
        
        self.context.mouse.process_data((center_x, center_y, box[2] - box[0], box[3] - box[1], cls, id))
        self.context.visuals.draw_aim_point(center_x, center_y)
        self.previous_target = Target(*box, cls, id, body_y_offset=self.body_y_offset)

    def _get_locked_index(self, tracker_ids):
        try:
            return int(np.where(tracker_ids == self.current_locked_target_id)[0][0])
        except (IndexError, ValueError):
            return -1

    def _extract_target_info(self, frame, locked_idx, box):
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        if frame.class_id[locked_idx] != HEADSHOT_CLASS:
            center_y -= self.body_y_offset * (box[3] - box[1])
        cls = frame.class_id[locked_idx]
        id = frame.tracker_id[locked_idx]
        return center_x, center_y, cls, id

    def _switch_target(self, new_target, old_target):
        if old_target is None or new_target.id != old_target.id:
            self.current_locked_target_id = new_target.id
            self.context.mouse.process_data((new_target.center_x, new_target.center_y, new_target.w, 
                                           new_target.h, new_target.cls, new_target.id))
        self.previous_target = new_target

