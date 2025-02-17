import math
import numpy as np
import torch
from numba import cuda
from numba.cuda.cudadrv.error import CudaDriverError
import supervision as sv
from config_watcher import cfg
from utils import get_torch_device, log_error

HEADSHOT_CLASS = 7
DEFAULT_COLOR = sv.ColorPalette.DEFAULT
DEFAULT_TEXT_COLOR = sv.Color.WHITE
DEFAULT_TEXT_SCALE = 0.5
DEFAULT_TEXT_THICKNESS = 1
DEFAULT_TEXT_PADDING = 5
DEFAULT_TEXT_POSITION = sv.Position.TOP_LEFT
HEADSHOT_WEIGHT = 0.1  # Reduced to prioritize headshots
NON_HEADSHOT_WEIGHT = 1.0
MIN_DETECTIONS_FOR_CUDA = 2

class Target:
    def __init__(self, x1, y1, x2, y2, cls, id=None):
        self.x = x1
        offset = cfg.body_y_offset if isinstance(getattr(cfg, 'body_y_offset', None), (int, float)) else 0.0
        self.y = y1 if cls == HEADSHOT_CLASS else y1 - offset * (y2 - y1)
        self.w = x2 - x1
        self.h = y2 - y1
        self.cls = cls
        self.id = id
        self.center_x = self.x + self.w // 2
        self.center_y = self.y + self.h // 2

@cuda.jit
def _find_nearest_target_cuda(boxes_array, classes_array, ids_array, center, weights, prioritize_headshot, result):
    idx = cuda.grid(1)
    if idx < boxes_array.shape[0]:
        cx = boxes_array[idx, 0] + boxes_array[idx, 2] / 2.0
        cy = boxes_array[idx, 1] + boxes_array[idx, 3] / 2.0
        dx = cx - center[0]
        dy = cy - center[1]
        distance_sq = dx * dx + dy * dy

        if prioritize_headshot:
            if classes_array[idx] == HEADSHOT_CLASS:
                distance_sq *= HEADSHOT_WEIGHT  # Headshots are considered closer
            else:
                distance_sq *= NON_HEADSHOT_WEIGHT
        else:
            area = boxes_array[idx, 2] * boxes_array[idx, 3]
            if area > 0:
                distance_sq = weights[idx] * (distance_sq / area)

        old = cuda.atomic.min(result, 0, distance_sq)
        if distance_sq < old:
            result[1] = ids_array[idx]

class FrameParser:
    def __init__(self, context):
        self.context = context
        self.device = get_torch_device(cfg)
        self.current_locked_target_id = None
        self.previous_target = None
        self.switch_threshold = getattr(cfg, 'switch_threshold', 1.0)
        self.smoothing_factor = getattr(cfg, 'smoothing_factor', 0.999)
        self.screen_width = getattr(cfg, 'detection_window_width', 1920)
        self.screen_height = getattr(cfg, 'detection_window_height', 1080)
        self.bounding_box_annotator = sv.BoxAnnotator(
            color=DEFAULT_COLOR,
            thickness=2,
            color_lookup=sv.ColorLookup.CLASS
        )
        self.label_annotator = sv.LabelAnnotator(
            color=DEFAULT_COLOR,
            text_color=DEFAULT_TEXT_COLOR,
            text_scale=DEFAULT_TEXT_SCALE,
            text_thickness=DEFAULT_TEXT_THICKNESS,
            text_padding=DEFAULT_TEXT_PADDING,
            text_position=DEFAULT_TEXT_POSITION,
            color_lookup=sv.ColorLookup.CLASS
        )
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
        self.player_head_map = {}

    def parse(self, result):
        if isinstance(result, sv.Detections):
            self._process_sv_detections(result)
        else:
            self._process_yolo_detections(result)

    def _process_sv_detections(self, detections):
        if detections.xyxy.size > 0:
            target = self.sort_targets(detections)
            self._handle_target(target, detections)
            self._annotate_frame(detections)

    def _process_yolo_detections(self, results):
        for frame in results:
            if hasattr(frame, "boxes") and frame.boxes:
                target = self.sort_targets(frame)
                self._handle_target(target, frame)
                self._annotate_frame(frame)

    def _handle_target(self, target, frame):
        if target:
            if self.context.hotkeys_watcher.clss is None:
                self.context.hotkeys_watcher.clss = self.context.hotkeys_watcher.active_classes()
            
            if target.cls in self.context.hotkeys_watcher.clss:
                if self.current_locked_target_id is not None:
                    tracker_ids = frame.tracker_id
                    locked_idx = self._get_locked_index(tracker_ids)
                    
                    if locked_idx != -1:
                        box = frame.xyxy[locked_idx]
                        center_x, center_y, cls, id = self._extract_target_info(frame, locked_idx, box)
                        current_pos = np.array([center_x, center_y])
                        prev_pos = np.array([self.previous_target.x, self.previous_target.y]) if self.previous_target else current_pos
                        
                        if np.linalg.norm(current_pos - prev_pos) <= self.switch_threshold:
                            if cls == 0:  # If the target is a player
                                # Check for associated head
                                head_targets = [t for t in zip(frame.xyxy, frame.class_id, frame.tracker_id) if t[1] == HEADSHOT_CLASS and t[2] == id]
                                if head_targets:
                                    head_target = Target(*head_targets[0][0], HEADSHOT_CLASS, head_targets[0][2])
                                    self.context.mouse.process_data((head_target.center_x, head_target.center_y, head_target.w, head_target.h, HEADSHOT_CLASS, head_target.id))
                                    self.context.visuals.draw_aim_point(head_target.center_x, head_target.center_y)
                                    self.previous_target = head_target
                                    self.current_locked_target_id = head_target.id
                                else:
                                    # No head detected, keep the player body
                                    self.context.mouse.process_data((center_x, center_y, box[2] - box[0], box[3] - box[1], cls, id))
                                    self.context.visuals.draw_aim_point(center_x, center_y)
                                    self.previous_target = Target(*box, cls, id)
                            else:
                                # Use the target data directly if not a player
                                self.context.mouse.process_data((center_x, center_y, box[2] - box[0], box[3] - box[1], cls, id))
                                self.context.visuals.draw_aim_point(center_x, center_y)
                                self.previous_target = Target(*box, cls, id)
                        else:
                            # Target has moved too far, potentially switch to a new one
                            new_target_info = Target(*box, cls, id)
                            self.context.mouse.process_data((center_x, center_y, box[2] - box[0], box[3] - box[1], cls, id))
                            self.current_locked_target_id = id
                            self.previous_target = new_target_info
                    else:
                        self._switch_target(target, None)
                else:
                    self._switch_target(target, None)
            else:
                self.current_locked_target_id = None
        else:
            self.current_locked_target_id = None
            self.previous_target = None

    def _get_locked_index(self, tracker_ids):
        try:
            return int(np.where(tracker_ids == self.current_locked_target_id)[0][0])
        except (IndexError, ValueError):
            return -1

    def _extract_target_info(self, frame, locked_idx, box):
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        if frame.class_id[locked_idx] != HEADSHOT_CLASS:
            center_y -= cfg.body_y_offset * (box[3] - box[1])
        cls = frame.class_id[locked_idx]
        id = frame.tracker_id[locked_idx]
        return center_x, center_y, cls, id

    def _switch_target(self, new_target, old_target):
        if old_target is None or new_target.id != old_target.id:
            self.current_locked_target_id = new_target.id
            self.context.mouse.process_data((new_target.x, new_target.y, new_target.w, new_target.h, new_target.cls, new_target.id))
        self.previous_target = new_target

    def _annotate_frame(self, frame):
        pass

    def _draw_boxes_and_info(self, frame):
        pass

    def sort_targets(self, frame):
        # Validate input structure early
        if hasattr(frame, "xyxy") and hasattr(frame, "class_id"):
            if isinstance(frame, sv.Detections):
                boxes_array, classes_array, ids_array = self._convert_sv_to_numpy(frame)
                confidence_array = frame.confidence
            else:
                if not (hasattr(frame, "boxes") and hasattr(frame.boxes, "xywh") and hasattr(frame.boxes, "cls")):
                    return None
                boxes_array = np.ascontiguousarray(frame.boxes.xywh.cpu().numpy())
                classes_array = np.ascontiguousarray(frame.boxes.cls.cpu().numpy())
                confidence_array = frame.boxes.conf.cpu().numpy()
                ids_array = np.ascontiguousarray(
                    frame.boxes.id.cpu().numpy() if frame.boxes.id is not None 
                    else np.zeros_like(classes_array)
                )
        else:
            return None

        if classes_array.size == 0:
            return None

        # If number of detections is small, use CPU path to avoid overhead
        if boxes_array.shape[0] < MIN_DETECTIONS_FOR_CUDA:
            weights = np.ones(boxes_array.shape[0], dtype=np.float32)
            if not cfg.disable_headshot:
                weights[classes_array == HEADSHOT_CLASS] *= 0.5
                weights *= (1 + confidence_array)
            else:
                weights[classes_array == HEADSHOT_CLASS] *= NON_HEADSHOT_WEIGHT
                weights *= (1 + confidence_array)
            return self._find_nearest_target_cpu(boxes_array, classes_array, ids_array, weights)

        # Compute weight vector once and reuse it in the CUDA call
        weights = np.ones(boxes_array.shape[0], dtype=np.float32)
        if not cfg.disable_headshot:
            weights[classes_array != HEADSHOT_CLASS] = HEADSHOT_WEIGHT
        else:
            weights[classes_array == HEADSHOT_CLASS] = NON_HEADSHOT_WEIGHT

        # Ensure correct data types and contiguity
        boxes_array = np.ascontiguousarray(boxes_array.astype(np.float32))
        classes_array = np.ascontiguousarray(classes_array.astype(np.float32))
        ids_array = np.ascontiguousarray(ids_array.astype(np.float32))
        weights = np.ascontiguousarray(weights)

        return self._find_nearest_target_cuda_wrapper(boxes_array, classes_array, ids_array, weights)

    def _convert_sv_to_numpy(self, frame):
        xyxy = frame.xyxy
        xywh = np.column_stack([
            (xyxy[:, 0] + xyxy[:, 2]) / 2,
            (xyxy[:, 1] + xyxy[:, 3]) / 2,
            xyxy[:, 2] - xyxy[:, 0],
            xyxy[:, 3] - xyxy[:, 1]
        ])
        classes_array = frame.class_id.astype(np.float32)
        ids_array = frame.tracker_id.astype(np.int32) if frame.tracker_id is not None else np.zeros_like(classes_array)
        return xywh, classes_array, ids_array

    def _find_nearest_target_cuda_wrapper(self, boxes_array, classes_array, ids_array, weights):
        center = np.array([self.screen_width / 2, self.screen_height / 2], dtype=np.float32)

        d_boxes = cuda.to_device(boxes_array)
        d_classes = cuda.to_device(classes_array)
        d_ids = cuda.to_device(ids_array)
        d_center = cuda.to_device(center)
        d_weights = cuda.to_device(weights)

        init_result = np.array([np.inf, -1], dtype=np.float32)
        d_result = cuda.to_device(init_result)

        threadsperblock = min(1024, boxes_array.shape[0])
        blockspergrid = (boxes_array.shape[0] + threadsperblock - 1) // threadsperblock

        try:
            _find_nearest_target_cuda[blockspergrid, threadsperblock](
                d_boxes, d_classes, d_ids, d_center, d_weights, (not cfg.disable_headshot), d_result
            )
            result = d_result.copy_to_host()
        except CudaDriverError as e:
            log_error("CUDA error in target selection", e)
            return self._find_nearest_target_cpu(boxes_array, classes_array, ids_array, weights)

        if result[0] == np.inf:
            if ids_array.size > 0:
                first_detection = (
                    boxes_array[0, 0],
                    boxes_array[0, 1],
                    boxes_array[0, 0] + boxes_array[0, 2],
                    boxes_array[0, 1] + boxes_array[0, 3],
                    classes_array[0],
                    ids_array[0]
                )
                return Target(*first_detection)
            return None

        nearest_id = int(result[1])
        nearest_idx = np.where(ids_array == nearest_id)[0][0]
        target_info = (
            boxes_array[nearest_idx, 0],
            boxes_array[nearest_idx, 1],
            boxes_array[nearest_idx, 0] + boxes_array[nearest_idx, 2],
            boxes_array[nearest_idx, 1] + boxes_array[nearest_idx, 3],
            classes_array[nearest_idx],
            nearest_id
        )
        return Target(*target_info)

    def _find_nearest_target_cpu(self, boxes_array, classes_array, ids_array, weights):
        center = np.array([self.screen_width / 2, self.screen_height / 2], dtype=np.float32)
        cx = boxes_array[:, 0] + boxes_array[:, 2] / 2.0
        cy = boxes_array[:, 1] + boxes_array[:, 3] / 2.0
        dx = cx - center[0]
        dy = cy - center[1]
        distance_sq = dx ** 2 + dy ** 2

        if not cfg.disable_headshot:
            distance_sq = np.where(classes_array == HEADSHOT_CLASS, distance_sq * HEADSHOT_WEIGHT, distance_sq * NON_HEADSHOT_WEIGHT)
        else:
            area = boxes_array[:, 2] * boxes_array[:, 3]
            distance_sq = np.where(area > 0, distance_sq / area, distance_sq)

        best_index = np.argmin(distance_sq)
        if distance_sq[best_index] == np.inf:
            return None
        target_info = (
            boxes_array[best_index, 0],
            boxes_array[best_index, 1],
            boxes_array[best_index, 0] + boxes_array[best_index, 2],
            boxes_array[best_index, 1] + boxes_array[best_index, 3],
            classes_array[best_index],
            ids_array[best_index]
        )
        return Target(*target_info)