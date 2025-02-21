# utils.py ---

import logging
import logging.handlers
import queue
import time
import cv2
import torch

class CudaLogFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        # Add mouse events to filtered messages
        return not ("dealloc: cuMemFree_v2" in msg or 
                   "add pending dealloc: cuMemFree_v2" in msg or
                   "mouse_event" in msg.lower())


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.handlers.QueueHandler(queue.Queue(-1))  # Unbounded queue
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    listener = logging.handlers.QueueListener(handler.queue, logging.FileHandler('app.log'))
    listener.start()
    
def log_error(message, exception=None):
    """Log an error message"""
    logging.error(f"{message}: {str(exception)}" if exception else message)

def cleanup_resources(capture_obj, visuals_obj, overlay_obj):
    """Clean up all application resources"""
    try:
        if capture_obj:
            capture_obj.quit()
        if visuals_obj and hasattr(visuals_obj, 'queue'):
            visuals_obj.queue.put(None)
            time.sleep(0.1)
            visuals_obj.cleanup()
        if overlay_obj and hasattr(overlay_obj, 'root') and overlay_obj.root:
            overlay_obj.root.destroy()
        cv2.destroyAllWindows()
    except Exception as e:
        log_error("Error during cleanup", e)

_torch_device_cache = None
def get_torch_device(cfg_obj):
    global _torch_device_cache
    if _torch_device_cache is None:
        ai_device = str(cfg_obj.ai_device).strip().lower()
        if cfg_obj.ai_enable_amd:
            _torch_device_cache = torch.device(f'hip:{ai_device}')
        elif 'cpu' in ai_device:
            _torch_device_cache = torch.device('cpu')
        elif ai_device.isdigit():
            _torch_device_cache = torch.device(f'cuda:{ai_device}')
        else:
            _torch_device_cache = torch.device('cuda:0')
    return _torch_device_cache