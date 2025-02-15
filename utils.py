# utils.py
import logging
import time
import cv2
import torch

class CudaLogFilter(logging.Filter):
    """Filter to suppress CUDA memory deallocation logs"""
    def filter(self, record):
        msg = record.getMessage()
        return not ("dealloc: cuMemFree_v2" in msg or "add pending dealloc: cuMemFree_v2" in msg)

def setup_logging():
    """Set up logging configuration with CUDA log filtering"""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add filter to suppress CUDA deallocation logs
    file_handler.addFilter(CudaLogFilter())

    # Clear any existing handlers and add our handler
    logger.handlers = [file_handler]

    # Reduce verbosity of external libraries
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)

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

def get_torch_device(cfg_obj):
    """Select the torch device based on configuration"""
    try:
        ai_device = str(cfg_obj.ai_device).strip().lower()
        if cfg_obj.ai_enable_amd:
            return torch.device(f'hip:{ai_device}')
        elif 'cpu' in ai_device:
            return torch.device('cpu')
        elif ai_device.isdigit():
            return torch.device(f'cuda:{ai_device}')
        return torch.device('cuda:0')
    except Exception:
        return torch.device('cpu')