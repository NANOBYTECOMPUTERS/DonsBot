import time
from config_watcher import cfg
import cv2
from utils import setup_logging, get_torch_device, cleanup_resources, log_error
from capture import Capture
from visual import Visuals
from frame_parser import FrameParser
from hotkeys_watcher import HotkeysWatcher
from shooting import Shooting
from overlay import Overlay
from mouse import MouseThread

class AppContext:
    """Container for all initialized modules"""
    def __init__(self):
        setup_logging()
        self.capture = None
        self.visuals = None
        self.frame_parser = None
        self.hotkeys_watcher = None
        self.shooting = None
        self.overlay = None
        self.mouse = None
        self.tracker = None

    def initialize(self):
        """Initialize all modules in the correct order"""
        try:
            log_error("Starting application initialization")
            
            # Initialize capture first (no context needed)
            self.capture = Capture()
            self.capture.start()
            
            # Initialize overlay (no context needed in current implementation)
            self.overlay = Overlay()
            
            # Initialize hotkeys_watcher before frame_parser (dependency)
            self.hotkeys_watcher = HotkeysWatcher(self)
            
            # Initialize modules that need context
            self.visuals = Visuals(self)
            self.frame_parser = FrameParser(self)  # Now hotkeys_watcher is ready
            self.mouse = MouseThread(self)
            self.shooting = Shooting(self)
            self.shooting.start()
            
            log_error("Application initialization completed successfully")
        except Exception as e:
            log_error("Failed to initialize application", e)
            self.cleanup()
            raise

    def cleanup(self):
        """Clean up all initialized modules"""
        try:
            if self.capture:
                self.capture.quit()
            if self.visuals and hasattr(self.visuals, 'queue'):
                self.visuals.queue.put(None)
                time.sleep(0.1)
                self.visuals.cleanup()
            if self.overlay and hasattr(self.overlay, 'root') and self.overlay.root:
                self.overlay.root.destroy()
            if self.shooting and self.shooting.is_alive():
                self.shooting.queue.put((False, False))
                self.shooting.join(timeout=1.0)
            if self.hotkeys_watcher:
                self.hotkeys_watcher.quit()  # Add quit method if missing
            cv2.destroyAllWindows()
            time.sleep(0.2)  # Give time for threads to settle
        except Exception as e:
            log_error(f"Error during cleanup: {e}")

def get_app_context():
    """Return initialized application context"""
    context = AppContext()
    context.initialize()
    return context