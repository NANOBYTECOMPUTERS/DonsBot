# init.py ---

from config_watcher import cfg
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
            
            # Initialize modules that need context
            self.visuals = Visuals(self)
            self.frame_parser = FrameParser(self)
            self.mouse = MouseThread(self)
            self.shooting = Shooting(self)
            self.shooting.start()
            
            # Initialize hotkeys last (needs access to all other modules)
            self.hotkeys_watcher = HotkeysWatcher(self)
            
            log_error("Application initialization completed successfully")
        except Exception as e:
            log_error("Failed to initialize application", e)
            self.cleanup()
            raise

    def cleanup(self):
        """Clean up all initialized modules"""
        cleanup_resources(self.capture, self.visuals, self.overlay)
        if self.shooting and self.shooting.is_alive():
            self.shooting.queue.put((False, False))

def get_app_context():
    """Return initialized application context"""
    context = AppContext()
    context.initialize()
    return context