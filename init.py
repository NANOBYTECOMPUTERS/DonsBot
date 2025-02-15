from utils import setup_logging, get_torch_device, cleanup_resources, log_error  # Consider replacing log_error with logger.info/info 
from capture import Capture
from visual import Visuals
from frame_parser import FrameParser
from hotkeys_watcher import HotkeysWatcher
from shooting import Shooting
from overlay import Overlay
from mouse import MouseThread
import logging

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
            logging.info("Starting application initialization")
            
            # Initialize capture first (no context needed)
            self.capture = Capture()
            self.capture.start()
            
            # Initialize overlay (no context needed in current implementation)
            self.overlay = Overlay()
            
            # Initialize modules that might depend on context
            self.visuals = Visuals(self)
            self.frame_parser = FrameParser(self)
            self.mouse = MouseThread(self)
            
            # Initialize and start the shooting thread
            self.shooting = Shooting(self)
            self.shooting.start()
            
            # Initialize hotkeys last (needs access to all other modules)
            self.hotkeys_watcher = HotkeysWatcher(self)
            
            logging.info("Application initialization completed successfully")
        except Exception as e:
            logging.error("Failed to initialize application", exc_info=True)
            self.cleanup()
            raise

    def cleanup(self):
        """Clean up all initialized modules"""
        # Clean up resources (assuming cleanup_resources handles None values safely)
        cleanup_resources(self.capture, self.visuals, self.overlay)
        
        # Signal the shooting thread to stop and wait for it to finish, if it exists
        if self.shooting is not None and self.shooting.is_alive():
            # Send shutdown message, assuming the shooting thread is listening on its queue
            self.shooting.queue.put((False, False))
            # Optionally join the shooting thread to ensure it shuts down before cleanup completes
            self.shooting.join(timeout=5)
            if self.shooting.is_alive():
                logging.warning("Shooting thread did not terminate properly.")

def get_app_context():
    """Return initialized application context"""
    context = AppContext()
    context.initialize()
    return context