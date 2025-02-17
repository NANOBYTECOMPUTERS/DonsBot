# rzctl.py ---
import ctypes

def enum(**enums):
    return type('Enum', (), enums)
MOUSE_CLICK = enum(
    LEFT_DOWN=1,
    LEFT_UP=2,
    RIGHT_DOWN=4,
    RIGHT_UP=8,
    SCROLL_CLICK_DOWN=16,
    SCROLL_CLICK_UP=32,
    BACK_DOWN=64,
    BACK_UP=128,
    FOWARD_DOWN=256,
    FOWARD_UP=512,
    SCROLL_DOWN=4287104000,
    SCROLL_UP=7865344
)

KEYBOARD_INPUT_TYPE = enum(
    KEYBOARD_DOWN=0,
    KEYBOARD_UP=1
)

class RZControl:
    def __init__(self, dll_path):
        """
        Initialize RZCONTROL with the specified DLL path.

        :param dll_path: Path to the DLL file
        """
        try:
            self.dll = ctypes.WinDLL(dll_path)
            self._setup_functions()
        except OSError as e:
            raise RuntimeError(f"Failed to load DLL from {dll_path}: {e}")

    def _setup_functions(self):
        """Set up the function signatures for DLL methods."""
        self.init = self.dll.init
        self.init.argtypes = []
        self.init.restype = ctypes.c_bool

        self.mouse_move = self.dll.mouse_move
        self.mouse_move.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_bool]
        self.mouse_move.restype = None

        self.mouse_click = self.dll.mouse_click
        self.mouse_click.argtypes = [ctypes.c_int]
        self.mouse_click.restype = None

        self.keyboard_input = self.dll.keyboard_input
        self.keyboard_input.argtypes = [ctypes.c_short, ctypes.c_int]
        self.keyboard_input.restype = None

    def init(self):
        """Initialize the RZCONTROL device.

        Finds the symbolic link that contains name RZCONTROL and opens a handle to the respective device.

        :return: Boolean indicating if a valid device handle was obtained.
        """
        if not self.init():
            raise RuntimeError("Failed to initialize RZCONTROL device.")
        return True

    def mouse_move(self, x, y, from_start_point):
        """
        Move the mouse cursor.

        If `from_start_point` is True, x and y are offsets from the current position.
        Otherwise, x and y are absolute positions on the screen, in the range of 1 to 65536.
        Note: x and/or y cannot be 0 unless moving from the start point.

        :param x: Integer for X-coordinate
        :param y: Integer for Y-coordinate
        :param from_start_point: Boolean indicating if movement is relative to the current position
        """
        if not from_start_point and (x == 0 or y == 0):
            raise ValueError("When not from start point, x and y cannot be 0.")
        self.mouse_move(x, y, from_start_point)

    def mouse_click(self, click_mask):
        """
        Perform a mouse click based on the click mask provided.

        :param click_mask: An integer from MOUSE_CLICK enum representing the type of click
        """
        self.mouse_click(click_mask)

    def keyboard_input(self, scan_code, up_down):
        """
        Send keyboard input.

        :param scan_code: Scan code of the key (short type)
        :param up_down: 0 for key down, 1 for key up from KEYBOARD_INPUT_TYPE enum
        """
        self.keyboard_input(scan_code, up_down)