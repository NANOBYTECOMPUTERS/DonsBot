
from tkinter import Canvas
import tkinter as tk
import tkinter.font as tkFont
import threading
import queue
import numpy as np
import supervision as sv
from config_watcher import cfg

class Overlay:
    # Moved class_map to a class-level constant to avoid recreating it on every call.
    CLASS_MAP = {
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
    
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = None
        # Optionally, remove or adjust frame_skip_counter if processing on every update.
        self.frame_skip_counter = 0

    def run(self, width, height):
        if cfg.show_overlay:
            self.root = tk.Tk()
            self.root.overrideredirect(True)
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            x = (screen_width - width) // 2
            y = (screen_height - height) // 2
            self.root.geometry(f"{width}x{height}+{x}+{y}")
            self.root.attributes('-topmost', True)
            self.root.attributes('-transparentcolor', 'black')

            self.canvas = Canvas(self.root, bg='black', highlightthickness=0, cursor="none")
            self.canvas.pack(fill=tk.BOTH, expand=True)

            if cfg.overlay_show_borders:
                pass
                #if cfg.circle_capture:
                    #self.canvas.create_oval(0, 0, width, height, outline='red', width=2, tag="border")
               # else:
                    #self.canvas.create_rectangle(0, 0, width, height, outline='red', width=2, tag="border")
            
            self.process_queue()
            # Note: Ideally Tkinter mainloop should be run in the main thread.
            self.root.mainloop()

    def process_queue(self):
        # Optional: If you decide not to skip frames, comment out this counter-based logic.
        self.frame_skip_counter += 1
        update_interval = 16  # Approximately 60FPS

        # If you wish to process on every update, simply remove the frame_skip logic.
        if self.frame_skip_counter % 3 == 0:
            # Delete previous drawn elements.
            self.canvas.delete("drawn")
            while not self.queue.empty():
                command, args = self.queue.get()
                command(*args)
                
        self.root.after(update_interval, self.process_queue)

    def draw_detections(self, detections):
        if isinstance(detections, sv.Detections):
            # If possible, iterate directly over detections rather than using range and indexing.
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                self.queue.put((self._draw_square, (x1, y1, x2, y2, 'green', 2)))
                if cfg.overlay_show_labels or cfg.overlay_show_conf:
                    label = f"{self.get_class_name(detections.class_id[i])} {detections.confidence[i]:.2f}"
                    self.queue.put((self._draw_text, (x1, y1 - 10, label, 12, 'white')))
        else:
            print("Detections not in expected sv.Detections format")

    def get_class_name(self, class_id):
        # Use the class constant instead of recreating the dict.
        return self.CLASS_MAP.get(int(class_id), 'Unknown')

    def _draw_square(self, x1, y1, x2, y2, color='white', size=1):
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=size, tag="drawn")

    def draw_text(self, x, y, text, size, color):
        self.queue.put((self._draw_text, (x, y, text, size, color)))

    def _draw_text(self, x, y, text, size, color):
        self.canvas.create_text(x, y, text=text, font=('Arial', size), fill=color, tag="drawn")

    def draw_line(self, x1, y1, x2, y2, color='green', size=2):
        self.queue.put((self._draw_line, (x1, y1, x2, y2, color, size)))

    def _draw_line(self, x1, y1, x2, y2, color, size):
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=size, tag="drawn")

    def draw_circle(self, x, y, radius, color='red', size=1):
        self.queue.put((self._draw_circle, (x, y, radius, color, size)))

    def _draw_circle(self, x, y, radius, color, size):
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=color, width=size, tag="drawn")

    def show(self, width, height):
        # Consider running tk.Tk() mainloop on the main thread.
        if self.thread is None:
            self.thread = threading.Thread(target=self.run, args=(width, height), daemon=True, name="Overlay")
            self.thread.start()