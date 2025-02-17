from tkinter import Canvas
import tkinter as tk
import threading
import cv2
import queue
import numpy as np
import supervision as sv
from config_watcher import cfg
from PIL import Image, ImageTk
from utils import log_error

class Overlay:
    def __init__(self):
        self.queue = queue.Queue(maxsize=5)  # Increased queue size for better performance
        self.thread = None
        self.frame_skip_counter = 0
        self.bounding_box_annotator = sv.BoxAnnotator(
            color=sv.ColorPalette.DEFAULT,
            thickness=2,
            color_lookup=sv.ColorLookup.CLASS
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.DEFAULT,
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1,
            text_padding=5,
            text_position=sv.Position.TOP_LEFT,
            color_lookup=sv.ColorLookup.CLASS
        )

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
                if cfg.circle_capture:
                    self.canvas.create_oval(0, 0, width, height, outline='red', width=2, tag="border")
                else:
                    self.canvas.create_rectangle(0, 0, width, height, outline='red', width=2, tag="border")
            self.process_queue()
            self.root.mainloop()

    def process_queue(self):
        self.frame_skip_counter += 1
        update_interval = 16
        if self.frame_skip_counter % 2 == 0:  # Adjust this for your performance needs
            self.canvas.delete("all")
            try:
                while not self.queue.empty():
                    item = self.queue.get(timeout=0.1)
                    if isinstance(item, tuple) and len(item) == 2:
                        image, detections = item
                    else:
                        log_error(f"Warning: Unexpected item format in overlay queue: {type(item)} - {item}")
                        image = item if not isinstance(item, tuple) else item[0]
                        detections = sv.Detections.empty() if hasattr(sv, 'Detections') else None
                    
                    if image is not None and detections is not None and cfg.show_overlay:
                        annotated_image = self.annotate_with_supervision(detections, image)
                        self.update_overlay_with_image(annotated_image)
                    else:
                        log_error("Overlay received None or invalid data")
            except queue.Empty:
                pass  # No data in queue, continue
            except Exception as e:
                log_error(f"Error in overlay processing: {e}")
        
        self.root.update()  # Ensure Tkinter window updates
        self.root.after(update_interval, self.process_queue)

    def annotate_with_supervision(self, detections, image):
        if cfg.overlay_show_boxes:
            annotated_image = self.bounding_box_annotator.annotate(
                scene=image,
                detections=detections
            )
            if cfg.overlay_show_labels or cfg.overlay_show_conf:
                labels = [f"{self.get_class_name(cls)} {conf:.2f}" for cls, conf in zip(detections.class_id, detections.confidence)]
                annotated_image = self.label_annotator.annotate(
                    scene=annotated_image,
                    detections=detections,
                    labels=labels
                )
            return annotated_image
        return image

    def get_class_name(self, class_id):
        class_map = {0: 'player', 1: 'bot', 2: 'weapon', 3: 'outline', 4: 'dead_body',
                     5: 'hideout_target_human', 6: 'hideout_target_balls', 7: 'head',
                     8: 'smoke', 9: 'fire', 10: 'third_person'}
        return class_map.get(int(class_id), 'Unknown')

    def update_overlay_with_image(self, annotated_image):
        img = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo  # keep a reference to avoid garbage collection

    def show(self, width, height):
        if self.thread is None:
            self.thread = threading.Thread(target=self.run, args=(width, height), daemon=True, name="Overlay")
            self.thread.start()

