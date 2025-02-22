#configeditor.py ---
import tkinter as tk
import time
import os
from tkinter import ttk
from utils import log_error
class ConfigEditor:
    def __init__(self, config_obj, restart_callback=None):
        self.root = tk.Tk()
        self.root.title("Config Editor")
        self.restart_callback = restart_callback
        self.config = config_obj.config
        self.restart_callback = restart_callback
        
        try:
            config_obj.read(verbose=True)  # Force reload config
            current_config = self.config
        except Exception as e:
            log_error("Error loading config: {e}")
        
        notebook = ttk.Notebook(self.root)
        
        # Store widgets and variables for saving
        self.widgets = {}
        self.variables = {}
        
        # Get current config values
        current_config = self.config
        
        for section in config_obj.CONFIG_SECTIONS.values():
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=section)
            
            section_items = current_config[section]
            
            row = 0
            for key, value in section_items.items():
                label = ttk.Label(tab, text=key)
                label.grid(row=row, column=0, padx=5, pady=5)
                
                if isinstance(value, bool):
                    var = tk.BooleanVar(value=value)
                    widget = ttk.Combobox(tab, values=['True', 'False'], state='readonly')
                    widget.set(str(value))
                    widget.bind('<<ComboboxSelected>>', lambda e, v=var: v.set(e.widget.get() == 'True'))
                else:
                    var = tk.StringVar(value=str(value))
                    widget = ttk.Entry(tab, textvariable=var)
                
                widget.grid(row=row, column=1, padx=5, pady=5)
                
                self.variables[(section, key)] = var
                self.widgets[(section, key)] = widget
                row += 1
        
        notebook.pack(expand=True, fill='both', padx=5, pady=5)
        
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        save_btn = ttk.Button(button_frame, text="Save", command=self.save_config)
        save_btn.pack(side='left', padx=5)
        
        restart_btn = ttk.Button(button_frame, text="Save & Restart", 
                              command=self.save_and_restart)
        restart_btn.pack(side='left', padx=5)
        
        close_btn = ttk.Button(button_frame, text="Close", command=self.close)
        close_btn.pack(side='left', padx=5)
        
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def save_and_restart(self):
        """Save config and fully restart the application."""
        self.save_config()
        if self.restart_callback:
            try:
                self.close()  # Close editor UI
                time.sleep(0.1)  # Brief delay to release Tkinter
                self.restart_callback()  # Restart app
                # Ensure this process exits if callback doesn't
                os._exit(0)
            except Exception as e:
                log_error(f"Error during restart: {e}")
                os._exit(1)
        else:
            self.close()

    def close(self):
        if self.root and self.root.winfo_exists():
            self.root.destroy()
            self.root.quit()

    def save_config(self):
        for (section, key), var in self.variables.items():
            value = var.get()
            self.config[section][key] = str(value)
        directory = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(directory, "config.ini")
        with open(config_path, 'w') as configfile:
            self.config.write(configfile)
            
    def save_and_restart(self):
        self.save_config()
        if self.restart_callback:
            try:
                self.close()  # Close editor UI first
                self.restart_callback()  # Then restart
            except Exception as e:
                log_error(f"Error during restart: {e}")
        else:
            self.close()  # Fallback to just closing if no callback        

    def show(self):
        self.root.mainloop()

if __name__ == "__main__":
    from config_watcher import cfg
    editor = ConfigEditor(cfg)
    editor.show()