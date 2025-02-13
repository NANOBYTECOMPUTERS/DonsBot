import tkinter as tk
import os
from tkinter import ttk
from config_watcher import cfg, Config

class ConfigEditor:
    # Fix for blank values in ConfigEditor
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Config Editor")
        self.config = cfg.config
        notebook = ttk.Notebook(self.root)
        
        # Store widgets and variables for saving
        self.widgets = {}
        self.variables = {}
        
        # Get current config values
        current_config = cfg.config
        
        for section in Config.CONFIG_SECTIONS.values():
      
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=section)
            
            # Get the current section's items
            section_items = current_config[section]
            
            row = 0
            for key, value in section_items.items():
                # Create label
                label = ttk.Label(tab, text=key)
                label.grid(row=row, column=0, padx=5, pady=5)
                
                # Create appropriate widget based on value type
                if isinstance(value, bool):
                    var = tk.BooleanVar(value=value)
                    widget = ttk.Checkbutton(tab, variable=var)
                else:
                    var = tk.StringVar(value=str(value))
                    widget = ttk.Entry(tab, textvariable=var)
                
                widget.grid(row=row, column=1, padx=5, pady=5)
                
                # Store references
                self.variables[(section, key)] = var
                self.widgets[(section, key)] = widget
                row += 1
        
        notebook.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Button frame for Save and Close
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # Save button
        save_btn = ttk.Button(button_frame, text="Save", command=self.save_config)
        save_btn.pack(side='left', padx=5)
        
        # Close button
        close_btn = ttk.Button(button_frame, text="Close", command=self.root.destroy)
        close_btn.pack(side='left', padx=5)

    def save_config(self):
        # Update config with new values from widgets
        for (section, key), var in self.variables.items():
            value = var.get()
            # Convert to string for ConfigParser
            self.config[section][key] = str(value)
        
        directory = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(directory, "config.ini")
        
        with open(config_path, 'w') as configfile:
            self.config.write(configfile)

    def show(self):
        self.root.mainloop()

editor = ConfigEditor()
editor.show()