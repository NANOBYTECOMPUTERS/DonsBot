#arduino.py ---
import os
import psutil
import serial
import serial.tools.list_ports
from config_watcher import Config

class ArduinoMouse:
    MAX_VALUE = 127

    def __init__(self):
        self.cfg = Config()
        self.serial_port = serial.Serial(
            baudrate=self.cfg.arduino_baudrate,
            timeout=0,
            write_timeout=0
        )
        self.serial_port.port = self._detect_port() if self.cfg.arduino_port == 'auto' else self.cfg.arduino_port
        try:
            self.serial_port.open()
            print(f'Arduino: Connected! Port: {self.serial_port.port}')
        except serial.SerialException as e:
            print(f'Arduino: Not Connected...\n{e}')
            self.checks()
            return 

        if not self.serial_port.is_open:
            raise RuntimeError("Serial port could not be opened.")

    def click(self):
        self._send_command('c')

    def press(self):
        self._send_command('p')

    def release(self):
        self._send_command('r')

    def move(self, x, y):
        if self.cfg.arduino_16_bit_mouse:
            data = f'm{x},{y}\n'.encode()
            self.serial_port.write(data)
        else:
            x_parts = self._split_value(x)
            y_parts = self._split_value(y)
            max_steps = max(len(x_parts), len(y_parts))
            x_parts.extend([0] * (max_steps - len(x_parts)))
            y_parts.extend([0] * (max_steps - len(y_parts)))
            messages = ''.join(f'm{x_val},{y_val}\n' for x_val, y_val in zip(x_parts, y_parts))
            self.serial_port.write(messages.encode())

    def _split_value(self, value):
        sign = -1 if value < 0 else 1
        abs_value = abs(value)
        count, remainder = divmod(abs_value, self.MAX_VALUE)
        values = [sign * self.MAX_VALUE] * count
        if remainder:
            values.append(sign * remainder)
        return values

    def close(self):
        if self.serial_port.is_open:
            self.serial_port.close()

    def __del__(self):
        self.close()

    def _detect_port(self):
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if "Arduino" in port.description:
                return port.device
        return None

    def _send_command(self, command):
        self.serial_port.write(f'{command}\n'.encode())

    def find_library_directory(self, base_path, library_name_start):
        for root, dirs, files in os.walk(base_path):
            for dir_name in dirs:
                if dir_name.startswith(library_name_start):
                    return os.path.join(root, dir_name)
        return None

    def checks(self):
        for process in psutil.process_iter(['pid', 'name']):
            if process.info['name'] == 'Arduino IDE.exe':
                print('Arduino: Arduino IDE is open, close IDE and restart app.')
                break

        try:
            documents_path = os.path.join(os.environ['USERPROFILE'], 'Documents')
            arduino_libraries_path = os.path.join(documents_path, 'Arduino', 'libraries')
            usb_host_shield_library_path = self.find_library_directory(arduino_libraries_path, 'USB_Host_Shield')
            if not usb_host_shield_library_path:
                print('Arduino: USB_Host_Shield lib not found.')
                return

            hid_settings = os.path.join(usb_host_shield_library_path, 'settings.h')
            with open(hid_settings, 'r') as file:
                for line in file:
                    if line.startswith('#define ENABLE_UHS_DEBUGGING'):
                        parts = line.split()
                        if len(parts) == 3 and parts[1] == 'ENABLE_UHS_DEBUGGING':
                            value = parts[2]
                            if value == '1':
                                print(f'Arduino: Disable `ENABLE_UHS_DEBUGGING` setting in {hid_settings} file.')
                                break
        except (FileNotFoundError, IOError) as e:
            print(f'Arduino: Error checking USB_Host_Shield lib settings.\n{e}')

arduino = ArduinoMouse()