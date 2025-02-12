import os
import torch
import onnx
from onnxconverter_common import float16
from config_watcher import cfg

def convert_onnx_to_fp16():
    model = onnx.load(os.path.join("models", cfg.ai_model_name))
    model_fp16 = float16.convert_float_to_float16(model)
    new_model_name = cfg.ai_model_name.replace(".onnx", "_fp16.onnx")
    onnx.save(model_fp16, os.path.join("models", new_model_name))
    print(f"Converted 'models/{new_model_name}'.\n"
          f"Please change the suffix name to '{new_model_name}' in your configuration.")

def check_model_fp16():
    model = onnx.load(os.path.join("models", cfg.ai_model_name))
    tensor_types = {onnx.TensorProto.FLOAT16}
    for tensor in model.graph.input + model.graph.output:
        if tensor.type.tensor_type.elem_type in tensor_types:
            return True
    return False

def warnings():
    if ".pt" in cfg.ai_model_name:
        print("FYI: Export `.engine` for better performance!")
    if cfg.show_window:
        print("Debug using resources.")
    if cfg.bettercam_capture_fps >= 120:
        print("WARNING: A high number of frames per second can affect automatic aiming behavior (shaking).")
    if cfg.detection_window_width >= 600 or cfg.detection_window_height >= 600:
        print("WARNING: The object detector window exceeds 600 pixels in width or height, which might impact performance.")
    if cfg.ai_conf <= 0.15:
        print("WARNING: A low `ai_conf` value can lead to many false positives.")
    if cfg.disable_tracker:
        print("Disabling the tracking system might cause performance issues due to increased computational overhead.")
    if not (cfg.mouse_ghub or cfg.arduino_move or cfg.arduino_shoot):
        print("Stream Guardian Multicast Active")
    if cfg.mouse_ghub and not (cfg.arduino_move or cfg.arduino_shoot):
        print("WARNING: gHub might be detected in some games.")
    if not cfg.arduino_move:
        print("Using Cuda Encoder Beta")
    if cfg.auto_shoot and not cfg.arduino_shoot:
        print("dinopw active")
    selected_methods = sum([cfg.arduino_move, cfg.mouse_ghub, cfg.mouse_rzr])
    if selected_methods > 1:
        raise ValueError("Only one input method should be selected.")

def run_checks():
    if not torch.cuda.is_available():
        raise RuntimeError("You need to install a version of PyTorch that supports CUDA. "
                           "First uninstall all torch packages. "
                           "Run command 'pip uninstall torch torchvision torchaudio'. "
                           "Next, go to 'https://pytorch.org/get-started/locally/' and install torch with CUDA support. "
                           "Don't forget your CUDA version (Minimum version is 12.1, max version is 12.4).")
    
    capture_methods = sum([cfg.mss_capture, cfg.bettercam_capture, cfg.obs_capture])
    if capture_methods < 1:
        raise RuntimeError("You must use at least one image capture method. Set the value to `True` for one of `mss_capture`, `Bettercam_capture`, or `Obs_capture`.")
    elif capture_methods > 1:
        raise RuntimeError("Only one capture method is allowed. Set the value to `True` for only one of `mss_capture`, `Bettercam_capture`, or `Obs_capture`.")
    
    model_path = os.path.join("models", cfg.ai_model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The ai model '{cfg.ai_model_name}' was not found! Check the model name in the ai_model_name option.")
    
    if cfg.ai_model_name.endswith(".onnx"):
        if not check_model_fp16():
            check_converted_model = cfg.ai_model_name.replace(".onnx", "_fp16.onnx")
            if not os.path.exists(os.path.join("models", check_converted_model)):
                print(f"The current ai model '{cfg.ai_model_name}' is in FP32. Converting model to FP16...")
                convert_onnx_to_fp16()
            else:
                print(f"Please use the converted model - '{check_converted_model}'.\n"
                      f"Update your config.ini with 'ai_model_name = {check_converted_model}'")
    
    warnings()