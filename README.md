
## Overview
Dons Bot is a rewrite of Sunone_Aimbot. It leverages the YOLOV11 models, PyTorch, and various other tools to automatically target and aim at enemies within the game. The AI model in repository has been trained on more than 30,000 images from popular first-person shooter games like Warface, Destiny 2, Battlefield (all series), Fortnite, The Finals, CS2 and more.
> [!WARNING]
> Use it at your own risk, There is NO guarentee this will not get you banned!

> [!NOTE]
> This requires an NVIDIA RTX+ card. Old video cards CANNOT handle this
> [!First_Run]
> If you are comming from another python bot try pip install -r requirements.txt and run bot with run.bat.
> If you are totally new to this try running installer.bat as administrator than run.bat after.


## Requirements
  CUDA 12.4 or higher
  Python 3.11.6
  PyTorch 2.7.0 or higher
  cuda_python
  bettercam
  numpy
  pywin32
  screeninfo
  asyncio
  onnxruntime
  onnxruntime-gpu
  pyserial
  requests
  opencv-python
  packaging
  ultralytics
  keyboard
  mss
  supervision
  tensorrt-cu12==10.3.0
  tensorrt-cu12_bindings==10.3.0
  tensorrt-cu12_libs==10.3.0


## MANUAL Installation links
- Download and install [Python](https://www.python.org/downloads/).
- Download and install [CUDA](https://developer.nvidia.com/cuda-toolkit).
- Download and install [TensorRT](https://developer.nvidia.com/tensorrt).
- Download and install [Ultralytics](https://github.com/ultralytics/yolov5).
- Download and install [OpenCV](https://pypi.org/project/opencv-python/).

<br></br>
- To launch the aimbot after all installations, double click deploy.bat or type `run.py` in cdm from the project folder.



## Notes / Recommendations
- Set in game FPS 60 and 1080 x1920 resolution.
- medium or lower graphics settings in games.
- close all other apps dont try to stream unless you have a great gpu setup.
- Use the F3 key to pause the aimbot and F2 to wuit.
- export to `.engine`.
- Turn off the debug window.
- Do not use the `--half` option, it will not improve performance.
- Do not increase the object search window resolution, this may affect your search speed.
- If you have started the application and nothing happens, it may be working, close it with the F2 key and change the `show_window` option to `True` in the file [config.ini](https://github.com/SunOner/sunone_aimbot/blob/main/config.ini) to make sure that the application is working.

## Support the project
[cashapp] $DonArrington

# For Sunones Models
[Boosty](https://boosty.to/sunone)

## License
This project is licensed under the MIT License. see attached license file.
