@echo off
echo Installing required Python packages...

pip install cuda-python
pip install bettercam
pip install numpy
pip install numba==0.61.0
pip install numba-cuda==0.3.0
pip install lazy_loader==0.4
pip install pywin32
pip install screeninfo
pip install asyncio
pip install cupy-cuda12x
pip install onnx==1.17.0
pip install onnxruntime-gpu==1.20.1
pip install onnxslim==0.1.48
pip install onnxruntime==1.20.1
pip install roboflow
pip install pyserial
pip install requests
pip install opencv-python
pip install packaging
pip install ultralytics
pip install keyboard
pip install mss
pip install supervision
pip install tensorrt-cu12==10.3.0
pip install tensorrt-cu12_bindings==10.3.0
pip install tensorrt-cu12_libs==10.3.0

echo Installation completed.
pause