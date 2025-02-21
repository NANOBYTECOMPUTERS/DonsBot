@echo off
setlocal enabledelayedexpansion

:: Create a temporary directory
mkdir temp

:: Check if Python 3.11.6 is installed
python --version 2>nul
if %errorlevel% neq 0 (
    echo Python 3.11.6 not found. Downloading installer...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe' -OutFile 'temp\python-3.11.6-amd64.exe'"
    echo Installing Python...
    start /wait temp\python-3.11.6-amd64.exe /verysilent /PrependPath=1 /InstallAllUsers=1
) else (
    echo Python 3.11.6 is already installed.
)

:: Check if pip is installed with Python
python -m pip --version 2>nul
if %errorlevel% neq 0 (
    echo pip is not installed. Installing...
    python -m ensurepip --default-pip
    if %errorlevel% neq 0 (
        echo Failed to install pip automatically. Please install pip manually.
        exit /b 1
    )
)

:: Find Python installation path (assuming it's added to PATH or in common locations)
for %%i in (python.exe) do set "PYTHON_PATH=%%~$PATH:i"
if not defined PYTHON_PATH set PYTHON_PATH=C:\Python311\python.exe

:: Create registry entry for DirectX user GPU preferences
set REG_PATH=HKCU\Software\Microsoft\DirectX\UserGpuPreferences
set REG_VALUE_NAME=%PYTHON_PATH%
set REG_VALUE_DATA=SwapEffectUpgradeEnable=0;GpuPreference=1;

reg add "%REG_PATH%" /v "%REG_VALUE_NAME%" /t REG_SZ /d "%REG_VALUE_DATA%" /f

echo Downloading CUDA Toolkit online installer...
curl -o cuda_12.8.0_windows_network.exe https://developer.download.nvidia.com/compute/cuda/12.8.0/network_installers/cuda_12.8.0_windows_network.exe
echo Download complete. Starting installation...
cuda_12.8.0_windows_network.exe
echo Press any key when finished with CUDA installation...
pause

:: Installing pip modules

echo Now we will install python modules and fix the bugs it makes.
REM Generate list of installed packages
pip freeze > list.txt
if %ERRORLEVEL% NEQ 0 (
    echo Error generating package list!
    pause
    exit /b %ERRORLEVEL%
)

REM Uninstall all packages from the list
pip uninstall -r list.txt -y
if %ERRORLEVEL% NEQ 0 (
    echo No packages found moving on!
)

REM Purge pip cache
pip cache purge
if %ERRORLEVEL% NEQ 0 (
    echo Error purging cache!
    pause
    exit /b %ERRORLEVEL%
)

REM Install PyTorch and related packages
echo installing pytorch
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
if %ERRORLEVEL% NEQ 0 (
    echo Error installing PyTorch packages!
    pause
    exit /b %ERRORLEVEL%
)

REM Install packages from requirements.txt if it exists
if exist requirements.txt (
    pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo Error installing from requirements.txt!
        pause
        exit /b %ERRORLEVEL%
    )
) else (
    echo requirements.txt not found, skipping...
)

REM Installing inference
echo installing inference package
pip install inference -y
if %ERRORLEVEL% NEQ 0 (
    echo Error uninstalling opencv-contrib-python!
    pause
)

REM Reinstall opencv-contrib-python and reinstall again
echo Fix the openCV issues inference just made  ¯\_(o.o)_/¯ ...
pip uninstall opencv-contrib-python -y
if %ERRORLEVEL% NEQ 0 (
    echo Error uninstalling opencv-contrib-python!
    pause
    exit /b %ERRORLEVEL%
)
echo uninstalling redundant opencv the inference package roboflow is on the pipe...
pip uninstall opencv-python -y
if %ERRORLEVEL% NEQ 0 (
    echo Error uninstalling opencv-contrib-python!
    pause
    exit /b %ERRORLEVEL%
)
pip install opencv-contrib-python==4.11.0.86 --no-cache-dir --force-reinstall --verbose
if %ERRORLEVEL% NEQ 0 (
    echo Error installing opencv-contrib-python!
    pause
    exit /b %ERRORLEVEL%
)

REM Fix a compatibility issue we just made by installing specific numpy version
echo Installing numpy 2.1.0 for compatibility fix ...
pip install numpy==2.1.0 --no-cache-dir --force-reinstall
if %ERRORLEVEL% NEQ 0 (
    echo Error installing numpy!
    pause
    exit /b %ERRORLEVEL%
)
REM Installing tensorRT
echo installing tensorrt for cuda12.8
python -m pip install --pre torch torch-tensorrt tensorrt --extra-index-url https://download.pytorch.org/whl/nightly/cu128
if %ERRORLEVEL% NEQ 0 (
    echo Error installing numpy!
    pause
    exit /b %ERRORLEVEL%
)

echo rerunning pytorch install so it knows its there (IDK But it worked for me)
pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
if %ERRORLEVEL% NEQ 0 (
    echo Error installing numpy!
    pause
    exit /b %ERRORLEVEL%
)

REM Exporting model to tensor engine
echo Exporting 1.pt to a tensor engine. things may have just changed on your system just let it happen.
cd models
yolo export model=1.pt format=engine half=True
if %ERRORLEVEL% NEQ 0 (
    echo Error installing numpy!
    pause
    exit /b %ERRORLEVEL%
)

echo Setup complete completed successfully ignore the warning of opencv missing it is installed with contrib!
echo Dont forget to export your existing model to engine again somthing may have just change causing this application to crash. 
echo 1.pt was automatically exported to engine for you yolo export model=1.pt format=engine half=True
echo If you want to use a different model, you can do so by changing the model name in the command above in models directory with CMD.
echo Save anything you are working on and press any key to start reboot process.

pause

your computer will reeboot now
shutdown /r /t 10
echo Setup complete.
endlocal