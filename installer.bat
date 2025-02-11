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

:: Download CUDA Toolkit
echo Downloading CUDA Toolkit...
powershell -Command "Invoke-WebRequest -Uri 'https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.76_windows.exe' -OutFile 'temp\cuda_12.6.0_560.76_windows.exe'"

:: Install CUDA Toolkit silently
echo Installing CUDA Toolkit...
start /wait temp\cuda_12.6.0_560.76_windows.exe -s


echo Setup complete.
endlocal