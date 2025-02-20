@echo off
echo Starting Don's fixer script...

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
    echo Error uninstalling packages!
    pause
    exit /b %ERRORLEVEL%
)

REM Purge pip cache
pip cache purge
if %ERRORLEVEL% NEQ 0 (
    echo Error purging cache!
    pause
    exit /b %ERRORLEVEL%
)

REM Install PyTorch and related packages
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

REM Uninstall and reinstall opencv-contrib-python and reinstall again ¯\_(ツ)_/¯ 
echo Reinstalling opencv-contrib-python...
pip uninstall opencv-contrib-python -y
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
echo Installing numpy 2.1.0 for compatibility...
pip install numpy==2.1.0 --no-cache-dir --force-reinstall
if %ERRORLEVEL% NEQ 0 (
    echo Error installing numpy!
    pause
    exit /b %ERRORLEVEL%
)

echo Script completed successfully!
pause