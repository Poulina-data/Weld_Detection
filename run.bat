@echo off
setlocal enabledelayedexpansion

echo =======================================================================
echo                 YOLO Vision Studio - Windows Launcher
echo =======================================================================
echo.

cd /d "%~dp0"

:: 1. Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in your PATH.
    echo Please install Python 3.10 and check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: 2. Check virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo [SETUP] Creating Python virtual environment in .\venv ...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [SETUP] Activating virtual environment...
    call venv\Scripts\activate.bat
    echo [SETUP] Upgrading pip...
    python -m pip install --upgrade pip
    echo [SETUP] Installing dependencies from requirements.txt ...
    pip install -r requirements.txt
) else (
    echo [INFO] Activating existing virtual environment...
    call venv\Scripts\activate.bat
)

:: 3. Check for default model file
if not exist "models\best.pt" (
    if exist "runs\detect\YOLO weld detection project\weld_yolo_training\weights\best.pt" (
        echo [INFO] Linking trained weld model to models\best.pt ...
        copy "runs\detect\YOLO weld detection project\weld_yolo_training\weights\best.pt" "models\best.pt" >nul
    )
)

echo.
echo =======================================================================
echo    Application starting at: http://127.0.0.1:5000
echo    Press Ctrl+C to stop the server
echo =======================================================================
echo.

python app.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with an error.
    pause
)
