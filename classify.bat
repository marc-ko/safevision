@echo off
REM Batch script to run content classifier with correct Python environment
REM This script uses Python 3.13 where nudenet is installed

set PYTHON_PATH=C:\Users\skip2\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe

if "%1"=="" (
    echo Usage: classify.bat [image_path_or_directory] [options]
    echo.
    echo Examples:
    echo   classify.bat image.jpg
    echo   classify.bat images --output results.json
    echo   classify.bat images --threshold 0.7
    echo.
    echo Run with --help for more options:
    "%PYTHON_PATH%" batch_classify.py --help
    exit /b 1
)

"%PYTHON_PATH%" batch_classify.py %*


