@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo [!] ERROR: Virtual environment not found or pythonw.exe missing.
    echo [!] Please run setup.bat first.
    pause
    exit /b
)

:: Launching silently
start "" ".\.venv\Scripts\pythonw.exe" "run.py"
exit