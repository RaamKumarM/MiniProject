@echo off
setlocal
title DDoS GNN Setup

:: --- AUTO-ADMIN ELEVATION ---
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if '%errorlevel%' NEQ '0' (
    echo [!] Requesting Admin privileges to fix folder permissions...
    goto UACPrompt
) else ( goto gotAdmin )
:UACPrompt
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
    "%temp%\getadmin.vbs" & del "%temp%\getadmin.vbs" & exit /B
:gotAdmin
    pushd "%CD%" & CD /D "%~dp0"

:: --- START OF SETUP ---
cls
echo =========================================
echo         DDoS GNN SETUP SYSTEM
echo =========================================

:: 1. Create Venv
if exist ".venv\Scripts\python.exe" (
    echo [*] Virtual environment already exists. Skipping creation.
) else (
    <nul set /p ="[*] Creating virtual environment... "
    python -m venv .venv >nul 2>&1 || python3 -m venv .venv >nul 2>&1 || py -m venv .venv >nul 2>&1
    if %errorlevel% equ 0 (echo DONE.) else (echo FAILED. & goto :error)
)

:: 2. Upgrade Pip (Crucial for Scapy/GNN libs)
<nul set /p ="[*] Upgrading pip... "
.\.venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1
echo DONE.

:: 3. Requirements
<nul set /p ="[*] Installing requirements (this takes time)... "
.\.venv\Scripts\pip install -r requirements.txt >nul 2>&1
if %errorlevel% equ 0 (echo DONE.) else (echo FAILED. & goto :error)

:: 4. Live Capture Tools
<nul set /p ="[*] Installing Scapy and Requests... "
.\.venv\Scripts\pip install scapy requests >nul 2>&1
echo DONE.

:: 5. Create Desktop Shortcut
<nul set /p ="[*] Creating Desktop Shortcut... "
powershell.exe -ExecutionPolicy Bypass -Command ^
    "$WshShell = New-Object -ComObject WScript.Shell; " ^
    "$ShortcutPath = [System.IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), 'DDoS GNN Detector.lnk'); " ^
    "$Shortcut = $WshShell.CreateShortcut($ShortcutPath); " ^
    "$Shortcut.TargetPath = '%~dp0run.bat'; " ^
    "$Shortcut.WorkingDirectory = '%~dp0'; " ^
    "$Shortcut.Description = 'Launch DDoS GNN Detection System'; " ^
    "$Shortcut.IconLocation = 'shell32.dll, 18'; " ^
    "$Shortcut.Save()"
echo DONE.

echo.
echo =========================================
echo    SUCCESS: Environment is ready.
echo    A shortcut has been created on your Desktop.
echo    Use the shortcut or run.bat to start.
echo =========================================
pause
exit

:error
echo.
echo -----------------------------------------
echo ERROR: Setup failed. 
echo 1. Check your internet connection.
echo 2. Ensure Python 3.x is in your PATH.
echo -----------------------------------------
pause