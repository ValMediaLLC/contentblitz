REM Developer environment bootstrap script

@echo off
title ContentBlitz AMD64 Environment Setup

echo.
echo ==========================================
echo   ContentBlitz AMD64 Python Setup
echo ==========================================
echo.

echo Creating .venv-x64...
py -3.13 -m venv .venv-x64

if not exist ".venv-x64\Scripts\python.exe" (
    echo Failed to create .venv-x64
    pause
    exit /b 1
)

echo.
echo Verifying architecture...
".venv-x64\Scripts\python.exe" -c "import sysconfig; print(sysconfig.get_platform())"

echo.
echo Upgrading pip...
".venv-x64\Scripts\python.exe" -m pip install --upgrade pip

echo.
echo Installing requirements...
".venv-x64\Scripts\python.exe" -m pip install -r requirements.txt

echo.
echo Launching activated PowerShell session...
echo.

powershell -NoExit -ExecutionPolicy Bypass -Command ".\.venv-x64\Scripts\Activate.ps1"