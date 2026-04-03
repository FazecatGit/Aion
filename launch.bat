@echo off
title Aion Launcher
cd /d "%~dp0"

REM Try venv first, then system Python
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe launch.py %*
) else if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe launch.py %*
) else (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        python launch.py %*
    ) else (
        echo ERROR: Python not found.
        echo Install Python from https://www.python.org/downloads/
        echo or create a virtual environment: python -m venv .venv
    )
)
pause
