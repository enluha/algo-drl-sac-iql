@echo off
REM Run OPT-STM Generator in Interactive Mode
REM This batch file uses the virtual environment Python

cd /d "%~dp0"
chcp 65001 >nul 2>&1
set MPLCONFIGDIR=%~dp0.matplotlib
set PYTHONIOENCODING=utf-8
.\venv\Scripts\python.exe main.py --interactive
pause
