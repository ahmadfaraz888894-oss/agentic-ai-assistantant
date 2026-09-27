@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Please run setup_windows.bat first.
    pause
    exit /b 1
)
.venv\Scripts\python.exe -m streamlit run apps.py
pause
