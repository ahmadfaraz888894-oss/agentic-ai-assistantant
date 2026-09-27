@echo off
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail
if not exist .env copy .env.example .env >nul
echo Setup complete. Double-click start_windows.bat to launch.
pause
exit /b 0
:fail
echo Setup failed. Check Python 3.10 or newer and your internet connection.
pause
exit /b 1
