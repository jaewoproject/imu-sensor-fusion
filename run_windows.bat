@echo off
echo ===================================================
echo Hybrid-AirScribe Digital Twin Server - Windows Setup
echo ===================================================

echo [1/3] Creating virtual environment...
python -m venv venv

echo [2/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/3] Installing dependencies...
pip install -r requirements.txt

echo.
echo Setup Complete! Starting the server...
echo.
python main.py

pause
