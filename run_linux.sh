#!/bin/bash
echo "==================================================="
echo "Hybrid-AirScribe Digital Twin Server - Linux/Mac Setup"
echo "==================================================="

echo "[1/3] Creating virtual environment..."
python3 -m venv venv

echo "[2/3] Activating virtual environment..."
source venv/bin/activate

echo "[3/3] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup Complete! Starting the server..."
echo ""
python3 main.py
