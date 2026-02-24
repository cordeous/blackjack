@echo off
echo Starting Blackjack AI API server...
cd /d "%~dp0"
uvicorn api:app --reload --host 0.0.0.0 --port 8000
