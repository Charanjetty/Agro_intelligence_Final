@echo off
setlocal
cd /d %~dp0

echo --- AgroIntelligence Environment Manager ---

:: 1. Check for Virtual Environment
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

:: 2. Update Requirements
echo Checking and updating dependencies (this may take a minute)...
.\.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

:: 3. Start Application
echo Starting AgroIntelligence application...
echo The application will be accessible at: http://127.0.0.1:5000
.\.venv\Scripts\python.exe app.py

pause
