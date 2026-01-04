# AgroIntelligence Startup Script
# This script manages the virtual environment, updates dependencies, and starts the application.

$ProjectDir = $PSScriptRoot
if ($null -eq $ProjectDir) { $ProjectDir = Get-Location }
Set-Location $ProjectDir

Write-Host "--- AgroIntelligence Environment Manager ---" -ForegroundColor Cyan

# 1. Check for Virtual Environment
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# 2. Update Requirements
Write-Host "Checking and updating dependencies..." -ForegroundColor Yellow
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

# 3. Start Application
Write-Host "Starting AgroIntelligence application..." -ForegroundColor Green
Write-Host "The application will be accessible at: http://127.0.0.1:5000" -ForegroundColor Cyan
$env:DATABASE_URL="postgresql://postgres:Charan%402005@localhost/agrointelligence"
& ".\.venv\Scripts\python.exe" app.py
