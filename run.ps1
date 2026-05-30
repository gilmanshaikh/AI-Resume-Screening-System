# DevOps AI Resume Screening — local launcher
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\streamlit.exe")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
    .\.venv\Scripts\pip.exe install -r requirements.txt
}

Write-Host ""
Write-Host "Starting app at http://localhost:8501"
Write-Host "Press Ctrl+C to stop."
Write-Host ""

.\.venv\Scripts\streamlit.exe run webapp.py --server.address localhost --server.port 8501
