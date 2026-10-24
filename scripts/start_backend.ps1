$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
