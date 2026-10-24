$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\smoke_phase2.py
.\.venv\Scripts\python.exe scripts\evaluate_phase2.py --output docs\phase2_eval_results.json
