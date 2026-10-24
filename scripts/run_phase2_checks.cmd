@echo off
cd /d "%~dp0.."
".\.venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 exit /b %errorlevel%
".\.venv\Scripts\python.exe" scripts\smoke_phase2.py
if errorlevel 1 exit /b %errorlevel%
".\.venv\Scripts\python.exe" scripts\evaluate_phase2.py --output docs\phase2_eval_results.json
if errorlevel 1 exit /b %errorlevel%
