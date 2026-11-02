@echo off
cd /d "%~dp0"
where pythonw >nul 2>nul
if errorlevel 1 (
    python run.pyw
) else (
    start "" pythonw run.pyw
)
