@echo off
setlocal
"%~dp0.venv\Scripts\python.exe" "%~dp0setup_check.py" || exit /b 1
"%~dp0.venv\Scripts\python.exe" "%~dp0acceptance_harness.py" %*
endlocal
