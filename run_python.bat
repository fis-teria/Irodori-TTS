@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_python.ps1" %*
exit /b %ERRORLEVEL%
