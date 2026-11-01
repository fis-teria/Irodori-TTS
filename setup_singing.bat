@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_singing.ps1" %*
exit /b %ERRORLEVEL%
