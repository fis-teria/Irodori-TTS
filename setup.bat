@echo off
setlocal
powershell -NoProfile -STA -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1" %*
exit /b %ERRORLEVEL%
