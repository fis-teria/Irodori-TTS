@echo off
setlocal
call "%~dp0run_python.bat" gradio_app_voicedesign.py --server-name 127.0.0.1 --server-port 7861 %*
exit /b %ERRORLEVEL%
