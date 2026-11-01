@echo off
setlocal
call "%~dp0run_python.bat" gradio_app.py --server-name 127.0.0.1 --server-port 7860 %*
exit /b %ERRORLEVEL%
