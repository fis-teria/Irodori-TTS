@echo off
setlocal
call "%~dp0run_python.bat" -m singing_vc.webui --server-name 127.0.0.1 --server-port 7862 %*
exit /b %ERRORLEVEL%
