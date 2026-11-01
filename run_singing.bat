@echo off
setlocal
rem Singing conversion CLI. For Japanese text prefer run_singing.ps1 or --tts-text-file.
call "%~dp0run_python.bat" -m singing_vc.cli %*
exit /b %ERRORLEVEL%
