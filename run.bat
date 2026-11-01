@echo off
setlocal
echo Irodori-TTS V4 / V4.1
echo 1. Reference Audio WebUI
echo 2. VoiceDesign WebUI
choice /c 12 /n /m "Select UI [1/2]: "
if errorlevel 2 (
    call "%~dp0run_webui_voice_design.bat" %*
) else (
    call "%~dp0run_webui_reference_audio.bat" %*
)
exit /b %ERRORLEVEL%
