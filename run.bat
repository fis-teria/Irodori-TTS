@echo off
setlocal

pushd "%~dp0" >nul

echo Irodori-TTS WebUI launcher
echo.
echo   1. Reference Audio WebUI  - voice cloning with reference audio
echo   2. VoiceDesign WebUI      - style control with caption prompt
echo.
set /p "CHOICE=Select UI to start [1/2]: "

if "%CHOICE%"=="1" (
    call "%~dp0run_webui_reference_audio.bat"
) else if "%CHOICE%"=="2" (
    call "%~dp0run_webui_voice_design.bat"
) else (
    echo Invalid selection.
    popd >nul
    exit /b 1
)

popd >nul
exit /b %ERRORLEVEL%
