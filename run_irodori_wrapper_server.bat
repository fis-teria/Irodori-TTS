@echo off
setlocal

pushd "%~dp0" >nul

set "UV_PROJECT_ENVIRONMENT=.venv"
set "IRODORI_TTS_MODEL_DIR=%CD%\model_data"
set "IRODORI_TTS_OUTPUT_DIR=%CD%\outputs"
set "IRODORI_TTS_DATA_DIR=%CD%\data"
set "HF_HOME=%IRODORI_TTS_MODEL_DIR%\huggingface"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "HUGGINGFACE_HUB_CACHE=%HF_HUB_CACHE%"
set "HF_DATASETS_CACHE=%HF_HOME%\datasets"
set "TRANSFORMERS_CACHE=%HF_HUB_CACHE%"
set "TORCH_HOME=%IRODORI_TTS_MODEL_DIR%\torch"
set "WANDB_DIR=%IRODORI_TTS_OUTPUT_DIR%\wandb"
set "WANDB_CACHE_DIR=%IRODORI_TTS_MODEL_DIR%\wandb\cache"
set "WANDB_ARTIFACT_DIR=%IRODORI_TTS_MODEL_DIR%\wandb\artifacts"

if not exist "%HF_HUB_CACHE%" mkdir "%HF_HUB_CACHE%" >nul 2>nul
if not exist "%HF_DATASETS_CACHE%" mkdir "%HF_DATASETS_CACHE%" >nul 2>nul
if not exist "%TORCH_HOME%" mkdir "%TORCH_HOME%" >nul 2>nul
if not exist "%IRODORI_TTS_OUTPUT_DIR%" mkdir "%IRODORI_TTS_OUTPUT_DIR%" >nul 2>nul
if not exist "%IRODORI_TTS_DATA_DIR%" mkdir "%IRODORI_TTS_DATA_DIR%" >nul 2>nul
if not exist "%WANDB_DIR%" mkdir "%WANDB_DIR%" >nul 2>nul
if not exist "%WANDB_CACHE_DIR%" mkdir "%WANDB_CACHE_DIR%" >nul 2>nul
if not exist "%WANDB_ARTIFACT_DIR%" mkdir "%WANDB_ARTIFACT_DIR%" >nul 2>nul

echo Starting Irodori-TTS SBV2-compatible wrapper...
echo URL: http://127.0.0.1:6001
echo.
echo Example:
echo   run_irodori_wrapper_server.bat --reference-audio path\to\voice.wav
echo.

uv run python irodori_wrapper_server.py %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo wrapper stopped with error code %EXIT_CODE%.
    pause
)

popd >nul
exit /b %EXIT_CODE%
