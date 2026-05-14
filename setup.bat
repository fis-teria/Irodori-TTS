@echo off
setlocal

pushd "%~dp0" >nul

where uv >nul 2>nul
if errorlevel 1 (
    echo uv is not installed or is not available on PATH.
    echo Install uv first, then run setup.bat again.
    popd >nul
    exit /b 1
)

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

echo Creating or updating .venv in:
echo   %CD%\.venv
echo Model data will be stored in:
echo   %IRODORI_TTS_MODEL_DIR%
echo Outputs will be stored in:
echo   %IRODORI_TTS_OUTPUT_DIR%
echo Data will be stored in:
echo   %IRODORI_TTS_DATA_DIR%
echo.

uv sync
if errorlevel 1 (
    echo.
    echo setup failed.
    popd >nul
    exit /b 1
)

echo.
echo setup complete.
echo To activate the environment, run:
echo   .venv\Scripts\activate

popd >nul
exit /b 0
