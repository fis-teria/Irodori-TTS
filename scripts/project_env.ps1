$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
# This helper uses only the standard library, before installing any dependencies.
$BootstrapPython = (Get-Command python -ErrorAction Stop).Source
$EnvironmentJson = & $BootstrapPython (Join-Path $ProjectRoot 'irodori_tts\local_env.py')
if ($LASTEXITCODE -ne 0) { throw 'Could not configure the workspace environment.' }
$EnvironmentValues = $EnvironmentJson | ConvertFrom-Json
foreach ($Property in $EnvironmentValues.PSObject.Properties) {
    [Environment]::SetEnvironmentVariable($Property.Name, [string]$Property.Value, 'Process')
}
Remove-Item Env:TRANSFORMERS_CACHE -ErrorAction SilentlyContinue
Remove-Item Env:UV_NATIVE_TLS -ErrorAction SilentlyContinue
$ProjectPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ProjectUv = Join-Path $ProjectRoot '.tools\uv\uv.exe'
