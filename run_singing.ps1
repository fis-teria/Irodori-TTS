# Singing conversion CLI (PowerShell keeps Japanese arguments intact).
& (Join-Path $PSScriptRoot 'run_python.ps1') -m singing_vc.cli @args
exit $LASTEXITCODE
