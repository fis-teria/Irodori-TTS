param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PythonArgs)
. (Join-Path $PSScriptRoot 'scripts\project_env.ps1')
Set-Location -LiteralPath $ProjectRoot
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw 'Virtual environment missing. Run setup.bat first.'
}
& $ProjectPython @PythonArgs
exit $LASTEXITCODE
