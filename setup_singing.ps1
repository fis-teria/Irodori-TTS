[CmdletBinding()]
param(
    # Existing Irodori-TTS_V4 checkout whose Hugging Face cache is hardlinked (read-only) to avoid re-downloads.
    [string]$SeedFrom = '',
    [ValidateSet('cu128', 'cpu', 'xpu')][string]$Backend = 'cu128',
    [switch]$SkipIrodori
)

# Singing prototype setup: Irodori venv (.venv) + separate Seed-VC v1 venv (.singing\venv).
# Safe to rerun; cached environments and models are reused, and reruns work offline
# once everything has been fetched.
$ErrorActionPreference = 'Stop'
$SeedVcRepo = 'https://github.com/Plachtaa/seed-vc.git'
$SeedVcCommit = '51383efd921027683c89e5348211d93ff12ac2a8'

. (Join-Path $PSScriptRoot 'scripts\project_env.ps1')
Set-Location -LiteralPath $ProjectRoot
$SingingDir = Join-Path $ProjectRoot '.singing'
$SeedVcDir = Join-Path $SingingDir 'seed-vc'
$SeedVcPython = Join-Path $SingingDir 'venv\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path $SingingDir | Out-Null

function Invoke-Checked([string]$Label, [scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Label failed: exit $LASTEXITCODE" }
}

# 1. Seed-VC source at the pinned commit (GPL-3.0; kept outside Git tracking).
if (-not (Test-Path -LiteralPath (Join-Path $SeedVcDir '.git'))) {
    if (Test-Path -LiteralPath $SeedVcDir) {
        throw "$SeedVcDir exists but is not a Git checkout; move it aside and rerun."
    }
    Write-Host "Cloning Seed-VC $SeedVcCommit"
    Invoke-Checked 'git clone' { git clone --quiet $SeedVcRepo $SeedVcDir }
}
$Head = (git -C $SeedVcDir rev-parse HEAD).Trim()
if ($Head -ne $SeedVcCommit) {
    if (git -C $SeedVcDir status --porcelain --untracked-files=no) {
        throw "Seed-VC checkout has local changes; refusing to move it to $SeedVcCommit."
    }
    Invoke-Checked 'git fetch' { git -C $SeedVcDir fetch --quiet origin $SeedVcCommit }
    Invoke-Checked 'git checkout' { git -C $SeedVcDir checkout --quiet $SeedVcCommit }
}
Write-Host "Seed-VC source: $SeedVcDir @ $SeedVcCommit"

# 2. Optional: hardlink already-downloaded models from another workspace (source is only read).
if ($SeedFrom) {
    Invoke-Checked 'cache seeding' { python (Join-Path $ProjectRoot 'singing_vc\seed_caches.py') $SeedFrom }
}

# 3. Irodori environment via the existing setup (no interactive model picker).
if (-not $SkipIrodori) {
    Invoke-Checked 'Irodori setup' { powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $ProjectRoot 'setup_windows.ps1') -Backend $Backend -SkipModels }
    . (Join-Path $ProjectRoot 'scripts\project_env.ps1')
    & $ProjectPython scripts\download_models.py --local-files-only
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Irodori models not cached; downloading v4.1-Small runtime assets.'
        Invoke-Checked 'Irodori model download' { & $ProjectPython scripts\download_models.py }
    }
}

# 4. Seed-VC environment from the pinned lock file.
if (-not (Test-Path -LiteralPath $SeedVcPython)) {
    Invoke-Checked 'Seed-VC venv' { & $ProjectUv venv (Join-Path $SingingDir 'venv') --python 3.10 }
}
Invoke-Checked 'Seed-VC packages' {
    & $ProjectUv pip sync --python $SeedVcPython --index-strategy unsafe-best-match --system-certs configs\singing\seedvc-requirements.lock.txt
}

# 5. Seed-VC singing assets at pinned revisions (verified offline first).
$AssetArgs = @('singing_vc\seedvc_assets.py', '--seedvc-dir', $SeedVcDir, '--manifest', (Join-Path $SingingDir 'seedvc_assets.json'))
& $SeedVcPython @AssetArgs --local-files-only
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Seed-VC assets not cached; downloading pinned revisions.'
    Invoke-Checked 'Seed-VC asset download' { & $SeedVcPython @AssetArgs }
}
Invoke-Checked 'CUDA check' { & $SeedVcPython -c "import torch; print('Seed-VC torch', torch.__version__, 'CUDA', torch.cuda.is_available())" }
Write-Host 'Singing setup complete. Use run_singing.ps1 / run_singing.bat (CLI) or run_singing_webui.bat.'
