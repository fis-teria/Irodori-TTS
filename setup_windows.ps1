[CmdletBinding()]
param(
    [ValidateSet('cu128', 'cpu', 'xpu')][string]$Backend = 'cu128',
    [string]$ModelRepo = 'Aratako/Irodori-TTS-v4.1-Small',
    [switch]$SkipModels,
    [switch]$SkipSilentCipher
)

$ErrorActionPreference = 'Stop'
$RequestedModelRepos = @($ModelRepo)
if (-not $SkipModels -and -not $PSBoundParameters.ContainsKey('ModelRepo')) {
    . (Join-Path $PSScriptRoot 'scripts\select_setup_models.ps1')
    $RequestedModelRepos = @(Select-IrodoriSetupModels -ProjectRoot $PSScriptRoot)
    if ($RequestedModelRepos.Count -eq 0) {
        Write-Host 'Setup cancelled.'
        exit 0
    }
}

. (Join-Path $PSScriptRoot 'scripts\project_env.ps1')
Set-Location -LiteralPath $ProjectRoot

function Get-ArchiveSha256([string]$Path) {
    $HashAlgorithm = [Security.Cryptography.SHA256]::Create()
    $ArchiveStream = [IO.File]::OpenRead($Path)
    try {
        return [BitConverter]::ToString($HashAlgorithm.ComputeHash($ArchiveStream)).Replace('-', '').ToLowerInvariant()
    } finally {
        $ArchiveStream.Dispose()
        $HashAlgorithm.Dispose()
    }
}

Add-Type -AssemblyName System.IO.Compression.FileSystem

# Pin a workspace-owned uv; the machine's uv 0.6 cannot read the V4 lock format.
$UvVersion = '0.13.5'
$UvSha256 = 'a38b8d1902b7cfbc76b423ba1bb456f623f42064cdc72628b6075b89afd1019a'
if (-not (Test-Path -LiteralPath $ProjectUv)) {
    $UvDir = Split-Path -Parent $ProjectUv
    New-Item -ItemType Directory -Force -Path $UvDir | Out-Null
    $Archive = Join-Path $ProjectRoot '.cache\uv-windows.zip'
    Write-Host "Downloading workspace uv $UvVersion"
    Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" -OutFile $Archive
    if ((Get-ArchiveSha256 $Archive) -ne $UvSha256) {
        throw 'uv archive SHA256 mismatch.'
    }
    [IO.Compression.ZipFile]::ExtractToDirectory($Archive, $UvDir)
}

# TorchCodec and dataset audio decoding require FFmpeg's shared DLLs on Windows.
$FfmpegDir = Join-Path $ProjectRoot '.tools\ffmpeg'
if (-not (Test-Path -LiteralPath (Join-Path $FfmpegDir 'bin\ffmpeg.exe'))) {
    $FfmpegFolder = 'ffmpeg-n8.1.3-24-g92611900bb-win64-lgpl-shared-8.1'
    $FfmpegArchive = Join-Path $ProjectRoot '.cache\ffmpeg-windows.zip'
    $FfmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-23-13-04/$FfmpegFolder.zip"
    Write-Host 'Downloading workspace FFmpeg 8.1 shared runtime'
    Invoke-WebRequest -UseBasicParsing -Uri $FfmpegUrl -OutFile $FfmpegArchive
    if ((Get-ArchiveSha256 $FfmpegArchive) -ne 'e7827a3897d0914b538c14efb8dc2d88020a4fd2b593bad3dbd668a691d076ad') {
        throw 'FFmpeg archive SHA256 mismatch.'
    }
    $ToolsDir = Join-Path $ProjectRoot '.tools'
    [IO.Compression.ZipFile]::ExtractToDirectory($FfmpegArchive, $ToolsDir)
    $ExtractedFfmpeg = [IO.Path]::GetFullPath((Join-Path $ToolsDir $FfmpegFolder))
    $WorkspaceBoundary = $ProjectRoot + [IO.Path]::DirectorySeparatorChar
    foreach ($MovePath in @($ExtractedFfmpeg, [IO.Path]::GetFullPath($FfmpegDir))) {
        if (-not $MovePath.StartsWith($WorkspaceBoundary, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'FFmpeg installation path is outside the workspace.'
        }
    }
    Move-Item -LiteralPath $ExtractedFfmpeg -Destination $FfmpegDir
    . (Join-Path $ProjectRoot 'scripts\project_env.ps1')
}

$SyncArgs = @('sync', '--frozen', '--system-certs', '--extra', $Backend)
if ($SkipSilentCipher) { $SyncArgs += @('--no-install-package', 'silentcipher') }
Write-Host "Syncing $ProjectRoot\.venv ($Backend)"
& $ProjectUv @SyncArgs
if ($LASTEXITCODE -ne 0) { throw "uv sync failed: $LASTEXITCODE" }

if (-not $SkipModels) {
    $DownloadArgs = @('scripts/download_models.py')
    foreach ($RequestedModelRepo in $RequestedModelRepos) {
        $DownloadArgs += @('--model-repo', $RequestedModelRepo)
    }
    if ($SkipSilentCipher) { $DownloadArgs += '--skip-silentcipher' }
    & $ProjectPython @DownloadArgs
    if ($LASTEXITCODE -ne 0) { throw "Model download failed: $LASTEXITCODE" }
}
& $ProjectPython scripts/check_environment.py
if ($LASTEXITCODE -ne 0) { throw "Environment check failed: $LASTEXITCODE" }
Write-Host 'Setup complete. Use run_python.bat or the WebUI launchers.'
