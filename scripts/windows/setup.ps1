<#
.SYNOPSIS
  One-time setup for The Unseen Archive on Windows 11 (Milestone 1: video benchmark).

.DESCRIPTION
  1. Checks git, ffmpeg/ffprobe and the NVIDIA driver (nvidia-smi).
  2. Installs uv (Python package/venv manager) with winget if it is missing.
  3. Creates the app environment (.venv, Python 3.12) with `uv sync --extra audio` (audio = M3 TTS).
  4. Installs the pinned ComfyUI + ComfyUI-GGUF and a separate CUDA PyTorch venv
     (runtime\comfy-venv, Python 3.13) via `uv run unseen setup-comfy`.
  5. Runs `uv run unseen doctor`.
  Safe to re-run; finished steps are verified and skipped.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
#>
[CmdletBinding()]
param(
    [switch]$SkipComfy
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $RepoRoot

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Stop-Setup([string]$Message) {
    Write-Host ""
    Write-Host "SETUP FAILED: $Message" -ForegroundColor Red
    exit 1
}

function Test-Tool([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    Write-Host "> $Exe $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        Stop-Setup "'$Exe $($Arguments -join ' ')' exited with code $LASTEXITCODE"
    }
}

function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
}

Write-Step "Repository: $RepoRoot"
if ($RepoRoot.Length -gt 80) {
    Write-Host "Note: the repo path is long; if you hit path-length errors, clone to e.g. C:\dev\TheUnseenArchive" -ForegroundColor Yellow
}

Write-Step "Checking required tools"
if (-not (Test-Tool 'git')) { Stop-Setup "git not found. Install it: winget install Git.Git  (then reopen PowerShell)" }
Write-Host "git: ok"
if (-not (Test-Tool 'ffmpeg') -or -not (Test-Tool 'ffprobe')) {
    Stop-Setup "ffmpeg/ffprobe not on PATH. Install: winget install Gyan.FFmpeg  (then reopen PowerShell)"
}
Write-Host "ffmpeg: ok"
if (-not (Test-Tool 'nvidia-smi')) { Stop-Setup "nvidia-smi not found. Install the latest NVIDIA driver from nvidia.com" }
$smi = (& nvidia-smi) -join "`n"
if ($smi -match 'CUDA Version:\s*([\d\.]+)') {
    Write-Host "NVIDIA driver supports CUDA $($Matches[1]) (ComfyUI's PyTorch needs 13.0 or newer)"
} else {
    Write-Host "Could not read the CUDA version from nvidia-smi" -ForegroundColor Yellow
}

Write-Step "Checking uv"
if (-not (Test-Tool 'uv')) {
    if (-not (Test-Tool 'winget')) {
        Stop-Setup "uv not found and winget unavailable. Install uv from https://docs.astral.sh/uv/getting-started/installation/ and re-run."
    }
    Invoke-Checked 'winget' @('install', '--id', 'astral-sh.uv', '-e', '--accept-source-agreements', '--accept-package-agreements')
    Update-SessionPath
    if (-not (Test-Tool 'uv')) { Stop-Setup "uv was installed but is not on PATH yet. Close and reopen PowerShell, then re-run this script." }
}
Invoke-Checked 'uv' @('--version')

Write-Step "Creating the app environment (.venv, Python 3.12)"
Invoke-Checked 'uv' @('sync', '--extra', 'audio')

if (-not $SkipComfy) {
    Write-Step "Installing pinned ComfyUI + CUDA PyTorch (first run downloads ~4 GB)"
    Invoke-Checked 'uv' @('run', 'unseen', 'setup-comfy')
}

Write-Step "Running doctor"
& uv run unseen doctor
$doctorExit = $LASTEXITCODE

Write-Host ""
Write-Host "IMPORTANT (one-time, manual): stop the driver from silently using system RAM as VRAM." -ForegroundColor Yellow
Write-Host "  NVIDIA Control Panel > Manage 3D settings > Program Settings > Add >"
Write-Host "  browse to $RepoRoot\runtime\comfy-venv\Scripts\python.exe"
Write-Host "  then set 'CUDA - Sysmem Fallback Policy' = 'Prefer No Sysmem Fallback' > Apply."
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Green
Write-Host "  uv run pytest"
Write-Host "  uv run unseen fetch-models --set ltxv"
Write-Host "  uv run unseen bench-video --model ltxv --quick"
Write-Host "(see README.md, section 'Milestone 1')"
if ($doctorExit -ne 0) {
    Write-Host ""
    Write-Host "Doctor reported failures above; fix them before benchmarking. Missing models are expected until you run fetch-models." -ForegroundColor Yellow
}
