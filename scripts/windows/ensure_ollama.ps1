<#
.SYNOPSIS
  Makes sure the local Ollama server (the story engine's last-resort model) is running, then lists
  the models it has. Starts `ollama serve` in the background if Ollama is installed but not running.
  Never fails the run: hosted models work without Ollama.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows\ensure_ollama.ps1
#>
$ErrorActionPreference = 'Continue'
$VersionUrl = 'http://127.0.0.1:11434/api/version'

function Get-OllamaVersion {
    try {
        return (Invoke-RestMethod -Uri $VersionUrl -TimeoutSec 3).version
    } catch {
        return $null
    }
}

function Find-Ollama {
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidate = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
    if (Test-Path -LiteralPath $candidate) { return $candidate }
    return $null
}

$exe = Find-Ollama
$version = Get-OllamaVersion
if (-not $version) {
    if (-not $exe) {
        Write-Output 'Ollama is not installed (optional: winget install Ollama.Ollama). Hosted models still work.'
        exit 0
    }
    Write-Output "Ollama is not running; starting it in the background: $exe serve"
    Start-Process -FilePath $exe -ArgumentList 'serve' -WindowStyle Hidden
    for ($i = 0; ($i -lt 30) -and (-not $version); $i++) {
        Start-Sleep -Seconds 1
        $version = Get-OllamaVersion
    }
}
if (-not $version) {
    Write-Output 'Ollama did not answer within 30 s; the story engine will use hosted models only.'
    exit 0
}
Write-Output "Ollama $version is running at http://127.0.0.1:11434"
if ($exe) {
    Write-Output ''
    Write-Output 'Installed local models (ollama list):'
    & $exe list
}
exit 0
