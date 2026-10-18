<#
.SYNOPSIS
  Removes local Ollama models the project no longer uses, to free disk space. Only the names given
  in -Models are touched, and never the model config/story.yaml uses. Needs Ollama running (the
  `ollama` step of run_this.ps1 starts it).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows\remove_ollama_models.ps1 -Models llama3.2:3b
#>
param([string[]]$Models)
$ErrorActionPreference = 'Continue'
# "-Models a,b" arrives as one string under -File: split it
$Models = @($Models | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })

function Find-Ollama {
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $candidate = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
    if (Test-Path -LiteralPath $candidate) { return $candidate }
    return $null
}

function Get-NormalName([string]$Name) {
    return ($Name -replace ':latest$', '').ToLowerInvariant()
}

$exe = Find-Ollama
if (-not $exe) {
    Write-Output 'Ollama is not installed; nothing to remove.'
    exit 0
}

# the local model the story engine uses (llm.ollama.model in config/story.yaml) is never removed
$configured = $null
$cfgPath = Join-Path $PSScriptRoot '..\..\config\story.yaml'
if (Test-Path -LiteralPath $cfgPath) {
    foreach ($line in (Get-Content -LiteralPath $cfgPath)) {
        if ($line -match '^\s+model:\s*(\S+)') {
            $configured = $Matches[1]
            break
        }
    }
}

$installed = @(& $exe list 2>$null | Select-Object -Skip 1 | ForEach-Object { ($_ -split '\s+')[0] } |
               Where-Object { $_ })
foreach ($m in $Models) {
    if ($configured -and ((Get-NormalName $m) -eq (Get-NormalName $configured))) {
        Write-Output ('Keeping {0}: config/story.yaml uses it.' -f $m)
        continue
    }
    $match = $installed | Where-Object { (Get-NormalName $_) -eq (Get-NormalName $m) } | Select-Object -First 1
    if (-not $match) {
        Write-Output ('{0} is not installed; nothing to do.' -f $m)
        continue
    }
    Write-Output ('Removing {0} ...' -f $match)
    & $exe rm $match
}
Write-Output ''
Write-Output 'Local models now (ollama list):'
& $exe list
exit 0
