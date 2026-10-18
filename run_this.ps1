<#
.SYNOPSIS
  Runs this round's checks and writes ONE structured report to paste back into the chat.

.DESCRIPTION
  1. git pull + uv sync (restarts itself if this script was updated by the pull)
  2. runs the steps listed in $Steps below (live output + one log file per step)
  3. writes run_logs\<timestamp>\REPORT.md with: commit, machine, configured API keys (names only,
     never values), each step's exit code / duration / output, and the story.md / report.md of
     anything created during the run
  4. copies REPORT.md to the clipboard

  The step list is updated every round. A failing step never stops the run.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\run_this.ps1
  powershell -ExecutionPolicy Bypass -File .\run_this.ps1 -Only story-fiction,kaggle-status
#>
[CmdletBinding()]
param(
    [switch]$SkipPull,
    [switch]$AfterUpdate,
    [string[]]$Only
)
# "-Only a,b" arrives as one string under -File: split it
$Only = @($Only | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })

# ----------------------------------------------------------------------------------------------
# THIS ROUND (edited by Claude each round)
# ----------------------------------------------------------------------------------------------
$Round = 'M2 round 7: model rotation per provider (best tier first), waits instead of failing, honest real cases'
$Steps = @(
    @{ Name = 'tests';          Cmd = 'uv run pytest -q -p no:cacheprovider' },
    # starts Ollama in the background if it is installed but not running (the last-resort model)
    @{ Name = 'ollama';         Cmd = 'powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\ensure_ollama.ps1' },
    # shows every provider's model rotation; pulls the new small local model (~5 GB, once)
    @{ Name = 'llm-setup';      Cmd = 'uv run unseen llm-setup' },
    # downloads the latest finished Kaggle run (the story render that was queued) -> bench_results\compare.html
    @{ Name = 'kaggle-fetch';   Cmd = 'uv run unseen kaggle-fetch' },
    @{ Name = 'story-real';     Cmd = 'uv run unseen story new --category real_mystery' },
    @{ Name = 'story-fiction';  Cmd = 'uv run unseen story new' }
)
$MaxLinesPerStep = 400
# ----------------------------------------------------------------------------------------------

$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath $PSScriptRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:NO_COLOR = '1'
$env:PY_COLORS = '0'

$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$LogDir = Join-Path $PSScriptRoot "run_logs\$Stamp"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$ReportPath = Join-Path $LogDir 'REPORT.md'
$Bar = [string][char]0x2588   # progress-bar glyph, filtered out of logs

function Invoke-Logged([string]$Name, [string]$Cmd, [int]$Index) {
    $log = Join-Path $LogDir ('{0:D2}-{1}.log' -f $Index, $Name)
    Write-Host ''
    Write-Host ('=== [{0}] {1}: {2}' -f $Index, $Name, $Cmd) -ForegroundColor Cyan
    $start = Get-Date
    # Out-Host: show live output without it becoming this function's return value
    cmd /c "chcp 65001 >nul & $Cmd 2>&1" | Tee-Object -FilePath $log | Out-Host
    $code = $LASTEXITCODE
    $secs = [int]((Get-Date) - $start).TotalSeconds
    $color = 'Green'
    if ($code -ne 0) { $color = 'Red' }
    Write-Host ('=== [{0}] {1}: exit {2} after {3} s' -f $Index, $Name, $code, $secs) -ForegroundColor $color
    return [pscustomobject]@{ Index = $Index; Name = $Name; Cmd = $Cmd; Exit = $code; Seconds = $secs; Log = $log }
}

function Get-CleanLines([string]$Path, [int]$Max) {
    if (-not (Test-Path -LiteralPath $Path)) { return @('(no output)') }
    $clean = @()
    foreach ($line in (Get-Content -LiteralPath $Path)) {
        $t = ($line -split "`r")[-1]
        $t = $t -replace '\x1B\[[0-9;?]*[A-Za-z]', ''
        if ($t.Contains($Bar)) { continue }
        $clean += $t
    }
    if ($clean.Count -gt $Max) {
        $omitted = $clean.Count - $Max
        $head = $clean[0..39]
        $tail = $clean[($clean.Count - ($Max - 40))..($clean.Count - 1)]
        return $head + @("... [$omitted lines omitted; full log in run_logs] ...") + $tail
    }
    return $clean
}

function Get-Dirs([string]$Path) {
    if (Test-Path -LiteralPath $Path) {
        return @(Get-ChildItem -LiteralPath $Path -Directory | ForEach-Object { $_.FullName })
    }
    return @()
}

# ---- 1. update (and restart if this script changed) -------------------------------------------
$Results = @()
if (-not $SkipPull) {
    $before = (Get-FileHash -LiteralPath $PSCommandPath).Hash
    $Results += Invoke-Logged 'git-pull' 'git pull --ff-only' 0
    $after = (Get-FileHash -LiteralPath $PSCommandPath).Hash
    if ($before -ne $after) {
        Write-Host ''
        Write-Host 'run_this.ps1 was updated by git pull; restarting the new version...' -ForegroundColor Yellow
        $argsList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $PSCommandPath, '-SkipPull', '-AfterUpdate')
        if ($Only) { $argsList += @('-Only', ($Only -join ',')) }
        & powershell @argsList
        exit $LASTEXITCODE
    }
    $Results += Invoke-Logged 'uv-sync' 'uv sync' 0
} elseif ($AfterUpdate) {
    $Results += Invoke-Logged 'uv-sync' 'uv sync' 0
}

# ---- 2. this round's steps ----------------------------------------------------------------------
$storiesBefore = Get-Dirs 'runs\stories'
$benchBefore = Get-Dirs 'bench_results'
$i = 1
foreach ($step in $Steps) {
    if ($Only -and -not ($Only -contains $step.Name)) { continue }
    $Results += Invoke-Logged $step.Name $step.Cmd $i
    $i++
}
$newStories = @(Get-Dirs 'runs\stories' | Where-Object { $storiesBefore -notcontains $_ })
$newBench = @(Get-Dirs 'bench_results' | Where-Object { $benchBefore -notcontains $_ })

# ---- 3. report ----------------------------------------------------------------------------------
$commit = (git log -1 --format='%h %s' 2>$null)
$gpu = (nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null)
$keys = @()
if (Test-Path -LiteralPath '.env') {
    foreach ($line in (Get-Content -LiteralPath '.env')) {
        if ($line -match '^\s*([A-Z0-9_]+)\s*=\s*(.*)$') {
            $value = $Matches[2].Trim()
            if ($value -and -not $value.StartsWith('your-')) { $keys += $Matches[1] }
        }
    }
}

$r = @()
$r += "# Run report: $Round"
$r += ''
$r += "- When: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$r += "- Commit: $commit"
$r += "- GPU: $gpu"
$r += "- Configured keys in .env (names only): $(if ($keys) { $keys -join ', ' } else { 'none' })"
$r += ''
$r += '## Summary'
$r += ''
$r += '| # | step | exit | seconds |'
$r += '|---|---|---|---|'
foreach ($res in $Results) {
    $status = 'ok'
    if ($res.Exit -ne 0) { $status = "FAILED ($($res.Exit))" }
    $r += "| $($res.Index) | $($res.Name) | $status | $($res.Seconds) |"
}
foreach ($res in $Results) {
    $r += ''
    $r += "## [$($res.Index)] $($res.Name) (exit $($res.Exit), $($res.Seconds) s)"
    $r += ''
    $r += "Command: ``$($res.Cmd)``"
    $r += ''
    $r += '```'
    $r += Get-CleanLines $res.Log $MaxLinesPerStep
    $r += '```'
}
foreach ($dir in $newStories) {
    $md = Join-Path $dir 'story.md'
    if (Test-Path -LiteralPath $md) {
        $r += ''
        $r += "## New story: $(Split-Path -Leaf $dir)"
        $r += ''
        $r += (Get-Content -LiteralPath $md -Encoding UTF8)
    }
}
foreach ($dir in $newBench) {
    $md = Join-Path $dir 'report.md'
    if (Test-Path -LiteralPath $md) {
        $r += ''
        $r += "## New benchmark run: $(Split-Path -Leaf $dir)"
        $r += ''
        $r += (Get-Content -LiteralPath $md -Encoding UTF8 | Where-Object { $_ -notmatch '^!\[' })
    }
}
$r -join "`r`n" | Out-File -FilePath $ReportPath -Encoding utf8

Write-Host ''
Write-Host "Report: $ReportPath" -ForegroundColor Green
try {
    Get-Content -LiteralPath $ReportPath -Raw -Encoding UTF8 | Set-Clipboard
    Write-Host 'REPORT.md has been copied to the clipboard: paste it into the chat.' -ForegroundColor Green
} catch {
    Write-Host 'Could not copy to the clipboard; open the report file and paste its contents.' -ForegroundColor Yellow
}
