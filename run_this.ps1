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
$Round = 'M3 round 2: campfire narrator styles (bm_lewis), sound from the rendered frames, fixed loudness'
$Story = '20261018-163824-the-warm-hull'   # round 1's story: rendered now so its sound can come from the footage
$Steps = @(
    @{ Name = 'tests';          Cmd = 'uv run pytest -q -p no:cacheprovider' },
    # bm_lewis in four narrator treatments: listen and pick one
    @{ Name = 'voices';         Cmd = 'uv run unseen voices --voices bm_lewis --styles dry,campfire,campfire_fire,deep' },
    # which saved stories have rendered clips
    @{ Name = 'clips';          Cmd = 'uv run unseen clips' },
    @{ Name = 'kaggle-status';  Cmd = 'uv run unseen kaggle-status' },
    # renders its 10 scenes on the free Kaggle GPU at preview size (~40-45 min, waits and downloads)
    @{ Name = 'render';         Cmd = "uv run unseen story render $Story --kaggle" },
    # narration + sound chosen from the rendered frames + mastering to -14 LUFS
    @{ Name = 'audio';          Cmd = "uv run unseen audio $Story" }
)
$MaxLinesPerStep = 400
# ----------------------------------------------------------------------------------------------

$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath $PSScriptRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'   # Python writes each line immediately even into a pipe: live output
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
    # every line is shown and written to the log the moment it arrives (live output)
    $writer = New-Object System.IO.StreamWriter($log, $false, (New-Object System.Text.UTF8Encoding($false)))
    $writer.AutoFlush = $true
    try {
        cmd /c "chcp 65001 >nul & $Cmd 2>&1" | ForEach-Object {
            $line = [string]$_
            $writer.WriteLine($line)
            Write-Host $line
        }
    } finally {
        $writer.Close()
    }
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
    foreach ($line in (Get-Content -LiteralPath $Path -Encoding UTF8)) {
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

function Get-GeminiReset {
    # Gemini's free daily quotas (20 requests per model) reset at midnight US Pacific time
    try {
        $pacific = [System.TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow, 'Pacific Standard Time')
        $left = $pacific.Date.AddDays(1) - $pacific
        return ('{0}h {1:D2}m' -f [int][math]::Floor($left.TotalHours), $left.Minutes)
    } catch {
        return 'unknown'
    }
}

$GeminiReset = Get-GeminiReset
Write-Host ("Gemini's free daily quotas reset in $GeminiReset (midnight US Pacific). Stories written " +
            "while they are used up come from weaker models.") -ForegroundColor Yellow

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
    $Results += Invoke-Logged 'uv-sync' 'uv sync --extra audio' 0
} elseif ($AfterUpdate) {
    $Results += Invoke-Logged 'uv-sync' 'uv sync --extra audio' 0
}

# ---- 2. this round's steps ----------------------------------------------------------------------
$storiesBefore = Get-Dirs 'runs\stories'
$benchBefore = Get-Dirs 'bench_results'
$voicesBefore = Get-Dirs 'runs\voices'
$stepsStart = Get-Date
$i = 1
foreach ($step in $Steps) {
    if ($Only -and -not ($Only -contains $step.Name)) { continue }
    $Results += Invoke-Logged $step.Name $step.Cmd $i
    $i++
}
$newStories = @(Get-Dirs 'runs\stories' | Where-Object { $storiesBefore -notcontains $_ })
$newBench = @(Get-Dirs 'bench_results' | Where-Object { $benchBefore -notcontains $_ })
$newVoices = @(Get-Dirs 'runs\voices' | Where-Object { $voicesBefore -notcontains $_ })
$newAudio = @()
if (Test-Path -LiteralPath 'runs\stories') {
    $newAudio = @(Get-ChildItem -Path 'runs\stories\*\audio\audio_report.md' -ErrorAction SilentlyContinue |
                  Where-Object { $_.LastWriteTime -ge $stepsStart } | ForEach-Object { $_.FullName })
}

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
$r += "- Gemini free quotas reset (midnight US Pacific) in: $GeminiReset (at the start of the run)"
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
foreach ($dir in $newVoices) {
    $md = Join-Path $dir 'voices.md'
    if (Test-Path -LiteralPath $md) {
        $r += ''
        $r += "## New voice audition: $(Split-Path -Leaf $dir)"
        $r += ''
        $r += (Get-Content -LiteralPath $md -Encoding UTF8)
    }
}
foreach ($md in $newAudio) {
    $r += ''
    $r += "## New audio: $(Split-Path -Leaf (Split-Path -Parent (Split-Path -Parent $md)))"
    $r += ''
    $r += (Get-Content -LiteralPath $md -Encoding UTF8)
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
