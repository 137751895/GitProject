param(
    [string]$Workspace = ".",
    [string]$Periods = "1min,5min,15min",
    [string]$FeaturesUsed = "all_features",
    [string]$KeyParams = "daily_batch",
    [string]$Notes = "scheduled batch run",
    [switch]$Strict,
    [string]$PipelineExtraArgs = ""
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$workspacePath = Resolve-Path $Workspace
Set-Location $workspacePath

$logsDir = Join-Path $workspacePath "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}

$lockPath = Join-Path $logsDir "daily_batch_update.lock"
$lockAcquired = $false

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logsDir "daily_batch_update_$timestamp.log"

function Write-Log {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [daily_batch] $Message"
    Write-Host $line
    Add-Content -Path $logFile -Value $line -Encoding utf8
}

function Convert-FileToUtf8 {
    param([string]$Path)

    if (-not (Test-Path $Path)) { return }

    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -eq 0) { return }

    $utf8Strict = [System.Text.UTF8Encoding]::new($false, $true)
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)

    try {
        $text = $utf8Strict.GetString($bytes)
    }
    catch {
        $ansiCodePage = [System.Globalization.CultureInfo]::CurrentCulture.TextInfo.ANSICodePage
        $ansiEncoding = [System.Text.Encoding]::GetEncoding($ansiCodePage)
        $text = $ansiEncoding.GetString($bytes)
    }

    [System.IO.File]::WriteAllText($Path, $text, $utf8NoBom)
}

function Acquire-BatchLock {
    param([string]$Path)

    if (Test-Path $Path) {
        $existing = Get-Content $Path -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($existing) {
            $existingPid = 0
            [void][int]::TryParse($existing.Trim(), [ref]$existingPid)
            if ($existingPid -gt 0) {
                $proc = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
                if ($proc) {
                    throw "Another daily batch is already running (PID=$existingPid); this run is blocked."
                }
            }
        }
        Remove-Item $Path -Force -ErrorAction SilentlyContinue
    }

    Set-Content -Path $Path -Value "$PID" -Encoding ascii
}

try {
    Acquire-BatchLock -Path $lockPath
    $lockAcquired = $true

    Write-Log "start"

    $oldNativePreference = $null
    if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue) {
        $oldNativePreference = $Global:PSNativeCommandUseErrorActionPreference
        $Global:PSNativeCommandUseErrorActionPreference = $false
    }

    $pythonCmd = "python"
    $pythonCandidates = @(
        (Join-Path $workspacePath ".venv\Scripts\python.exe"),
        (Join-Path $workspacePath "venv\Scripts\python.exe"),
        "D:\pyvenv\hfml\Scripts\python.exe"
    )
    foreach ($candidate in $pythonCandidates) {
        if (Test-Path $candidate) {
            $pythonCmd = $candidate
            break
        }
    }

    $scriptPath = Join-Path $workspacePath "scripts\update_experiment_log.py"
    if (-not (Test-Path $scriptPath)) {
        throw "script not found: $scriptPath"
    }

    $args = @(
        "-X", "utf8",
        $scriptPath,
        "--periods", $Periods,
        "--run-pipeline",
        "--features-used", $FeaturesUsed,
        "--key-params", $KeyParams,
        "--notes", $Notes
    )

    if ($Strict) {
        $args += "--strict"
    }

    if ($PipelineExtraArgs -and $PipelineExtraArgs.Trim().Length -gt 0) {
        $args += "--pipeline-extra-args"
        $args += $PipelineExtraArgs
    }

    Write-Log ("exec: {0} {1}" -f $pythonCmd, ($args -join " "))

    $stdoutPath = Join-Path $logsDir "daily_batch_stdout_$timestamp.log"
    $stderrPath = Join-Path $logsDir "daily_batch_stderr_$timestamp.log"

    $oldPythonUtf8 = $env:PYTHONUTF8
    $oldPythonIoEncoding = $env:PYTHONIOENCODING
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"

    $argString = ($args | ForEach-Object {
        if ($_ -match '\s') { '"' + ($_ -replace '"', '\\"') + '"' } else { $_ }
    }) -join ' '

    $proc = Start-Process -FilePath $pythonCmd `
        -ArgumentList $argString `
        -NoNewWindow `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath

    Convert-FileToUtf8 -Path $stdoutPath
    Convert-FileToUtf8 -Path $stderrPath

    if (Test-Path $stdoutPath) {
        Get-Content $stdoutPath -Encoding utf8 | Tee-Object -FilePath $logFile -Append -Encoding utf8
    }
    if (Test-Path $stderrPath) {
        Get-Content $stderrPath -Encoding utf8 | Tee-Object -FilePath $logFile -Append -Encoding utf8
    }

    if ($null -eq $oldPythonUtf8) { Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue } else { $env:PYTHONUTF8 = $oldPythonUtf8 }
    if ($null -eq $oldPythonIoEncoding) { Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue } else { $env:PYTHONIOENCODING = $oldPythonIoEncoding }

    if ($null -ne $oldNativePreference) {
        $Global:PSNativeCommandUseErrorActionPreference = $oldNativePreference
    }

    $exitCode = $proc.ExitCode

    if ($exitCode -ne 0) {
        Write-Log "failed with exit code $exitCode"
        if ($lockAcquired) { Remove-Item $lockPath -Force -ErrorAction SilentlyContinue }
        exit $exitCode
    }

    Write-Log "success"
    if ($lockAcquired) { Remove-Item $lockPath -Force -ErrorAction SilentlyContinue }
    exit 0
}
catch {
    Write-Log ("exception: " + $_.Exception.Message)
    Write-Log ("exception_detail: " + ($_ | Out-String).Trim())
    if ($_.ScriptStackTrace) {
        Write-Log ("script_stack: " + $_.ScriptStackTrace)
    }

    if (Get-Variable -Name oldPythonUtf8 -Scope Local -ErrorAction SilentlyContinue) {
        if ($null -eq $oldPythonUtf8) { Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue } else { $env:PYTHONUTF8 = $oldPythonUtf8 }
    }
    if (Get-Variable -Name oldPythonIoEncoding -Scope Local -ErrorAction SilentlyContinue) {
        if ($null -eq $oldPythonIoEncoding) { Remove-Item Env:PYTHONIOENCODING -ErrorAction SilentlyContinue } else { $env:PYTHONIOENCODING = $oldPythonIoEncoding }
    }
    if (Get-Variable -Name oldNativePreference -Scope Local -ErrorAction SilentlyContinue) {
        if ($null -ne $oldNativePreference) {
            $Global:PSNativeCommandUseErrorActionPreference = $oldNativePreference
        }
    }
    if ($lockAcquired) { Remove-Item $lockPath -Force -ErrorAction SilentlyContinue }
    exit 1
}
