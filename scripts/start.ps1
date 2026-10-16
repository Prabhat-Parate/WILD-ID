<#
.SYNOPSIS
    Starts the Wild-ID server with configurable port, UTF-8 environment, and diagnostic pre-flight check.

.DESCRIPTION
    1. Sets UTF-8 environment variables.
    2. Runs scripts/doctor.py to verify environment and dependencies.
    3. Launches demo.py on 127.0.0.1 with the requested port.

.PARAMETER Port
    Port number to bind (default: 7861).

.PARAMETER HostName
    Host address to bind (default: 127.0.0.1).

.PARAMETER SkipDoctor
    Skip the pre-flight doctor check.

.EXAMPLE
    .\scripts\start.ps1 -Port 7861
    .\scripts\start.ps1 -Port 8080 -SkipDoctor
#>

param(
    [int]$Port = 7861,
    [string]$HostName = "127.0.0.1",
    [switch]$SkipDoctor,
    [switch]$Help
)

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Full
    exit 0
}

# Ensure repository root is working directory
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $RepoRoot

# Enforce UTF-8 in PowerShell and Python child processes
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  WILD-ID TIGER RE-IDENTIFICATION SYSTEM (STARTUP)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Repository Root : $RepoRoot"
Write-Host "  Target Host     : $HostName"
Write-Host "  Target Port     : $Port"
Write-Host ""

# Locate Python executable
$VenvPython = Join-Path $RepoRoot "venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $PythonExe = $VenvPython
    Write-Host "  [OK] Using virtualenv Python: $PythonExe" -ForegroundColor Green
} else {
    $PythonExe = "python"
    Write-Host "  [WARN] Virtualenv Python not found at venv\Scripts\python.exe, using system python" -ForegroundColor Yellow
}

# Run pre-flight diagnostic check
if (-not $SkipDoctor) {
    Write-Host "`n--> Running pre-flight diagnostic check (doctor.py)..." -ForegroundColor Yellow
    & $PythonExe "scripts\doctor.py" --host $HostName --port $Port
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[ERROR] Doctor diagnostic failed with exit code $LASTEXITCODE." -ForegroundColor Red
        Write-Host "Fix the errors above, or re-run with -SkipDoctor if you wish to override." -ForegroundColor Red
        exit $LASTEXITCODE
    }
} else {
    Write-Host "  [INFO] Pre-flight doctor check skipped by user flag." -ForegroundColor DarkYellow
}

Write-Host "`n--> Launching Wild-ID Server on http://${HostName}:${Port}..." -ForegroundColor Green
Write-Host "    Press Ctrl+C to stop the server.`n" -ForegroundColor DarkGray

& $PythonExe "demo.py" --host $HostName --port $Port
exit $LASTEXITCODE
