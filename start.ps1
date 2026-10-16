param(
    [int]$Port = 7861,
    [string]$HostName = "127.0.0.1",
    [switch]$SkipDoctor,
    [switch]$Help
)

$ScriptPath = Join-Path $PSScriptRoot "scripts\start.ps1"
& $ScriptPath -Port $Port -HostName $HostName -SkipDoctor:$SkipDoctor -Help:$Help
exit $LASTEXITCODE
