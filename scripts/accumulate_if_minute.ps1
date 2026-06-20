# Unattended daily IF minute-bar accumulator wrapper for Windows Task Scheduler.
# Tiny + isolated from the paper job (its failure must never touch the paper record). Logs to a
# timestamped file and propagates the exit code. Register weekdays after futures close (~15:40):
#   schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 15:40 /TN hermes-if-accum `
#     /TR "powershell -NoProfile -ExecutionPolicy Bypass -File F:\hermes-quant\scripts\accumulate_if_minute.ps1"
# Portable: repo root derived from this script's location; python overridable via HERMES_PYTHON env var.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = if ($env:HERMES_PYTHON) { $env:HERMES_PYTHON } else { "D:\Anaconda3\envs\hermes\python.exe" }
$logdir = Join-Path $repo "results\paper\logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$log = Join-Path $logdir ("if_accum_{0:yyyyMMdd}.log" -f (Get-Date))
"=== run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Append -Encoding utf8
& $py (Join-Path $repo "scripts\accumulate_if_minute.py") *>> $log
exit $LASTEXITCODE
