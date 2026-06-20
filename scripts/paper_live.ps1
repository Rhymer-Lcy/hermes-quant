# Unattended daily paper-trading wrapper for Windows Task Scheduler.
# Captures stdout+stderr to a timestamped logfile (Task Scheduler discards them otherwise)
# and propagates the Python exit code (nonzero on a degraded pull or crash -> visible as the
# task's "last result"). Register it with:
#   schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 15:35 /TN hermes-paper `
#     /TR "powershell -NoProfile -ExecutionPolicy Bypass -File F:\hermes-quant\scripts\paper_live.ps1"
$ErrorActionPreference = "Stop"
$repo = "F:\hermes-quant"
$py = "D:\Anaconda3\envs\hermes\python.exe"
$logdir = Join-Path $repo "results\paper\logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$log = Join-Path $logdir ("paper_{0:yyyyMMdd}.log" -f (Get-Date))
"=== run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Append -Encoding utf8
& $py (Join-Path $repo "scripts\paper_live.py") *>> $log
exit $LASTEXITCODE
