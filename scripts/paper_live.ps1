# Unattended daily paper-trading wrapper for Windows Task Scheduler.
# Captures stdout+stderr to a timestamped logfile (Task Scheduler discards them otherwise)
# and propagates the Python exit code (nonzero on a degraded pull or crash -> visible as the
# task's "last result"). Register it with:
#   schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 15:35 /TN hermes-paper `
#     /TR "powershell -NoProfile -ExecutionPolicy Bypass -File F:\hermes-quant\scripts\paper_live.ps1"
# Portable: repo root is derived from this script's location (scripts/ -> repo); the python
# interpreter is overridable via the HERMES_PYTHON env var (falls back to the conda env path).
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = if ($env:HERMES_PYTHON) { $env:HERMES_PYTHON } else { "D:\Anaconda3\envs\hermes\python.exe" }
$logdir = Join-Path $repo "results\paper\logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$log = Join-Path $logdir ("paper_{0:yyyyMMdd}.log" -f (Get-Date))
# Capture the child's output to a UTF-8 log with no stream-object corruption: PYTHONIOENCODING
# makes Python emit UTF-8; Start-Process writes those raw bytes to temp files (no PowerShell
# re-encoding and no stderr-as-error wrapping); both are then appended as UTF-8, in order.
$env:PYTHONIOENCODING = "utf-8"
"=== run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Append -Encoding utf8
$out = [System.IO.Path]::GetTempFileName(); $err = [System.IO.Path]::GetTempFileName()
$proc = Start-Process -FilePath $py -ArgumentList "`"$(Join-Path $repo 'scripts\paper_live.py')`"" `
  -NoNewWindow -Wait -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
Get-Content -LiteralPath $out, $err -Encoding UTF8 | Out-File -FilePath $log -Append -Encoding utf8
Remove-Item -LiteralPath $out, $err -ErrorAction SilentlyContinue
exit $proc.ExitCode
