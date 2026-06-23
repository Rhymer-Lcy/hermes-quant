# Unattended daily IF minute-bar accumulator wrapper for Windows Task Scheduler.
# Tiny + isolated from the paper job (its failure must never touch the paper record). Logs to a
# timestamped file and propagates the exit code. Register weekdays after futures close (~15:40):
#   schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 15:40 /TN hermes-if-accum `
#     /TR "powershell -NoProfile -ExecutionPolicy Bypass -File <repo>\scripts\accumulate_if_minute.ps1"
# Portable: repo root derived from this script's location; python from HERMES_PYTHON (set at user scope
# by `schedule_tasks.ps1 register`), else `python` on PATH for a manually-run, env-activated invocation.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = if ($env:HERMES_PYTHON) { $env:HERMES_PYTHON } else { "python" }
$logdir = Join-Path $repo "results\paper\logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$log = Join-Path $logdir ("if_accum_{0:yyyyMMdd}.log" -f (Get-Date))
# Capture the child's output to a UTF-8 log with no stream-object corruption: PYTHONIOENCODING
# makes Python emit UTF-8; Start-Process writes those raw bytes to temp files (no PowerShell
# re-encoding and no stderr-as-error wrapping); both are then appended as UTF-8, in order.
$env:PYTHONIOENCODING = "utf-8"
"=== run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Append -Encoding utf8
$out = [System.IO.Path]::GetTempFileName(); $err = [System.IO.Path]::GetTempFileName()
$proc = Start-Process -FilePath $py -ArgumentList "`"$(Join-Path $repo 'scripts\accumulate_if_minute.py')`"" `
  -NoNewWindow -Wait -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
Get-Content -LiteralPath $out, $err -Encoding UTF8 | Out-File -FilePath $log -Append -Encoding utf8
Remove-Item -LiteralPath $out, $err -ErrorAction SilentlyContinue
exit $proc.ExitCode
