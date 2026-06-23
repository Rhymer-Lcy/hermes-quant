# Unattended daily paper-trading wrapper for Windows Task Scheduler.
# Captures stdout+stderr to a timestamped logfile (Task Scheduler discards them otherwise)
# and propagates the Python exit code (nonzero on a degraded pull or crash -> visible as the
# task's "last result"). Register it with:
#   schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 19:00 /TN hermes-paper `
#     /TR "powershell -NoProfile -ExecutionPolicy Bypass -File F:\hermes-quant\scripts\paper_live.ps1"
# Portable: repo root is derived from this script's location (scripts/ -> repo); the python
# interpreter is overridable via the HERMES_PYTHON env var (falls back to the conda env path).
#
# RETRY-WITH-BACKOFF: paper_live.py exits 75 (EX_TEMPFAIL) for a TRANSIENT data failure -- BaoStock
# unreachable (e.g. a VPN blackholing it at the scheduled time) or still mid-publication -- and 1 for
# a fatal error. On exit 75 this wrapper waits and retries, up to HERMES_RETRY_MAX attempts spaced
# HERMES_RETRY_DELAY_SEC apart (default 24 x 300 s ~= 2 h), so a run blocked at 19:00 self-heals the
# moment connectivity returns (or the VPN is briefly bypassed) WITHOUT a manual re-trigger. A fatal
# error (any nonzero != 75) or success (0) returns immediately. Each python run is idempotent
# (recompute-from-seed), so retrying is safe.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = if ($env:HERMES_PYTHON) { $env:HERMES_PYTHON } else { "D:\Anaconda3\envs\hermes\python.exe" }
$logdir = Join-Path $repo "results\paper\logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$log = Join-Path $logdir ("paper_{0:yyyyMMdd}.log" -f (Get-Date))

$maxAttempts = if ($env:HERMES_RETRY_MAX) { [int]$env:HERMES_RETRY_MAX } else { 24 }
$delaySec    = if ($env:HERMES_RETRY_DELAY_SEC) { [int]$env:HERMES_RETRY_DELAY_SEC } else { 300 }
$EX_TEMPFAIL = 75

# Capture the child's output to a UTF-8 log with no stream-object corruption: PYTHONIOENCODING
# makes Python emit UTF-8; Start-Process writes those raw bytes to temp files (no PowerShell
# re-encoding and no stderr-as-error wrapping); both are then appended as UTF-8, in order.
$env:PYTHONIOENCODING = "utf-8"
$script = Join-Path $repo 'scripts\paper_live.py'

for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
  "=== run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') (attempt $attempt/$maxAttempts) ===" |
    Out-File -FilePath $log -Append -Encoding utf8
  $out = [System.IO.Path]::GetTempFileName(); $err = [System.IO.Path]::GetTempFileName()
  $proc = Start-Process -FilePath $py -ArgumentList "`"$script`"" `
    -NoNewWindow -Wait -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
  Get-Content -LiteralPath $out, $err -Encoding UTF8 | Out-File -FilePath $log -Append -Encoding utf8
  Remove-Item -LiteralPath $out, $err -ErrorAction SilentlyContinue

  if ($proc.ExitCode -ne $EX_TEMPFAIL) { exit $proc.ExitCode }   # success (0) or fatal (!=75) -> done
  if ($attempt -lt $maxAttempts) {
    "transient data failure (exit 75); retrying in $delaySec s (attempt $attempt/$maxAttempts) ..." |
      Out-File -FilePath $log -Append -Encoding utf8
    Start-Sleep -Seconds $delaySec
  }
}
"retries exhausted ($maxAttempts attempts); BaoStock never became reachable. StartWhenAvailable plus the next scheduled run will catch up (recompute-from-seed)." |
  Out-File -FilePath $log -Append -Encoding utf8
exit $EX_TEMPFAIL
