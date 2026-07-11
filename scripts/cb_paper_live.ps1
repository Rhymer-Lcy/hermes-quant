# Unattended daily CB paper wrapper for Windows Task Scheduler.
# Mirrors paper_live.ps1: captures stdout+stderr to a timestamped UTF-8 log (Task Scheduler
# discards them otherwise) and propagates the Python exit code. Registered by
# scripts\schedule_tasks.ps1 (hermes-cb-paper, 19:40 Beijing weekdays -- after hermes-paper,
# and after Eastmoney/Sina post the day's EOD data).
#
# RETRY-WITH-BACKOFF: cb_paper_live.py exits 75 (EX_TEMPFAIL) when the pull was transiently
# unreachable or degraded, and 1 on a fatal error. On 75 this wrapper waits and retries, up
# to HERMES_CB_RETRY_MAX attempts spaced HERMES_CB_RETRY_DELAY_SEC apart (default 6 x 600 s
# ~= 1 h -- the incremental refresh itself takes ~10-15 min, so retries are spaced wider
# than the equity wrapper's). Each run is idempotent (recompute-from-inception).
# No DNS pinning here: the pull targets Eastmoney/Sina, which the system resolver handles;
# a residual failure is covered by the retry loop and by tomorrow's recompute.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = if ($env:HERMES_PYTHON) { $env:HERMES_PYTHON } else { "python" }
$logdir = Join-Path $repo "results\paper\logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$log = Join-Path $logdir ("cb_paper_{0:yyyyMMdd}.log" -f (Get-Date))

$maxAttempts = if ($env:HERMES_CB_RETRY_MAX) { [int]$env:HERMES_CB_RETRY_MAX } else { 6 }
$delaySec    = if ($env:HERMES_CB_RETRY_DELAY_SEC) { [int]$env:HERMES_CB_RETRY_DELAY_SEC } else { 600 }
$EX_TEMPFAIL = 75

# PYTHONIOENCODING=utf-8 so Python emits UTF-8; Start-Process writes raw child output to temp
# files (no PowerShell re-encoding, no stderr-as-error wrapping); both are appended in order.
$env:PYTHONIOENCODING = "utf-8"
$script = Join-Path $repo 'scripts\cb_paper_live.py'

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
    "transient failure (exit 75); retrying in $delaySec s (attempt $attempt/$maxAttempts) ..." |
      Out-File -FilePath $log -Append -Encoding utf8
    Start-Sleep -Seconds $delaySec
  }
}
"retries exhausted ($maxAttempts attempts); the sources never became reachable. StartWhenAvailable plus the next scheduled run will catch up (recompute-from-inception)." |
  Out-File -FilePath $log -Append -Encoding utf8
exit $EX_TEMPFAIL
