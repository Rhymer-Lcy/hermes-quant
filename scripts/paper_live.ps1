# Unattended daily paper-trading wrapper for Windows Task Scheduler.
# Captures stdout+stderr to a timestamped logfile (Task Scheduler discards them otherwise)
# and propagates the Python exit code (nonzero on a degraded pull or crash -> visible as the
# task's "last result"). Register it with:
#   schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 19:00 /TN hermes-paper `
#     /TR "powershell -NoProfile -ExecutionPolicy Bypass -File <repo>\scripts\paper_live.ps1"
# Portable: repo root is derived from this script's location (scripts/ -> repo); the python
# interpreter comes from HERMES_PYTHON (set at user scope by `schedule_tasks.ps1 register`, which a
# scheduled task inherits), else `python` on PATH for a manually-run, env-activated invocation.
#
# RETRY-WITH-BACKOFF: paper_live.py exits 75 (EX_TEMPFAIL) for a residual TRANSIENT data failure -- a
# server-side blip during the publication window, a momentary network drop, or a mid-publication run
# -- and 1 for a fatal error. (The common VPN-breaks-DNS failure is handled earlier by resolving and
# pinning the server IP; see below.) On exit 75 this wrapper waits and retries, up to HERMES_RETRY_MAX
# attempts spaced HERMES_RETRY_DELAY_SEC apart (default 24 x 300 s ~= 2 h), self-healing without a
# manual re-trigger. A fatal error (any nonzero != 75) or success (0) returns immediately. Each python
# run is idempotent (recompute-from-seed), so retrying is safe.
#
# WATCHDOG: an attempt that has not exited after HERMES_ATTEMPT_TIMEOUT_SEC (default 7200 s, about
# twice the slowest complete run on record, 3380 s on 2026-08-26) is killed and treated as exit 75.
# A hung attempt otherwise blocks forever: on 2026-10-22 one produced no output for 6.8 h, before
# its first lake write, until the machine was shut down (most likely a stalled BaoStock socket; the
# client sets no timeout). The task is registered with MultipleInstances=IgnoreNew, so a hang that
# outlives the day would also suppress the next day's run. No attempt starts once
# HERMES_RETRY_BUDGET_SEC (default 72000 s = 20 h) has elapsed, so even a series of hung attempts
# ends within about 22 h. Python runs unbuffered so a killed attempt still leaves its partial
# output in the log. Killing is safe: every lake and ledger write of the canonical run is atomic
# and each run recomputes from the seed. The D=5 shadow runs after the attempt, outside the watchdog.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = if ($env:HERMES_PYTHON) { $env:HERMES_PYTHON } else { "python" }
$logdir = Join-Path $repo "results\paper\logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$log = Join-Path $logdir ("paper_{0:yyyyMMdd}.log" -f (Get-Date))
"=== wrapper start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') pid $PID ===" |
  Out-File -FilePath $log -Append -Encoding utf8

# D=5 FORWARD SHADOW (issue #22) -- strictly subordinate to the canonical record.
# Called ONLY after the canonical run exits 0, and its result is deliberately discarded: the shadow
# is an experiment racing a historical champion that FAILED issue #21's battery, and it must never
# be able to fail, delay or invalidate this task. It does not refresh the lake (the canonical run
# already did) and writes only under results/paper_shadow/d5/.
# Disable with HERMES_SHADOW_D5=0; the canonical path is unchanged either way.
function Invoke-ShadowD5 {
  param($repo, $py)
  try {
    $shadow = Join-Path $repo 'scripts\paper_shadow_d5.ps1'
    if (Test-Path $shadow) {
      & powershell -NoProfile -ExecutionPolicy Bypass -File $shadow | Out-Null
    }
  } catch { }        # a shadow fault is logged by its own wrapper and ignored here, by design
}

$maxAttempts = if ($env:HERMES_RETRY_MAX) { [int]$env:HERMES_RETRY_MAX } else { 24 }
$delaySec    = if ($env:HERMES_RETRY_DELAY_SEC) { [int]$env:HERMES_RETRY_DELAY_SEC } else { 300 }
$timeoutSec  = if ($env:HERMES_ATTEMPT_TIMEOUT_SEC) { [int]$env:HERMES_ATTEMPT_TIMEOUT_SEC } else { 7200 }
$budgetSec   = if ($env:HERMES_RETRY_BUDGET_SEC) { [int]$env:HERMES_RETRY_BUDGET_SEC } else { 72000 }
$started     = Get-Date
$EX_TEMPFAIL = 75

# Capture the child's output to a UTF-8 log with no stream-object corruption: PYTHONIOENCODING
# makes Python emit UTF-8; Start-Process writes those raw bytes to temp files (no PowerShell
# re-encoding and no stderr-as-error wrapping); both are then appended as UTF-8, in order.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$script = Join-Path $repo 'scripts\paper_live.py'

# Pin BaoStock's server IP so the pull works even when a corporate VPN (e.g. Sangfor SSL) breaks DNS
# for public-api.baostock.com while leaving the server's physical route intact. Resolve via a PUBLIC
# DNS server (the VPN does not block UDP/53 to these) and pass the IP to Python via HERMES_BAOSTOCK_IP.
# Best-effort: if every public resolver is unreachable, Python falls back to its cached/seed IP. A
# manual `python paper_live.py` resolves the same way inside session().
foreach ($dns in '223.5.5.5', '119.29.29.29', '8.8.8.8') {
  try {
    $a = Resolve-DnsName -Server $dns -Name public-api.baostock.com -Type A -DnsOnly -ErrorAction Stop |
      Where-Object { $_.Type -eq 'A' } | Select-Object -First 1
    if ($a -and $a.IPAddress) {
      $env:HERMES_BAOSTOCK_IP = $a.IPAddress
      "resolved public-api.baostock.com -> $($a.IPAddress) via DNS $dns" | Out-File -FilePath $log -Append -Encoding utf8
      break
    }
  } catch { }
}

for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
  if ($attempt -gt 1 -and ((Get-Date) - $started).TotalSeconds -ge $budgetSec) {
    "retry budget of $budgetSec s spent before attempt $attempt; giving up." |
      Out-File -FilePath $log -Append -Encoding utf8
    break
  }
  "=== run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') (attempt $attempt/$maxAttempts) ===" |
    Out-File -FilePath $log -Append -Encoding utf8
  $out = [System.IO.Path]::GetTempFileName(); $err = [System.IO.Path]::GetTempFileName()
  $proc = Start-Process -FilePath $py -ArgumentList "`"$script`"" `
    -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
  $null = $proc.Handle           # hold the handle so ExitCode stays readable without -Wait
  $timedOut = -not $proc.WaitForExit($timeoutSec * 1000)
  if ($timedOut) {
    & taskkill.exe /PID $proc.Id /T /F | Out-Null
    $proc.WaitForExit()
  }
  Get-Content -LiteralPath $out, $err -Encoding UTF8 | Out-File -FilePath $log -Append -Encoding utf8
  Remove-Item -LiteralPath $out, $err -ErrorAction SilentlyContinue
  $code = $proc.ExitCode
  if ($timedOut) {
    "attempt exceeded $timeoutSec s and was killed; treated as a transient failure (exit 75)." |
      Out-File -FilePath $log -Append -Encoding utf8
    $code = $EX_TEMPFAIL
  }

  if ($code -eq 0) { Invoke-ShadowD5 $repo $py }   # canonical succeeded -> shadow (issue #22)
  if ($code -ne $EX_TEMPFAIL) { exit $code }        # success (0) or fatal (!=75) -> done
  if ($attempt -lt $maxAttempts) {
    "transient data failure (exit 75); retrying in $delaySec s (attempt $attempt/$maxAttempts) ..." |
      Out-File -FilePath $log -Append -Encoding utf8
    Start-Sleep -Seconds $delaySec
  }
}
"retries exhausted; BaoStock never delivered a complete run. StartWhenAvailable plus the next scheduled run will catch up (recompute-from-seed)." |
  Out-File -FilePath $log -Append -Encoding utf8
exit $EX_TEMPFAIL
