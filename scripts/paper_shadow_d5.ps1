# D=5 forward-shadow wrapper (issue #22). Runs AFTER the canonical paper step succeeds.
#
# ISOLATION IS THE POINT. This script never refreshes the lake -- the canonical run owns the data
# pull and this reads what it already validated -- and it never writes outside
# results/paper_shadow/d5/. It is invoked by paper_live.ps1 only on a canonical exit code of 0, and
# that caller discards this script's exit code, so a shadow failure can never roll back, overwrite
# or invalidate the canonical D=1 record.
#
# To DISABLE the shadow: set HERMES_SHADOW_D5=0 (or delete the call in paper_live.ps1). The
# canonical task is unaffected either way. To re-enable: unset the variable.
# To run manually:  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\paper_shadow_d5.ps1
$ErrorActionPreference = "Continue"     # a shadow fault must never abort the caller
$repo = Split-Path -Parent $PSScriptRoot
$py = if ($env:HERMES_PYTHON) { $env:HERMES_PYTHON } else { "python" }
$logdir = Join-Path $repo "results\paper_shadow\d5\logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$log = Join-Path $logdir ("shadow_{0:yyyyMMdd}.log" -f (Get-Date))

if ($env:HERMES_SHADOW_D5 -eq "0") {
  "=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') shadow DISABLED via HERMES_SHADOW_D5=0 ===" |
    Out-File -FilePath $log -Append -Encoding utf8
  exit 0
}

$env:PYTHONIOENCODING = "utf-8"
$script = Join-Path $repo 'scripts\paper_shadow_d5.py'
"=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') d5 forward shadow ===" |
  Out-File -FilePath $log -Append -Encoding utf8

# No retry loop: the canonical run has already pulled and validated the lake, so a transient data
# failure here means the shadow simply skips a day and catches up on the next run
# (recompute-from-seed makes a missed day harmless).
$out = [System.IO.Path]::GetTempFileName(); $err = [System.IO.Path]::GetTempFileName()
$proc = Start-Process -FilePath $py -ArgumentList "`"$script`"" `
  -NoNewWindow -Wait -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
Get-Content -LiteralPath $out, $err -Encoding UTF8 | Out-File -FilePath $log -Append -Encoding utf8
Remove-Item -LiteralPath $out, $err -ErrorAction SilentlyContinue

if ($proc.ExitCode -ne 0) {
  "shadow exited $($proc.ExitCode) -- recorded, canonical D=1 record is unaffected" |
    Out-File -FilePath $log -Append -Encoding utf8
}
exit $proc.ExitCode
