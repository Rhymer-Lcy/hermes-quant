# Manage the Hermes scheduled tasks (version-controlled definitions so they are reproducible).
# Two weekday jobs, isolated from each other:
#   hermes-paper     19:00  -> paper_live.ps1          (EOD paper trading; updates results/paper/)
#   hermes-if-accum  15:40  -> accumulate_if_minute.ps1 (IF minute-bar data accumulation)
# hermes-paper runs in the evening because BaoStock posts the day's EOD daily bars ~2-3 h after
# close; an earlier run would either miss today's data or (mid-publication) mis-liquidate the
# not-yet-posted names. hermes-if-accum stays at 15:40 (Sina minute bars are available at close).
# Both use StartWhenAvailable: a run missed because the PC was off/asleep/logged-out fires on the
# next boot/wake (cannot run while powered off; one catch-up, which suffices given recompute-from-seed).
# hermes-paper additionally retries a TRANSIENT data failure with backoff inside paper_live.ps1 (BaoStock
# unreachable, e.g. a VPN blocking it at 19:00, or mid-publication) -- so it self-heals without a manual
# re-trigger; a run can therefore stay in the Running state for up to the retry window (~2 h by default).
#
# Usage (run from anywhere; paths are derived from this script's location):
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_tasks.ps1 register   # (re)create both (idempotent)
#   powershell ... schedule_tasks.ps1 status     # show state + next run time
#   powershell ... schedule_tasks.ps1 disable    # PAUSE both (keep definitions; resume later)
#   powershell ... schedule_tasks.ps1 enable     # RESUME both
#   powershell ... schedule_tasks.ps1 remove     # DELETE both
# Run one once, now (without waiting for the schedule):  Start-ScheduledTask -TaskName hermes-paper
# Per-task: replace the loop with a single -TaskName, e.g.  Disable-ScheduledTask -TaskName hermes-if-accum
param([ValidateSet('register', 'status', 'disable', 'enable', 'remove')] [string]$action = 'status')
$ErrorActionPreference = 'Stop'
$tasks = @{
  'hermes-paper'    = @{ file = Join-Path $PSScriptRoot 'paper_live.ps1';          time = '19:00'; desc = 'Hermes daily EOD paper trading (weekdays 19:00, after BaoStock posts EOD)' }
  'hermes-if-accum' = @{ file = Join-Path $PSScriptRoot 'accumulate_if_minute.ps1'; time = '15:40'; desc = 'Hermes daily IF minute-bar accumulator (weekdays 15:40)' }
}
switch ($action) {
  'register' {
    foreach ($name in $tasks.Keys) {
      $t = $tasks[$name]
      $a = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$($t.file)`""
      $trig = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $t.time
      $p = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
      # StartWhenAvailable: if the PC was off/asleep/logged-out at the scheduled time, run the missed
      # job once on the next boot/wake (one catch-up, not one-per-missed-day -- which suffices because
      # the paper ledger is recompute-from-seed, so a single late run reconstructs every skipped bar).
      $s = New-ScheduledTaskSettingsSet -StartWhenAvailable
      Register-ScheduledTask -TaskName $name -Action $a -Trigger $trig -Principal $p -Settings $s -Description $t.desc -Force | Out-Null
      "registered $name @ $($t.time) weekdays (LogonType Interactive; StartWhenAvailable = catch up a missed run on next boot/wake)"
    }
  }
  'disable' { $tasks.Keys | ForEach-Object { Disable-ScheduledTask -TaskName $_ | Out-Null; "disabled (paused) $_" } }
  'enable'  { $tasks.Keys | ForEach-Object { Enable-ScheduledTask -TaskName $_ | Out-Null; "enabled $_" } }
  'remove'  { $tasks.Keys | ForEach-Object { Unregister-ScheduledTask -TaskName $_ -Confirm:$false; "removed $_" } }
  'status'  {
    Get-ScheduledTask -TaskName 'hermes-*' | Select-Object TaskName, State | Format-Table -AutoSize
    Get-ScheduledTask -TaskName 'hermes-*' | ForEach-Object { "{0}: next run {1}" -f $_.TaskName, (Get-ScheduledTaskInfo -TaskName $_.TaskName).NextRunTime }
  }
}
