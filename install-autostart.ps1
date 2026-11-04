<#
.SYNOPSIS
  Start the Vriksha workshop worker by itself when this PC's user signs in, and restart it if it stops.

.DESCRIPTION
  Registers a Windows scheduled task ("Vriksha Scan Worker") that runs workshop_worker.py in the
  background with no window. It does not need you to start it, and it keeps its own log file.

  Do these once first, in this folder:
    1. python -m venv .venv ; .venv\Scripts\pip install -r requirements.txt
       (and, for flat carvings from one photo: .venv\Scripts\pip install -r requirements-relief.txt)
    2. .venv\Scripts\python workshop_worker.py --url https://api.vrikshastudios.com --user workshop-pc --save-password
       It asks for the password (nothing shows as you type), checks it, and keeps it in the
       Windows Credential Manager. The password is not written to any file or to this script.

  Then run this script. It never asks for or stores a password.

.PARAMETER Url      The server address, for example https://api.vrikshastudios.com
.PARAMETER User     The owner or Artisan account the worker signs in as (use its own account, for example workshop-pc).
.PARAMETER Worker   A name for this PC (shown to the server). Default: workshop-pc
.PARAMETER Python   The pythonw.exe to use. Default: .venv\Scripts\pythonw.exe next to this script.
.PARAMETER Remove   Stop the worker and remove the scheduled task.
.PARAMETER DryRun   Show what would be registered, change nothing.

.EXAMPLE
  .\install-autostart.ps1 -Url https://api.vrikshastudios.com -User workshop-pc
.EXAMPLE
  .\install-autostart.ps1 -Remove
#>
[CmdletBinding()]
param(
    [string]$Url,
    [string]$User,
    [string]$Worker = "workshop-pc",
    [string]$Python,
    [switch]$Remove,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$TaskName = "Vriksha Scan Worker"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = Join-Path $env:LOCALAPPDATA "VrikshaScan\worker.log"
$Me = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name  # DOMAIN\user, who signs in at this PC

if ($Remove) {
    if ($DryRun) { Write-Host "Would stop and remove the task '$TaskName'."; return }
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed '$TaskName'. The saved password is still in the Credential Manager; remove it with: workshop_worker.py --url ... --user ... --forget-password"
    } else {
        Write-Host "There is no task called '$TaskName'."
    }
    return
}

if (-not $Url -or -not $User) { throw "Give -Url and -User, for example: .\install-autostart.ps1 -Url https://api.vrikshastudios.com -User workshop-pc" }

if (-not $Python) { $Python = Join-Path $Here ".venv\Scripts\pythonw.exe" }
if (-not (Test-Path $Python)) {
    throw "Cannot find $Python. Create the environment first (python -m venv .venv ; .venv\Scripts\pip install -r requirements.txt) or pass -Python."
}
$Console = Join-Path (Split-Path -Parent $Python) "python.exe"

# The worker must be able to sign in with no one there: check a password has been saved.
& $Console -c "import sys, keyring; sys.exit(0 if keyring.get_password('vriksha-scan', sys.argv[1] + '@' + sys.argv[2].rstrip('/')) else 1)" $User $Url
if ($LASTEXITCODE -ne 0) {
    throw "No saved password for $User on $Url. Run once: $Console workshop_worker.py --url $Url --user $User --save-password"
}

$Arguments = "workshop_worker.py --url `"$Url`" --user `"$User`" --worker `"$Worker`" --unattended --log-file `"$LogFile`""
$Action = New-ScheduledTaskAction -Execute $Python -Argument $Arguments -WorkingDirectory $Here
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $Me
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 99 -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew
$Principal = New-ScheduledTaskPrincipal -UserId $Me -LogonType Interactive -RunLevel Limited

if ($DryRun) {
    Write-Host "Would register the task '$TaskName' for ${Me}:"
    Write-Host "  runs     $Python $Arguments"
    Write-Host "  in       $Here"
    Write-Host "  when     this user signs in; restarts every minute if it stops; one copy at a time; no time limit"
    Write-Host "  log      $LogFile"
    return
}

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName

Write-Host "Done. '$TaskName' is running now and will start by itself each time $Me signs in."
Write-Host "Log:     $LogFile"
Write-Host "Status:  Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
Write-Host "Stop:    Stop-ScheduledTask -TaskName '$TaskName'      Remove: .\install-autostart.ps1 -Remove"
Write-Host "Tip: set this PC to sign in by itself after a restart and to never sleep while plugged in, or the worker waits until someone signs in."
