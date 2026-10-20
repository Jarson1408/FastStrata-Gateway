# setup_autostart.ps1 - the Windows Scheduled Task for LODESTAR's warm embedding service
#
#   .\setup_autostart.ps1 [-Action Enable|Disable|Status] [-InstallDir <gateway folder>]
#
# One task: `faststrata-embeddings`, at logon, the semantic-memory service on
# 127.0.0.1:18001. Without it every embed.py call pays ~7 s to import
# sentence-transformers, and LODESTAR makes two per task. It is an
# optimisation, not a dependency: embed.py loads the model in-process when
# nothing is listening.
#
# The gateway itself is NOT a task: the desktop app (start_gui.bat, or the
# desktop shortcut FastStrata's setup makes) supervises its own
# `python -m gateway` child, and a task starting a second one races it for
# port 18000. To run the gateway headless, run install\windows\start_proxy.bat.
param(
    [ValidateSet("Enable", "Disable", "Status")]
    [string]$Action = "Enable",
    [string]$InstallDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
)

$ErrorActionPreference = "Stop"

$embedTaskName = "faststrata-embeddings"

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "    OK $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    !! $msg" -ForegroundColor Yellow }

if ($Action -eq "Disable") {
    Step "Removing the scheduled task..."
    Unregister-ScheduledTask -TaskName $embedTaskName -Confirm:$false -ErrorAction SilentlyContinue
    Ok "'$embedTaskName' removed from Windows Task Scheduler."
    exit 0
}

if ($Action -eq "Status") {
    $t = Get-ScheduledTask -TaskName $embedTaskName -ErrorAction SilentlyContinue
    if ($t) { Ok "$embedTaskName : $($t.State)" }
    else { Warn "$embedTaskName : not registered (semantic memory will be ~16 s/task slower)" }
    exit 0
}

Step "Registering '$embedTaskName' (runs $InstallDir\start_embed_service.bat at logon)..."
Unregister-ScheduledTask -TaskName $embedTaskName -Confirm:$false -ErrorAction SilentlyContinue

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$action = New-ScheduledTaskAction -Execute "$InstallDir\start_embed_service.bat" -WorkingDirectory $InstallDir
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0)

Register-ScheduledTask `
    -TaskName $embedTaskName `
    -Description "LODESTAR semantic memory: warm all-MiniLM-L6-v2 embedding service on CPU (127.0.0.1:18001)" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal | Out-Null
Ok "'$embedTaskName' registered."
