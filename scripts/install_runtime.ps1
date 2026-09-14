# Install psytrainer-ml runtime on Windows (Skill install time).
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\install_runtime.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\install_runtime.ps1 -Wheel 'D:\wheels\PsyTrainer-*.whl'
#   $env:PSYTRAINER_WHEEL = 'D:\wheels\PsyTrainer-0.2.0-cp314-none-any.whl'
#   powershell -ExecutionPolicy Bypass -File scripts\install_runtime.ps1

param(
    [string]$Wheel = $env:PSYTRAINER_WHEEL,
    [string]$Python = "",
    [switch]$Recreate,
    [string]$Wheelhouse = "",
    [switch]$Legacy
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$scriptArgs = @("scripts\install_runtime.py")
if ($Wheel) { $scriptArgs += @("--wheel", $Wheel) }
if ($Python) { $scriptArgs += @("--python", $Python) }
if ($Recreate) { $scriptArgs += "--recreate" }
if ($Wheelhouse) { $scriptArgs += @("--wheelhouse", $Wheelhouse) }
if ($Legacy) { $scriptArgs += "--legacy" }

if (Get-Command py -ErrorAction SilentlyContinue) {
    Write-Host "+ py -3 $($scriptArgs -join ' ')"
    & py -3 @scriptArgs
    exit $LASTEXITCODE
}
if (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "+ python $($scriptArgs -join ' ')"
    & python @scriptArgs
    exit $LASTEXITCODE
}

Write-Error "Python not found. Install CPython 3.12-3.14 (3.14 for -Legacy) and enable the py launcher."
exit 1
