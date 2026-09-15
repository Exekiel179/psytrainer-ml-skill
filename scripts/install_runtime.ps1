# Configure the local Python runtime and check dependencies on Windows.
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\install_runtime.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\install_runtime.ps1 -Wheel 'D:\wheels\PsyTrainer-*.whl'

param(
    [string]$Wheel = "",
    [string]$Python = "",
    [switch]$Recreate,
    [string]$Wheelhouse = "",
    [string]$IndexUrl = "",
    [ValidateSet("pip", "uv")]
    [string]$Installer = "pip",
    [int]$Timeout = 20,
    [int]$Retries = 2,
    [int]$MaxSeconds = 600,
    [switch]$Check,
    [switch]$All,
    [string[]]$Packages = @(),
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
if ($IndexUrl) { $scriptArgs += @("--index-url", $IndexUrl) }
$scriptArgs += @("--installer", $Installer)
$scriptArgs += @("--timeout", "$Timeout", "--retries", "$Retries", "--max-seconds", "$MaxSeconds")
if ($Check) { $scriptArgs += "--check" }
if ($All) { $scriptArgs += "--all" }
if ($Packages) { $scriptArgs += @("--packages") + $Packages }

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

Write-Error "Python not found. Install CPython 3.12-3.14 and enable the py launcher."
exit 1
