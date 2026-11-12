<#
.SYNOPSIS
Create or update the two native Windows Conda environments used by OSDTrack.

.DESCRIPTION
This script is safe to run repeatedly. It uses only the current user's Conda
installation, creates no system-wide files, downloads no model weights, and
persists the resolved model paths as Conda environment variables.
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$WeightsDir = "",
    [string]$LoRATEnvName = "osdtrack-lorat",
    [string]$YoloEnvName = "osdtrack-yolo",
    [string]$CondaExecutable = "conda",
    [switch]$SkipEnvironmentUpdate,
    [switch]$SkipVerify,
    [switch]$SkipGpuCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "setup_windows.ps1 supports native Windows only (no WSL)."
}

$resolvedRepo = (Resolve-Path -LiteralPath $RepoRoot).Path
$loratSpec = Join-Path $resolvedRepo "environment\lorat-windows.yml"
$yoloSpec = Join-Path $resolvedRepo "environment\yolo-windows.yml"
foreach ($required in @($loratSpec, $yoloSpec)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required environment file not found: $required"
    }
}

if ([string]::IsNullOrWhiteSpace($WeightsDir)) {
    $WeightsDir = Join-Path $resolvedRepo "weights"
} elseif (-not [IO.Path]::IsPathRooted($WeightsDir)) {
    $WeightsDir = Join-Path $resolvedRepo $WeightsDir
}
$null = New-Item -ItemType Directory -Force -Path $WeightsDir
$resolvedWeights = (Resolve-Path -LiteralPath $WeightsDir).Path

try {
    $script:CondaCommand = Get-Command $CondaExecutable -ErrorAction Stop
} catch {
    throw "Conda was not found. Open 'Miniconda Prompt' or pass -CondaExecutable with the full path to conda.exe."
}

function Invoke-CondaCommand {
    param([Parameter(Mandatory = $true)][string[]]$CondaArgs)
    Write-Host ("[conda] conda " + ($CondaArgs -join " ")) -ForegroundColor Cyan
    & $script:CondaCommand @CondaArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Conda command failed with exit code $LASTEXITCODE."
    }
}

function Resolve-CondaPython {
    param([Parameter(Mandatory = $true)][string]$EnvironmentName)
    $raw = @(& $script:CondaCommand run -n $EnvironmentName python -c "import sys; print(sys.executable)" 2>&1)
    if ($LASTEXITCODE -ne 0) {
        $raw | ForEach-Object { Write-Host $_ }
        throw "Cannot resolve python.exe for Conda environment '$EnvironmentName'."
    }
    foreach ($line in ($raw | Select-Object -Last 10)) {
        $candidate = ([string]$line).Trim()
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "Conda environment '$EnvironmentName' did not report a usable python.exe path."
}

Invoke-CondaCommand -CondaArgs @("--version")

if (-not $SkipEnvironmentUpdate) {
    # These are project-owned environments. `--prune` removes dependencies no
    # longer present in the YAML so repeated invocations converge to the specs.
    Invoke-CondaCommand -CondaArgs @("env", "update", "--prune", "-n", $LoRATEnvName, "-f", $loratSpec)
    Invoke-CondaCommand -CondaArgs @("env", "update", "--prune", "-n", $YoloEnvName, "-f", $yoloSpec)
} else {
    Write-Host "Skipping Conda environment update as requested." -ForegroundColor Yellow
}

$yoloPython = Resolve-CondaPython -EnvironmentName $YoloEnvName
$weightVariables = [ordered]@{
    OSDTRACK_WEIGHTS_DIR  = $resolvedWeights
    OSDTRACK_DINO_WEIGHT  = (Join-Path $resolvedWeights "dinov2_vitl14_pretrain.pth")
    OSDTRACK_LORAT_WEIGHT = (Join-Path $resolvedWeights "LoRAT_large-378.bin")
    OSDTRACK_B_WEIGHT     = (Join-Path $resolvedWeights "LoRAT_B_UAV_final.safetensors")
    OSDTRACK_D0_WEIGHT    = (Join-Path $resolvedWeights "D0_static_final.safetensors")
    OSDTRACK_YOLO_WEIGHT  = (Join-Path $resolvedWeights "yolo26_uav_best.pt")
    OSDTRACK_YOLO_PYTHON  = $yoloPython
}

foreach ($entry in $weightVariables.GetEnumerator()) {
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
}

# Persist the same values in both project-owned environments. This avoids
# relying on a global PATH or user/machine environment mutation in later shells.
$variableArgs = @("env", "config", "vars", "set", "-n", $LoRATEnvName)
$variableArgs += @($weightVariables.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" })
Invoke-CondaCommand -CondaArgs $variableArgs

$variableArgs = @("env", "config", "vars", "set", "-n", $YoloEnvName)
$variableArgs += @($weightVariables.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" })
Invoke-CondaCommand -CondaArgs $variableArgs

Write-Host "Resolved repository: $resolvedRepo" -ForegroundColor Green
Write-Host "Resolved weights:    $resolvedWeights" -ForegroundColor Green
Write-Host "YOLO python:          $yoloPython" -ForegroundColor Green

if (-not $SkipVerify) {
    $verifyScript = Join-Path $PSScriptRoot "verify_windows.ps1"
    $verifyArgs = @{
        RepoRoot = $resolvedRepo
        WeightsDir = $resolvedWeights
        LoRATEnvName = $LoRATEnvName
        YoloEnvName = $YoloEnvName
        CondaExecutable = $CondaExecutable
    }
    if ($SkipGpuCheck) {
        $verifyArgs["SkipGpuCheck"] = $true
    }
    & $verifyScript @verifyArgs
} else {
    Write-Host "Verification skipped. Run scripts\verify_windows.ps1 after all five weights are present." -ForegroundColor Yellow
}
