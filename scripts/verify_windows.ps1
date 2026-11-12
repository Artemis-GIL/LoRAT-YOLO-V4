<#
.SYNOPSIS
Verify a native Windows OSDTrack installation without running a full sequence.

.DESCRIPTION
Checks the five immutable model files, both Python environments, strict D0 CPU
loading and a synthetic forward pass, the Ultralytics checkpoint loader, the
portable runtime preflight, and (unless skipped) a CUDA operation in each
environment. No training, dataset access, or model download is performed.
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$WeightsDir = "",
    [string]$LoRATEnvName = "osdtrack-lorat",
    [string]$YoloEnvName = "osdtrack-yolo",
    [string]$CondaExecutable = "conda",
    [switch]$SkipGpuCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw "verify_windows.ps1 supports native Windows only (no WSL)."
}

$resolvedRepo = (Resolve-Path -LiteralPath $RepoRoot).Path
if ([string]::IsNullOrWhiteSpace($WeightsDir)) {
    $WeightsDir = Join-Path $resolvedRepo "weights"
} elseif (-not [IO.Path]::IsPathRooted($WeightsDir)) {
    $WeightsDir = Join-Path $resolvedRepo $WeightsDir
}
if (-not (Test-Path -LiteralPath $WeightsDir -PathType Container)) {
    throw "Weights directory does not exist: $WeightsDir"
}
$resolvedWeights = (Resolve-Path -LiteralPath $WeightsDir).Path

try {
    $script:CondaCommand = Get-Command $CondaExecutable -ErrorAction Stop
} catch {
    throw "Conda was not found. Open 'Miniconda Prompt' or pass -CondaExecutable with the full path to conda.exe."
}
$script:PythonEnvironmentJson = "{}"

function Invoke-CondaPython {
    param(
        [Parameter(Mandatory = $true)][string]$EnvironmentName,
        [Parameter(Mandatory = $true)][string[]]$PythonArgs,
        [Parameter(Mandatory = $true)][string]$Label
    )
    Write-Host "[check] $Label" -ForegroundColor Cyan
    # `conda run` may restore environment-scoped OSDTRACK variables after it
    # inherits this PowerShell process.  A tiny Python bootstrap reapplies this
    # verification run's resolved paths before executing -c, -m, or a script.
    $bootstrap = @'
import json
import os
import runpy
import sys

os.environ.update(json.loads(sys.argv[1]))
args = sys.argv[2:]
if not args:
    raise SystemExit("missing Python target")
if args[0] == "-c":
    if len(args) < 2:
        raise SystemExit("-c requires code")
    sys.argv = ["-c", *args[2:]]
    namespace = {"__name__": "__main__", "__package__": None}
    exec(compile(args[1], "<string>", "exec"), namespace)
elif args[0] == "-m":
    if len(args) < 2:
        raise SystemExit("-m requires a module")
    sys.argv = [args[1], *args[2:]]
    runpy.run_module(args[1], run_name="__main__", alter_sys=False)
else:
    sys.argv = [args[0], *args[1:]]
    runpy.run_path(args[0], run_name="__main__")
'@
    & $script:CondaCommand run --no-capture-output -n $EnvironmentName python -c $bootstrap $script:PythonEnvironmentJson @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
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

$weightSpecs = @(
    [pscustomobject]@{
        Name = "DINOv2-L backbone"
        File = "dinov2_vitl14_pretrain.pth"
        Bytes = [long]1217586395
        Sha256 = "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"
        Variable = "OSDTRACK_DINO_WEIGHT"
    },
    [pscustomobject]@{
        Name = "LoRAT constructor"
        File = "LoRAT_large-378.bin"
        Bytes = [long]130101948
        Sha256 = "601529735439da5d7ad0136b63e894ee02923d4694ce17d3568e333c298a06b0"
        Variable = "OSDTRACK_LORAT_WEIGHT"
    },
    [pscustomobject]@{
        Name = "UAV-adapted B"
        File = "LoRAT_B_UAV_final.safetensors"
        Bytes = [long]130101948
        Sha256 = "e62112dd6582552d1668130b58b0a0c3382775670ccc6c714e9b00e94dbd58bc"
        Variable = "OSDTRACK_B_WEIGHT"
    },
    [pscustomobject]@{
        Name = "static D0 (paper final)"
        File = "D0_static_final.safetensors"
        Bytes = [long]1002996
        Sha256 = "0a27352804839714f7e691f60ae288ef06dc0b358209e379de156831d24871bd"
        Variable = "OSDTRACK_D0_WEIGHT"
    },
    [pscustomobject]@{
        Name = "YOLO detector"
        File = "yolo26_uav_best.pt"
        Bytes = [long]60285155
        Sha256 = "efa76597c9912fbda8ff150208996a88597544fb5ba2e52a6ad02e4dc964b433"
        Variable = "OSDTRACK_YOLO_WEIGHT"
    }
)

Write-Host "[check] SHA-256 and byte length for all five weights" -ForegroundColor Cyan
$weightErrors = [Collections.Generic.List[string]]::new()
$weightRows = foreach ($spec in $weightSpecs) {
    $path = Join-Path $resolvedWeights $spec.File
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $weightErrors.Add("Missing $($spec.Name): $path")
        [pscustomobject]@{ Weight = $spec.Name; Status = "MISSING"; Bytes = "-"; Sha256 = "-" }
        continue
    }
    $item = Get-Item -LiteralPath $path
    $actualHash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    $status = "PASS"
    if ($item.Length -ne $spec.Bytes) {
        $status = "FAIL"
        $weightErrors.Add("$($spec.Name) byte length is $($item.Length), expected $($spec.Bytes): $path")
    }
    if ($actualHash -ne $spec.Sha256) {
        $status = "FAIL"
        $weightErrors.Add("$($spec.Name) SHA-256 is $actualHash, expected $($spec.Sha256): $path")
    }
    [Environment]::SetEnvironmentVariable($spec.Variable, $item.FullName, "Process")
    [pscustomobject]@{ Weight = $spec.Name; Status = $status; Bytes = $item.Length; Sha256 = $actualHash }
}
$weightRows | Format-Table -AutoSize | Out-Host

if ($weightErrors.Count -gt 0) {
    $d0Hint = "Download D0_static_final.safetensors from release osdtrack-d0-windows-rc1-20261112; do not substitute the historical GRU_D_final.safetensors."
    throw ("Weight verification failed:`n" + ($weightErrors -join "`n") + "`n" + $d0Hint)
}

$yoloPython = Resolve-CondaPython -EnvironmentName $YoloEnvName
[Environment]::SetEnvironmentVariable("OSDTRACK_WEIGHTS_DIR", $resolvedWeights, "Process")
[Environment]::SetEnvironmentVariable("OSDTRACK_YOLO_PYTHON", $yoloPython, "Process")
$runtimeVariables = [ordered]@{
    OSDTRACK_WEIGHTS_DIR  = $resolvedWeights
    OSDTRACK_DINO_WEIGHT  = (Join-Path $resolvedWeights "dinov2_vitl14_pretrain.pth")
    OSDTRACK_LORAT_WEIGHT = (Join-Path $resolvedWeights "LoRAT_large-378.bin")
    OSDTRACK_B_WEIGHT     = (Join-Path $resolvedWeights "LoRAT_B_UAV_final.safetensors")
    OSDTRACK_D0_WEIGHT    = (Join-Path $resolvedWeights "D0_static_final.safetensors")
    OSDTRACK_YOLO_WEIGHT  = (Join-Path $resolvedWeights "yolo26_uav_best.pt")
    OSDTRACK_YOLO_PYTHON  = $yoloPython
}
$script:PythonEnvironmentJson = $runtimeVariables | ConvertTo-Json -Compress

$importSmoke = @'
import cv2
import numpy
import safetensors
import torch
from osdtrack import build_tracker, load_config, validate_weights
print({"status": "PASS", "torch": torch.__version__, "opencv": cv2.__version__, "numpy": numpy.__version__})
'@

$cpuSmoke = @'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
from safetensors.torch import load_file
from research.de_r1.model_d import ActionValue

torch.set_num_threads(1)
assert not torch.cuda.is_available(), "CPU smoke unexpectedly exposed a CUDA device"
model = ActionValue(temporal=False).cpu().eval()
model.load_state_dict(load_file(os.environ["OSDTRACK_D0_WEIGHT"], device="cpu"), strict=True)
features = torch.zeros((1, 2, 4, 4, 1024), dtype=torch.float32)
metadata = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
geometry = torch.zeros((1, 2, 4, 5), dtype=torch.float32)
history_valid = torch.ones((1, 2, 4), dtype=torch.bool)
with torch.inference_mode():
    output = model(features, metadata, geometry, history_valid, torch.tensor([0]))
assert torch.isfinite(output["quality"]).all()
b_state = load_file(os.environ["OSDTRACK_B_WEIGHT"], device="cpu")
assert len(b_state) == 301, len(b_state)
print({"status": "PASS", "device": "CPU", "variant": "D0", "parameters": sum(p.numel() for p in model.parameters()), "B_keys": len(b_state)})
'@

$yoloSmoke = @'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["YOLO_OFFLINE"] = "true"
import torch
import ultralytics
from ultralytics import YOLO

assert not torch.cuda.is_available(), "YOLO CPU smoke unexpectedly exposed a CUDA device"
model = YOLO(os.environ["OSDTRACK_YOLO_WEIGHT"], task="detect")
assert model.model is not None
print({"status": "PASS", "device": "CPU", "torch": torch.__version__, "ultralytics": ultralytics.__version__})
'@

$cudaSmoke = @'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
x = torch.ones(1, device="cuda")
assert x.item() == 1.0
print({"status": "PASS", "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
'@

Push-Location $resolvedRepo
try {
    Invoke-CondaPython -EnvironmentName $LoRATEnvName -PythonArgs @("tools/verify_snapshot.py") -Label "frozen Ubuntu snapshot hashes"
    Invoke-CondaPython -EnvironmentName $LoRATEnvName -PythonArgs @("-c", $importSmoke) -Label "LoRAT/runtime imports"
    Invoke-CondaPython -EnvironmentName $LoRATEnvName -PythonArgs @("-m", "osdtrack.runtime", "show-config") -Label "portable configuration"
    Invoke-CondaPython -EnvironmentName $LoRATEnvName -PythonArgs @("-m", "osdtrack.runtime", "preflight") -Label "portable runtime preflight"
    Invoke-CondaPython -EnvironmentName $LoRATEnvName -PythonArgs @("-c", $cpuSmoke) -Label "strict D0 CPU load and synthetic forward"
    Invoke-CondaPython -EnvironmentName $LoRATEnvName -PythonArgs @("-m", "pytest", "-q", "tests/test_protocol.py") -Label "simulator protocol smoke tests"
    Invoke-CondaPython -EnvironmentName $YoloEnvName -PythonArgs @("-c", $yoloSmoke) -Label "YOLO checkpoint CPU load"

    if (-not $SkipGpuCheck) {
        $smi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
        if ($null -eq $smi) {
            $smi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
        }
        if ($null -eq $smi) {
            throw "nvidia-smi was not found. Install a compatible NVIDIA Windows driver."
        }
        Write-Host "[check] NVIDIA driver" -ForegroundColor Cyan
        & $smi "--query-gpu=name,driver_version" "--format=csv,noheader"
        if ($LASTEXITCODE -ne 0) {
            throw "nvidia-smi failed with exit code $LASTEXITCODE."
        }
        Invoke-CondaPython -EnvironmentName $LoRATEnvName -PythonArgs @("-c", $cudaSmoke) -Label "LoRAT CUDA operation"
        Invoke-CondaPython -EnvironmentName $YoloEnvName -PythonArgs @("-c", $cudaSmoke) -Label "YOLO CUDA operation"
    } else {
        Write-Host "GPU checks skipped as requested; this is not a complete target-machine verification." -ForegroundColor Yellow
    }
} finally {
    Pop-Location
}

Write-Host "OSDTrack native Windows verification: PASS" -ForegroundColor Green
