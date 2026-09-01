param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("full", "lora_r32", "last_layer_only", "frozen_vision")]
    [string]$Mode,
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir,
    [Parameter(Mandatory = $true)]
    [int]$Port,
    [int]$Seed = 0,
    [int]$Steps = 300
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$base = Join-Path $workspace "models\experiment_4_6_openvla_4l\base"
$checkpointPath = (Resolve-Path (Join-Path $workspace $Checkpoint)).Path
$outputPath = Join-Path $workspace $OutputDir
$openvlaPython = "C:\Users\sjtu101\miniconda3\envs\openvla\python.exe"
$robocasaPython = "C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe"
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$stdout = Join-Path $outputPath "server.stdout.log"
$stderr = Join-Path $outputPath "server.stderr.log"
$server = Start-Process `
    -FilePath $openvlaPython `
    -ArgumentList @(
        "-m", "tools.openvla_4l_tcp_server",
        "--mode", $Mode,
        "--checkpoint", $checkpointPath,
        "--base", $base,
        "--port", $Port
    ) `
    -WorkingDirectory $workspace `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 360; $attempt++) {
        if ($server.HasExited) {
            throw "Inference server exited before becoming ready. See $stderr"
        }
        if (
            (Test-Path -LiteralPath $stdout) -and
            ((Get-Content -LiteralPath $stdout -Raw) -match "OPENVLA_4L_SERVER_READY")
        ) {
            $ready = $true
            break
        }
        Start-Sleep -Milliseconds 1000
    }
    if (-not $ready) {
        throw "Inference server did not become ready within 360 seconds."
    }
    & $robocasaPython `
        -m tools.openvla_gym_water_cup_diagnostic `
        --steps $Steps `
        --seed $Seed `
        --port $Port `
        --object-scale 0.7 `
        --output-dir $outputPath
}
finally {
    if (-not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
        $server.WaitForExit()
    }
}
