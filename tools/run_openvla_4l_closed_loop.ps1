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

    [string]$Seeds = "0,1,2,3,5",
    [int]$MaxSteps = 300
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$base = Join-Path $workspace "models\experiment_4_6_openvla_4l\base"
$checkpointPath = (Resolve-Path (Join-Path $workspace $Checkpoint)).Path
$outputPath = Join-Path $workspace $OutputDir
$openvlaPython = "C:\Users\sjtu101\miniconda3\envs\openvla\python.exe"
$robocasaPython = "C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe"

New-Item -ItemType Directory -Force -Path $outputPath | Out-Null
$serverStdout = Join-Path $outputPath "server.stdout.log"
$serverStderr = Join-Path $outputPath "server.stderr.log"

$serverArgs = @(
    "-m", "tools.openvla_4l_tcp_server",
    "--mode", $Mode,
    "--checkpoint", $checkpointPath,
    "--base", $base,
    "--port", $Port
)

$server = Start-Process `
    -FilePath $openvlaPython `
    -ArgumentList $serverArgs `
    -WorkingDirectory $workspace `
    -WindowStyle Hidden `
    -RedirectStandardOutput $serverStdout `
    -RedirectStandardError $serverStderr `
    -PassThru

try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 180; $attempt++) {
        if ($server.HasExited) {
            throw "Inference server exited before becoming ready. See $serverStderr"
        }
        try {
            $client = [System.Net.Sockets.TcpClient]::new()
            $client.Connect("127.0.0.1", $Port)
            $client.Close()
            $ready = $true
            break
        }
        catch {
            Start-Sleep -Milliseconds 1000
        }
    }
    if (-not $ready) {
        throw "Inference server did not become ready within 180 seconds."
    }

    & $robocasaPython `
        -m tools.openvla_4l_robocasa_eval `
        --mode $Mode `
        --checkpoint $checkpointPath `
        --base $base `
        --output-dir $outputPath `
        --seeds $Seeds `
        --max-steps $MaxSteps `
        --query-interval 1 `
        --object-scale 0.7 `
        --port $Port
    if ($LASTEXITCODE -ne 0) {
        throw "RoboCasa evaluation failed with exit code $LASTEXITCODE"
    }
}
finally {
    if (-not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
        $server.WaitForExit()
    }
}
