param(
    [int]$TrainingProcessId = 24372
)

$ErrorActionPreference = "Stop"
$projectRoot = "C:\OpenVLA-Simulator"
$pythonOpenVla = "C:\Users\sjtu101\miniconda3\envs\openvla\python.exe"
$pythonRoboCasa = "C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe"
$adapterDir = Join-Path $projectRoot "models\openvla-water-cup-lora-e12-v2"
$adapterFile = Join-Path $adapterDir "adapter_model.safetensors"
$serverLog = Join-Path $projectRoot "outputs\water_cup_e12_v2_server.log"
$serverErrorLog = Join-Path $projectRoot "outputs\water_cup_e12_v2_server.err.log"
$evaluationLog = Join-Path $projectRoot "outputs\water_cup_e12_v2_eval.log"
$evaluationErrorLog = Join-Path $projectRoot "outputs\water_cup_e12_v2_eval.err.log"
$evaluationDir = Join-Path $projectRoot "outputs\openvla_gym_water_cup_e12_v2_seed2"
$statusFile = Join-Path $projectRoot "outputs\water_cup_e12_pipeline_status.json"
$port = 8768

function Write-Status([string]$stage, [hashtable]$extra = @{}) {
    $status = @{
        stage = $stage
        timestamp = (Get-Date).ToString("o")
        training_process_id = $TrainingProcessId
        adapter_dir = $adapterDir
        evaluation_dir = $evaluationDir
        port = $port
    }
    foreach ($key in $extra.Keys) {
        $status[$key] = $extra[$key]
    }
    $status | ConvertTo-Json -Depth 5 | Set-Content -Path $statusFile -Encoding UTF8
}

Set-Location $projectRoot
Write-Status "waiting_for_training"

$trainingProcess = Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue
if ($null -ne $trainingProcess) {
    Wait-Process -Id $TrainingProcessId
}

if (-not (Test-Path $adapterFile)) {
    Write-Status "training_failed" @{ reason = "adapter_model.safetensors was not created" }
    exit 2
}

$listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($null -ne $listener) {
    Write-Status "server_start_failed" @{ reason = "port $port is already in use" }
    exit 3
}

Write-Status "starting_server"
$server = Start-Process -FilePath $pythonOpenVla `
    -ArgumentList "-m", "tools.openvla_tcp_server_water_cup", "--port", "$port", "--adapter-dir", $adapterDir `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $serverLog `
    -RedirectStandardError $serverErrorLog `
    -WindowStyle Hidden `
    -PassThru

$ready = $false
for ($attempt = 0; $attempt -lt 180; $attempt++) {
    Start-Sleep -Seconds 2
    if ($server.HasExited) {
        break
    }
    if ((Test-Path $serverLog) -and (Select-String -Path $serverLog -Pattern "OPENVLA_WATER_CUP_SERVER_READY" -Quiet)) {
        $ready = $true
        break
    }
}

if (-not $ready) {
    Write-Status "server_start_failed" @{ server_process_id = $server.Id }
    exit 4
}

Write-Status "evaluating" @{ server_process_id = $server.Id }
$evaluation = Start-Process -FilePath $pythonRoboCasa `
    -ArgumentList "-m", "tools.openvla_gym_lora_rollout", "--steps", "300", "--query-interval", "1", "--seed", "2", "--port", "$port", "--output-dir", $evaluationDir `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $evaluationLog `
    -RedirectStandardError $evaluationErrorLog `
    -WindowStyle Hidden `
    -Wait `
    -PassThru

$reportPath = Join-Path $evaluationDir "report.json"
if (-not (Test-Path $reportPath)) {
    Write-Status "evaluation_failed" @{
        server_process_id = $server.Id
        evaluation_exit_code = $evaluation.ExitCode
        reason = "report.json was not created"
    }
    exit 5
}

$report = Get-Content $reportPath -Raw | ConvertFrom-Json
$stage = if ($report.success) { "complete_success" } else { "complete_no_success" }
Write-Status $stage @{
    server_process_id = $server.Id
    evaluation_exit_code = $evaluation.ExitCode
    success = [bool]$report.success
    executed_steps = @($report.steps).Count
    report = $reportPath
}

if ($report.success) {
    exit 0
}
exit 1
