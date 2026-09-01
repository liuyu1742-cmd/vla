param(
    [int]$CollectionProcessId = 24508
)

$ErrorActionPreference = "Stop"
$projectRoot = "C:\OpenVLA-Simulator"
$pythonOpenVla = "C:\Users\sjtu101\miniconda3\envs\openvla\python.exe"
$pythonRoboCasa = "C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe"
$adapterDir = Join-Path $projectRoot "models\openvla-water-cup-lora-stride1-e1"
$statusFile = Join-Path $projectRoot "outputs\water_cup_stride1_pipeline_status.json"
$serverLog = Join-Path $projectRoot "outputs\water_cup_stride1_server.log"
$serverErrorLog = Join-Path $projectRoot "outputs\water_cup_stride1_server.err.log"
$port = 8771

function Write-Status([string]$stage, [hashtable]$extra = @{}) {
    $status = @{
        stage = $stage
        timestamp = (Get-Date).ToString("o")
        collection_process_id = $CollectionProcessId
        adapter_dir = $adapterDir
        port = $port
    }
    foreach ($key in $extra.Keys) {
        $status[$key] = $extra[$key]
    }
    $status | ConvertTo-Json -Depth 6 | Set-Content -Path $statusFile -Encoding UTF8
}

Set-Location $projectRoot
Write-Status "waiting_for_stride1_collection"
$collection = Get-Process -Id $CollectionProcessId -ErrorAction SilentlyContinue
if ($null -ne $collection) {
    Wait-Process -Id $CollectionProcessId
}

Write-Status "preparing_stride1_manifest"
& $pythonOpenVla -m tools.prepare_water_cup_stride1_manifest
if ($LASTEXITCODE -ne 0) {
    Write-Status "manifest_failed" @{ exit_code = $LASTEXITCODE }
    exit 2
}

$manifestPath = Join-Path $projectRoot "datasets\water_cup_expert\training_manifest.json"
$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
$trainSamples = ($manifest.train | Measure-Object samples -Sum).Sum
Write-Status "training_stride1" @{
    train_episodes = @($manifest.train).Count
    train_samples = $trainSamples
    planned_updates = $trainSamples
}

& $pythonOpenVla -m tools.finetune_water_cup_openvla_epochs_v2 `
    --epochs 1 `
    --output $adapterDir
if ($LASTEXITCODE -ne 0) {
    Write-Status "training_failed" @{ exit_code = $LASTEXITCODE }
    exit 3
}

$adapterFile = Join-Path $adapterDir "adapter_model.safetensors"
if (-not (Test-Path $adapterFile)) {
    Write-Status "training_failed" @{ reason = "adapter file was not created" }
    exit 4
}

Write-Status "starting_stride1_server"
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
    if ($server.HasExited) { break }
    if ((Test-Path $serverLog) -and (Select-String -Path $serverLog -Pattern "OPENVLA_WATER_CUP_SERVER_READY" -Quiet)) {
        $ready = $true
        break
    }
}
if (-not $ready) {
    Write-Status "server_start_failed" @{ server_process_id = $server.Id }
    exit 5
}

$results = @()
foreach ($seed in @(0, 2)) {
    if (($seed -eq 2) -and (($results.Count -eq 0) -or (-not $results[0].success))) {
        break
    }
    $evaluationDir = Join-Path $projectRoot "outputs\openvla_gym_water_cup_stride1_seed$seed"
    $evaluationLog = Join-Path $projectRoot "outputs\water_cup_stride1_eval_seed$seed.log"
    $evaluationErrorLog = Join-Path $projectRoot "outputs\water_cup_stride1_eval_seed$seed.err.log"
    Write-Status "evaluating_stride1_seed_$seed" @{
        server_process_id = $server.Id
        completed_results = $results
    }
    $evaluation = Start-Process -FilePath $pythonRoboCasa `
        -ArgumentList "-m", "tools.openvla_gym_water_cup_rollout_v2", "--steps", "700", "--query-interval", "1", "--seed", "$seed", "--port", "$port", "--object-scale", "0.7", "--output-dir", $evaluationDir `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $evaluationLog `
        -RedirectStandardError $evaluationErrorLog `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    $reportPath = Join-Path $evaluationDir "report.json"
    if (-not (Test-Path $reportPath)) {
        $results += @{ seed = $seed; success = $false; reason = "report missing" }
        break
    }
    $report = Get-Content $reportPath -Raw | ConvertFrom-Json
    $results += @{
        seed = $seed
        success = [bool]$report.success
        executed_steps = @($report.steps).Count
        report = $reportPath
    }
}

$stage = if (($results.Count -eq 2) -and $results[0].success -and $results[1].success) {
    "complete_stride1_success"
} else {
    "complete_stride1_needs_iteration"
}
Write-Status $stage @{
    server_process_id = $server.Id
    train_episodes = @($manifest.train).Count
    train_samples = $trainSamples
    results = $results
}

if ($stage -eq "complete_stride1_success") { exit 0 }
exit 1
