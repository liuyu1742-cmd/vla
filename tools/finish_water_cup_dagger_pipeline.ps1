param(
    [Parameter(Mandatory = $true)][int]$TrainingProcessId,
    [string]$AdapterDir = "C:\OpenVLA-Simulator\models\openvla-water-cup-dagger-r1-e1",
    [int]$Port = 8772
)

$ErrorActionPreference = "Stop"
$root = "C:\OpenVLA-Simulator"
$openvlaPython = "C:\Users\sjtu101\miniconda3\envs\openvla\python.exe"
$robocasaPython = "C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe"
$statusPath = Join-Path $root "outputs\water_cup_dagger_r1_pipeline_status.json"
$serverOut = Join-Path $root "outputs\water_cup_dagger_r1_server.log"
$serverErr = Join-Path $root "outputs\water_cup_dagger_r1_server.err.log"
$serverProcess = $null

function Write-Stage([string]$stage, [hashtable]$extra = @{}) {
    $payload = @{
        stage = $stage
        timestamp = (Get-Date).ToString("o")
        training_process_id = $TrainingProcessId
        adapter_dir = $AdapterDir
        port = $Port
    }
    foreach ($key in $extra.Keys) { $payload[$key] = $extra[$key] }
    $payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statusPath -Encoding utf8
    Write-Output ($payload | ConvertTo-Json -Compress)
}

function Wait-TcpPort([int]$targetPort, [int]$timeoutSeconds = 180) {
    $deadline = (Get-Date).AddSeconds($timeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $client = [System.Net.Sockets.TcpClient]::new()
        try {
            $result = $client.BeginConnect("127.0.0.1", $targetPort, $null, $null)
            if ($result.AsyncWaitHandle.WaitOne(1000) -and $client.Connected) {
                $client.EndConnect($result)
                return
            }
        } catch {
        } finally {
            $client.Dispose()
        }
        Start-Sleep -Seconds 2
    }
    throw "OpenVLA server did not become ready on port $targetPort"
}

try {
    Write-Stage "waiting_for_training"
    Wait-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue
    $trainingReport = Join-Path $AdapterDir "training_report.json"
    $adapterWeights = Join-Path $AdapterDir "adapter_model.safetensors"
    if (-not (Test-Path -LiteralPath $trainingReport) -or -not (Test-Path -LiteralPath $adapterWeights)) {
        throw "training ended without adapter artifacts"
    }

    Write-Stage "starting_server"
    $serverArgs = @(
        "-u", "-m", "tools.openvla_tcp_server_water_cup",
        "--port", "$Port", "--adapter-dir", $AdapterDir
    )
    $serverProcess = Start-Process -FilePath $openvlaPython -ArgumentList $serverArgs `
        -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $serverOut `
        -RedirectStandardError $serverErr -PassThru
    Wait-TcpPort $Port

    Write-Stage "evaluating_seed0" @{ server_process_id = $serverProcess.Id }
    & $robocasaPython -u -m tools.openvla_gym_water_cup_diagnostic `
        --steps 700 --seed 0 --port $Port --object-scale 0.7 `
        --output-dir (Join-Path $root "outputs\water_cup_dagger_r1_eval_seed0")
    $seed0Exit = $LASTEXITCODE
    if ($seed0Exit -ne 0) {
        Write-Stage "seed0_failed" @{ seed0_exit_code = $seed0Exit; server_process_id = $serverProcess.Id }
        exit $seed0Exit
    }

    Write-Stage "evaluating_seed2" @{ seed0_exit_code = 0; server_process_id = $serverProcess.Id }
    & $robocasaPython -u -m tools.openvla_gym_water_cup_diagnostic `
        --steps 700 --seed 2 --port $Port --object-scale 0.7 `
        --output-dir (Join-Path $root "outputs\water_cup_dagger_r1_eval_seed2")
    $seed2Exit = $LASTEXITCODE
    if ($seed2Exit -ne 0) {
        Write-Stage "seed2_failed" @{ seed0_exit_code = 0; seed2_exit_code = $seed2Exit; server_process_id = $serverProcess.Id }
        exit $seed2Exit
    }
    Write-Stage "complete" @{ seed0_exit_code = 0; seed2_exit_code = 0 }
} catch {
    Write-Stage "pipeline_error" @{ error = $_.Exception.Message }
    throw
} finally {
    if ($null -ne $serverProcess -and -not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
