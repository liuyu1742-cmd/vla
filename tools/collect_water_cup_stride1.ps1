$ErrorActionPreference = "Continue"
$projectRoot = "C:\OpenVLA-Simulator"
$pythonRoboCasa = "C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe"
$outputDirectory = Join-Path $projectRoot "datasets\water_cup_expert_stride1"
$seeds = @(0, 1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 18)

Set-Location $projectRoot
foreach ($seed in $seeds) {
    Write-Output "STRIDE1_SEED_START seed=$seed timestamp=$((Get-Date).ToString('o'))"
    & $pythonRoboCasa -m tools.collect_water_cup_expert `
        --seed $seed `
        --frame-stride 1 `
        --output-dir $outputDirectory
    Write-Output "STRIDE1_SEED_END seed=$seed exit_code=$LASTEXITCODE timestamp=$((Get-Date).ToString('o'))"
}
Write-Output "STRIDE1_COLLECTION_COMPLETE timestamp=$((Get-Date).ToString('o'))"
