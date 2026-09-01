$ErrorActionPreference = "Continue"
$projectRoot = "C:\OpenVLA-Simulator"
$pythonRoboCasa = "C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe"
$outputDirectory = Join-Path $projectRoot "datasets\water_cup_expert"

Set-Location $projectRoot
foreach ($seed in 7..18) {
    Write-Output "COLLECT_SEED_START seed=$seed timestamp=$((Get-Date).ToString('o'))"
    & $pythonRoboCasa -m tools.collect_water_cup_expert `
        --seed $seed `
        --frame-stride 4 `
        --output-dir $outputDirectory
    Write-Output "COLLECT_SEED_END seed=$seed exit_code=$LASTEXITCODE timestamp=$((Get-Date).ToString('o'))"
}
Write-Output "COLLECTION_RANGE_COMPLETE timestamp=$((Get-Date).ToString('o'))"
