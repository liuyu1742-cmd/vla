$ErrorActionPreference = 'Stop'
$ProjectRoot = 'C:\OpenVLA-Simulator'
$LogPath = Join-Path $ProjectRoot 'outputs\water_cup_full_train.log'
Set-Location $ProjectRoot
& 'C:\Users\sjtu101\miniconda3\envs\openvla\python.exe' -m tools.finetune_water_cup_openvla_local --steps 100 --output models\openvla-water-cup-lora-full 2>&1 | Tee-Object -FilePath $LogPath
