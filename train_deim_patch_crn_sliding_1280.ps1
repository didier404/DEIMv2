$ErrorActionPreference = "Continue"
# DEIMv2-L + DINOv3-S(vits16) @ crn_sliding 1280 patch, 本地 4090 48G 单卡训练。
# 三个 patch 策略实验共用同一超参/seed(42)/COCO 预训练权重，保证单变量对比。
# 中断续跑: 加参数 -r <输出目录>/last.pth（替换 -t 权重加载行）。

Set-Location -LiteralPath $PSScriptRoot
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONWARNINGS = "ignore::FutureWarning"
$env:PYTHONUNBUFFERED = "1"

$OutputDir = "F:/Data/Small_fetting_defect/model_outputs/deimv2_outputs/deimv2_dinov3_l_patch_crn_sliding_1280"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

D:\miniconda3\envs\deimv2\python.exe train.py `
  -c configs/deimv2/deimv2_patch_convnext_large_crn_sliding_1280.yml `
  --use-amp `
  -u "print_freq=100" `
  --seed 42 `
  --output-dir $OutputDir `
  -t ckpt/deimv2_dinov3_l_coco.pth `
  2>&1 | Tee-Object -FilePath "$OutputDir/train_console.log"
