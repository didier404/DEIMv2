$ErrorActionPreference = "Stop"

# Train DEIMv2-L with DINOv3-S backbone on the 5-class small fitting defect dataset.
# Run from PowerShell:
#   cd C:\Users\zd\Desktop\Compare\DEIMv2
#   .\train_small_fitting_dinov3_l.ps1
#
# Resume interrupted training:
#   D:\miniconda3\envs\deimv2\python.exe train.py -c configs/deimv2/deimv2_dinov3_l_5cls.yml --use-amp -r outputs/deimv2_dinov3_l_5cls/checkpoint.pth
#
# Validate a trained checkpoint:
#   D:\miniconda3\envs\deimv2\python.exe train.py -c configs/deimv2/deimv2_dinov3_l_5cls.yml --test-only -r outputs/deimv2_dinov3_l_5cls/best.pth

$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

D:\miniconda3\envs\deimv2\python.exe train.py `
  -c configs/deimv2/deimv2_dinov3_l_5cls.yml `
  --use-amp `
  --seed 0 `
  --output-dir outputs/deimv2_dinov3_l_5cls `
  -t ckpt/deimv2_dinov3_l_coco.pth
