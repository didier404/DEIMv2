<#
脚本功能：
  这是 DEIMv2-L + DINOv3-S(vits16) backbone + 小金具五类缺陷数据的通用训练入口。
  它可以做三件事：
    1. smoke：用很小的 smoke annotations 检查环境、权重、数据读取、显存和训练链路。
    2. small：使用 G:/Data/Small_fetting_defect/data/DEIMv2_small 抽样小数据集正式训练。
    3. full ：使用 G:/Data/Small_fetting_defect/data/DEIMv2 全量数据集正式训练。

使用指令：
  先进入仓库目录：
    cd C:\Users\zd\Desktop\Compare\DEIMv2

  训练前 smoke 检查，默认就是 smoke：
    .\smoke_train_small_fitting_dinov3_l.ps1

  使用小数据集训练：
    .\smoke_train_small_fitting_dinov3_l.ps1 -Mode small

  使用全量数据集训练：
    .\smoke_train_small_fitting_dinov3_l.ps1 -Mode full

  常用参数示例：
    .\smoke_train_small_fitting_dinov3_l.ps1 -Mode small -BatchSize 2 -ValBatchSize 2 -NumWorkers 0 -InputSize 1280
    .\smoke_train_small_fitting_dinov3_l.ps1 -Mode small -BatchSize 4 -NumWorkers 2 -Epochs 68
    .\smoke_train_small_fitting_dinov3_l.ps1 -Mode small -ResumeCheckpoint outputs/deimv2_dinov3_l_deimv2_small_5cls/last.pth

参数说明：
  -Mode              smoke / small / full 三选一。
  -BatchSize         训练 total_batch_size。4090 先用 2 或 4；OOM 就调小。
  -ValBatchSize      验证 batch size。验证也占显存，OOM 就调小。
  -NumWorkers        Windows 建议 0-2，稳定后再试 4。
  -InputSize         训练和验证输入尺寸。小目标建议 1280；显存不够可试 1024。
  -Epochs            small/full 正式训练轮数；smoke 固定由 smoke yml 控制为 1。
  -UseAmp            是否启用混合精度。默认 true，想关闭用 -UseAmp:$false。
  -TuningCheckpoint  COCO 检测权重，用于迁移微调；默认 ckpt/deimv2_dinov3_l_coco.pth。
  -ResumeCheckpoint  中断后继续训练的 last.pth/best.pth。填它以后脚本会自动不用 -TuningCheckpoint。
  -DryRun            只打印将要执行的命令，不真正启动训练。

注意：
  1. 这个脚本只保存图片路径和 COCO annotations，不会复制原始图片。
  2. small 数据集对应的配置文件是 configs/deimv2/deimv2_dinov3_l_deimv2_small_5cls.yml。
  3. 这里保留 Dense O2O 的 Mosaic、MixUp、CopyBlend 思路；Mosaic/MixUp/CopyBlend 的概率可在参数区调整。
  4. 如果使用 -ResumeCheckpoint，不能同时从 -TuningCheckpoint 微调；脚本会优先 resume。
#>

param(
  [ValidateSet("smoke", "small", "full")]
  [string]$Mode = "smoke",

  [string]$CudaVisibleDevices = "0",
  [string]$PythonExe = "D:\miniconda3\envs\deimv2\python.exe",

  [int]$BatchSize = 4,
  [int]$ValBatchSize = 4,
  [int]$NumWorkers = 2,
  [int]$InputSize = 1280,
  [int]$Epochs = 68,
  [int]$Seed = 0,
  [bool]$UseAmp = $true,

  [string]$TuningCheckpoint = "ckpt/deimv2_dinov3_l_coco.pth",
  [string]$ResumeCheckpoint = "",

  [double]$MosaicProb = 0.3,
  [double]$MixupProb = 0.3,
  [double]$CopyBlendProb = 0.5,
  [int]$CopyBlendAreaThreshold = 64,

  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ($PSScriptRoot) {
  Set-Location $PSScriptRoot
}

$env:CUDA_VISIBLE_DEVICES = $CudaVisibleDevices
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

if (-not (Test-Path $PythonExe)) {
  throw "PythonExe 不存在：$PythonExe"
}

if ($ResumeCheckpoint -ne "") {
  $TuningCheckpoint = ""
}

switch ($Mode) {
  "smoke" {
    $Config = "configs/deimv2/deimv2_dinov3_l_small_fitting_5cls_smoke.yml"
    $OutputDir = "outputs/smoke_deimv2_dinov3_l_small_fitting_5cls"
    $ExtraUpdates = @(
      "eval_spatial_size=[$InputSize,$InputSize]",
      "train_dataloader.total_batch_size=$BatchSize",
      "val_dataloader.total_batch_size=$ValBatchSize",
      "train_dataloader.num_workers=$NumWorkers",
      "val_dataloader.num_workers=$NumWorkers",
      "train_dataloader.collate_fn.base_size=$InputSize",
      "train_dataloader.collate_fn.mixup_prob=$MixupProb",
      "train_dataloader.collate_fn.copyblend_prob=$CopyBlendProb",
      "train_dataloader.collate_fn.area_threshold=$CopyBlendAreaThreshold"
    )
  }
  "small" {
    $Config = "configs/deimv2/deimv2_dinov3_l_deimv2_small_5cls.yml"
    $OutputDir = "outputs/deimv2_dinov3_l_deimv2_small_5cls"
    $ExtraUpdates = @(
      "epoches=$Epochs",
      "eval_spatial_size=[$InputSize,$InputSize]",
      "train_dataloader.total_batch_size=$BatchSize",
      "val_dataloader.total_batch_size=$ValBatchSize",
      "train_dataloader.num_workers=$NumWorkers",
      "val_dataloader.num_workers=$NumWorkers",
      "train_dataloader.collate_fn.base_size=$InputSize",
      "train_dataloader.collate_fn.mixup_prob=$MixupProb",
      "train_dataloader.collate_fn.copyblend_prob=$CopyBlendProb",
      "train_dataloader.collate_fn.area_threshold=$CopyBlendAreaThreshold",
      "train_dataloader.dataset.transforms.mosaic_prob=$MosaicProb"
    )
  }
  "full" {
    $Config = "configs/deimv2/deimv2_dinov3_l_5cls.yml"
    $OutputDir = "outputs/deimv2_dinov3_l_5cls"
    $ExtraUpdates = @(
      "epoches=$Epochs",
      "eval_spatial_size=[$InputSize,$InputSize]",
      "train_dataloader.total_batch_size=$BatchSize",
      "val_dataloader.total_batch_size=$ValBatchSize",
      "train_dataloader.num_workers=$NumWorkers",
      "val_dataloader.num_workers=$NumWorkers",
      "train_dataloader.collate_fn.base_size=$InputSize",
      "train_dataloader.collate_fn.mixup_prob=$MixupProb",
      "train_dataloader.collate_fn.copyblend_prob=$CopyBlendProb",
      "train_dataloader.collate_fn.area_threshold=$CopyBlendAreaThreshold",
      "train_dataloader.dataset.transforms.mosaic_prob=$MosaicProb"
    )
  }
}

if (-not (Test-Path $Config)) {
  throw "Config 不存在：$Config"
}
if ($TuningCheckpoint -ne "" -and -not (Test-Path $TuningCheckpoint)) {
  throw "TuningCheckpoint 不存在：$TuningCheckpoint"
}
if ($ResumeCheckpoint -ne "" -and -not (Test-Path $ResumeCheckpoint)) {
  throw "ResumeCheckpoint 不存在：$ResumeCheckpoint"
}

$ArgsList = @(
  "train.py",
  "-c", $Config,
  "--seed", "$Seed",
  "--output-dir", $OutputDir
)

if ($UseAmp) {
  $ArgsList += "--use-amp"
}

if ($ResumeCheckpoint -ne "") {
  $ArgsList += @("-r", $ResumeCheckpoint)
}
elseif ($TuningCheckpoint -ne "") {
  $ArgsList += @("-t", $TuningCheckpoint)
}

if ($ExtraUpdates.Count -gt 0) {
  $ArgsList += "-u"
  $ArgsList += $ExtraUpdates
}

Write-Host "Mode: $Mode"
Write-Host "Config: $Config"
Write-Host "Output: $OutputDir"
Write-Host "Python: $PythonExe"
Write-Host "CUDA_VISIBLE_DEVICES: $CudaVisibleDevices"
Write-Host "Command: $PythonExe $($ArgsList -join ' ')"

if ($DryRun) {
  Write-Host "DryRun=true，只打印命令，不启动训练。"
  exit 0
}

& $PythonExe @ArgsList
