# -*- coding: utf-8 -*-
r"""
脚本功能：
  通用训练入口。这个脚本不会重写 DEIMv2 的训练逻辑，只是把常用参数整理好，
  自动生成一份临时 yml 配置，然后调用仓库原本的 train.py。

推荐用法：
  1. 先进入仓库目录：
       cd C:\Users\zd\Desktop\Compare\DEIMv2

  2. 使用 DEIMv2_small 小数据集训练：
       D:\miniconda3\envs\deimv2\python.exe train_model.py --preset small

  3. 使用全量数据集训练：
       D:\miniconda3\envs\deimv2\python.exe train_model.py --preset full

  3.1 使用 D 盘 SSD 图片缓存版 DEIMv2_small 训练：
       D:\miniconda3\envs\deimv2\python.exe train_model.py --preset small-ssd

  3.2 使用 D 盘 SSD 数据、3072 输入训练，并把输出保存到 F 盘：
       D:\miniconda3\envs\deimv2\python.exe train_model.py --preset small-ssd-3072 --img-size 3072 --batch-size 2 --val-batch-size 1 --workers 2 --seed 0
       如果指定 GPU：
       python train_model.py --preset small-ssd-3072 --device 2 --nproc-per-node 1

  3.3 单台服务器使用两张 GPU 训练（batch size 是两张卡合计值）：
       python train_model.py --preset server-3072 --device 0,1 --nproc-per-node 2 --batch-size 4 --val-batch-size 2 --workers 2 --seed 0

  3.4 单台服务器使用多张 GPU 训练：
       python **** 和双卡一致，--device 1,3,5,7 --nproc-per-node 4

  4. 只做 smoke 链路检查：
       D:\miniconda3\envs\deimv2\python.exe train_model.py --preset smoke --img-size 640 --batch-size 1 --workers 0

  5. 常用自定义参数：
       D:\miniconda3\envs\deimv2\python.exe train_model.py --preset small --epochs 68 --batch-size 2 --val-batch-size 2 --img-size 1280 --workers 0

  6. 中断后继续训练：
       D:\miniconda3\envs\deimv2\python.exe train_model.py --preset small --resume outputs/deimv2_dinov3_l_deimv2_small_5cls/last.pth

说明：
  --preset small/full/smoke 会自动选择当前项目已经准备好的 yml。
  --config 可以手动指定任意 yml；如果同时给了 --config，就以 --config 为准。
  --tuning 是迁移微调权重，默认使用 ckpt/deimv2_dinov3_l_coco.pth。
  --resume 是续训 checkpoint；使用 --resume 时会自动不使用 --tuning。
  新训练默认会在 preset 的输出根目录下创建时间子文件夹，例如 outputs/.../20260629_153000。
  使用 --resume 且不指定 --output-dir 时，会自动沿用 checkpoint 所在文件夹。
  --img-size 会同时覆盖训练 Resize、验证 Resize、eval_spatial_size 和 collate_fn.base_size。
  --nproc-per-node 大于 1 时会通过 torch.distributed.run 启动单机 DDP 多卡训练。
  --batch-size 和 --val-batch-size 都是所有 GPU 合计的 total_batch_size，必须能被进程数整除。
  --save-period 控制每隔多少个 epoch 额外保存一次 checkpoint；last.pth 每个 epoch 都会覆盖保存。
  --print-freq 控制每隔多少个 step 打印一次训练进度。
  --console-log 控制完整终端输出保存到哪里；默认保存到输出目录的 train_console.log。
  --dry-run 只打印命令，不启动训练，适合检查参数。
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from engine.core import yaml_utils


REPO_DIR = Path(__file__).resolve().parent

PRESETS = {
    "small": {
        "config": "configs/deimv2/deimv2_dinov3_l_deimv2_small_5cls.yml",
        "output_dir": "outputs/deimv2_dinov3_l_deimv2_small_5cls",
        "img_size": 1280,
        "batch_size": 4,
        "val_batch_size": 4,
        "workers": 2,
    },
    "small-ssd": {
        "config": "configs/deimv2/deimv2_dinov3_l_deimv2_small_ssd_5cls.yml",
        "output_dir": "outputs/deimv2_dinov3_l_deimv2_small_ssd_5cls",
        "img_size": 1280,
        "batch_size": 4,
        "val_batch_size": 4,
        "workers": 2,
    },
    "small-ssd-3072": {
        "config": "configs/deimv2/deimv2_dinov3_l_deimv2_small_ssd_3072_5cls.yml",
        "output_dir": "F:/Data/Small_fetting_defect/model_outputs/deimv2_dinov3_l_deimv2_small_ssd_3072_5cls",
        "img_size": 3072,
        "batch_size": 2,
        "val_batch_size": 1,
        "workers": 2,
    },
    "server-3072": {
        "config": "configs/deimv2/deimv2_dinov3_l_server_3072_5cls.yml",
        "output_dir": "outputs/deimv2_dinov3_l_server_3072_5cls",
        "img_size": 3072,
        "batch_size": 2,
        "val_batch_size": 4,
        "workers": 2,
    },
    "full": {
        "config": "configs/deimv2/deimv2_dinov3_l_5cls.yml",
        "output_dir": "outputs/deimv2_dinov3_l_5cls",
        "img_size": 1280,
        "batch_size": 4,
        "val_batch_size": 4,
        "workers": 2,
    },
    "smoke": {
        "config": "configs/deimv2/deimv2_dinov3_l_small_fitting_5cls_smoke.yml",
        "output_dir": "outputs/smoke_deimv2_dinov3_l_small_fitting_5cls",
        "img_size": 640,
        "batch_size": 1,
        "val_batch_size": 1,
        "workers": 0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DEIMv2 通用训练入口：整理常用参数后调用原始 train.py"
    )

    parser.add_argument("--preset", type=str,
                        choices=["small", "small-ssd", "small-ssd-3072", "server-3072", "full", "smoke"], default="small",
                        help="预设配置：server-3072=服务器数据、3072输入、outputs输出")
    parser.add_argument("--config", type=str, default="",
                        help="训练配置 yml。留空时根据 --preset 自动选择")
    parser.add_argument("--output-dir", type=str, default="",
                        help="输出目录。留空时根据 --preset 自动选择，并为新训练自动加时间子文件夹")
    parser.add_argument("--run-name", type=str, default="",
                        help="新训练的子文件夹名称。留空时自动使用当前时间，例如 20260629_153000")
    parser.add_argument("--no-time-dir", action="store_true",
                        help="新训练时不自动创建时间子文件夹，直接写入输出根目录")
    parser.add_argument("--tuning", type=str, default="ckpt/deimv2_dinov3_l_coco.pth",
                        help="迁移微调权重，默认使用 DEIMv2-L + DINOv3-S 的 COCO 权重")
    parser.add_argument("--resume", type=str, default="",
                        help="从指定 checkpoint 续训，例如 outputs/.../last.pth。使用后自动不加载 --tuning")
    parser.add_argument("--epochs", type=int, default=68,
                        help="训练轮数，默认 68。smoke 配置一般保持 1")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="训练 total_batch_size；留空时使用预设默认值")
    parser.add_argument("--val-batch-size", type=int, default=None,
                        help="验证 total_batch_size；留空时使用预设默认值")
    parser.add_argument("--img-size", type=int, default=None,
                        help="输入图像尺寸；留空时使用预设默认值")
    parser.add_argument("--device", type=str, default="0",
                        help="训练设备，默认 0；CPU 可填 cpu；多卡通常用 0,1")
    parser.add_argument("--nproc-per-node", type=int, default=1,
                        help="单机训练进程数，通常等于 GPU 数量；两卡填写 2")
    parser.add_argument("--master-port", type=int, default=7777,
                        help="DDP 主进程通信端口，默认 7777；端口占用时换一个")
    parser.add_argument("--workers", type=int, default=None,
                        help="数据加载线程数；留空时使用预设默认值。Windows 建议 0-2")
    parser.add_argument("--save-period", type=int, default=5,
                        help="每隔多少个 epoch 额外保存一次 checkpoint，默认 5。last.pth 每个 epoch 都保存")
    parser.add_argument("--print-freq", type=int, default=100,
                        help="每隔多少个 step 打印一次训练日志，默认 100")
    parser.add_argument("--console-log", type=str, default="train_console.log",
                        help="完整终端输出日志文件名或路径，默认保存到 output-dir/train_console.log；填空字符串可关闭")
    parser.add_argument("--lr", type=float, default=None,
                        help="主干以外模块的基础学习率。留空则使用 yml 默认值")
    parser.add_argument("--backbone-lr", type=float, default=None,
                        help="DINOv3 backbone 学习率。留空则使用 yml 默认值")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="权重衰减。留空则使用 yml 默认值")
    parser.add_argument("--warmup-iters", type=int, default=None,
                        help="warmup iteration 数。留空则使用 yml 默认值")
    parser.add_argument("--mosaic-prob", type=float, default=0.5,
                        help="Mosaic 概率，默认 0.5，贴近原版 DEIMv2")
    parser.add_argument("--mixup-prob", type=float, default=0.5,
                        help="MixUp 概率，默认 0.5，贴近原版 DEIMv2")
    parser.add_argument("--copyblend-prob", type=float, default=0.5,
                        help="CopyBlend 概率，默认 0.5，贴近原版 DEIMv2")
    parser.add_argument("--copyblend-area-threshold", type=int, default=100,
                        help="CopyBlend 目标面积阈值，默认 100，贴近原版 DEIMv2")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子，默认 0")
    parser.add_argument("--no-amp", action="store_true",
                        help="关闭混合精度。默认启用 AMP")
    parser.add_argument("--test-only", action="store_true",
                        help="只验证不训练")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将要执行的命令，不启动训练")

    return parser.parse_args()


def set_if_not_none(dct: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        dct[key] = value


def apply_preset_defaults(args: argparse.Namespace, preset: dict[str, Any]) -> None:
    for name in ("img_size", "batch_size", "val_batch_size", "workers"):
        if getattr(args, name) is None:
            setattr(args, name, preset[name])


def validate_distributed_args(args: argparse.Namespace) -> None:
    if args.nproc_per_node < 1:
        raise ValueError("--nproc-per-node 必须大于等于 1")
    if args.nproc_per_node == 1:
        return
    if args.device.lower() == "cpu":
        raise ValueError("多进程训练需要 CUDA GPU，不能同时使用 --device cpu")

    visible_devices = [item.strip() for item in args.device.split(",") if item.strip()]
    if len(visible_devices) < args.nproc_per_node:
        raise ValueError(
            f"--device 只指定了 {len(visible_devices)} 张卡，但 --nproc-per-node={args.nproc_per_node}"
        )
    if args.batch_size % args.nproc_per_node != 0:
        raise ValueError("--batch-size 是总 batch size，必须能被 --nproc-per-node 整除")
    if args.val_batch_size % args.nproc_per_node != 0:
        raise ValueError("--val-batch-size 是总 batch size，必须能被 --nproc-per-node 整除")


def set_all_resize_ops(transforms: dict[str, Any], img_size: int) -> None:
    for op in transforms.get("ops", []) or []:
        if isinstance(op, dict) and op.get("type") == "Resize":
            op["size"] = [img_size, img_size]


def set_mosaic_output_size(transforms: dict[str, Any], img_size: int) -> None:
    for op in transforms.get("ops", []) or []:
        if isinstance(op, dict) and op.get("type") == "Mosaic":
            op["output_size"] = img_size // 2


def build_config(args: argparse.Namespace, source_config: str, output_dir: str) -> Path:
    cfg = yaml_utils.load_config(str(REPO_DIR / source_config), cfg={})
    cfg.pop("__include__", None)

    if args.preset != "smoke":
        cfg["epoches"] = args.epochs

    cfg["output_dir"] = output_dir
    cfg["eval_spatial_size"] = [args.img_size, args.img_size]
    cfg["checkpoint_freq"] = args.save_period
    cfg["print_freq"] = args.print_freq

    train_loader = cfg["train_dataloader"]
    val_loader = cfg["val_dataloader"]

    train_loader["total_batch_size"] = args.batch_size
    val_loader["total_batch_size"] = args.val_batch_size
    train_loader["num_workers"] = args.workers
    val_loader["num_workers"] = args.workers

    set_all_resize_ops(train_loader["dataset"]["transforms"], args.img_size)
    set_all_resize_ops(val_loader["dataset"]["transforms"], args.img_size)
    set_mosaic_output_size(train_loader["dataset"]["transforms"], args.img_size)

    train_transforms = train_loader["dataset"]["transforms"]
    train_transforms["mosaic_prob"] = args.mosaic_prob

    collate_fn = train_loader["collate_fn"]
    collate_fn["base_size"] = args.img_size
    collate_fn["mixup_prob"] = args.mixup_prob
    collate_fn["copyblend_prob"] = args.copyblend_prob
    collate_fn["area_threshold"] = args.copyblend_area_threshold

    optimizer = cfg["optimizer"]
    set_if_not_none(optimizer, "lr", args.lr)
    set_if_not_none(optimizer, "weight_decay", args.weight_decay)
    set_if_not_none(cfg, "warmup_iter", args.warmup_iters)

    if args.backbone_lr is not None:
        optimizer["params"][0]["lr"] = args.backbone_lr
        optimizer["params"][1]["lr"] = args.backbone_lr

    generated_dir = REPO_DIR / output_dir / "_generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated_config = generated_dir / f"train_model_{args.preset}_{args.img_size}.yml"
    with generated_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return generated_config


def resolve_output_dir(args: argparse.Namespace, preset_output_dir: str) -> str:
    if args.output_dir:
        return args.output_dir

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = REPO_DIR / resume_path
        try:
            return str(resume_path.parent.relative_to(REPO_DIR))
        except ValueError:
            return str(resume_path.parent)

    if args.no_time_dir:
        return preset_output_dir

    run_name = args.run_name or _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(preset_output_dir) / run_name)


def resolve_console_log_path(console_log: str, output_dir: str) -> Path | None:
    if not console_log:
        return None
    path = Path(console_log)
    if not path.is_absolute():
        path = REPO_DIR / output_dir / path
    return path


def run_and_tee(cmd: list[str], env: dict[str, str], console_log_path: Path | None) -> int:
    if console_log_path is None:
        return subprocess.call(cmd, cwd=REPO_DIR, env=env)

    console_log_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "",
        "=" * 100,
        f"Start time: {_datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Command: {' '.join(cmd)}",
        f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}",
        "=" * 100,
        "",
    ]

    with console_log_path.open("a", encoding="utf-8", errors="replace") as log_f:
        for line in header:
            print(line)
            log_f.write(line + "\n")
        log_f.flush()

        process = subprocess.Popen(
            cmd,
            cwd=REPO_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_f.write(line)
            log_f.flush()

        return_code = process.wait()
        footer = [
            "",
            "=" * 100,
            f"End time: {_datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Return code: {return_code}",
            "=" * 100,
            "",
        ]
        for line in footer:
            print(line)
            log_f.write(line + "\n")
        log_f.flush()
        return return_code


def main() -> int:
    args = parse_args()
    preset = PRESETS[args.preset]
    apply_preset_defaults(args, preset)
    validate_distributed_args(args)

    source_config = args.config or preset["config"]
    output_dir = resolve_output_dir(args, preset["output_dir"])

    if not (REPO_DIR / source_config).exists():
        raise FileNotFoundError(f"配置文件不存在：{REPO_DIR / source_config}")

    if args.resume:
        checkpoint = REPO_DIR / args.resume
        if not checkpoint.exists():
            raise FileNotFoundError(f"续训 checkpoint 不存在：{checkpoint}")
    elif args.tuning:
        checkpoint = REPO_DIR / args.tuning
        if not checkpoint.exists():
            raise FileNotFoundError(f"微调权重不存在：{checkpoint}")

    generated_config = build_config(args, source_config, output_dir)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.device.lower() == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        device_arg = "cpu"
    else:
        env["CUDA_VISIBLE_DEVICES"] = args.device
        device_arg = ""

    cmd = [sys.executable]
    if args.nproc_per_node > 1:
        cmd.extend([
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={args.nproc_per_node}",
            f"--master_port={args.master_port}",
        ])

    cmd.extend([
        "train.py",
        "-c",
        str(generated_config),
        "--seed",
        str(args.seed),
        "--output-dir",
        output_dir,
    ])

    if not args.no_amp:
        cmd.append("--use-amp")
    if device_arg:
        cmd.extend(["--device", device_arg])
    if args.test_only:
        cmd.append("--test-only")

    if args.resume:
        cmd.extend(["-r", args.resume])
    elif args.tuning:
        cmd.extend(["-t", args.tuning])

    print("已生成临时配置：")
    print(generated_config)
    print("即将执行训练命令：")
    print(" ".join(cmd))
    print(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}")
    console_log_path = resolve_console_log_path(args.console_log, output_dir)
    if console_log_path is not None:
        print(f"完整终端输出日志：{console_log_path}")

    if args.dry_run:
        print("dry-run 模式：只打印命令，不启动训练。")
        return 0

    return run_and_tee(cmd, env, console_log_path)


if __name__ == "__main__":
    raise SystemExit(main())
