#!/usr/bin/env python3
"""下载 lada 模型权重到 ComfyUI/models/lada/。

用法：
    # 1. 默认（最高质量组合，约 332 MB；不考虑速度，要最好的效果）
    #    检测: lada_mosaic_detection_model_v4_accurate.pt    (45 MB)
    #    修复: lada_mosaic_restoration_model_generic_v1.2_full.pth (287 MB)
    python scripts/download_models.py

    # 2. 速度优先组合（约 84 MB）
    python scripts/download_models.py --preset fast

    # 3. 指定输出目录
    python scripts/download_models.py --dest /root/autodl-tmp/comfyui/models/lada

    # 4. 国内/AutoDL 用 HF 镜像
    python scripts/download_models.py --mirror

    # 5. 下载全部模型（含 NSFW / 水印检测等，约 800 MB）
    python scripts/download_models.py --all

    # 6. 自定义文件
    python scripts/download_models.py --files \\
        lada_mosaic_detection_model_v3.1_accurate.pt \\
        lada_mosaic_restoration_model_generic_v1.2_full.pth

依赖：只用 Python 标准库（urllib），不需要装 huggingface_hub。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional

HF_HOST_OFFICIAL = "https://huggingface.co"
HF_HOST_MIRROR = "https://hf-mirror.com"
HF_REPO = "ladaapp/lada"

PRESETS = {
    # 默认：最高质量，不考虑速度（45 MB + 287 MB ≈ 332 MB）
    "best": [
        "lada_mosaic_detection_model_v4_accurate.pt",
        "lada_mosaic_restoration_model_generic_v1.2_full.pth",
    ],
    # 平衡：精度优先但更省体积（20 MB + 78 MB ≈ 98 MB）
    "balanced": [
        "lada_mosaic_detection_model_v3.1_accurate.pt",
        "lada_mosaic_restoration_model_generic_v1.2.pth",
    ],
    # 速度优先：最小体积（6 MB + 78 MB ≈ 84 MB）
    "fast": [
        "lada_mosaic_detection_model_v3.1_fast.pt",
        "lada_mosaic_restoration_model_generic_v1.2.pth",
    ],
}
DEFAULT_PRESET = "best"

ALL_FILES = [
    "lada_mosaic_detection_model_v2.pt",
    "lada_mosaic_detection_model_v3.pt",
    "lada_mosaic_detection_model_v3.1_fast.pt",
    "lada_mosaic_detection_model_v3.1_accurate.pt",
    "lada_mosaic_detection_model_v4_fast.pt",
    "lada_mosaic_detection_model_v4_accurate.pt",
    "lada_mosaic_restoration_model_generic_v1.1.pth",
    "lada_mosaic_restoration_model_generic_v1.2.pth",
    "lada_mosaic_restoration_model_generic_v1.2_full.pth",
    "lada_nsfw_detection_model.pt",
    "lada_nsfw_detection_model_v1.1.pt",
    "lada_nsfw_detection_model_v1.2.pt",
    "lada_nsfw_detection_model_v1.3.pt",
    "lada_watermark_detection_model.pt",
    "lada_watermark_detection_model_v1.1.pt",
    "lada_watermark_detection_model_v1.2.pt",
    "lada_watermark_detection_model_v1.3.pt",
]


def detect_default_dest() -> Path:
    """猜 ComfyUI/models/lada 的位置。

    优先级：
    1. 环境变量 LADA_MODEL_WEIGHTS_DIR
    2. 脚本所在仓库的父级 ../models/lada（也就是 custom_nodes/.../scripts/ 上面那一层 models/lada）
    3. 当前工作目录下的 ./models/lada
    """
    if env := os.environ.get("LADA_MODEL_WEIGHTS_DIR"):
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve().parent  # .../ComfyUI_RemoveMosaic/scripts
    plugin_dir = here.parent  # .../ComfyUI_RemoveMosaic
    custom_nodes_dir = plugin_dir.parent  # .../custom_nodes
    if custom_nodes_dir.name == "custom_nodes":
        comfy_root = custom_nodes_dir.parent
        candidate = comfy_root / "models" / "lada"
        return candidate.resolve()

    return (Path.cwd() / "models" / "lada").resolve()


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0:
            return f"{n:6.1f} {unit}"
        n /= 1024.0
    return f"{n:6.1f} TB"


def _print_progress(downloaded: int, total: int, speed: float, prefix: str) -> None:
    if total > 0:
        pct = downloaded * 100.0 / total
        bar_len = 30
        filled = int(bar_len * downloaded / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        msg = f"\r{prefix} [{bar}] {pct:5.1f}%  {human_bytes(downloaded)}/{human_bytes(total)}  {human_bytes(speed)}/s"
    else:
        msg = f"\r{prefix} {human_bytes(downloaded)}  {human_bytes(speed)}/s"
    sys.stdout.write(msg)
    sys.stdout.flush()


def download_one(
    url: str,
    dest: Path,
    *,
    overwrite: bool = False,
    timeout: int = 30,
    chunk_size: int = 1 << 20,  # 1 MB
) -> bool:
    """断点续传 + 进度条下载。返回是否成功。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and not overwrite:
        print(f"  ✓ 已存在，跳过：{dest.name}  ({human_bytes(dest.stat().st_size)})")
        return True

    resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
    headers = {"User-Agent": "ComfyUI_RemoveMosaic/downloader"}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"
        print(f"  ↻ 续传：从 {human_bytes(resume_from)} 继续")

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            content_length = resp.headers.get("Content-Length")
            total = int(content_length) + resume_from if content_length else 0
            mode = "ab" if (resume_from > 0 and status == 206) else "wb"
            if mode == "wb":
                resume_from = 0  # server ignored Range; restart

            downloaded = resume_from
            t0 = time.time()
            t_last = t0
            with open(tmp_path, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - t_last >= 0.3 or downloaded == total:
                        speed = (downloaded - resume_from) / max(now - t0, 1e-6)
                        _print_progress(downloaded, total, speed, f"  ↓ {dest.name}")
                        t_last = now
            sys.stdout.write("\n")
    except urllib.error.HTTPError as e:
        sys.stdout.write("\n")
        print(f"  ✗ HTTP {e.code}: {e.reason}")
        return False
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        sys.stdout.write("\n")
        print(f"  ✗ 网络错误：{e}")
        return False
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        print("  ⏸  用户中断，已保留 .part 文件，下次运行可断点续传。")
        raise

    shutil.move(str(tmp_path), str(dest))
    return True


def build_url(host: str, filename: str) -> str:
    return f"{host.rstrip('/')}/{HF_REPO}/resolve/main/{filename}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="下载 lada 模型权重到 ComfyUI/models/lada/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="目标目录，默认自动定位 ComfyUI/models/lada",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help=f"使用镜像 {HF_HOST_MIRROR}（国内/AutoDL 推荐）",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=f"自定义 HF 镜像地址（覆盖 --mirror，例如 {HF_HOST_MIRROR}）",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default=DEFAULT_PRESET,
        help=(
            "选择模型组合：\n"
            "  best     (默认) 最高质量，不考虑速度，约 332 MB\n"
            "  balanced 精度优先但更小，约 98 MB\n"
            "  fast     速度优先，约 84 MB"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="下载全部模型（含 NSFW/水印/full 等，约 800 MB）",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="只下载指定文件名（覆盖 --preset / --all）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="即使本地已存在也重新下载",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="只列出可下载的文件名，不下载",
    )
    args = parser.parse_args(argv)

    if args.list:
        for name, items in PRESETS.items():
            tag = " (默认)" if name == DEFAULT_PRESET else ""
            print(f"# preset: {name}{tag}")
            for f in items:
                print(f"  - {f}")
            print()
        print("# 全部可选：")
        for f in ALL_FILES:
            print(f"  - {f}")
        return 0

    host = args.host or (HF_HOST_MIRROR if args.mirror else HF_HOST_OFFICIAL)

    if args.files:
        files: List[str] = list(args.files)
    elif args.all:
        files = list(ALL_FILES)
    else:
        files = list(PRESETS[args.preset])

    dest = (args.dest or detect_default_dest()).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    print(f"目标目录: {dest}")
    print(f"下载源:   {host}")
    print(f"文件 ({len(files)}):")
    for f in files:
        print(f"  - {f}")
    print()

    failed: List[str] = []
    for filename in files:
        url = build_url(host, filename)
        print(f"==> {filename}")
        ok = download_one(url, dest / filename, overwrite=args.overwrite)
        if not ok:
            failed.append(filename)
        print()

    print("=" * 60)
    print(f"完成: {len(files) - len(failed)}/{len(files)} 成功")
    if failed:
        print("失败列表（可重新执行脚本继续）:")
        for f in failed:
            print(f"  - {f}")
        if not args.mirror and host == HF_HOST_OFFICIAL:
            print("\n提示: 网络不通可以加 --mirror 用 hf-mirror.com 重试。")
        return 1
    print(f"模型目录: {dest}")
    print("重启 ComfyUI 后即可在节点下拉里看到这些模型。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
