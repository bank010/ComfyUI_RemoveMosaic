"""Adapter between ComfyUI's IMAGE batches and the vendored Lada pipeline.

The vendored ``lada.restorationpipeline.frame_restorer.FrameRestorer`` is a
multi-threaded pipeline built around an on-disk video file (it instantiates
its own ``av.open`` based ``VideoReader``). To keep behaviour identical to
upstream lada we don't re-implement that pipeline; instead we encode the
ComfyUI ``IMAGE`` batch into a temporary lossless ``ffv1`` MKV, run the
restorer, and decode the result back into an ``IMAGE`` tensor.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger("ComfyUI_RemoveMosaic")


def _ensure_ffmpeg_binaries() -> None:
    """Make sure ``ffmpeg`` and ``ffprobe`` are available on PATH.

    Lada's pipeline shells out to both binaries via ``subprocess`` (e.g. for
    ``video_utils.get_video_meta_data``). When they are missing the failure
    happens deep inside the worker threads and the resulting ``FileNotFoundError``
    isn't very helpful, so we surface a clear hint here instead.
    """
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if not missing:
        return
    raise RuntimeError(
        "ComfyUI_RemoveMosaic requires {missing} on PATH but could not find "
        "{count}. Install ffmpeg and try again:\n"
        "  - apt:   apt update && apt install -y ffmpeg\n"
        "  - conda: conda install -y -c conda-forge ffmpeg\n"
        "  - macOS: brew install ffmpeg\n"
        "  - Win:   choco install ffmpeg  (or download from ffmpeg.org and add to PATH)"
        .format(missing=", ".join(missing), count="them" if len(missing) > 1 else "it")
    )


# ---------------------------------------------------------------------------
# Lazy lada import ----------------------------------------------------------
# ---------------------------------------------------------------------------

def _import_lada():
    try:
        import lada  # noqa: F401
        from lada import ModelFiles  # noqa: F401
        from lada.models.basicvsrpp.inference import load_model as _load_basicvsrpp  # noqa: F401
        from lada.models.yolo.yolo11_segmentation_model import (  # noqa: F401
            Yolo11SegmentationModel,
        )
        from lada.restorationpipeline.basicvsrpp_mosaic_restorer import (  # noqa: F401
            BasicvsrppMosaicRestorer,
        )
        from lada.restorationpipeline.frame_restorer import FrameRestorer  # noqa: F401
        from lada.utils.threading_utils import STOP_MARKER, ErrorMarker  # noqa: F401
        from lada.utils.os_utils import (  # noqa: F401
            gpu_has_fp16_acceleration,
            get_default_torch_device,
            has_mps,
        )
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "ComfyUI_RemoveMosaic could not import its vendored lada code. "
            "Make sure the following packages are installed in the ComfyUI "
            "Python env: torch, torchvision, mmengine, ultralytics, av, "
            "opencv-python. Original error: " + str(e)
        ) from e


# ---------------------------------------------------------------------------
# Wrappers around the loaded models ----------------------------------------
# ---------------------------------------------------------------------------

@dataclass
class LadaDetectionModel:
    """Mosaic-detection model handle (YOLOv11 segmentation)."""
    model: object  # lada.models.yolo.yolo11_segmentation_model.Yolo11SegmentationModel
    name: str
    path: str
    device: torch.device
    fp16: bool


@dataclass
class LadaRestorationModel:
    """Mosaic-restoration model handle (BasicVSR++ GAN or DeepMosaics)."""
    restorer: object  # BasicvsrppMosaicRestorer or DeepmosaicsMosaicRestorer
    backbone: str  # 'basicvsrpp' or 'deepmosaics'
    name: str
    path: str
    device: torch.device
    fp16: bool
    pad_mode: str  # 'zero' for basicvsrpp, 'reflect' for deepmosaics


# ---------------------------------------------------------------------------
# Device / fp16 helpers -----------------------------------------------------
# ---------------------------------------------------------------------------

def resolve_device(device: str) -> torch.device:
    _import_lada()
    from lada.utils.os_utils import get_default_torch_device, has_mps  # type: ignore

    if device in (None, "", "auto"):
        return torch.device(get_default_torch_device())
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if device == "mps" and not has_mps():
        logger.warning("MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def resolve_fp16(fp16: str | bool | None, device: torch.device) -> bool:
    _import_lada()
    from lada.utils.os_utils import gpu_has_fp16_acceleration  # type: ignore

    if isinstance(fp16, bool):
        return fp16
    if fp16 in (None, "auto"):
        return gpu_has_fp16_acceleration(device)
    return fp16 == "enable"


# ---------------------------------------------------------------------------
# Listing models on disk ----------------------------------------------------
# ---------------------------------------------------------------------------

def list_detection_files() -> List[str]:
    """Return a list of available detection model display names.

    The names mirror lada's well-known names (``v4-fast`` etc.) when the
    canonical filenames are present, plus any other ``.pt`` files found in
    ``ComfyUI/models/lada/``.
    """
    _import_lada()
    from lada import ModelFiles, MODEL_WEIGHTS_DIR

    ModelFiles.reset_cache()
    names = [m.name for m in ModelFiles.get_detection_models()]
    # Also surface raw .pt files that don't follow lada's naming.
    if os.path.isdir(MODEL_WEIGHTS_DIR):
        for fn in sorted(os.listdir(MODEL_WEIGHTS_DIR)):
            if fn.endswith(".pt") and fn not in names and not any(
                fn == os.path.basename(m.path)
                for m in ModelFiles.get_detection_models()
            ):
                names.append(fn)
    return names


def list_restoration_files() -> List[str]:
    """Return a list of available restoration model display names."""
    _import_lada()
    from lada import ModelFiles, MODEL_WEIGHTS_DIR

    ModelFiles.reset_cache()
    names = [m.name for m in ModelFiles.get_restoration_models()]
    if os.path.isdir(MODEL_WEIGHTS_DIR):
        for fn in sorted(os.listdir(MODEL_WEIGHTS_DIR)):
            if fn.endswith(".pth") and fn not in names and not any(
                fn == os.path.basename(m.path)
                for m in ModelFiles.get_restoration_models()
            ):
                names.append(fn)
    return names


def _resolve_detection_path(name: str) -> tuple[str, str]:
    _import_lada()
    from lada import ModelFiles, MODEL_WEIGHTS_DIR

    mf = ModelFiles.get_detection_model_by_name(name)
    if mf is not None and os.path.isfile(mf.path):
        return mf.name, mf.path
    candidate = os.path.join(MODEL_WEIGHTS_DIR, name)
    if os.path.isfile(candidate):
        return os.path.splitext(name)[0], candidate
    raise FileNotFoundError(
        f"Detection model '{name}' not found. Place a .pt file under {MODEL_WEIGHTS_DIR}/"
    )


def _resolve_restoration_path(name: str) -> tuple[str, str, str]:
    """Returns (display_name, file_path, backbone) where backbone is 'basicvsrpp' or 'deepmosaics'."""
    _import_lada()
    from lada import ModelFiles, MODEL_WEIGHTS_DIR

    mf = ModelFiles.get_restoration_model_by_name(name)
    if mf is not None and os.path.isfile(mf.path):
        backbone = "deepmosaics" if mf.name.startswith("deepmosaics") else "basicvsrpp"
        return mf.name, mf.path, backbone
    candidate = os.path.join(MODEL_WEIGHTS_DIR, name)
    if os.path.isfile(candidate):
        backbone = "deepmosaics" if "deepmosaics" in name.lower() or "clean_youknow" in name.lower() else "basicvsrpp"
        return os.path.splitext(name)[0], candidate, backbone
    raise FileNotFoundError(
        f"Restoration model '{name}' not found. Place a .pth file under {MODEL_WEIGHTS_DIR}/"
    )


# ---------------------------------------------------------------------------
# Model loaders -------------------------------------------------------------
# ---------------------------------------------------------------------------

def load_detection_model(name: str, device: str = "auto", fp16: str = "auto") -> LadaDetectionModel:
    _import_lada()
    from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel  # type: ignore

    display_name, path = _resolve_detection_path(name)
    torch_device = resolve_device(device)
    use_fp16 = resolve_fp16(fp16, torch_device)
    yolo_model = Yolo11SegmentationModel(path, torch_device, classes=None, conf=0.15, fp16=use_fp16)
    return LadaDetectionModel(
        model=yolo_model,
        name=display_name,
        path=path,
        device=torch_device,
        fp16=use_fp16,
    )


def load_restoration_model(name: str, device: str = "auto", fp16: str = "auto") -> LadaRestorationModel:
    _import_lada()

    display_name, path, backbone = _resolve_restoration_path(name)
    torch_device = resolve_device(device)
    use_fp16 = resolve_fp16(fp16, torch_device)

    if backbone == "basicvsrpp":
        from lada.models.basicvsrpp.inference import load_model as _load_basicvsrpp  # type: ignore
        from lada.restorationpipeline.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer  # type: ignore

        net = _load_basicvsrpp(None, path, torch_device, use_fp16)
        restorer = BasicvsrppMosaicRestorer(net, torch_device, use_fp16)
        pad_mode = "zero"
        # 'basicvsrpp-v1.2' style name is required by FrameRestorer to dispatch to the right code path.
        full_name = display_name if display_name.startswith("basicvsrpp") else f"basicvsrpp-{display_name}"
    else:
        from lada.models.deepmosaics.models import loadmodel as deepmosaics_loadmodel  # type: ignore
        from lada.restorationpipeline.deepmosaics_mosaic_restorer import DeepmosaicsMosaicRestorer  # type: ignore

        net = deepmosaics_loadmodel.video(torch_device, path, use_fp16)
        restorer = DeepmosaicsMosaicRestorer(net, torch_device)
        pad_mode = "reflect"
        full_name = display_name if display_name.startswith("deepmosaics") else f"deepmosaics-{display_name}"

    return LadaRestorationModel(
        restorer=restorer,
        backbone=backbone,
        name=full_name,
        path=path,
        device=torch_device,
        fp16=use_fp16,
        pad_mode=pad_mode,
    )


# ---------------------------------------------------------------------------
# IMAGE batch <-> temp video -----------------------------------------------
# ---------------------------------------------------------------------------

def _comfy_images_to_uint8_rgb(images: torch.Tensor) -> np.ndarray:
    if images.dim() != 4 or images.shape[-1] not in (3, 4):
        raise ValueError(f"Expected IMAGE batch [B,H,W,C], got {tuple(images.shape)}")
    arr = images.detach().to("cpu", dtype=torch.float32).clamp_(0.0, 1.0)
    arr = (arr * 255.0 + 0.5).to(torch.uint8).numpy()
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def _write_lossless_temp_video(images_uint8_rgb: np.ndarray, fps: float, path: str) -> None:
    """Encode frames into a lossless ``ffv1`` MKV via PyAV (fallback to libx264 yuv420p)."""
    import av

    h, w = images_uint8_rgb.shape[1], images_uint8_rgb.shape[2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        images_uint8_rgb = np.pad(
            images_uint8_rgb,
            ((0, 0), (0, pad_h), (0, pad_w), (0, 0)),
            mode="edge",
        )
        h, w = images_uint8_rgb.shape[1], images_uint8_rgb.shape[2]

    container = av.open(path, mode="w", format="matroska")
    try:
        rate = Fraction(fps).limit_denominator(60000)
        try:
            stream = container.add_stream("ffv1", rate=rate)
            stream.pix_fmt = "bgr0"
        except Exception:
            stream = container.add_stream("libx264", rate=rate)
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "0", "preset": "ultrafast"}
        stream.width = w
        stream.height = h

        for frame_arr in images_uint8_rgb:
            frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _bgr_uint8_to_rgb_float(frame) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8, copy=False)
    rgb = frame[..., ::-1]
    return rgb.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# The actual restore call --------------------------------------------------
# ---------------------------------------------------------------------------

def restore_image_batch(
    detection: LadaDetectionModel,
    restoration: LadaRestorationModel,
    images: torch.Tensor,
    *,
    fps: float = 25.0,
    max_clip_length: int = 180,
    progress_cb=None,
) -> torch.Tensor:
    """Run the full lada detection + restoration pipeline on a frame batch."""
    _import_lada()
    _ensure_ffmpeg_binaries()
    from lada.restorationpipeline.frame_restorer import FrameRestorer  # type: ignore
    from lada.utils.threading_utils import STOP_MARKER, ErrorMarker  # type: ignore

    if images.shape[0] == 0:
        return images.clone()

    if detection.device != restoration.device:
        raise RuntimeError(
            f"Detection model is on {detection.device} but restoration model is on "
            f"{restoration.device}; both must share the same device."
        )

    rgb = _comfy_images_to_uint8_rgb(images)
    orig_h, orig_w = images.shape[1], images.shape[2]

    tmp_dir = tempfile.mkdtemp(prefix="lada_comfy_")
    tmp_in = os.path.join(tmp_dir, "input.mkv")
    try:
        _write_lossless_temp_video(rgb, fps=float(fps), path=tmp_in)

        frame_restorer = FrameRestorer(
            detection.device,
            tmp_in,
            int(max_clip_length),
            restoration.name,
            detection.model,
            restoration.restorer,
            restoration.pad_mode,
        )

        restored: List[np.ndarray] = []
        try:
            frame_restorer.start()
            for elem in frame_restorer:
                if elem is STOP_MARKER or isinstance(elem, ErrorMarker):
                    raise RuntimeError(f"Lada FrameRestorer stopped prematurely: {elem!r}")
                restored_frame, _pts = elem
                rgb_frame = _bgr_uint8_to_rgb_float(restored_frame)
                if rgb_frame.shape[0] != orig_h or rgb_frame.shape[1] != orig_w:
                    rgb_frame = rgb_frame[:orig_h, :orig_w, :]
                restored.append(rgb_frame)
                if progress_cb is not None:
                    progress_cb(len(restored), images.shape[0])
        finally:
            frame_restorer.stop()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if len(restored) != images.shape[0]:
        logger.warning(
            "Lada returned %d frames for %d input frames; padding with originals.",
            len(restored),
            images.shape[0],
        )
        original_np = images.detach().cpu().float().clamp_(0, 1).numpy()
        for i in range(len(restored), images.shape[0]):
            restored.append(original_np[i])

    out = np.stack(restored, axis=0).astype(np.float32, copy=False)
    return torch.from_numpy(out)
