"""ComfyUI node definitions for the (vendored) Lada mosaic-removal pipeline."""

from __future__ import annotations

import logging
from typing import List

import torch

from .pipeline import (
    LadaDetectionModel,
    LadaRestorationModel,
    list_detection_files,
    list_restoration_files,
    load_detection_model,
    load_restoration_model,
    restore_image_batch,
)

logger = logging.getLogger("ComfyUI_RemoveMosaic")

try:
    from comfy.utils import ProgressBar  # type: ignore
except Exception:  # pragma: no cover
    ProgressBar = None  # type: ignore

CATEGORY = "RemoveMosaic"

DEVICE_OPTIONS = ["auto", "cuda", "cuda:0", "cuda:1", "mps", "cpu"]
FP16_OPTIONS = ["auto", "enable", "disable"]


def _safe_list(fn, fallback_label: str) -> List[str]:
    try:
        names = fn()
        return names if names else [fallback_label]
    except Exception as e:
        logger.warning("Failed to enumerate lada models: %s", e)
        return [fallback_label]


_NO_MODEL_LABEL = "<no models found in ComfyUI/models/lada>"


# ---------------------------------------------------------------------------
# 1. Load detection model ---------------------------------------------------
# ---------------------------------------------------------------------------

class LadaLoadDetectionModel:
    """Load a YOLOv11 mosaic-detection model.

    Drops a list of all ``.pt`` files (and the well-known names like
    ``v4-fast``) found in ``ComfyUI/models/lada/``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_safe_list(list_detection_files, _NO_MODEL_LABEL),),
                "device": (DEVICE_OPTIONS, {"default": "auto"}),
                "fp16": (FP16_OPTIONS, {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("LADA_DETECTION_MODEL",)
    RETURN_NAMES = ("detection_model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, model_name: str, device: str, fp16: str):
        if model_name.startswith("<"):
            raise RuntimeError(
                "No detection model found. Place a YOLOv11 `.pt` file under "
                "ComfyUI/models/lada/ (e.g. lada_mosaic_detection_model_v4_fast.pt)."
            )
        return (load_detection_model(model_name, device=device, fp16=fp16),)


# ---------------------------------------------------------------------------
# 2. Load restoration model -------------------------------------------------
# ---------------------------------------------------------------------------

class LadaLoadRestorationModel:
    """Load a mosaic-restoration model.

    Supports both BasicVSR++ GAN checkpoints (``.pth``, recommended) and the
    legacy DeepMosaics checkpoint (``clean_youknow_video.pth``). The dropdown
    lists every restoration model lada finds in ``ComfyUI/models/lada/``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_safe_list(list_restoration_files, _NO_MODEL_LABEL),),
                "device": (DEVICE_OPTIONS, {"default": "auto"}),
                "fp16": (FP16_OPTIONS, {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("LADA_RESTORATION_MODEL",)
    RETURN_NAMES = ("restoration_model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, model_name: str, device: str, fp16: str):
        if model_name.startswith("<"):
            raise RuntimeError(
                "No restoration model found. Place a `.pth` file under "
                "ComfyUI/models/lada/ (e.g. lada_mosaic_restoration_model_generic_v1.2.pth)."
            )
        return (load_restoration_model(model_name, device=device, fp16=fp16),)


# ---------------------------------------------------------------------------
# 3. Remove mosaic ----------------------------------------------------------
# ---------------------------------------------------------------------------

class LadaRemoveMosaic:
    """Run lada's full detection + restoration pipeline on a batch of frames.

    Inputs / outputs are standard ComfyUI ``IMAGE`` tensors of shape
    ``[B, H, W, 3]`` (float32 RGB in ``[0, 1]``). Internally the frames are
    encoded into a temporary lossless ``ffv1`` MKV so lada's video-oriented
    pipeline can operate on them; the temp file is removed automatically.

    Lada's restoration model is **temporal**: it expects a coherent video
    clip. A single frame or very short clip will produce noticeably worse
    results than a real sequence (>= 30 frames is a reasonable lower bound).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "detection_model": ("LADA_DETECTION_MODEL",),
                "restoration_model": ("LADA_RESTORATION_MODEL",),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 240.0, "step": 0.1}),
                "max_clip_length": ("INT", {"default": 180, "min": 1, "max": 1024, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "remove"
    CATEGORY = CATEGORY

    def remove(
        self,
        images: torch.Tensor,
        detection_model: LadaDetectionModel,
        restoration_model: LadaRestorationModel,
        fps: float = 25.0,
        max_clip_length: int = 180,
    ):
        if not isinstance(detection_model, LadaDetectionModel):
            raise TypeError("'detection_model' must come from LadaLoadDetectionModel.")
        if not isinstance(restoration_model, LadaRestorationModel):
            raise TypeError("'restoration_model' must come from LadaLoadRestorationModel.")

        if images.shape[0] < 2:
            logger.warning(
                "Lada is built for temporally consistent restoration; "
                "a single frame may yield poor results."
            )

        progress = ProgressBar(images.shape[0]) if ProgressBar is not None else None

        def _cb(done: int, total: int):
            if progress is not None:
                progress.update_absolute(done, total)

        restored = restore_image_batch(
            detection_model,
            restoration_model,
            images,
            fps=float(fps),
            max_clip_length=int(max_clip_length),
            progress_cb=_cb,
        )
        return (restored,)


# ---------------------------------------------------------------------------
# Registration --------------------------------------------------------------
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "LadaLoadDetectionModel": LadaLoadDetectionModel,
    "LadaLoadRestorationModel": LadaLoadRestorationModel,
    "LadaRemoveMosaic": LadaRemoveMosaic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LadaLoadDetectionModel": "Load Mosaic Detection Model (Lada)",
    "LadaLoadRestorationModel": "Load Mosaic Restoration Model (Lada)",
    "LadaRemoveMosaic": "Remove Mosaic (Lada)",
}
