"""Vendored subset of https://github.com/ladaapp/lada (AGPL-3.0).

This is a stripped-down copy used by the ComfyUI_RemoveMosaic plugin to run
mosaic detection + restoration without requiring the full lada package
(which also ships GUI, CLI, training and dataset-creation code).

Only the inference pipeline, models (BasicVSR++ / DeepMosaics / YOLOv11) and
the supporting utils are vendored. See the original repository for the full
project.
"""

from __future__ import annotations

import builtins
import gettext as _gettext
import os
from dataclasses import dataclass
from functools import cache

# ---------------------------------------------------------------------------
# Translation shim ----------------------------------------------------------
# ---------------------------------------------------------------------------
# The upstream lada package installs gettext's `_` builtin during import.
# The vendored code uses `_("...")` in many places. We don't ship .mo files,
# so we just register an identity translation if `_` is not defined yet.
if not hasattr(builtins, "_"):
    builtins._ = _gettext.gettext  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Constants -----------------------------------------------------------------
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING")

# Default location for model weights. The ComfyUI plugin sets this to
# ComfyUI/models/lada via env var before importing this module.
MODEL_WEIGHTS_DIR = os.environ.get("LADA_MODEL_WEIGHTS_DIR", "model_weights")

VERSION = "0.11.0-vendored"
IS_FLATPAK = False

# Silence noisy 3rd-party libs.
os.environ.setdefault("ALBUMENTATIONS_OFFLINE", "1")
os.environ.setdefault("ALBUMENTATIONS_NO_TELEMETRY", "1")
os.environ.setdefault("YOLO_VERBOSE", "false")


# ---------------------------------------------------------------------------
# Model registry ------------------------------------------------------------
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelFile:
    name: str
    description: str | None
    path: str


class ModelFiles:
    """File-system based registry for lada model weights.

    Detection models are ``.pt`` (YOLOv11). Restoration models are ``.pth``
    (BasicVSR++ GAN, or the legacy DeepMosaics ``clean_youknow_video.pth``).
    The registry resolves model *names* to absolute file paths so the upstream
    pipeline code keeps working unchanged.
    """

    _WELL_KNOWN_RESTORATION_MODELS = [
        ModelFile('basicvsrpp-v1.0', None, os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_restoration_model_generic.pth')),
        ModelFile('basicvsrpp-v1.1', None, os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_restoration_model_generic_v1.1.pth')),
        ModelFile('basicvsrpp-v1.2', "Latest Lada restoration model. Recommended", os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_restoration_model_generic_v1.2.pth')),
        ModelFile('deepmosaics', "Restoration model from abandoned DeepMosaics project", os.path.join(MODEL_WEIGHTS_DIR, '3rd_party', 'clean_youknow_video.pth')),
    ]
    _WELL_KNOWN_DETECTION_MODELS = [
        ModelFile('v2', None, os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v2.pt')),
        ModelFile('v3', None, os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v3.pt')),
        ModelFile('v3.1-fast', None, os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v3.1_fast.pt')),
        ModelFile('v3.1-accurate', None, os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v3.1_accurate.pt')),
        ModelFile('v4-fast', "Fast and efficient. Recommended", os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v4_fast.pt')),
        ModelFile('v4-accurate', "Slightly more accurate than v4-fast but slower", os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v4_accurate.pt')),
    ]

    @staticmethod
    def _scan_custom(suffix: str, prefix: str, well_known: list[ModelFile]) -> list[ModelFile]:
        models: list[ModelFile] = []
        if not os.path.exists(MODEL_WEIGHTS_DIR):
            return models
        well_known_filenames = {os.path.basename(m.path) for m in well_known}
        for filename in os.listdir(MODEL_WEIGHTS_DIR):
            if not filename.endswith(suffix) or not filename.startswith(prefix):
                continue
            if filename in well_known_filenames:
                continue
            model_name = os.path.splitext(filename)[0].split(prefix)[1]
            if not model_name:
                continue
            if suffix == ".pth":
                if not model_name.startswith("deepmosaics") and "deepmosaics" in model_name:
                    model_name = f"deepmosaics-{model_name}"
                elif not model_name.startswith("basicvsrpp"):
                    model_name = f"basicvsrpp-{model_name}"
            models.append(ModelFile(model_name, None, os.path.join(MODEL_WEIGHTS_DIR, filename)))
        return models

    @staticmethod
    def _existing(models: list[ModelFile]) -> list[ModelFile]:
        return [m for m in models if os.path.exists(m.path)]

    @staticmethod
    @cache
    def get_detection_models() -> list[ModelFile]:
        return (ModelFiles._existing(ModelFiles._WELL_KNOWN_DETECTION_MODELS)
                + ModelFiles._scan_custom(".pt", "lada_mosaic_detection_model_", ModelFiles._WELL_KNOWN_DETECTION_MODELS))

    @staticmethod
    @cache
    def get_restoration_models() -> list[ModelFile]:
        return (ModelFiles._existing(ModelFiles._WELL_KNOWN_RESTORATION_MODELS)
                + ModelFiles._scan_custom(".pth", "lada_mosaic_restoration_model_", ModelFiles._WELL_KNOWN_RESTORATION_MODELS))

    @staticmethod
    def get_restoration_model_by_name(model_name: str) -> ModelFile | None:
        for m in ModelFiles.get_restoration_models():
            if m.name == model_name:
                return m
        return None

    @staticmethod
    def get_detection_model_by_name(model_name: str) -> ModelFile | None:
        for m in ModelFiles.get_detection_models():
            if m.name == model_name:
                return m
        return None

    @staticmethod
    def get_detection_model_by_path(model_path: str) -> ModelFile | None:
        for m in ModelFiles.get_detection_models():
            if m.path == model_path:
                return m
        return None

    @staticmethod
    def reset_cache():
        ModelFiles.get_detection_models.cache_clear()
        ModelFiles.get_restoration_models.cache_clear()
