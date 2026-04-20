"""ComfyUI_RemoveMosaic
ComfyUI nodes that wrap the (vendored) Lada mosaic-removal pipeline.

The whole `lada/` subpackage in this directory is a stripped-down vendor
copy of https://github.com/ladaapp/lada (AGPL-3.0). Users do NOT need to
`pip install lada`; the ComfyUI Python environment only has to provide the
common scientific deps (torch, torchvision, mmengine, ultralytics, av, ...).
"""

from __future__ import annotations

import logging
import os
import sys

# ---------------------------------------------------------------------------
# Locate ComfyUI/models/lada and tell the vendored lada to look there.
# ---------------------------------------------------------------------------
try:
    import folder_paths  # type: ignore  # provided by ComfyUI runtime

    _lada_models_dir = os.path.join(folder_paths.models_dir, "lada")
    os.makedirs(_lada_models_dir, exist_ok=True)
    os.environ["LADA_MODEL_WEIGHTS_DIR"] = _lada_models_dir
    folder_paths.add_model_folder_path("lada", _lada_models_dir)
except Exception:  # pragma: no cover - allow import outside ComfyUI for tests
    pass

# Make the vendored `lada` package importable as a top-level module.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

os.environ.setdefault("LOG_LEVEL", "WARNING")

logger = logging.getLogger("ComfyUI_RemoveMosaic")

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as e:  # pragma: no cover
    logger.exception("ComfyUI_RemoveMosaic failed to load: %s", e)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
