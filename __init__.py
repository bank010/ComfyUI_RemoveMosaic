"""ComfyUI_RemoveMosaic
ComfyUI nodes that wrap the (vendored) Lada mosaic-removal pipeline.

The whole `lada/` subpackage in this directory is a stripped-down vendor
copy of https://github.com/ladaapp/lada (AGPL-3.0). Users do NOT need to
`pip install lada`; the ComfyUI Python environment only has to provide the
common scientific deps (torch, torchvision, mmengine, ultralytics, av, ...).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import types

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


def _manual_load_nodes():
    """Load nodes/pipeline without relying on relative imports.

    Some ComfyUI variants (notably the AutoDL build and a few forks) load
    custom nodes via ``importlib.util.spec_from_file_location`` using the
    *absolute filesystem path* as the module name. That breaks ``from .nodes
    import ...`` because the relative import resolves to e.g.
    ``"/root/.../ComfyUI_RemoveMosaic.nodes"``. We work around it by
    registering a clean synthetic package and exec-loading both submodules
    under stable dotted names.
    """
    pkg_name = "comfyui_removemosaic_pkg"

    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [_THIS_DIR]
        sys.modules[pkg_name] = pkg

    def _load(submod, filename):
        full_name = f"{pkg_name}.{submod}"
        if full_name in sys.modules:
            return sys.modules[full_name]
        spec = importlib.util.spec_from_file_location(
            full_name, os.path.join(_THIS_DIR, filename)
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("pipeline", "pipeline.py")
    return _load("nodes", "nodes.py")


try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as primary_err:
    try:
        _nodes_mod = _manual_load_nodes()
        NODE_CLASS_MAPPINGS = _nodes_mod.NODE_CLASS_MAPPINGS
        NODE_DISPLAY_NAME_MAPPINGS = _nodes_mod.NODE_DISPLAY_NAME_MAPPINGS
        logger.info(
            "ComfyUI_RemoveMosaic loaded via manual fallback (host ComfyUI uses "
            "path-as-module-name; original error: %s)",
            primary_err,
        )
    except Exception as fallback_err:  # pragma: no cover
        logger.exception(
            "ComfyUI_RemoveMosaic failed to load. Primary error: %s | Fallback error: %s",
            primary_err,
            fallback_err,
        )
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
