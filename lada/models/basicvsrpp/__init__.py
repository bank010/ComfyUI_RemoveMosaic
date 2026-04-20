# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
#
# Patched for vendoring inside ComfyUI_RemoveMosaic.
#
# Only modules required for INFERENCE of BasicVSR++ / BasicVSR++-GAN are
# registered with the MODELS registry. Training-only modules (GAN losses,
# perceptual loss, EMA hook, optimizer constructors, evaluators, visualizers)
# are skipped because they pull in extra dependencies (e.g. termcolor,
# tensorboard) that aren't useful inside a ComfyUI workflow.

from __future__ import annotations

import logging

logger = logging.getLogger("ComfyUI_RemoveMosaic")

_REGISTERED = False


def register_all_modules() -> None:
    """Register the minimal set of MMagic modules required for inference."""
    global _REGISTERED
    if _REGISTERED:
        return

    from mmengine import DefaultScope

    from lada.models.basicvsrpp.mmagic import SCOPE
    # Importing these for their `@MODELS.register_module()` side-effects.
    from lada.models.basicvsrpp.mmagic.base_edit_model import BaseEditModel  # noqa: F401
    from lada.models.basicvsrpp.mmagic.data_preprocessor import DataPreprocessor  # noqa: F401
    from lada.models.basicvsrpp.mmagic.pixelwise_loss import CharbonnierLoss  # noqa: F401
    from lada.models.basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet  # noqa: F401
    from lada.models.basicvsrpp.mmagic.basicvsr import BasicVSR  # noqa: F401
    from lada.models.basicvsrpp.mmagic.real_basicvsr import RealBasicVSR  # noqa: F401
    from lada.models.basicvsrpp.basicvsrpp_gan import (  # noqa: F401
        BasicVSRPlusPlusGanNet,
        BasicVSRPlusPlusGan,
    )

    if DefaultScope.get_current_instance() is None or not DefaultScope.check_instance_created(SCOPE):
        DefaultScope.get_instance(SCOPE, scope_name=SCOPE)

    _REGISTERED = True
