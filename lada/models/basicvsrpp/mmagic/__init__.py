# SPDX-FileCopyrightText: OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND AGPL-3.0
# Code vendored from: https://github.com/open-mmlab/mmagic
#
# Patched: the bulk `register_all_modules` was moved up one level
# (lada.models.basicvsrpp.__init__) where we register only what's strictly
# needed for inference. This file keeps the SCOPE constant only.

SCOPE = 'lada.models.basicvsrpp.mmagic'
