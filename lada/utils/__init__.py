# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
#
# Patched for vendoring: replaced Python 3.12 ``type X = ...`` aliases with
# plain assignments so the module imports under Python >= 3.10.

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple

import numpy as np
import torch

# A bounding box (top-Y, left-X, bottom-Y, right-X).
Box = Tuple[int, int, int, int]

# Segmentation mask, (H, W, 1) uint8 with values 0 / mask_value.
Mask = np.ndarray
MaskTensor = torch.Tensor

# Color image, (H, W, 3) uint8 in BGR order.
Image = np.ndarray
ImageTensor = torch.Tensor

# Padding of an image: (top, bottom, left, right).
Pad = Tuple[int, int, int, int]


@dataclass
class VideoMetadata:
    video_file: str
    video_height: int
    video_width: int
    video_fps: float
    average_fps: float
    video_fps_exact: Fraction
    codec_name: str
    frames_count: int
    duration: float
    time_base: Fraction
    start_pts: int


@dataclass
class Detection:
    cls: int
    box: Box
    mask: Mask
    confidence: float | None = None


@dataclass
class Detections:
    frame: Image
    detections: list[Detection]


DETECTION_CLASSES = {
    "nsfw": dict(cls=0, mask_value=255),
    "sfw_head": dict(cls=1, mask_value=127),
    "sfw_face": dict(cls=2, mask_value=192),
    "watermark": dict(cls=3, mask_value=60),
    "mosaic": dict(cls=4, mask_value=90),
}
