"""The six-level Gaussian blur schedule used during paper training."""

from __future__ import annotations

import albumentations as A

BLUR_KERNELS = (3, 7, 11, 15, 19, 23)


def six_level_blur() -> A.BasicTransform:
    """Uniformly sample one kernel for every image (Albumentations 1.3.0).

    Place this transform before image resizing and normalization in the training
    Compose. Albumentations applies it to the RGB image and leaves masks sharp.
    OpenCV chooses sigma from the sampled kernel size, as in the experiment.
    """
    return A.OneOf(
        [A.GaussianBlur(blur_limit=(kernel, kernel), p=1.0) for kernel in BLUR_KERNELS],
        p=1.0,
    )
