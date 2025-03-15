import torch
import torch.nn as nn
from typing import Any, Dict, List, Sequence, Union

IMAGE_NAME_PATTERN = "{}/frame_{:07d}.png"
CLIP_NAME_PATTERN = "{}.mp4"
BBOX_NAME_PATTERN = "{}/{:07d}.npz"
DATASET_FILE_TEMPLATE = "{}_annot.json.gz"


def get_clip_name_from_clip_uid(clip_uid: str) -> str:
    return CLIP_NAME_PATTERN.format(clip_uid)


def extract_window_with_context(
    image: torch.Tensor,
    bbox: Sequence[Union[int, float]],
    p: int = 16,
    size: int = 256,
    pad_value: int = 0,
) -> torch.Tensor:
    """
    Extracts region from a bounding box in the image with some context padding.

    Arguments:
        image - (1, c, h, w) Tensor
        bbox - bounding box specifying (x1, y1, x2, y2)
        p - number of pixels of context to include around window
        size - final size of the window
        pad_value - value of pixels padded
    """
    H, W = image.shape[2:]
    bbox = tuple([int(x) for x in bbox])
    x1, y1, x2, y2 = bbox
    x1 = max(x1 - p, 0)
    y1 = max(y1 - p, 0)
    x2 = min(x2 + p, W)
    y2 = min(y2 + p, H)
    window = image[:, :, y1:y2, x1:x2]
    H, W = window.shape[2:]
    # Zero pad and resize
    left_pad = 0
    right_pad = 0
    top_pad = 0
    bot_pad = 0
    if H > W:
        left_pad = (H - W) // 2
        right_pad = (H - W) - left_pad
    elif H < W:
        top_pad = (W - H) // 2
        bot_pad = (W - H) - top_pad
    if H != W:
        window = nn.functional.pad(
            window, (left_pad, right_pad, top_pad, bot_pad), value=pad_value
        )
    window = nn.functional.interpolate(
        window, size=size, mode="bilinear", align_corners=False
    )

    return window
