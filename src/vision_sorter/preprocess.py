"""Small, explicit image preprocessing functions."""

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

IMAGE_SIZE = (128, 128)

transform = Compose(
    [
        Resize(IMAGE_SIZE),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)


def load_image(path: str | Path) -> torch.Tensor:
    """Load one RGB image and return a normalized CHW tensor."""

    with Image.open(path) as image:
        return transform(image.convert("RGB"))
