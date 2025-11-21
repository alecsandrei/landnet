from __future__ import annotations

import collections.abc as c
import typing as t
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2 import (
    Compose,
    InterpolationMode,
    Lambda,
    ToDtype,
    ToImage,
    functional,
)

from landnet.enums import Mode
from landnet.logger import create_logger

if t.TYPE_CHECKING:
    from landnet.features.grids import Grid

logger = create_logger(__name__)


@dataclass
class ResizeTensor:
    size: list[int]
    interpolation: InterpolationMode = InterpolationMode.BILINEAR

    def __call__(self, image: Tensor) -> Tensor:
        return functional.resize(
            image, self.size, interpolation=self.interpolation
        )


class RandomRotateTensor:
    def __init__(self, angles: c.Sequence[int]):
        self.angles = angles

    def __call__(self, images: Tensor | list[Tensor]) -> list[Tensor]:
        if isinstance(images, Tensor):
            images = [images]
        choice = np.random.choice(self.angles)
        if choice != 0:
            images = [
                functional.rotate(img, int(choice), expand=False)
                for img in images
            ]
        return images


class ConsistentVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images: Tensor | list[Tensor]) -> list[Tensor]:
        if isinstance(images, Tensor):
            images = [images]
        do_flip = np.random.random() < self.p
        if do_flip:
            images = [functional.vflip(img) for img in images]
        return images


class ConsistentHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images: Tensor | list[Tensor]) -> list[Tensor]:
        if isinstance(images, Tensor):
            images = [images]
        do_flip = np.random.random() < self.p
        if do_flip:
            images = [functional.hflip(img) for img in images]
        return images


def get_default_transform(size: int = 224):
    return Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            ResizeTensor([size, size]),
            Lambda(lambda x: (x - x.mean()) / x.std()),
        ]
    )


def get_default_mask_transform(size: int = 224):
    return Compose(
        [
            ToImage(),
            ResizeTensor([size, size], interpolation=InterpolationMode.NEAREST),
        ]
    )


def get_default_augment_transform():
    return Compose(
        [
            ConsistentHorizontalFlip(p=0.5),
            ConsistentVerticalFlip(p=0.5),
            RandomRotateTensor([0, 90, 180, 270]),
        ]
    )


@dataclass
class LandslideImages(Dataset):
    grid: Grid
    mode: Mode
    transform: c.Callable[..., torch.Tensor] | None = get_default_transform()
    augment_transform: c.Callable[..., list[torch.Tensor]] | None = None
    transforms: c.Callable[..., torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if self.augment_transform is not None:
            logger.debug(
                'augment_transform parameter was provided for %r'
                % self.__class__.__name__
            )

    def __len__(self) -> int:
        return self.grid.get_tiles_length()
