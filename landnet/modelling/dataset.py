from __future__ import annotations

import collections.abc as c
import typing as t
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.v2 import (
    Compose,
    InterpolationMode,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ToTensor,
    functional,
)

from landnet.config import (
    GRIDS,
)
from landnet.enums import Mode
from landnet.logger import create_logger

if t.TYPE_CHECKING:
    from torch import Tensor

    from landnet.features.grids import Grid

logger = create_logger(__name__)


@dataclass
class ResizeTensor:
    size: list[int]
    interpolation: InterpolationMode = InterpolationMode.BILINEAR

    def __call__(self, img: Tensor) -> Tensor:
        return functional.resize(
            img, self.size, interpolation=self.interpolation
        )


class RotateTensor:
    def __init__(self, angles: c.Sequence[int]):
        self.angles = angles

    def __call__(self, *imgs: Tensor) -> list[Tensor]:
        choice = np.random.choice(self.angles)
        return [functional.rotate(img, int(choice)) for img in imgs]


def get_default_transform(size: int = 224):
    return Compose(
        [
            ToTensor(),
            ResizeTensor([size, size]),
            Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
            Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def get_default_mask_transform(size: int = 224):
    return Compose(
        [
            ToTensor(),
            ResizeTensor([size, size], interpolation=InterpolationMode.NEAREST),
        ]
    )


def get_default_augment_transform():
    return Compose(
        [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RotateTensor([0, 90, 180, 270]),
        ]
    )


@dataclass
class LandslideImages(Dataset):
    grid: Grid
    mode: Mode
    transform: c.Callable | None = get_default_transform()
    augment_transform: c.Callable | None = None
    transforms: c.Callable | None = None

    def __post_init__(self) -> None:
        self.overlap = 0
        self.mode = (
            Mode.TRAIN
            if self.grid.path.is_relative_to(GRIDS / 'train')
            else Mode.TEST
        )
        if self.augment_transform is not None:
            logger.debug(
                'augment_transform parameter was provided for %r'
                % self.__class__.__name__
            )

    def __len__(self) -> int:
        return self.grid.get_tiles_length()
