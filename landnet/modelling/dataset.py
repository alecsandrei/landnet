from __future__ import annotations

import collections.abc as c
import typing as t
from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ToTensor,
    functional,
)

from landnet.config import (
    GRIDS,
    LANDSLIDE_DENSITY_THRESHOLD,
)
from landnet.enums import Mode

if t.TYPE_CHECKING:
    from torch import Tensor

    from landnet.features.grids import Grid


class ResizeTensor:
    def __init__(self, size):
        self.size = size

    def __call__(self, img: Tensor) -> Tensor:
        return functional.resize(img, self.size)


class RotateTensor:
    def __init__(self, angles: c.Sequence[int]):
        self.angles = angles

    def __call__(self, img: Tensor) -> Tensor:
        choice = np.random.choice(self.angles)
        return functional.rotate(img, int(choice))


def get_default_transform():
    return Compose(
        [
            ToTensor(),
            ResizeTensor((224, 224)),
            Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
            Normalize(mean=0.5, std=0.5),
        ]
    )


def get_default_augument_transform():
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
    landslide_density_threshold: float = LANDSLIDE_DENSITY_THRESHOLD
    transforms: c.Callable | None = None
    transform: c.Callable | None = None

    def __post_init__(
        self,
    ) -> None:
        self.overlap = 0
        self.transform = (
            self.transform
            if self.transform is not None
            else get_default_transform()
        )
        self.augument_transform = get_default_augument_transform()
        self.mode = (
            Mode.TRAIN
            if self.grid.path.is_relative_to(GRIDS / 'train')
            else Mode.TEST
        )

    def __len__(self) -> int:
        return self.grid.get_tiles_length()
