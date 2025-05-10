from __future__ import annotations

import typing as t
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from landnet.enums import LandslideClass, Mode
from landnet.logger import create_logger
from landnet.modelling.dataset import (
    LandslideImages,
    get_default_augment_transform,
    get_default_mask_transform,
)

if t.TYPE_CHECKING:
    import collections.abc as c

DEFAULT_CLASS_BALANCE = {
    LandslideClass.NO_LANDSLIDE: 0.3,
    LandslideClass.LANDSLIDE: 0.7,
}

logger = create_logger(__name__)


@dataclass
class ConcatLandslideImageSegmentation(Dataset):
    landslide_images: c.Sequence[LandslideImageSegmentation]
    augment_transform: c.Callable | None = get_default_augment_transform()

    def __len__(self):
        return len(self.landslide_images[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if any(
            landslide_images.augment_transform is not None
            for landslide_images in self.landslide_images
        ):
            logger.error(
                'Found "augment_transform" for landslide_images. This might result in unexpected behaviour'
            )
        image_batch = [images[index] for images in self.landslide_images]
        images = [images[0] for images in image_batch]
        mask = image_batch[0][1]
        if self.augment_transform is not None:
            mask, *images = self.augment_transform(mask, *images)
        cat = torch.cat(images, dim=0)
        return (cat, mask)


@dataclass
class LandslideImageSegmentation(LandslideImages):
    mask_transform: c.Callable | None = get_default_mask_transform()

    def _get_tile_mask(self, index: int) -> torch.Tensor:
        mask = self.grid.get_tile_mask(index, self.mode)[1]
        return torch.from_numpy(mask)

    def _get_tile(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        _, image, _ = self.grid.get_tile(index)
        mask = self._get_tile_mask(index)
        image = image.squeeze(0)
        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        assert isinstance(image, torch.Tensor)
        return image, mask

    def _get_item_train(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        tile, mask = self._get_tile(index)
        if self.augment_transform is not None:
            tile, mask = self.augment_transform(tile, mask)
        return (tile, mask)

    def _get_item_test(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._get_tile(index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode is Mode.TRAIN:
            return self._get_item_train(index)
        elif self.mode is Mode.TEST:
            return self._get_item_test(index)
        raise ValueError('Mode should only be "train" or "test"')


def get_segmentation_dataloader(dataset: Dataset, size: int, **kwargs):
    sampler = RandomSampler(
        data_source=dataset,  # type: ignore
        num_samples=size,
        replacement=True,
    )
    dataloader = DataLoader(dataset, **kwargs, sampler=sampler)
    return dataloader
