from __future__ import annotations

import collections.abc as c
import typing as t
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    WeightedRandomSampler,
)

from landnet.config import LANDSLIDE_DENSITY_THRESHOLD
from landnet.enums import LandslideClass, Mode
from landnet.logger import create_logger
from landnet.modelling.dataset import (
    LandslideImages,
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
    augment_transform: c.Callable | None = None

    def __len__(self):
        return len(self.landslide_images[0])

    def __getitem__(self, key) -> tuple[torch.Tensor, torch.Tensor]:
        if any(
            landslide_images.augment_transform is not None
            for landslide_images in self.landslide_images
        ):
            logger.error(
                'Found "augment_transform" for landslide_images. This might result in unexpected behaviour'
            )
        image_batch = [images[key] for images in self.landslide_images]
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
        mask = self.grid.get_tile_mask(index)[1]
        return torch.from_numpy(mask)

    def _get_tile(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        _, image, _ = self.grid.get_tile(index)
        mask = self._get_tile_mask(index)
        image = image.squeeze(0)
        if self.transform is not None:
            t_image = self.transform(image)
        else:
            t_image = torch.from_numpy(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return t_image, mask

    def _get_item_train(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        tile, mask = self._get_tile(index)
        if self.augment_transform is not None:
            tile, mask = self.augment_transform(tile, mask)
        return (tile, mask)

    def _get_item_test(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._get_tile(index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index >= len(self):
            raise StopIteration
        if self.mode is Mode.TRAIN:
            return self._get_item_train(index)
        elif self.mode in (Mode.TEST, Mode.VALIDATION, Mode.INFERENCE):
            return self._get_item_test(index)
        raise ValueError('Mode %r is not supported' % self.mode)


def get_weights(dataset: ConcatLandslideImageSegmentation) -> np.ndarray:
    samples_weight = np.zeros(len(dataset))
    for i in range(len(dataset)):
        image, mask = dataset[i]
        if (
            mask[1].count_nonzero() / mask[1].numel()
            < LANDSLIDE_DENSITY_THRESHOLD
        ):
            samples_weight[i] = 0.0
        else:
            samples_weight[i] = np.nan
        i += 1
    nan_weights = np.isnan(samples_weight)
    zero_count = (samples_weight == 0).sum()
    samples_weight[nan_weights] = 1 / (len(dataset) - zero_count)

    np.testing.assert_almost_equal(samples_weight.sum(), 1, decimal=5)
    return samples_weight


def get_weighted_segmentation_dataloader(
    dataset: ConcatLandslideImageSegmentation, size: int, **kwargs
) -> DataLoader:
    sampler = WeightedRandomSampler(
        get_weights(dataset).tolist(), num_samples=size, replacement=True
    )
    dataloader = DataLoader(dataset, **kwargs, sampler=sampler)
    return dataloader


def get_segmentation_dataloader(dataset: Dataset, size: int, **kwargs):
    sampler = RandomSampler(
        data_source=dataset,  # type: ignore
        num_samples=size,
        replacement=True,
    )
    dataloader = DataLoader(dataset, **kwargs, sampler=sampler)
    return dataloader
