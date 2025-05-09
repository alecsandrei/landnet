from __future__ import annotations

import collections.abc as c
import time
import typing as t
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from landnet.config import LANDSLIDE_DENSITY_THRESHOLD
from landnet.enums import LandslideClass, Mode
from landnet.logger import create_logger
from landnet.modelling.classification.models import apply_pca_on_channels
from landnet.modelling.dataset import LandslideImages

logger = create_logger(__name__)


class PCAConcatLandslideImageClassification(Dataset):
    def __init__(
        self,
        landslide_images: c.Sequence[LandslideImageClassification],
        num_components: int,
    ):
        self.landslide_images = landslide_images
        self.concat = ConcatLandslideImageClassification(self.landslide_images)
        self.data_indices = self.concat.data_indices  # type: ignore
        self.num_components = num_components

    def __len__(self):
        return len(self.landslide_images[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Return the PCA-reduced image.
        """

        tile, class_ = self.concat[index]
        return (apply_pca_on_channels(tile, self.num_components), class_)


class ConcatLandslideImageClassification(Dataset):
    def __init__(
        self,
        landslide_images: c.Sequence[LandslideImageClassification],
    ):
        self.landslide_images = landslide_images
        self.data_indices = landslide_images[0].data_indices

    def __len__(self):
        return len(self.landslide_images[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_batch = [images[index] for images in self.landslide_images]
        cat = torch.cat([batch[0] for batch in image_batch], dim=0)
        class_ = image_batch[0][1]
        assert all(batch[1] == class_ for batch in image_batch[1:])
        return (cat, class_)


@dataclass
class LandslideImageClassification(LandslideImages):
    landslide_density_threshold: float = LANDSLIDE_DENSITY_THRESHOLD
    data_indices: dict[int, list[int]] = field(init=False)

    def __post_init__(self):
        self.data_indices = self._get_data_indices()

    def _get_data_indices(self) -> dict[int, list[int]]:
        start = time.perf_counter()
        class_to_indices: dict[int, list[int]] = {}
        indices_range = range(self.grid.get_tiles_length())
        gdf = self.grid.get_landslide_percentage_intersection(
            list(indices_range), self.mode
        )
        for row in gdf.itertuples():
            class_ = (
                LandslideClass.LANDSLIDE.value
                if t.cast(float, row.landslide_density)
                >= self.landslide_density_threshold
                else LandslideClass.NO_LANDSLIDE.value
            )
            assert isinstance(row.Index, t.SupportsInt)
            class_to_indices.setdefault(class_, []).append(int(row.Index))
        end = time.perf_counter()
        logger.info(
            'Took %f seconds to compute data indices for %r at mode=%r. Length of classes: %r'
            % (
                end - start,
                self.grid.tile_config.size,
                self.mode.value,
                {k: len(v) for k, v in class_to_indices.items()},
            )
        )
        return class_to_indices

    def _get_tile_class(self, index: int) -> int:
        for class_, indices in self.data_indices.items():
            if index in indices:
                return class_
        raise ValueError

    def _get_tile(self, index: int) -> tuple[torch.Tensor, int]:
        _, arr, _ = self.grid.get_tile(index)
        tile_class = self._get_tile_class(index)
        tile = arr.squeeze(0)
        if self.transform is not None:
            tile = self.transform(arr)
        assert isinstance(tile, torch.Tensor)
        return tile, tile_class

    def _get_item_train(self, index: int) -> tuple[torch.Tensor, int]:
        tile, label = self._get_tile(index)
        if self.augment_transform is not None:
            tile = self.augment_transform(tile)
        return (tile, label)

    def _get_item_test(self, index: int) -> tuple[torch.Tensor, int]:
        return self._get_tile(index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self.mode is Mode.TRAIN:
            return self._get_item_train(index)
        elif self.mode is Mode.TEST:
            return self._get_item_test(index)
        raise ValueError('Mode should only be "train" or "test"')


def get_classification_dataloader(
    dataset: Dataset, weights: np.ndarray, size: int | None = None, **kwargs
):
    samples_weight = torch.tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        samples_weight,  # type: ignore
        num_samples=size or len(samples_weight),
        replacement=True,
    )
    data_loader = DataLoader(dataset, **kwargs, sampler=sampler)
    return data_loader


def create_classification_dataloader_from_subset(
    dataset: Subset[
        LandslideImageClassification | ConcatLandslideImageClassification
    ],
    class_balance: dict[LandslideClass, float] | None = None,
    size: int | None = None,
    **kwargs,
):
    # Count occurrences of each class
    class_sample_count = {
        cls: len([index for index in indices if index in dataset.indices])
        for cls, indices in dataset.dataset.data_indices.items()  # type: ignore
    }

    # Compute inverse frequency weights
    weight = {
        cls: 1.0 / count if count > 0 else 0
        for cls, count in class_sample_count.items()
    }

    # Assign sample weights based on their class
    samples_weight = np.zeros(len(dataset.dataset))  # type: ignore

    for cls, indices in dataset.dataset.data_indices.items():  # type: ignore
        indices = [index for index in indices if index in dataset.indices]
        class_weight = weight[cls]
        if class_balance is not None:
            class_weight *= class_balance[LandslideClass(cls)]
        samples_weight[indices] = [class_weight] * len(indices)

    return get_classification_dataloader(
        dataset.dataset, weights=samples_weight, size=size, **kwargs
    )


def create_classification_dataloader(
    dataset: LandslideImageClassification
    | ConcatLandslideImageClassification
    | Subset[LandslideImageClassification | ConcatLandslideImageClassification],
    class_balance: dict[LandslideClass, float] | None = None,
    size: int | None = None,
    **kwargs,
) -> DataLoader:
    if isinstance(dataset, Subset):
        return create_classification_dataloader_from_subset(
            dataset, class_balance=class_balance, size=size, **kwargs
        )
    # Count occurrences of each class
    class_sample_count = {
        cls: len(indices) for cls, indices in dataset.data_indices.items()
    }

    # Compute inverse frequency weights
    weight = {
        cls: 1.0 / count if count > 0 else 0
        for cls, count in class_sample_count.items()
    }

    # Assign sample weights based on their class
    samples_weight = np.zeros(len(dataset))

    for cls, indices in dataset.data_indices.items():
        if isinstance(dataset, Subset):
            indices = [index for index in indices if index in dataset.indices]
        class_weight = weight[cls]
        if class_balance is not None:
            class_weight *= class_balance[LandslideClass(cls)]
        samples_weight[indices] = [class_weight] * len(indices)
    return get_classification_dataloader(
        dataset, weights=samples_weight, size=size, **kwargs
    )
