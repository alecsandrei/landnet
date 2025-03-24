from __future__ import annotations

import collections.abc as c
import typing as t
from functools import cache

import lightning.pytorch as pl
import ray
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, random_split

from landnet.enums import GeomorphometricalVariable, Mode
from landnet.features.tiles import (
    LandslideImages,
    TileSize,
    get_landslide_images_for_variable,
)
from landnet.modelling.stats import BinaryClassificationMetricCollection


@cache
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    print('Warning: CUDA is not available.')
    return torch.get_default_device()


class ConfigSpec(t.TypedDict):
    batch_size: int
    learning_rate: float
    tile_size: TileSize


class LandslideImageClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module, config: ConfigSpec):
        super().__init__()
        self.config = config
        self.model = model
        self.criterion = nn.BCELoss()
        self.train_metrics = BinaryClassificationMetricCollection(
            prefix='train_'
        )

        self.val_metrics = self.train_metrics.clone(prefix='val_')
        self.test_metrics = self.train_metrics.clone(prefix='test_')

    def forward(self, x):
        return self.model(x).flatten()

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        self.train_metrics.update(logits, y)
        return self.criterion(logits, y.float())

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), sync_dist=True)
        self.train_metrics.reset()

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        self.val_metrics.update(logits, y)
        loss = self.criterion(logits, y.float())
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.val_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config['learning_rate']
        )
        return optimizer


class LandslideImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: ConfigSpec,
        variables: c.Sequence[GeomorphometricalVariable],
        cacher: ray.ObjectRef[LandslideImagesCacher],  # type: ignore
    ):
        super().__init__()
        self.cacher = cacher
        self.config = config
        self.variables = variables

    def setup(self, stage=None):
        self.train_dataset = ConcatDataset(
            ray.get(
                [
                    self.cacher.setdefault.remote(  # type: ignore
                        variable, self.config['tile_size'], Mode.TRAIN
                    )
                    for variable in self.variables
                ]
            )
        )
        self.train_dataset, self.validation_dataset = random_split(
            self.train_dataset, (0.7, 0.3)
        )
        self.test_dataset = ConcatDataset(
            ray.get(
                [
                    self.cacher.setdefault.remote(  # type: ignore
                        variable, self.config['tile_size'], Mode.TEST
                    )
                    for variable in self.variables
                ]
            )
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            num_workers=4,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.config['batch_size'],
            num_workers=4,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            num_workers=4,
            shuffle=False,
            pin_memory=True,
        )


@ray.remote
class LandslideImagesCacher:
    """Actor used to cache the LandslideImages.

    Because of the _get_data_indices() method, LandslideImages takes some time
    to load."""

    def __init__(self):
        self.map: dict[  # type: ignore
            TileSize, dict[GeomorphometricalVariable, LandslideImages]
        ] = {}

    def get(
        self, tile_size: TileSize, variable: GeomorphometricalVariable
    ) -> LandslideImages | None:
        try:
            return self.map[tile_size][variable]
        except KeyError:
            return None

    def setdefault(
        self,
        variable: GeomorphometricalVariable,
        tile_size: TileSize,
        mode: Mode,
    ) -> LandslideImages:
        map_ = self.map.setdefault(tile_size, {})
        if variable not in map_:
            map_[variable] = get_landslide_images_for_variable(
                variable, tile_size, mode
            )
        return map_[variable]

    def set(
        self,
        tile_size: TileSize,
        variable: GeomorphometricalVariable,
        landslide_images: LandslideImages,
    ) -> None:
        self.map.setdefault(tile_size, {})[variable] = landslide_images

    def to_map(
        self,
    ) -> dict[TileSize, dict[GeomorphometricalVariable, LandslideImages]]:
        return self.map
