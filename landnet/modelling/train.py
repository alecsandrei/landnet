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
from landnet.logger import create_logger
from landnet.modelling.models import read_legacy_checkpoint
from landnet.modelling.stats import BinaryClassificationMetricCollection

logger = create_logger(__name__)


@cache
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    print('Warning: CUDA is not available.')
    return torch.get_default_device()


class ConfigSpec(t.TypedDict):
    batch_size: int
    tile_size: TileSize
    learning_rate: t.NotRequired[float]


class LandslideImageClassifier(pl.LightningModule):
    def __init__(self, config: ConfigSpec, model: nn.Module):
        super().__init__()
        self.config = config
        self.model = model
        self.criterion = nn.BCELoss()
        self.train_metrics = BinaryClassificationMetricCollection(
            prefix='train_'
        )

        self.val_metrics = self.train_metrics.clone(prefix='val_')
        self.test_metrics = self.train_metrics.clone(prefix='test_')

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=None,
        model: nn.Module | None = None,
        config: ConfigSpec | None = None,
        **kwargs: t.Any,
    ) -> t.Self:
        try:
            return super().load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                map_location=map_location,
                hparams_file=hparams_file,
                strict=strict,
                **kwargs,
            )
        except Exception as e:
            logger.error(
                'Failed to load the checkpoint at %s with %e. Attempting the legacy way.'
                % (e, checkpoint_path)
            )
            if model is None or config is None:
                raise ValueError(
                    'Will not attempt reading the checkpoint in a legacy way.'
                )
            model = read_legacy_checkpoint(model, checkpoint_path)
            return cls(config=config, model=model)

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

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        self.test_metrics.update(logits, y)
        loss = self.criterion(logits, y.float())
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.log('test_loss', loss, sync_dist=True)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.val_metrics.reset()

    def configure_optimizers(self):
        assert 'learning_rate' in self.config
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

    def train_dataloader(self):
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
            TileSize,
            dict[GeomorphometricalVariable, dict[Mode, LandslideImages]],
        ] = {}

    def get(
        self,
        tile_size: TileSize,
        mode: Mode,
        variable: GeomorphometricalVariable,
    ) -> LandslideImages | None:
        try:
            return self.map[tile_size][variable][mode]
        except KeyError:
            return None

    def setdefault(
        self,
        variable: GeomorphometricalVariable,
        tile_size: TileSize,
        mode: Mode,
    ) -> LandslideImages:
        map_ = self.map.setdefault(tile_size, {}).setdefault(variable, {})
        if mode not in map_:
            logger.info('%r' % mode)
            map_[mode] = get_landslide_images_for_variable(
                variable, tile_size, mode
            )
        return map_[mode]

    def set(
        self,
        tile_size: TileSize,
        variable: GeomorphometricalVariable,
        landslide_images: LandslideImages,
        mode: Mode,
    ) -> None:
        self.map.setdefault(tile_size, {}).setdefault(variable, {})[mode] = (
            landslide_images
        )

    def to_map(
        self,
    ) -> dict[
        TileSize,
        dict[GeomorphometricalVariable, dict[Mode, LandslideImages]],
    ]:
        return self.map
