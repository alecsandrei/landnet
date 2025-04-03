from __future__ import annotations

import collections.abc as c
import typing as t

import lightning.pytorch as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from landnet._typing import (
    AnyLandslideDataset,
    LandslideDataset,
)
from landnet.enums import GeomorphometricalVariable
from landnet.features.tiles import (
    DEFAULT_CLASS_BALANCE,
    create_dataloader,
)
from landnet.logger import create_logger
from landnet.modelling.models import read_legacy_checkpoint
from landnet.modelling.stats import BinaryClassificationMetricCollection

if t.TYPE_CHECKING:
    from landnet._typing import TuneSpace

logger = create_logger(__name__)


class LandslideImageClassifier(pl.LightningModule):
    def __init__(self, config: TuneSpace, model: nn.Module):
        super().__init__()
        self.config = config
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2]))
        self.train_metrics = BinaryClassificationMetricCollection(
            prefix='train_'
        )

        self.val_metrics = self.train_metrics.clone(prefix='val_')
        self.test_metrics = self.train_metrics.clone(prefix='test_')
        self.predict_metrics = self.train_metrics.clone(prefix='predict_')

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=None,
        model: nn.Module | None = None,
        config: TuneSpace | None = None,
        **kwargs: t.Any,
    ) -> t.Self:
        try:
            return super().load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                map_location=map_location,
                hparams_file=hparams_file,
                strict=strict,
                model=model,
                config=config,
                **kwargs,
            )
        except Exception as e:
            logger.error(
                'Failed to load the checkpoint at %s with %s. Attempting the legacy way.'
                % (e, checkpoint_path)
            )
            if model is None or config is None:
                raise ValueError(
                    'Will not attempt reading the checkpoint in a legacy way.'
                )
            model = read_legacy_checkpoint(model, checkpoint_path)
            return cls(config=config, model=model)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x = torch.sigmoid(self.forward(x))
        return (x, y)

    def forward(self, x):
        return self.model(x).flatten()

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        self.train_metrics.update(logits, y)
        loss = self.criterion(logits, y.float())
        return loss

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
        config: TuneSpace,
        variables: c.Sequence[GeomorphometricalVariable],
        train_dataset: AnyLandslideDataset | None = None,
        validation_dataset: AnyLandslideDataset | None = None,
        test_dataset: LandslideDataset | None = None,
    ):
        super().__init__()
        self.config = config
        self.variables = variables
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

    def setup(self, stage=None): ...

    def train_dataloader(self):
        assert self.train_dataset is not None
        return create_dataloader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            num_workers=4,
            prefetch_factor=4,  # Load 4 batches ahead
            persistent_workers=True,  # Keeps workers alive
            class_balance=DEFAULT_CLASS_BALANCE,
            pin_memory=True,
        )

    def val_dataloader(self):
        assert self.validation_dataset is not None
        return DataLoader(
            self.validation_dataset,
            batch_size=self.config['batch_size'],
            num_workers=4,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            num_workers=4,
            shuffle=False,
            pin_memory=True,
        )
