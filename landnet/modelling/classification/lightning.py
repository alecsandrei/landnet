from __future__ import annotations

import collections.abc as c
import typing as t

import lightning.pytorch as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from landnet import RandomSeedContext, seed_worker
from landnet.config import BATCH_SIZE, DEFAULT_CLASS_BALANCE, TRAIN_NUM_SAMPLES
from landnet.enums import GeomorphometricalVariable, Mode
from landnet.logger import create_logger
from landnet.modelling.classification.dataset import (
    create_classification_dataloader,
)
from landnet.modelling.classification.stats import (
    BinaryClassificationMetricCollection,
)
from landnet.modelling.models import read_legacy_checkpoint

if t.TYPE_CHECKING:
    from landnet.typing import (
        AnyLandslideClassificationDataset,
        LandslideClassificationDataset,
        ModelConfig,
    )

logger = create_logger(__name__)


class LandslideImageClassifier(pl.LightningModule):
    def __init__(self, config: ModelConfig, model: nn.Module):
        super().__init__()
        self.config = config
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2]))
        self.train_metrics = BinaryClassificationMetricCollection(
            prefix=f'{Mode.TRAIN.value}_'
        )

        self.validation_metrics = self.train_metrics.clone(
            prefix=f'{Mode.VALIDATION.value}_'
        )
        self.test_metrics = self.train_metrics.clone(
            prefix=f'{Mode.TEST.value}_'
        )

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=None,
        model: nn.Module | None = None,
        config: ModelConfig | None = None,
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

    def on_train_epoch_start(self):
        # We do not want the same augmentation each epoch so we change the seed
        random_seed_context: RandomSeedContext | None = self.config.get(
            'random_seed_context', None
        )
        if random_seed_context is not None:
            random_seed_context.set_random_seed(random_seed_context.seed + 1)
            logger.info(
                'Set seed to %i in on_train_epoch_start'
                % random_seed_context.seed
            )

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.criterion(logits, y.float())

        # Log step metrics
        self.log(
            f'{Mode.TRAIN.value}_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        preds = torch.sigmoid(logits)
        self.train_metrics.update(preds, y)
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.criterion(logits, y.float())

        # Update metrics
        self.validation_metrics.update(torch.sigmoid(logits), y)

        # Log per batch if needed (optional), mostly we care about epoch
        self.log(
            f'{Mode.VALIDATION.value}_loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Compute and log metrics at epoch level
        metrics = self.validation_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_validation_epoch_end(self):
        self.validation_metrics.reset()

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        self.test_metrics.update(torch.sigmoid(logits), y)
        loss = self.criterion(logits, y.float())
        self.log('test_loss', loss, sync_dist=True)
        self.log_dict(self.test_metrics.compute(), sync_dist=True)

    def configure_optimizers(self):
        assert 'learning_rate' in self.config
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config['learning_rate']
        )
        return optimizer


class LandslideImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: ModelConfig,
        variables: c.Sequence[GeomorphometricalVariable],
        train_dataset: AnyLandslideClassificationDataset | None = None,
        validation_dataset: AnyLandslideClassificationDataset | None = None,
        test_dataset: LandslideClassificationDataset | None = None,
        random_seed_context: RandomSeedContext | None = None,
    ):
        super().__init__()
        self.config = config
        self.variables = variables
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.random_seed_context = random_seed_context

    def setup(self, stage=None): ...

    def train_dataloader(self):
        assert self.train_dataset is not None
        logger.info(
            'Length of train dataset: %d, picking randomly with replacement %d samples for training.'
            % (len(self.train_dataset), TRAIN_NUM_SAMPLES)
        )
        return create_classification_dataloader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            random_seed_context=self.config.get('random_seed_context', None),
            num_workers=4,
            prefetch_factor=4,  # Load 4 batches ahead
            persistent_workers=True,  # Keeps workers alive
            class_balance=DEFAULT_CLASS_BALANCE,
            pin_memory=True,
            worker_init_fn=seed_worker,
            size=TRAIN_NUM_SAMPLES,
        )

    def val_dataloader(self):
        assert self.validation_dataset is not None
        return DataLoader(
            self.validation_dataset,
            batch_size=BATCH_SIZE,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=BATCH_SIZE,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
        )
