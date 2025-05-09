from __future__ import annotations

import collections.abc as c
import typing as t

import lightning.pytorch as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from landnet.enums import GeomorphometricalVariable, Mode
from landnet.features.grids import get_grid_for_variable
from landnet.logger import create_logger
from landnet.modelling.dataset import get_default_augment_transform
from landnet.modelling.models import read_legacy_checkpoint
from landnet.modelling.segmentation.dataset import (
    ConcatLandslideImageSegmentation,
    LandslideImageSegmentation,
    get_segmentation_dataloader,
)
from landnet.modelling.segmentation.stats import SegmentationMetricCollection

if t.TYPE_CHECKING:
    from landnet._typing import (
        AnyLandslideSegmentationDataset,
        TuneSpace,
    )

logger = create_logger(__name__)


class LandslideImageSegmenter(pl.LightningModule):
    def __init__(
        self, config: TuneSpace, model: nn.Module, num_classes: int = 2
    ):
        super().__init__()
        self.config = config
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.train_metrics = SegmentationMetricCollection(
            num_classes=num_classes, prefix='train_'
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
        image, mask = batch
        output = self.forward(image)
        if isinstance(output, c.Mapping):
            # DeepLabV3 Model outputs a dictionary for some reason ?????
            output = output['out']

        return (image, mask)

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        image, mask = train_batch
        output = self.forward(image)
        if isinstance(output, c.Mapping):
            # DeepLabV3 Model outputs a dictionary for some reason ?????
            output = output['out']
        output = output.sigmoid()
        loss = self.criterion(output, mask.float())
        output_mask = output > 0.5
        self.train_metrics.update(output_mask, mask)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(
            self.train_metrics.compute(),
            sync_dist=True,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self.train_metrics.reset()

    def validation_step(self, val_batch, batch_idx):
        image, mask = val_batch
        output = self.forward(image)
        if isinstance(output, c.Mapping):
            # DeepLabV3 Model outputs a dictionary for some reason ?????
            output = output['out']
        output = output.sigmoid()
        loss = self.criterion(output, mask.float())
        output_mask = output > 0.5
        self.val_metrics.update(output_mask, mask)
        self.log(
            'val_loss',
            loss,
            sync_dist=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, test_batch, batch_idx):
        image, mask = test_batch
        output = self.forward(image)
        if isinstance(output, c.Mapping):
            # DeepLabV3 Model outputs a dictionary for some reason ?????
            output = output['out']
        output = output.sigmoid()
        loss = self.criterion(output, mask.float())
        output_mask = output > 0.5
        self.test_metrics.update(output_mask, mask)
        self.log_dict(
            self.test_metrics.compute(),
            sync_dist=True,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            'test_loss',
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

    def on_validation_epoch_end(self):
        self.log_dict(
            self.val_metrics.compute(),
            sync_dist=True,
            prog_bar=True,
            logger=True,
        )
        self.val_metrics.reset()

    def configure_optimizers(self):
        assert 'learning_rate' in self.config
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config['learning_rate']
        )
        return optimizer


class LandslideImageSegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: TuneSpace,
        variables: c.Sequence[GeomorphometricalVariable],
        train_dataset: AnyLandslideSegmentationDataset | None = None,
        validation_dataset: AnyLandslideSegmentationDataset | None = None,
        test_dataset: AnyLandslideSegmentationDataset | None = None,
    ):
        super().__init__()
        self.config = config
        self.variables = variables
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

    def setup(self, stage=None):
        if self.train_dataset is None or self.validation_dataset is None:
            logger.info(
                'Setting up train, test and validation datasets based on %r'
                % self.variables
            )
            train_grids = [
                get_grid_for_variable(
                    variable,
                    tile_config=self.config['tile_config'],
                    mode=Mode.TRAIN,
                )
                for variable in self.variables
            ]
            test_grids = [
                get_grid_for_variable(
                    variable,
                    tile_config=self.config['tile_config'],
                    mode=Mode.TEST,
                )
                for variable in self.variables
            ]
            dataset = ConcatLandslideImageSegmentation(
                landslide_images=[
                    LandslideImageSegmentation(grid) for grid in train_grids
                ],
                augment_transform=None,
            )
            self.train_dataset, self.validation_dataset = (
                torch.utils.data.random_split(dataset, (0.7, 0.3))
            )
            t.cast(
                ConcatLandslideImageSegmentation, self.train_dataset.dataset
            ).augment_transform = get_default_augment_transform()
            self.test_dataset = ConcatLandslideImageSegmentation(
                landslide_images=[
                    LandslideImageSegmentation(grid) for grid in test_grids
                ],
                augment_transform=None,
            )
            logger.info('Finished setting up train and validation datasets.')

    def train_dataloader(self):
        assert self.train_dataset is not None
        return get_segmentation_dataloader(
            self.train_dataset,
            # size=len(self.train_dataset),
            size=10000,
            batch_size=self.config['batch_size'],
            num_workers=4,
            prefetch_factor=4,  # Load 4 batches ahead
            persistent_workers=True,  # Keeps workers alive
            pin_memory=True,
        )

    def val_dataloader(self):
        assert self.validation_dataset is not None
        data_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.config['batch_size'],
            num_workers=4,
            shuffle=False,
            pin_memory=True,
        )
        return data_loader

    def test_dataloader(self):
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            num_workers=4,
            shuffle=False,
            pin_memory=True,
        )
