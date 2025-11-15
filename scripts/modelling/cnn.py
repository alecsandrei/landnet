from __future__ import annotations

import collections.abc as c

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from landnet.config import ARCHITECTURE, EPOCHS, GPUS, OVERLAP, TILE_SIZE
from landnet.enums import GeomorphometricalVariable, Mode
from landnet.features.grids import get_grid_for_variable
from landnet.features.tiles import TileConfig, TileSize
from landnet.logger import create_logger
from landnet.modelling import torch_clear
from landnet.modelling.classification.dataset import (
    ConcatLandslideImageClassification,
    LandslideImageClassification,
)
from landnet.modelling.classification.lightning import (
    LandslideImageClassifier,
    LandslideImageDataModule,
)
from landnet.modelling.classification.models import get_architecture
from landnet.modelling.dataset import (
    get_default_augment_transform,
    get_default_transform,
)
from landnet.modelling.tune import MetricSorter
from landnet.typing import TuneSpace

if GPUS:
    torch_clear()


logger = create_logger(__name__)

TRAIN_TILE_CONFIG = TileConfig(TileSize(TILE_SIZE, TILE_SIZE), overlap=OVERLAP)
TRAIN_MODEL_CONFIG: TuneSpace = {
    'batch_size': 8,
    'learning_rate': 3.146057909839248e-05,
    'tile_config': TRAIN_TILE_CONFIG,
}

TEST_TILE_CONFIG = TileConfig(TileSize(TILE_SIZE, TILE_SIZE), overlap=0)
TEST_MODEL_CONFIG: TuneSpace = {
    'batch_size': 4,
    'tile_config': TEST_TILE_CONFIG,
}


def get_datasets(variables: c.Sequence[GeomorphometricalVariable]):
    train_grids = [
        get_grid_for_variable(
            variable,
            tile_config=TRAIN_TILE_CONFIG,
            mode=Mode.TRAIN,
        )
        for variable in variables
    ]

    validation_grids = [
        get_grid_for_variable(
            variable,
            tile_config=TEST_TILE_CONFIG,
            mode=Mode.VALIDATION,
        )
        for variable in variables
    ]

    train_dataset = ConcatLandslideImageClassification(
        landslide_images=[
            LandslideImageClassification(
                grid,
                Mode.TRAIN,
                transform=get_default_transform(),
            )
            for grid in train_grids
        ],
        augment_transform=get_default_augment_transform(),
        # augment_transform=None,
    )

    validation_dataset = ConcatLandslideImageClassification(
        landslide_images=[
            LandslideImageClassification(
                grid,
                Mode.VALIDATION,
                transform=get_default_transform(),
            )
            for grid in validation_grids
        ],
        augment_transform=None,
    )

    test_grids = [
        get_grid_for_variable(
            variable,
            tile_config=TEST_TILE_CONFIG,
            mode=Mode.TEST,
        )
        for variable in variables
    ]
    test_dataset = ConcatLandslideImageClassification(
        landslide_images=[
            LandslideImageClassification(
                grid,
                Mode.TEST,
                transform=get_default_transform(),
            )
            for grid in test_grids
        ],
        augment_transform=None,
    )
    return (train_dataset, validation_dataset, test_dataset)


def train_cnn(variables: c.Sequence[GeomorphometricalVariable]):  # type: ignore
    sorter = MetricSorter('val_f2_score', 'max')
    train_dataset, validation_dataset, test_dataset = get_datasets(variables)
    dm = LandslideImageDataModule(
        TRAIN_MODEL_CONFIG,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        variables=variables,
    )
    architecture = get_architecture(ARCHITECTURE)
    model = architecture(len(variables), Mode.TRAIN)
    model = LandslideImageClassifier(model=model, config=TRAIN_MODEL_CONFIG)
    checkpoint_callback = ModelCheckpoint(
        verbose=True, save_top_k=1, mode=sorter.mode, monitor=sorter.metric
    )
    trainer = L.Trainer(
        # enable_checkpointing=True,
        # callbacks=[EarlyStopping(monitor='val_mIoU', mode='max', patience=5)],
        callbacks=[checkpoint_callback],
        max_epochs=EPOCHS,
    )

    trainer.fit(model, datamodule=dm)
    checkpoint = checkpoint_callback.best_model_path
    result = trainer.test(
        model, datamodule=dm, ckpt_path=checkpoint, verbose=True
    )
    logger.info(result)


if __name__ == '__main__':
    variables_1 = [
        GeomorphometricalVariable('cgene'),
        # GeomorphometricalVariable('clo'),
        # GeomorphometricalVariable('slope'),
        # GeomorphometricalVariable('shade'),
        # GeomorphometricalVariable('cdl'),
        # GeomorphometricalVariable('wind'),
        # GeomorphometricalVariable('cprof'),
        # GeomorphometricalVariable('vld'),
        # GeomorphometricalVariable('tri'),
        # GeomorphometricalVariable('clong'),
        # GeomorphometricalVariable('tpi'),
    ]
    variables_2 = [
        GeomorphometricalVariable('slope'),
    ]
    train_cnn(variables_1)
    train_cnn(variables_2)
