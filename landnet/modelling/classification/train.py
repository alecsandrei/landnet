from __future__ import annotations

import collections.abc as c
import typing as t

import lightning.pytorch as pl
import torch
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.tune import Result
from torch import nn

from landnet.config import (
    ARCHITECTURE,
    EPOCHS,
    GPUS,
    MODELS_DIR,
    OVERWRITE,
    TEMP_RAY_TUNE_DIR,
    TRIAL_NAME,
)
from landnet.enums import GeomorphometricalVariable, Mode
from landnet.features.grids import get_grid_for_variable
from landnet.features.tiles import (
    TileConfig,
)
from landnet.logger import create_logger
from landnet.modelling.classification.dataset import (
    ConcatLandslideImageClassification,
    LandslideImageClassification,
)
from landnet.modelling.classification.lightning import (
    LandslideImageClassifier,
    LandslideImageDataModule,
)
from landnet.modelling.classification.models import get_architecture
from landnet.modelling.dataset import get_default_augment_transform
from landnet.modelling.tune import (
    get_best_result_from_experiment,
    get_tuner,
    save_experiment,
)
from landnet.typing import (
    LandslideClassificationDataset,
)

if t.TYPE_CHECKING:
    from landnet.modelling.tune import MetricSorter
    from landnet.typing import (
        ClassificationTrainTestValidation,
        TuneSpace,
    )

logger = create_logger(__name__)


def train_model(
    variables: c.Sequence[GeomorphometricalVariable],
    model_name: str,
    sorter: MetricSorter,
) -> Result | None:
    assert TRIAL_NAME is not None
    TEMP_RAY_TUNE_DIR.mkdir(exist_ok=True)
    experiment_dir = MODELS_DIR / TRIAL_NAME / model_name
    if experiment_dir.exists() and not OVERWRITE:
        logger.info(
            'Skipping training for %s as %s already exists.'
            % (variables, experiment_dir)
        )
        return get_best_result_from_experiment(experiment_dir, sorter)
    # try:
    tuner = get_tuner(
        train_func,
        func_kwds={
            'model': get_architecture(ARCHITECTURE),
            'variables': variables,
        },
        sorter=sorter,
        variables=variables,
        trial_dir=TEMP_RAY_TUNE_DIR / model_name,
        run_config_kwargs={'name': model_name},
    )
    results = tuner.fit()
    best_result = results.get_best_result(
        sorter.metric, sorter.mode, scope='all'
    )
    assert best_result.config
    assert best_result.metrics

    logger.info('Best trial config: %s' % best_result.config)
    logger.info('Best trial final validation metrics: %s' % best_result.metrics)
    save_experiment(results, sorter, variables, model_name)
    return best_result


def train_func(
    config: TuneSpace,
    variables: c.Sequence[GeomorphometricalVariable],
    model: c.Callable[[int, Mode], nn.Module],
    **kwargs,
):
    train_dataset, validation_dataset, test_dataset = get_datasets(
        config, variables, return_test=False
    )
    dm = LandslideImageDataModule(
        config,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        variables=variables,
    )
    model = model(len(variables), Mode.TRAIN)
    model = LandslideImageClassifier(model=model, config=config)

    trainer = prepare_trainer(get_trainer())
    return trainer.fit(model, datamodule=dm)


def get_trainer():
    return pl.Trainer(
        devices='auto',
        accelerator='gpu' if bool(GPUS) else 'cpu',
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
        max_epochs=EPOCHS,
    )


def get_dataset_for_mode(
    variable: GeomorphometricalVariable,
    tile_config: TileConfig,
    mode: Mode,
    augment_transform: c.Callable[..., list[torch.Tensor]] | None = None,
) -> LandslideImageClassification:
    grid = get_grid_for_variable(variable, tile_config, mode)
    return LandslideImageClassification(
        grid, mode, augment_transform=augment_transform
    )


def get_datasets(
    config: TuneSpace,
    variables: c.Sequence[GeomorphometricalVariable],
    return_test: bool = True,
) -> ClassificationTrainTestValidation:
    augment_transform = (
        get_default_augment_transform() if len(variables) == 1 else None
    )
    train = [
        get_dataset_for_mode(
            variable,
            config['tile_config'],
            Mode.TRAIN,
            augment_transform=augment_transform,
        )
        for variable in variables
    ]
    test_dataset: None | LandslideClassificationDataset = None
    tile_config_test = TileConfig(config['tile_config'].size, overlap=0)
    validation = [
        get_dataset_for_mode(variable, tile_config_test, Mode.VALIDATION)
        for variable in variables
    ]
    if return_test:
        test = [
            get_dataset_for_mode(variable, tile_config_test, Mode.TEST)
            for variable in variables
        ]
    if len(variables) > 1:
        train_dataset = ConcatLandslideImageClassification(
            train, augment_transform=get_default_augment_transform()
        )
        validation_dataset = ConcatLandslideImageClassification(validation)
        if return_test:
            test_dataset = ConcatLandslideImageClassification(test)
        return (train_dataset, validation_dataset, test_dataset)
    else:
        if return_test:
            test_dataset = test[0]
        return (train[0], validation[0], test_dataset)
