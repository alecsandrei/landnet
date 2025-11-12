from __future__ import annotations

import collections.abc as c
import os
import shutil
import typing as t

import lightning.pytorch as pl
import ray
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.tune import ResultGrid
from torch import nn

from landnet.config import (
    ARCHITECTURE,
    EPOCHS,
    GPUS,
    MODELS_DIR,
    OVERWRITE,
    TEMP_RAY_TUNE_DIR,
    TRIAL_NAME,
    save_vars_as_json,
)
from landnet.enums import GeomorphometricalVariable, Mode
from landnet.features.grids import get_grid_for_variable
from landnet.features.tiles import (
    TileConfig,
    TileSize,
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
from landnet.modelling.tune import get_tuner
from landnet.utils import get_utc_now

if t.TYPE_CHECKING:
    from landnet.modelling.tune import MetricSorter
    from landnet.typing import (
        CachedImages,
        ClassificationTrainTestValidation,
        TuneSpace,
    )

logger = create_logger(__name__)


def save_experiment(
    results: ResultGrid,
    sorter: MetricSorter,
    variables: c.Sequence[GeomorphometricalVariable],
    model_name: str,
) -> None:
    experiment_out_dir = MODELS_DIR / TRIAL_NAME / model_name / get_utc_now()
    experiment_dir = TEMP_RAY_TUNE_DIR / model_name
    os.makedirs(experiment_out_dir, exist_ok=True)
    results.get_dataframe().T.to_csv(experiment_out_dir / 'trials.csv')
    best_result = results.get_best_result(
        sorter.metric, sorter.mode, scope='all'
    )
    assert best_result.config
    assert best_result.metrics

    logger.info('Best trial config: %s' % best_result.config)
    logger.info('Best trial final validation metrics: %s' % best_result.metrics)
    best_checkpoint = best_result.get_best_checkpoint(
        metric=sorter.metric, mode=sorter.mode
    )
    assert best_checkpoint is not None
    # shutil.move(Path(best_result.path) / 'predictions', experiment_out_dir)
    for content in experiment_dir.iterdir():
        shutil.move(content, experiment_out_dir)
    with (experiment_out_dir / 'geomorphometrical_variables').open(
        mode='w'
    ) as file:
        for var in variables:
            file.write(f'{str(var)}\n')
    save_vars_as_json(experiment_out_dir / 'config.json')


def train_model(
    variables: c.Sequence[GeomorphometricalVariable],
    model_name: str,
    cacher: ray.ObjectRef[LandslideImageClassificationCacher],  # type: ignore
    sorter: MetricSorter,
):
    assert TRIAL_NAME is not None
    TEMP_RAY_TUNE_DIR.mkdir(exist_ok=True)
    experiment_dir = MODELS_DIR / TRIAL_NAME / model_name
    if experiment_dir.exists() and not OVERWRITE:
        logger.info(
            'Skipping training for %s as %s already exists.'
            % (variables, experiment_dir)
        )
        return None
    try:
        tuner = get_tuner(
            train_func,
            func_kwds={
                'landslide_images_cacher': cacher,
                'model': get_architecture(ARCHITECTURE),
                'variables': variables,
            },
            sorter=sorter,
            variables=variables,
            trial_dir=TEMP_RAY_TUNE_DIR / model_name,
            run_config_kwargs={'name': model_name},
        )
        results = tuner.fit()
        save_experiment(results, sorter, variables, model_name)
    except Exception as e:
        logger.info('Error %s occured for variables %s' % (e, variables))


def train_func(
    config: TuneSpace,
    variables: c.Sequence[GeomorphometricalVariable],
    landslide_images_cacher: ray.ObjectRef[LandslideImageClassificationCacher],  # type: ignore
    model: c.Callable[[int, Mode], nn.Module],
    **kwargs,
):
    train_dataset, validation_dataset, test_dataset = get_datsets_from_cacher(
        landslide_images_cacher, config, variables, return_test=False
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


@ray.remote
class LandslideImageClassificationCacher:
    """Actor used to cache the LandslideImageClassification.

    Because of the _get_data_indices() method, LandslideImageClassification takes some time
    to load."""

    def __init__(self):
        self.map: c.MutableMapping[
            TileSize, CachedImages[LandslideImageClassification]
        ] = {}

    def get(
        self,
        tile_size: TileSize,
        mode: Mode,
        variable: GeomorphometricalVariable,
    ) -> LandslideImageClassification | None:
        try:
            return self.map[tile_size][variable][mode]
        except KeyError:
            return None

    def setdefault(
        self,
        variable: GeomorphometricalVariable,
        tile_config: TileConfig,
        mode: Mode,
        augment_transform: nn.Module | None = None,
    ) -> LandslideImageClassification:
        map_ = self.map.setdefault(tile_config.size, {}).setdefault(
            variable, {}
        )
        if mode not in map_:
            logger.info('%r' % mode)
            grid = get_grid_for_variable(variable, tile_config, mode)
            map_[mode] = LandslideImageClassification(
                grid, mode, augment_transform=augment_transform
            )
        return map_[mode]

    def set(
        self,
        tile_size: TileSize,
        variable: GeomorphometricalVariable,
        landslide_images: LandslideImageClassification,
        mode: Mode,
    ) -> None:
        self.map.setdefault(tile_size, {}).setdefault(variable, {})[mode] = (
            landslide_images
        )

    def to_map(
        self,
    ) -> c.MutableMapping[TileSize, CachedImages]:
        return self.map


def get_datsets_from_cacher(
    cacher: ray.ObjectRef[LandslideImageClassificationCacher],  # type: ignore
    config: TuneSpace,
    variables: c.Sequence[GeomorphometricalVariable],
    return_test: bool = True,
) -> ClassificationTrainTestValidation:
    train = ray.get(
        [
            cacher.setdefault.remote(  # type: ignore
                variable, config['tile_config'], Mode.TRAIN
            )
            for variable in variables
        ]
    )
    test_dataset = None
    if return_test:
        test = ray.get(
            [
                cacher.setdefault.remote(  # type: ignore
                    variable, config['tile_config'], Mode.TEST
                )
                for variable in variables
            ]
        )
    validation = ray.get(
        [
            cacher.setdefault.remote(  # type: ignore
                variable, config['tile_config'], Mode.VALIDATION
            )
            for variable in variables
        ]
    )
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
