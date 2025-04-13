from __future__ import annotations

import collections.abc as c
import os
import shutil
import typing as t
from pathlib import Path

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
from torch.utils.data import random_split

from landnet.config import (
    ARCHITECTURE,
    CHECKPOINTS_DIR,
    EPOCHS,
    GPUS,
    MODELS_DIR,
    OVERWRITE,
    TEMP_RAY_TUNE_DIR,
    TRIAL_NAME,
)
from landnet.enums import GeomorphometricalVariable, Mode
from landnet.features.tiles import (
    ConcatLandslideImages,
    LandslideImages,
    TileConfig,
    TileSize,
    get_landslide_images_for_variable,
)
from landnet.logger import create_logger
from landnet.modelling.lightning import (
    LandslideImageClassifier,
    LandslideImageDataModule,
)
from landnet.modelling.models import get_architecture
from landnet.modelling.tune import get_tuner
from landnet.utils import get_utc_now

if t.TYPE_CHECKING:
    from landnet._typing import TrainTestValidation, TuneSpace
    from landnet.modelling.tune import MetricSorter

logger = create_logger(__name__)


def save_experiment(
    results: ResultGrid,
    sorter: MetricSorter,
    variables: c.Sequence[GeomorphometricalVariable],
) -> None:
    experiment_out_dir = MODELS_DIR / TRIAL_NAME / get_utc_now()
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
    shutil.move(Path(best_result.path) / 'predictions', experiment_out_dir)
    for content in CHECKPOINTS_DIR.iterdir():
        shutil.move(content, experiment_out_dir)
    with (experiment_out_dir / 'geomorphometrical_variables').open(
        mode='w'
    ) as file:
        for var in variables:
            file.write(f'{str(var)}\n')


def train_model(
    variables: c.Sequence[GeomorphometricalVariable],
    model_name: str,
    cacher: ray.ObjectRef[LandslideImagesCacher],  # type: ignore
    sorter: MetricSorter,
):
    assert TRIAL_NAME is not None
    TEMP_RAY_TUNE_DIR.mkdir(exist_ok=True)
    out_name = MODELS_DIR / TRIAL_NAME / f'{model_name}_{ARCHITECTURE.value}'
    os.makedirs(out_name.parent, exist_ok=True)
    try:
        if (
            not OVERWRITE
            and (
                out_name.with_suffix('.ckpt').exists()
                or out_name.with_suffix('.pt').exists()
            )
            and out_name.with_suffix('.csv').exists()
        ):
            return None

        tuner = get_tuner(
            train_func,
            func_kwds={
                'landslide_images_cacher': cacher,
                'model': get_architecture(ARCHITECTURE),
                'variables': variables,
            },
            sorter=sorter,
            variables=variables,
            trial_dir=TEMP_RAY_TUNE_DIR / TRIAL_NAME,
            run_config_kwargs={
                'name': model_name,
            },
        )
        results = tuner.fit()
        save_experiment(results, sorter, variables)
    except Exception as e:
        logger.info(
            'Error %s occured. Will cleanup %s' % (e, TEMP_RAY_TUNE_DIR)
        )
        shutil.rmtree(TEMP_RAY_TUNE_DIR, ignore_errors=True)


def train_func(
    config: TuneSpace,
    variables: c.Sequence[GeomorphometricalVariable],
    landslide_images_cacher: ray.ObjectRef[LandslideImagesCacher],  # type: ignore
    model: c.Callable[[int, Mode], nn.Module],
    **kwargs,
):
    train_dataset, validation_dataset, test_dataset = get_datsets_from_cacher(
        landslide_images_cacher, config, variables
    )
    dm = LandslideImageDataModule(
        config,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        # test_dataset=test_dataset,
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
        # profiler='simple',
    )


type CachedImages = c.MutableMapping[
    GeomorphometricalVariable, dict[Mode, LandslideImages]
]


@ray.remote
class LandslideImagesCacher:
    """Actor used to cache the LandslideImages.

    Because of the _get_data_indices() method, LandslideImages takes some time
    to load."""

    def __init__(self):
        self.map: c.MutableMapping[TileSize, CachedImages] = {}

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
        tile_config: TileConfig,
        mode: Mode,
    ) -> LandslideImages:
        map_ = self.map.setdefault(tile_config.size, {}).setdefault(
            variable, {}
        )
        if mode not in map_:
            logger.info('%r' % mode)
            map_[mode] = get_landslide_images_for_variable(
                variable, tile_config, mode
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
    ) -> c.MutableMapping[TileSize, CachedImages]:
        return self.map


def get_datsets_from_cacher(
    cacher: ray.ObjectRef[LandslideImagesCacher],  # type: ignore
    config: TuneSpace,
    variables: c.Sequence[GeomorphometricalVariable],
) -> TrainTestValidation:
    train = ray.get(
        [
            cacher.setdefault.remote(  # type: ignore
                variable, config['tile_config'], Mode.TRAIN
            )
            for variable in variables
        ]
    )
    test = ray.get(
        [
            cacher.setdefault.remote(  # type: ignore
                variable, config['tile_config'], Mode.TEST
            )
            for variable in variables
        ]
    )
    if len(variables) > 1:
        dataset = ConcatLandslideImages(
            train
        )
        train_dataset, validation_dataset = random_split(
            dataset, (0.7, 0.3)
        )
        test_dataset = ConcatLandslideImages(test)
        return (train_dataset, validation_dataset, test_dataset)
    else:
        train_dataset, validation_dataset = random_split(train[0], (0.7, 0.3))
        return (train_dataset, validation_dataset, test[0])
