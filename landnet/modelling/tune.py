from __future__ import annotations

import collections.abc as c
import os
import shutil
import typing as t
from pathlib import Path

import pandas as pd
from ray import train, tune
from ray.train import Checkpoint, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.tune import ExperimentAnalysis, Result, ResultGrid
from ray.tune.experiment import Trial
from ray.tune.search.hyperopt import HyperOptSearch

from landnet.config import (
    GPUS,
    NUM_SAMPLES,
    OVERLAP,
    SEED,
    TEMP_RAY_TUNE_DIR,
    TILE_SIZE,
    save_vars_as_json,
)
from landnet.enums import GeomorphometricalVariable, Mode
from landnet.features.tiles import TileConfig, TileSize
from landnet.logger import create_logger
from landnet.modelling.classification.inference import (
    InferTrainTest,
    has_predictions,
    save_predictions,
)
from landnet.typing import TuneSpace

logger = create_logger(__name__)


class MetricSorter(t.NamedTuple):
    metric: str
    mode: t.Literal['max', 'min']


def get_tune_space() -> TuneSpace:
    return {
        'learning_rate': tune.loguniform(1e-6, 1e-4),
        'batch_size': tune.choice([2, 4, 8]),
        'tile_config': tune.choice(
            [TileConfig(TileSize.from_size(TILE_SIZE), OVERLAP)]
        ),
    }


class SavePredictions(tune.Callback):
    """Computes the predictions for train, validation and test with the best checkpoint."""

    def __init__(
        self,
        sorter: MetricSorter,
        variables: c.Sequence[GeomorphometricalVariable],
        **kwargs,
    ):
        self.sorter = sorter
        self.variables = variables
        super().__init__(**kwargs)

    def _get_best_result(self, trials: t.List[Trial]) -> Result:
        trial_path = trials[0].path
        if trial_path is None:
            raise FileNotFoundError(
                'Could not find the trial path of %r' % trials[0]
            )
        experiment_dir = Path(trial_path).parent
        logger.debug('Finding best result of experiment at %s' % experiment_dir)
        return get_best_result_from_experiment(experiment_dir, self.sorter)

    def get_best_checkpoint(self, result: Result) -> Path:
        ckpt = t.cast(
            Checkpoint | None,
            result.get_best_checkpoint(self.sorter.metric, self.sorter.mode),
        )
        assert ckpt is not None
        return Path(ckpt.path) / 'checkpoint.ckpt'

    def on_experiment_end(self, trials: list[Trial], **info):
        result = self._get_best_result(trials)
        assert result.config is not None
        checkpoint = self.get_best_checkpoint(result)
        infer = InferTrainTest(
            self.variables, checkpoint.parent / 'predictions'
        )
        infer.handle_checkpoint(
            checkpoint,
            result.config['train_loop_config'],
            modes=(Mode.TEST, Mode.TRAIN, Mode.VALIDATION),
        )


def get_metrics_from_checkpoint(checkpoint: Path) -> dict[str, float]:
    predictions = checkpoint.parent / 'predictions'

    all_metrics: dict[str, float] = {}
    for mode in (Mode.TRAIN, Mode.TEST, Mode.VALIDATION):
        metrics = pd.read_csv(
            predictions / mode.value / 'metrics.csv', index_col=0
        )
        metrics = metrics.iloc[:, 0]
        metrics.index = mode.value + '_' + metrics.index
        all_metrics.update(metrics.to_dict())
    return all_metrics


def get_result_as_dict(
    result: Result, sorter: MetricSorter, fix_missing_predictions: bool = False
) -> dict[str, t.Any]:
    assert result.config is not None
    experiment_dir = Path(result.path).parent
    result_map: dict[str, t.Any] = {}
    result_map['model'] = experiment_dir.stem

    checkpoint_path = (
        Path(
            result.get_best_checkpoint(
                metric=sorter.metric, mode=sorter.mode
            ).path
        )
        / 'checkpoint.ckpt'
    )
    if fix_missing_predictions and not has_predictions(checkpoint_path):
        save_predictions(
            GeomorphometricalVariable.parse_file(
                experiment_dir / 'geomorphometrical_variables'
            ),
            checkpoint_path,
        )
    result_map.update(get_metrics_from_checkpoint(checkpoint_path))
    result_map.update(result.config['train_loop_config'])
    result_map['checkpoint'] = checkpoint_path.parent.name
    result_map['epoch'] = int(checkpoint_path.parent.name.split('_')[1])
    return result_map


def get_results_df(
    results: c.Sequence[Result],
    sorter: MetricSorter,
    fix_missing_predictions: bool = False,
) -> pd.DataFrame:
    result_maps = [
        get_result_as_dict(result, sorter, fix_missing_predictions)
        for result in results
    ]
    return pd.DataFrame(result_maps)


def get_tuner(
    train_func,
    func_kwds: dict,
    sorter: MetricSorter,
    variables: c.Sequence[GeomorphometricalVariable],
    trial_dir: Path,
    run_config_kwargs: dict[str, t.Any],
) -> tune.Tuner:
    def get_run_config():
        return train.RunConfig(
            **run_config_kwargs,
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute=sorter.metric,
                checkpoint_score_order=sorter.mode,
            ),
            callbacks=[SavePredictions(sorter, variables)],  # type: ignore
            storage_path=TEMP_RAY_TUNE_DIR.as_posix(),
        )

    def get_tune_config():
        return tune.TuneConfig(
            num_samples=NUM_SAMPLES,
            search_alg=HyperOptSearch(
                metric=sorter.metric, mode=sorter.mode, random_state_seed=SEED
            ),
        )

    def get_trainable(restore: bool = False):
        checkpoint = None
        if restore:
            experiment = ExperimentAnalysis(trial_dir)
            trial = experiment.trials[-1]
            checkpoint = experiment.get_last_checkpoint(
                trial=trial, metric='training_iteration'
            )
        return TorchTrainer(
            tune.with_parameters(train_func, **func_kwds),
            scaling_config=train.ScalingConfig(use_gpu=bool(GPUS)),
            run_config=get_run_config(),
            resume_from_checkpoint=checkpoint,
        )

    def get_param_space():
        return {
            'train_loop_config': get_tune_space(),
        }

    if tune.Tuner.can_restore(trial_dir):
        logger.info('Attempting to restore trial.')
        tuner = tune.Tuner.restore(
            trial_dir.as_posix(),
            trainable=get_trainable(restore=True),  # type: ignore
            param_space=get_param_space(),
        )
        logger.info('Trial %s restored' % trial_dir)
    else:
        tuner = tune.Tuner(
            get_trainable(),
            param_space=get_param_space(),
            tune_config=get_tune_config(),
        )
    return tuner


def get_best_result_from_experiment(
    experiment_dir: Path, sorter: MetricSorter
) -> Result:
    analysis = ExperimentAnalysis(
        experiment_dir, default_metric=sorter.metric, default_mode=sorter.mode
    )
    result_grid = ResultGrid(analysis)
    return result_grid.get_best_result(
        metric=sorter.metric, mode=sorter.mode, scope='all'
    )


def save_experiment(
    results: ResultGrid,
    variables: c.Sequence[GeomorphometricalVariable],
    model_name: str,
    out_dir: Path,
    sorter: MetricSorter,
):
    experiment_dir = TEMP_RAY_TUNE_DIR / model_name
    os.makedirs(out_dir, exist_ok=True)
    results.get_dataframe().T.to_csv(out_dir / 'trials.csv')
    for content in experiment_dir.iterdir():
        shutil.move(content, out_dir)
    with (out_dir / 'geomorphometrical_variables').open(mode='w') as file:
        for var in variables:
            file.write(f'{str(var)}\n')
    save_vars_as_json(out_dir / 'config.json')
