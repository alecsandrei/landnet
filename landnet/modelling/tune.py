from __future__ import annotations

import collections.abc as c
import typing as t
from pathlib import Path

from ray import train, tune
from ray.train import CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.tune import ExperimentAnalysis
from ray.tune.experiment import Trial
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from landnet.config import (
    CHECKPOINTS_DIR,
    CPUS,
    EPOCHS,
    NUM_SAMPLES,
    OVERLAP,
    TEMP_RAY_TUNE_DIR,
)
from landnet.enums import GeomorphometricalVariable
from landnet.features.tiles import TileConfig, TileSize
from landnet.logger import create_logger
from landnet.modelling.inference import InferTrainTest

logger = create_logger(__name__)


class MetricSorter(t.NamedTuple):
    metric: str
    mode: t.Literal['max', 'min']


def get_tune_space():
    return {
        'learning_rate': tune.loguniform(1e-6, 1e-3),
        'batch_size': tune.choice([2, 4, 8]),
        'tile_config': tune.choice([TileConfig(TileSize(100, 100), OVERLAP)]),
    }


class SaveTrial(tune.Callback):
    def __init__(
        self,
        sorter: MetricSorter,
        variables: c.Sequence[GeomorphometricalVariable],
        **kwargs,
    ):
        self.sorter = sorter
        self.variables = variables
        super().__init__(**kwargs)

    def _get_best_result(self, trials: t.List[Trial]):
        analysis = ExperimentAnalysis(
            CHECKPOINTS_DIR,
            trials=trials,
            default_metric=self.sorter.metric,
            default_mode=self.sorter.mode,
        )
        results = tune.ResultGrid(analysis)
        return results.get_best_result(scope='all')

    def on_trial_complete(
        self, iteration: int, trials: t.List[Trial], trial: Trial, **info
    ):
        result = self._get_best_result([trial])
        ckpt = result.get_best_checkpoint(self.sorter.metric, self.sorter.mode)
        assert result.config is not None and ckpt is not None
        logger.info('Trainable name: %s' % trial.path)
        infer = InferTrainTest(self.variables, self.sorter, Path(trial.path))  # type: ignore
        infer.handle_checkpoint(ckpt, result.config['train_loop_config'])

    def _on_experiment_end(self, trials: list[Trial], **info):
        result = self._get_best_result(trials)

        ckpt = result.get_best_checkpoint(self.sorter.metric, self.sorter.mode)
        assert result.config is not None and ckpt is not None
        infer = InferTrainTest(self.variables, self.sorter)
        infer.handle_checkpoint(ckpt, result.config['train_loop_config'])


def get_scheduler(sorter: MetricSorter):
    return ASHAScheduler(
        metric=sorter.metric,
        mode=sorter.mode,
        max_t=EPOCHS,
        grace_period=EPOCHS,
        reduction_factor=2,
    )


def get_tuner(
    train_func,
    func_kwds: dict,
    sorter: MetricSorter,
    variables: c.Sequence[GeomorphometricalVariable],
    run_config_kwargs: dict[str, t.Any],
) -> tune.Tuner:
    return tune.Tuner(
        TorchTrainer(
            tune.with_parameters(train_func, **func_kwds),
            scaling_config=train.ScalingConfig({'CPU': CPUS}, use_gpu=True),
            run_config=train.RunConfig(
                **run_config_kwargs,
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute=sorter.metric,
                    checkpoint_score_order=sorter.mode,
                ),
                callbacks=[SaveTrial(sorter, variables)],
                storage_path=TEMP_RAY_TUNE_DIR.as_posix(),
            ),
        ),
        param_space={
            'train_loop_config': get_tune_space(),
        },
        tune_config=tune.TuneConfig(
            scheduler=get_scheduler(sorter),
            num_samples=NUM_SAMPLES,
            search_alg=HyperOptSearch(metric='val_loss', mode='min'),
        ),
    )
