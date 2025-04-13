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
    EPOCHS,
    GPUS,
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
        # grace_period=EPOCHS,
        grace_period=3,
        reduction_factor=2,
    )


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
            callbacks=[SaveTrial(sorter, variables)],  # type: ignore
            storage_path=TEMP_RAY_TUNE_DIR.as_posix(),
        )

    def get_tune_config():
        return tune.TuneConfig(
            scheduler=get_scheduler(sorter),
            num_samples=NUM_SAMPLES,
            search_alg=HyperOptSearch(metric='val_loss', mode='min'),
        )

    def get_trainable():
        return TorchTrainer(
            tune.with_parameters(train_func, **func_kwds),
            scaling_config=train.ScalingConfig(use_gpu=bool(GPUS)),
            run_config=get_run_config(),
        )

    def get_param_space():
        return {
            'train_loop_config': get_tune_space(),
        }

    if tune.Tuner.can_restore(trial_dir):
        return tune.Tuner.restore(
            trial_dir.as_posix(),
            trainable=get_trainable(),  # type: ignore
            param_space=get_param_space(),
            resume_errored=True,
            resume_unfinished=True,
            restart_errored=True,
        )
    return tune.Tuner(
        get_trainable(),  # type: ignore
        param_space=get_param_space(),
        tune_config=get_tune_config(),
    )
