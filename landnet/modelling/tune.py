from __future__ import annotations

import collections.abc as c
import typing as t
from pathlib import Path

from ray import train, tune
from ray.train import Checkpoint, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.tune import ExperimentAnalysis, Result, ResumeConfig
from ray.tune.experiment import Trial
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from landnet.config import (
    EPOCHS,
    GPUS,
    NUM_SAMPLES,
    OVERLAP,
    TEMP_RAY_TUNE_DIR,
    TILE_SIZE,
)
from landnet.enums import GeomorphometricalVariable
from landnet.features.tiles import TileConfig, TileSize
from landnet.logger import create_logger
from landnet.modelling.classification.inference import InferTrainTest

logger = create_logger(__name__)


class MetricSorter(t.NamedTuple):
    metric: str
    mode: t.Literal['max', 'min']


def get_tune_space():
    return {
        'learning_rate': tune.loguniform(1e-6, 1e-4),
        'batch_size': tune.choice([2, 4, 8]),
        'tile_config': tune.choice(
            [TileConfig(TileSize(TILE_SIZE, TILE_SIZE), OVERLAP)]
        ),
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
        # trials[0].local_experiment_path
        analysis = ExperimentAnalysis(
            trials[0].local_experiment_path,
            trials=trials,
            default_metric=self.sorter.metric,
            default_mode=self.sorter.mode,
        )
        results = tune.ResultGrid(analysis)
        return results.get_best_result(scope='all')

    def get_best_checkpoint(self, result: Result) -> Path:
        ckpt = t.cast(
            Checkpoint | None,
            result.get_best_checkpoint(self.sorter.metric, self.sorter.mode),
        )
        assert ckpt is not None
        return Path(ckpt.path) / 'checkpoint.ckpt'

    def on_trial_complete(
        self, iteration: int, trials: t.List[Trial], trial: Trial, **info
    ):
        result = self._get_best_result([trial])
        assert result.config is not None
        logger.info('Trainable name: %s' % trial.path)
        assert trial.path is not None
        infer = InferTrainTest(self.variables, Path(trial.path))
        infer.handle_checkpoint(
            self.get_best_checkpoint(result), result.config['train_loop_config']
        )

    def on_experiment_end(self, trials: list[Trial], **info):
        # if len(trials) == 1:
        #     return None
        result = self._get_best_result(trials)

        assert result.config is not None
        infer = InferTrainTest(self.variables, Path(result.path))
        infer.handle_checkpoint(
            self.get_best_checkpoint(result), result.config['train_loop_config']
        )


def get_scheduler(sorter: MetricSorter):
    return ASHAScheduler(
        metric=sorter.metric,
        mode=sorter.mode,
        max_t=EPOCHS,
        grace_period=EPOCHS // 2,
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
            search_alg=HyperOptSearch(metric=sorter.metric, mode=sorter.mode),
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
        # ResumeConfig is kind of buggy, should be used with caution
        # It also has wrong type hints, so we need to ignore them
        resume_config = ResumeConfig(
            finished=ResumeConfig.ResumeType.SKIP,  # type: ignore
            unfinished=ResumeConfig.ResumeType.RESTART,  # type: ignore
            errored=ResumeConfig.ResumeType.RESTART,  # type: ignore
        )
        restored = tune.Tuner.restore(
            trial_dir.as_posix(),
            trainable=get_trainable(),  # type: ignore
            param_space=get_param_space(),
            _resume_config=resume_config,
        )
        return restored
    else:
        restored = tune.Tuner(
            get_trainable(),
            param_space=get_param_space(),
            tune_config=get_tune_config(),
        )
    return restored
