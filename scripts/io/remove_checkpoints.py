"""Removes all checkpoints other than the best one for all experiments."""

from __future__ import annotations

import shutil
from pathlib import Path

from ray.tune import ExperimentAnalysis, ResultGrid

from landnet.config import EXPERIMENTS_NAME, MODELS_DIR
from landnet.modelling.tune import MetricSorter

if __name__ == '__main__':
    metric = MetricSorter('val_f2_score', 'max')
    experiments = []
    experiments_dir = MODELS_DIR / EXPERIMENTS_NAME
    best_result_paths = []
    for experiment_dir in experiments_dir.iterdir():
        if not experiment_dir.is_dir():
            continue

        experiment = ExperimentAnalysis(
            experiment_dir,
            default_metric=metric.metric,
            default_mode=metric.mode,
        )
        best_result = ResultGrid(experiment).get_best_result(
            metric=metric.metric, mode=metric.mode, scope='all'
        )
        best_checkpoint_path = (
            Path(
                best_result.get_best_checkpoint(
                    metric=metric.metric, mode=metric.mode
                ).path
            )
            / 'checkpoint.ckpt'
        )
        for checkpoint in experiment_dir.rglob('*.ckpt'):
            if checkpoint == best_checkpoint_path:
                print('will not delete', checkpoint)
                continue
            shutil.rmtree(checkpoint.parent, ignore_errors=False)
