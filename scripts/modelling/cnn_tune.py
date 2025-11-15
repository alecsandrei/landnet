from __future__ import annotations

import pandas as pd
import ray
import ray.util

from landnet.config import CPUS, EXPERIMENTS_NAME, GPUS, MODELS_DIR
from landnet.enums import GeomorphometricalVariable
from landnet.logger import create_logger
from landnet.modelling import torch_clear
from landnet.modelling.classification.models import *
from landnet.modelling.classification.train import train_model
from landnet.modelling.tune import MetricSorter

if GPUS:
    torch_clear()


logger = create_logger(__name__)


def train_models() -> None:  # type: ignore
    sorter = MetricSorter('val_f2_score', 'max')

    results = pd.read_csv(MODELS_DIR / EXPERIMENTS_NAME / 'results.csv')

    sorted = results[['variable', 'val_f2_score']].sort_values(
        'val_f2_score', ascending=False
    )
    sorted = sorted[
        sorted['variable'].isin(
            [var.value for var in GeomorphometricalVariable]
        )
    ]
    ray.init(num_cpus=CPUS, num_gpus=GPUS)
    vars = [GeomorphometricalVariable(var) for var in sorted['variable']]
    train_model(vars, '36_variables_test', sorter)


if __name__ == '__main__':
    train_models()
