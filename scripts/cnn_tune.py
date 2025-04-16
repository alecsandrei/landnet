from __future__ import annotations

import ray
import ray.util

from landnet.config import (
    CPUS,
    GPUS,
)
from landnet.features.grids import GeomorphometricalVariable
from landnet.logger import create_logger
from landnet.modelling import torch_clear
from landnet.modelling.models import *
from landnet.modelling.train import (
    LandslideImagesCacher,
    train_model,
)
from landnet.modelling.tune import MetricSorter

ray.init(num_cpus=CPUS, num_gpus=GPUS)
if GPUS:
    torch_clear()


logger = create_logger(__name__)


def train_models():  # type: ignore
    sorter = MetricSorter('val_f_beta', 'max')
    for var in GeomorphometricalVariable:
        cacher = LandslideImagesCacher.remote()  # type: ignore
        train_model([var], var.value, cacher, sorter)
    # train_model(
    #     [
    #         # GeomorphometricalVariable('area'),
    #         # GeomorphometricalVariable('cbl'),
    #         # GeomorphometricalVariable('ccros'),
    #         # GeomorphometricalVariable('cdl'),
    #         # GeomorphometricalVariable('cdo'),
    #         # GeomorphometricalVariable('cgene'),
    #         # GeomorphometricalVariable('clong'),
    #         # GeomorphometricalVariable('clo'),
    #         # GeomorphometricalVariable('clu'),
    #         # GeomorphometricalVariable('cmaxi'),
    #         # GeomorphometricalVariable('cmini'),
    #         # GeomorphometricalVariable('nego'),
    #         # GeomorphometricalVariable('northness'),
    #         # GeomorphometricalVariable('poso'),
    #         # GeomorphometricalVariable('shade'),
    #         GeomorphometricalVariable('slope'),
    #         # GeomorphometricalVariable('cprof'),
    #         # GeomorphometricalVariable('croto'),
    #         # GeomorphometricalVariable('ctang'),
    #         # GeomorphometricalVariable('cup'),
    #         # GeomorphometricalVariable('dem'),
    #         # GeomorphometricalVariable('eastness'),
    #         # GeomorphometricalVariable('tpi'),
    #         # GeomorphometricalVariable('tri'),
    #         # GeomorphometricalVariable('wind'),
    #     ],
    #     TRIAL_NAME,
    #     cacher,
    #     sorter,
    # )


if __name__ == '__main__':
    train_models()
