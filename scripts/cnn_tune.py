from __future__ import annotations

import ray
import ray.util
import torch

from landnet.config import (
    CPUS,
    GPUS,
    TRIAL_NAME,
)
from landnet.features.grids import GeomorphometricalVariable
from landnet.logger import create_logger
from landnet.modelling.models import *
from landnet.modelling.train import (
    LandslideImagesCacher,
    train_model,
)
from landnet.modelling.tune import MetricSorter

ray.init(num_cpus=CPUS, num_gpus=GPUS)
torch.cuda.ipc_collect()
torch.cuda.empty_cache()


logger = create_logger(__name__)


def train_models(cacher: ray.ObjectRef[LandslideImagesCacher]):  # type: ignore
    sorter = MetricSorter('val_f_beta', 'max')
    train_model(
        [
            # GeomorphometricalVariable('area'),
            # GeomorphometricalVariable('cbl'),
            # GeomorphometricalVariable('ccros'),
            # GeomorphometricalVariable('cdl'),
            GeomorphometricalVariable('cdo'),
            GeomorphometricalVariable('cgene'),
            # GeomorphometricalVariable('clong'),
            # GeomorphometricalVariable('clo'),
            GeomorphometricalVariable('clu'),
            # GeomorphometricalVariable('cmaxi'),
            # GeomorphometricalVariable('cmini'),
            GeomorphometricalVariable('nego'),
            # GeomorphometricalVariable('northness'),
            # GeomorphometricalVariable('poso'),
            # GeomorphometricalVariable('shade'),
            GeomorphometricalVariable('slope'),
            # GeomorphometricalVariable('cprof'),
            # GeomorphometricalVariable('croto'),
            # GeomorphometricalVariable('ctang'),
            # GeomorphometricalVariable('cup'),
            # GeomorphometricalVariable('dem'),
            # GeomorphometricalVariable('eastness'),
            # GeomorphometricalVariable('tpi'),
            # GeomorphometricalVariable('tri'),
            # GeomorphometricalVariable('wind'),
        ],
        TRIAL_NAME,
        cacher,
        sorter,
    )


if __name__ == '__main__':
    cacher = LandslideImagesCacher.remote()
    train_models(cacher)
