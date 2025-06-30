from __future__ import annotations

import ray
import ray.util

from landnet.config import CPUS, GPUS, TRIAL_NAME
from landnet.enums import GeomorphometricalVariable
from landnet.logger import create_logger
from landnet.modelling import torch_clear
from landnet.modelling.classification.models import *
from landnet.modelling.classification.train import (
    LandslideImageClassificationCacher,
    train_model,
)
from landnet.modelling.tune import MetricSorter

ray.init(num_cpus=CPUS, num_gpus=GPUS)
if GPUS:
    torch_clear()


logger = create_logger(__name__)


def train_models():  # type: ignore
    sorter = MetricSorter('val_f2_score', 'max')
    # vars = [
    #     GeomorphometricalVariable.HILLSHADE,
    #     GeomorphometricalVariable.TOPOGRAPHIC_POSITION_INDEX,
    #     GeomorphometricalVariable.NEGATIVE_TOPOGRAPHIC_OPENNESS,
    #     GeomorphometricalVariable.DIGITAL_ELEVATION_MODEL,
    #     GeomorphometricalVariable.EASTNESS,
    #     GeomorphometricalVariable.SLOPE,
    #     GeomorphometricalVariable.REAL_SURFACE_AREA,
    #     GeomorphometricalVariable.FLOW_LINE_CURVATURE,
    #     GeomorphometricalVariable.TERRAIN_RUGGEDNESS_INDEX,
    #     GeomorphometricalVariable.LOCAL_CURVATURE,
    # ]
    # for var in vars:
    #     cacher = LandslideImageClassificationCacher.remote()  # type: ignore
    #     train_model([var], var.value, cacher, sorter)

    cacher = LandslideImageClassificationCacher.remote()  # type: ignore
    train_model(list(GeomorphometricalVariable), TRIAL_NAME, cacher, sorter)
    # train_model(
    #     [
    #         GeomorphometricalVariable.HILLSHADE,
    #         GeomorphometricalVariable.TOPOGRAPHIC_POSITION_INDEX,
    #         GeomorphometricalVariable.NEGATIVE_TOPOGRAPHIC_OPENNESS,
    #         GeomorphometricalVariable.DIGITAL_ELEVATION_MODEL,
    #         GeomorphometricalVariable.EASTNESS,
    #         GeomorphometricalVariable.SLOPE,
    #         GeomorphometricalVariable.REAL_SURFACE_AREA,
    #         GeomorphometricalVariable.FLOW_LINE_CURVATURE,
    #         GeomorphometricalVariable.TERRAIN_RUGGEDNESS_INDEX,
    #         GeomorphometricalVariable.LOCAL_CURVATURE,
    #     ],
    #     TRIAL_NAME,
    #     cacher,
    #     sorter,
    # )


if __name__ == '__main__':
    train_models()
