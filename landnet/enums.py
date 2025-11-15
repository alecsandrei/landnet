from __future__ import annotations

import typing as t
from enum import Enum

if t.TYPE_CHECKING:
    from pathlib import Path


class LandslideClass(Enum):
    NO_LANDSLIDE = 0
    LANDSLIDE = 1


class Mode(Enum):
    TRAIN = 'train'
    TEST = 'test'
    VALIDATION = 'validation'
    INFERENCE = 'inference'


class GeomorphometricalVariable(Enum):
    SLOPE = 'slope'
    HILLSHADE = 'shade'
    TERRAIN_RUGGEDNESS_INDEX = 'tri'
    POSITIVE_TOPOGRAPHIC_OPENNESS = 'poso'
    NEGATIVE_TOPOGRAPHIC_OPENNESS = 'nego'
    TOPOGRAPHIC_POSITION_INDEX = 'tpi'
    PROFILE_CURVATURE = 'cprof'
    GENERAL_CURVATURE = 'cgene'
    INDEX_OF_CONVERGENCE = 'ioc'
    TERRAIN_SURFACE_CONVEXITY = 'conv'
    ASPECT = 'aspect'
    NORTHNESS = 'northness'
    EASTNESS = 'eastness'
    PLAN_CURVATURE = 'cplan'
    FLOW_LINE_CURVATURE = 'croto'
    TANGENTIAL_CURVATURE = 'ctang'
    LONGITUDINAL_CURVATURE = 'clong'
    CROSS_SECTIONAL_CURVATURE = 'ccros'
    MINIMAL_CURVATURE = 'cmini'
    MAXIMAL_CURVATURE = 'cmaxi'
    TOTAL_CURVATURE = 'ctota'
    DIGITAL_ELEVATION_MODEL = 'dem'
    REAL_SURFACE_AREA = 'area'
    VALLEY_DEPTH = 'vld'
    VECTOR_RUGGEDNESS_MEASURE = 'vrm'
    LOCAL_CURVATURE = 'clo'
    UPSLOPE_CURVATURE = 'cup'
    LOCAL_UPSLOPE_CURVATURE = 'clu'
    DOWNSLOPE_CURVATURE = 'cdo'
    LOCAL_DOWNSLOPE_CURVATURE = 'cdl'
    FLOW_ACCUMULATION = 'flow'
    FLOW_PATH_LENGTH = 'fpl'
    SLOPE_LENGTH = 'spl'
    CELL_BALANCE = 'cbl'
    TOPOGRAPHIC_WETNESS_INDEX = 'twi'
    WIND_EXPOSITION_INDEX = 'wind'

    @classmethod
    def parse_file(cls, path: Path) -> list[GeomorphometricalVariable]:
        with path.open(mode='r') as file:
            return [
                cls._member_map_[variable.strip().split('.')[1]]
                for variable in file.readlines()
            ]


class Architecture(Enum):
    ALEXNET = 'alexnet'
    RESNET50 = 'resnet50'
    CONVNEXT = 'convnext'
    CONVNEXTKAN = 'convnextkan'
    RESNET50KAN = 'resnet50kan'
    DEEPLABV3 = 'deeplabv3'
    FCN = 'fcn'
    UNET = 'unet'
