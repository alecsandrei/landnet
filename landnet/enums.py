from __future__ import annotations

from enum import Enum


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
    INDEX_OF_CONVERGENCE = 'ioc'
    TERRAIN_SURFACE_CONVEXITY = 'conv'
    POSITIVE_TOPOGRAPHIC_OPENNESS = 'poso'
    NEGATIVE_TOPOGRAPHIC_OPENNESS = 'nego'
    ASPECT = 'aspect'
    NORTHNESS = 'northness'
    EASTNESS = 'eastness'
    PROFILE_CURVATURE = 'cprof'
    PLAN_CURVATURE = 'cplan'
    GENERAL_CURVATURE = 'cgene'
    FLOW_LINE_CURVATURE = 'croto'
    TANGENTIAL_CURVATURE = 'ctang'
    LONGITUDINAL_CURVATURE = 'clong'
    CROSS_SECTIONAL_CURVATURE = 'ccros'
    MINIMAL_CURVATURE = 'cmini'
    MAXIMAL_CURVATURE = 'cmaxi'
    TOTAL_CURVATURE = 'ctota'
    DIGITAL_ELEVATION_MODEL = 'dem'
    REAL_SURFACE_AREA = 'area'
    TOPOGRAPHIC_POSITION_INDEX = 'tpi'
    VALLEY_DEPTH = 'vld'
    TERRAIN_RUGGEDNESS_INDEX = 'tri'
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


class Architecture(Enum):
    ALEXNET = 'alexnet'
    RESNET50 = 'resnet50'
    CONVNEXT = 'convnext'
    CONVNEXTKAN = 'convnextkan'
    RESNET50KAN = 'resnet50kan'
    DEEPLABV3 = 'deeplabv3'
    FCN = 'fcn'
