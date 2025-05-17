from __future__ import annotations

import os
import typing as t

if t.TYPE_CHECKING:
    from torch.utils.data import Subset

    from landnet.features.grids import Grid
    from landnet.features.tiles import TileConfig
    from landnet.modelling.classification.dataset import (
        ConcatLandslideImageClassification,
        LandslideImageClassification,
    )
    from landnet.modelling.segmentation.dataset import (
        ConcatLandslideImageSegmentation,
        LandslideImageSegmentation,
    )

Metadata = dict[str, t.Any]
PathLike = os.PathLike | str

type LandslideClassifiicationDataset = (
    LandslideImageClassification | ConcatLandslideImageClassification
)
type AnyLandslideClassificationDataset = (
    LandslideClassifiicationDataset | Subset[LandslideClassifiicationDataset]
)
type ClassificationTrainTestValidation = tuple[
    AnyLandslideClassificationDataset,
    AnyLandslideClassificationDataset,
    LandslideClassifiicationDataset,
]

type LandslideSegmentationDataset = (
    LandslideImageSegmentation | ConcatLandslideImageSegmentation
)
type AnyLandslideSegmentationDataset = (
    LandslideSegmentationDataset | Subset[LandslideSegmentationDataset]
)
type SegmentationTrainTestValidation = tuple[
    AnyLandslideSegmentationDataset,
    AnyLandslideSegmentationDataset,
    LandslideSegmentationDataset,
]


class GridTypes(t.TypedDict):
    train: Grid
    test: Grid


class TileProperties(t.TypedDict):
    path: str
    mode: t.NotRequired[t.Literal['train', 'test']]
    landslide_density: t.NotRequired[float]


class Geometry(t.TypedDict):
    type: str
    coordinates: list


class Feature(t.TypedDict):
    type: str
    properties: TileProperties
    geometry: Geometry
    bbox: t.NotRequired[list[float]]


class CRSProperties(t.TypedDict):
    name: str


class CRS(t.TypedDict):
    type: str
    properties: CRSProperties


class GeoJSON(t.TypedDict):
    """The GeoJSON representation of the DEM tile bounds."""

    type: str
    crs: t.NotRequired[CRS]
    features: list[Feature]


class TuneSpace(t.TypedDict):
    batch_size: int
    tile_config: TileConfig
    learning_rate: t.NotRequired[float]
