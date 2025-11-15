from __future__ import annotations

import os
import typing as t

from landnet.enums import GeomorphometricalVariable

if t.TYPE_CHECKING:
    import collections.abc as c

    from torch.utils.data import Subset

    from landnet.enums import Mode
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

type LandslideClassificationDataset = (
    LandslideImageClassification | ConcatLandslideImageClassification
)
type AnyLandslideClassificationDataset = (
    LandslideClassificationDataset | Subset[LandslideClassificationDataset]
)
type ClassificationTrainTestValidation = tuple[
    AnyLandslideClassificationDataset,
    LandslideClassificationDataset,
    LandslideClassificationDataset | None,
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

type AnyLandslideImages = (
    LandslideImageClassification | LandslideImageSegmentation
)

type CachedImages[T: AnyLandslideImages] = c.MutableMapping[
    GeomorphometricalVariable, c.MutableMapping[Mode, T]
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
    learning_rate: t.NotRequired[float]
    batch_size: int
    tile_config: t.NotRequired[TileConfig]
