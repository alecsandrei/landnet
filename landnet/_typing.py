from __future__ import annotations

import os
import typing as t

if t.TYPE_CHECKING:
    from landnet.features.tiles import LandslideImages

Metadata = dict[str, t.Any]
PathLike = os.PathLike | str


class ImageFolders(t.TypedDict):
    train: LandslideImages
    test: LandslideImages


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
