from __future__ import annotations

import collections.abc as c
import json
import os
import typing as t
from pathlib import Path

import rasterio
from rasterio.features import dataset_features

from landnet.config import DEM_TILES, EPSG, INTERIM_DATA_DIR

PathLike = os.PathLike | str
Mode = t.Literal['train', 'test']


class TileProperties(t.TypedDict):
    path: str


class ModelTileProperties(TileProperties):
    mode: Mode
    landslide_density: float


class Geometry(t.TypedDict):
    type: str
    coordinates: c.Sequence


class Feature(t.TypedDict):
    type: str
    properties: TileProperties
    geometry: Geometry
    bbox: t.NotRequired[c.Sequence[float]]


class GeoJSON(t.TypedDict):
    """The GeoJSON representation of the DEM tile bounds."""

    type: str
    crs: t.NotRequired[c.Mapping[str, t.Any]]
    features: c.MutableSequence[Feature]


def get_empty_geojson() -> GeoJSON:
    return {
        'type': 'FeatureCollection',
        'crs': {'type': 'name', 'properties': {'name': f'EPSG:{EPSG}'}},
        'features': [],
    }


def create_tile_bounds_geojson() -> GeoJSON:
    geojson = get_empty_geojson()
    images = Path(DEM_TILES).rglob('*.tif')

    def process_feature(feature: dict[str, t.Any], path: Path) -> Feature:
        keys_to_remove = [
            k
            for k in feature
            if k not in Feature.__required_keys__
            and k not in Feature.__optional_keys__
        ]
        for k in keys_to_remove:
            feature.pop(k)
        feature['properties'] = {
            'path': f'{path.parents[1].stem}/{path.parents[0].stem}/{path.name}'
        }
        return t.cast(Feature, feature)

    def get_features(tif: Path) -> list[Feature]:
        with rasterio.open(tif) as raster:
            features = [
                process_feature(feature, tif)
                for feature in dataset_features(
                    raster,
                    bidx=1,
                    as_mask=True,
                    geographic=False,
                    band=False,
                )
            ]

        return features

    for image in list(images):
        geojson['features'].extend(get_features(image))
    tile_bounds = INTERIM_DATA_DIR / 'tiles.geojson'
    with tile_bounds.open(mode='w') as file:
        json.dump(geojson, file, indent=2)
    return geojson
