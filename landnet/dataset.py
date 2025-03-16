from __future__ import annotations

import os
import typing as t

import geopandas as gpd

from landnet.config import EPSG, RAW_DATA_DIR

PathLike = os.PathLike | str
Mode = t.Literal['train', 'test']


class TileProperties(t.TypedDict):
    path: str
    mode: t.NotRequired[Mode]
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


def get_empty_geojson() -> GeoJSON:
    return {
        'type': 'FeatureCollection',
        'crs': {'type': 'name', 'properties': {'name': f'EPSG:{EPSG}'}},
        'features': [],
    }


def geojson_to_gdf(geojson: GeoJSON) -> gpd.GeoDataFrame:
    crs = None
    if (crs_dict := geojson.get('crs', None)) is not None:
        crs = crs_dict['properties']['name']
    return gpd.GeoDataFrame.from_features(geojson['features'], crs=crs)


def read_landslide_shapes(mode: Mode) -> gpd.GeoSeries:
    return gpd.read_file(
        RAW_DATA_DIR / 'shapes.gpkg',
        layer='landslides_train' if mode == 'train' else 'landslides_test',
    ).geometry
