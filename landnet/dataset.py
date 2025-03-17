from __future__ import annotations

import os
import typing as t
from functools import cache

import geopandas as gpd

from landnet.config import EPSG, RAW_DATA_DIR

if t.TYPE_CHECKING:
    from shapely import Polygon

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


@cache
def get_dem_tiles() -> gpd.GeoDataFrame:
    # Respects the GeoJSON type defined in this module
    return gpd.read_file(RAW_DATA_DIR / 'dem_tiles.geojson')


def get_tile_relative_path(
    tiles: gpd.GeoDataFrame, raster_bounds: Polygon
) -> str:
    return t.cast(str, tiles[tiles.within(raster_bounds)].iloc[0].path)


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
