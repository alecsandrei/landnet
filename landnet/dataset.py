from __future__ import annotations

import json
import typing as t
from functools import cache

import geopandas as gpd
from shapely import MultiPolygon, Polygon, from_geojson
from shapely.geometry.base import BaseGeometry

from landnet.config import EPSG, RAW_DATA_DIR
from landnet.enums import Mode

if t.TYPE_CHECKING:
    from pathlib import Path


from landnet._typing import Feature, GeoJSON


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


# @cache
def get_landslide_shapes(mode: Mode) -> gpd.GeoSeries:
    return gpd.read_file(
        RAW_DATA_DIR / 'shapes.gpkg',
        layer='landslides_train' if mode is Mode.TRAIN else 'landslides_test',
    ).geometry


def get_percentage_intersection(
    feature: Polygon | MultiPolygon, other: gpd.GeoSeries
) -> float:
    intersection = other.intersection(feature).union_all()
    intersection_area = intersection.area if not intersection.is_empty else 0
    polygon_area = feature.area
    if polygon_area == 0:
        return 0
    return intersection_area / polygon_area


def feature_to_geojson(feature: dict[str, t.Any]) -> BaseGeometry:
    return from_geojson(json.dumps(feature['geometry']))


def process_feature(
    feature: dict[str, t.Any], mode: Mode, path: Path, parent_dir: Path
) -> Feature:
    keys_to_remove = [
        k
        for k in feature
        if k not in Feature.__required_keys__
        and k not in Feature.__optional_keys__
    ]
    for k in keys_to_remove:
        feature.pop(k)
    feature['properties'] = {
        'path': path.relative_to(parent_dir).as_posix(),
        'mode': mode.value,
        'landslide_density': get_percentage_intersection(
            t.cast(
                Polygon | MultiPolygon,
                feature_to_geojson(feature),
            ),
            get_landslide_shapes(mode=mode),
        ),
    }
    return t.cast(Feature, feature)
