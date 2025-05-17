from __future__ import annotations

import json
import typing as t
from dataclasses import dataclass
from functools import cache

import geopandas as gpd
import numpy as np
from shapely import MultiPolygon, Polygon, box, from_geojson
from shapely.geometry.base import BaseGeometry

from landnet.typing import Feature, GeoJSON
from landnet.config import EPSG, RAW_DATA_DIR
from landnet.enums import Mode

if t.TYPE_CHECKING:
    from pathlib import Path
    from landnet.features.grids import Grid


@dataclass
class Fishnet:
    reference_geometry: Polygon
    rows: int
    cols: int

    def generate_grid(self, buffer: float | None = None) -> list[Polygon]:
        """Generates a grid of polygons within the reference geometry."""
        minx, miny, maxx, maxy = self.reference_geometry.bounds
        cell_width = (maxx - minx) / self.cols
        cell_height = (maxy - miny) / self.rows

        grid = []
        for i in range(self.rows):
            for j in range(self.cols):
                cell_minx = minx + j * cell_width
                cell_maxx = cell_minx + cell_width
                cell_miny = miny + i * cell_height
                cell_maxy = cell_miny + cell_height
                cell = box(cell_minx, cell_miny, cell_maxx, cell_maxy)
                if self.reference_geometry.intersects(cell):
                    if buffer:
                        cell = cell.buffer(buffer, join_style='mitre')
                    grid.append(cell.intersection(self.reference_geometry))

        return grid


def get_empty_geojson() -> GeoJSON:
    return {
        'type': 'FeatureCollection',
        'crs': {'type': 'name', 'properties': {'name': f'EPSG:{EPSG}'}},
        'features': [],
    }


@cache
def get_dem_tiles() -> gpd.GeoDataFrame:
    dem_tiles = gpd.read_file(RAW_DATA_DIR / 'dem_tiles.geojson')
    path_split = dem_tiles['path'].str.split('/')
    dem_tiles['id1'] = path_split.str.get(0)
    dem_tiles['id2'] = path_split.str.get(1)
    return dem_tiles


def get_tile_relative_path(
    tiles: gpd.GeoDataFrame, raster_bounds: Polygon
) -> str:
    return t.cast(str, tiles[tiles.within(raster_bounds)].iloc[0].path)


def geojson_to_gdf(geojson: GeoJSON) -> gpd.GeoDataFrame:
    crs = None
    if (crs_dict := geojson.get('crs', None)) is not None:
        crs = crs_dict['properties']['name']
    return gpd.GeoDataFrame.from_features(geojson['features'], crs=crs)


@cache
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


def logits_to_dem_tiles(
    logits: np.ndarray, y: np.ndarray, mode: Mode, grid: Grid
) -> gpd.GeoDataFrame:
    landslide_density = grid.get_landslide_percentage_intersection(
        list(range(grid.get_tiles_length())), mode=mode
    )
    assert len(y) == landslide_density.shape[0]
    assert len(logits) == landslide_density.shape[0]

    landslide_density['logits'] = logits
    landslide_density['y'] = y
    yhat = logits.round()

    conditions = [
        (yhat == 1) & (y == 1),  # tp
        (yhat == 1) & (y == 0),  # fp
        (yhat == 0) & (y == 0),  # tn
        (yhat == 0) & (y == 1),  # fn
    ]
    labels = np.select(conditions, ['tp', 'fp', 'tn', 'fn'], default='unknown')
    landslide_density['yhat'] = labels

    return landslide_density
