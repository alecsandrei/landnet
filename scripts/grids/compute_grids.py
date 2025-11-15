from __future__ import annotations

import collections.abc as c
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
from PySAGA_cmd import SAGA

from landnet.config import DEM_TILES, GRIDS, INFERENCE_TILES
from landnet.enums import GeomorphometricalVariable, Mode
from landnet.features.dataset import get_dem_tiles
from landnet.features.grids import (
    Grid,
    TerrainAnalysis,
)
from landnet.features.tiles import RasterTiles, TileConfig, TileSize
from landnet.logger import create_logger

logger = create_logger(__name__)


def get_neighbouring_tiles(tiles: gpd.GeoDataFrame, other: gpd.GeoDataFrame):
    other_tiles = dem_tiles[dem_tiles['lat_long'] != lat_long]
    return other_tiles[other_tiles.intersects(tiles.union_all().buffer(50))]


def compute_grids_for_dem(
    dem: Path,
    saga: SAGA,
    mode: Mode,
    variables: c.Sequence[GeomorphometricalVariable] | None = None,
):
    for tool_name, _ in TerrainAnalysis(
        dem,
        saga=saga,
        verbose=False,
        infer_obj_type=False,
        ignore_stderr=False,
        variables=variables,
    ).execute():
        logger.info('%s finished executing for %r' % (tool_name, mode))


def compute_grids(
    tiles: RasterTiles,
    saga: SAGA,
    resample_dir: Path,
    out_dir: Path,
    variables: c.Sequence[GeomorphometricalVariable],
    mask_geometry,
):
    logger.debug('Resampling %r for %sing' % (tiles, Mode.INFERENCE.value))
    resampled = tiles.resample(resample_dir, Mode.INFERENCE)
    logger.debug('Merging %r for %sing' % (resampled, Mode.INFERENCE.value))
    merged = resampled.merge(out_dir / 'dem.tif')
    compute_grids_for_dem(merged, saga, Mode.INFERENCE, variables)
    variable_names = [variable.value for variable in variables]
    for grid in out_dir.glob('*.tif'):
        if grid.stem not in variable_names:
            continue
        logger.info('Masking %s' % grid)
        Grid(grid, TileConfig(TileSize(100, 100)), mode=Mode.INFERENCE).mask(
            mask_geometry, overwrite=True
        )


if __name__ == '__main__':
    # Load DEM tiles
    saga = SAGA('saga_cmd')
    variables = [
        GeomorphometricalVariable.HILLSHADE,
        GeomorphometricalVariable.TOPOGRAPHIC_POSITION_INDEX,
        GeomorphometricalVariable.NEGATIVE_TOPOGRAPHIC_OPENNESS,
        GeomorphometricalVariable.DIGITAL_ELEVATION_MODEL,
        GeomorphometricalVariable.EASTNESS,
        GeomorphometricalVariable.SLOPE,
        GeomorphometricalVariable.REAL_SURFACE_AREA,
        GeomorphometricalVariable.FLOW_LINE_CURVATURE,
        GeomorphometricalVariable.TERRAIN_RUGGEDNESS_INDEX,
        GeomorphometricalVariable.LOCAL_CURVATURE,
    ]
    dem_tiles = get_dem_tiles()
    dem_tiles = dem_tiles[dem_tiles['id1'].astype(int) >= 73]
    path_split = dem_tiles['path'].str.split('/')
    dem_tiles['lat_long'] = path_split.str.get(0) + '/' + path_split.str.get(1)
    for group in dem_tiles.groupby('lat_long'):
        lat_long, group_tiles = group
        group_tiles_with_neighbours = pd.concat(  # type: ignore
            [group_tiles, get_neighbouring_tiles(group_tiles, dem_tiles)],  # type: ignore
            ignore_index=True,
            axis='index',
        )
        print(group_tiles.shape[0], group_tiles_with_neighbours.shape[0])
        tiles = RasterTiles(group_tiles_with_neighbours, DEM_TILES)
        lat, long = tuple(str(lat_long).split('/'))
        out_dir = GRIDS / Mode.INFERENCE.value / lat / long
        resample_dir = INFERENCE_TILES / Mode.INFERENCE.value / lat / long
        os.makedirs(resample_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        compute_grids(
            tiles,
            saga,
            resample_dir,
            out_dir,
            variables,
            group_tiles.union_all(),
        )
