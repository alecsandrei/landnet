from __future__ import annotations

import collections.abc as c
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
from PySAGA_cmd import SAGA

from landnet.config import DEM_TILES, GRIDS, INFERENCE_TILES
from landnet.dataset import get_dem_tiles
from landnet.enums import Mode
from landnet.features.grids import (
    GeomorphometricalVariable,
    TerrainAnalysis,
)
from landnet.features.tiles import Grid, RasterTiles, TileConfig, TileSize
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
        mode=mode,
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
    logger.debug('Resampling %r for %sing' % (tiles, Mode.INFERENCE))
    resampled = tiles.resample(resample_dir, Mode.INFERENCE)
    logger.debug('Merging %r for %sing' % (resampled, Mode.INFERENCE))
    merged = resampled.merge(out_dir / 'dem.tif')
    compute_grids_for_dem(merged, saga, Mode.INFERENCE, variables)
    for grid in out_dir.glob('*.tif'):
        logger.info('Masking %s' % grid)
        Grid(grid, TileConfig(TileSize(100, 100))).mask(
            mask_geometry, overwrite=True
        )


if __name__ == '__main__':
    # Load DEM tiles
    saga = SAGA('saga_cmd')
    variables = [
        GeomorphometricalVariable.DOWNSLOPE_CURVATURE,
        GeomorphometricalVariable.GENERAL_CURVATURE,
        GeomorphometricalVariable.LOCAL_UPSLOPE_CURVATURE,
        GeomorphometricalVariable.NEGATIVE_TOPOGRAPHIC_OPENNESS,
        GeomorphometricalVariable.SLOPE,
    ]
    dem_tiles = get_dem_tiles()
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

    # dem_tiles['grouper'] = .str.get(
    #     0
    # ) + dem_tiles['path'].str.split('/').str.get(1)
    # for group in dem_tiles.groupby('grouper'):
    #     # Generate fishnet grid based on DEM extent
    #     fishnet = Fishnet(
    #         box(*dem_tiles.total_bounds), rows=10, cols=10
    #     ).generate_grid()
    #     grid = gpd.GeoSeries(fishnet)

    #     # Create figure and axis
    #     fig, ax = plt.subplots(figsize=(8, 8))

    #     # Plot DEM tiles (filled polygons for better visualization)
    #     group[1].plot(
    #         ax=ax,
    #         facecolor='lightgray',
    #         edgecolor='black',
    #         linewidth=0.5,
    #         alpha=0.6,
    #         label='DEM Tiles',
    #     )

    #     # Plot Fishnet Grid
    #     grid.plot(
    #         ax=ax,
    #         facecolor='none',
    #         edgecolor='blue',
    #         linewidth=1,
    #         linestyle='--',
    #         label='Fishnet Grid',
    #     )

    #     # Add a border around the entire grid
    #     full_extent = gpd.GeoSeries(box(*grid.total_bounds))
    #     full_extent.plot(
    #         ax=ax,
    #         facecolor='none',
    #         edgecolor='red',
    #         linewidth=2,
    #         linestyle='--',
    #         label='Grid Border',
    #     )

    #     # Improve aesthetics
    #     ax.set_title(
    #         'DEM Tiles and Fishnet Grid', fontsize=14, fontweight='bold'
    #     )
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_frame_on(False)  # Remove default axis frame
    #     ax.legend()

    #     plt.show()
