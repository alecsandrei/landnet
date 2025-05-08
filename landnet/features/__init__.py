from __future__ import annotations

import os
import typing as t

import geopandas as gpd
from PySAGA_cmd.saga import SAGA

from landnet.config import DEM_TILES, GRIDS, OVERLAP
from landnet.enums import Mode
from landnet.features.dataset import get_dem_tiles
from landnet.features.grids import (
    GeomorphometricalVariable,
    compute_grids,
    get_grid_types,
)
from landnet.features.tiles import (
    RasterTiles,
    TileConfig,
    TileSize,
)

if t.TYPE_CHECKING:
    from landnet._typing import GridTypes


def get_raster_tiles(mode: Mode) -> RasterTiles:
    tiles = t.cast(gpd.GeoDataFrame, get_dem_tiles()).dropna(subset='mode')
    os.makedirs((GRIDS / mode.value), exist_ok=True)
    return RasterTiles(tiles[tiles.loc[:, 'mode'] == mode], DEM_TILES)


def main(tile_size: TileSize) -> dict[GeomorphometricalVariable, GridTypes]:
    saga = SAGA('saga_cmd')
    compute_grids(get_raster_tiles(Mode.TRAIN), Mode.TRAIN, saga)
    compute_grids(get_raster_tiles(Mode.TEST), Mode.TEST, saga)
    return get_grid_types(TileConfig(tile_size, overlap=OVERLAP))
