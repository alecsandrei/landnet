from __future__ import annotations

import os
import typing as t

import geopandas as gpd
from PySAGA_cmd.saga import SAGA, Version

from landnet.config import DEM_TILES, GRIDS
from landnet.dataset import get_dem_tiles
from landnet.features.grids import GeomorphometricalVariable, compute_grids
from landnet.features.tiles import RasterTiles, TileSize, get_image_folders

if t.TYPE_CHECKING:
    from landnet.features.tiles import ImageFolders

Mode = t.Literal['train', 'test']


def get_raster_tiles(mode: Mode) -> RasterTiles:
    tiles = t.cast(gpd.GeoDataFrame, get_dem_tiles()).dropna(subset='mode')
    os.makedirs((GRIDS / mode), exist_ok=True)
    return RasterTiles(tiles[tiles.loc[:, 'mode'] == mode], DEM_TILES)


def main(tile_size: TileSize) -> dict[GeomorphometricalVariable, ImageFolders]:
    saga = SAGA('saga_cmd', Version(9, 8, 0))
    compute_grids(get_raster_tiles('train'), 'train', saga)
    compute_grids(get_raster_tiles('test'), 'test', saga)
    return get_image_folders(tile_size)
