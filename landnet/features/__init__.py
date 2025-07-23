from __future__ import annotations

import os
import typing as t
from pathlib import Path

import geopandas as gpd
from PySAGA_cmd.saga import SAGA

from landnet.config import (
    DEM_TILES,
    GRIDS,
    INFERENCE_TILES,
    OVERLAP,
    TEST_TILES,
    TRAIN_TILES,
    VALIDATION_TILES,
)
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
from landnet.logger import create_logger

if t.TYPE_CHECKING:
    from landnet.typing import GridTypes

logger = create_logger(__name__)


def get_merged_dem(mode: Mode, out_dir: Path | None = None) -> Path:
    tiles = t.cast(gpd.GeoDataFrame, get_dem_tiles()).dropna(subset='mode')
    os.makedirs((GRIDS / mode.value), exist_ok=True)
    dir_map = {
        Mode.TRAIN: TRAIN_TILES,
        Mode.TEST: TEST_TILES,
        Mode.INFERENCE: INFERENCE_TILES,
        Mode.VALIDATION: VALIDATION_TILES,
    }
    if out_dir is None:
        out_dir = dir_map[mode] / 'dem' / '100x100'
    tiles = RasterTiles(tiles[tiles.loc[:, 'mode'] == mode.value], DEM_TILES)
    logger.debug('Resampling %r for %s' % (tiles, mode))
    resampled = tiles.resample(out_dir, mode)
    logger.debug('Merging %r for %s' % (resampled, mode))
    return resampled.merge(GRIDS / mode.value / 'dem.tif')


def main(tile_size: TileSize) -> dict[GeomorphometricalVariable, GridTypes]:
    saga = SAGA('saga_cmd')
    compute_grids(get_merged_dem(Mode.TRAIN), Mode.TRAIN, saga)
    compute_grids(get_merged_dem(Mode.TEST), Mode.TEST, saga)
    compute_grids(get_merged_dem(Mode.VALIDATION), Mode.VALIDATION, saga)
    return get_grid_types(TileConfig(tile_size, overlap=OVERLAP))


if __name__ == '__main__':
    main(TileSize(100, 100))
