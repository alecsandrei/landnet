from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import rasterio
import rasterio.windows
import shapely

from landnet.features.tiles import TileConfig, TileHandler, TileSize

DATA_DIR = Path(__file__).parent.parent / 'data'
TILES_100x100_OVERLAP50 = gpd.read_file(
    DATA_DIR / 'tiles_100x100_50overlap.fgb'
)
GRID = DATA_DIR / 'test_grid.tif'


def test_tile_handler():
    config = TileConfig(size=TileSize(100, 100), overlap=50)
    handler = TileHandler(config)
    shapes: set[str] = set()
    with rasterio.open(GRID) as src:
        tiles_gen = handler.get_tiles(src)
        for tile in tiles_gen:
            shape = (
                shapely.box(*rasterio.windows.bounds(tile[0], src.transform))
                .normalize()
                .wkt
            )
            shapes.add(shape)
    tiles: set[str] = set(
        TILES_100x100_OVERLAP50.normalize().to_wkt().to_list()
    )
    assert len(shapes) == len(tiles)
    assert shapes == tiles
