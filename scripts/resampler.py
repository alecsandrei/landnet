from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from landnet.config import DEM_TILES
from landnet.enums import Mode
from landnet.features.tiles import RasterTiles

if __name__ == '__main__':
    # tiles = get_dem_tiles()
    # tiles = tiles[tiles['path'].str.split('/').str.get(0) == '71']
    parent = Path(
        '/media/alex/alex/python-modules-packages-utils/slope-area/data/saveni_aval'
    )
    tiles = gpd.read_file(parent / 'tiles.fgb')
    # raster_tiles = RasterTiles.from_dir(
    #     INTERIM_DATA_DIR / 'DEM' / 'resampled', Mode.TRAIN
    # ).merge(INTERIM_DATA_DIR / 'DEM' / 'resampled' / '71_merged.tif')
    out_dir = parent / 'DEM' / 'resampled'
    raster_tiles = RasterTiles(tiles, DEM_TILES)
    raster_tiles.resample(out_dir, mode=Mode.TRAIN).merge(
        out_dir.parent / 'merged.tif'
    )
