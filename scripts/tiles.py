from __future__ import annotations

import rasterio

from landnet.config import GRIDS, INTERIM_DATA_DIR
from landnet.features.tiles import TileConfig, TileHandler, TileSize

if __name__ == '__main__':
    # grid = Grid(GRIDS / 'train' / 'area.tif')
    # test_dir = INTERIM_DATA_DIR / 'test_dir'
    # test_dir.mkdir(exist_ok=True)
    tile_size = TileSize(100, 100)
    overlap = 10
    # RasterTiles.from_raster(
    #     grid.path, tile_size, overlap=0, out_dir=test_dir, mode='train'
    # )
    # for i in range(grid.get_tiles_length(tile_size.columns, tile_size.rows)):
    #     tile_getter = grid.get_tile(i, tile_size.columns, tile_size.rows)
    # for mode in Mode:
    #     for grid in (GRIDS / mode.value).glob('*.tif'):
    #         out_dir = (
    #             TRAIN_TILES if mode is Mode.TRAIN else TEST_TILES
    #         ) / grid.stem
    #         out_dir.mkdir(exist_ok=True)
    #         tiles = RasterTiles.from_raster(grid, tile_size, 0, out_dir, mode)
    #         no_ldl_dir = out_dir / '0'
    #         no_ldl_dir.mkdir(exist_ok=True)
    #         ldl_dir = out_dir / '1'
    #         ldl_dir.mkdir(exist_ok=True)
    #         for tile in tiles.tiles.itertuples():
    #             path = tiles.parent_dir / str(tile.path)
    #             if float(tile.landslide_density) > 0:
    #                 path.rename(ldl_dir / path.name)
    #             else:
    #                 path.rename(no_ldl_dir / path.name)
    test_dir = INTERIM_DATA_DIR / 'test_tiles2'
    test_dir.mkdir(exist_ok=True)
    with rasterio.open(GRIDS / 'test' / 'shade.tif') as src:
        handler = TileHandler(TileConfig(tile_size, overlap))
        metadata = src.meta.copy()
        for i in range(handler.get_tiles_length(src)):
            tile = handler.get_tile(src, i)

            def handle_window():
                window, transform = tile
                metadata['transform'] = transform
                metadata['width'], metadata['height'] = (
                    window.width,
                    window.height,
                )
                out_filepath = (
                    test_dir / f'{window.col_off}_{window.row_off}_{i}.tif'
                )

                with rasterio.open(out_filepath, 'w', **metadata) as dst:
                    dst.write(src.read(window=window))

            handle_window()

        # print(handler.get_tiles_length(src))
        # breakpoint()
