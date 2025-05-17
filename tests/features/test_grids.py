from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
import rasterio

from landnet.enums import Mode
from landnet.features.grids import Grid
from landnet.features.tiles import TileConfig, TileSize

DATA_DIR = Path(__file__).parent.parent / 'data'
LANDSLIDES = gpd.read_file(DATA_DIR / 'test_landslides.fgb')
TILES = gpd.read_file(DATA_DIR / 'test_tiles.fgb')
GRID = DATA_DIR / 'test_grid.tif'


def test_get_masked_tile():
    expected_shape = (2, 25, 25)
    config = TileConfig(
        TileSize(expected_shape[1], expected_shape[2]), overlap=0
    )
    grid = Grid(GRID, config, landslides=LANDSLIDES.geometry)
    _, array, _ = grid.get_tile_mask(399, Mode.TRAIN)  # has landslides
    assert array.shape == expected_shape
    assert 1 in array[1, :, :]
    _, array, _ = grid.get_tile_mask(0, Mode.TRAIN)  # does not have landslides
    assert (array[0, :, :] == 1).all()
    assert array.shape == expected_shape


@pytest.mark.dependency()
def test_get_tile():
    expected_shape = (1, 25, 25)
    config = TileConfig(
        TileSize(expected_shape[1], expected_shape[2]), overlap=0
    )
    grid = Grid(GRID, config, landslides=LANDSLIDES.geometry)
    _, array, _ = grid.get_tile(0)
    assert array.shape == expected_shape


@pytest.mark.dependency(depends=['test_get_tile'])
def test_get_tiles():
    expected_shape = (1, 25, 25)
    config = TileConfig(
        TileSize(expected_shape[1], expected_shape[2]), overlap=0
    )
    grid = Grid(GRID, config, landslides=LANDSLIDES.geometry)
    for metadata, array in grid.get_tiles():
        assert array.shape == expected_shape


@pytest.mark.dependency(depends=['test_get_tile'])
def test_write_tile(tmp_path: Path):
    expected_shape = (1, 25, 25)
    config = TileConfig(
        TileSize(expected_shape[1], expected_shape[2]), overlap=0
    )
    grid = Grid(GRID, config, landslides=LANDSLIDES.geometry)
    _, tile_array, _ = grid.get_tile(0)
    out_file = grid.write_tile(0, tile_array, out_dir=tmp_path)
    with rasterio.open(out_file) as src:
        array = src.read(1)
        assert array.shape == tile_array.shape[1:]
