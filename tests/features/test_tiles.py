from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest

from landnet.enums import Mode
from landnet.features.grids import Grid
from landnet.features.tiles import TileConfig, TileSize

DATA_DIR = Path(__file__).parent.parent / 'data'
LANDSLIDES = gpd.read_file(DATA_DIR / 'test_landslides.fgb')
TILES = gpd.read_file(DATA_DIR / 'test_tiles.fgb')
GRID = DATA_DIR / 'test_grid.tif'


def test_get_masked_tile():
    expected_shape = (1, 25, 25)
    config = TileConfig(
        TileSize(expected_shape[1], expected_shape[2]), overlap=0
    )
    grid = Grid(GRID, config, landslides=LANDSLIDES.geometry)
    _, array, _ = grid.get_masked_tile(399, Mode.TRAIN)  # has landslides
    assert 0 in array
    assert 1 in array
    assert array.shape == expected_shape
    _, array, _ = grid.get_masked_tile(
        0, Mode.TRAIN
    )  # does not have landslides
    assert (array == 0).all()
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
