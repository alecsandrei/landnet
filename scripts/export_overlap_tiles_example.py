from __future__ import annotations

from geopandas import gpd

from landnet.config import (
    DEFAULT_CLASS_BALANCE,
    EPSG,
    GRIDS,
    INTERIM_DATA_DIR,
    OVERLAP,
)
from landnet.enums import GeomorphometricalVariable, Mode
from landnet.features.grids import Grid
from landnet.features.tiles import TileConfig, TileSize
from landnet.modelling.classification.dataset import (
    LandslideImageClassification,
    create_classification_dataloader,
)

if __name__ == '__main__':
    tile_config = TileConfig(TileSize(100, 100), OVERLAP)
    grid = Grid(
        (
            GRIDS
            / Mode.VALIDATION.value
            / GeomorphometricalVariable.SLOPE.value
        ).with_suffix('.tif'),
        tile_config,
        mode=Mode.VALIDATION,
    )

    landslide_image_classification = LandslideImageClassification(
        grid, mode=Mode.VALIDATION
    )
    dataloader = create_classification_dataloader(
        landslide_image_classification,
        batch_size=1,
        # num_workers=4,
        # pin_memory=True,
        size=500,
        class_balance=DEFAULT_CLASS_BALANCE,
    )
    for _ in dataloader:
        ...
    all_tiles = gpd.GeoSeries(
        [grid.get_tile_bounds(i)[2] for i in range(grid.get_tiles_length())],
        crs=EPSG,
    )
    all_tiles.to_file(
        INTERIM_DATA_DIR / f'tiles_overlap_{OVERLAP}_validation.fgb'
    )
    subset = gpd.GeoSeries(
        [
            grid.get_tile_bounds(i)[2]
            for i in landslide_image_classification.sampled_indices
        ],
        crs=EPSG,
    )
    subset.to_file(
        INTERIM_DATA_DIR / f'tiles_overlap_{OVERLAP}_validation_subset.fgb'
    )
