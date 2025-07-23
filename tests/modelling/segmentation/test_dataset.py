from __future__ import annotations

import collections.abc as c
import math

from landnet.enums import GeomorphometricalVariable, Mode
from landnet.features.grids import get_grid_for_variable
from landnet.features.tiles import TileConfig, TileSize
from landnet.modelling import torch_clear
from landnet.modelling.dataset import (
    get_default_mask_transform,
    get_default_transform,
)
from landnet.modelling.segmentation.dataset import (
    ConcatLandslideImageSegmentation,
    LandslideImageSegmentation,
    get_weighted_segmentation_dataloader,
)

torch_clear()


def get_concat_landslide_image_segmentation(
    variables: c.Sequence[GeomorphometricalVariable],
) -> ConcatLandslideImageSegmentation:
    tile_config = TileConfig(TileSize(100, 100), overlap=5)
    train_grids = [
        get_grid_for_variable(
            variable,
            tile_config=tile_config,
            mode=Mode.TRAIN,
        )
        for variable in variables
    ]

    dataset = ConcatLandslideImageSegmentation(
        landslide_images=[
            LandslideImageSegmentation(
                grid,
                Mode.TRAIN,
                transform=get_default_transform(),
                mask_transform=get_default_mask_transform(),
            )
            for grid in train_grids
        ],
        augment_transform=None,
    )
    return dataset


def test_get_dataloader():
    variables = [
        GeomorphometricalVariable.SLOPE,
        GeomorphometricalVariable.REAL_SURFACE_AREA,
        GeomorphometricalVariable.PROFILE_CURVATURE,
    ]
    size = 501
    batch_size = 5
    dataset = get_concat_landslide_image_segmentation(variables)
    dataloader = get_weighted_segmentation_dataloader(
        dataset, size=size, batch_size=batch_size
    )
    assert len(dataloader) == math.ceil(size / batch_size)
    for batch in dataloader:
        images, masks = batch
        assert images.shape[0] == batch_size
        assert masks.shape[0] == batch_size
        assert images.shape[1] == len(variables)
        assert masks.shape[1] == 2
        break


class TestLandslideImageSegmentation:
    def test_get_tile(self):
        variables = [
            GeomorphometricalVariable.SLOPE,
            GeomorphometricalVariable.REAL_SURFACE_AREA,
            GeomorphometricalVariable.PROFILE_CURVATURE,
        ]
        dataset = get_concat_landslide_image_segmentation(variables)
        dataset.landslide_images[0]._get_tile(0)
