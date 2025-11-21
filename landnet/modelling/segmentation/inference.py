from __future__ import annotations

import collections.abc as c
import os
import typing as t
from dataclasses import dataclass

import lightning as pl
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms.v2.functional import InterpolationMode, resize

from landnet.config import ARCHITECTURE, GRIDS, PROCESSED_DATA_DIR
from landnet.enums import GeomorphometricalVariable, Mode
from landnet.features.grids import Grid
from landnet.logger import create_logger
from landnet.modelling.segmentation.dataset import (
    ConcatLandslideImageSegmentation,
    LandslideImageSegmentation,
)
from landnet.modelling.segmentation.lightning import LandslideImageSegmenter
from landnet.modelling.segmentation.models import get_architecture
from landnet.typing import ModelConfig

if t.TYPE_CHECKING:
    from pathlib import Path

logger = create_logger(__name__)


@dataclass
class Infer:
    variables: c.Sequence[GeomorphometricalVariable]
    config: ModelConfig

    def get_grid_for_variable(
        self, variable: GeomorphometricalVariable, mode: Mode
    ) -> Grid:
        grid = (GRIDS / mode.value / variable.value).with_suffix('.tif')
        return Grid(grid, self.config['tile_config'], mode=mode)

    def _get_landslide_images(
        self, variable: GeomorphometricalVariable, mode: Mode
    ) -> LandslideImageSegmentation:
        grid = (GRIDS / mode.value / variable.value).with_suffix('.tif')
        tile_config = self.config['tile_config']
        if tile_config.overlap != 0:
            logger.warning(
                'TileConfig overlap is not 0, this may result in unexpected behaviour'
            )
        # tile_config.overlap = 0  # In inferance mode, this should be 0
        return LandslideImageSegmentation(
            Grid(grid, tile_config, mode=mode), mode=mode
        )

    def handle_checkpoint(
        self, checkpoint_path: os.PathLike | str, model: nn.Module | None = None
    ) -> None:
        if model is None:
            model = get_architecture(ARCHITECTURE)(
                len(self.variables), Mode.INFERENCE
            )
        segmenter = LandslideImageSegmenter.load_from_checkpoint(
            checkpoint_path,
            model=model,
            tune_space=self.config,
            strict=False,
        )
        for mode in (Mode.TRAIN, Mode.TEST, Mode.VALIDATION):
            self._handle_mode(segmenter, mode)

    def _handle_mode(
        self,
        segmenter: LandslideImageSegmenter,
        mode: Mode,
    ):
        images = ConcatLandslideImageSegmentation(
            [
                self._get_landslide_images(variable, mode)
                for variable in self.variables
            ]
        )
        loader = DataLoader(
            images,
            shuffle=False,
            batch_size=self.config['batch_size'],
            num_workers=4,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
        )
        grid = images.landslide_images[0].grid
        output = self._get_predictions(segmenter, loader)
        for i, batch in enumerate(output):
            batch_mask = batch[1]
            for j in range(batch_mask.shape[0]):
                index = i * self.config['batch_size'] + j
                mask = batch_mask.select(0, j)
                resized_mask = resize(
                    mask.unsqueeze(0),
                    [grid.tile_config.size.height, grid.tile_config.size.width],
                    interpolation=InterpolationMode.NEAREST_EXACT,
                )
                self._handle_output_mask(resized_mask.numpy(), index, mode)

    def _handle_output_mask(
        self, mask: np.ndarray, tile_index: int, mode: Mode
    ) -> Path:
        variable = self.variables[0]
        grid = self.get_grid_for_variable(variable, mode)

        out_dir = (
            PROCESSED_DATA_DIR / 'segmentation_results' / mode.name.lower()
        )
        os.makedirs(out_dir, exist_ok=True)
        return grid.write_tile(
            tile_index,
            mask,
            prefix=f'tile{tile_index}_',
            out_dir=out_dir,
        )

    def _get_predictions(
        eslf, segmenter: LandslideImageSegmenter, dataloader: DataLoader
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        trainer = pl.Trainer()
        return trainer.predict(segmenter, dataloader, return_predictions=True)  # type: ignore
