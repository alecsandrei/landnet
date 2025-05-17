from __future__ import annotations

import collections.abc as c
import concurrent.futures
import os
import typing as t
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, Lambda, Normalize

from landnet.config import (
    ARCHITECTURE,
    GRIDS,
)
from landnet.enums import GeomorphometricalVariable, LandslideClass, Mode
from landnet.features.dataset import logits_to_dem_tiles
from landnet.features.grids import Grid
from landnet.features.tiles import (
    TileConfig,
)
from landnet.logger import create_logger
from landnet.modelling import torch_clear
from landnet.modelling.classification.dataset import (
    ConcatLandslideImageClassification,
    LandslideImageClassification,
)
from landnet.modelling.classification.lightning import LandslideImageClassifier
from landnet.modelling.classification.models import get_architecture
from landnet.modelling.classification.stats import (
    BinaryClassificationMetricCollection,
)
from landnet.modelling.dataset import ResizeTensor
from landnet.plots import get_confusion_matrix, get_roc_curve
from landnet.utils import save_fig

if t.TYPE_CHECKING:
    from landnet.typing import TuneSpace
    from landnet.modelling.tune import MetricSorter

logger = create_logger(__name__)


@dataclass
class InferTrainTest:
    variables: c.Sequence[GeomorphometricalVariable]
    sorter: MetricSorter
    out_dir: Path

    @staticmethod
    def _get_landslide_images(
        variable: GeomorphometricalVariable, tune_space: TuneSpace, mode: Mode
    ) -> LandslideImageClassification:
        grid = (GRIDS / mode.value / variable.value).with_suffix('.tif')
        tile_config = tune_space['tile_config']
        tile_config.overlap = 0  # In inferance mode, this should be 0
        return LandslideImageClassification(Grid(grid, tile_config), mode=mode)

    def _get_predictions(
        self, classifier: LandslideImageClassifier, dataloader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trainer = pl.Trainer()
        predictions: list[tuple[torch.Tensor, torch.Tensor]] = trainer.predict(
            classifier, dataloader, return_predictions=True
        )  # type: ignore
        logits = torch.cat([prediction[0] for prediction in predictions])
        y = torch.cat([prediction[1] for prediction in predictions])

        return (logits, y)

    @staticmethod
    def _export_metrics(
        out_dir, logits: torch.Tensor, targets: torch.Tensor, mode: Mode
    ):
        metrics = BinaryClassificationMetricCollection()
        metrics.update(logits, targets)
        computed = metrics.compute()
        logger.info('For mode %r, computed metrics %s' % (mode, computed))
        pd.DataFrame.from_dict(computed, orient='index').to_csv(
            out_dir / 'metrics.csv'
        )

    @staticmethod
    def _export_tiles(
        out_dir,
        logits: np.ndarray,
        targets: np.ndarray,
        mode: Mode,
        grid: Grid,
    ):
        tiles = logits_to_dem_tiles(logits, targets, mode, grid)
        tiles.to_file(out_dir / 'tiles.fgb', driver='FlatGeobuf')

    @staticmethod
    def _export_plots(out_dir, logits: np.ndarray, targets: np.ndarray):
        _, ax = plt.subplots()
        curve = get_roc_curve(logits, targets, ax=ax)
        save_fig(out_dir / 'roc_auc_curve.png', curve.figure_)  # type: ignore

        _, ax = plt.subplots()
        get_confusion_matrix(
            logits.round(),
            targets,
            display_labels=(
                LandslideClass.NO_LANDSLIDE.name.replace('_', ' ').capitalize(),
                LandslideClass.LANDSLIDE.name.capitalize(),
            ),
            ax=ax,
        )
        save_fig(out_dir / 'confusion_matrix.png')  # type: ignore

    def _export_predictions(
        self, logits: torch.Tensor, y: torch.Tensor, mode: Mode, grid: Grid
    ) -> None:
        out_dir = self.out_dir / 'predictions' / mode.value
        os.makedirs(out_dir, exist_ok=True)
        logits_np, y_np = (
            logits.cpu().numpy(),
            y.cpu().numpy(),
        )
        self._export_metrics(out_dir, logits, y, mode)
        self._export_tiles(out_dir, logits_np, y_np, mode, grid)
        self._export_plots(out_dir, logits_np, y_np)

    def _handle_mode(
        self,
        classifier: LandslideImageClassifier,
        mode: Mode,
        tune_space: TuneSpace,
    ) -> None:
        images = ConcatLandslideImageClassification(
            [
                self._get_landslide_images(variable, tune_space, mode)
                for variable in self.variables
            ]
        )
        loader = DataLoader(
            images,
            shuffle=False,
            batch_size=tune_space['batch_size'],
            num_workers=4,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
        )
        logits, y = self._get_predictions(classifier, loader)
        self._export_predictions(
            logits, y, mode, images.landslide_images[0].grid
        )

    def handle_checkpoint(
        self, checkpoint_path: os.PathLike | str, tune_space: TuneSpace
    ) -> None:
        torch_clear()
        tune_space_copy = tune_space.copy()
        tune_space_copy['batch_size'] = 1
        classifier = LandslideImageClassifier.load_from_checkpoint(
            checkpoint_path,
            model=get_architecture(ARCHITECTURE)(
                len(self.variables), Mode.INFERENCE
            ),
            config=tune_space_copy,
            strict=False,
        )
        for mode in (Mode.TRAIN, Mode.TEST):
            self._handle_mode(classifier, mode, tune_space)


@dataclass
class InferenceFolder:
    parent: Path
    variables: c.Sequence[GeomorphometricalVariable]
    tile_config: TileConfig
    tiles: gpd.GeoDataFrame

    @property
    def enum_to_grid(self) -> dict[GeomorphometricalVariable, Grid]:
        return {
            variable: Grid(
                (self.parent / variable.value).with_suffix('.tif'),
                self.tile_config,
            )
            for variable in self.variables
        }


class DEMTilesDataset(Dataset):
    def __init__(self, folder: InferenceFolder, transform):
        self.folder = folder
        self.transform = transform
        self.tiles = list(folder.tiles.itertuples())
        self.grids = list(folder.enum_to_grid.values())

    def get_tensor(self, grid: Grid, geometry):
        """Apply grid masking and transformation"""
        array = grid.mask(
            geometry,
            mask_kwargs={'crop': True, 'filled': False},
        )
        return self.transform(torch.from_numpy(array))

    def __getitem__(self, index):
        tile = self.tiles[index]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            tensors = list(
                executor.map(
                    lambda grid: self.get_tensor(grid, tile.geometry),
                    self.grids,
                )
            )

        return torch.cat(tensors)

    def __len__(self):
        return len(self.tiles)


def perform_inference_on_tiles(
    classifier: LandslideImageClassifier,
    folder: InferenceFolder,
    batch_size: int = 128,
    num_workers: int = 4,
) -> np.ndarray:
    transform = get_inference_transform()
    dataset = DEMTilesDataset(folder, transform)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    logits: list[np.ndarray] = []
    classifier.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(classifier.device, non_blocking=True)
            logit = torch.sigmoid(classifier(batch))
            logits.extend(logit.cpu().numpy())
    return np.array(logits)


class Pad:
    def __init__(self, target_size: int):
        self.target_size = target_size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        _, H, W = image.shape

        pad_top = max(0, (self.target_size - H) // 2)
        pad_bottom = max(0, self.target_size - H - pad_top)
        pad_left = max(0, (self.target_size - W) // 2)
        pad_right = max(0, self.target_size - W - pad_left)

        padded_image = F.pad(
            image,
            [pad_left, pad_top, pad_right, pad_bottom],
            fill=0,
            padding_mode='constant',
        )

        return padded_image


def get_inference_transform() -> Compose:
    return Compose(
        [
            Pad(100),
            CenterCrop((100, 100)),
            ResizeTensor([224, 224]),
            Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
            Normalize(mean=0.5, std=0.5),
        ]
    )
