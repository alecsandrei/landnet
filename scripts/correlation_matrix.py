from __future__ import annotations

import itertools
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

from landnet import features
from landnet.config import FIGURES_DIR, TEST_TILES, TRAIN_TILES
from landnet.modelling import stats
from landnet.plots import get_correlation_matrix_plot


def pil_loader(path: str, size: tuple[int, int]) -> Image.Image:
    with open(path, 'rb') as f:
        return Image.open(f).resize(size)


def to_image_folder(path: Path):
    loader = partial(pil_loader, size=(100, 100))
    return features.LandslideImageFolder(
        root=path,
        transform=features.get_default_transform(),
        loader=loader,
    )


if __name__ == '__main__':
    folders = map(
        to_image_folder,
        itertools.chain(
            [path for path in TRAIN_TILES.iterdir() if path.is_dir()],
            [path for path in TEST_TILES.iterdir() if path.is_dir()],
        ),
    )

    corr = stats.get_correlation_matrix(folders)
    fig = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0,
    )
    fig.write_html(FIGURES_DIR / 'plotly_correlation_matrix.html')
    get_correlation_matrix_plot(corr)
    plt.savefig(
        FIGURES_DIR / 'correlation_matrix.png',
        dpi=300,
    )
