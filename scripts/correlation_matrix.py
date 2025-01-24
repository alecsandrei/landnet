import itertools
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

from landnet import raster, stats

PARENT = Path(__file__).parent.parent
DATA_FOLDER = PARENT / 'data'
TRAIN_FOLDER = DATA_FOLDER / 'train_rasters'
TEST_FOLDER = DATA_FOLDER / 'test_rasters'


def pil_loader(path: str, size: tuple[int, int]) -> Image.Image:
    with open(path, 'rb') as f:
        return Image.open(f).resize(size)


def to_image_folder(path: Path):
    loader = partial(pil_loader, size=(100, 100))
    return raster.LandslideImageFolder(
        root=path,
        transform=raster.get_default_transform(),
        loader=loader,
    )


if __name__ == '__main__':
    folders = map(
        to_image_folder,
        itertools.chain(
            [path for path in TRAIN_FOLDER.iterdir() if path.is_dir()],
            [path for path in TEST_FOLDER.iterdir() if path.is_dir()],
        ),
    )

    corr = stats.get_correlation_matrix(folders)
    breakpoint()
    fig = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0,
    )
    out_folder = PARENT / 'figures' / 'plotly_corr_matrix'
    out_folder.mkdir(exist_ok=True)
    fig.write_html(out_folder / 'graph.html')
    exit()
    stats.get_correlation_matrix_plot(corr)
    plt.savefig(
        Path(__file__).parent.parent / 'figures' / 'correlation_matrix.png',
        dpi=300,
    )
