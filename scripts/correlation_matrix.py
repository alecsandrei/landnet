import itertools
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
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
    stats.get_correlation_matrix_plot(corr)
    plt.savefig(
        Path(__file__).parent.parent / 'figures' / 'correlation_matrix.png',
        dpi=300,
    )
