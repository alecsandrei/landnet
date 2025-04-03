from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import rasterio
from torchvision.datasets import ImageFolder

from landnet.config import FIGURES_DIR, GRIDS
from landnet.enums import Mode
from landnet.features.tiles import TileConfig, TileHandler, TileSize
from landnet.plots import get_correlation_matrix_plot


def rasterio_loader(path):
    with rasterio.open(path, nodata=-99999.0) as raster:
        print(raster.width, raster.height)
        return raster.read(1)


def to_image_folder(path: Path):
    # loader = partial(pil_loader, size=(100, 100))
    return ImageFolder(
        root=path,
        # transform=get_default_transform(),
        loader=rasterio_loader,
    )


if __name__ == '__main__':
    config = TileConfig(TileSize(100, 100))
    grids = list((GRIDS / Mode.TRAIN.value).glob('*.tif'))
    data_map: dict[str, np.ndarray] = {}
    with rasterio.open(grids[0]) as src:
        handler = TileHandler(config)
        length = handler.get_tiles_length(src)
        for i in range(length):
            window, transform = TileHandler(config).get_tile(src, i)
            data_map.setdefault(grids[0].stem, []).append(
                src.read(1, window=window).flatten()
            )
            for grid in grids[1:]:
                with rasterio.open(grid) as other_src:
                    data_map.setdefault(grid.stem, []).append(
                        other_src.read(1, window=window).flatten()
                    )

        for k, v in data_map.items():
            data_map[k] = np.concatenate(v)

    corr = pd.DataFrame.from_dict(data_map).corr(method='spearman')
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
