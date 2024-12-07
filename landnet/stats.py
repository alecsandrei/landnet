from __future__ import annotations

import collections.abc as c
import concurrent.futures
import itertools
import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if t.TYPE_CHECKING:
    from matplotlib.axes import Axes
    from torchvision.datasets import ImageFolder


def get_correlation_matrix(folders: c.Iterable[ImageFolder]) -> pd.DataFrame:
    def handle_one_image(folder: ImageFolder, index: int):
        return folder[index][0].numpy().flatten()

    data_map: dict[str, np.ndarray] = {}

    def handle_one_folder(folder: ImageFolder):
        name = Path(folder.root).name
        with concurrent.futures.ThreadPoolExecutor() as executor:
            length = len(folder.imgs)
            results = executor.map(
                handle_one_image,
                itertools.repeat(folder, length),
                range(length),
            )
            data_map[name] = np.concatenate(list(results))

    for folder in folders:
        handle_one_folder(folder)

    return pd.DataFrame.from_dict(data_map).corr()


def get_correlation_matrix_plot(
    correlation_matrix: pd.DataFrame,
    ax: Axes | None = None,
) -> Axes:
    def plot_matrix():
        return sns.heatmap(
            correlation_matrix,
            fmt='.2f',
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            vmin=-1.0,
            vmax=1.0,
            square=True,
            ax=ax,
            annot=True,
        )

    if not ax:
        _, ax = plt.subplots(figsize=(20, 18))
    plot = plot_matrix()
    plt.tight_layout()
    return plot
