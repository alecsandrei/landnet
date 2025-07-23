from __future__ import annotations

import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay

if t.TYPE_CHECKING:
    from matplotlib.axes import Axes


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


def get_confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
    display_labels: tuple[str, str],
    ax: Axes,
):
    import seaborn as sns

    confusion_matrix = metrics.confusion_matrix(targets, preds)
    array_labels = np.array([[0, 1], [2, 3]])
    cmap = ListedColormap(
        [
            '#1eb316',  # Color 4
            '#ffe101',  # Color 3
            '#5fa2ff',  # Color 1
            '#fe0d00',  # Color 2
        ]
    )

    df_cm = pd.DataFrame(
        array_labels, index=display_labels, columns=display_labels
    )
    # Plot the heatmap
    plot = sns.heatmap(
        df_cm,
        square=True,
        annot=confusion_matrix,
        cmap=cmap,
        cbar=False,
        fmt='d',
        annot_kws={'size': 40},
        ax=ax,
    )
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Set axis labels
    plt.xlabel('Prediction label', fontsize=30)
    plt.ylabel('True label', fontsize=30)

    return plot


def get_roc_curve(logits: np.ndarray, targets: np.ndarray, ax: Axes):
    plot = RocCurveDisplay.from_predictions(targets, logits)
    return plot.plot(
        linewidth=5,
        c='navy',
        chance_level_kw={'linewidth': 5},
        plot_chance_level=True,
        ax=ax,
    )
