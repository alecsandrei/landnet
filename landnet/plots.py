from __future__ import annotations

import typing as t

import matplotlib.pyplot as plt
import seaborn as sns

if t.TYPE_CHECKING:
    import pandas as pd
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
