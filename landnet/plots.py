from __future__ import annotations

import collections.abc as c
import typing as t
from dataclasses import dataclass

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
            '#33a02c',  # Color 4
            '#ffff00',  # Color 3
            '#5fa2ff',  # Color 1
            '#ff0000',  # Color 2
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


@dataclass
class ExperimentsResultPlot:
    df: pd.DataFrame

    def make_long(
        self, value_vars: c.Sequence[str], var_name: str
    ) -> pd.DataFrame:
        return self.df.melt(
            id_vars='model',
            value_vars=value_vars,
            var_name=var_name,
            value_name='value',
        )

    def model_line_plot(
        self, value_vars: c.Sequence[str], ax: Axes | None = None
    ) -> Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 7))
        long_df = self.make_long(value_vars, var_name='metric')
        sns.lineplot(long_df, x='model', y='value', hue='metric', ax=ax)
        return ax

    def model_bar_plot(
        self, value_vars: c.Sequence[str], ax: Axes | None = None
    ) -> Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 7))
        long_df = self.make_long(value_vars, var_name='metric')
        order = (
            long_df[long_df['metric'] == value_vars[0]]
            .sort_values('value', ascending=False)['model']
            .tolist()
        )
        long_df['metric'] = pd.Categorical(
            long_df['metric'], categories=value_vars, ordered=True
        )
        long_df['metric'] = long_df['metric'].str.replace('_', ' ').str.title()
        long_df['model'] = pd.Categorical(
            long_df['model'], categories=order, ordered=True
        )
        sns.barplot(
            long_df,
            x='value',
            hue='metric',
            y='model',
            dodge=True,
            orient='y',
            ax=ax,
        )
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', padding=3, fontsize=8)  # type: ignore
        ax.legend()
        ax.set_xlabel('Metric')
        ax.set_ylabel('Model')
        ax.set_xlim(long_df['value'].min() - 0.05, 1)
        ax.margins(y=0.01)

        plt.tight_layout()
        return ax
