from __future__ import annotations

import collections.abc as c
import concurrent.futures
import itertools
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
import torchmetrics.classification

if t.TYPE_CHECKING:
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


class BinaryClassificationMetricCollection(torchmetrics.MetricCollection):
    def __init__(self, prefix: str):
        super().__init__(
            {
                'f1_score': torchmetrics.classification.BinaryF1Score(),
                'sensitivity': torchmetrics.classification.BinaryRecall(),
                'specificity': torchmetrics.classification.BinarySpecificity(),
                'accuracy': torchmetrics.classification.BinaryAccuracy(),
                'negative_predictive_value': torchmetrics.classification.BinaryNegativePredictiveValue(),
                'positive_predictive_value': torchmetrics.classification.BinaryPrecision(),
                'roc_auc': torchmetrics.classification.BinaryAUROC(),
            },
            prefix=prefix,
        )
