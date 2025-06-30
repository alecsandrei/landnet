from __future__ import annotations

import collections.abc as c
import concurrent.futures
import itertools
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
import torchmetrics.classification
import torchmetrics.segmentation

if t.TYPE_CHECKING:
    from torchvision.datasets import ImageFolder


def get_correlation_matrix(folders: c.Iterable[ImageFolder]) -> pd.DataFrame:
    def handle_one_image(folder: ImageFolder, index: int):
        array, _ = folder[index]
        grid_type = Path(folder.imgs[index][0]).stem
        return (array.flatten(), grid_type)

    data_map: dict[str, np.ndarray] = {}

    def handle_one_folder(folder: ImageFolder):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            length = len(folder)
            results = executor.map(
                handle_one_image,
                itertools.repeat(folder, length),
                range(length),
            )

            for result in results:
                data_map.setdefault(result[1], []).append(result[0])  # type: ignore
            for k, v in data_map.items():
                data_map[k] = np.concatenate(v)
                data_map[k] = data_map[k][
                    (data_map[k] != -99999) & (data_map[k] != 0)
                ]

    for folder in folders:
        handle_one_folder(folder)
    corr = pd.DataFrame.from_dict(data_map).corr()
    return corr


class BinaryClassificationMetricCollection(torchmetrics.MetricCollection):
    def __init__(self, prefix: str | None = None):
        super().__init__(
            {
                'sensitivity': torchmetrics.classification.BinaryRecall(),
                'specificity': torchmetrics.classification.BinarySpecificity(),
                'accuracy': torchmetrics.classification.BinaryAccuracy(),
                'negative_predictive_value': torchmetrics.classification.BinaryNegativePredictiveValue(),
                'positive_predictive_value': torchmetrics.classification.BinaryPrecision(),
                'roc_auc': torchmetrics.classification.BinaryAUROC(),
                'f1_score': torchmetrics.classification.BinaryF1Score(),
                'f2_score': torchmetrics.classification.BinaryFBetaScore(
                    beta=2.0
                ),
                'f3_score': torchmetrics.classification.BinaryFBetaScore(
                    beta=3.0
                ),
            },
            prefix=prefix,
        )
