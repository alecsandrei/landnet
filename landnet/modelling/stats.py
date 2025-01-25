from __future__ import annotations

import collections.abc as c
import concurrent.futures
import itertools
import typing as t
import warnings
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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


class Result(t.NamedTuple):
    true: c.Sequence[int]
    pred: c.Sequence[int]
    logits: c.Sequence[float]
    loss: float

    def confusion_matrix(self):
        return confusion_matrix(self.true, self.pred)

    def metrics(self) -> Metrics:
        return Metrics.from_y(self.true, self.pred, self.logits)

    def binary_classification_labels(
        self,
    ) -> list[t.Literal['tn', 'fn', 'tp', 'fp']]:
        labels: list[t.Literal['tn', 'fn', 'tp', 'fp']] = []
        for true, pred in zip(self.true, self.pred):
            match (true, pred):
                case (0, 0):
                    labels.append('tn')
                case (1, 0):
                    labels.append('fn')
                case (1, 1):
                    labels.append('tp')
                case (0, 1):
                    labels.append('fp')
        return labels

    @classmethod
    def from_results(
        cls,
        results: c.Sequence[Result],
        loss_agg: c.Callable[[c.Sequence[float]], float] = mean,
    ) -> Result:
        if not results:
            raise ValueError('At least one Result instance is required.')

        true = [item for result in results for item in result.true]
        pred = [item for result in results for item in result.pred]
        logits = [item for result in results for item in result.logits]
        loss = loss_agg([result.loss for result in results])

        return cls(true, pred, logits, loss)


class Metrics(t.NamedTuple):
    accuracy: float
    f1_score: float
    negative_predictive_value: float
    positive_predictive_value: float
    specificity: float
    sensitivity: float
    balanced_accuracy_score: float
    roc_auc_score: float

    @classmethod
    def from_y(
        cls,
        true: c.Sequence[int],
        pred: c.Sequence[int],
        logits: c.Sequence[float],
    ) -> Metrics:
        labels: c.Sequence[int] = (0, 1)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            if np.unique(true).shape[0] == 2:
                _roc_auc_score = roc_auc_score(true, logits, labels=labels)
            else:
                _roc_auc_score = 0

            return cls(
                float(accuracy_score(true, pred)),
                float(f1_score(true, pred, zero_division=0, labels=labels)),
                float(precision_score(true, pred, labels=labels, pos_label=0)),
                float(precision_score(true, pred, labels=labels, pos_label=1)),
                float(recall_score(true, pred, labels=labels, pos_label=0)),
                float(recall_score(true, pred, labels=labels, pos_label=1)),
                float(balanced_accuracy_score(true, pred)),
                float(_roc_auc_score),
            )

    def formatted(self) -> str:
        return ', '.join(
            f'{metric.title()} is {score}'
            for metric, score in self._asdict().items()
        )
