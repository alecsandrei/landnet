from __future__ import annotations

import collections.abc as c
import typing as t
from functools import cache
from statistics import mean

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

if t.TYPE_CHECKING:
    from torch.optim import Optimizer


@cache
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    print('Warning: CUDA is not available.')
    return torch.get_default_device()


class EpochResult(t.NamedTuple):
    train: PredictionResult
    validation: PredictionResult


class PredictionResult(t.NamedTuple):
    metrics: Metrics
    loss: float


class BatchResult(t.NamedTuple):
    true: c.Sequence[int]
    pred: c.Sequence[int]
    result: PredictionResult


class Metrics(t.NamedTuple):
    accuracy: float
    f1_score: float
    precision_score: float
    recall_score: float
    balanced_accuracy_score: float
    roc_auc_score: float

    @classmethod
    def from_y(
        cls,
        y_true: c.Sequence[float],
        y_pred: c.Sequence[float],
        labels: c.Sequence[t.Any] = (0, 1),
    ) -> Metrics:
        return cls(
            float(accuracy_score(y_true, y_pred)),
            float(f1_score(y_true, y_pred, zero_division=0, labels=labels)),
            float(precision_score(y_true, y_pred, labels=labels)),
            float(recall_score(y_true, y_pred, labels=labels)),
            float(balanced_accuracy_score(y_true, y_pred)),
            float(roc_auc_score(y_true, y_pred, labels=labels)),
        )

    def formatted(self) -> str:
        return ', '.join(
            f'{metric.title()} is {score}'
            for metric, score in self._asdict().items()
        )


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, loss_fn: nn.Module
) -> PredictionResult:
    model.eval()
    y_true = []
    y_pred = []
    loss_values = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device()), y.to(device()).reshape(-1, 1)
            pred = model(x)
            labels = pred.round()
            loss = loss_fn(pred, y.float())
            loss_value = loss.item()
            y_pred.extend(labels.tolist())
            y_true.extend(y.tolist())
            loss_values.append(loss_value)
    return PredictionResult(
        Metrics.from_y(y_true, y_pred),
        mean(loss_values),
    )


def one_epoch(
    model: nn.Module,
    train_batch: DataLoader,
    validation_batch: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
) -> EpochResult:
    def one_batch(
        train_batch: DataLoader,
        loss_fn: nn.Module,
        optimizer: Optimizer,
    ) -> BatchResult:
        x, y = train_batch
        x, y = (
            x.to(device()),
            y.to(device()).reshape(-1, 1),
        )
        optimizer.zero_grad()
        pred = model(x)
        labels = pred.flatten().int()
        loss = loss_fn(pred, y.float())
        loss_value = loss.item()
        loss.backward()
        optimizer.step()
        labels_list = labels.tolist()
        y_list = y.flatten().tolist()
        result = PredictionResult(
            Metrics.from_y(y_list, labels_list), loss_value
        )
        return BatchResult(
            y_list,
            labels_list,
            result,
        )

    model.train()
    y_true: list[float] = []
    y_pred: list[float] = []
    loss_values = []
    for batch in train_batch:
        batch_result = one_batch(batch, loss_fn, optimizer)
        y_true.extend(batch_result.true)
        y_pred.extend(batch_result.pred)
        loss_values.append(float(batch_result.result.loss))
    validation = evaluate_model(model, validation_batch, loss_fn)
    training = PredictionResult(
        Metrics.from_y(y_true, y_pred), mean(loss_values)
    )
    return EpochResult(
        training,
        validation,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    test_loader: DataLoader | None = None,
) -> pd.DataFrame:
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_metrics = pd.DataFrame(
        columns=(
            'epoch',
            *('training_{}'.format(metric) for metric in Metrics._fields),
            *('validation_{}'.format(metric) for metric in Metrics._fields),
        )
    )
    for epoch in range(num_epochs):
        result = one_epoch(
            model, train_loader, validation_loader, loss_fn, optimizer
        )
        epoch_metrics.loc[epoch_metrics.shape[0], :] = [
            epoch,
            *result.train,
            *result.validation,
        ]
        print(
            f'Epoch {epoch+1}/{num_epochs}',
            f'|| Metrics for training: {result.train.metrics.formatted()}',
            f'|| Metrics for validation: {result.validation.metrics.formatted()}',
        )
    if test_loader is not None:
        print('Test result: ', evaluate_model(model, test_loader, loss_fn))
    return epoch_metrics
