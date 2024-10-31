from __future__ import annotations

import collections.abc as c
import typing as t
from functools import cache
from statistics import mean

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
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


class EpochMetrics(t.NamedTuple):
    train: Metrics
    validation: Metrics


class BatchResult(t.NamedTuple):
    true: c.Sequence[float]
    pred: c.Sequence[float]
    loss: float


class Metrics(t.NamedTuple):
    accuracy: t.SupportsFloat
    f1_score: t.SupportsFloat
    loss: t.SupportsFloat

    def _formatted(self) -> str:
        return ', '.join(
            f'{metric.title()} is {score}'
            for metric, score in self._asdict().items()
        )


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, loss_fn: nn.Module
) -> Metrics:
    model.eval()
    y_true = []
    y_pred = []
    loss_values = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device()), y.to(device()).reshape(-1, 1)
            pred = model(x)
            labels = pred.round()
            y_pred.extend(labels.tolist())
            y_true.extend(y.tolist())
            loss = loss_fn(pred, y.float())
            loss_values.append(loss.item())
    return Metrics(
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        mean(loss_values),
    )


def one_epoch(
    model: nn.Module,
    train_batch: DataLoader,
    validation_batch: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
) -> EpochMetrics:
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
        labels = pred.round()
        loss = loss_fn(pred, y.float())
        loss_value = loss.item()
        loss.backward()
        optimizer.step()
        return BatchResult(
            y.tolist(),
            labels.tolist(),
            t.cast(float, loss_value),
        )

    model.train()
    y_true: list[float] = []
    y_pred: list[float] = []
    loss_values = []
    for batch in train_batch:
        result = one_batch(batch, loss_fn, optimizer)
        y_true.extend(result.true)
        y_pred.extend(result.pred)
        loss_values.append(result.loss)
    validation = evaluate_model(model, validation_batch, loss_fn)
    training = Metrics(
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        mean(loss_values),
    )
    return EpochMetrics(
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
            f'|| Metrics for training: {result.train._formatted()}',
            f'|| Metrics for validation: {result.validation._formatted()}',
        )
    if test_loader is not None:
        print('Test result: ', evaluate_model(model, test_loader, loss_fn))
    return epoch_metrics
