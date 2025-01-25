from __future__ import annotations

import typing as t
from enum import Enum, auto
from functools import cache

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from landnet.modelling.stats import Metrics, Result

if t.TYPE_CHECKING:
    from torch.optim import Optimizer


@cache
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    print('Warning: CUDA is not available.')
    return torch.get_default_device()


class Epoch(t.NamedTuple):
    train: Result
    validation: Result


class Mode(Enum):
    TRAINING = auto()
    EVALUATION = auto()


def evaluate_model(
    model: nn.Module, test_loader: DataLoader, loss_fn: nn.Module
) -> Result:
    model.eval()
    results = []
    with torch.no_grad():
        for batch in test_loader:
            results.append(
                handle_prediction(
                    batch=batch,
                    model=model,
                    loss_fn=loss_fn,
                    mode=Mode.EVALUATION,
                )
            )
    return Result.from_results(results)


def handle_prediction(
    *,
    batch: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    mode: Mode = Mode.TRAINING,
    optimizer: Optimizer | None = None,
) -> Result:
    if mode is Mode.TRAINING and optimizer is None:
        raise ValueError('The optimizer was not provided in training mode.')
    x, y = batch
    x, y = x.to(device()), y.to(device()).reshape(-1, 1)
    if optimizer:
        optimizer.zero_grad()
    logits = model(x)
    labels = logits.round().int().flatten().tolist()
    loss = loss_fn(logits, y.float())
    logits = logits.flatten().tolist()
    y = y.flatten().tolist()
    loss_value = loss.item()
    if mode is Mode.TRAINING:
        loss.backward()
        assert optimizer is not None
        optimizer.step()
    return Result(
        y,
        labels,
        logits,
        loss_value,
    )


def one_epoch(
    model: nn.Module,
    train_batch: DataLoader,
    validation_batch: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
) -> Epoch:
    model.train()
    results = []
    for batch in train_batch:
        results.append(
            handle_prediction(
                batch=batch,
                model=model,
                loss_fn=loss_fn,
                mode=Mode.TRAINING,
                optimizer=optimizer,
            )
        )
    validation = evaluate_model(model, validation_batch, loss_fn)
    training = Result.from_results(results)
    return Epoch(
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
        train_result, validation_result = one_epoch(
            model, train_loader, validation_loader, loss_fn, optimizer
        )
        train_metrics = train_result.metrics()
        validation_metrics = validation_result.metrics()
        epoch_metrics.loc[epoch_metrics.shape[0], :] = [
            epoch,
            *train_metrics,
            *validation_metrics,
        ]
        print(
            f'Epoch {epoch+1}/{num_epochs}',
            f'|| Metrics for training: {train_metrics.formatted()}',
            f'|| Metrics for validation: {validation_metrics.formatted()}',
            f'|| Train loss: {train_result.loss}',
        )
    if test_loader is not None:
        print('Test result: ', evaluate_model(model, test_loader, loss_fn))
    return epoch_metrics
