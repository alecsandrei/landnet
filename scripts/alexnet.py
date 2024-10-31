from __future__ import annotations

import collections.abc as c
import typing as t
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import AlexNet_Weights, alexnet
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToTensor

from landnet.cnn import device, train_model

if t.TYPE_CHECKING:
    from torchvision.models import AlexNet


BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 10
LEARNING_RATE = 0.0001
WEIGHTS = AlexNet_Weights.IMAGENET1K_V1
NUM_SAMPLES = 10


def pil_loader(path: str, size: tuple[int, int]) -> Image.Image:
    # open path as file to avoid
    # ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        return Image.open(f).resize(size)


def get_model() -> AlexNet:
    model = alexnet(weights=WEIGHTS)
    # Our data is single channel: insert a conv2d which outputs three channels
    model.features.insert(0, nn.Conv2d(1, 3, kernel_size=2))
    model.classifier[-1] = nn.Linear(
        4096, 1
    )  # we need 1 as output, binary classification
    model.classifier.append(nn.Sigmoid())
    model.to(device())
    return model


def get_transform() -> Compose:
    return Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
            Normalize(mean=0.5, std=0.5),
        ]
    )


def get_data_loaders(
    train_folder: Path, test_folder: Path, batch_size: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = get_transform()
    loader = partial(pil_loader, size=(100, 100))
    train_image_folder = ImageFolder(
        train_folder,
        transform,
        loader=loader,
    )
    test_image_folder = ImageFolder(
        test_folder,
        transform,
        loader=loader,
    )
    train_dataset, validation_dataset = torch.utils.data.random_split(
        train_image_folder, (0.7, 0.3)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_image_folder,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return (train_loader, validation_loader, test_loader)


def train_and_save(
    folders: c.Iterable[tuple[Path, Path]],
    out_csv: Path,
    out_folder: Path,
    overwrite: bool = False,
):
    """Used to train a model for each geomorphometric parameter."""

    out_folder.mkdir(exist_ok=True)
    model_metrics = pd.DataFrame()
    for train_folder, test_folder in folders:
        assert train_folder.name == test_folder.name, (
            train_folder.name,
            test_folder.name,
        )
        print(f'Starting {train_folder.name}')
        if WEIGHTS is not None:
            out_model = out_folder / f'{train_folder.stem}_{WEIGHTS.name}'
        else:
            out_model = out_folder / f'{train_folder.stem}'
        if not overwrite and out_model.exists():
            print(f'{out_model!r} already exists, skipping')
            continue
        train_loader, validation_loader, test_loader = get_data_loaders(
            train_folder, test_folder, BATCH_SIZE
        )
        model = get_model()

        df = train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            num_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
        )
        df['parameter'] = train_folder.name
        model_metrics = pd.concat([model_metrics, df], ignore_index=True)
        torch.save(model.state_dict(), out_model)
        model_metrics.to_csv(out_csv)


if __name__ == '__main__':
    data_folder = Path(__file__).parent.parent / 'data'
    train_folder = data_folder / 'train_rasters'
    test_folder = data_folder / 'test_rasters'
    folders = zip(
        [path for path in train_folder.iterdir() if path.is_dir()],
        [path for path in test_folder.iterdir() if path.is_dir()],
    )

    train_and_save(
        folders=folders,
        out_csv=data_folder / 'metrics.csv',
        out_folder=data_folder / 'models',
        overwrite=True,
    )
