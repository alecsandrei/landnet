from __future__ import annotations

import pickle
import shutil
import tempfile
import typing as t
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from ray import train, tune
from ray.air import CheckpointConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import AlexNet_Weights, alexnet
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToTensor

from landnet.cnn import device, evaluate_model, one_epoch
from landnet.raster import LandslideImageFolder

if t.TYPE_CHECKING:
    from torchvision.models import AlexNet


NUM_WORKERS = 4
EPOCHS = 10
WEIGHTS = AlexNet_Weights.IMAGENET1K_V1
GPUS = 1
CPUS = 4
NUM_SAMPLES = 20


def pil_loader(path: str, size: tuple[int, int]) -> Image.Image:
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


def get_train_loaders(
    train_folder: Path, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    transform = get_transform()
    loader = partial(pil_loader, size=(100, 100))
    train_image_folder = ImageFolder(
        train_folder,
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
    return (train_loader, validation_loader)


def get_test_loader(test_folder: Path, batch_size):
    loader = partial(pil_loader, size=(100, 100))
    test_image_folder = ImageFolder(
        test_folder,
        get_transform(),
        loader=loader,
    )
    return DataLoader(
        test_image_folder,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


def ray_train_wrapper(
    config,
    train_folder: Path,
):
    batch_size, learning_rate = config['batch_size'], config['learning_rate']
    print(f'Starting {train_folder.name}')
    train_loader, validation_loader = get_train_loaders(
        train_folder, batch_size
    )
    model = get_model()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(EPOCHS):
        result = one_epoch(
            model, train_loader, validation_loader, loss_fn, optimizer
        )
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = Path(temp_checkpoint_dir) / 'checkpoint.pt'
            checkpoint_data = {
                'epoch': epoch,
                'net_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            with open(path, 'wb') as fp:
                pickle.dump(checkpoint_data, fp)
            checkpoint = train.Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(result.validation._asdict(), checkpoint=checkpoint)


def get_tune_config():
    return {
        'learning_rate': tune.loguniform(1e-6, 1e-1),
        'batch_size': tune.choice([8, 16, 32, 64, 128]),
    }


def get_scheduler():
    return ASHAScheduler(
        metric='loss',
        mode='min',
        max_t=EPOCHS,
        grace_period=2,
        reduction_factor=2,
    )


def get_tuner(train_func, **run_config_kwargs):
    return tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_func),
            resources={'cpu': CPUS, 'gpu': GPUS},
        ),
        param_space=get_tune_config(),
        tune_config=tune.TuneConfig(
            scheduler=get_scheduler(),
            num_samples=NUM_SAMPLES,
            search_alg=HyperOptSearch(metric='loss', mode='min'),
        ),
        run_config=train.RunConfig(
            **run_config_kwargs,
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute='loss',
                checkpoint_score_order='min',
            ),
        ),
    )


def reclassify_rasters(
    image_folder: Path,
    landslide_threshold: float,
    landslide_percentage_csv: Path,
):
    df = pd.read_csv(landslide_percentage_csv)
    df_subset = (
        df[['id', 'ldl']]
        .rename(columns={'id': 'tile', 'ldl': 'landslide_percentage'})
        .set_index('tile')
    )
    df_subset['landslide_percentage'] = df_subset['landslide_percentage'] / 100
    tiles = df_subset.to_dict()['landslide_percentage']
    LandslideImageFolder(image_folder).reclassify(tiles, landslide_threshold)


if __name__ == '__main__':
    threshold = 0.05
    data_folder = Path(__file__).parent.parent / 'data'
    train_folder = data_folder / 'train_rasters'
    train_landslide_percentage = data_folder / '66_65_ldl.csv'
    test_landslide_percentage = data_folder / '66_64_ldl.csv'
    test_folder = data_folder / 'test_rasters'
    models_folder = data_folder / 'models'
    final_models_folder = models_folder / 'final'
    models_folder.mkdir(exist_ok=True)
    folders = zip(
        [path for path in train_folder.iterdir() if path.is_dir()],
        [path for path in test_folder.iterdir() if path.is_dir()],
    )

    to_skip = ['mpi', 'txt']
    for train_folder, test_folder in folders:
        if (models_folder / f'{train_folder.name}.pt').exists():
            continue
        reclassify_rasters(train_folder, threshold, train_landslide_percentage)
        reclassify_rasters(test_folder, threshold, test_landslide_percentage)
        assert train_folder.name == test_folder.name, (
            train_folder.name,
            test_folder.name,
        )
        train_func = partial(
            ray_train_wrapper,
            train_folder=train_folder,
        )
        tuner = get_tuner(
            train_func, name=train_folder.name, storage_path=models_folder
        )
        results = tuner.fit()
        results.get_dataframe().to_csv(
            (models_folder / train_folder.name).with_suffix('.csv')
        )
        best_result = results.get_best_result('loss', 'min')
        assert best_result.metrics

        print('Best trial config: {}'.format(best_result.config))
        print(
            'Best trial final validation metrics: {}'.format(
                best_result.metrics
            )
        )
        best_checkpoint = best_result.get_best_checkpoint(
            metric='loss', mode='min'
        )
        with best_checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / 'checkpoint.pt'
            rename_path = (models_folder / train_folder.name).with_suffix('.pt')
            rename_path.unlink(missing_ok=True)
            data_path.rename(rename_path)
            shutil.rmtree(data_path.parents[2])
        with open(rename_path, 'rb') as fp:
            best_checkpoint_data = pickle.load(fp)

        model = get_model()
        model.load_state_dict(best_checkpoint_data['net_state_dict'])
        test_acc = evaluate_model(
            model,
            get_test_loader(test_folder, best_result.config['batch_size']),
            nn.BCELoss(),
        )
        print('Best trial test set accuracy: {}'.format(test_acc._formatted()))
