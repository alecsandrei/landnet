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

LANDSLIDE_THRESHOLD = 0.05
DATA_FOLDER = Path(__file__).parent.parent / 'data'
TRAIN_FOLDER = DATA_FOLDER / 'train_rasters'
TRAIN_LANDSLIDE_PERCENTAGE = DATA_FOLDER / '66_65_ldl.csv'
TEST_LANDSLIDE_PERCENTAGE = DATA_FOLDER / '66_64_ldl.csv'
TEST_FOLDER = DATA_FOLDER / 'test_rasters'
MODELS_FOLDER = DATA_FOLDER / 'models'


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


def train_models():
    MODELS_FOLDER.mkdir(exist_ok=True)
    folders = zip(
        [path for path in TRAIN_FOLDER.iterdir() if path.is_dir()],
        [path for path in TEST_FOLDER.iterdir() if path.is_dir()],
    )

    for train_folder, test_folder in folders:
        if (
            MODELS_FOLDER / f'{train_folder.name}.pt'
        ).exists() or train_folder.name in ['txt', 'mpi']:
            continue
        reclassify_rasters(
            train_folder, LANDSLIDE_THRESHOLD, TRAIN_LANDSLIDE_PERCENTAGE
        )
        reclassify_rasters(
            test_folder, LANDSLIDE_THRESHOLD, TEST_LANDSLIDE_PERCENTAGE
        )
        assert train_folder.name == test_folder.name, (
            train_folder.name,
            test_folder.name,
        )
        train_func = partial(
            ray_train_wrapper,
            train_folder=train_folder,
        )
        tuner = get_tuner(
            train_func, name=train_folder.name, storage_path=MODELS_FOLDER
        )
        results = tuner.fit()
        results.get_dataframe().to_csv(
            (MODELS_FOLDER / train_folder.name).with_suffix('.csv')
        )
        best_result = results.get_best_result('loss', 'min')
        assert best_result.config
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
        assert best_checkpoint is not None
        with best_checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / 'checkpoint.pt'
            rename_path = (MODELS_FOLDER / train_folder.name).with_suffix('.pt')
            rename_path.unlink(missing_ok=True)
            data_path.rename(rename_path)
            shutil.rmtree(data_path.parents[2])


def evaluate_models():
    for model_path in MODELS_FOLDER.glob('*.pt'):
        df = pd.read_csv(
            (model_path.parent / model_path.stem).with_suffix('.csv')
        )
        if not df.shape[0] == NUM_SAMPLES:
            print(model_path, 'failed', df.shape[0])
            continue
        batch_size = int(df.sort_values(by='loss').iloc[0]['config/batch_size'])
        with open(model_path, 'rb') as fp:
            best_checkpoint_data = pickle.load(fp)

        model = get_model()
        model.load_state_dict(best_checkpoint_data['net_state_dict'])
        test_acc = evaluate_model(
            model,
            get_test_loader(TEST_FOLDER / model_path.stem, batch_size),
            nn.BCELoss(),
        )
        print(
            'Best test set accuracy for model {}: {}'.format(
                model_path.stem, test_acc._formatted()
            )
        )


if __name__ == '__main__':
    train_models()
    evaluate_models()
