"""Same as cnn_tune but instead stacks all of the GeomorphometricalVariables."""

from __future__ import annotations

import collections.abc as c
import pickle
import shutil
import tempfile
import typing as t
from functools import partial
from pathlib import Path

import pandas as pd
import torch
import torchvision
from PIL import Image
from ray import train, tune
from ray.air import CheckpointConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import AlexNet_Weights, alexnet
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToTensor

from landnet.features import LandslideImageFolder
from landnet.modelling.train import device, evaluate_model, one_epoch

if t.TYPE_CHECKING:
    from torchvision.models import AlexNet


USE_PRETRAINED_WEIGHTS = True
NUM_WORKERS = 4
EPOCHS = 10
GPUS = 1
CPUS = 4
NUM_SAMPLES = 5
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
    weights = None
    if USE_PRETRAINED_WEIGHTS:
        weights = AlexNet_Weights.IMAGENET1K_V1
    model = alexnet(weights=weights)
    # Our data is single channel: insert a conv2d which outputs three channels
    model.features.insert(0, nn.Conv2d(1, 3, kernel_size=2))
    model.classifier[-1] = nn.Linear(
        4096, 1
    )  # we need 1 as output, binary classification
    model.classifier.append(nn.Sigmoid())
    model.to(device())
    return model


def convnext():
    weights = None
    if USE_PRETRAINED_WEIGHTS:
        weights = torchvision.models.ConvNeXt_Base_Weights.DEFAULT
    model = torchvision.models.convnext_base(weights=weights)
    conv_1x1 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
    model.classifier[2] = nn.Linear(1024, 1, bias=True)
    model = nn.Sequential(conv_1x1, model, nn.Sigmoid())
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
    train_folders: Path | c.Sequence[Path], batch_size: int
) -> tuple[DataLoader, DataLoader]:
    if not isinstance(train_folders, c.Sequence):
        train_folders = [train_folders]
    transform = get_transform()
    loader = partial(pil_loader, size=(100, 100))
    train_image_folder: ConcatDataset[ImageFolder] = ConcatDataset(
        [
            LandslideImageFolder(
                root=train_folder,
                transform=transform,
                loader=loader,
            )
            for train_folder in train_folders
        ]
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


def get_test_loader(test_folders: Path | c.Sequence[Path], batch_size):
    if not isinstance(test_folders, c.Sequence):
        test_folders = [test_folders]
    loader = partial(pil_loader, size=(100, 100))
    test_image_folder: ConcatDataset[ImageFolder] = ConcatDataset(
        [
            ImageFolder(
                test_folder,
                get_transform(),
                loader=loader,
            )
            for test_folder in test_folders
        ]
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
    train_folders: c.Sequence[Path],
):
    batch_size, learning_rate = config['batch_size'], config['learning_rate']
    train_loader, validation_loader = get_train_loaders(
        train_folders, batch_size
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
            metrics = result.validation.metrics()
            train.report(
                {
                    'loss': result.validation.loss,
                    'f1_score': metrics.f1_score,
                    'specificity': metrics.specificity,
                    'sensitivity': metrics.sensitivity,
                    'accuracy': metrics.accuracy,
                },
                checkpoint=checkpoint,
            )


def get_tune_config():
    return {
        'learning_rate': tune.loguniform(1e-6, 1e-1),
        'batch_size': tune.choice([2, 4, 8]),
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
    LandslideImageFolder(root=image_folder).reclassify(
        tiles, landslide_threshold
    )


def train_model():
    MODELS_FOLDER.mkdir(exist_ok=True)
    variables = ['area', 'slope', 'shade', 'cmini', 'wind']
    train_folders = [
        path
        for path in TRAIN_FOLDER.iterdir()
        if path.is_dir()
        if path.name in variables
    ]
    test_folders = [
        path
        for path in TEST_FOLDER.iterdir()
        if path.is_dir()
        if path.name in variables
    ]
    model_output_name = 'all_variables'

    for train_folder, test_folder in zip(train_folders, test_folders):
        assert train_folder.name == test_folder.name, (
            train_folder.name,
            test_folder.name,
        )
        reclassify_rasters(
            train_folder, LANDSLIDE_THRESHOLD, TRAIN_LANDSLIDE_PERCENTAGE
        )
        reclassify_rasters(
            test_folder, LANDSLIDE_THRESHOLD, TEST_LANDSLIDE_PERCENTAGE
        )

    if not (MODELS_FOLDER / model_output_name).with_suffix('.pt').exists():
        print('EXISTS')
        train_func = partial(
            ray_train_wrapper,
            train_folders=train_folders,
        )
        tuner = get_tuner(
            train_func, name=model_output_name, storage_path=MODELS_FOLDER
        )
        results = tuner.fit()
        results.get_dataframe().to_csv(
            (MODELS_FOLDER / model_output_name).with_suffix('.csv')
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
            rename_path = (MODELS_FOLDER / model_output_name).with_suffix('.pt')
            rename_path.unlink(missing_ok=True)
            data_path.rename(rename_path)
            shutil.rmtree(data_path.parents[2])
    test_model(test_folders)


def test_model(test_folders: c.Sequence[Path]):
    path = MODELS_FOLDER / 'all_variables.pt'
    df = pd.read_csv((path.parent / path.stem).with_suffix('.csv'))
    batch_size = int(df.sort_values(by='loss').iloc[0]['config/batch_size'])
    with open(path, 'rb') as fp:
        best_checkpoint_data = pickle.load(fp)

    model = get_model()
    model.load_state_dict(best_checkpoint_data['net_state_dict'])
    test_acc = evaluate_model(
        model,
        get_test_loader(test_folders, batch_size),
        nn.BCELoss(),
    )
    print(
        'Best test set accuracy for model {}: {}'.format(
            path.stem, test_acc.metrics().formatted()
        )
    )


if __name__ == '__main__':
    train_model()
