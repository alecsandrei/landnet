from __future__ import annotations

import collections.abc as c
import os
import pickle
import shutil
import tempfile
import typing as t
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.models
from matplotlib.colors import ListedColormap
from PIL import Image
from ray import train, tune
from ray.train import CheckpointConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.metrics import RocCurveDisplay
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToTensor

from landnet.raster import LandslideClass, LandslideImageFolder
from landnet.training import Metrics, device, evaluate_model, one_epoch

if t.TYPE_CHECKING:
    from torchvision.models import AlexNet

    from landnet.training import Result

torch.cuda.empty_cache()

# model settings
ARCHITECTURE = os.getenv('ARCHITECTURE', 'alexnet')  # or 'resnet50' 'convnext'
USE_PRETRAINED_WEIGHTS = True

# training with ray settings
EPOCHS = int(os.getenv('EPOCHS', 10))
GPUS = 1
CPUS = 4
NUM_SAMPLES = int(os.getenv('NUM_SAMPLES', 10))

# data settings
LANDSLIDE_THRESHOLD = 0.05

# dirs
PARENT = Path(__file__).parent.parent
DATA_FOLDER = PARENT / 'data'
FIGURES_FOLDER = PARENT / 'figures'
TRAIN_FOLDER = DATA_FOLDER / 'train_rasters'
TRAIN_LANDSLIDE_PERCENTAGE = DATA_FOLDER / '66_65_ldl.csv'
TEST_LANDSLIDE_PERCENTAGE = DATA_FOLDER / '66_64_ldl.csv'
TEST_FOLDER = DATA_FOLDER / 'test_rasters'
MODELS_FOLDER = DATA_FOLDER / 'models'


def pil_loader(path: str, size: tuple[int, int]) -> Image.Image:
    with open(path, 'rb') as f:
        return Image.open(f).resize(size)


def alexnet() -> AlexNet:
    weights = None
    if USE_PRETRAINED_WEIGHTS:
        weights = torchvision.models.AlexNet_Weights.DEFAULT
    model = torchvision.models.alexnet(weights=weights)
    conv_1x1 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
    model.features.insert(0, conv_1x1)
    model.classifier[-1] = nn.Linear(4096, 1, bias=True)
    model.classifier.append(nn.Sigmoid())
    model.to(device())
    return model


def resnet50() -> nn.Sequential:
    weights = None
    if USE_PRETRAINED_WEIGHTS:
        weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights)
    conv_1x1 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
    model.fc = nn.Linear(2048, 1, bias=True)
    model = nn.Sequential(conv_1x1, model, nn.Sigmoid())
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


def get_train_datasets(train_folders: Path | c.Sequence[Path]):
    if not isinstance(train_folders, c.Sequence):
        train_folders = [train_folders]
    transform = get_transform()
    loader = partial(pil_loader, size=(100, 100))
    train_image_folder: ConcatDataset[ImageFolder] = ConcatDataset(
        [
            ImageFolder(
                train_folder,
                transform,
                loader=loader,
            )
            for train_folder in train_folders
        ]
    )
    train_dataset, validation_dataset = torch.utils.data.random_split(
        train_image_folder, (0.7, 0.3)
    )
    return (train_dataset, validation_dataset)


def get_train_loaders(
    train_folders: Path | c.Sequence[Path], batch_size: int
) -> tuple[DataLoader, DataLoader]:
    train_dataset, validation_dataset = get_train_datasets(train_folders)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return (train_loader, validation_loader)


def get_tile_number_from_image(image: tuple[str, int]) -> int:
    return int(''.join(filter(str.isdigit, Path(image[0]).stem)))


def get_test_loader(test_folders: Path | c.Sequence[Path], batch_size):
    if not isinstance(test_folders, c.Sequence):
        test_folders = [test_folders]
    loader = partial(pil_loader, size=(100, 100))

    def get_image_folder(folder: Path):
        image_folder = ImageFolder(folder, get_transform(), loader=loader)
        image_folder.imgs.sort(key=get_tile_number_from_image)
        return image_folder

    image_folders = [
        get_image_folder(test_folder) for test_folder in test_folders
    ]
    test_image_folder: ConcatDataset[ImageFolder] = ConcatDataset(image_folders)
    return DataLoader(
        test_image_folder,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


# def _ray_train_wrapper_cv(config, train_folder: Path):
#     batch_size, learning_rate = config['batch_size'], config['learning_rate']
#     print(f'Starting {train_folder.name}')
#     train_datasets = get_train_datasets(train_folder)
#     kfold = KFold(n_splits=KFOLDS, shuffle=True)
#     dataset = ConcatDataset(train_datasets)
#     for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
#         # Print
#         print(f'FOLD {fold}')
#         print('--------------------------------')
#         model: nn.Module = MODEL()
#         loss_fn = nn.BCELoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#         # Sample elements randomly from a given list of ids, no replacement.
#         train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
#         test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

#         # Define data loaders for training and testing data in this fold
#         train_loader = torch.utils.data.DataLoader(
#             dataset, batch_size=batch_size, sampler=train_subsampler
#         )
#         validation_loader = torch.utils.data.DataLoader(
#             dataset, batch_size=batch_size, sampler=test_subsampler
#         )
#         for epoch in range(EPOCHS):
#             result = one_epoch(
#                 model,
#                 train_loader,
#                 validation_loader,
#                 loss_fn,
#                 optimizer,
#             )
#             with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
#                 path = Path(temp_checkpoint_dir) / 'checkpoint.pt'
#                 checkpoint_data = {
#                     'epoch': epoch,
#                     'fold': fold,
#                     'net_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                 }
#                 with open(path, 'wb') as fp:
#                     pickle.dump(checkpoint_data, fp)
#                 checkpoint = train.Checkpoint.from_directory(
#                     temp_checkpoint_dir
#                 )
#                 train.report(
#                     result.validation.metrics._asdict()
#                     | {'loss': result.validation.loss},
#                     checkpoint=checkpoint,
#                 )


def ray_train_wrapper(config, train_folder: Path):
    batch_size, learning_rate = config['batch_size'], config['learning_rate']
    print(f'Starting {train_folder.name}')
    train_loader, validation_loader = get_train_loaders(
        train_folder, batch_size
    )
    model: nn.Module = MODEL()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Sample elements randomly from a given list of ids, no replacement.
    for epoch in range(EPOCHS):
        result = one_epoch(
            model,
            train_loader,
            validation_loader,
            loss_fn,
            optimizer,
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


def get_param_space():
    return {
        'learning_rate': tune.loguniform(1e-6, 1e-3),
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


def get_tuner(train_func, **run_config_kwargs) -> tune.Tuner:
    return tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_func),
            resources={'cpu': CPUS, 'gpu': GPUS},
        ),
        param_space=get_param_space(),
        tune_config=tune.TuneConfig(
            scheduler=get_scheduler(),
            num_samples=NUM_SAMPLES,
            search_alg=HyperOptSearch(metric='loss', mode='min'),
        ),
        run_config=train.RunConfig(
            **run_config_kwargs,
            checkpoint_config=CheckpointConfig(
                num_to_keep=EPOCHS,
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
    # variables = ['shade']
    folders = zip(
        [
            path
            for path in TRAIN_FOLDER.iterdir()
            if path.is_dir()
            # if path.name in variables
        ],
        [
            path
            for path in TEST_FOLDER.iterdir()
            if path.is_dir()
            # if path.name in variables
        ],
    )

    for train_folder, test_folder in folders:
        out_name = MODELS_FOLDER / f'{train_folder.stem}_{ARCHITECTURE}'
        if (
            out_name.with_suffix('.pt').exists()
            and out_name.with_suffix('.csv').exists()
        ):
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
        results.get_dataframe().to_csv(out_name.with_suffix('.csv'))
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
            rename_path = out_name.with_suffix('.pt')
            rename_path.unlink(missing_ok=True)
            data_path.rename(rename_path)
            shutil.rmtree(data_path.parents[2])


def save_fig(path: Path) -> None:
    plt.savefig(
        path,
        transparent=True,
        dpi=600,
        bbox_inches='tight',
    )


def save_confusion_matrix(
    result: Result, display_labels: tuple[str, str], out_fig: Path
):
    import numpy as np
    import seaborn as sns

    array = result.confusion_matrix()
    array_labels = np.array([[0, 1], [2, 3]])
    cmap = ListedColormap(
        [
            '#fe0d00',  # Color 2
            '#ffe101',  # Color 3
            '#5fa2ff',  # Color 1
            '#1eb316',  # Color 4
        ]
    )

    df_cm = pd.DataFrame(
        array_labels, index=display_labels, columns=display_labels
    )
    # Plot the heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        df_cm,
        square=True,
        annot=array,
        cmap=cmap,
        cbar=False,
        # xticklabels=False,
        # yticklabels=False,
        fmt='d',
        annot_kws={'size': 40},
    )
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    # Set axis labels
    plt.xlabel('Prediction label', fontsize=40)
    plt.ylabel('True label', fontsize=40)

    # Save the figure
    save_fig(out_fig)
    plt.close()


def save_roc_curve(result: Result, out_fig: Path):
    fig = RocCurveDisplay.from_predictions(result.true, result.logits)
    fig.plot(
        plot_chance_level=True,
        linewidth=5,
        c='navy',
        chance_level_kw={'linewidth': 5},
    )
    save_fig(out_fig)
    plt.close()


def evaluate_models():
    df_test_results_metrics = pd.DataFrame()
    df_test_results_labels = pd.DataFrame()
    for model_path in list(MODELS_FOLDER.glob(f'*{ARCHITECTURE}.pt'))[::-1]:
        df = pd.read_csv(
            (model_path.parent / model_path.stem).with_suffix('.csv')
        )
        if not df.shape[0] == NUM_SAMPLES:
            print(model_path, 'failed', df.shape[0])
            continue
        batch_size = int(df.sort_values(by='loss').iloc[0]['config/batch_size'])
        with open(model_path, 'rb') as fp:
            best_checkpoint_data = pickle.load(fp)
        model: nn.Module = MODEL()
        model.load_state_dict(best_checkpoint_data['net_state_dict'])
        test_loader = get_test_loader(
            TEST_FOLDER / model_path.stem.replace(f'_{ARCHITECTURE}', ''),
            batch_size,
        )
        test_results = evaluate_model(
            model,
            test_loader,
            nn.BCELoss(),
        )
        save_confusion_matrix(
            test_results,
            display_labels=(
                LandslideClass.NO_LANDSLIDE.name.replace('_', ' ').capitalize(),
                LandslideClass.LANDSLIDE.name.capitalize(),
            ),
            out_fig=(FIGURES_FOLDER / f'cm_{model_path.stem}').with_suffix(
                '.png'
            ),
        )
        save_roc_curve(
            test_results,
            out_fig=(FIGURES_FOLDER / f'roc_{model_path.stem}').with_suffix(
                '.png'
            ),
        )
        if 'true' not in df_test_results_labels:
            df_test_results_labels['true'] = test_results.true
        metrics = test_results.metrics()
        df_test_results_labels[f'pred_{model_path.stem}'] = test_results.pred
        df_test_results_labels[f'pred_{model_path.stem}_logits'] = (
            test_results.logits
        )
        df_test_results_labels[f'pred_{model_path.stem}_confusion'] = (
            test_results.binary_classification_labels()
        )
        df_test_results_metrics.loc[model_path.stem, Metrics._fields] = list(
            metrics
        )
        df_test_results_metrics.loc[model_path.stem, 'loss'] = test_results.loss

        print(
            'Best test set accuracy for model {}: {}'.format(
                model_path.stem, metrics.formatted()
            )
        )
    df_test_results_metrics.to_csv(
        MODELS_FOLDER / f'{ARCHITECTURE}_test_results.csv'
    )
    df_test_results_labels.to_csv(
        MODELS_FOLDER / f'{ARCHITECTURE}_test_labels.csv'
    )


if __name__ == '__main__':
    MODEL = locals()[ARCHITECTURE]
    train_models()
    evaluate_models()
