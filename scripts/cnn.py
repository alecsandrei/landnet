from __future__ import annotations

import collections.abc as c
import pickle
import typing as t
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.models
from matplotlib.colors import ListedColormap
from PIL import Image
from sklearn.metrics import RocCurveDisplay
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder

from landnet.config import (
    ARCHITECTURE,
    EPOCHS,
    FIGURES_DIR,
    GRIDS,
    MODELS_DIR,
    NUM_SAMPLES,
    OVERWRITE,
    PRETRAINED,
    TEST_TILES,
)
from landnet.features.grids import GeomorphometricalVariable
from landnet.features.tiles import (
    LandslideClass,
    LandslideImages,
    TileSize,
    get_image_folders_for_variable,
)
from landnet.logger import create_logger
from landnet.modelling.train import (
    Metrics,
    device,
    evaluate_model,
    one_epoch,
)

if t.TYPE_CHECKING:
    from torchvision.models import AlexNet

    from landnet.modelling.train import Result

torch.cuda.empty_cache()

logger = create_logger(__name__)


TILE_SIZE = TileSize(100, 100)
BATCH_SIZE = 32
LEARNING_RATE = 0.00001


def pil_loader(path: str, size: tuple[int, int]) -> Image.Image:
    with open(path, 'rb') as f:
        return Image.open(f).resize(size)


def alexnet() -> AlexNet:
    weights = None
    if PRETRAINED:
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
    if PRETRAINED:
        weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights)
    conv_1x1 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
    model.fc = nn.Linear(2048, 1, bias=True)
    model = nn.Sequential(conv_1x1, model, nn.Sigmoid())
    model.to(device())
    return model


def convnext():
    weights = None
    if PRETRAINED:
        weights = torchvision.models.ConvNeXt_Base_Weights.DEFAULT
    model = torchvision.models.convnext_base(weights=weights)
    conv_1x1 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
    model.classifier[2] = nn.Linear(1024, 1, bias=True)
    model = nn.Sequential(conv_1x1, model, nn.Sigmoid())
    model.to(device())
    return model


def get_train_datasets(image_folders: c.Sequence[LandslideImages]):
    # transform = get_transform()
    # loader = partial(pil_loader, size=(tile_size.rows, tile_size.columns))
    train_image_folder: ConcatDataset[ImageFolder] = ConcatDataset(
        image_folders
    )
    train_dataset, validation_dataset = torch.utils.data.random_split(
        train_image_folder, (0.7, 0.3)
    )
    return (train_dataset, validation_dataset)


def get_train_loaders(
    image_folders: c.Sequence[LandslideImages],
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_dataset, validation_dataset = get_train_datasets(image_folders)
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


def train(grids: Path | c.Sequence[Path]):
    if not isinstance(grids, c.Sequence):
        grids = [grids]
    logger.info('Starting %s' % [grid.stem for grid in grids])
    image_folders = [
        get_image_folders_for_variable(
            GeomorphometricalVariable(grid.stem), TILE_SIZE
        )
        for grid in grids
    ]
    train_loader, validation_loader = get_train_loaders(
        [folder['train'] for _, folder in image_folders], BATCH_SIZE
    )
    model: nn.Module = MODEL()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        result = one_epoch(
            model,
            train_loader,
            validation_loader,
            loss_fn,
            optimizer,
        )

        metrics = result.validation.metrics()
        print(
            {
                'loss': result.validation.loss,
                'f1_score': metrics.f1_score,
                'specificity': metrics.specificity,
                'sensitivity': metrics.sensitivity,
                'accuracy': metrics.accuracy,
            }
        )


def train_models():
    # folders = zip(
    #     [path for path in TRAIN_TILES.iterdir() if path.is_dir()],
    #     [path for path in TEST_TILES.iterdir() if path.is_dir()],
    # )

    train_grids = GRIDS / 'train'
    test_grids = GRIDS / 'test'
    grids = ['shade']
    for train_grid in train_grids.glob('*.tif'):
        if train_grid.stem not in grids:
            continue
        test_grid = test_grids / train_grid.name
        out_name = MODELS_DIR / f'{train_grid.stem}_{ARCHITECTURE}'
        if (
            not OVERWRITE
            and out_name.with_suffix('.pt').exists()
            and out_name.with_suffix('.csv').exists()
        ):
            continue

        train(train_grid)


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
    for model_path in list(MODELS_DIR.glob(f'*{ARCHITECTURE}.pt'))[::-1]:
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
            TEST_TILES / model_path.stem.replace(f'_{ARCHITECTURE}', ''),
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
            out_fig=(FIGURES_DIR / f'cm_{model_path.stem}').with_suffix('.png'),
        )
        save_roc_curve(
            test_results,
            out_fig=(FIGURES_DIR / f'roc_{model_path.stem}').with_suffix(
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
        MODELS_DIR / f'{ARCHITECTURE}_test_results.csv'
    )
    df_test_results_labels.to_csv(
        MODELS_DIR / f'{ARCHITECTURE}_test_labels.csv'
    )


if __name__ == '__main__':
    MODEL = locals()[ARCHITECTURE]
    train_models()
    # evaluate_models()
