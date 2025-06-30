from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from torch import nn

from landnet.config import ARCHITECTURE, FIGURES_DIR, GRIDS, MODELS_DIR
from landnet.enums import GeomorphometricalVariable, Mode
from landnet.features.dataset import get_dem_tiles
from landnet.features.tiles import TileConfig, TileSize
from landnet.modelling.classification.inference import (
    InferenceFolder,
    perform_inference_on_tiles,
)
from landnet.modelling.classification.lightning import LandslideImageClassifier
from landnet.modelling.classification.models import get_architecture


def get_all_conv2d_layers(model):
    conv_layers = {}

    def recurse(module, prefix=''):
        for name, layer in module.named_children():
            full_name = f'{prefix}.{name}' if prefix else name
            if isinstance(layer, nn.Conv2d):
                conv_layers[full_name] = (
                    layer,
                    layer.weight.detach().cpu().numpy(),
                )  # Convert to NumPy
            elif isinstance(layer, nn.Module):  # If it's a module, recurse
                recurse(layer, full_name)

    recurse(model)
    return conv_layers


# Hook function to store activations
activations = {}
kernels = {}


def hook_fn(layer_name: str):
    def inner_hook(module, input, output):
        if layer_name == 'model.0':
            activations['model_input'] = input[0].detach()

        activations[layer_name] = output.detach()
        kernels[layer_name] = module.weight.detach().cpu().numpy()

    return inner_hook


# Register hooks on specific layers
def register_hooks(model, layers_to_hook):
    registered = []
    for name, module in model.named_modules():
        if name in layers_to_hook:
            module.register_forward_hook(hook_fn(name))
            if name == 'model.0':
                registered.append('model_input')
            registered.append(name)
    return registered


# Function to plot feature maps in a grid
def plot_feature_maps(
    activations, out_file, layer_name, num_maps=36, split=False
):
    cmap = 'viridis'
    try:
        feature_maps = activations[layer_name].squeeze(
            0
        )  # Remove batch dimension
    except KeyError:
        print(f'Failed to get feature maps for {layer_name}')
        return None

    num_channels = feature_maps.shape[0]
    num_maps = min(num_maps, num_channels)  # Limit to available channels

    grid_size = int(np.ceil(np.sqrt(num_maps)))  # Define grid dimensions
    if split:
        fig, ax = plt.subplots()
        for i, map in enumerate(feature_maps):
            if i > num_maps:
                break
            map_array = map.cpu().numpy()
            ax.imshow(map_array, cmap=cmap)
            # ax.set_title(f'Ch {i+1}')
            ax.axis('off')  # Hide axis
            out_dir = out_file.with_suffix('')
            out_dir.mkdir(exist_ok=True)
            plt.tight_layout()
            plt.savefig(
                out_dir / f'{i}_{map_array.shape}.png',
                dpi=300,
                transparent=True,
                bbox_inches='tight',
                pad_inches=0,
            )
            # plt.close()
    else:
        fig, axes = plt.subplots(
            grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2)
        )

        for i, ax in enumerate(axes.flat):
            if i < num_maps:
                ax.imshow(feature_maps[i].cpu().numpy(), cmap=cmap)
                ax.set_title(f'Ch {i+1}')
            ax.axis('off')  # Hide axis

        plt.suptitle(f'Feature Maps from {layer_name}')
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        plt.close()


def plot_kernels(weights, out_file, layer_name, num_kernels=36, split=False):
    cmap = 'grey'
    if layer_name not in weights or weights[layer_name] is None:
        print(f'No weights found for {layer_name}')
        return

    kernels = weights[layer_name]  # Shape: (out_channels, in_channels, kH, kW)
    num_kernels = min(
        num_kernels, kernels.shape[0]
    )  # Limit to available kernels

    grid_size = int(np.ceil(np.sqrt(num_kernels)))  # Define grid size
    fig, axes = plt.subplots(
        grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2)
    )

    if split:
        fig, ax = plt.subplots()
        for i in range(num_kernels):
            if i > num_kernels:
                break
            ax.imshow(kernels[i, 0], cmap=cmap)
            # ax.set_title(f'Ch {i+1}')
            ax.axis('off')  # Hide axis
            out_dir = out_file.with_suffix('')
            out_dir.mkdir(exist_ok=True)
            plt.tight_layout()
            plt.savefig(
                out_dir / f'{i}.png',
                dpi=300,
                transparent=False,
                bbox_inches='tight',
            )
            # plt.close()
    else:
        for i, ax in enumerate(axes.flat):
            if i < num_kernels:
                kernel = kernels[i, 0]  # Take first input channel
                ax.imshow(kernel, cmap=cmap)
                ax.set_title(f'Kernel {i+1}')
            ax.axis('off')

        plt.suptitle(f'Kernels from {layer_name}')
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        plt.close()


if __name__ == '__main__':
    import torch

    torch.cuda.empty_cache()
    tile_path = '72/58/72_58_S11.tif'

    variables = [
        GeomorphometricalVariable.DOWNSLOPE_CURVATURE,
        GeomorphometricalVariable.GENERAL_CURVATURE,
        GeomorphometricalVariable.LOCAL_UPSLOPE_CURVATURE,
        GeomorphometricalVariable.NEGATIVE_TOPOGRAPHIC_OPENNESS,
        GeomorphometricalVariable.SLOPE,
    ]
    tiles = get_dem_tiles()
    checkpoint = (
        MODELS_DIR
        / '5vars_conv1x1_weightedBce_convnext_100x100/2025-04-02_09-54-28/TorchTrainer_2fc8ded1_4_batch_size=4,learning_rate=0.0001,tile_config=ref_ph_c793cfd2_2025-04-02_09-52-52/checkpoint_000004/checkpoint.ckpt'
    )
    classifier = LandslideImageClassifier.load_from_checkpoint(
        checkpoint,
        model=get_architecture(ARCHITECTURE)(len(variables), Mode.INFERENCE),
    )

    tile = tiles[tiles['path'] == tile_path]
    folder = InferenceFolder(
        parent=GRIDS
        / Mode.INFERENCE.value
        / tile['id1'].iloc[0]
        / tile['id2'].iloc[0],
        tile_config=TileConfig(TileSize(100, 100)),
        tiles=tile,
        variables=variables,
    )

    # model_layers = list(classifier.model.children())
    layers = get_all_conv2d_layers(classifier)
    to_plot = list(layers)[:20]
    registered = register_hooks(classifier, to_plot)

    perform_inference_on_tiles(classifier, folder)

    out_dir = FIGURES_DIR / 'feature_maps' / tile_path.split('/')[-1]
    os.makedirs(out_dir, exist_ok=True)
    for layer in registered:
        plot_kernels(
            kernels,
            (out_dir / f'{layer}_kernel.png').with_suffix('.png'),
            layer,
            split=True,
            num_kernels=5,
        )
        plot_feature_maps(
            activations,
            (out_dir / layer).with_suffix('.png'),
            layer,
            split=False,
            num_maps=36,
        )
    assert folder.parent.exists()
