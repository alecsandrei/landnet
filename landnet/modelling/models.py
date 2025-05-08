from __future__ import annotations

import collections.abc as c
import pickle
import typing as t
from abc import ABC, abstractmethod

import torch
import torchvision.models
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import AlexNet, ConvNeXt, ResNet, WeightsEnum

from landnet._vendor.kcn import ConvNeXtKAN, ResNetKAN
from landnet.config import PRETRAINED
from landnet.enums import Architecture, Mode
from landnet.logger import create_logger

if t.TYPE_CHECKING:
    from pathlib import Path

    from landnet.features.tiles import LandslideImages

logger = create_logger(__name__)


def get_architecture(
    architecture: Architecture,
) -> c.Callable[[int, Mode], nn.Module]:
    return MODEL_BUILDERS[architecture].build


def conv1x1(in_channels, out_channels) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        groups=1,
        bias=True,
    )


T = t.TypeVar('T', bound=nn.Module)
M = t.TypeVar('M', bound=nn.Module)


class ModelBuilder(ABC, t.Generic[T]):
    """Base class for building models with common functionality."""

    def __init__(
        self,
        model: c.Callable[..., T],
        out_features: int = 1,
        weights: WeightsEnum | None = None,
    ):
        self.model = model
        self.out_features = out_features
        self.weights = weights

    def _get_model(self, mode: Mode) -> T:
        model = self.model()
        if (
            PRETRAINED
            and mode not in (Mode.INFERENCE, Mode.TEST)
            and self.weights is not None
        ):
            logger.info('Set weights to %s' % self.weights)
            self.load_weights_partial(model, self.weights)
        return model

    def build(self, in_channels: int, mode: Mode):
        """Build the model with the specified input channels and mode."""
        model = self._get_model(mode)

        with_adapted_input = self._adapt_input_channels(model, in_channels)
        with_adapted_output = self._adapt_output_features(with_adapted_input)

        return self._finalize_model(with_adapted_output)

    def _adapt_input_channels(
        self, model: M, in_channels: int
    ) -> nn.Sequential | M:
        """Adapt the model for the specified number of input channels."""
        if in_channels == 3:
            return model
        return nn.Sequential(self._create_conv1x1(in_channels, 3), model)

    @abstractmethod
    def _adapt_output_features(self, model: M) -> M:
        """Override in subclasses to adapt the output layer."""

    def _finalize_model(self, model: M) -> M:
        """Final processing before returning the model."""
        return model

    @staticmethod
    def _create_conv1x1(in_channels: int, out_channels: int) -> nn.Conv2d:
        """Create a 1x1 convolution layer."""
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=True,
        )

    @staticmethod
    def load_weights_partial(model: nn.Module, weights: WeightsEnum) -> None:
        """Load weights partially, ignoring missing keys."""
        model.load_state_dict(
            weights.get_state_dict(check_hash=True), strict=False
        )


class AlexNetBuilder(ModelBuilder):
    def __init__(self):
        super().__init__(
            model=torchvision.models.alexnet,
            weights=torchvision.models.AlexNet_Weights.DEFAULT
            if PRETRAINED
            else None,
        )

    def _adapt_output_features(self, model: AlexNet) -> AlexNet:
        model.classifier[-1] = nn.Linear(4096, self.out_features, bias=True)
        return model

    def _finalize_model(self, model: AlexNet) -> AlexNet:
        """Return the model directly without wrapping in Sequential."""
        return model


class ResNet50Builder(ModelBuilder):
    def __init__(self):
        super().__init__(
            model=torchvision.models.resnet50,
            weights=torchvision.models.ResNet50_Weights.DEFAULT
            if PRETRAINED
            else None,
        )

    def _adapt_output_features(self, model: ResNet) -> ResNet:
        model.fc = nn.Linear(2048, self.out_features, bias=True)
        return model


class ConvNextBuilder(ModelBuilder):
    def __init__(self):
        super().__init__(
            model=torchvision.models.convnext_base,
            weights=torchvision.models.ConvNeXt_Base_Weights.DEFAULT
            if PRETRAINED
            else None,
        )

    # def _adapt_input_channels(
    #     self, model: M, in_channels: int
    # ) -> nn.Sequential | M:
    #     model.features[0][0] = nn.Conv2d(
    #         in_channels,
    #         128,
    #         4,
    #         4,
    #         0,
    #         dilation=1,
    #         groups=1,
    #         bias=True,
    #     )
    #     return model
    # return super()._adapt_input_channels(model, in_channels)

    def _adapt_output_features(self, model: M) -> M:
        layer = nn.Linear(1024, self.out_features, bias=True)
        if isinstance(model, nn.Sequential):
            assert isinstance(model[1], ConvNeXt)
            convnext = model[1]
            assert isinstance(convnext, ConvNeXt)
            convnext.classifier[2] = layer
        if isinstance(model, ConvNeXt):
            model.classifier[2] = layer
        return model


class ResNet50KANBuilder(ModelBuilder):
    def __init__(self):
        super().__init__(
            model=torchvision.models.resnet50,
            weights=torchvision.models.ResNet50_Weights.DEFAULT
            if PRETRAINED
            else None,
        )

    def build(self, in_channels: int, mode: Mode) -> nn.Module:
        resnet = super()._get_model(mode)
        resnet = self._adapt_output_features(resnet)

        # Assuming ResNetKAN class is defined elsewhere
        kcn = ResNetKAN(resnet)
        conv_1x1 = self._create_conv1x1(in_channels, 3)

        return nn.Sequential(conv_1x1, kcn, nn.Sigmoid())

    def _adapt_output_features(self, model: ResNet) -> ResNet:
        model.fc = nn.Linear(2048, self.out_features, bias=True)
        return model


class ConvNextKANBuilder(ModelBuilder):
    def __init__(self):
        super().__init__(
            model=torchvision.models.convnext_base,
            weights=torchvision.models.ConvNeXt_Base_Weights.DEFAULT
            if PRETRAINED
            else None,
        )

    def build(self, in_channels: int, mode: Mode) -> nn.Module:
        resnet = super()._get_model(mode)
        resnet = self._adapt_output_features(resnet)

        kcn = ConvNeXtKAN(resnet)
        conv_1x1 = self._create_conv1x1(in_channels, 3)

        return nn.Sequential(conv_1x1, kcn, nn.Sigmoid())

    def _adapt_output_features(self, model: ConvNeXt) -> ConvNeXt:
        model.classifier[2] = nn.Linear(1024, self.out_features, bias=True)
        return model


MODEL_BUILDERS: dict[Architecture, ModelBuilder] = {
    Architecture.ALEXNET: AlexNetBuilder(),
    Architecture.RESNET50: ResNet50Builder(),
    Architecture.CONVNEXT: ConvNextBuilder(),
    Architecture.RESNET50KAN: ResNet50KANBuilder(),
    Architecture.CONVNEXTKAN: ConvNextKANBuilder(),
}


def read_legacy_checkpoint(model: T, checkpoint: Path) -> T:
    with checkpoint.open(mode='rb') as fp:
        model_data = pickle.load(fp)
    model.load_state_dict(model_data['net_state_dict'], strict=False)
    return model


# Function to apply PCA on the channel dimension of an image
def apply_pca_on_channels(
    image: torch.Tensor, n_components: int = 8
) -> torch.Tensor:
    """
    Apply PCA on the channel dimension of the image to reduce the number of channels.

    Args:
        image: A tensor of shape (C, H, W), where C is the number of channels.
        n_components: Number of principal components to retain.

    Returns:
        A tensor of shape (n_components, H, W), with reduced channels.
    """
    # Reshape the image to (C, H*W) to treat each channel as a feature
    image_flat = image.view(image.shape[0], -1).numpy()  # Shape (C, H*W)

    # Apply PCA to reduce channels (C -> n_components)
    pca = PCA(n_components=n_components)
    image_pca = pca.fit_transform(
        image_flat.T
    ).T  # Apply PCA and transpose back

    # Convert back to a tensor and reshape to (n_components, H, W)
    return torch.tensor(image_pca).view(
        n_components, image.shape[1], image.shape[2]
    )


# Function to apply PCA on all images in the ConcatDataset
class PCAConcatDataset(Dataset):
    def __init__(
        self,
        landslide_images: c.Sequence[LandslideImages],
        n_components: int = 8,
    ):
        """
        Apply PCA on all images in the ConcatDataset object to reduce the channels.

        Args:
            concat_dataset: The concatenated dataset containing images.
            n_components: Number of principal components to retain for PCA.
        """
        self.landslide_images = landslide_images
        self.n_components = n_components

        # Apply PCA to all images in the dataset
        self.reduced_data: list[tuple[torch.Tensor, int]] = (
            self._apply_pca_to_all_images()
        )

    def _apply_pca_to_all_images(self):
        """
        Apply PCA to all images in the dataset and store the reduced images.

        Returns:
            A list of PCA-reduced images.
        """
        reduced_images: list[tuple[torch.Tensor, int]] = []
        for image_batch in zip(*self.landslide_images):
            cat = torch.cat([batch[0] for batch in image_batch], dim=0)
            class_ = image_batch[0][1]
            assert all(batch[1] == class_ for batch in image_batch[1:])
            reduced_image = apply_pca_on_channels(cat, self.n_components)
            reduced_images.append((reduced_image, class_))
        return reduced_images

    def __len__(self):
        return len(self.reduced_data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Return the PCA-reduced image.
        """
        return self.reduced_data[index]


# class CNBlock(nn.Module):
#     def __init__(
#         self,
#         dim,
#         layer_scale: float,
#         stochastic_depth_prob: float,
#         norm_layer: c.Callable[..., nn.Module] | None = None,
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = partial(nn.LayerNorm, eps=1e-6)
#         self.conv1 = nn.Conv2d(
#             dim, dim, kernel_size=7, padding=3, groups=dim, bias=True
#         )
#         self.perm1 = Permute([0, 2, 3, 1])
#         self.norm_layer = norm_layer(dim)
#         self.linear1 = nn.Linear(
#             in_features=dim, out_features=4 * dim, bias=True
#         )
#         self.act = nn.GELU()
#         self.perm2 = Permute([0, 3, 1, 2])
#         self.ca = ChannelAttention(4 * dim)
#         self.sa = SpatialAttention()
#         self.perm3 = Permute([0, 2, 3, 1])
#         self.linear2 = nn.Linear(
#             in_features=4 * dim, out_features=dim, bias=True
#         )
#         self.perm4 = Permute([0, 3, 1, 2])
#         self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
#         self.stochastic_depth = StochasticDepth(stochastic_depth_prob, 'row')

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         x = self.conv1(input)
#         x = self.perm1(x)
#         x = self.norm_layer(x)
#         x = self.linear1(x)
#         x = self.act(x)
#         x = self.perm2(x)
#         x = x * self.ca(x)
#         x = x * self.sa(x)
#         x = self.perm3(x)
#         x = self.linear2(x)
#         x = self.perm4(x)
#         result = self.layer_scale * x
#         result = self.stochastic_depth(result)
#         result += input
#         return result


# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc = nn.Sequential(
#             nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)


# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()

#         self.conv1 = nn.Conv2d(
#             2, 1, kernel_size, padding=kernel_size // 2, bias=False
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
