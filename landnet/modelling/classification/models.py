from __future__ import annotations

import collections.abc as c
import typing as t

import torch
import torchvision.models
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import AlexNet, ConvNeXt, ResNet

from landnet._vendor.kcn import ConvNeXtKAN, ResNetKAN
from landnet.config import PRETRAINED
from landnet.enums import Architecture, Mode
from landnet.logger import create_logger
from landnet.modelling.models import ModelBuilder

if t.TYPE_CHECKING:
    from landnet.modelling.classification.dataset import LandslideImages

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


class AlexNetBuilder(ModelBuilder[AlexNet]):
    def __init__(self):
        super().__init__(
            model=torchvision.models.alexnet,
            weights=torchvision.models.AlexNet_Weights.DEFAULT
            if PRETRAINED
            else None,
        )

    def _adapt_input_channels(
        self, model: AlexNet, in_channels: int
    ) -> AlexNet:
        conv1 = nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2)
        model.features[0] = conv1
        return model

    def _adapt_output_features(self, model: AlexNet) -> AlexNet:
        layer = nn.Linear(4096, self.out_features, bias=True)
        model.classifier[-1] = layer
        return model

    def _finalize_model(self, model: AlexNet) -> AlexNet:
        """Return the model directly without wrapping in Sequential."""
        return model


class ResNet50Builder(ModelBuilder[ResNet]):
    def __init__(self):
        super().__init__(
            model=torchvision.models.resnet50,
            weights=torchvision.models.ResNet50_Weights.DEFAULT
            if PRETRAINED
            else None,
        )

    def _adapt_output_features(self, model: ResNet) -> ResNet:
        layer = nn.Linear(2048, self.out_features, bias=True)
        model.fc = layer
        return model

    def _adapt_input_channels(self, model: ResNet, in_channels: int) -> ResNet:
        model.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        return model


class ConvNextBuilder(ModelBuilder[ConvNeXt]):
    def __init__(self):
        super().__init__(
            model=torchvision.models.convnext_tiny,
            weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
            if PRETRAINED
            else None,
        )

    def _adapt_input_channels(
        self, model: ConvNeXt, in_channels: int
    ) -> ConvNeXt:
        current = model.features[0][0]
        new = nn.Conv2d(
            in_channels,
            out_channels=current.out_channels,
            kernel_size=current.kernel_size,
            stride=current.stride,
            padding=current.padding,
            dilation=current.dilation,
            groups=current.groups,
            bias=True,
            padding_mode=current.padding_mode,
        )

        model.features[0][0] = new
        return model

    def _adapt_output_features(self, model: ConvNeXt) -> ConvNeXt:
        layer = nn.Linear(
            model.classifier[2].in_features, self.out_features, bias=True
        )
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
