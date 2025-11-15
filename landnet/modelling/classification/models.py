from __future__ import annotations

import collections.abc as c

import torchvision.models
from torch import nn
from torchvision.models import AlexNet, ConvNeXt, ResNet

from landnet._vendor.kcn import ConvNeXtKAN, ResNetKAN
from landnet.config import PRETRAINED
from landnet.enums import Architecture, Mode
from landnet.logger import create_logger
from landnet.modelling.models import ModelBuilder

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
