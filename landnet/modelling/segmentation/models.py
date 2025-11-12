from __future__ import annotations

import collections.abc as c

import segmentation_models_pytorch as smp
from torch import hub, nn
from torchvision.models.segmentation import (
    FCN,
    DeepLabV3,
    DeepLabV3_ResNet50_Weights,
    FCN_ResNet50_Weights,
    deeplabv3_resnet50,
    fcn_resnet50,
)

from landnet.config import PRETRAINED
from landnet.enums import Architecture, Mode
from landnet.modelling.models import ModelBuilder


def get_architecture(
    architecture: Architecture,
) -> c.Callable[[int, Mode], nn.Module]:
    return MODEL_BUILDERS[architecture].build


class DeepLabV3ResNet50Builder(ModelBuilder[DeepLabV3]):
    def __init__(self):
        # we do not provide num_classes argument here because it gets
        # overriden if weights are specified (check source code of torchvision)
        super().__init__(
            model=deeplabv3_resnet50,
            weights=DeepLabV3_ResNet50_Weights.DEFAULT if PRETRAINED else None,
        )

    def _adapt_output_features(self, model: DeepLabV3) -> DeepLabV3:
        current = model.classifier[-1]
        new = nn.Conv2d(
            current.in_channels,
            out_channels=2,
            kernel_size=current.kernel_size,
            stride=current.stride,
            padding=current.padding,
            dilation=current.dilation,
            groups=current.groups,
            bias=True,
            padding_mode=current.padding_mode,
        )

        model.classifier[-1] = new
        return model


class FCNResNet50Builder(ModelBuilder[FCN]):
    def __init__(self):
        super().__init__(
            model=fcn_resnet50,
            weights=FCN_ResNet50_Weights.DEFAULT if PRETRAINED else None,
        )

    def _adapt_output_features(self, model: FCN) -> FCN:
        current = model.classifier[-1]
        new = nn.Conv2d(
            current.in_channels,
            out_channels=2,
            kernel_size=current.kernel_size,
            stride=current.stride,
            padding=current.padding,
            dilation=current.dilation,
            groups=current.groups,
            bias=True,
            padding_mode=current.padding_mode,
        )

        model.classifier[-1] = new
        return model


def get_unet_model_milesial():
    return hub.load(
        'milesial/Pytorch-UNet',
        'unet_carvana',
        pretrained=PRETRAINED,
        scale=0.5,
    )


def get_unet_model_smp(in_channels: int, classes: int) -> smp.Unet:
    return smp.Unet(
        encoder_name='resnet101',
        in_channels=in_channels,
        classes=classes,
        encoder_weights='imagenet' if PRETRAINED else None,
    )


class UNetBuilder(ModelBuilder[smp.Unet]):
    def __init__(self):
        super().__init__(model=get_unet_model_smp, weights=None)

    def _adapt_output_features(self, model: smp.Unet) -> smp.Unet:
        return model

    def _adapt_input_channels(
        self, model: smp.Unet, in_channels: int
    ) -> smp.Unet:
        current = model.encoder.conv1
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
        model.classifier[-1] = new
        return model


MODEL_BUILDERS: dict[Architecture, ModelBuilder] = {
    Architecture.DEEPLABV3: DeepLabV3ResNet50Builder(),
    Architecture.FCN: FCNResNet50Builder(),
    Architecture.UNET: UNetBuilder(),
}
