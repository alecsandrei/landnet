from __future__ import annotations

import typing as t

from torch import nn
from torchvision.models.segmentation import (
    FCN,
    DeepLabV3,
    DeepLabV3_ResNet50_Weights,
    FCN_ResNet50_Weights,
    deeplabv3_resnet50,
    fcn_resnet50,
)

from landnet.config import PRETRAINED
from landnet.modelling.models import ModelBuilder

M = t.TypeVar('M', bound=nn.Module)


class DeepLabV3ResNet50Builder(ModelBuilder):
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes

        # we do not provide num_classes argument here because it gets
        # overriden if weights are specified (check source code of torchvision)
        super().__init__(
            model=deeplabv3_resnet50,
            weights=DeepLabV3_ResNet50_Weights.DEFAULT if PRETRAINED else None,
        )

    def _adapt_output_features(self, model: M) -> M:
        if isinstance(model, nn.Sequential):
            deeplabv3 = model[1]
        else:
            deeplabv3 = model
        deeplabv3 = t.cast(DeepLabV3, deeplabv3)
        assert isinstance(deeplabv3.classifier, nn.Sequential)
        deeplabv3.classifier[-1] = nn.Conv2d(
            256, 2, kernel_size=(1, 1), stride=(1, 1)
        )
        return model


class FCNResNet50Builder(ModelBuilder):
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes

        # we do not provide num_classes argument here because it gets
        # overriden if weights are specified (check source code of torchvision)
        super().__init__(
            model=fcn_resnet50,
            weights=FCN_ResNet50_Weights.DEFAULT if PRETRAINED else None,
        )

    def _adapt_output_features(self, model: M) -> M:
        if isinstance(model, nn.Sequential):
            fcn = model[1]
        else:
            fcn = model
        fcn = t.cast(FCN, fcn)
        assert isinstance(fcn.classifier, nn.Sequential)
        fcn.classifier[-1] = nn.Conv2d(
            512, 2, kernel_size=(1, 1), stride=(1, 1)
        )
        return model
