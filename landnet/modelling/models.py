from __future__ import annotations

import pickle
import typing as t

import torchvision.models
from torch import nn

from landnet._vendor.kcn import ConvNeXtKAN, ResNetKAN
from landnet.config import PRETRAINED
from landnet.logger import create_logger
from landnet.modelling.train import device

if t.TYPE_CHECKING:
    from pathlib import Path

    from torchvision.models import AlexNet

logger = create_logger(__name__)


def alexnet() -> AlexNet:
    weights = None
    if PRETRAINED:
        weights = torchvision.models.AlexNet_Weights.DEFAULT
        logger.info('Set weights to %r' % weights)
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
        logger.info('Set weights to %r' % weights)
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
        logger.info('Set weights to %r' % weights)
    model = torchvision.models.convnext_base(weights=weights)
    conv_1x1 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
    model.classifier[2] = nn.Linear(1024, 1, bias=True)
    model = nn.Sequential(conv_1x1, model, nn.Sigmoid())
    model.to(device())
    return model


def convnextkan() -> nn.Sequential:
    weights = None
    if PRETRAINED:
        weights = torchvision.models.ConvNeXt_Base_Weights.DEFAULT
        logger.info('Set weights to %r' % weights)
    convnext = torchvision.models.convnext_base(weights=weights)
    conv_1x1 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
    convnext.classifier[2] = nn.Linear(1024, 1, bias=True)
    kcn = ConvNeXtKAN(convnext)
    model = nn.Sequential(conv_1x1, kcn, nn.Sigmoid())
    model.to(device())
    return model


def resnet50kan() -> nn.Sequential:
    weights = None
    if PRETRAINED:
        weights = torchvision.models.ResNet50_Weights.DEFAULT
        logger.info('Set weights to %r' % weights)
    resnet50 = torchvision.models.resnet50(weights=weights)
    resnet50.fc = nn.Linear(2048, 1, bias=True)
    kcn = ResNetKAN(resnet50)
    conv_1x1 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
    model = nn.Sequential(conv_1x1, kcn, nn.Sigmoid())
    model.to(device())
    return model


T = t.TypeVar('T', bound=nn.Module)


def read_legacy_checkpoint(model: T, checkpoint: Path) -> T:
    with checkpoint.open(mode='rb') as fp:
        model_data = pickle.load(fp)
    model.load_state_dict(model_data['net_state_dict'])
    return model
