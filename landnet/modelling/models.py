from __future__ import annotations

import collections.abc as c
import pickle
import typing as t
from abc import ABC, abstractmethod

from torch import nn
from torchvision.models import WeightsEnum

from landnet.config import PRETRAINED
from landnet.enums import Mode
from landnet.logger import create_logger

if t.TYPE_CHECKING:
    from pathlib import Path

logger = create_logger(__name__)


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
        logger.debug('Adapting model input channels from %d to 3', in_channels)
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


def read_legacy_checkpoint(model: T, checkpoint: Path) -> T:
    with checkpoint.open(mode='rb') as fp:
        model_data = pickle.load(fp)
    model.load_state_dict(model_data['net_state_dict'], strict=False)
    return model
