from __future__ import annotations

import random

import numpy as np
import torch

from landnet.config import LOGGING_ENABLED
from landnet.logger import (
    ErrorFilter,
    JSONFormatter,
    NonErrorFilter,
    setup_logging,
)

__all__ = ['ErrorFilter', 'JSONFormatter', 'NonErrorFilter']

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

if LOGGING_ENABLED:
    setup_logging()
