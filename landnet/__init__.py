from __future__ import annotations

import random

import numpy as np
import torch

from landnet.config import LOGGING_ENABLED, SEED
from landnet.logger import (
    ErrorFilter,
    JSONFormatter,
    NonErrorFilter,
    setup_logging,
)

__all__ = ['ErrorFilter', 'JSONFormatter', 'NonErrorFilter']


# Reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


TORCH_GEN = torch.Generator()
TORCH_GEN.manual_seed(SEED)

torch.use_deterministic_algorithms(True)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


if LOGGING_ENABLED:
    setup_logging()
