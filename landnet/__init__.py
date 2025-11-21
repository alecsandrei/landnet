from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from landnet.config import LOGGING_ENABLED
from landnet.logger import (
    ErrorFilter,
    JSONFormatter,
    NonErrorFilter,
    create_logger,
    setup_logging,
)

__all__ = [
    'ErrorFilter',
    'JSONFormatter',
    'NonErrorFilter',
    'seed_worker',
    'set_random_seed',
]
logger = create_logger(__name__)


if LOGGING_ENABLED:
    setup_logging()


def set_random_seed(seed_value: int) -> None:
    logger.info('Setting random seed to %d', seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass(init=False)
class RandomSeedContext:
    def __init__(self, seed_value: int) -> None:
        assert isinstance(seed_value, int)
        self.seed = seed_value
        self.set_random_seed(seed_value)

    def set_random_seed(self, seed_value: int) -> None:
        self.seed = seed_value
        set_random_seed(seed_value)

    def get_torch_generator(self) -> torch.Generator:
        return torch.Generator().manual_seed(self.seed)


torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
