from __future__ import annotations

import contextlib
import json
import os
import random
import typing as t
import uuid
from pathlib import Path

import numpy as np
import torch

from landnet.enums import Architecture, LandslideClass

LOGGING_ENABLED = bool(int(os.getenv('LOGGING_ENABLED', 1)))

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXTERNAL_DATA_DIR = DATA_DIR / 'external'
DEM_TILES = EXTERNAL_DATA_DIR / 'DEM'
TRAIN_TILES = INTERIM_DATA_DIR / 'train_tiles'
TEST_TILES = INTERIM_DATA_DIR / 'test_tiles'
VALIDATION_TILES = INTERIM_DATA_DIR / 'validation_tiles'
INFERENCE_TILES = INTERIM_DATA_DIR / 'inference_tiles'
GRIDS = INTERIM_DATA_DIR / 'computed_grids'

MODELS_DIR = PROJ_ROOT / 'models'

REPORTS_DIR = PROJ_ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

LOGGING_DIR = PROJ_ROOT / 'logging'
LOGGING_CONFIG = LOGGING_DIR / 'config.json'

# GIS configs
EPSG = 3844
SAGA_CMD: str | None = None  # could be configured to specify path to saga_cmd
NODATA = float(os.getenv('NODATA', -32767.0))
SAGAGIS_NODATA = float(os.getenv('SAGAGIS_NODATA', -99999))

# Model configs
SEED = 0
LANDSLIDE_DENSITY_THRESHOLD = float(
    os.getenv('LANDSLIDE_DENSITY_THRESHOLD', 0.05)
)
ARCHITECTURE = Architecture(os.getenv('ARCHITECTURE', 'resnet50'))
PRETRAINED = bool(int(os.getenv('PRETRAINED', 1)))
EPOCHS = int(os.getenv('EPOCHS', 5))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 4))
TILE_SIZE = int(os.getenv('TILE_SIZE', 100))  # Size of the tiles in pixels
OVERLAP = int(os.getenv('OVERLAP', 0))
DEFAULT_CLASS_BALANCE = {
    LandslideClass.NO_LANDSLIDE: 0.5,
    LandslideClass.LANDSLIDE: 0.5,
}
TRAIN_NUM_SAMPLES = int(os.getenv('TRAIN_NUM_SAMPLES', 1000))

# Tune configs
EXPERIMENTS_NAME = os.getenv('EXPERIMENTS_NAME', uuid.uuid4().hex)
NUM_SAMPLES = int(
    os.getenv('NUM_SAMPLES', 5)
)  # Number of models to train with ray for hyperparameter tuning
GPUS = int(os.getenv('GPUS', 1))
CPUS = os.getenv('CPUS', None)
OVERWRITE = bool(
    int(os.getenv('OVERWRITE', 0))
)  # Whether or not to overwrite the existing models in EXPERIMENTS_NAME
TEMP_RAY_TUNE_DIR = MODELS_DIR / 'temp_ray_tune'


# Random seed for reproducibility
TORCH_GEN = torch.Generator()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_random_seed(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    global TORCH_GEN
    TORCH_GEN.manual_seed(seed_value)

    global SEED
    if SEED != seed_value:
        SEED = seed_value


def get_current_random_seed() -> int:
    seed = random.getstate()[1][0]
    assert seed == np.random.get_state()[1][0]  # type: ignore[index]
    assert seed == torch.random.get_rng_state()[1]
    return seed


set_random_seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


@contextlib.contextmanager
def temp_seed(seed_value: int) -> t.Generator[None, None, None]:
    """Context manager to set the random seed temporarily."""

    # Save current states
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    if torch.cuda.is_available():
        torch_cuda_state = torch.cuda.get_rng_state_all()
    else:
        torch_cuda_state = None

    try:
        # Set new seed
        set_random_seed(seed_value)
        yield
    finally:
        # Restore previous states
        random.setstate(random_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if torch_cuda_state is not None:
            torch.cuda.set_rng_state_all(torch_cuda_state)


def save_vars_as_json(out_file: Path):
    def is_json_serializable(obj: t.Any) -> bool:
        try:
            json.dumps(obj)
            return True
        except (TypeError, OverflowError):
            return False

    vars = {
        k: v if is_json_serializable(v) else str(v)
        for k, v in globals().items()
        if k.isupper()
    }
    with out_file.open(mode='w') as f:
        json.dump(vars, f, indent=2)
    return vars
