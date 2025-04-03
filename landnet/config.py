from __future__ import annotations

import os
import uuid
from pathlib import Path

from landnet.enums import Architecture

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
INFERENCE_TILES = INTERIM_DATA_DIR / 'inference_tiles'
GRIDS = INTERIM_DATA_DIR / 'computed_grids'

MODELS_DIR = PROJ_ROOT / 'models'

REPORTS_DIR = PROJ_ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

LOGGING_DIR = PROJ_ROOT / 'logging'
LOGGING_CONFIG = LOGGING_DIR / 'config.json'

# GIS configs
EPSG = 3844
SAGA_CMD = None  # could be configures to specify path to saga_cmd
RASTER_CELL_SIZE = (5, 5)  # meters
NODATA = float(os.getenv('NODATA', -32767.0))
DEFAULT_TILE_SIZE = 100

# Model configs
TRIAL_NAME = os.getenv('TRIAL_NAME', uuid.uuid4().hex)
LANDSLIDE_DENSITY_THRESHOLD = float(
    os.getenv('LANDSLIDE_DENSITY_THRESHOLD', 0.05)
)
ARCHITECTURE = Architecture(os.getenv('ARCHITECTURE', 'alexnet'))
PRETRAINED = int(os.getenv('PRETRAINED', 1))
EPOCHS = int(os.getenv('EPOCHS', 10))
GPUS = int(os.getenv('GPUS', 1))
CPUS = int(os.getenv('CPUS', 4))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 4))
NUM_SAMPLES = int(
    os.getenv('NUM_SAMPLES', 10)
)  # Number of models to train with ray for hyperparameter tuning
OVERWRITE = bool(
    int(os.getenv('OVERWRITE', 0))
)  # Whether or not to overwrite the existing models
OVERLAP = int(os.getenv('OVERLAP', 0))

TEMP_RAY_TUNE_DIR = MODELS_DIR / 'temp_ray_tune'
CHECKPOINTS_DIR = TEMP_RAY_TUNE_DIR / TRIAL_NAME
