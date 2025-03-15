from __future__ import annotations

import os
from pathlib import Path

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
NODATA = -32767
# Model configs
LANDSLIDE_DENSITY_THRESHOLD = float(
    os.getenv('LANDSLIDE_DENSITY_THRESHOLD', 0.05)
)
ARCHITECTURE = os.getenv('ARCHITECTURE', 'alexnet')  # or 'resnet50' 'convnext'
PRETRAINED = bool(os.getenv('PRETRAINED_WEIGHTS', False))
EPOCHS = int(os.getenv('EPOCHS', 10))
GPUS = int(os.getenv('GPUS', 1))
CPUS = int(os.getenv('CPUS', 4))
NUM_SAMPLES = int(
    os.getenv('NUM_SAMPLES', 10)
)  # Number of models to train with ray for hyperparameter tuning
OVERWRITE = bool(
    os.getenv('OVERWRITE', False)
)  # Whether or not to overwrite the existing models
