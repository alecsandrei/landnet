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
TRAIN_TILES = PROCESSED_DATA_DIR / 'train_tiles'
TEST_TILES = PROCESSED_DATA_DIR / 'test_tiles'

MODELS_DIR = PROJ_ROOT / 'models'

REPORTS_DIR = PROJ_ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

# GIS configs
EPSG = 3844

# Model configs
LANDSLIDE_DENSITY_THRESHOLD = float(
    os.getenv('LANDSLIDE_DENSITY_THRESHOLD', 0.05)
)
ARCHITECTURE = os.getenv('ARCHITECTURE', 'alexnet')  # or 'resnet50' 'convnext'
PRETRAINED = bool(os.getenv('PRETRAINED_WEIGHTS', False))
EPOCHS = int(os.getenv('EPOCHS', 10))
GPUS = int(os.getenv('GPUS', 1))
CPUS = int(os.getenv('CPUS', 4))
NUM_SAMPLES = int(os.getenv('NUM_SAMPLES', 10))
OVERWRITE = bool(os.getenv('OVERWRITE', False))
