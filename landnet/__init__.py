from enum import Enum
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent / 'data'


class Data(Enum):
    TRAIN_DEM = DATA / 'train.tif'
    TEST_DEM = DATA / 'test.tif'
