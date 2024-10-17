from PySAGA_cmd import SAGA

from landnet import Data
from landnet.raster import split_raster

if __name__ == '__main__':
    saga = SAGA()
    out_dir = Data.TEST_DEM.value.parent / 'tiled'
    out_dir.mkdir(exist_ok=True)
    split_raster(saga, Data.TEST_DEM.value, (100, 100), out_dir)
