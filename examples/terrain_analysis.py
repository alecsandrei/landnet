from pathlib import Path

from PySAGA_cmd import SAGA

from landnet import Data
from landnet.terrain_analysis import TerrainAnalysis


def main(dem: Path, saga: SAGA, out_dir: Path):
    for tool in TerrainAnalysis(dem, saga, out_dir).execute():
        print('Executed', tool[0])


if __name__ == '__main__':
    saga = SAGA()
    main(
        Data.TEST_DEM.value,
        saga,
        Data.TEST_DEM.value.parent / 'test_rasters',
    )
    main(
        Data.TRAIN_DEM.value,
        saga,
        Data.TRAIN_DEM.value.parent / 'train_rasters',
    )
