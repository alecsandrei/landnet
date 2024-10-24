import rasterio

from landnet import Data

if __name__ == '__main__':
    # saga = SAGA()
    # out_dir = Data.TEST_DEM.value.parent / 'tiled'
    # out_dir.mkdir(exist_ok=True)
    # output = split_raster(saga, Data.TEST_DEM.value, (100, 100), out_dir)
    # print(output.rasters)
    slope_train = Data.TEST_DEM.value.parent
    slope_test = Data.TEST_DEM.value.parent

    for folder in (slope_train, slope_test):
        for raster_path in folder.rglob('*.tif'):
            if raster_path.parent.name in ('0', '1'):
                with rasterio.open(raster_path) as raster:
                    if raster.height != 100 or raster.width != 100:
                        print(raster_path)
