import geopandas as gpd

from landnet.features import get_merged_dem

if __name__ == '__main__':
    model_tiles = './data/model_tiles.geojson'
    tiles = './data/tiles.geojson'
    out_file = './data/merged_{mode}.tif'
    model_tiles_gdf = gpd.read_file(model_tiles)

    def train():
        get_merged_dem(
            '/media/alex/My Passport/LiDAR/TIF',
            out_file.format(mode='train'),
            gpd.read_file(tiles),
            model_tiles_gdf[model_tiles_gdf['mode'] == 'train'],
            5.0,
            20,
            '3844',
        )

    def test():
        get_merged_dem(
            '/media/alex/My Passport/LiDAR/TIF',
            out_file.format(mode='test'),
            gpd.read_file(tiles),
            model_tiles_gdf[model_tiles_gdf['mode'] == 'test'],
            5.0,
            20,
            '3844',
        )

    train()
    test()
