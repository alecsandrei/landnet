from __future__ import annotations

import json
import typing as t

import pandas as pd
import rasterio

from landnet.config import (
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
    TEST_TILES,
    TRAIN_TILES,
)

if t.TYPE_CHECKING:
    from landnet.dataset import GeoJSON

if __name__ == '__main__':
    geojson: GeoJSON = {
        'type': 'FeatureCollection',
        'crs': {'type': 'name', 'properties': {'name': 'EPSG:3844'}},
        'features': [],
    }
    for mode in ('train', 'test'):
        path = (TRAIN_TILES if mode == 'train' else TEST_TILES) / 'dem'
        landslide_density_df = pd.read_csv(
            RAW_DATA_DIR / f'{mode}_landslide_density.csv'
        )
        for tif in path.rglob('*.tif'):
            with rasterio.open(tif) as raster:
                image_id = int(tif.stem[3:])
                landslide_density = float(
                    landslide_density_df.loc[
                        landslide_density_df['id'] == image_id, 'ldl'
                    ].iloc[0]
                )
                left, bottom, right, top = raster.bounds
                geojson['features'].append(
                    {
                        'type': 'Feature',
                        'properties': {
                            'mode': mode,
                            'image_id': image_id,
                            'landslide_density': landslide_density,
                        },
                        'geometry': {
                            'type': 'Polygon',
                            'coordinates': [
                                [
                                    [left, top],
                                    [right, top],
                                    [right, bottom],
                                    [left, bottom],
                                    [left, top],
                                ]
                            ],
                        },
                    }
                )

    with open(INTERIM_DATA_DIR / 'tiles.geojson', mode='w') as file:
        json.dump(geojson, file, indent=2)
