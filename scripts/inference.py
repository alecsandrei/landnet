from __future__ import annotations

import geopandas as gpd
import numpy as np

from landnet.config import (
    ARCHITECTURE,
    GRIDS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)
from landnet.enums import GeomorphometricalVariable, Mode
from landnet.features.dataset import get_dem_tiles
from landnet.features.tiles import TileConfig, TileSize
from landnet.modelling.classification.inference import (
    InferenceFolder,
    perform_inference_on_tiles,
)
from landnet.modelling.classification.lightning import LandslideImageClassifier
from landnet.modelling.classification.models import get_architecture

if __name__ == '__main__':
    variables = [
        GeomorphometricalVariable.HILLSHADE,
        GeomorphometricalVariable.TOPOGRAPHIC_POSITION_INDEX,
        GeomorphometricalVariable.NEGATIVE_TOPOGRAPHIC_OPENNESS,
        GeomorphometricalVariable.DIGITAL_ELEVATION_MODEL,
        # GeomorphometricalVariable.EASTNESS,
        # GeomorphometricalVariable.SLOPE,
        # GeomorphometricalVariable.REAL_SURFACE_AREA,
        # GeomorphometricalVariable.FLOW_LINE_CURVATURE,
        GeomorphometricalVariable.TERRAIN_RUGGEDNESS_INDEX,
        # GeomorphometricalVariable.LOCAL_CURVATURE,
    ]
    out_tiles = PROCESSED_DATA_DIR / 'dem_tiles_infered_5vars.fgb'
    try:
        tiles = gpd.read_file(out_tiles)
    except FileNotFoundError:
        tiles = get_dem_tiles()
        tiles['prediction'] = np.nan
    checkpoint = (
        MODELS_DIR
        / 'convnext_100x100_5vars/convnext_100x100_5vars/2025-06-27_16-46-08/TorchTrainer_05f6ed3b_2_batch_size=2,learning_rate=0.0000,tile_config=ref_ph_c793cfd2_2025-06-27_17-10-49/checkpoint_000008/checkpoint.ckpt'
    )
    classifier = LandslideImageClassifier.load_from_checkpoint(
        checkpoint,
        model=get_architecture(ARCHITECTURE)(len(variables), Mode.INFERENCE),
    )

    missing = tiles['prediction'].isna()
    for group in tiles[missing].groupby(['id1', 'id2']):
        (id1, id2), subset = group
        folder = InferenceFolder(
            parent=GRIDS / Mode.INFERENCE.value / id1 / id2,
            tile_config=TileConfig(TileSize(100, 100), overlap=0),
            tiles=subset,
            variables=variables,
        )
        logits = perform_inference_on_tiles(classifier, folder)
        tiles.loc[subset.index, 'logits'] = logits
        tiles.loc[subset.index, 'prediction'] = np.round(logits).astype(int)
        tiles.to_file(out_tiles)
        print(
            'Remaining:',
            tiles.shape[0] - tiles.dropna(subset='prediction').shape[0],
        )
