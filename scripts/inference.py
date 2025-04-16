from __future__ import annotations

import numpy as np

from landnet.config import ARCHITECTURE, GRIDS, MODELS_DIR, PROCESSED_DATA_DIR
from landnet.enums import GeomorphometricalVariable, Mode
from landnet.features.tiles import TileConfig, TileSize
from landnet.modelling.inference import (
    InferenceFolder,
    perform_inference_on_tiles,
)
import geopandas as gpd
from landnet.modelling.lightning import LandslideImageClassifier
from landnet.modelling.models import get_architecture

if __name__ == '__main__':
    variables = [
        GeomorphometricalVariable.DOWNSLOPE_CURVATURE,
        GeomorphometricalVariable.GENERAL_CURVATURE,
        GeomorphometricalVariable.LOCAL_UPSLOPE_CURVATURE,
        GeomorphometricalVariable.NEGATIVE_TOPOGRAPHIC_OPENNESS,
        GeomorphometricalVariable.SLOPE,
    ]
    out_tiles = PROCESSED_DATA_DIR / 'dem_tiles_infered.fgb'
    tiles = gpd.read_file(out_tiles)
    checkpoint = (
        MODELS_DIR
        / '5vars_conv1x1_weightedBce_convnext_100x100/2025-04-02_09-54-28/TorchTrainer_2fc8ded1_4_batch_size=4,learning_rate=0.0001,tile_config=ref_ph_c793cfd2_2025-04-02_09-52-52/checkpoint_000004/checkpoint.ckpt'
    )
    classifier = LandslideImageClassifier.load_from_checkpoint(
        checkpoint, model=get_architecture(ARCHITECTURE)(len(variables))
    )

    missing = tiles['prediction'].isna()
    for group in tiles[missing].groupby(['id1', 'id2']):
        (id1, id2), subset = group
        folder = InferenceFolder(
            parent=GRIDS / Mode.INFERENCE.value / id1 / id2,
            tile_config=TileConfig(TileSize(100, 100)),
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
