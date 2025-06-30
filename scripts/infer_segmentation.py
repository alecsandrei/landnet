from __future__ import annotations

from pathlib import Path

from landnet.enums import GeomorphometricalVariable
from landnet.features.tiles import TileConfig, TileSize
from landnet.modelling import torch_clear
from landnet.modelling.segmentation.inference import Infer
from landnet.typing import TuneSpace

if __name__ == '__main__':
    torch_clear()

    variables = [
        GeomorphometricalVariable.SLOPE,
        GeomorphometricalVariable.REAL_SURFACE_AREA,
        GeomorphometricalVariable.PROFILE_CURVATURE,
        GeomorphometricalVariable.DIGITAL_ELEVATION_MODEL,
        # GeomorphometricalVariable.GENERAL_CURVATURE,
        # GeomorphometricalVariable.HILLSHADE,
        # GeomorphometricalVariable.TOPOGRAPHIC_POSITION_INDEX,
        GeomorphometricalVariable.NEGATIVE_TOPOGRAPHIC_OPENNESS,
    ]
    tile_config = TileConfig(TileSize(100, 100), overlap=50)
    model_config: TuneSpace = {
        'batch_size': 4,
        'learning_rate': 0.00001,
        'tile_config': tile_config,
    }

    infer = Infer(variables, model_config)
    parent = Path(
        '/media/alex/alex/python-modules-packages-utils/landnet/notebooks/lightning_logs'
    )
    last_version = sorted(
        map(lambda x: int(x.name.split('_')[-1]), parent.glob('version*'))
    )[-1]
    last_version = '60'
    ckpt = f'/media/alex/alex/python-modules-packages-utils/landnet/notebooks/lightning_logs/version_{last_version}/checkpoints/epoch=9-step=1000.ckpt'
    infer.handle_checkpoint(ckpt)
