from __future__ import annotations

from landnet.config import MODELS_DIR, TRIAL_NAME
from landnet.enums import GeomorphometricalVariable
from landnet.features.tiles import TileConfig, TileSize
from landnet.modelling.inference import InferTrainTest
from landnet.modelling.tune import MetricSorter

if __name__ == '__main__':
    parent = MODELS_DIR / TRIAL_NAME
    metric = 'val_sensitivity'
    vals = []
    for ckpt in parent.glob('*pt'):
        out_dir = parent / f'{ckpt.stem}_predictions'
        if (out_dir / 'predictions' / 'train').exists() and (
            out_dir / 'predictions' / 'test'
        ).exists():
            continue
        out_dir.mkdir(exist_ok=True)
        InferTrainTest(
            [GeomorphometricalVariable(ckpt.stem.split('_')[0])],
            MetricSorter('f1_score', 'max'),
            out_dir,
        ).handle_checkpoint(
            ckpt,
            {
                'batch_size': 32,
                'tile_config': TileConfig(TileSize(100, 100), 0),
            },
        )
