from __future__ import annotations

from pathlib import Path

from ray.train import Result

from landnet.enums import GeomorphometricalVariable
from landnet.features.tiles import TileConfig, TileSize
from landnet.modelling.classification.inference import InferTrainTest
from landnet.modelling.tune import MetricSorter
from landnet.typing import TuneSpace

if __name__ == '__main__':
    sorter = MetricSorter('val_f2_score', 'max')
    infer = InferTrainTest(
        variables=[
            GeomorphometricalVariable.HILLSHADE,
            GeomorphometricalVariable.TOPOGRAPHIC_POSITION_INDEX,
            GeomorphometricalVariable.NEGATIVE_TOPOGRAPHIC_OPENNESS,
            GeomorphometricalVariable.DIGITAL_ELEVATION_MODEL,
            GeomorphometricalVariable.TERRAIN_RUGGEDNESS_INDEX,
        ],
        out_dir=None,
    )
    trial_path = Path(
        '/media/alex/alex/python-modules-packages-utils/landnet/models/convnext_100x100_5vars/convnext_100x100_5vars/2025-06-27_16-46-08'
    )
    for trial in trial_path.iterdir():
        if not trial.is_dir() or (trial / 'predictions').exists():
            continue
        try:
            ...
        except FileNotFoundError:
            continue
        if not any(
            obj.stem.startswith('checkpoint') for obj in trial.iterdir()
        ):
            continue
        best_checkpoint = Result.from_path(trial).get_best_checkpoint(
            sorter.metric, sorter.mode
        )
        tune_space: TuneSpace = {
            'batch_size': 1,
            'tile_config': TileConfig(TileSize(100, 100), overlap=0),
        }
        infer.out_dir = trial
        infer.handle_checkpoint(
            Path(best_checkpoint.path) / 'checkpoint.ckpt', tune_space
        )
