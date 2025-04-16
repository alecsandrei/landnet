from __future__ import annotations

from ray.train import Result

from landnet._typing import TuneSpace
from landnet.config import MODELS_DIR, TRIAL_NAME
from landnet.enums import GeomorphometricalVariable
from landnet.features.tiles import TileConfig, TileSize
from landnet.modelling.inference import InferTrainTest
from landnet.modelling.tune import MetricSorter

if __name__ == '__main__':
    infer = InferTrainTest(
        variables=[
            # GeomorphometricalVariable.DOWNSLOPE_CURVATURE,
            # GeomorphometricalVariable.GENERAL_CURVATURE,
            # GeomorphometricalVariable.LOCAL_UPSLOPE_CURVATURE,
            # GeomorphometricalVariable.NEGATIVE_TOPOGRAPHIC_OPENNESS,
            GeomorphometricalVariable.SLOPE,
        ],
        sorter=MetricSorter('val_f_beta', 'max'),
    )
    for trial in (MODELS_DIR / TRIAL_NAME / '2025-04-12_16-13-19').iterdir():
        if not trial.is_dir() or (trial / 'predictions').exists():
            continue
        try:
            ...
            # trial = Trial.from_directory(trial)
        except FileNotFoundError:
            continue
        if not any(
            obj.stem.startswith('checkpoint') for obj in trial.iterdir()
        ):
            continue
        best_checkpoint = Result.from_path(trial).get_best_checkpoint(
            infer.sorter.metric, infer.sorter.mode
        )
        tune_space: TuneSpace = {
            'batch_size': 1,
            'tile_config': TileConfig(TileSize(100, 100), overlap=0),
        }
        infer.out_dir = trial
        infer.handle_checkpoint(best_checkpoint, tune_space)
