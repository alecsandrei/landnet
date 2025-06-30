from __future__ import annotations

from pathlib import Path

from ray.train import Result

from landnet.config import MODELS_DIR, TRIAL_NAME
from landnet.enums import GeomorphometricalVariable
from landnet.features.tiles import TileConfig, TileSize
from landnet.modelling.classification.inference import InferTrainTest
from landnet.modelling.tune import MetricSorter
from landnet.typing import TuneSpace

if __name__ == '__main__':
    infer = InferTrainTest(
        variables=[
            # GeomorphometricalVariable('area'),
            # GeomorphometricalVariable('cbl'),
            # GeomorphometricalVariable('ccros'),
            # GeomorphometricalVariable('cdl'),
            # GeomorphometricalVariable('cdo'),
            # GeomorphometricalVariable('cgene'),
            GeomorphometricalVariable('clong'),
            # GeomorphometricalVariable('clo'),
            # GeomorphometricalVariable('clu'),
            # GeomorphometricalVariable('cmaxi'),
            GeomorphometricalVariable('cmini'),
            GeomorphometricalVariable('conv'),
            GeomorphometricalVariable('cplan'),
            # GeomorphometricalVariable('nego'),
            # GeomorphometricalVariable('northness'),
            # GeomorphometricalVariable('poso'),
            # GeomorphometricalVariable('shade'),
            # GeomorphometricalVariable('slope'),
            GeomorphometricalVariable('cprof'),
            # GeomorphometricalVariable('croto'),
            # GeomorphometricalVariable('ctang'),
            # GeomorphometricalVariable('cup'),
            GeomorphometricalVariable('dem'),
            # GeomorphometricalVariable('eastness'),
            GeomorphometricalVariable('tpi'),
            GeomorphometricalVariable('tri'),
            GeomorphometricalVariable('twi'),
            # GeomorphometricalVariable('wind'),
            # GeomorphometricalVariable('vld'),
        ],
        sorter=MetricSorter('val_f_beta', 'max'),
        out_dir=None,
    )
    for trial in (
        MODELS_DIR / TRIAL_NAME / TRIAL_NAME / '2025-04-24_08-13-50'
    ).iterdir():
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
        infer.handle_checkpoint(
            Path(best_checkpoint.path) / 'checkpoint.ckpt', tune_space
        )
