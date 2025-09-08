from __future__ import annotations

import datetime
import functools
import time
import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from landnet.enums import GeomorphometricalVariable
from landnet.logger import create_logger

logger = create_logger(__name__)


def timeit(func):
    @functools.wraps(func)
    def wrapper(*arg, **kw):
        t1 = time.perf_counter()
        res = func(*arg, **kw)
        t2 = time.perf_counter()
        logger.info('Took %f seconds to run %r' % (t2 - t1, func.__name__))
        return res

    return wrapper


def get_utc_now():
    return datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d_%H-%M-%S')


class SaveFigKwargs(t.TypedDict):
    fname: Path
    transparent: bool
    dpi: int
    bbox_inches: str


def save_fig(path: Path, figure: Figure | None = None) -> None:
    """General purpose figure saver."""
    kwargs: SaveFigKwargs = {
        'fname': path,
        'transparent': True,
        'dpi': 600,
        'bbox_inches': 'tight',
    }
    if figure is None:
        plt.savefig(**kwargs)
    else:
        figure.savefig(**kwargs)


def geomorphometrical_variables_from_file(
    path: Path,
) -> list[GeomorphometricalVariable]:
    with path.open(mode='r') as file:
        return t.cast(
            list[GeomorphometricalVariable],
            [
                GeomorphometricalVariable._member_map_[
                    variable.strip().split('.')[1]
                ]
                for variable in file.readlines()
            ],
        )
