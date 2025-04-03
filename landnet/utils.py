from __future__ import annotations

import datetime
import functools
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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


def save_fig(path: Path, figure: Figure | None = None) -> None:
    """General purpose figure saver."""
    kwargs = {
        'fname': path,
        'transparent': True,
        'dpi': 600,
        'bbox_inches': 'tight',
    }
    if figure is None:
        plt.savefig(**kwargs)
    else:
        figure.savefig(**kwargs)
