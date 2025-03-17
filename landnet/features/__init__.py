from __future__ import annotations

import typing as t

from PySAGA_cmd.saga import SAGA, Version

from landnet.features.grids import compute_grids
from landnet.features.tiles import TileSize, get_image_folders

if t.TYPE_CHECKING:
    from landnet.features.grids import GeomorphometricalVariable
    from landnet.features.tiles import ImageFolders

Mode = t.Literal['train', 'test']


def main(tile_size: TileSize) -> dict[GeomorphometricalVariable, ImageFolders]:
    saga = SAGA('saga_cmd', Version(9, 8, 0))
    compute_grids('train', saga)
    compute_grids('test', saga)
    return get_image_folders(tile_size)


if __name__ == '__main__':
    main(TileSize(100, 100))
