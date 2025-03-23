from __future__ import annotations

import collections.abc as c
import concurrent.futures
import os
import time
import typing as t
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.crs
import rasterio.mask
import rasterio.merge
import rasterio.warp
import torch
from PIL import Image
from rasterio import DatasetReader, windows
from rasterio.enums import Resampling
from rasterio.features import dataset_features
from shapely import Polygon, box
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    Lambda,
    Normalize,
    RandomApply,
    RandomHorizontalFlip,
    ToTensor,
    functional,
)

from landnet._typing import ImageFolders, Metadata
from landnet.config import (
    DEFAULT_TILE_SIZE,
    EPSG,
    GRIDS,
    LANDSLIDE_DENSITY_THRESHOLD,
    NODATA,
    RASTER_CELL_SIZE,
)
from landnet.dataset import (
    geojson_to_gdf,
    get_empty_geojson,
    get_landslide_shapes,
    get_percentage_intersection,
    get_tile_relative_path,
    process_feature,
)
from landnet.enums import LandslideClass, Mode
from landnet.features.grids import GeomorphometricalVariable
from landnet.logger import create_logger

if t.TYPE_CHECKING:
    from torch import Tensor

logger = create_logger(__name__)


@dataclass
class TileSize:
    columns: int
    rows: int

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self):
        return f'{self.columns}x{self.rows}'

    @classmethod
    def from_string(cls, string: str):
        split = string.split('x')
        return TileSize(int(split[0]), int(split[1]))


def get_tiles(
    ds: DatasetReader, tile_width, tile_height, overlap: int | None = None
):
    if overlap is None:
        overlap = 0
    ncols, nrows = ds.meta['width'], ds.meta['height']
    xstep = tile_width
    ystep = tile_height
    for x in range(0, ncols + xstep, xstep):
        if x != 0:
            x -= overlap
        width = tile_width + overlap * (2 if x != 0 else 1)
        if x + width > ncols:
            # logger.warning('Could not create tiles from x=%i' % x)
            break
        for y in range(0, nrows + ystep, ystep):
            if y != 0:
                y -= overlap
            height = tile_height + overlap * (2 if y != 0 else 1)
            if y + height > nrows:
                # logger.warning('Could not create tiles from y=%i' % y)
                break
            window = windows.Window(x, y, width, height)  # type: ignore
            transform = windows.transform(window, ds.transform)
            yield window, transform


def get_tile(
    src: DatasetReader,
    index: int,
    tile_width: int,
    tile_height: int,
    overlap: int = 0,
):
    """Fetch a specific tile by index, computing the window based on the index."""

    # Compute the number of tiles in both x and y directions
    ncols, nrows = src.width, src.height
    xstep = tile_width
    ystep = tile_height
    max_x = ncols // tile_width
    max_y = nrows // tile_height
    index_x = index // max_x
    index_y = index % max_x
    x = index_x * tile_width
    y = index_y * tile_height
    window = windows.Window(x, y, tile_width, tile_height)  # type: ignore
    transform = windows.transform(window, src.transform)
    return window, transform


def get_tiles_length(
    src: DatasetReader, tile_width: int, tile_height: int, overlap: int = 0
) -> int:
    ncols, nrows = src.width, src.height
    return (nrows // tile_width) * (ncols // tile_height)


@dataclass
class Grid:
    path: Path

    def get_tiles_length(
        self, tile_width: int, tile_height: int, overlap: int = 0
    ) -> int:
        with rasterio.open(self.path) as src:
            ncols, nrows = src.width, src.height
            return (nrows // tile_width) * (ncols // tile_height)

    def get_tile(
        self, index: int, tile_width: int, tile_height: int, overlap: int = 0
    ) -> tuple[Metadata, np.ndarray, Polygon]:
        with rasterio.open(self.path) as src:
            metadata = src.meta.copy()

            window, transform = get_tile(
                src, index, tile_width, tile_height, overlap
            )
            metadata['transform'] = transform
            metadata['width'], metadata['height'] = (
                window.width,
                window.height,
            )
            bounds = box(*windows.bounds(window, src.transform))
            return (
                metadata,
                src.read(window=window),
                bounds,
            )

    def get_tiles(
        self, tile_width, tile_height, overlap: int = 0
    ) -> c.Generator[tuple[Metadata, np.ndarray]]:
        tiles_length = self.get_tiles_length(tile_width, tile_height, overlap)
        with rasterio.open(self.path) as src:
            metadata = src.meta.copy()
            for i in range(tiles_length):
                window, transform = get_tile(src, i, tile_width, tile_height)
                metadata['transform'] = transform
                metadata['width'], metadata['height'] = (
                    window.width,
                    window.height,
                )
                yield (
                    metadata,
                    src.read(window=window),
                )

    def get_features(
        self,
        window: windows.Window | None = None,
        metadata: Metadata | None = None,
    ) -> list[dict[str, t.Any]]:
        metadata = metadata or {}
        with rasterio.open(self.path, window=window, **metadata) as raster:
            return list(
                dataset_features(
                    raster,
                    bidx=1,
                    as_mask=True,
                    geographic=False,
                    band=False,
                )
            )


@dataclass
class RasterTiles:
    tiles: gpd.GeoDataFrame = field(repr=False)
    parent_dir: Path

    @classmethod
    def from_dir(
        cls, tiles_parent: Path, mode: Mode, suffix: str = '.tif'
    ) -> t.Self:
        geojson = get_empty_geojson()
        images = tiles_parent.rglob(f'*{suffix}')

        for image in images:
            features = Grid(image).get_features()
            processed_features = [
                process_feature(feature, mode, image, tiles_parent)
                for feature in features
            ]
            if len(processed_features) == 0:
                logger.error(
                    'Failed to process dataset feature for raster %r' % image
                )
            geojson['features'].extend(processed_features)
        return cls(geojson_to_gdf(geojson), tiles_parent)

    @classmethod
    def from_raster(
        cls,
        raster: Path,
        tile_size: TileSize,
        overlap: int,
        out_dir: Path,
        mode: Mode,
        suffix: str = '.tif',
    ) -> t.Self:
        with rasterio.open(raster, mode='r') as src:
            metadata = src.meta.copy()

            def handle_window(input: tuple[windows.Window, t.Any]):
                window, transform = input
                metadata['transform'] = transform
                metadata['width'], metadata['height'] = (
                    window.width,
                    window.height,
                )
                out_filepath = (
                    out_dir / f'{window.col_off}_{window.row_off}{suffix}'
                )

                with rasterio.open(out_filepath, 'w', **metadata) as dst:
                    dst.write(src.read(window=window))

            for i in range(
                get_tiles_length(src, tile_size.columns, tile_size.rows)
            ):
                handle_window(
                    get_tile(src, i, tile_size.columns, tile_size.rows)
                )

        return cls.from_dir(out_dir, mode, suffix)

    def add_overlap(
        self,
        overlap: int,
        mode: Mode,
        tile_size: TileSize,
    ) -> t.Self:
        merged = self.merge(self.parent_dir / 'merged.tif')
        parent_dir = self.parent_dir.parent / f'overlap_{self.parent_dir.name}'
        with rasterio.open(merged) as src:
            metadata = src.meta.copy()

            for window, transform in get_tiles(
                src,
                tile_size.rows,
                tile_size.columns,
                overlap,
            ):
                metadata['transform'] = transform
                metadata['width'], metadata['height'] = (
                    window.width,
                    window.height,
                )
                bounds = box(*windows.bounds(window, src.transform))
                out_filepath = parent_dir / get_tile_relative_path(
                    self.tiles, bounds
                )
                os.makedirs(out_filepath.parent, exist_ok=True)
                with rasterio.open(out_filepath, 'w', **metadata) as dst:
                    dst.write(src.read(window=window))
        return self.from_dir(parent_dir, mode)

    def resample(
        self,
        out_dir: Path,
        mode: Mode,
        resampling: Resampling = Resampling.bilinear,
    ) -> t.Self:
        os.makedirs(out_dir, exist_ok=True)
        crs = rasterio.crs.CRS.from_epsg(EPSG)

        def from_path(tile_path: str):
            path = self.parent_dir / tile_path
            dest = out_dir / Path(tile_path).name
            os.makedirs(dest.parent, exist_ok=True)
            with rasterio.open(path) as ds:
                arr = ds.read(1)
                meta = ds.meta.copy()
                newaff, width, height = (
                    rasterio.warp.calculate_default_transform(
                        ds.crs,
                        ds.crs,
                        ds.width,
                        ds.height,
                        *ds.bounds,
                        resolution=RASTER_CELL_SIZE,
                    )
                )
                newarr = np.ma.asanyarray(
                    np.empty(
                        shape=(1, t.cast(int, height), t.cast(int, width)),
                        dtype=arr.dtype,
                    )
                )
                rasterio.warp.reproject(
                    arr,
                    newarr,
                    src_transform=ds.transform,
                    dst_transform=newaff,
                    width=width,
                    height=height,
                    src_nodata=NODATA,
                    dst_nodata=NODATA,
                    src_crs=crs,
                    dst_crs=crs,
                    resample=resampling,
                )
                meta.update(
                    {
                        'transform': newaff,
                        'width': width,
                        'height': height,
                        'nodata': NODATA,
                        'crs': crs,
                    }
                )
                with rasterio.open(dest, mode='w', **meta) as dest_raster:
                    dest_raster.write(newarr)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(from_path, self.tiles['path'])
            for result in results:
                if isinstance(result, Exception):
                    raise result

        return self.from_dir(out_dir, mode)

    def merge(self, out_file: Path) -> Path:
        paths = (
            self.parent_dir.as_posix()
            + '/'
            + self.tiles['path'].str.lstrip('/')
        )
        out_file.unlink(missing_ok=True)
        rasterio.merge.merge(
            paths.values,
            res=RASTER_CELL_SIZE,
            resampling=Resampling.bilinear,
            dst_path=out_file,
            nodata=NODATA,
        )
        return out_file


class ResizeTensor:
    def __init__(self, size):
        self.size = size

    def __call__(self, img: Tensor) -> Tensor:
        return functional.resize(img, self.size)


def get_default_transform():
    return Compose(
        [
            ToTensor(),
            ResizeTensor((224, 224)),
            Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
            Normalize(mean=0.5, std=0.5),
        ]
    )


def get_default_augument_transform():
    return Compose(
        [
            RandomHorizontalFlip(p=0.5),
            RandomApply(
                [
                    Lambda(
                        lambda img: img.rotate(np.random.choice([90, 180, 270]))
                    )
                ],
                p=1.0,
            ),
        ]
    )


def get_default_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()
        return img


def get_landslide_density_threshold(tile_size: TileSize) -> float:
    """The LANDSLIDE_DENSITY_THRESHOLD is relative to the DEFAULT_TILE_SIZE."""
    default_area = DEFAULT_TILE_SIZE * DEFAULT_TILE_SIZE
    return LANDSLIDE_DENSITY_THRESHOLD * (
        (tile_size.rows * tile_size.columns) / default_area
    )


class LandslideImages(Dataset):
    def __init__(
        self,
        grid: Grid,
        tile_size: TileSize,
        landslide_density_threshold: float = LANDSLIDE_DENSITY_THRESHOLD,
        transforms: c.Callable | None = None,
        transform: c.Callable | None = None,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.landslide_density_threshold = landslide_density_threshold
        self.tile_size = tile_size
        self.transform = transform or get_default_transform()
        self.transforms = transforms
        self.mode = (
            Mode.TRAIN
            if grid.path.is_relative_to(GRIDS / 'train')
            else Mode.TEST
        )
        if self.mode is Mode.TRAIN:
            self.data_indices = self._get_data_indices()

        # TODO FIX tile_size col rows

    @staticmethod
    def augment_tensor(tensor):
        """Apply a random flip and/or 90-degree rotation."""
        if np.random.random() < 0.5:
            tensor = functional.hflip(tensor)
        angle = np.random.choice([90, 180, 270])
        tensor = functional.rotate(tensor, int(angle))
        return tensor

    def _get_data_indices(self):
        start = time.perf_counter()
        indices = {}

        def handle_tile(i):
            _, class_ = self._get_tile(i)
            indices.setdefault(class_, []).append(i)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(
                handle_tile,
                range(
                    self.grid.get_tiles_length(
                        self.tile_size.columns, self.tile_size.rows
                    )
                ),
            )

        end = time.perf_counter()
        logger.info(
            'Took %f seconds to compute data indices for %r.Length of classes: %r'
            % (
                end - start,
                self.tile_size,
                {k: len(v) for k, v in indices.items()},
            )
        )
        return indices

    def __len__(self) -> int:
        # return self.grid.get_tiles_length(
        #     self.tile_size.columns, self.tile_size.rows
        # )
        if self.mode is Mode.TRAIN:
            return max(map(len, self.data_indices.values())) * len(
                self.data_indices
            )  # Make the dataset balanced
        elif self.mode == Mode.TEST:
            return self.grid.get_tiles_length(
                self.tile_size.columns, self.tile_size.rows
            )
        raise ValueError('Mode should only be "train" or "test"')

    def _get_tile(self, index: int) -> tuple[torch.Tensor, int]:
        metadata, arr, bounds = self.grid.get_tile(
            index, self.tile_size.columns, self.tile_size.rows
        )
        # print(index, arr.shape)
        # test_dir = INTERIM_DATA_DIR / 'test_dir'
        # test_dir.mkdir(exist_ok=True)
        # with rasterio.open(
        #     test_dir / f'{tile_class}_{index}.tif', mode='w', **metadata
        # ) as dst:
        #     dst.write(tile)

        tile_class = self._classify_tile(bounds)
        tile = self.transform(arr.squeeze(0))
        assert isinstance(tile, torch.Tensor)
        return tile, tile_class

    def _get_item_train(self, index: int) -> tuple[torch.Tensor, int]:
        cls = index % len(self.data_indices)  # Cycle through classes
        class_idx = index // len(self.data_indices)  # Which sample in the class
        if class_idx >= len(self.data_indices[cls]):
            # print(cls, class_idx, 'AUGUMENTED')
            real_idx = np.random.choice(
                self.data_indices[cls]
            )  # Randomly sample from real images
            tile, label = self._get_tile(real_idx)

            tile = self.augment_tensor(tile)
        else:
            # print(cls, class_idx)
            real_idx = self.data_indices[cls][class_idx]
            tile, label = self._get_tile(real_idx)
        return (tile, label)

    def _get_item_test(self, index: int) -> tuple[torch.Tensor, int]:
        return self._get_tile(index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self.mode is Mode.TRAIN:
            return self._get_item_train(index)
        elif self.mode is Mode.TEST:
            return self._get_item_test(index)
        raise ValueError('Mode should only be "train" or "test"')

    def _classify_tile(self, bounds: Polygon) -> int:
        """Classify the tile based on landslide density threshold."""
        perc = get_percentage_intersection(
            bounds, get_landslide_shapes(mode=self.mode)
        )
        threshold = get_landslide_density_threshold(self.tile_size)
        if perc >= threshold:
            return LandslideClass.LANDSLIDE.value
        else:
            return LandslideClass.NO_LANDSLIDE.value


def get_dem_cell_size(raster: Path) -> tuple[float, float]:
    with rasterio.open(raster, mode='r') as r:
        x, y = r.res
    return (x, y)


def get_landslide_images_for_variable(
    variable: GeomorphometricalVariable, tile_size: TileSize, mode: Mode
) -> LandslideImages:
    key = (variable, tile_size, mode)

    # for cached_key in _cache:
    #     if cached_key == key:
    #         return _cache[cached_key]

    # Compute and store if not found
    result = LandslideImages(
        Grid((GRIDS / mode.value / variable.value).with_suffix('.tif')),
        tile_size,
    )
    # _cache[key] = result
    return result


T = t.TypeVar('T', bound=GeomorphometricalVariable)


def get_image_folders_for_variable(
    variable: T, tile_size: TileSize
) -> tuple[T, ImageFolders]:
    image_folders = {}
    for mode in Mode:
        image_folders[mode] = get_landslide_images_for_variable(
            variable, tile_size, mode
        )
    return (variable, t.cast(ImageFolders, image_folders))


def get_image_folders(
    tile_size: TileSize,
) -> dict[GeomorphometricalVariable, ImageFolders]:
    func = partial(get_image_folders_for_variable, tile_size=tile_size)
    logger.info(
        'Creating tiles with %r for all of the geomorphometrical variables'
        % tile_size
    )
    # results = [func(variable) for variable in GeomorphometricalVariable]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(func, GeomorphometricalVariable)
    return {variable: image_folders for variable, image_folders in results}
