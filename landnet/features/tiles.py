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
from shapely.geometry.base import BaseGeometry
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision.transforms import (
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
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
from landnet.modelling.models import apply_pca_on_channels

if t.TYPE_CHECKING:
    from torch import Tensor

logger = create_logger(__name__)


@dataclass
class TileSize:
    width: int
    height: int

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self):
        return f'{self.width}x{self.height}'

    @classmethod
    def from_string(cls, string: str):
        split = string.split('x')
        return TileSize(int(split[0]), int(split[1]))


@dataclass
class TileConfig:
    size: TileSize
    overlap: int = 0


@dataclass
class TileHandler:
    config: TileConfig

    def _compute_tiles_grid(
        self, src: DatasetReader
    ) -> tuple[int, int, int, int]:
        """Compute the number of tiles and step sizes in x and y directions."""
        ncols, nrows = src.width, src.height
        xstep = self.config.size.width - self.config.overlap
        ystep = self.config.size.height - self.config.overlap
        max_x = (ncols - self.config.overlap) // xstep
        max_y = (nrows - self.config.overlap) // ystep
        return ncols, nrows, max_x, max_y

    def get_tiles(
        self, src: DatasetReader
    ) -> c.Generator[tuple[windows.Window, windows.Affine], None, None]:
        """Generate all tiles with overlap."""
        ncols, nrows, max_x, max_y = self._compute_tiles_grid(src)

        for index_x in range(max_x):
            x = index_x * (self.config.size.width - self.config.overlap)
            for index_y in range(max_y):
                y = index_y * (self.config.size.height - self.config.overlap)

                window = windows.Window(
                    x, y, self.config.size.width, self.config.size.height
                )
                transform = windows.transform(window, src.transform)
                yield window, transform

    def get_tile(
        self, src: DatasetReader, tile_index: int
    ) -> tuple[windows.Window, windows.Affine]:
        """Fetch a specific tile by index."""
        _, _, max_x, max_y = self._compute_tiles_grid(src)

        index_x = tile_index % max_x
        index_y = tile_index // max_x
        x = index_x * (self.config.size.width - self.config.overlap)
        y = index_y * (self.config.size.height - self.config.overlap)

        window = windows.Window(
            x, y, self.config.size.width, self.config.size.height
        )
        transform = windows.transform(window, src.transform)
        return window, transform

    def get_tiles_length(self, src: DatasetReader) -> int:
        """Return the total number of tiles."""
        _, _, max_x, max_y = self._compute_tiles_grid(src)
        return max_x * max_y

    # def get_tiles(self, src: DatasetReader):
    #     ncols, nrows = src.meta['width'], src.meta['height']
    #     xstep = self.config.size.width
    #     ystep = self.config.size.height
    #     for x in range(0, ncols + xstep, xstep):
    #         if x != 0:
    #             x -= self.config.overlap
    #         width = self.config.size.width + self.config.overlap * (
    #             2 if x != 0 else 1
    #         )
    #         if x + width > ncols:
    #             # logger.warning('Could not create tiles from x=%i' % x)
    #             break
    #         for y in range(0, nrows + ystep, ystep):
    #             if y != 0:
    #                 y -= self.config.overlap
    #             height = self.config.size.height + self.config.overlap * (
    #                 2 if y != 0 else 1
    #             )
    #             if y + height > nrows:
    #                 # logger.warning('Could not create tiles from y=%i' % y)
    #                 break
    #             window = windows.Window(x, y, width, height)  # type: ignore
    #             transform = windows.transform(window, src.transform)
    #             yield window, transform


def get_raster_features(
    path: Path,
    window: windows.Window | None = None,
    metadata: Metadata | None = None,
) -> list[dict[str, t.Any]]:
    metadata = metadata or {}
    with rasterio.open(path, window=window, **metadata) as raster:
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
class Grid:
    path: Path
    tile_config: TileConfig
    tile_handler: TileHandler = field(init=False)
    cached_tile: dict[int, tuple[Metadata, np.ndarray, Polygon]] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self):
        if self.tile_config:
            self.tile_handler = TileHandler(self.tile_config)

    def get_tiles_length(self) -> int:
        assert self.tile_handler is not None
        with rasterio.open(self.path) as src:
            return self.tile_handler.get_tiles_length(src)

    def get_landslide_percentage_intersection(
        self, indices: c.Sequence[int], mode: Mode
    ) -> gpd.GeoDataFrame:
        assert self.tile_handler is not None
        bounds = gpd.GeoSeries()
        bounds.set_crs(inplace=True, epsg=EPSG)
        with rasterio.open(self.path) as src:
            for i in indices:
                window, _ = self.tile_handler.get_tile(src, i)
                bounds[i] = box(*windows.bounds(window, src.transform))
        landslides = get_landslide_shapes(mode)
        bounds = t.cast(gpd.GeoDataFrame, bounds.to_frame('geometry'))
        bounds['landslide_density'] = bounds['geometry'].apply(
            lambda geom: get_percentage_intersection(geom, other=landslides)  # type: ignore
        )
        return bounds

    def get_tile_bounds(
        self, index: int
    ) -> tuple[Metadata, windows.Window, Polygon]:
        assert self.tile_handler is not None

        with rasterio.open(self.path) as src:
            metadata = src.meta.copy()

            window, transform = self.tile_handler.get_tile(src, index)
            metadata['transform'] = transform
            metadata['width'], metadata['height'] = (
                window.width,
                window.height,
            )
            bounds = box(*windows.bounds(window, src.transform))
            return (metadata, window, bounds)

    def get_tile(self, index: int) -> tuple[Metadata, np.ndarray, Polygon]:
        if (vals := self.cached_tile.get(index, None)) is None:
            with rasterio.open(self.path) as src:
                metadata = src.meta.copy()

                metadata, window, bounds = self.get_tile_bounds(index)
                self.cached_tile[index] = (
                    metadata,
                    src.read(window=window),
                    bounds,
                )
                return self.cached_tile[index]
        return vals

    def get_tiles(self) -> c.Generator[tuple[Metadata, np.ndarray]]:
        assert self.tile_handler is not None

        with rasterio.open(self.path) as src:
            metadata = src.meta.copy()
            for i in range(self.get_tiles_length()):
                window, transform = self.tile_handler.get_tile(src, i)
                metadata['transform'] = transform
                metadata['width'], metadata['height'] = (
                    window.width,
                    window.height,
                )
                yield (
                    metadata,
                    src.read(window=window),
                )

    def mask(
        self,
        geometry: BaseGeometry,
        overwrite: bool = False,
        mask_kwargs: dict[str, t.Any] | None = None,
    ) -> np.ndarray:
        if mask_kwargs is None:
            mask_kwargs = {}
        mask_kwargs.setdefault('crop', True)
        mask_kwargs.setdefault('filled', True)
        with rasterio.open(self.path) as src:
            out_image, transformed = rasterio.mask.mask(
                src, [geometry], **mask_kwargs
            )
            out_profile = src.profile.copy()

            out_profile.update(
                {
                    'width': out_image.shape[2],
                    'height': out_image.shape[1],
                    'transform': transformed,
                }
            )
            out_profile.setdefault('crs', rasterio.crs.CRS.from_epsg(EPSG))
        if overwrite:
            with rasterio.open(self.path, 'w', **out_profile) as dst:
                dst.write(out_image)
        return out_image


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
            features = get_raster_features(image)
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
        config: TileConfig,
        out_dir: Path,
        mode: Mode,
        suffix: str = '.tif',
    ) -> t.Self:
        handler = TileHandler(config)
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

            for i in range(handler.get_tiles_length(src)):
                handle_window(handler.get_tile(src, i))

        return cls.from_dir(out_dir, mode, suffix)

    def add_overlap(self, mode: Mode, tile_config: TileConfig) -> t.Self:
        handler = TileHandler(tile_config)
        merged = self.merge(self.parent_dir / 'merged.tif')
        parent_dir = self.parent_dir.parent / f'overlap_{self.parent_dir.name}'
        with rasterio.open(merged) as src:
            metadata = src.meta.copy()

            for window, transform in handler.get_tiles(src):
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
        tile_size: TileSize = TileSize(5, 5),
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
                        resolution=(tile_size.width, tile_size.height),
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

    def merge(
        self, out_file: Path, merge_kwargs: dict[str, t.Any] | None = None
    ) -> Path:
        paths = (
            self.parent_dir.as_posix()
            + '/'
            + self.tiles['path'].str.lstrip('/')
        )
        out_file.unlink(missing_ok=True)
        if merge_kwargs is None:
            merge_kwargs = {}
        merge_kwargs.setdefault('nodata', NODATA)
        merge_kwargs.setdefault('resampling', Resampling.bilinear)
        rasterio.merge.merge(paths.values, dst_path=out_file, **merge_kwargs)
        return out_file


class ResizeTensor:
    def __init__(self, size):
        self.size = size

    def __call__(self, img: Tensor) -> Tensor:
        return functional.resize(img, self.size)


class RotateTensor:
    def __init__(self, angles: c.Sequence[int]):
        self.angles = angles

    def __call__(self, img: Tensor) -> Tensor:
        choice = np.random.choice(self.angles)
        return functional.rotate(img, int(choice))


def get_default_transform():
    return Compose(
        [
            ToTensor(),
            ResizeTensor((224, 224)),
            Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
            Normalize(mean=0.5, std=0.5),
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
        (tile_size.width * tile_size.height) / default_area
    )


class PCAConcatLandslideImages(Dataset):
    def __init__(
        self, landslide_images: c.Sequence[LandslideImages], num_components: int
    ):
        self.landslide_images = landslide_images
        self.concat = ConcatLandslideImages(self.landslide_images)
        self.data_indices = self.concat.data_indices  # type: ignore
        self.num_components = num_components

    def __len__(self):
        return len(self.landslide_images[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Return the PCA-reduced image.
        """

        tile, class_ = self.concat[index]
        return (apply_pca_on_channels(tile, self.num_components), class_)


class ConcatLandslideImages(Dataset):
    def __init__(
        self,
        landslide_images: c.Sequence[LandslideImages],
    ):
        self.landslide_images = landslide_images
        self.data_indices = landslide_images[0].data_indices

    def __len__(self):
        return len(self.landslide_images[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        # print(self.i)
        # self.i += 1
        image_batch = [images[index] for images in self.landslide_images]
        cat = torch.cat([batch[0] for batch in image_batch], dim=0)
        class_ = image_batch[0][1]
        assert all(batch[1] == class_ for batch in image_batch[1:])
        return (cat, class_)


def get_default_augument_transform():
    return Compose(
        [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RotateTensor([0, 90, 180, 270]),
        ]
    )


DEFAULT_CLASS_BALANCE = {
    LandslideClass.NO_LANDSLIDE: 0.5,
    LandslideClass.LANDSLIDE: 0.5,
}


def get_dataloader(
    dataset: Dataset, weights: np.ndarray, size: int | None = None, **kwargs
):
    samples_weight = torch.tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        samples_weight,  # type: ignore
        num_samples=size or len(samples_weight),
        replacement=True,
    )
    data_loader = DataLoader(dataset, **kwargs, sampler=sampler)
    return data_loader


def create_dataloader_from_subset(
    dataset: Subset[LandslideImages | ConcatLandslideImages],
    class_balance: dict[LandslideClass, float] | None = None,
    size: int | None = None,
    **kwargs,
):
    # Count occurrences of each class
    class_sample_count = {
        cls: len([index for index in indices if index in dataset.indices])
        for cls, indices in dataset.dataset.data_indices.items()  # type: ignore
    }

    # Compute inverse frequency weights
    weight = {
        cls: 1.0 / count if count > 0 else 0
        for cls, count in class_sample_count.items()
    }

    # Assign sample weights based on their class
    samples_weight = np.zeros(len(dataset.dataset))  # type: ignore

    for cls, indices in dataset.dataset.data_indices.items():  # type: ignore
        indices = [index for index in indices if index in dataset.indices]
        class_weight = weight[cls]
        if class_balance is not None:
            class_weight *= class_balance[LandslideClass(cls)]
        samples_weight[indices] = [class_weight] * len(indices)

    return get_dataloader(
        dataset.dataset, weights=samples_weight, size=size, **kwargs
    )


def create_dataloader(
    dataset: LandslideImages
    | ConcatLandslideImages
    | Subset[LandslideImages | ConcatLandslideImages],
    class_balance: dict[LandslideClass, float] | None = None,
    size: int | None = None,
    **kwargs,
) -> DataLoader:
    if isinstance(dataset, Subset):
        return create_dataloader_from_subset(
            dataset, class_balance=class_balance, size=size, **kwargs
        )
    # Count occurrences of each class
    class_sample_count = {
        cls: len(indices) for cls, indices in dataset.data_indices.items()
    }

    # Compute inverse frequency weights
    weight = {
        cls: 1.0 / count if count > 0 else 0
        for cls, count in class_sample_count.items()
    }

    # Assign sample weights based on their class
    samples_weight = np.zeros(len(dataset))

    for cls, indices in dataset.data_indices.items():
        if isinstance(dataset, Subset):
            indices = [index for index in indices if index in dataset.indices]
        class_weight = weight[cls]
        if class_balance is not None:
            class_weight *= class_balance[LandslideClass(cls)]
        samples_weight[indices] = [class_weight] * len(indices)
    return get_dataloader(dataset, weights=samples_weight, size=size, **kwargs)


class LandslideImages(Dataset):
    def __init__(
        self,
        grid: Grid,
        landslide_density_threshold: float = LANDSLIDE_DENSITY_THRESHOLD,
        transforms: c.Callable | None = None,
        transform: c.Callable | None = None,
    ) -> None:
        super().__init__()
        self.overlap = 0
        self.grid = grid
        self.landslide_density_threshold = landslide_density_threshold
        self.transform = transform or get_default_transform()
        self.augument_transform = get_default_augument_transform()
        self.transforms = transforms
        self.mode = (
            Mode.TRAIN
            if grid.path.is_relative_to(GRIDS / 'train')
            else Mode.TEST
        )
        self.data_indices = self._get_data_indices()

    def _get_data_indices(self) -> dict[int, list[int]]:
        start = time.perf_counter()
        class_to_indices: dict[int, list[int]] = {}
        indices_range = range(self.grid.get_tiles_length())
        gdf = self.grid.get_landslide_percentage_intersection(
            list(indices_range), self.mode
        )
        for row in gdf.itertuples():
            class_ = (
                LandslideClass.LANDSLIDE.value
                if t.cast(float, row.landslide_density)
                >= self.landslide_density_threshold
                else LandslideClass.NO_LANDSLIDE.value
            )
            assert isinstance(row.Index, t.SupportsInt)
            class_to_indices.setdefault(class_, []).append(int(row.Index))
        end = time.perf_counter()
        logger.info(
            'Took %f seconds to compute data indices for %r at mode=%r. Length of classes: %r'
            % (
                end - start,
                self.grid.tile_config.size,
                self.mode.value,
                {k: len(v) for k, v in class_to_indices.items()},
            )
        )
        return class_to_indices

    def __len__(self) -> int:
        return self.grid.get_tiles_length()

    def _get_tile_class(self, index: int) -> int:
        for class_, indices in self.data_indices.items():
            if index in indices:
                return class_
        raise ValueError

    def _get_tile(self, index: int) -> tuple[torch.Tensor, int]:
        _, arr, _ = self.grid.get_tile(index)
        tile_class = self._get_tile_class(index)
        tile = self.transform(arr.squeeze(0))
        assert isinstance(tile, torch.Tensor)
        return tile, tile_class

    def _get_item_train(self, index: int) -> tuple[torch.Tensor, int]:
        tile, label = self._get_tile(index)
        tile = self.augument_transform(tile)
        return (tile, label)

    def _get_item_test(self, index: int) -> tuple[torch.Tensor, int]:
        return self._get_tile(index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self.mode is Mode.TRAIN:
            return self._get_item_train(index)
        elif self.mode is Mode.TEST:
            return self._get_item_test(index)
        raise ValueError('Mode should only be "train" or "test"')


def get_dem_cell_size(raster: Path) -> tuple[float, float]:
    with rasterio.open(raster, mode='r') as r:
        x, y = r.res
    return (x, y)


def get_landslide_images_for_variable(
    variable: GeomorphometricalVariable, tile_config: TileConfig, mode: Mode
) -> LandslideImages:
    result = LandslideImages(
        Grid(
            (GRIDS / mode.value / variable.value).with_suffix('.tif'),
            tile_config,
        ),
    )
    return result


T = t.TypeVar('T', bound=GeomorphometricalVariable)


def get_image_folders_for_variable(
    variable: T, tile_config: TileConfig
) -> tuple[T, ImageFolders]:
    image_folders = {}
    for mode in Mode:
        image_folders[mode] = get_landslide_images_for_variable(
            variable, tile_config, mode
        )
    return (variable, t.cast(ImageFolders, image_folders))


def get_image_folders(
    tile_config: TileConfig,
) -> dict[GeomorphometricalVariable, ImageFolders]:
    func = partial(get_image_folders_for_variable, tile_config=tile_config)
    logger.info(
        'Creating tiles with %r for all of the geomorphometrical variables'
        % tile_config
    )
    # results = [func(variable) for variable in GeomorphometricalVariable]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(func, GeomorphometricalVariable)
    return {variable: image_folders for variable, image_folders in results}
