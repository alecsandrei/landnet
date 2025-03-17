from __future__ import annotations

import collections.abc as c
import concurrent.futures
import json
import os
import typing as t
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.crs
import rasterio.mask
import rasterio.merge
import rasterio.warp
from PIL import Image
from rasterio import DatasetReader, windows
from rasterio.enums import Resampling
from rasterio.features import dataset_features
from shapely import MultiPolygon, Polygon, box, from_geojson
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToTensor

from landnet.config import (
    EPSG,
    GRIDS,
    LANDSLIDE_DENSITY_THRESHOLD,
    NODATA,
    RASTER_CELL_SIZE,
    TEST_TILES,
    TRAIN_TILES,
)
from landnet.dataset import (
    Feature,
    geojson_to_gdf,
    get_empty_geojson,
    get_tile_relative_path,
    read_landslide_shapes,
)
from landnet.features.grids import GeomorphometricalVariable
from landnet.logger import create_logger

logger = create_logger(__name__)

PathLike = os.PathLike | str
LandslideTile = int
LandslideDensity = float
TilePaths = dict[int, Path]
ClassFolder = Path
Mode = t.Literal['train', 'test']


@dataclass
class TileSize:
    rows: int
    columns: int

    def __str__(self):
        return f'{self.rows}x{self.columns}'


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
            break
        for y in range(0, nrows + ystep, ystep):
            if y != 0:
                y -= overlap
            height = tile_height + overlap * (2 if y != 0 else 1)
            if y + height > nrows:
                break
            window = windows.Window(x, y, width, height)  # type: ignore
            transform = windows.transform(window, ds.transform)
            yield window, transform


@dataclass
class RasterTiles:
    tiles: gpd.GeoDataFrame = field(repr=False)
    parent_dir: Path

    @classmethod
    def from_dir(
        cls, tiles_parent: Path, mode: Mode, suffix: str = '.tif'
    ) -> t.Self:
        landslide_shapes = read_landslide_shapes(mode)
        geojson = get_empty_geojson()
        images = tiles_parent.rglob(f'*{suffix}')

        def get_percentage_intersection(
            feature: Polygon | MultiPolygon,
        ) -> float:
            intersection = landslide_shapes.intersection(feature).union_all()
            intersection_area = (
                intersection.area if not intersection.is_empty else 0
            )
            polygon_area = feature.area
            if polygon_area == 0:
                return 0
            return intersection_area / polygon_area

        def process_feature(feature: dict[str, t.Any], path: Path) -> Feature:
            keys_to_remove = [
                k
                for k in feature
                if k not in Feature.__required_keys__
                and k not in Feature.__optional_keys__
            ]
            for k in keys_to_remove:
                feature.pop(k)
            feature['properties'] = {
                'path': path.relative_to(tiles_parent).as_posix(),
                'mode': mode,
                'landslide_density': get_percentage_intersection(
                    t.cast(
                        Polygon | MultiPolygon,
                        from_geojson(json.dumps(feature['geometry'])),
                    )
                ),
            }
            return t.cast(Feature, feature)

        def get_features(tif: Path) -> list[Feature]:
            with rasterio.open(tif) as raster:
                features = [
                    process_feature(feature, tif)
                    for feature in dataset_features(
                        raster,
                        bidx=1,
                        as_mask=True,
                        geographic=False,
                        band=False,
                    )
                ]

            return features

        for i, image in enumerate(list(images)):
            features = get_features(image)
            if len(features) == 0:
                logger.error(
                    'Failed to process dataset feature for raster %r' % image
                )
            geojson['features'].extend(features)
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
        with rasterio.open(raster) as src:
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

            for input in get_tiles(
                src,
                tile_size.rows,
                tile_size.columns,
                overlap,
            ):
                handle_window(input)

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


def get_default_transform():
    return Compose(
        [
            Resize((224, 224)),
            ToTensor(),
            Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
            Normalize(mean=0.5, std=0.5),
        ]
    )


def get_default_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()
        return img


class LandslideClass(Enum):
    NO_LANDSLIDE = 0
    LANDSLIDE = 1


class LandslideImageFolder(VisionDataset):
    def __init__(
        self,
        raster_tiles: RasterTiles,
        loader: c.Callable[[str], t.Any] = get_default_loader,
        landslide_density_threshold: float = LANDSLIDE_DENSITY_THRESHOLD,
        transforms: c.Callable | None = None,
        transform: c.Callable | None = None,
        target_transform: c.Callable | None = None,
    ) -> None:
        if transform is None:
            transform = get_default_transform()
        super().__init__(
            raster_tiles.parent_dir, transforms, transform, target_transform
        )
        self.samples = self.make_dataset(
            raster_tiles, landslide_density_threshold
        )
        self.loader = loader

    def __getitem__(self, index: int) -> tuple[t.Any, t.Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def make_dataset(
        raster_tiles: RasterTiles,
        landslide_density_threshold: float = 0.05,
    ) -> list[tuple[str, int]]:
        samples = []
        for tile in raster_tiles.tiles.itertuples():
            path = t.cast(str, tile.path)
            density = t.cast(float, tile.landslide_density)
            tile_path = raster_tiles.parent_dir / path
            tile_class = (
                LandslideClass.LANDSLIDE.value
                if density >= landslide_density_threshold
                else LandslideClass.NO_LANDSLIDE.value
            )
            samples.append((tile_path.as_posix(), tile_class))
        return samples


def get_dem_cell_size(raster: Path) -> tuple[float, float]:
    with rasterio.open(raster, mode='r') as r:
        x, y = r.res
    return (x, y)


class ImageFolders(t.TypedDict):
    train: LandslideImageFolder
    test: LandslideImageFolder


T = t.TypeVar('T', bound=GeomorphometricalVariable)


def get_image_folders_for_variable(
    variable: T, tile_size: TileSize
) -> tuple[T, ImageFolders]:
    image_folders = {}
    for mode in ('train', 'test'):
        out_dir = (
            (TRAIN_TILES if mode == 'train' else TEST_TILES)
            / variable.value
            / str(tile_size)
        )
        if False and out_dir.exists():
            tiles = RasterTiles.from_dir(out_dir, mode)
        else:
            os.makedirs(out_dir, exist_ok=True)

            grid = (GRIDS / mode / variable.value).with_suffix('.tif')
            tiles = RasterTiles.from_raster(
                grid,
                tile_size,
                overlap=0,
                mode=mode,
                out_dir=out_dir,
            )
        image_folders[mode] = LandslideImageFolder(tiles)
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
