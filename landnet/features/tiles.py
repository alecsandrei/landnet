from __future__ import annotations

import collections.abc as c
import concurrent.futures
import os
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.crs
import rasterio.mask
import rasterio.merge
import rasterio.warp
from rasterio import DatasetReader, windows
from rasterio.enums import Resampling
from rasterio.features import dataset_features
from shapely import box

from landnet.config import (
    EPSG,
    NODATA,
)
from landnet.enums import Mode
from landnet.features.dataset import (
    geojson_to_gdf,
    get_empty_geojson,
    get_tile_relative_path,
    process_feature,
)
from landnet.logger import create_logger
from landnet.typing import Metadata

logger = create_logger(__name__)


@dataclass
class TileSize:
    width: float
    height: float

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self):
        return f"{self.width}x{self.height}"

    @classmethod
    def from_string(cls, string: str):
        split = string.split("x")
        return TileSize(float(split[0]), float(split[1]))


@dataclass
class TileConfig:
    size: TileSize
    overlap: int = 0


@dataclass
class TileHandler:
    config: TileConfig

    def _compute_tiles_grid(self, src: DatasetReader) -> tuple[int, int, int, int]:
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

        window = windows.Window(x, y, self.config.size.width, self.config.size.height)
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
    metadata = metadata if metadata else {}
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
class RasterTiles:
    tiles: gpd.GeoDataFrame = field(repr=False)
    parent_dir: Path

    @classmethod
    def from_dir(cls, tiles_parent: Path, mode: Mode, suffix: str = ".tif") -> t.Self:
        geojson = get_empty_geojson()
        images = list(tiles_parent.rglob(f"*{suffix}"))
        if not images:
            logger.error(
                "No raster images found in %r with suffix %s" % (tiles_parent, suffix)
            )
            return cls(gpd.GeoDataFrame(geojson), tiles_parent)

        for image in images:
            features = get_raster_features(image)
            processed_features = [
                process_feature(feature, mode, image, tiles_parent)
                for feature in features
            ]
            if len(processed_features) == 0:
                logger.error("Failed to process dataset feature for raster %r" % image)
            geojson["features"].extend(processed_features)
        return cls(geojson_to_gdf(geojson), tiles_parent)

    @classmethod
    def from_raster(
        cls,
        raster: Path,
        config: TileConfig,
        out_dir: Path,
        mode: Mode,
        suffix: str = ".tif",
    ) -> t.Self:
        handler = TileHandler(config)
        with rasterio.open(raster, mode="r") as src:
            metadata = src.meta.copy()

            def handle_window(input: tuple[windows.Window, t.Any]):
                window, transform = input
                metadata["transform"] = transform
                metadata["width"], metadata["height"] = (
                    window.width,
                    window.height,
                )
                out_filepath = out_dir / f"{window.col_off}_{window.row_off}{suffix}"

                with rasterio.open(out_filepath, "w", **metadata) as dst:
                    dst.write(src.read(window=window))

            for i in range(handler.get_tiles_length(src)):
                handle_window(handler.get_tile(src, i))

        return cls.from_dir(out_dir, mode, suffix)

    def add_overlap(self, mode: Mode, tile_config: TileConfig) -> t.Self:
        handler = TileHandler(tile_config)
        merged = self.merge(self.parent_dir / "merged.tif")
        parent_dir = self.parent_dir.parent / f"overlap_{self.parent_dir.name}"
        with rasterio.open(merged) as src:
            metadata = src.meta.copy()

            for window, transform in handler.get_tiles(src):
                metadata["transform"] = transform
                metadata["width"], metadata["height"] = (
                    window.width,
                    window.height,
                )
                bounds = box(*windows.bounds(window, src.transform))
                out_filepath = parent_dir / get_tile_relative_path(self.tiles, bounds)
                os.makedirs(out_filepath.parent, exist_ok=True)
                with rasterio.open(out_filepath, "w", **metadata) as dst:
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
                newaff, width, height = rasterio.warp.calculate_default_transform(
                    ds.crs,
                    ds.crs,
                    ds.width,
                    ds.height,
                    *ds.bounds,
                    resolution=(tile_size.width, tile_size.height),
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
                        "transform": newaff,
                        "width": width,
                        "height": height,
                        "nodata": NODATA,
                        "crs": crs,
                    }
                )
                with rasterio.open(dest, mode="w", **meta) as dest_raster:
                    dest_raster.write(newarr)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(from_path, self.tiles["path"])
            # TODO: use submit instead and handle each error separately
            try:
                for _ in results:
                    ...
            except Exception as e:
                logger.error("Error %s occured." % e)

        return self.from_dir(out_dir, mode)

    def merge(
        self, out_file: Path, merge_kwargs: dict[str, t.Any] | None = None
    ) -> Path:
        paths = self.parent_dir.as_posix() + "/" + self.tiles["path"].str.lstrip("/")
        out_file.unlink(missing_ok=True)
        if merge_kwargs is None:
            merge_kwargs = {}
        merge_kwargs.setdefault("nodata", NODATA)
        merge_kwargs.setdefault("resampling", Resampling.bilinear)
        # merge_kwargs.setdefault('method', RasterTiles.custom_merge_mean)
        merge_kwargs.setdefault("target_aligned_pixels", True)
        rasterio.merge.merge(paths.values, dst_path=out_file, **merge_kwargs)
        return out_file

    @staticmethod
    def custom_merge_mean(merged_data, new_data, merged_mask, new_mask, **kwargs):
        """Returns the maximum value pixel."""
        mask = np.empty_like(merged_mask, dtype="bool")
        np.logical_or(merged_mask, new_mask, out=mask)
        np.logical_not(mask, out=mask)
        merged_data[mask] = np.mean(
            np.concatenate([merged_data[mask], new_data[mask]], axis=0)
        )
        np.logical_not(new_mask, out=mask)
        np.logical_and(merged_mask, mask, out=mask)
        np.copyto(merged_data, new_data, where=mask, casting="unsafe")


# @dataclass
# class LandslideImageSemanticSegmentation(LandslideImages): ...


def get_dem_cell_size(raster: Path) -> tuple[float, float]:
    with rasterio.open(raster, mode="r") as r:
        x, y = r.res
    return (x, y)
