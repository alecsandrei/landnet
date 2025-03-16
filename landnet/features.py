from __future__ import annotations

import collections.abc as c
import concurrent.futures
import functools
import json
import logging
import os
import typing as t
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.crs
import rasterio.mask
import rasterio.merge
import rasterio.warp
import shapely
from PIL import Image
from PySAGA_cmd import SAGA
from PySAGA_cmd.saga import Version
from rasterio import DatasetReader, windows
from rasterio.enums import Resampling
from rasterio.features import dataset_features
from shapely import MultiPolygon, Polygon, box
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToTensor

from landnet.config import (
    DEM_TILES,
    EPSG,
    GRIDS,
    INTERIM_DATA_DIR,
    NODATA,
    PROJ_ROOT,
    RASTER_CELL_SIZE,
    RAW_DATA_DIR,
    TEST_TILES,
    TRAIN_TILES,
)
from landnet.dataset import (
    Feature,
    geojson_to_gdf,
    get_empty_geojson,
    read_landslide_shapes,
)

if t.TYPE_CHECKING:
    from PySAGA_cmd.saga import Library, ToolOutput

logger = logging.getLogger(__name__)

PathLike = os.PathLike | str
LandslideTile = int
LandslideDensity = float
TilePaths = dict[int, Path]
ClassFolder = Path
Mode = t.Literal['train', 'test']
Rows = int
Columns = int


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
            intersection = landslide_shapes.intersection(feature).unary_union
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
                        shapely.from_geojson(json.dumps(feature['geometry'])),
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

        for image in list(images):
            geojson['features'].extend(get_features(image))
        return cls(geojson_to_gdf(geojson), tiles_parent)

    @classmethod
    def from_raster(
        cls,
        raster: Path,
        tile_size: tuple[Rows, Columns],
        overlap: int,
        out_dir: Path,
        mode: Mode,
        suffix: str = '.tif',
    ) -> t.Self:
        with rasterio.open(raster) as src:
            metadata = src.meta.copy()

            for window, transform in get_tiles(
                src,
                tile_size[0],
                tile_size[1],
                overlap,
            ):
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
        return cls.from_dir(out_dir, mode, suffix)

    def add_overlap(
        self,
        overlap: int,
        mode: Mode,
        tile_size: tuple[Rows, Columns] = (100, 100),
    ) -> t.Self:
        merged = self.merge(self.parent_dir / 'merged.tif')
        parent_dir = self.parent_dir.parent / f'overlap_{self.parent_dir.name}'
        with rasterio.open(merged) as src:
            metadata = src.meta.copy()

            for window, transform in get_tiles(
                src,
                tile_size[0],
                tile_size[1],
                overlap,
            ):
                metadata['transform'] = transform
                metadata['width'], metadata['height'] = (
                    window.width,
                    window.height,
                )
                bounds = box(*windows.bounds(window, src.transform))
                tile = self.tiles[self.tiles.within(bounds)].iloc[0]

                out_filepath = parent_dir / tile['path']
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
            dest = out_dir / tile_path
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
                resolution=RASTER_CELL_SIZE,
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
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


@dataclass
class Fishnet:
    reference_geometry: Polygon
    rows: Rows
    cols: Columns

    def generate_grid(self, buffer: float | None = None) -> list[Polygon]:
        """Generates a grid of polygons within the reference geometry."""
        minx, miny, maxx, maxy = self.reference_geometry.bounds
        cell_width = (maxx - minx) / self.cols
        cell_height = (maxy - miny) / self.rows

        grid = []
        for i in range(self.rows):
            for j in range(self.cols):
                cell_minx = minx + j * cell_width
                cell_maxx = cell_minx + cell_width
                cell_miny = miny + i * cell_height
                cell_maxy = cell_miny + cell_height
                cell = box(cell_minx, cell_miny, cell_maxx, cell_maxy)
                if self.reference_geometry.intersects(cell):
                    if buffer:
                        cell = cell.buffer(buffer, join_style='mitre')
                    grid.append(cell.intersection(self.reference_geometry))

        return grid


def get_merged_dem(
    tiles: gpd.GeoDataFrame,
    mode: Mode,
    cell_size: tuple[float, float],
    dst_path: Path,
):
    paths = os.fspath(PROJ_ROOT) + '/' + tiles['path'].str.lstrip('/')
    assert mode in ('train', 'test')
    dst_path.unlink(missing_ok=True)
    rasterio.merge.merge(
        paths.values,
        res=cell_size,
        resampling=Resampling.bilinear,
        dst_path=dst_path,
        dst_kwds={'crs': rasterio.crs.CRS.from_epsg(EPSG)},
        dtype=np.float32,
        # precision=2,
    )
    return dst_path


class GeomorphometricalVariable(Enum):
    INDEX_OF_CONVERGENCE = 'ioc'
    HILLSHADE = 'shade'
    TERRAIN_SURFACE_CONVEXITY = 'conv'
    POSITIVE_TOPOGRAPHIC_OPENNESS = 'poso'
    NEGATIVE_TOPOGRAPHIC_OPENNESS = 'nego'
    SLOPE = 'slope'
    GENERAL_CURVATURE = 'cgene'
    PROFILE_CURVATURE = 'cprof'
    PLAN_CURVATURE = 'cplan'
    TANGENTIAL_CURVATURE = 'ctang'
    LONGITUDINAL_CURVATURE = 'clong'
    CROSS_SECTIONAL_CURVATURE = 'ccros'
    MINIMAL_CURVATURE = 'cmini'
    MAXIMAL_CURVATURE = 'cmaxi'
    TOTAL_CURVATURE = 'ctota'
    FLOW_LINE_CURVATURE = 'croto'
    DIGITAL_ELEVATION_MODEL = 'dem'
    REAL_SURFACE_AREA = 'area'
    TOPOGRAPHIC_POSITION_INDEX = 'tpi'
    VALLEY_DEPTH = 'vld'
    TERRAIN_RUGGEDNESS_INDEX = 'tri'
    VECTOR_RUGGEDNESS_MEASURE = 'vrm'
    LOCAL_CURVATURE = 'clo'
    UPSLOPE_CURVATURE = 'cup'
    LOCAL_UPSLOPE_CURVATURE = 'clu'
    DOWNSLOPE_CURVATURE = 'cdo'
    LOCAL_DOWNSLOPE_CURVATURE = 'cdl'
    FLOW_ACCUMULATION = 'flow'
    FLOW_PATH_LENGTH = 'fpl'
    SLOPE_LENGTH = 'spl'
    CELL_BALANCE = 'cbl'
    TOPOGRAPHIC_WETNESS_INDEX = 'twi'
    WIND_EXPOSITION_INDEX = 'wind'


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
        return Image.open(f)


class LandslideClass(Enum):
    NO_LANDSLIDE = auto()
    LANDSLIDE = auto()


class LandslideImageFolder(VisionDataset):
    def __init__(
        self,
        raster_tiles: RasterTiles,
        loader: c.Callable[[str], t.Any] = get_default_loader,
        landslide_density_threshold: float = 0.05,
        transforms: c.Callable | None = None,
        transform: c.Callable | None = None,
        target_transform: c.Callable | None = None,
    ) -> None:
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


class TerrainAnalysis:
    def __init__(
        self,
        dem: PathLike,
        mode: Mode,
        saga: SAGA,
        verbose: bool,
        infer_obj_type: bool,
        ignore_stderr: bool,
        dem_edge: PathLike | None = None,
    ):
        self.dem = Path(dem)
        self.dem_edge = Path(dem_edge) if dem_edge is not None else None
        assert self.dem.is_relative_to(INTERIM_DATA_DIR)
        if self.dem_edge:
            assert self.dem_edge.is_relative_to(INTERIM_DATA_DIR)
        self.mode = mode
        self.saga = saga
        self.verbose = verbose
        self.infer_obj_type = infer_obj_type
        self.ignore_stderr = ignore_stderr

        self.tools: list[c.Callable[..., ToolOutput]] = [
            self.index_of_convergence,
            self.terrain_surface_convexity,
            self.topographic_openness,
            self.slope_aspect_curvature,
            self.real_surface_area,
            self.wind_exposition_index,
            self.topographic_position_index,
            self.valley_depth,
            self.terrain_ruggedness_index,
            self.vector_ruggedness_measure,
            self.upslope_and_downslope_curvature,
            self.flow_accumulation_parallelizable,
            self.flow_path_length,
            self.slope_length,
            self.cell_balance,
            self.topographic_wetness_index,
        ]

    def execute(self) -> c.Generator[tuple[str, ToolOutput]]:
        for tool in self.tools:
            yield (tool.__name__, tool())

    @property
    def morphometry(self) -> Library:
        return self.saga / 'ta_morphometry'

    @property
    def lighting(self) -> Library:
        return self.saga / 'ta_lighting'

    @property
    def channels(self) -> Library:
        return self.saga / 'ta_channels'

    @property
    def hydrology(self) -> Library:
        return self.saga / 'ta_hydrology'

    def get_out_path(
        self, dem: Path, variable: GeomorphometricalVariable
    ) -> Path:
        # parts = list(dem.parts)
        # parts[-4] = variable.value
        # path = Path(*parts)
        # os.makedirs(path.parent, exist_ok=True)
        # return path
        return (GRIDS / self.mode / variable.value).with_suffix('.tif')

    def index_of_convergence(self) -> ToolOutput:
        """Requires 1 or 2 units of buffer depending on the neighbours parameter."""
        tool = self.morphometry / 'Convergence Index'
        dem = self.dem_edge if self.dem_edge is not None else self.dem
        return tool.execute(
            elevation=dem,
            result=self.get_out_path(
                dem, GeomorphometricalVariable.INDEX_OF_CONVERGENCE
            ),
            method=0,
            neighbours=0,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def terrain_surface_convexity(self) -> ToolOutput:
        """Requires 1 unit of buffer."""
        tool = self.morphometry / 'Terrain Surface Convexity'
        dem = self.dem_edge if self.dem_edge is not None else self.dem
        return tool.execute(
            dem=dem,
            convexity=self.get_out_path(
                dem,
                GeomorphometricalVariable.TERRAIN_SURFACE_CONVEXITY,
            ),
            kernel=0,
            type=0,
            epsilon=0,
            scale=10,
            method=1,
            dw_weighting=3,
            dw_idw_power=2,
            dw_bandwidth=0.7,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def analytical_hillshading(self) -> ToolOutput:
        tool = self.lighting / 'Analytical Hillshading'
        return tool.execute(
            elevation=self.dem,
            method='5',
            shade=self.get_out_path(
                self.dem, GeomorphometricalVariable.HILLSHADE
            ),
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def topographic_openness(self) -> ToolOutput:
        """Uses radius / resolution units of buffer."""
        tool = self.lighting / 'Topographic Openness'
        return tool.execute(
            dem=self.dem,
            pos=self.get_out_path(
                self.dem,
                GeomorphometricalVariable.POSITIVE_TOPOGRAPHIC_OPENNESS,
            ),
            neg=self.get_out_path(
                self.dem,
                GeomorphometricalVariable.NEGATIVE_TOPOGRAPHIC_OPENNESS,
            ),
            radius=100,
            directions=1,
            direction=315,
            ndirs=8,
            method=0,
            dlevel=3.0,
            unit=0,
            nadir=1,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def slope_aspect_curvature(self) -> ToolOutput:
        """Requires 1 unit of buffer.

        TODO: Add Northness and Eastness"""
        tool = self.morphometry / 'Slope, Aspect, Curvature'
        dem = self.dem_edge if self.dem_edge is not None else self.dem
        return tool.execute(
            elevation=dem,
            slope=self.get_out_path(dem, GeomorphometricalVariable.SLOPE),
            c_gene=self.get_out_path(
                dem, GeomorphometricalVariable.GENERAL_CURVATURE
            ),
            c_prof=self.get_out_path(
                dem, GeomorphometricalVariable.PROFILE_CURVATURE
            ),
            c_plan=self.get_out_path(
                dem, GeomorphometricalVariable.PLAN_CURVATURE
            ),
            c_tang=self.get_out_path(
                dem, GeomorphometricalVariable.TANGENTIAL_CURVATURE
            ),
            c_long=self.get_out_path(
                dem, GeomorphometricalVariable.LONGITUDINAL_CURVATURE
            ),
            c_cros=self.get_out_path(
                dem,
                GeomorphometricalVariable.CROSS_SECTIONAL_CURVATURE,
            ),
            c_mini=self.get_out_path(
                dem, GeomorphometricalVariable.MINIMAL_CURVATURE
            ),
            c_maxi=self.get_out_path(
                dem, GeomorphometricalVariable.MAXIMAL_CURVATURE
            ),
            c_tota=self.get_out_path(
                dem, GeomorphometricalVariable.TOTAL_CURVATURE
            ),
            c_roto=self.get_out_path(
                dem, GeomorphometricalVariable.FLOW_LINE_CURVATURE
            ),
            method=6,
            unit_slope=0,
            unit_aspect=0,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def real_surface_area(self) -> ToolOutput:
        tool = self.morphometry / 'Real Surface Area'
        return tool.execute(
            dem=self.dem,
            area=self.get_out_path(
                self.dem, GeomorphometricalVariable.REAL_SURFACE_AREA
            ),
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def wind_exposition_index(self) -> ToolOutput:
        tool = self.morphometry / 'Wind Exposition Index'
        return tool.execute(
            dem=self.dem,
            exposition=self.get_out_path(
                self.dem, GeomorphometricalVariable.WIND_EXPOSITION_INDEX
            ),
            maxdist=0.1,
            step=90,
            oldver=0,
            accel=1.5,
            pyramids=0,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def topographic_position_index(self) -> ToolOutput:
        """Uses radius_max / resolution units of buffer."""
        tool = self.morphometry / 'Topographic Position Index (TPI)'
        return tool.execute(
            dem=self.dem,
            tpi=self.get_out_path(
                self.dem, GeomorphometricalVariable.TOPOGRAPHIC_POSITION_INDEX
            ),
            standard=0,
            dw_weighting=0,
            dw_idw_power=2,
            dw_bandwidth=75,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def valley_depth(self) -> ToolOutput:
        tool = self.channels / 'Valley Depth'
        return tool.execute(
            elevation=self.dem,
            valley_depth=self.get_out_path(
                self.dem, GeomorphometricalVariable.VALLEY_DEPTH
            ),
            threshold=1,
            maxiter=0,
            nounderground=1,
            order=4,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def terrain_ruggedness_index(self) -> ToolOutput:
        """Uses radius units of buffer."""
        tool = self.morphometry / 'Terrain Ruggedness Index (TRI)'
        dem = self.dem_edge if self.dem_edge is not None else self.dem
        return tool.execute(
            dem=dem,
            tri=self.get_out_path(
                dem,
                GeomorphometricalVariable.TERRAIN_RUGGEDNESS_INDEX,
            ),
            mode=1,
            radius=1,
            dw_weighting=0,
            dw_idw_power=2,
            dw_bandwidth=75,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def vector_ruggedness_measure(self) -> ToolOutput:
        """Uses radius units of buffer."""
        tool = self.morphometry / 'Vector Ruggedness Measure (VRM)'
        dem = self.dem_edge if self.dem_edge is not None else self.dem
        return tool.execute(
            dem=dem,
            vrm=self.get_out_path(
                dem,
                GeomorphometricalVariable.VECTOR_RUGGEDNESS_MEASURE,
            ),
            mode=1,
            radius=1,
            dw_weighting=0,
            dw_idw_power=2,
            dw_bandwidth=75,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def upslope_and_downslope_curvature(self) -> ToolOutput:
        """Uses one unit of buffer."""
        tool = self.morphometry / 'Upslope and Downslope Curvature'
        dem = self.dem_edge if self.dem_edge is not None else self.dem
        return tool.execute(
            dem=dem,
            c_local=self.get_out_path(
                dem, GeomorphometricalVariable.LOCAL_CURVATURE
            ),
            c_up=self.get_out_path(
                dem, GeomorphometricalVariable.UPSLOPE_CURVATURE
            ),
            c_up_local=self.get_out_path(
                dem, GeomorphometricalVariable.LOCAL_UPSLOPE_CURVATURE
            ),
            c_down=self.get_out_path(
                dem, GeomorphometricalVariable.DOWNSLOPE_CURVATURE
            ),
            c_down_local=self.get_out_path(
                dem,
                GeomorphometricalVariable.LOCAL_DOWNSLOPE_CURVATURE,
            ),
            weighting=0.5,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def flow_accumulation_parallelizable(self) -> ToolOutput:
        tool = self.hydrology / 'Flow Accumulation (Parallelizable)'
        return tool.execute(
            dem=self.dem,
            flow=self.get_out_path(
                self.dem, GeomorphometricalVariable.FLOW_ACCUMULATION
            ),
            update=0,
            method=2,
            convergence=1.1,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def flow_path_length(self) -> ToolOutput:
        tool = self.hydrology / 'Flow Path Length'
        return tool.execute(
            elevation=self.dem,
            # seed=None,
            length=self.get_out_path(
                self.dem, GeomorphometricalVariable.FLOW_PATH_LENGTH
            ),
            seeds_only=0,
            method=1,
            convergence=1.1,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def slope_length(self) -> ToolOutput:
        tool = self.hydrology / 'Slope Length'
        return tool.execute(
            dem=self.dem,
            length=self.get_out_path(
                self.dem, GeomorphometricalVariable.SLOPE_LENGTH
            ),
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def cell_balance(self) -> ToolOutput:
        tool = self.hydrology / 'Cell Balance'
        return tool.execute(
            dem=self.dem,
            # weights=None,
            weights_default=1,
            balance=self.get_out_path(
                self.dem, GeomorphometricalVariable.CELL_BALANCE
            ),
            method=1,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def topographic_wetness_index(self) -> ToolOutput:
        tool = self.hydrology / 'twi'
        return tool.execute(
            dem=self.dem,
            twi=self.get_out_path(
                self.dem, GeomorphometricalVariable.TOPOGRAPHIC_WETNESS_INDEX
            ),
            flow_method=0,
        )


def handle_tile(
    tile_path: str,
    tiles: RasterTiles,
    tiles_with_overlap: RasterTiles,
    saga: SAGA,
    mode: Mode,
) -> list[tuple[str, ToolOutput]]:
    tile = tiles.tiles[tiles.tiles['path'] == tile_path].iloc[0].path
    overlap_tile = (
        tiles_with_overlap.tiles[tiles_with_overlap.tiles['path'] == tile]
        .iloc[0]
        .path
    )
    tile_path = tiles.parent_dir / tile
    overlap_tile_path = tiles_with_overlap.parent_dir / overlap_tile
    return list(
        TerrainAnalysis(
            tile_path,
            dem_edge=overlap_tile_path,
            mode=mode,
            saga=saga,
            verbose=False,
            infer_obj_type=False,
            ignore_stderr=False,
        ).execute()
    )


def handle_dem(
    dem: Path,
    saga: SAGA,
    mode: Mode,
):
    for tool_name, _ in TerrainAnalysis(
        dem,
        mode=mode,
        saga=saga,
        verbose=False,
        infer_obj_type=False,
        ignore_stderr=False,
    ).execute():
        logger.info('%s finished executing for %r' % (tool_name, mode))


def handle_tiles(
    tiles: RasterTiles, tiles_with_overlap: RasterTiles, mode: Mode
):
    saga = SAGA('saga_cmd', version=Version(9, 8, 0))
    func = functools.partial(
        handle_tile,
        tiles=tiles,
        tiles_with_overlap=tiles_with_overlap,
        saga=saga,
        mode=mode,
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
        results = executor.map(func, tiles.tiles['path'])
    return results


def compute_grids(mode: Mode, saga: SAGA):
    tiles_dir = TRAIN_TILES if mode == 'train' else TEST_TILES
    tiles = t.cast(
        gpd.GeoDataFrame, gpd.read_file(RAW_DATA_DIR / 'dem_tiles.geojson')
    ).dropna(subset='mode')
    os.makedirs((GRIDS / mode), exist_ok=True)
    raster_tiles = RasterTiles(tiles[tiles.loc[:, 'mode'] == mode], DEM_TILES)
    logger.debug('Resampling %r for %r' % (raster_tiles, mode))
    resampled = raster_tiles.resample(tiles_dir / 'dem')
    logger.debug('Merging %r for %r' % (raster_tiles, mode))
    merged = resampled.merge(GRIDS / mode / 'dem.tif')
    handle_dem(merged, saga, mode)


def main(tile_size: tuple[int, int] = (100, 100)):
    saga = SAGA('saga_cmd', Version(9, 8, 0))
    compute_grids('train', saga)
    compute_grids('test', saga)


if __name__ == '__main__':
    main()
