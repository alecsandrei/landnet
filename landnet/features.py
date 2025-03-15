from __future__ import annotations

import collections.abc as c
import concurrent.futures
import functools
import os
import typing as t
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.crs
import rasterio.mask
import rasterio.merge
import rasterio.warp
from PySAGA_cmd import SAGA
from PySAGA_cmd.saga import Version
from rasterio import DatasetReader, windows
from rasterio.enums import Resampling
from rasterio.features import dataset_features
from shapely import Polygon, box
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToTensor

from landnet.config import (
    DEM_TILES,
    EPSG,
    INTERIM_DATA_DIR,
    NODATA,
    PROJ_ROOT,
    RASTER_CELL_SIZE,
    RAW_DATA_DIR,
    TEST_TILES,
    TRAIN_TILES,
)
from landnet.dataset import Feature, geojson_to_gdf, get_empty_geojson

if t.TYPE_CHECKING:
    from PySAGA_cmd.saga import Library, ToolOutput


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
        if x >= ncols:
            break
        # if ncols - x <= overlap:
        #     break
        if x != 0:
            x -= overlap
        width = tile_width + overlap * (2 if x != 0 else 1)
        if x + width > ncols:
            x = ncols - width
        for y in range(0, nrows + ystep, ystep):
            # print(x, y, nrows, ncols)
            if y >= nrows:
                break
            # if nrows - y <= overlap:
            #     break
            if y != 0:
                y -= overlap
            height = tile_height + overlap * (2 if y != 0 else 1)
            if y + height > nrows:
                y = nrows - height
            window = windows.Window(x, y, width, height)  # type: ignore
            transform = windows.transform(window, ds.transform)
            yield window, transform


@dataclass
class RasterTiles:
    tiles: gpd.GeoDataFrame
    parent_dir: Path

    @classmethod
    def from_dir(cls, tiles_parent: Path, suffix: str = '.tif') -> t.Self:
        geojson = get_empty_geojson()
        images = tiles_parent.rglob(f'*{suffix}')

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
                'path': path.relative_to(tiles_parent).as_posix()
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
        return cls.from_dir(out_dir, suffix)

    def add_overlap(
        self,
        overlap: int,
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
        return self.from_dir(parent_dir)

    def resample(
        self, out_dir: Path, resampling: Resampling = Resampling.bilinear
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

        return self.from_dir(out_dir)

    def merge(self, out_file: Path) -> Path:
        paths = (
            self.parent_dir.as_posix()
            + '/'
            + self.tiles['path'].str.lstrip('/')
        )
        # out_file.unlink(missing_ok=True)
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


def merge_rasters(datasets, **kwargs):
    """Wrapper for rasterio.merge.merge"""
    return rasterio.merge.merge(datasets, **kwargs)


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


def get_merged_dems(
    tiles: gpd.GeoDataFrame, fishnet: Fishnet, mode: Mode, workers: int = 4
) -> list[Path]:
    cell_size = get_dem_cell_size(PROJ_ROOT / tiles.iloc[0].path)
    dem_boundaries = fishnet.generate_grid(buffer=max(cell_size))

    def handle_bounds(bounds: tuple[int, Polygon]) -> Path:
        index, polygon = bounds
        subset = tiles[tiles.intersects(polygon)]
        return get_merged_dem(subset, mode, str(index), cell_size)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(handle_bounds, enumerate(dem_boundaries, start=1))
        )


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


def split_raster(
    saga: SAGA, raster: PathLike, tile_size: tuple[int, int], out_dir: PathLike
) -> ToolOutput:
    tiling = saga / 'grid_tools' / 'Tiling'
    return tiling.execute(
        grid=raster,
        tiles=out_dir,
        tiles_save=1,
        tiles_path=out_dir,
        nx=tile_size[0],
        ny=tile_size[1],
        verbose=True,
    )


class LandslideClass(Enum):
    NO_LANDSLIDE = auto()
    LANDSLIDE = auto()


@dataclass
class LandslideImageFolder(ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_tile_paths(self) -> TilePaths:
        tiles = {}
        for file in Path(self.root).rglob('*[0-9]*'):
            tile = int(''.join(filter(str.isdigit, file.name)))
            tiles[tile] = file
        return tiles

    def reclassify(
        self,
        tiles: c.Mapping[LandslideTile, LandslideDensity],
        threshold_percentage: float = 0.05,
    ) -> None:
        tile_paths = self.get_tile_paths()
        for tile, percentage in tiles.items():
            path = tile_paths[tile]
            if percentage >= threshold_percentage:
                landslide_class = str(LandslideClass.LANDSLIDE.value)
            else:
                landslide_class = str(LandslideClass.NO_LANDSLIDE.value)

            if path.parent.name != landslide_class:
                new_folder = path.parents[1] / landslide_class
                new_folder.mkdir(exist_ok=True)
                print(
                    '{} renamed to {}, {}, {}'.format(
                        path,
                        new_folder / path.name,
                        percentage,
                        landslide_class,
                    )
                )
                path.rename(new_folder / path.name)
        self.remove_empty_folders(Path(self.root))

    @staticmethod
    def remove_empty_folders(path: Path) -> None:
        for dir_ in path.iterdir():
            if dir_.is_dir():
                if not list(dir_.iterdir()):
                    dir_.rmdir()


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
        parts = list(dem.parts)
        parts[-4] = variable.value
        path = Path(*parts)
        os.makedirs(path.parent, exist_ok=True)
        return path

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
):
    tile = tiles.tiles[tiles.tiles['path'] == tile_path].iloc[0].path
    overlap_tile = (
        tiles_with_overlap.tiles[tiles_with_overlap.tiles['path'] == tile]
        .iloc[0]
        .path
    )
    tile_path = tiles.parent_dir / tile
    overlap_tile_path = tiles_with_overlap.parent_dir / overlap_tile
    terrain_analysis = list(
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


def handle_tiles(
    tiles: RasterTiles, tiles_with_overlap: RasterTiles, mode: Mode
):
    saga = SAGA('saga_cmd', version=Version(9, 8, 0))
    # terrain_analysis =
    #     TerrainAnalysis(
    #         dem,
    #         mode=mode,
    #         saga=saga,
    #         verbose=False,
    #         infer_obj_type=False,
    #         ignore_stderr=False,
    #     ).execute()

    # for tile in tiles.tiles['path']:
    #     handle_tile(tiles, tiles_with_overlap, saga, tile, mode)

    func = functools.partial(
        handle_tile,
        tiles=tiles,
        tiles_with_overlap=tiles_with_overlap,
        saga=saga,
        mode=mode,
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
        results = executor.map(func, tiles.tiles['path'])

    # for tool in terrain_analysis.execute():
    #     if tool[1].stderr is not None:
    #         print(
    #             'Failed for', path, 'tool name', tool[0], '\n', tool[1].stderr
    #         )
    # print('Processed', path)
    # return terrain_analysis


def get_dem_cell_size(raster: Path) -> tuple[float, float]:
    with rasterio.open(raster, mode='r') as r:
        x, y = r.res
    return (x, y)


def compute_train_tiles():
    tiles = t.cast(
        gpd.GeoDataFrame, gpd.read_file(RAW_DATA_DIR / 'dem_tiles.geojson')
    ).dropna(subset='mode')
    train_tiles = RasterTiles(
        tiles[tiles.loc[:, 'mode'] == 'train'], DEM_TILES
    ).resample(TRAIN_TILES / 'dem')
    train_tiles_with_overlap = train_tiles.add_overlap(1)
    train_tiles_with_overlap.tiles.to_file(
        INTERIM_DATA_DIR / 'overlap_tiles.shp'
    )
    handle_tiles(train_tiles, train_tiles_with_overlap, 'train')


def compute_test_tiles():
    tiles = t.cast(
        gpd.GeoDataFrame, gpd.read_file(RAW_DATA_DIR / 'dem_tiles.geojson')
    ).dropna(subset='mode')
    test_tiles = RasterTiles(
        tiles[tiles.loc[:, 'mode'] == 'test'], DEM_TILES
    ).resample(TEST_TILES / 'dem')
    test_tiles_with_overlap = test_tiles.add_overlap(1)
    test_tiles_with_overlap.tiles.to_file(
        INTERIM_DATA_DIR / 'overlap_tiles.shp'
    )
    handle_tiles(test_tiles, test_tiles_with_overlap, 'test')


def main(tile_size: tuple[int, int] = (100, 100)):
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        train = executor.submit(compute_train_tiles)
        test = executor.submit(compute_test_tiles)


if __name__ == '__main__':
    main()
