from __future__ import annotations

import collections.abc as c
import concurrent.futures
import os
import time
import typing as t
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import geopandas as gpd
import rasterio.crs
import rasterio.merge
from PySAGA_cmd import SAGA
from PySAGA_cmd.saga import Version
from shapely import Polygon, box
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToTensor

from landnet.config import (
    EPSG,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RAW_DATA_DIR,
)

if t.TYPE_CHECKING:
    import pandas as pd
    from PySAGA_cmd.saga import Library, ToolOutput


PathLike = os.PathLike | str
LandslideTile = int
LandslideDensity = float
TilePaths = dict[int, Path]
ClassFolder = Path
Mode = t.Literal['train', 'test']


@dataclass
class Fishnet:
    reference_geometry: Polygon
    rows: int
    cols: int

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
    suffix: str,
    scale: float | None = None,
    buffer_units: int | None = None,
):
    if scale is not None and buffer_units is not None:
        tiles['geometry'] = tiles.buffer(scale * buffer_units)
    paths = os.fspath(PROJ_ROOT) + '/' + tiles['path'].str.lstrip('/')
    assert mode in ('train', 'test')
    dst_path = INTERIM_DATA_DIR / f'{suffix}_{mode}_merged_dem.tif'
    # if dst_path.exists():
    #     return dst_path
    rasterio.merge.merge(
        paths.values,
        dst_path=dst_path,
        dst_kwds={'crs': rasterio.crs.CRS.from_epsg(EPSG)},
    )
    return dst_path


def get_merged_dems(
    tiles: gpd.GeoDataFrame, fishnet: Fishnet, mode: Mode, workers: int = 4
) -> list[Path]:
    cell_size = get_dem_cell_size(tiles.iloc[0])
    dem_boundaries = fishnet.generate_grid(buffer=max(cell_size))

    def handle_bounds(bounds: tuple[int, Polygon]) -> Path:
        index, polygon = bounds
        subset = tiles[tiles.intersects(polygon)]
        return get_merged_dem(subset, mode, str(index))

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
    SAGA_WETNESS_INDEX = 'twi'
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
        saga: SAGA,
        mode: Mode,
        verbose: bool,
        infer_obj_type: bool,
        ignore_stderr: bool,
    ):
        self.dem = Path(dem)
        self.saga = saga
        self.mode = mode
        self.verbose = verbose
        self.infer_obj_type = infer_obj_type
        self.ignore_stderr = ignore_stderr

        self.tools: list[c.Callable[..., ToolOutput]] = [
            self.index_of_convergence,
            self.terrain_surface_convexity,
            self.topographic_openness,
            self.slope_aspect_curvature,
            self.real_surface_area,
            # self.wind_exposition_index,
            # self.topographic_position_index,
            self.valley_depth,
            self.terrain_ruggedness_index,
            self.vector_ruggedness_measure,
            self.upslope_and_downslope_curvature,
            self.flow_accumulation_parallelizable,
            self.flow_path_length,
            self.slope_length,
            self.cell_balance,
            self.saga_wetness_index,
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

    def get_out_path(self, variable: GeomorphometricalVariable) -> Path:
        out_path = (
            PROCESSED_DATA_DIR
            / f'{self.mode}_tiles'
            / variable.value
            / self.dem.name
        )
        os.makedirs(out_path.parent, exist_ok=True)
        return out_path

    def index_of_convergence(self) -> ToolOutput:
        """Requires 1 or 2 units of buffer depending on the neighbours parameter."""
        tool = self.morphometry / 'Convergence Index'
        return tool.execute(
            elevation=self.dem,
            result=self.get_out_path(
                GeomorphometricalVariable.INDEX_OF_CONVERGENCE
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
        return tool.execute(
            dem=self.dem,
            convexity=self.get_out_path(
                GeomorphometricalVariable.TERRAIN_SURFACE_CONVEXITY
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
            shade=self.get_out_path(GeomorphometricalVariable.HILLSHADE),
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
                GeomorphometricalVariable.POSITIVE_TOPOGRAPHIC_OPENNESS
            ),
            neg=self.get_out_path(
                GeomorphometricalVariable.NEGATIVE_TOPOGRAPHIC_OPENNESS
            ),
            radius=100,
            directions=1,
            direction=315,
            ndirs=8,
            method=1,
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
        return tool.execute(
            elevation=self.dem,
            slope=self.get_out_path(GeomorphometricalVariable.SLOPE),
            c_gene=self.get_out_path(
                GeomorphometricalVariable.GENERAL_CURVATURE
            ),
            c_prof=self.get_out_path(
                GeomorphometricalVariable.PROFILE_CURVATURE
            ),
            c_plan=self.get_out_path(GeomorphometricalVariable.PLAN_CURVATURE),
            c_tang=self.get_out_path(
                GeomorphometricalVariable.TANGENTIAL_CURVATURE
            ),
            c_long=self.get_out_path(
                GeomorphometricalVariable.LONGITUDINAL_CURVATURE
            ),
            c_cros=self.get_out_path(
                GeomorphometricalVariable.CROSS_SECTIONAL_CURVATURE
            ),
            c_mini=self.get_out_path(
                GeomorphometricalVariable.MINIMAL_CURVATURE
            ),
            c_maxi=self.get_out_path(
                GeomorphometricalVariable.MAXIMAL_CURVATURE
            ),
            c_tota=self.get_out_path(GeomorphometricalVariable.TOTAL_CURVATURE),
            c_roto=self.get_out_path(
                GeomorphometricalVariable.FLOW_LINE_CURVATURE
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
            area=self.get_out_path(GeomorphometricalVariable.REAL_SURFACE_AREA),
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def wind_exposition_index(self) -> ToolOutput:
        tool = self.morphometry / 'Wind Exposition Index'
        return tool.execute(
            dem=self.dem,
            exposition=self.get_out_path(
                GeomorphometricalVariable.WIND_EXPOSITION_INDEX
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
                GeomorphometricalVariable.TOPOGRAPHIC_POSITION_INDEX
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
                GeomorphometricalVariable.VALLEY_DEPTH
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
        return tool.execute(
            dem=self.dem,
            tri=self.get_out_path(
                GeomorphometricalVariable.TERRAIN_RUGGEDNESS_INDEX
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
        return tool.execute(
            dem=self.dem,
            vrm=self.get_out_path(
                GeomorphometricalVariable.VECTOR_RUGGEDNESS_MEASURE
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
        return tool.execute(
            dem=self.dem,
            c_local=self.get_out_path(
                GeomorphometricalVariable.LOCAL_CURVATURE
            ),
            c_up=self.get_out_path(GeomorphometricalVariable.UPSLOPE_CURVATURE),
            c_up_local=self.get_out_path(
                GeomorphometricalVariable.LOCAL_UPSLOPE_CURVATURE
            ),
            c_down=self.get_out_path(
                GeomorphometricalVariable.DOWNSLOPE_CURVATURE
            ),
            c_down_local=self.get_out_path(
                GeomorphometricalVariable.LOCAL_DOWNSLOPE_CURVATURE
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
            flow=self.get_out_path(GeomorphometricalVariable.FLOW_ACCUMULATION),
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
                GeomorphometricalVariable.FLOW_PATH_LENGTH
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
            length=self.get_out_path(GeomorphometricalVariable.SLOPE_LENGTH),
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
            balance=self.get_out_path(GeomorphometricalVariable.CELL_BALANCE),
            method=1,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def saga_wetness_index(self) -> ToolOutput:
        tool = self.hydrology / 'SAGA Wetness Index'
        return tool.execute(
            dem=self.dem,
            # weight=None,
            # area=None,
            # slope=None,
            # area_mod=None,
            twi=self.get_out_path(GeomorphometricalVariable.SAGA_WETNESS_INDEX),
            suction=10,
            area_type=2,
            slope_type=1,
            slope_min=0,
            slope_off=0.1,
            slope_weight=1,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )


PROCESSED = 0


def handle_tile(attributes: dict[str, t.Any]):
    global PROCESSED
    saga = SAGA('saga_cmd', version=Version(9, 8, 0))
    path = PROJ_ROOT / attributes['path']
    assert path.exists(), 'DEM path does not exist.'
    terrain_analysis = TerrainAnalysis(
        path,
        saga,
        attributes['mode'],
        verbose=False,
        infer_obj_type=False,
        ignore_stderr=True,
    )

    for tool in terrain_analysis.execute():
        if tool[1].stderr is not None:
            print(
                'Failed for', path, 'tool name', tool[0], '\n', tool[1].stderr
            )
    PROCESSED += 1
    print('Processed', path, 'Total count:', PROCESSED)
    return terrain_analysis


def get_dem_cell_size(tile: pd.Series) -> tuple[float, float]:
    dem_path = PROJ_ROOT / tile.path
    with rasterio.open(dem_path, mode='r') as raster:
        x, y = raster.res
    return (x, y)


def main():
    tiles = t.cast(
        gpd.GeoDataFrame, gpd.read_file(RAW_DATA_DIR / 'dem_tiles.geojson')
    ).dropna(subset='mode')
    train_tiles = t.cast(
        gpd.GeoDataFrame, tiles[tiles.loc[:, 'mode'] == 'train']
    )
    test_tiles = t.cast(gpd.GeoDataFrame, tiles[tiles.loc[:, 'mode'] == 'test'])

    train_tiles_limit = box(
        *t.cast(Polygon, train_tiles.dissolve().geometry.iloc[0]).bounds
    )

    fishnet = Fishnet(train_tiles_limit, 4, 4)
    # gpd.GeoDataFrame(geometry=fishnet).to_file(RAW_DATA_DIR / 'fishnet.shp')
    start = time.perf_counter()
    merged_dems = get_merged_dems(train_tiles, fishnet, 'train')
    end = time.perf_counter()
    print('Execution duration:', end - start)
    breakpoint()

    # train_dem = get_merged_dem(
    #     train_tiles,
    #     scale=10,
    #     buffer_units=1,
    # )
    # train_dem = get_merged_dem(
    #     test_tiles,
    #     scale=10,
    #     buffer_units=1,
    # )
    # train_terrain_analysis = TerrainAnalysis(
    #     train_dem, saga, mode='train', verbose=True
    # )
    # test_terrain_analysis = TerrainAnalysis(
    #     train_dem, saga, mode='test', verbose=True
    # )
    with concurrent.futures.ProcessPoolExecutor(max_workers=100) as executor:
        results = executor.map(handle_tile, tiles.to_dict('index').values())

    breakpoint()

    # def handle_terrain_analysis(terrain_analysis: TerrainAnalysis):
    #     print(f'Executing tools for {terrain_analysis.mode=}')
    #     list(terrain_analysis.execute())

    # handle_terrain_analysis(train_terrain_analysis)
    # handle_terrain_analysis(test_terrain_analysis)


if __name__ == '__main__':
    main()
