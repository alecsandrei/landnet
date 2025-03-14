from __future__ import annotations

import collections.abc as c
import os
import typing as t
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import rasterio.crs
import rasterio.merge
import shapely
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToTensor

from landnet.config import DEM_TILES, EPSG

if t.TYPE_CHECKING:
    import geopandas as gpd
    from PySAGA_cmd.saga import SAGA, Library, ToolOutput


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
    if dst_path.exists():
        return dst_path
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
    def __init__(self, dem: PathLike, saga: SAGA, out_dir: PathLike):
        self.dem = dem
        self.saga = saga
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)
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

    def index_of_convergence(self) -> ToolOutput:
        """Requires 1 or 2 units of buffer depending on the neighbours parameter."""
        tool = self.morphometry / 'Convergence Index'
        out_path = (
            self.out_dir
            / f'{GeomorphometricalVariable.INDEX_OF_CONVERGENCE.value}.tif'
        )
        return tool.execute(
            elevation=self.dem,
            result=out_path,
            method=0,
            neighbours=0,
            verbose=True,
        )

    def terrain_surface_convexity(self) -> ToolOutput:
        """Requires 1 unit of buffer."""
        tool = self.morphometry / 'Terrain Surface Convexity'
        out_path = (
            self.out_dir
            / f'{GeomorphometricalVariable.TERRAIN_SURFACE_CONVEXITY.value}.tif'
        )
        return tool.execute(
            dem=self.dem,
            convexity=out_path,
            kernel=0,
            type=0,
            epsilon=0,
            scale=10,
            method=1,
            dw_weighting=3,
            dw_idw_power=2,
            dw_bandwidth=0.7,
            verbose=True,
        )

    def analytical_hillshading(self) -> ToolOutput:
        tool = self.lighting / 'Analytical Hillshading'
        return tool.execute(
            elevation=self.dem,
            method='5',
            shade=self.out_dir
            / f'{GeomorphometricalVariable.HILLSHADE.value}.tif',
        )

    def topographic_openness(self) -> ToolOutput:
        """Uses radius / resolution units of buffer."""
        tool = self.lighting / 'Topographic Openness'
        return tool.execute(
            dem=self.dem,
            pos=self.out_dir
            / f'{GeomorphometricalVariable.POSITIVE_TOPOGRAPHIC_OPENNESS.value}.tif',
            neg=self.out_dir
            / f'{GeomorphometricalVariable.NEGATIVE_TOPOGRAPHIC_OPENNESS.value}.tif',
            radius=100,
            directions=1,
            direction=315,
            ndirs=8,
            method=1,
            dlevel=3.0,
            unit=0,
            nadir=1,
            verbose=True,
        )

    def slope_aspect_curvature(self) -> ToolOutput:
        """Requires 1 unit of buffer.

        TODO: Add Northness and Eastness"""
        tool = self.morphometry / 'Slope, Aspect, Curvature'
        return tool.execute(
            elevation=self.dem,
            slope=self.out_dir / f'{GeomorphometricalVariable.SLOPE.value}.tif',
            c_gene=self.out_dir
            / f'{GeomorphometricalVariable.GENERAL_CURVATURE.value}.tif',
            c_prof=self.out_dir
            / f'{GeomorphometricalVariable.PROFILE_CURVATURE.value}.tif',
            c_plan=self.out_dir
            / f'{GeomorphometricalVariable.PLAN_CURVATURE.value}.tif',
            c_tang=self.out_dir
            / f'{GeomorphometricalVariable.TANGENTIAL_CURVATURE.value}.tif',
            c_long=self.out_dir
            / f'{GeomorphometricalVariable.LONGITUDINAL_CURVATURE.value}.tif',
            c_cros=self.out_dir
            / f'{GeomorphometricalVariable.CROSS_SECTIONAL_CURVATURE.value}.tif',
            c_mini=self.out_dir
            / f'{GeomorphometricalVariable.MINIMAL_CURVATURE.value}.tif',
            c_maxi=self.out_dir
            / f'{GeomorphometricalVariable.MAXIMAL_CURVATURE.value}.tif',
            c_tota=self.out_dir
            / f'{GeomorphometricalVariable.TOTAL_CURVATURE.value}.tif',
            c_roto=self.out_dir
            / f'{GeomorphometricalVariable.FLOW_LINE_CURVATURE.value}.tif',
            method=6,
            unit_slope=0,
            unit_aspect=0,
            verbose=True,
        )

    def real_surface_area(self) -> ToolOutput:
        tool = self.morphometry / 'Real Surface Area'
        return tool.execute(
            dem=self.dem,
            area=self.out_dir
            / f'{GeomorphometricalVariable.REAL_SURFACE_AREA.value}.tif',
            verbose=True,
        )

    def wind_exposition_index(self) -> ToolOutput:
        tool = self.morphometry / 'Wind Exposition Index'
        return tool.execute(
            dem=self.dem,
            exposition=self.out_dir
            / f'{GeomorphometricalVariable.WIND_EXPOSITION_INDEX.value}.tif',
            maxdist=300,
            step=15,
            oldver=0,
            accel=1.5,
            pyramids=0,
            verbose=True,
        )

    def topographic_position_index(self) -> ToolOutput:
        """Uses radius_max / resolution units of buffer."""
        tool = self.morphometry / 'Topographic Position Index (TPI)'
        return tool.execute(
            dem=self.dem,
            tpi=self.out_dir
            / f'{GeomorphometricalVariable.TOPOGRAPHIC_POSITION_INDEX.value}.tif',
            standard=0,
            radius_min=0,
            radius_max=100,
            dw_weighting=0,
            dw_idw_power=2,
            dw_bandwidth=75,
            verbose=True,
        )

    def valley_depth(self) -> ToolOutput:
        tool = self.channels / 'Valley Depth'
        return tool.execute(
            elevation=self.dem,
            valley_depth=self.out_dir
            / f'{GeomorphometricalVariable.VALLEY_DEPTH.value}.tif',
            threshold=1,
            maxiter=0,
            nounderground=1,
            order=4,
            verbose=True,
        )

    def terrain_ruggedness_index(self) -> ToolOutput:
        """Uses radius units of buffer."""
        tool = self.morphometry / 'Terrain Ruggedness Index (TRI)'
        return tool.execute(
            dem=self.dem,
            tri=self.out_dir
            / f'{GeomorphometricalVariable.TERRAIN_RUGGEDNESS_INDEX.value}.tif',
            mode=1,
            radius=1,
            dw_weighting=0,
            dw_idw_power=2,
            dw_bandwidth=75,
            verbose=True,
        )

    def vector_ruggedness_measure(self) -> ToolOutput:
        """Uses radius units of buffer."""
        tool = self.morphometry / 'Vector Ruggedness Measure (VRM)'
        return tool.execute(
            dem=self.dem,
            vrm=self.out_dir
            / f'{GeomorphometricalVariable.VECTOR_RUGGEDNESS_MEASURE.value}.tif',
            mode=1,
            radius=1,
            dw_weighting=0,
            dw_idw_power=2,
            dw_bandwidth=75,
            verbose=True,
        )

    def upslope_and_downslope_curvature(self) -> ToolOutput:
        """Uses one unit of buffer."""
        tool = self.morphometry / 'Upslope and Downslope Curvature'
        return tool.execute(
            dem=self.dem,
            c_local=self.out_dir
            / f'{GeomorphometricalVariable.LOCAL_CURVATURE.value}.tif',
            c_up=self.out_dir
            / f'{GeomorphometricalVariable.UPSLOPE_CURVATURE.value}.tif',
            c_up_local=self.out_dir
            / f'{GeomorphometricalVariable.LOCAL_UPSLOPE_CURVATURE.value}.tif',
            c_down=self.out_dir
            / f'{GeomorphometricalVariable.DOWNSLOPE_CURVATURE.value}.tif',
            c_down_local=self.out_dir
            / f'{GeomorphometricalVariable.LOCAL_DOWNSLOPE_CURVATURE.value}.tif',
            weighting=0.5,
            verbose=True,
        )

    def flow_accumulation_parallelizable(self) -> ToolOutput:
        tool = self.hydrology / 'Flow Accumulation (Parallelizable)'
        return tool.execute(
            dem=self.dem,
            flow=self.out_dir
            / f'{GeomorphometricalVariable.FLOW_ACCUMULATION.value}.tif',
            update=0,
            method=2,
            convergence=1.1,
            verbose=True,
        )

    def flow_path_length(self) -> ToolOutput:
        tool = self.hydrology / 'Flow Path Length'
        return tool.execute(
            elevation=self.dem,
            # seed=None,
            length=self.out_dir
            / f'{GeomorphometricalVariable.FLOW_PATH_LENGTH.value}.tif',
            seeds_only=0,
            method=1,
            convergence=1.1,
            verbose=True,
        )

    def slope_length(self) -> ToolOutput:
        tool = self.hydrology / 'Slope Length'
        return tool.execute(
            dem=self.dem,
            length=self.out_dir
            / f'{GeomorphometricalVariable.SLOPE_LENGTH.value}.tif',
            verbose=True,
        )

    def cell_balance(self) -> ToolOutput:
        tool = self.hydrology / 'Cell Balance'
        return tool.execute(
            dem=self.dem,
            # weights=None,
            weights_default=1,
            balance=self.out_dir
            / f'{GeomorphometricalVariable.CELL_BALANCE.value}.tif',
            method=1,
            verbose=True,
        )

    def saga_wetness_index(self) -> ToolOutput:
        tool = self.hydrology / 'SAGA Wetness Index'
        return tool.execute(
            dem=self.dem,
            # weight=None,
            # area=None,
            # slope=None,
            # area_mod=None,
            twi=self.out_dir
            / f'{GeomorphometricalVariable.SAGA_WETNESS_INDEX.value}.tif',
            suction=10,
            area_type=2,
            slope_type=1,
            slope_min=0,
            slope_off=0.1,
            slope_weight=1,
            verbose=True,
        )
