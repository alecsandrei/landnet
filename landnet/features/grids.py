from __future__ import annotations

import collections.abc as c
import typing as t
from enum import Enum
from pathlib import Path

from PySAGA_cmd.saga import SAGA

from landnet.config import (
    GRIDS,
    INTERIM_DATA_DIR,
    TEST_TILES,
    TRAIN_TILES,
)
from landnet.logger import create_logger

if t.TYPE_CHECKING:
    from os import PathLike

    from PySAGA_cmd.saga import SAGA, Library, ToolOutput

    from landnet.features.tiles import RasterTiles

logger = create_logger(__name__)

Mode = t.Literal['train', 'test']


class GeomorphometricalVariable(Enum):
    INDEX_OF_CONVERGENCE = 'ioc'
    HILLSHADE = 'shade'
    TERRAIN_SURFACE_CONVEXITY = 'conv'
    POSITIVE_TOPOGRAPHIC_OPENNESS = 'poso'
    NEGATIVE_TOPOGRAPHIC_OPENNESS = 'nego'
    ASPECT = 'aspect'
    SLOPE = 'slope'
    NORTHNESS = 'northness'
    EASTNESS = 'eastness'
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
            self.analytical_hillshading,
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

    def get_out_path(self, variable: GeomorphometricalVariable) -> Path:
        return (GRIDS / self.mode / variable.value).with_suffix('.tif')

    def index_of_convergence(self) -> ToolOutput:
        """Requires 1 or 2 units of buffer depending on the neighbours parameter."""
        tool = self.morphometry / 'Convergence Index'
        dem = self.dem_edge if self.dem_edge is not None else self.dem
        return tool.execute(
            elevation=dem,
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
        dem = self.dem_edge if self.dem_edge is not None else self.dem
        return tool.execute(
            dem=dem,
            convexity=self.get_out_path(
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
                GeomorphometricalVariable.POSITIVE_TOPOGRAPHIC_OPENNESS,
            ),
            neg=self.get_out_path(
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
            aspect=self.get_out_path(GeomorphometricalVariable.ASPECT),
            northness=self.get_out_path(GeomorphometricalVariable.NORTHNESS),
            eastness=self.get_out_path(GeomorphometricalVariable.EASTNESS),
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
                GeomorphometricalVariable.CROSS_SECTIONAL_CURVATURE,
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
        dem = self.dem_edge if self.dem_edge is not None else self.dem
        return tool.execute(
            dem=dem,
            tri=self.get_out_path(
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

    def topographic_wetness_index(self) -> ToolOutput:
        tool = self.hydrology / 'twi'
        return tool.execute(
            dem=self.dem,
            twi=self.get_out_path(
                GeomorphometricalVariable.TOPOGRAPHIC_WETNESS_INDEX
            ),
            flow_method=0,
        )


# def handle_tile(
#     tile_path: str,
#     tiles: RasterTiles,
#     tiles_with_overlap: RasterTiles,
#     saga: SAGA,
#     mode: Mode,
# ) -> list[tuple[str, ToolOutput]]:
#     tile = t.cast(
#         str, tiles.tiles[tiles.tiles['path'] == tile_path].iloc[0].path
#     )
#     overlap_tile = (
#         tiles_with_overlap.tiles[tiles_with_overlap.tiles['path'] == tile]
#         .iloc[0]
#         .path
#     )
#     overlap_tile_path = tiles_with_overlap.parent_dir / overlap_tile
#     return list(
#         TerrainAnalysis(
#             tiles.parent_dir / tile,
#             dem_edge=overlap_tile_path,
#             mode=mode,
#             saga=saga,
#             verbose=False,
#             infer_obj_type=False,
#             ignore_stderr=False,
#         ).execute()
#     )


# def handle_tiles(
#     tiles: RasterTiles, tiles_with_overlap: RasterTiles, mode: Mode
# ):
#     saga = SAGA('saga_cmd', version=Version(9, 8, 0))
#     func = functools.partial(
#         handle_tile,
#         tiles=tiles,
#         tiles_with_overlap=tiles_with_overlap,
#         saga=saga,
#         mode=mode,
#     )
#     with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
#         results = executor.map(func, tiles.tiles['path'])
#     return results


def compute_grids_for_dem(
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


def compute_grids(tiles: RasterTiles, mode: Mode, saga: SAGA):
    tiles_dir = TRAIN_TILES if mode == 'train' else TEST_TILES
    logger.debug('Resampling %r for %sing' % (tiles, mode))
    resampled = tiles.resample(tiles_dir / 'dem' / '100x100', mode)
    logger.debug('Merging %r for %sing' % (resampled, mode))
    merged = resampled.merge(GRIDS / mode / 'dem.tif')
    compute_grids_for_dem(merged, saga, mode)
