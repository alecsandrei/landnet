from __future__ import annotations

import collections.abc as c
import typing as t
from pathlib import Path

from PySAGA_cmd.saga import SAGA

from landnet.config import (
    GRIDS,
    INFERENCE_TILES,
    INTERIM_DATA_DIR,
    TEST_TILES,
    TRAIN_TILES,
)
from landnet.enums import GeomorphometricalVariable, Mode
from landnet.logger import create_logger

if t.TYPE_CHECKING:
    from os import PathLike

    from PySAGA_cmd.saga import SAGA, Library, ToolOutput

    from landnet.features.tiles import RasterTiles

logger = create_logger(__name__)


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
        variables: c.Sequence[GeomorphometricalVariable] | None = None,
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
        self.variables = variables

        self.tools: list[c.Callable[..., ToolOutput | None]] = [
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

    def execute(self) -> c.Generator[tuple[str, ToolOutput | None]]:
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

    def get_out_path(self, variable: GeomorphometricalVariable) -> Path | None:
        if self.should_compute(variable):
            return (self.dem.parent / variable.value).with_suffix('.tif')
        return None

    def should_compute(self, variable: GeomorphometricalVariable) -> bool:
        return self.variables is None or variable in self.variables

    def index_of_convergence(self) -> ToolOutput | None:
        """Requires 1 or 2 units of buffer depending on the neighbours parameter."""
        if not self.should_compute(
            GeomorphometricalVariable.INDEX_OF_CONVERGENCE
        ):
            return None
        tool = self.morphometry / 'Convergence Index'
        dem = self.dem_edge if self.dem_edge is not None else self.dem
        return tool.execute(
            dem=dem,
            method=0,
            neighbours=0,
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
            result=self.get_out_path(
                GeomorphometricalVariable.INDEX_OF_CONVERGENCE
            ),
        )

    def terrain_surface_convexity(self) -> ToolOutput | None:
        """Requires 1 unit of buffer."""
        if not self.should_compute(
            GeomorphometricalVariable.TERRAIN_SURFACE_CONVEXITY
        ):
            return None
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

    def analytical_hillshading(self) -> ToolOutput | None:
        if not self.should_compute(GeomorphometricalVariable.HILLSHADE):
            return None
        tool = self.lighting / 'Analytical Hillshading'
        return tool.execute(
            elevation=self.dem,
            method='5',
            shade=self.get_out_path(GeomorphometricalVariable.HILLSHADE),
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def topographic_openness(self) -> ToolOutput | None:
        """Uses radius / resolution units of buffer."""
        if not any(
            self.should_compute(variable)
            for variable in (
                GeomorphometricalVariable.POSITIVE_TOPOGRAPHIC_OPENNESS,
                GeomorphometricalVariable.NEGATIVE_TOPOGRAPHIC_OPENNESS,
            )
        ):
            return None
        tool = self.lighting / 'Topographic Openness'
        kwargs = {
            'dem': self.dem,
            'radius': 100,
            'directions': 1,
            'direction': 315,
            'ndirs': 8,
            'method': 0,
            'dlevel': 3.0,
            'unit': 0,
            'nadir': 1,
            'verbose': self.verbose,
            'infer_obj_type': self.infer_obj_type,
            'ignore_stderr': self.ignore_stderr,
            'pos': self.get_out_path(
                GeomorphometricalVariable.POSITIVE_TOPOGRAPHIC_OPENNESS
            ),
            'neg': self.get_out_path(
                GeomorphometricalVariable.NEGATIVE_TOPOGRAPHIC_OPENNESS
            ),
        }
        return tool.execute(**kwargs)

    def slope_aspect_curvature(self) -> ToolOutput | None:
        """Requires 1 unit of buffer."""
        if not any(
            self.should_compute(variable)
            for variable in (
                GeomorphometricalVariable.ASPECT,
                GeomorphometricalVariable.NORTHNESS,
                GeomorphometricalVariable.EASTNESS,
                GeomorphometricalVariable.SLOPE,
                GeomorphometricalVariable.GENERAL_CURVATURE,
                GeomorphometricalVariable.PROFILE_CURVATURE,
                GeomorphometricalVariable.PLAN_CURVATURE,
                GeomorphometricalVariable.TANGENTIAL_CURVATURE,
                GeomorphometricalVariable.LONGITUDINAL_CURVATURE,
                GeomorphometricalVariable.CROSS_SECTIONAL_CURVATURE,
                GeomorphometricalVariable.MINIMAL_CURVATURE,
                GeomorphometricalVariable.MAXIMAL_CURVATURE,
                GeomorphometricalVariable.TOTAL_CURVATURE,
                GeomorphometricalVariable.FLOW_LINE_CURVATURE,
            )
        ):
            return None

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

    def real_surface_area(self) -> ToolOutput | None:
        tool = self.morphometry / 'Real Surface Area'
        if not self.should_compute(GeomorphometricalVariable.REAL_SURFACE_AREA):
            return None
        return tool.execute(
            dem=self.dem,
            area=self.get_out_path(GeomorphometricalVariable.REAL_SURFACE_AREA),
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def wind_exposition_index(self) -> ToolOutput | None:
        if not self.should_compute(
            GeomorphometricalVariable.WIND_EXPOSITION_INDEX
        ):
            return None
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

    def topographic_position_index(self) -> ToolOutput | None:
        """Uses radius_max / resolution units of buffer."""
        if not self.should_compute(
            GeomorphometricalVariable.TOPOGRAPHIC_POSITION_INDEX
        ):
            return None
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

    def valley_depth(self) -> ToolOutput | None:
        if not self.should_compute(GeomorphometricalVariable.VALLEY_DEPTH):
            return None
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

    def terrain_ruggedness_index(self) -> ToolOutput | None:
        """Uses radius units of buffer."""
        if not self.should_compute(
            GeomorphometricalVariable.TERRAIN_RUGGEDNESS_INDEX
        ):
            return None
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

    def vector_ruggedness_measure(self) -> ToolOutput | None:
        """Uses radius units of buffer."""
        if not self.should_compute(
            GeomorphometricalVariable.VECTOR_RUGGEDNESS_MEASURE
        ):
            return None
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

    def upslope_and_downslope_curvature(self) -> ToolOutput | None:
        """Uses one unit of buffer."""
        if not any(
            self.should_compute(variable)
            for variable in (
                GeomorphometricalVariable.LOCAL_CURVATURE,
                GeomorphometricalVariable.UPSLOPE_CURVATURE,
                GeomorphometricalVariable.LOCAL_UPSLOPE_CURVATURE,
                GeomorphometricalVariable.DOWNSLOPE_CURVATURE,
                GeomorphometricalVariable.LOCAL_DOWNSLOPE_CURVATURE,
            )
        ):
            return None
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
        if not self.should_compute(GeomorphometricalVariable.FLOW_ACCUMULATION):
            return None
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

    def flow_path_length(self) -> ToolOutput | None:
        if not self.should_compute(GeomorphometricalVariable.FLOW_PATH_LENGTH):
            return None
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

    def slope_length(self) -> ToolOutput | None:
        if not self.should_compute(GeomorphometricalVariable.SLOPE_LENGTH):
            return None
        tool = self.hydrology / 'Slope Length'
        return tool.execute(
            dem=self.dem,
            length=self.get_out_path(GeomorphometricalVariable.SLOPE_LENGTH),
            verbose=self.verbose,
            infer_obj_type=self.infer_obj_type,
            ignore_stderr=self.ignore_stderr,
        )

    def cell_balance(self) -> ToolOutput | None:
        if not self.should_compute(GeomorphometricalVariable.CELL_BALANCE):
            return None
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

    def topographic_wetness_index(self) -> ToolOutput | None:
        if not self.should_compute(
            GeomorphometricalVariable.TOPOGRAPHIC_WETNESS_INDEX
        ):
            return None
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
    variables: c.Sequence[GeomorphometricalVariable] | None = None,
):
    for tool_name, _ in TerrainAnalysis(
        dem,
        mode=mode,
        saga=saga,
        verbose=False,
        infer_obj_type=False,
        ignore_stderr=False,
        variables=variables,
    ).execute():
        logger.info('%s finished executing for %r' % (tool_name, mode))


def compute_grids(
    tiles: RasterTiles,
    mode: Mode,
    saga: SAGA,
    out_dir: Path | None = None,
    variables: c.Sequence[GeomorphometricalVariable] | None = None,
):
    dir_map = {
        Mode.TRAIN: TRAIN_TILES,
        Mode.TEST: TEST_TILES,
        Mode.INFERENCE: INFERENCE_TILES,
    }
    if out_dir is None:
        out_dir = dir_map[mode] / 'dem' / '100x100'
    logger.debug('Resampling %r for %sing' % (tiles, mode))
    resampled = tiles.resample(out_dir, mode)
    logger.debug('Merging %r for %sing' % (resampled, mode))
    merged = resampled.merge(GRIDS / mode.value / 'dem.tif')
    compute_grids_for_dem(merged, saga, mode, variables)
