from __future__ import annotations

import collections.abc as c
import concurrent.futures
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
from PySAGA_cmd.saga import SAGA
from rasterio import DatasetReader, windows
from rasterio.io import MemoryFile
from rasterio.mask import raster_geometry_mask
from shapely import Polygon, box
from shapely.geometry.base import BaseGeometry

from landnet.config import (
    EPSG,
    GRIDS,
    INTERIM_DATA_DIR,
    SAGAGIS_NODATA,
)
from landnet.enums import GeomorphometricalVariable, Mode
from landnet.features.dataset import (
    get_landslide_shapes,
    get_limit,
)
from landnet.features.tiles import TileConfig, TileHandler
from landnet.logger import create_logger
from landnet.typing import GridTypes, Metadata

if t.TYPE_CHECKING:
    from os import PathLike

    from PySAGA_cmd.saga import SAGA, Library, ToolOutput


logger = create_logger(__name__)


class TerrainAnalysis:
    def __init__(
        self,
        dem: PathLike,
        saga: SAGA,
        verbose: bool,
        infer_obj_type: bool,
        ignore_stderr: bool,
        dem_edge: PathLike | None = None,
        variables: c.Sequence[GeomorphometricalVariable] | None = None,
    ):
        self.dem = Path(dem)
        self.dem_edge = Path(dem_edge) if dem_edge is not None else None
        if self.dem_edge:
            assert self.dem_edge.is_relative_to(INTERIM_DATA_DIR)
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
        if self.variables is None:
            return True
        elif variable in self.variables:
            return True
        return False

    def index_of_convergence(self) -> ToolOutput | None:
        """Requires 1 or 2 units of buffer depending on the neighbours parameter."""
        if not self.should_compute(
            GeomorphometricalVariable.INDEX_OF_CONVERGENCE
        ):
            return None
        tool = self.morphometry / 'Convergence Index'
        dem = self.dem_edge if self.dem_edge is not None else self.dem
        return tool.execute(
            elevation=dem,
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

    def flow_accumulation_parallelizable(self) -> ToolOutput | None:
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


@dataclass
class Grid:
    path: Path
    tile_config: TileConfig
    mode: Mode
    landslides: gpd.GeoSeries | None = None
    tile_handler: TileHandler = field(init=False)

    def __post_init__(self):
        if self.tile_config:
            self.tile_handler = TileHandler(self.tile_config)

    def get_tiles_length(self) -> int:
        assert self.tile_handler is not None
        with rasterio.open(self.path, nodata=SAGAGIS_NODATA) as src:
            return self.tile_handler.get_tiles_length(src)

    def get_bounds(self, indices: c.Sequence[int]) -> gpd.GeoSeries:
        geoms = []
        with rasterio.open(self.path) as src:
            for i in indices:
                geoms.append(self.get_tile_bounds(i, src)[2])
        bounds = gpd.GeoSeries(data=geoms)
        bounds.set_crs(inplace=True, epsg=EPSG)
        return bounds

    def get_tile_landslides(self, bounds: Polygon) -> gpd.GeoSeries:
        landslides = (
            get_landslide_shapes(self.mode)
            if self.landslides is None
            else self.landslides
        )
        intersection = landslides.intersection(bounds)
        return intersection[~intersection.is_empty]

    def get_landslide_percentage_intersection(
        self, indices: c.Sequence[int]
    ) -> gpd.GeoDataFrame:
        assert self.tile_handler is not None
        bounds = self.get_bounds(indices)
        bounds = t.cast(gpd.GeoDataFrame, bounds.to_frame('geometry'))
        bounds['index'] = indices
        bounds['landslide_density'] = 0.0
        intersection = bounds.overlay(
            get_landslide_shapes(self.mode), how='intersection'
        )
        dissolved = intersection.dissolve('index')
        bounds.loc[dissolved.index, 'landslide_density'] = (
            dissolved.area / bounds.loc[dissolved.index].area
        )
        return bounds

    def get_tile_bounds(
        self, index: int, src: DatasetReader | None = None
    ) -> tuple[Metadata, windows.Window, Polygon]:
        assert self.tile_handler is not None

        should_close = src is None
        if src is None:
            src = rasterio.open(self.path, nodata=SAGAGIS_NODATA)
        metadata = src.meta.copy()

        window, transform = self.tile_handler.get_tile(src, index)
        metadata['transform'] = transform
        metadata['width'], metadata['height'] = (
            window.width,
            window.height,
        )
        bounds = box(*windows.bounds(window, src.transform))
        if should_close:
            src.close()
        return (metadata, window, bounds)

    def get_tile(self, index: int) -> tuple[Metadata, np.ndarray, Polygon]:
        metadata, window, bounds = self.get_tile_bounds(index)
        with rasterio.open(self.path, nodata=SAGAGIS_NODATA) as src:
            return (metadata, src.read(window=window), bounds)

    def write_tile(
        self,
        index: int,
        array: np.ndarray,
        prefix: str | None = None,
        out_dir: Path | None = None,
    ) -> Path:
        if out_dir is None:
            out_dir = self.path.parent
        if prefix is None:
            prefix = f'_{str(index)}'
        out_file = out_dir / f'{prefix}{self.path.name}'

        metadata, tile_array, bounds = self.get_tile(index)
        metadata.update({'count': array.shape[0]})
        assert array.ndim == 3, ('array must be 3-dimensional, got', array.ndim)
        if not array.shape[1:] == tile_array.shape[1:]:
            raise Exception(
                f'shape mismatch between arrays: {array.shape} -> {tile_array.shape}'
            )
        with rasterio.open(out_file, mode='w', **metadata) as dest:
            for i in range(array.shape[0] - 1, -1, -1):
                dest.write(array[i], i + 1)
        return out_file

    def get_tiles(self) -> c.Generator[tuple[Metadata, np.ndarray]]:
        assert self.tile_handler is not None

        with rasterio.open(self.path, nodata=SAGAGIS_NODATA) as src:
            metadata = src.meta.copy()
            for i in range(self.get_tiles_length()):
                window, transform = self.tile_handler.get_tile(src, i)
                metadata['transform'] = transform
                metadata['width'], metadata['height'] = (
                    window.width,
                    window.height,
                )
                yield (metadata, src.read(window=window))

    def get_tile_mask(self, index: int) -> tuple[Metadata, np.ndarray, Polygon]:
        metadata, window, bounds = self.get_tile_bounds(index)
        landslides = self.get_tile_landslides(bounds)
        with rasterio.open(self.path, nodata=SAGAGIS_NODATA) as src:
            array = src.read(window=window)
            if not landslides.empty:
                memfile = MemoryFile()
                dataset = memfile.open(**metadata)
                dataset.write(array)
                landslide_mask_raw, _, _ = raster_geometry_mask(
                    dataset, landslides, all_touched=True, invert=True
                )
                landslide_mask = np.expand_dims(landslide_mask_raw, 0)
            else:
                landslide_mask = np.zeros(array.shape)
            mask = np.concatenate(
                [np.logical_not(landslide_mask), landslide_mask]
            ).astype('uint8')
            assert (mask.sum(axis=0) == 1).all()
            return (metadata, mask, bounds)

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
        with rasterio.open(self.path, nodata=SAGAGIS_NODATA) as src:
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


def compute_grids_for_dem(
    dem: Path,
    saga: SAGA,
    variables: c.Sequence[GeomorphometricalVariable] | None = None,
):
    for tool_name, _ in TerrainAnalysis(
        dem,
        saga=saga,
        verbose=False,
        infer_obj_type=False,
        ignore_stderr=False,
        variables=variables,
    ).execute():
        logger.info('%s finished executing' % tool_name)


def compute_grids(
    dem: Path,
    mode: Mode,
    saga: SAGA,
    out_dir: Path | None = None,
    variables: c.Sequence[GeomorphometricalVariable] | None = None,
):
    with rasterio.open(dem) as src:
        profile = src.profile.copy()
        out_image, out_transform = rasterio.mask.mask(
            src, shapes=[get_limit(mode)], crop=True, filled=False
        )
        data_window = rasterio.windows.get_data_window(out_image)
        data_transform = rasterio.windows.transform(data_window, out_transform)
        profile.update(
            transform=data_transform,
            height=data_window.height,
            width=data_window.width,
        )

        data = src.read(window=data_window)

    with rasterio.open(dem, 'w', **profile) as dst:
        dst.write(data)

    compute_grids_for_dem(dem, saga, variables)


def get_grid_for_variable(
    variable: GeomorphometricalVariable,
    tile_config: TileConfig,
    mode: Mode,
    dir: Path | None = None,
) -> Grid:
    if dir is None:
        dir = GRIDS / mode.value
    return Grid(
        (dir / variable.value).with_suffix('.tif'),
        tile_config,
        mode=mode,
    )


T = t.TypeVar('T', bound=GeomorphometricalVariable)


def get_grid_types_for_variable(
    variable: T, tile_config: TileConfig
) -> tuple[T, GridTypes]:
    image_folders = {}
    for mode in Mode:
        image_folders[mode] = get_grid_for_variable(variable, tile_config, mode)
    return (variable, t.cast(GridTypes, image_folders))


def get_grid_types(
    tile_config: TileConfig,
) -> dict[GeomorphometricalVariable, GridTypes]:
    func = partial(get_grid_types_for_variable, tile_config=tile_config)
    logger.info(
        'Creating tiles with %r for all of the geomorphometrical variables'
        % tile_config
    )
    # results = [func(variable) for variable in GeomorphometricalVariable]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(func, GeomorphometricalVariable)
    return {variable: image_folders for variable, image_folders in results}
