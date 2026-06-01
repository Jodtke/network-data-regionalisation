from __future__ import annotations

"""Generate raster-based renewable and hydro profiles for reduced-grid studies.

This module implements the weather-dependent part of the preprocessing pipeline.
It follows the broad structure of Atlite/PyPSA-Eur profile generation, but keeps
the masks, region definitions, and technology classes explicit so that scenario
capacity can later be allocated to reduced network buses. The main outputs are
cell-level availability masks, maximum installable capacities, capacity-factor
time series for PV and wind technologies, investment resource classes, and
country or bus-level hydro inflow profiles.

The code deliberately separates physical resource quality from scenario
allocation. Raster masks describe where a technology could be built; TYNDP
capacity targets are only applied in the renewable disaggregation modules.
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import xarray as xr
import yaml
import atlite
from atlite import gis

LOG = logging.getLogger(__name__)
os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")

CUTOUT_DIR = Path(r"\\IIP-COMP103\endata\MA_Lisa\atlite_cutouts\cutouts")
OUT_DIR = Path(r"\\IIP-COMP103\endata\MA_Eric")
DATASHAPES_DIR = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf\datashapes")
NETWORK_DIR = Path(
    r"Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf\grid\target_year_2030\electrical_spectral_line_equivalent_electrical"
)

PV_PANEL = "CSi" # others??
PV_ORIENTATION = "latitude_optimal"
ONWIND_TURBINE = "Vestas_V112_3MW"
OFFWIND_TURBINE = "Vestas_V164_7MW_offshore"
OFFWIND_CONNECTION_TYPE = "combined" # ac + dc; in pypsa is separated

COUNTRY_SHAPES = DATASHAPES_DIR / "country_shapes.geojson"
COUNTRY_REDUCTIONS = DATASHAPES_DIR / "country_reductions.geojson"
EUROPE_SHAPE = DATASHAPES_DIR / "europe_shape.geojson"
ONSHORE_SHAPES = DATASHAPES_DIR / "regions_onshore.geojson"
OFFSHORE_SHAPES = DATASHAPES_DIR / "regions_offshore.geojson"
ONSHORE_MASK_SHAPES = (EUROPE_SHAPE,)
OFFSHORE_MASK_SHAPES = (OFFSHORE_SHAPES,)
NATURA_PATH = DATASHAPES_DIR / "natura.tiff"
SHIPDENSITY_PATH = DATASHAPES_DIR / "shipdensity_raster.nc"
CORINE_LANDCOVER_PATH = DATASHAPES_DIR / "CORINE_LANDCOVER_U2018_CLC2018_V2020_20u1.tif"
LUISA_LANDCOVER_PATH = DATASHAPES_DIR / "LUISA_basemap_020321_100m.tif"
LUISA_LEGEND_PATH = DATASHAPES_DIR / "Legend_Base_Map_2018.csv"
GEBCO_PATH = DATASHAPES_DIR / "GEBCO_2025.nc"
WDPA_POLYGON_PATHS = [
    DATASHAPES_DIR / "WDPA_WDOECM_Aug2025_Public_EU_shp_0" / "WDPA_WDOECM_Aug2025_Public_EU_shp-polygons.shp",
    DATASHAPES_DIR / "WDPA_WDOECM_Aug2025_Public_EU_shp_1" / "WDPA_WDOECM_Aug2025_Public_EU_shp-polygons.shp",
    DATASHAPES_DIR / "WDPA_WDOECM_Aug2025_Public_EU_shp_2" / "WDPA_WDOECM_Aug2025_Public_EU_shp-polygons.shp",
]
WDPA_POINT_PATHS = [
    DATASHAPES_DIR / "WDPA_WDOECM_Aug2025_Public_EU_shp_0" / "WDPA_WDOECM_Aug2025_Public_EU_shp-points.shp",
    DATASHAPES_DIR / "WDPA_WDOECM_Aug2025_Public_EU_shp_1" / "WDPA_WDOECM_Aug2025_Public_EU_shp-points.shp",
    DATASHAPES_DIR / "WDPA_WDOECM_Aug2025_Public_EU_shp_2" / "WDPA_WDOECM_Aug2025_Public_EU_shp-points.shp",
]
WDPA_POINT_BUFFER_M = 1000.0
WDPA_ONSHORE_STRICT_FILTER = "terrestrial"
WDPA_ONSHORE_STRICT_IUCN = ("IA", "IB", "II", "III")
WDPA_ONSHORE_STRICT_STATUS = ("ADOPTED", "DESIGNATED", "INSCRIBED")
SHIPDENSITY_THRESHOLD = 1.0e7 # higher than pypsa, more conservative
HYDRO_REFERENCE = Path(
    r"Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf\renewables\hydro\hydro_country_profiles.nc"
)
HYDRO_REFERENCE_YEAR = 2013
HYDRO_WRITE_BUS_PROFILES = False

HYDRO_TYPES = ["ror", "reservoir", "phs"]
HYDRO_TECH_TO_TYPE = {
    "Run-Of-River": "ror",
    "Reservoir": "reservoir",
    "Pumped Storage": "phs",
}

PYPSA_CAPACITY_PER_SQKM_PV = 5.1
PYPSA_CAPACITY_PER_SQKM_ONWIND = 3.0
PYPSA_CAPACITY_PER_SQKM_OFFWIND = 2.0

CLC_CODE_TO_CLASS = {
    111: 1,
    112: 2,
    121: 3,
    122: 4,
    123: 5,
    124: 6,
    131: 7,
    132: 8,
    133: 9,
    141: 10,
    142: 11,
    211: 12,
    212: 13,
    213: 14,
    221: 15,
    222: 16,
    223: 17,
    231: 18,
    241: 19,
    242: 20,
    243: 21,
    244: 22,
    311: 23,
    312: 24,
    313: 25,
    321: 26,
    322: 27,
    323: 28,
    324: 29,
    331: 30,
    332: 31,
    333: 32,
    334: 33,
    335: 34,
    411: 35,
    412: 36,
    421: 37,
    422: 38,
    423: 39,
    511: 40,
    512: 41,
    521: 42,
    522: 43,
    523: 44,
}
CLC_CLASSES = set(CLC_CODE_TO_CLASS.values())
CLC_NODATA_CLASSES = {48}
CLC_NODATA_CODES = {999}
LUISA_NODATA_CODES = {0}

PYPSA_ONWIND_CLC_CODES = {
    211,
    212,
    213,
    221,
    222,
    223,
    231,
    241,
    242,
    243,
    244,
    311,
    312,
    313,
    321,
    322,
    323,
    324,
    332,
    333,
}
PYPSA_SOLAR_CLC_CODES = {
    111,
    112,
    121,
    122,
    123,
    124,
    131,
    132,
    133,
    141,
    142,
    211,
    212,
    213,
    221,
    222,
    223,
    231,
    241,
    242,
    321,
    332,
    333,
}
PYPSA_OFFWIND_DEFAULTS = {
    "ac": {"max_depth": 60.0, "max_shore_distance": 30000.0},
    "dc": {"max_depth": 60.0, "min_shore_distance": 30000.0},
    "acdc": {"max_depth": 60.0},
    "float": {"min_depth": 60.0, "max_depth": 1000.0},
}
PYPSA_SHIP_THRESHOLD = 400.0
PYPSA_SHIP_HOURS = 8760 * 6
PYPSA_ONWIND_URBAN_DISTANCE_M = 1000.0
PYPSA_ONWIND_URBAN_CLC_CODES = {111, 112, 121, 122, 123, 124}
PYPSA_LUISA_URBAN_CODES = {1111, 1121, 1122, 1123, 1130, 1210, 1221, 1222, 1230, 1241, 1242}
PYPSA_LUISA_ONWIND_CODES = {code * 10 for code in PYPSA_ONWIND_CLC_CODES}
PYPSA_LUISA_SOLAR_CODES = {
    1111,
    1121,
    1122,
    1123,
    1130,
    1210,
    1221,
    1222,
    1230,
    1241,
    1242,
    1310,
    1320,
    1330,
    1410,
    1421,
    1422,
    2110,
    2120,
    2130,
    2210,
    2220,
    2230,
    2310,
    2410,
    2420,
    3210,
    3320,
    3330,
}


@dataclass(frozen=True)
class LandcoverConfig:
    include: set[int] | None
    exclude: set[int] | None
    mode: str
    codes_input: str
    classes_mapped: str

    @property
    def active(self) -> bool:
        return bool(self.mode)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_yaml_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit("config YAML must be a mapping of keys to values")
    return data


def _normalize_config_keys(data: dict) -> dict:
    normalized: dict[str, object] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise SystemExit("config keys must be strings")
        dest = key.strip().replace("-", "_")
        if dest in normalized:
            raise SystemExit(f"duplicate config key: {dest}")
        normalized[dest] = value
    return normalized


def _config_to_argv(config: dict, parser: argparse.ArgumentParser) -> list[str]:
    action_by_dest = {
        action.dest: action
        for action in parser._actions
        if action.option_strings and action.dest != "help"
    }
    argv: list[str] = []
    for key, value in config.items():
        if key == "config":
            continue
        action = action_by_dest.get(key)
        if action is None:
            raise SystemExit(f"unknown config key: {key}")
        if isinstance(action, argparse._StoreTrueAction):
            if bool(value):
                argv.append(action.option_strings[-1])
            continue
        if isinstance(action, argparse._StoreFalseAction):
            if not bool(value):
                argv.append(action.option_strings[-1])
            continue
        if value is None:
            continue
        option = action.option_strings[-1]
        if action.nargs is None or action.nargs == 1 or action.nargs == "?":
            if isinstance(value, (list, tuple)):
                if len(value) != 1:
                    raise SystemExit(f"config key {key} expects a single value")
                value = value[0]
            argv.extend([option, str(value)])
        else:
            if isinstance(value, (list, tuple)):
                values = list(value)
            else:
                values = [value]
            if not values:
                continue
            argv.append(option)
            argv.extend(str(v) for v in values)
    return argv


def _validate_windturbine(turbine: str, label: str) -> None:
    if not turbine or not str(turbine).strip():
        raise SystemExit(f"{label} turbine is empty")
    try:
        atlite.resource.get_windturbineconfig(turbine)
    except Exception as exc:
        raise SystemExit(
            f"{label} turbine '{turbine}' not found or OEDB unavailable. Original error: {exc}"
        )


def _clip_p_max_pu(cf: xr.DataArray, threshold: float | None) -> xr.DataArray:
    if threshold is None or threshold <= 0:
        return cf
    return cf.where(cf >= threshold, 0.0)


def _cutout_path(year: int, cutout_dir: Path) -> Path:
    return cutout_dir / f"europe-{year}.nc"


def _cf_out_path(kind: str, year: int, out_dir: Path) -> Path:
    out_subdir = out_dir / kind
    _ensure_dir(out_subdir)
    return out_subdir / f"{kind}_cf_{year}.nc"


def _hydro_out_paths(year: int, out_dir: Path) -> tuple[Path, Path]:
    out_subdir = out_dir / "hydro"
    _ensure_dir(out_subdir)
    return (
        out_subdir / f"hydro_bus_profiles_{year}.nc",
        out_subdir / f"hydro_country_profiles_{year}.nc",
    )


def _write_cf(cf: xr.DataArray, out_path: Path, attrs: dict[str, str]) -> None:
    cf = cf.rename("cf").astype("float32").chunk({"time": 168})
    ds = cf.to_dataset()
    ds.attrs.update(attrs)
    encoding = {"cf": {"zlib": True, "complevel": 4, "dtype": "float32"}}
    LOG.info("writing %s", out_path)
    ds.to_netcdf(out_path, engine="netcdf4", encoding=encoding)


def _mask_from_geometry(
    cutout: atlite.Cutout, geometry, geometry_crs
) -> xr.DataArray:
    matrix = _indicator_matrix(cutout, [geometry], geometry_crs)
    coverage = np.asarray(matrix.sum(axis=0)).ravel()
    mask = xr.DataArray(
        coverage.reshape(cutout.shape),
        coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
        dims=("y", "x"),
    )
    return mask.clip(0.0, 1.0)


def _union_geometry(geometries) -> object:
    if hasattr(geometries, "union_all"):
        return geometries.union_all()
    return geometries.unary_union


def _make_valid_geometry(geometry) -> object:
    if geometry is None or getattr(geometry, "is_empty", False):
        return geometry
    if getattr(geometry, "is_valid", True):
        return geometry
    try:
        from shapely import make_valid as shapely_make_valid
    except Exception:
        try:
            return geometry.buffer(0)
        except Exception:
            return geometry
    try:
        return shapely_make_valid(geometry)
    except Exception:
        try:
            return geometry.buffer(0)
        except Exception:
            return geometry


def _geometry_precision_grid_size(geometry_crs) -> float:
    if geometry_crs is None:
        return 1.0
    try:
        crs = gpd.GeoSeries([], crs=geometry_crs).crs
    except Exception:
        return 1.0
    if crs is not None and getattr(crs, "is_geographic", False):
        return 1.0e-5
    return 1.0


def _snap_geometry_precision(geometry, geometry_crs):
    geometry = _make_valid_geometry(geometry)
    if geometry is None or getattr(geometry, "is_empty", False):
        return geometry
    try:
        from shapely import set_precision as shapely_set_precision
    except Exception:
        return geometry
    grid_size = _geometry_precision_grid_size(geometry_crs)
    try:
        geometry = shapely_set_precision(geometry, grid_size)
    except Exception:
        return geometry
    return _make_valid_geometry(geometry)


def _snap_shapes_precision(shapes, shapes_crs):
    if isinstance(shapes, gpd.GeoSeries):
        snapped = shapes.apply(lambda geom: _snap_geometry_precision(geom, shapes_crs))
        return gpd.GeoSeries(snapped, index=shapes.index, crs=shapes.crs or shapes_crs)
    if isinstance(shapes, tuple):
        return tuple(_snap_geometry_precision(geom, shapes_crs) for geom in shapes)
    if isinstance(shapes, list):
        return [_snap_geometry_precision(geom, shapes_crs) for geom in shapes]
    return [_snap_geometry_precision(shapes, shapes_crs)]


def _indicator_matrix(cutout: atlite.Cutout, shapes, shapes_crs):
    try:
        return cutout.indicatormatrix(shapes, shapes_crs=shapes_crs)
    except Exception as exc:
        try:
            from shapely.errors import GEOSException
        except Exception:
            GEOSException = None
        if GEOSException is None or not isinstance(exc, GEOSException):
            raise
    grid_size = _geometry_precision_grid_size(shapes_crs)
    LOG.warning(
        "indicatormatrix GEOSException; retrying with snapped geometry precision (grid_size=%s)",
        grid_size,
    )
    snapped_shapes = _snap_shapes_precision(shapes, shapes_crs)
    return cutout.indicatormatrix(snapped_shapes, shapes_crs=shapes_crs)


def _make_valid_geometries(geometries: gpd.GeoSeries) -> gpd.GeoSeries:
    geometries = geometries[geometries.notna() & ~geometries.is_empty]
    if geometries.empty:
        return geometries
    if hasattr(geometries, "make_valid"):
        try:
            return geometries.make_valid()
        except Exception:
            pass
    try:
        from shapely import make_valid as shapely_make_valid
    except Exception:
        return geometries.buffer(0)
    return geometries.apply(shapely_make_valid)


def _mask_from_shapes(
    cutout: atlite.Cutout, shapes_path: Path | list[Path] | tuple[Path, ...]
) -> xr.DataArray:
    if isinstance(shapes_path, (list, tuple)):
        if not shapes_path:
            return xr.DataArray(
                np.zeros(cutout.shape, dtype="float32"),
                coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
                dims=("y", "x"),
            )
        mask = None
        for path in shapes_path:
            next_mask = _mask_from_shapes(cutout, path)
            mask = next_mask if mask is None else mask * next_mask
        return mask.clip(0.0, 1.0)

    shapes = gpd.read_file(shapes_path)
    if shapes.empty:
        return xr.DataArray(
            np.zeros(cutout.shape, dtype="float32"),
            coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
            dims=("y", "x"),
        )

    geometries = _make_valid_geometries(shapes.geometry)
    if geometries.empty:
        return xr.DataArray(
            np.zeros(cutout.shape, dtype="float32"),
            coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
            dims=("y", "x"),
        )
    geom = _union_geometry(geometries)
    return _mask_from_geometry(cutout, geom, shapes.crs)


def _natura_mask(cutout: atlite.Cutout, natura_path: Path) -> xr.DataArray:
    with rio.open(natura_path) as src:
        dest = np.zeros(cutout.shape, dtype="float32")
        rio.warp.reproject(
            source=rio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=cutout.transform_r,
            dst_crs=cutout.crs,
            dst_nodata=0,
            resampling=rio.warp.Resampling.nearest,
        )
    dest = dest[::-1, :]
    eligible = 1.0 - np.where(dest > 0, 1.0, 0.0)
    return xr.DataArray(
        eligible,
        coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
        dims=("y", "x"),
    )


def _shore_distance_mask(
    cutout: atlite.Cutout,
    country_shapes_path: Path,
    min_distance: float | None,
    max_distance: float | None,
) -> xr.DataArray:
    if min_distance is None and max_distance is None:
        raise ValueError("shore distance mask requested without min/max distance")

    shapes = gpd.read_file(country_shapes_path)
    if shapes.empty:
        LOG.warning("empty country shapes for shore distance mask, returning all-ones mask")
        return xr.DataArray(
            np.ones(cutout.shape, dtype="float32"),
            coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
            dims=("y", "x"),
        )

    if shapes.crs is None:
        raise ValueError(f"country shapes missing CRS: {country_shapes_path}")

    shapes_proj = shapes
    if not shapes.crs.is_projected:
        shapes_proj = shapes.to_crs(3035)

    geometries = _make_valid_geometries(shapes_proj.geometry)
    if geometries.empty:
        LOG.warning("empty country shapes after validation, returning all-ones mask")
        return xr.DataArray(
            np.ones(cutout.shape, dtype="float32"),
            coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
            dims=("y", "x"),
        )
    geom = _union_geometry(geometries)
    mask = xr.DataArray(
        np.ones(cutout.shape, dtype="float32"),
        coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
        dims=("y", "x"),
    )
    if min_distance is not None:
        geom_min = geom.buffer(min_distance)
        min_mask = _mask_from_geometry(cutout, geom_min, shapes_proj.crs)
        mask = mask * (1.0 - min_mask)
    if max_distance is not None:
        geom_max = geom.buffer(max_distance)
        max_mask = _mask_from_geometry(cutout, geom_max, shapes_proj.crs)
        mask = mask * max_mask
    return mask.clip(0.0, 1.0)


def _urban_distance_mask(
    cutout: atlite.Cutout,
    onshore_shapes_path: Path | list[Path] | tuple[Path, ...],
    landuse_path: Path,
    urban_codes: set[int],
    distance_m: float,
    res: float = 100.0,
    *,
    normalize_fn,
    legend_path: Path | None = None,
    nodata_default: int | None = None,
) -> xr.DataArray:
    shapes_key = _normalize_shapes_paths(onshore_shapes_path)
    geometry = _load_union_geometry_cached(shapes_key, 3035)
    if geometry.empty:
        return xr.DataArray(
            np.zeros(cutout.shape, dtype="float32"),
            coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
            dims=("y", "x"),
        )

    urban_classes, _, _ = normalize_fn(urban_codes, legend_path)
    if urban_classes is None:
        return xr.DataArray(
            np.ones(cutout.shape, dtype="float32"),
            coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
            dims=("y", "x"),
        )

    excluder = atlite.ExclusionContainer(crs=3035, res=res)
    with rio.open(landuse_path) as src:
        nodata = src.nodata if src.nodata is not None else nodata_default
        if nodata is None:
            nodata = -128
    excluder.add_raster(
        landuse_path,
        codes=sorted(urban_classes),
        buffer=distance_m,
        nodata=nodata,
    )
    mask, _ = gis.shape_availability_reprojected(
        geometry,
        excluder,
        cutout.transform_r,
        cutout.crs,
        cutout.shape,
    )
    mask = mask[::-1, :]
    return xr.DataArray(
        mask,
        coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
        dims=("y", "x"),
    ).clip(0.0, 1.0)


def _shipdensity_mask(
    cutout: atlite.Cutout, shipdensity_path: Path, threshold: float
) -> xr.DataArray:
    ds = xr.open_dataset(shipdensity_path)
    try:
        var = list(ds.data_vars)[0]
        da = ds[var]
        if "band" in da.dims:
            da = da.isel(band=0, drop=True)
        ship = gis.regrid(
            da,
            cutout.data.coords["x"],
            cutout.data.coords["y"],
            resampling=rio.warp.Resampling.average,
        )
        mask = xr.where(ship > threshold, 0.0, 1.0).astype("float32")
        return mask
    finally:
        ds.close()


def _bathymetry_mask(
    cutout: atlite.Cutout,
    gebco_path: Path,
    min_depth: float | None,
    max_depth: float | None,
) -> xr.DataArray:
    if min_depth is None and max_depth is None:
        raise ValueError("bathymetry mask requested without min/max depth")

    ds = xr.open_dataset(gebco_path)
    try:
        var = "elevation" if "elevation" in ds.data_vars else list(ds.data_vars)[0]
        da = ds[var]
        if "band" in da.dims:
            da = da.isel(band=0, drop=True)
        if "lon" not in da.coords or "lat" not in da.coords:
            raise ValueError(f"bathymetry dataset missing lon/lat coords: {gebco_path}")

        lon = cutout.data.coords["x"]
        lat = cutout.data.coords["y"]
        lon_min, lon_max = float(lon.min()), float(lon.max())
        lat_min, lat_max = float(lat.min()), float(lat.max())
        da = da.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
        da = da.rename({"lon": "x", "lat": "y"})

        eligible = xr.ones_like(da, dtype="float32")
        if max_depth is not None:
            eligible = eligible.where(da >= -max_depth, 0.0)
        if min_depth is not None:
            eligible = eligible.where(da <= -min_depth, 0.0)
        eligible = eligible.where(np.isfinite(da), 0.0)

        mask = gis.regrid(
            eligible,
            lon,
            lat,
            resampling=rio.warp.Resampling.average,
        )
        return mask.clip(0.0, 1.0)
    finally:
        ds.close()


def _landcover_mask(
    cutout: atlite.Cutout,
    CORINE_LANDCOVER_PATH: Path,
    include_classes: set[int] | None,
    exclude_classes: set[int] | None,
) -> xr.DataArray:
    if include_classes is None and exclude_classes is None:
        raise ValueError("landcover mask requested without include/exclude classes")
    if include_classes is not None and exclude_classes is not None:
        raise ValueError("landcover mask supports only include or exclude classes")

    include_list = sorted(include_classes) if include_classes else None
    exclude_list = sorted(exclude_classes) if exclude_classes else None
    mode = "include" if include_list is not None else "exclude"
    count = len(include_list) if include_list is not None else len(exclude_list or [])
    LOG.info("building landcover mask (%s, codes=%d) from %s", mode, count, CORINE_LANDCOVER_PATH)

    with rio.open(CORINE_LANDCOVER_PATH) as src:
        src_arr = src.read(1, masked=True)
        raw = src_arr.data
        nodata_mask = np.ma.getmaskarray(src_arr).copy()
        if src.nodata is not None:
            nodata_mask |= raw == src.nodata
        if CLC_NODATA_CLASSES:
            nodata_mask |= np.isin(raw, list(CLC_NODATA_CLASSES))
        if include_list is not None:
            eligible = np.isin(raw, include_list)
        else:
            eligible = ~np.isin(raw, exclude_list)
        eligible = np.where(nodata_mask, 0, eligible)

        source = eligible.astype("float32")
        dest = np.zeros(cutout.shape, dtype="float32")
        rio.warp.reproject(
            source=source,
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=cutout.transform_r,
            dst_crs=cutout.crs,
            dst_nodata=0,
            resampling=rio.warp.Resampling.average,
        )
    dest = dest[::-1, :]
    return xr.DataArray(
        dest,
        coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
        dims=("y", "x"),
    ).clip(0.0, 1.0)


def _luisa_mask(
    cutout: atlite.Cutout,
    LUISA_LANDCOVER_PATH: Path,
    include_codes: set[int] | None,
    exclude_codes: set[int] | None,
) -> xr.DataArray:
    if include_codes is None and exclude_codes is None:
        raise ValueError("luisa mask requested without include/exclude codes")
    if include_codes is not None and exclude_codes is not None:
        raise ValueError("luisa mask supports only include or exclude codes")

    include_list = sorted(include_codes) if include_codes else None
    exclude_list = sorted(exclude_codes) if exclude_codes else None
    mode = "include" if include_list is not None else "exclude"
    count = len(include_list) if include_list is not None else len(exclude_list or [])
    LOG.info("building LUISA mask (%s, codes=%d) from %s", mode, count, LUISA_LANDCOVER_PATH)

    with rio.open(LUISA_LANDCOVER_PATH) as src:
        src_arr = src.read(1, masked=True)
        raw = src_arr.data
        nodata_mask = np.ma.getmaskarray(src_arr).copy()
        nodata_value = src.nodata if src.nodata is not None else 0
        nodata_mask |= raw == nodata_value
        if LUISA_NODATA_CODES:
            nodata_mask |= np.isin(raw, list(LUISA_NODATA_CODES))

        if include_list is not None:
            eligible = np.isin(raw, include_list)
        else:
            eligible = ~np.isin(raw, exclude_list)
        eligible = np.where(nodata_mask, 0, eligible)

        source = eligible.astype("float32")
        dest = np.zeros(cutout.shape, dtype="float32")
        rio.warp.reproject(
            source=source,
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=cutout.transform_r,
            dst_crs=cutout.crs,
            dst_nodata=0,
            resampling=rio.warp.Resampling.average,
        )
    dest = dest[::-1, :]
    return xr.DataArray(
        dest,
        coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
        dims=("y", "x"),
    ).clip(0.0, 1.0)


def _landcover_valid_mask(
    cutout: atlite.Cutout, CORINE_LANDCOVER_PATH: Path
) -> xr.DataArray:
    LOG.info("building landcover valid mask from %s", CORINE_LANDCOVER_PATH)
    with rio.open(CORINE_LANDCOVER_PATH) as src:
        src_arr = src.read(1, masked=True)
        raw = src_arr.data
        nodata_mask = np.ma.getmaskarray(src_arr).copy()
        if src.nodata is not None:
            nodata_mask |= raw == src.nodata
        if CLC_NODATA_CLASSES:
            nodata_mask |= np.isin(raw, list(CLC_NODATA_CLASSES))
        if CLC_NODATA_CODES:
            nodata_mask |= np.isin(raw, list(CLC_NODATA_CODES))

        valid = ~nodata_mask
        source = valid.astype("float32")
        dest = np.zeros(cutout.shape, dtype="float32")
        rio.warp.reproject(
            source=source,
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=cutout.transform_r,
            dst_crs=cutout.crs,
            dst_nodata=0,
            resampling=rio.warp.Resampling.average,
        )
    dest = dest[::-1, :]
    return xr.DataArray(
        dest,
        coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
        dims=("y", "x"),
    ).clip(0.0, 1.0)


def _luisa_valid_mask(
    cutout: atlite.Cutout, LUISA_LANDCOVER_PATH: Path
) -> xr.DataArray:
    LOG.info("building LUISA valid mask from %s", LUISA_LANDCOVER_PATH)
    with rio.open(LUISA_LANDCOVER_PATH) as src:
        src_arr = src.read(1, masked=True)
        raw = src_arr.data
        nodata_mask = np.ma.getmaskarray(src_arr).copy()
        nodata_value = src.nodata if src.nodata is not None else 0
        nodata_mask |= raw == nodata_value
        if LUISA_NODATA_CODES:
            nodata_mask |= np.isin(raw, list(LUISA_NODATA_CODES))

        valid = ~nodata_mask
        source = valid.astype("float32")
        dest = np.zeros(cutout.shape, dtype="float32")
        rio.warp.reproject(
            source=source,
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=cutout.transform_r,
            dst_crs=cutout.crs,
            dst_nodata=0,
            resampling=rio.warp.Resampling.average,
        )
    dest = dest[::-1, :]
    return xr.DataArray(
        dest,
        coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
        dims=("y", "x"),
    ).clip(0.0, 1.0)


def _combine_landuse_masks(
    landuse_dataset: str,
    landuse_fusion: str,
    corine_mask: xr.DataArray | None,
    luisa_mask: xr.DataArray | None,
    corine_valid: xr.DataArray | None,
    luisa_valid: xr.DataArray | None,
) -> xr.DataArray | None:
    if landuse_dataset == "corine":
        return corine_mask
    if landuse_dataset == "luisa":
        return luisa_mask
    if corine_mask is None:
        return luisa_mask
    if luisa_mask is None:
        return corine_mask
    if landuse_fusion == "prefer-corine":
        # Prefer one land-cover source only where it is valid; otherwise use the
        # other source to avoid unnecessary holes along national data boundaries.
        if corine_valid is None:
            return (corine_mask * luisa_mask).clip(0.0, 1.0)
        combined = corine_mask + luisa_mask * (1.0 - corine_valid)
        return combined.clip(0.0, 1.0)
    if landuse_fusion == "prefer-luisa":
        if luisa_valid is None:
            return (corine_mask * luisa_mask).clip(0.0, 1.0)
        combined = luisa_mask + corine_mask * (1.0 - luisa_valid)
        return combined.clip(0.0, 1.0)
    return (corine_mask * luisa_mask).clip(0.0, 1.0)


def _build_mask(
    cutout: atlite.Cutout,
    shapes_path: Path | list[Path] | tuple[Path, ...],
    exclude_natura: bool,
    natura_path: Path,
    exclude_shipdensity: bool,
    shipdensity_path: Path,
    shipdensity_threshold: float,
    landcover_mask: xr.DataArray | None,
    bathymetry_mask: xr.DataArray | None,
    shore_distance_mask: xr.DataArray | None,
) -> xr.DataArray:
    mask = _mask_from_shapes(cutout, shapes_path)
    if landcover_mask is not None:
        mask = mask * landcover_mask
    if bathymetry_mask is not None:
        mask = mask * bathymetry_mask
    if shore_distance_mask is not None:
        mask = mask * shore_distance_mask
    if exclude_natura:
        mask = mask * _natura_mask(cutout, natura_path)
    if exclude_shipdensity:
        mask = mask * _shipdensity_mask(cutout, shipdensity_path, shipdensity_threshold)
    return mask.clip(0.0, 1.0)


def _apply_mask(cf: xr.DataArray, mask: xr.DataArray | None) -> xr.DataArray:
    if mask is None:
        return cf
    attrs = cf.attrs.copy()
    # Availability already enters p_nom_max; CF masks should only exclude cells.
    cf = cf.where(mask > 0)
    cf.attrs.update(attrs)
    return cf


def _format_class_list(classes: set[int] | None) -> str:
    if not classes:
        return ""
    return ",".join(str(value) for value in sorted(classes))


def _format_string_list(values: tuple[str, ...] | list[str] | set[str] | None) -> str:
    if not values:
        return ""
    return ",".join(str(value) for value in values)


def _normalize_shapes_paths(
    shapes_paths: Path | list[Path] | tuple[Path, ...],
) -> tuple[str, ...]:
    if isinstance(shapes_paths, (list, tuple)):
        paths = shapes_paths
    else:
        paths = (shapes_paths,)
    return tuple(str(path) for path in paths)


def _format_path_list(paths: list[Path] | tuple[Path, ...] | None) -> str:
    if not paths:
        return ""
    return ";".join(str(path) for path in paths)


def _normalize_landcover_codes(
    codes: set[int] | None,
    _legend_path: Path | None = None,
) -> tuple[set[int] | None, str, str]:
    if not codes:
        return None, "", ""

    clean = set(codes)
    dropped_codes = clean & CLC_NODATA_CODES
    if dropped_codes:
        LOG.warning("dropping nodata CLC codes: %s", sorted(dropped_codes))
        clean -= dropped_codes
    if not clean:
        raise ValueError("no valid CLC codes after removing nodata")

    non_nodata = clean - CLC_NODATA_CLASSES
    if not non_nodata:
        raise ValueError("no valid CLC classes after removing nodata")
    has_clc = any(code >= 100 for code in non_nodata)
    has_class = any(code < 100 for code in non_nodata)
    if has_clc and has_class:
        raise ValueError("mixed CLC codes and raster classes are not supported")

    if has_clc:
        mapped: set[int] = set()
        unknown: list[int] = []
        for code in clean:
            if code in CLC_CODE_TO_CLASS:
                mapped.add(CLC_CODE_TO_CLASS[code])
            elif code in CLC_NODATA_CLASSES:
                LOG.warning("dropping nodata CLC class: %s", code)
            else:
                unknown.append(code)
        if unknown:
            raise ValueError(f"unknown CLC codes: {sorted(unknown)}")
        if not mapped:
            raise ValueError("no valid CLC codes after mapping")
        return mapped, _format_class_list(clean), _format_class_list(mapped)

    unknown = [code for code in clean if code not in CLC_CLASSES and code not in CLC_NODATA_CLASSES]
    if unknown:
        raise ValueError(f"unknown CLC raster classes: {sorted(unknown)}")
    mapped = clean - CLC_NODATA_CLASSES
    if clean != mapped:
        LOG.warning("dropping nodata CLC classes: %s", sorted(clean - mapped))
    if not mapped:
        raise ValueError("no valid CLC raster classes after removing nodata")
    return mapped, _format_class_list(clean), _format_class_list(mapped)


@lru_cache(maxsize=1)
def _load_luisa_legend_codes(legend_path: str) -> set[int] | None:
    path = Path(legend_path)
    if not path.exists():
        LOG.warning("LUISA legend not found: %s", path)
        return None

    df = pd.read_csv(path, sep=";")
    if "Code" not in df.columns:
        LOG.warning("LUISA legend missing 'Code' column: %s", path)
        return None
    return set(df["Code"].dropna().astype(int).tolist())


def _normalize_luisa_codes(
    codes: set[int] | None,
    legend_path: Path | None = None,
) -> tuple[set[int] | None, str, str]:
    if not codes:
        return None, "", ""

    clean = set(codes)
    dropped_codes = clean & LUISA_NODATA_CODES
    if dropped_codes:
        LOG.warning("dropping nodata LUISA codes: %s", sorted(dropped_codes))
        clean -= dropped_codes
    if not clean:
        raise ValueError("no valid LUISA codes after removing nodata")

    if legend_path is not None:
        valid = _load_luisa_legend_codes(str(legend_path))
        if valid is not None:
            unknown = [code for code in clean if code not in valid]
            if unknown:
                raise ValueError(f"unknown LUISA codes: {sorted(unknown)}")

    return clean, _format_class_list(clean), _format_class_list(clean)


def _make_landcover_config(
    include_raw: set[int] | None, exclude_raw: set[int] | None
) -> LandcoverConfig:
    include, include_input, include_mapped = _normalize_landcover_codes(include_raw)
    exclude, exclude_input, exclude_mapped = _normalize_landcover_codes(exclude_raw)
    if include:
        return LandcoverConfig(include, None, "include", include_input, include_mapped)
    if exclude:
        return LandcoverConfig(None, exclude, "exclude", exclude_input, exclude_mapped)
    return LandcoverConfig(None, None, "", "", "")


def _make_luisa_config(
    include_raw: set[int] | None, exclude_raw: set[int] | None, legend_path: Path
) -> LandcoverConfig:
    include, include_input, include_mapped = _normalize_luisa_codes(include_raw, legend_path)
    exclude, exclude_input, exclude_mapped = _normalize_luisa_codes(exclude_raw, legend_path)
    if include:
        return LandcoverConfig(include, None, "include", include_input, include_mapped)
    if exclude:
        return LandcoverConfig(None, exclude, "exclude", exclude_input, exclude_mapped)
    return LandcoverConfig(None, None, "", "", "")


def _landuse_cache_key(dataset: str, config: LandcoverConfig) -> str:
    return f"{dataset}:{config.mode}:{config.classes_mapped}"


def _select_landcover_raw(
    specific_include: set[int] | None,
    specific_exclude: set[int] | None,
    global_include: set[int] | None,
    global_exclude: set[int] | None,
    default_include: set[int] | None,
) -> tuple[set[int] | None, set[int] | None]:
    if specific_include or specific_exclude:
        return specific_include, specific_exclude
    if global_include or global_exclude:
        return global_include, global_exclude
    if default_include:
        return set(default_include), None
    return None, None


def _availability_out_path(kind: str, out_dir: Path) -> Path:
    return out_dir / f"availability_{kind}.nc"


def _cutout_area_km2(cutout: atlite.Cutout) -> xr.DataArray:
    area = cutout.grid.to_crs(3035).area / 1e6
    return xr.DataArray(
        area.values.reshape(cutout.shape),
        coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
        dims=("y", "x"),
    )


def _load_regions(regions_path: Path) -> gpd.GeoDataFrame:
    regions = gpd.read_file(regions_path)
    if regions.empty:
        return regions
    if "name" in regions.columns:
        regions = regions.set_index("name").rename_axis("bus")
    else:
        regions = regions.set_index(regions.index.astype(str)).rename_axis("bus")
    if regions.crs is None:
        raise ValueError(f"regions missing CRS: {regions_path}")
    return regions


def _availability_by_regions(
    cutout: atlite.Cutout, regions: gpd.GeoDataFrame, mask: xr.DataArray | None
) -> xr.DataArray:
    matrix = _indicator_matrix(cutout, regions.geometry, regions.crs)
    coverage = matrix.toarray().astype("float32")
    # The indicator matrix preserves partial cell overlap. Multiplying by the
    # technology mask afterwards keeps administrative geometry and siting rules
    # as separable modelling assumptions.
    availability = coverage.reshape((len(regions),) + cutout.shape)
    da = xr.DataArray(
        availability,
        coords={"bus": regions.index.values, "y": cutout.data.y.data, "x": cutout.data.x.data},
        dims=("bus", "y", "x"),
    )
    if mask is not None:
        da = da * mask
    return da.clip(0.0, 1.0)


def _mask_gt_zero(values: np.ndarray) -> np.ndarray:
    return values > 0


def _bathymetry_exclusion(
    values: np.ndarray, min_depth: float | None, max_depth: float | None
) -> np.ndarray:
    mask = ~np.isfinite(values)
    if max_depth is not None:
        mask |= values < -max_depth
    if min_depth is not None:
        mask |= values > -min_depth
    return mask


def _raster_nodata(path: Path, default: int) -> float:
    try:
        with rio.open(path) as src:
            nodata = src.nodata
    except Exception:
        return float(default)
    return float(default if nodata is None else nodata)


def _excluder_codes_from_config(
    config: LandcoverConfig,
    extra_exclude: set[int] | None,
    nodata_value: float | None,
) -> tuple[list[int] | None, bool]:
    if config.include:
        return sorted(config.include), True
    if config.exclude:
        codes = set(config.exclude)
        if extra_exclude:
            codes |= extra_exclude
        if nodata_value is not None:
            try:
                if not np.isnan(nodata_value):
                    codes.add(int(nodata_value))
            except TypeError:
                codes.add(int(nodata_value))
        return sorted(codes), False
    return None, False


@lru_cache(maxsize=8)
def _load_union_geometry_cached(
    shapes_paths: tuple[str, ...], target_crs: int
) -> gpd.GeoSeries:
    geometries_all: list[gpd.GeoSeries] = []
    for shapes_path in shapes_paths:
        shapes = gpd.read_file(shapes_path)
        if shapes.empty:
            continue
        if shapes.crs is None:
            raise ValueError(f"shapes missing CRS: {shapes_path}")
        shapes_proj = (
            shapes.to_crs(target_crs) if shapes.crs.to_epsg() != target_crs else shapes
        )
        geometries = _make_valid_geometries(shapes_proj.geometry)
        if not geometries.empty:
            geometries_all.append(geometries)
    if not geometries_all:
        return gpd.GeoSeries([], crs=target_crs)
    geometries = gpd.GeoSeries(
        pd.concat(geometries_all, ignore_index=True), crs=target_crs
    )
    geometry = _union_geometry(geometries)
    return gpd.GeoSeries([geometry], crs=target_crs)


def _normalize_wdpa_values(values: tuple[str, ...] | list[str] | None) -> set[str]:
    if not values:
        return set()
    return {str(v).strip().upper() for v in values if str(v).strip()}


def _find_wdpa_column(df: gpd.GeoDataFrame, candidates: list[str]) -> str | None:
    upper_to_actual = {c.upper(): c for c in df.columns}
    for name in candidates:
        if name in upper_to_actual:
            return upper_to_actual[name]
    return None


def _apply_wdpa_filter(
    df: gpd.GeoDataFrame,
    candidates: list[str],
    include_values: set[str],
    exclude_values: set[str],
    label: str,
) -> gpd.GeoDataFrame:
    if not include_values and not exclude_values:
        return df
    column = _find_wdpa_column(df, candidates)
    if column is None:
        LOG.warning("WDPA %s filter requested but columns not found (%s)", label, candidates)
        return df
    values = df[column].fillna("").astype(str).str.strip().str.upper()
    if include_values:
        return df[values.isin(include_values)]
    return df[~values.isin(exclude_values)]


def _filter_wdpa(
    df: gpd.GeoDataFrame,
    filter_mode: str,
    iucn_include: tuple[str, ...],
    iucn_exclude: tuple[str, ...],
    status_include: tuple[str, ...],
    status_exclude: tuple[str, ...],
    designation_include: tuple[str, ...],
    designation_exclude: tuple[str, ...],
    designation_type_include: tuple[str, ...],
    designation_type_exclude: tuple[str, ...],
) -> gpd.GeoDataFrame:
    if filter_mode == "marine":
        if "MARINE" in df.columns:
            col = df["MARINE"]
            if pd.api.types.is_numeric_dtype(col):
                df = df[col.fillna(0) > 0]
            else:
                values = col.fillna("").astype(str).str.strip().str.upper()
                df = df[~values.isin({"0", "N", "NO", "NONE", "FALSE", ""})]
    elif filter_mode == "terrestrial":
        if "MARINE" in df.columns:
            col = df["MARINE"]
            if pd.api.types.is_numeric_dtype(col):
                df = df[col.fillna(0) == 0]
            else:
                values = col.fillna("").astype(str).str.strip().str.upper()
                df = df[values.isin({"0", "N", "NO", "NONE", "FALSE", ""})]

    iucn_inc = _normalize_wdpa_values(iucn_include)
    iucn_exc = _normalize_wdpa_values(iucn_exclude)
    status_inc = _normalize_wdpa_values(status_include)
    status_exc = _normalize_wdpa_values(status_exclude)
    desig_inc = _normalize_wdpa_values(designation_include)
    desig_exc = _normalize_wdpa_values(designation_exclude)
    desig_type_inc = _normalize_wdpa_values(designation_type_include)
    desig_type_exc = _normalize_wdpa_values(designation_type_exclude)

    df = _apply_wdpa_filter(df, ["IUCN_CAT"], iucn_inc, iucn_exc, "IUCN")
    df = _apply_wdpa_filter(df, ["STATUS"], status_inc, status_exc, "STATUS")
    df = _apply_wdpa_filter(
        df,
        ["DESIG_ENG", "DESIG", "DESIGNATION", "DESIGN"],
        desig_inc,
        desig_exc,
        "DESIGNATION",
    )
    df = _apply_wdpa_filter(
        df,
        ["DESIG_TYPE", "DESIG_TY", "DESIGTYPE"],
        desig_type_inc,
        desig_type_exc,
        "DESIGNATION_TYPE",
    )
    return df


@lru_cache(maxsize=4)
@lru_cache(maxsize=4)
def _load_wdpa_geometries_cached(
    polygon_paths: tuple[str, ...],
    point_paths: tuple[str, ...],
    filter_mode: str,
    point_buffer_m: float,
    iucn_include: tuple[str, ...],
    iucn_exclude: tuple[str, ...],
    status_include: tuple[str, ...],
    status_exclude: tuple[str, ...],
    designation_include: tuple[str, ...],
    designation_exclude: tuple[str, ...],
    designation_type_include: tuple[str, ...],
    designation_type_exclude: tuple[str, ...],
    target_crs: int,
) -> gpd.GeoSeries:
    LOG.info(
        "loading WDPA geometries (filter=%s, buffer_m=%.0f, polygons=%d, points=%d)",
        filter_mode,
        point_buffer_m,
        len(polygon_paths),
        len(point_paths),
    )
    geometries: list[gpd.GeoSeries] = []
    for path in polygon_paths:
        path_obj = Path(path)
        if not path_obj.exists():
            continue
        LOG.info("reading WDPA polygons: %s", path_obj)
        gdf = gpd.read_file(path_obj)
        gdf = _filter_wdpa(
            gdf,
            filter_mode,
            iucn_include,
            iucn_exclude,
            status_include,
            status_exclude,
            designation_include,
            designation_exclude,
            designation_type_include,
            designation_type_exclude,
        )
        if gdf.empty:
            LOG.info("WDPA polygons empty after filter: %s", path_obj)
            continue
        if gdf.crs is None:
            LOG.warning("WDPA polygons missing CRS: %s", path_obj)
            continue
        LOG.info("WDPA polygons kept: %s (count=%d)", path_obj, len(gdf))
        gdf = gdf.to_crs(target_crs)
        geometries.append(gdf.geometry)

    for path in point_paths:
        path_obj = Path(path)
        if not path_obj.exists():
            continue
        LOG.info("reading WDPA points: %s", path_obj)
        gdf = gpd.read_file(path_obj)
        gdf = _filter_wdpa(
            gdf,
            filter_mode,
            iucn_include,
            iucn_exclude,
            status_include,
            status_exclude,
            designation_include,
            designation_exclude,
            designation_type_include,
            designation_type_exclude,
        )
        if gdf.empty:
            LOG.info("WDPA points empty after filter: %s", path_obj)
            continue
        if gdf.crs is None:
            LOG.warning("WDPA points missing CRS: %s", path_obj)
            continue
        LOG.info("WDPA points kept: %s (count=%d)", path_obj, len(gdf))
        gdf = gdf.to_crs(target_crs)
        if point_buffer_m > 0:
            gdf = gdf.copy()
            gdf["geometry"] = gdf.geometry.buffer(point_buffer_m)
        geometries.append(gdf.geometry)

    if not geometries:
        return gpd.GeoSeries([], crs=target_crs)

    series = gpd.GeoSeries(
        pd.concat(geometries, ignore_index=True), crs=target_crs
    )
    series = _make_valid_geometries(series)
    if series.empty:
        return gpd.GeoSeries([], crs=target_crs)
    LOG.info("WDPA geometries combined (count=%d)", len(series))
    return series


def _load_wdpa_union_cached(
    polygon_paths: tuple[str, ...],
    point_paths: tuple[str, ...],
    filter_mode: str,
    point_buffer_m: float,
    iucn_include: tuple[str, ...],
    iucn_exclude: tuple[str, ...],
    status_include: tuple[str, ...],
    status_exclude: tuple[str, ...],
    designation_include: tuple[str, ...],
    designation_exclude: tuple[str, ...],
    designation_type_include: tuple[str, ...],
    designation_type_exclude: tuple[str, ...],
    target_crs: int,
) -> gpd.GeoSeries:
    series = _load_wdpa_geometries_cached(
        polygon_paths,
        point_paths,
        filter_mode,
        point_buffer_m,
        iucn_include,
        iucn_exclude,
        status_include,
        status_exclude,
        designation_include,
        designation_exclude,
        designation_type_include,
        designation_type_exclude,
        target_crs,
    )
    if series.empty:
        return gpd.GeoSeries([], crs=target_crs)
    geometry = _union_geometry(series)
    geometry_series = gpd.GeoSeries([geometry], crs=target_crs)
    geometry_series = _make_valid_geometries(geometry_series)
    if geometry_series.empty:
        return gpd.GeoSeries([], crs=target_crs)
    geometry = _union_geometry(geometry_series)
    return gpd.GeoSeries([geometry], crs=target_crs)


def _wdpa_mask(
    cutout: atlite.Cutout,
    polygon_paths: list[Path],
    point_paths: list[Path],
    filter_mode: str,
    point_buffer_m: float,
    res: float,
    iucn_include: tuple[str, ...],
    iucn_exclude: tuple[str, ...],
    status_include: tuple[str, ...],
    status_exclude: tuple[str, ...],
    designation_include: tuple[str, ...],
    designation_exclude: tuple[str, ...],
    designation_type_include: tuple[str, ...],
    designation_type_exclude: tuple[str, ...],
) -> xr.DataArray:
    LOG.info(
        "rasterizing WDPA mask (filter=%s, res=%.0f, buffer_m=%.0f)",
        filter_mode,
        res,
        point_buffer_m,
    )
    geometry = _load_wdpa_geometries_cached(
        tuple(str(p) for p in polygon_paths),
        tuple(str(p) for p in point_paths),
        filter_mode,
        point_buffer_m,
        iucn_include,
        iucn_exclude,
        status_include,
        status_exclude,
        designation_include,
        designation_exclude,
        designation_type_include,
        designation_type_exclude,
        3035,
    )
    if geometry.empty:
        return xr.DataArray(
            np.ones(cutout.shape, dtype="float32"),
            coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
            dims=("y", "x"),
        )
    excluder = atlite.ExclusionContainer(crs=3035, res=res)
    mask, _ = gis.shape_availability_reprojected(
        geometry, excluder, cutout.transform_r, cutout.crs, cutout.shape
    )
    mask = mask[::-1, :]
    da = xr.DataArray(
        mask.astype("float32"),
        coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
        dims=("y", "x"),
    )
    LOG.info("WDPA mask rasterized")
    return (1.0 - da).clip(0.0, 1.0)


def _mask_from_excluder(
    cutout: atlite.Cutout,
    shapes_path: Path | list[Path] | tuple[Path, ...],
    excluder: atlite.ExclusionContainer,
) -> xr.DataArray:
    shapes_key = _normalize_shapes_paths(shapes_path)
    geometry = _load_union_geometry_cached(shapes_key, int(excluder.crs))
    if geometry.empty:
        return xr.DataArray(
            np.zeros(cutout.shape, dtype="float32"),
            coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
            dims=("y", "x"),
        )
    mask, _ = gis.shape_availability_reprojected(
        geometry, excluder, cutout.transform_r, cutout.crs, cutout.shape
    )
    mask = mask[::-1, :]
    return xr.DataArray(
        mask.astype("float32"),
        coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
        dims=("y", "x"),
    ).clip(0.0, 1.0)


def _availability_by_regions_excluder(
    cutout: atlite.Cutout, regions: gpd.GeoDataFrame, excluder: atlite.ExclusionContainer
) -> xr.DataArray:
    availability = cutout.availabilitymatrix(regions, excluder)
    return availability.clip(0.0, 1.0)


def _build_onshore_excluder(
    res: float,
    CORINE_LANDCOVER_PATH: Path,
    landcover_cfg: LandcoverConfig,
    LUISA_LANDCOVER_PATH: Path,
    luisa_cfg: LandcoverConfig,
    use_corine: bool,
    use_luisa: bool,
    exclude_natura: bool,
    natura_path: Path,
    urban_distance_onwind: float | None,
    luisa_legend_path: Path,
) -> atlite.ExclusionContainer:
    LOG.info(
        "building onshore excluder (res=%.0f, corine=%s, luisa=%s, natura=%s, urban_distance_m=%s)",
        res,
        use_corine,
        use_luisa,
        exclude_natura,
        "" if urban_distance_onwind is None else str(urban_distance_onwind),
    )
    excluder = atlite.ExclusionContainer(crs=3035, res=res)
    if exclude_natura:
        LOG.info("adding natura mask to onshore excluder: %s", natura_path)
        nodata = _raster_nodata(natura_path, 0)
        excluder.add_raster(
            natura_path,
            codes=_mask_gt_zero,
            nodata=nodata,
            allow_no_overlap=True,
        )
    if use_corine and landcover_cfg.active:
        nodata = _raster_nodata(CORINE_LANDCOVER_PATH, -128)
        codes, invert = _excluder_codes_from_config(
            landcover_cfg, CLC_NODATA_CLASSES, nodata
        )
        if codes:
            LOG.info(
                "adding CORINE landcover to onshore excluder (mode=%s, codes=%d)",
                landcover_cfg.mode,
                len(codes),
            )
            excluder.add_raster(
                CORINE_LANDCOVER_PATH,
                codes=codes,
                invert=invert,
                nodata=nodata,
                allow_no_overlap=True,
            )
    if use_luisa and luisa_cfg.active:
        nodata = _raster_nodata(LUISA_LANDCOVER_PATH, 0)
        codes, invert = _excluder_codes_from_config(
            luisa_cfg, LUISA_NODATA_CODES, nodata
        )
        if codes:
            LOG.info(
                "adding LUISA landcover to onshore excluder (mode=%s, codes=%d)",
                luisa_cfg.mode,
                len(codes),
            )
            excluder.add_raster(
                LUISA_LANDCOVER_PATH,
                codes=codes,
                invert=invert,
                nodata=nodata,
                allow_no_overlap=True,
            )
    if urban_distance_onwind is not None and urban_distance_onwind > 0:
        if use_corine:
            urban_classes, _, _ = _normalize_landcover_codes(
                PYPSA_ONWIND_URBAN_CLC_CODES
            )
            if urban_classes:
                LOG.info(
                    "adding CORINE urban distance buffer to onshore excluder (codes=%d, buffer_m=%.0f)",
                    len(urban_classes),
                    urban_distance_onwind,
                )
                nodata = _raster_nodata(CORINE_LANDCOVER_PATH, -128)
                excluder.add_raster(
                    CORINE_LANDCOVER_PATH,
                    codes=sorted(urban_classes),
                    buffer=urban_distance_onwind,
                    nodata=nodata,
                    allow_no_overlap=True,
                )
        if use_luisa:
            urban_codes, _, _ = _normalize_luisa_codes(
                PYPSA_LUISA_URBAN_CODES, luisa_legend_path
            )
            if urban_codes:
                LOG.info(
                    "adding LUISA urban distance buffer to onshore excluder (codes=%d, buffer_m=%.0f)",
                    len(urban_codes),
                    urban_distance_onwind,
                )
                nodata = _raster_nodata(LUISA_LANDCOVER_PATH, 0)
                excluder.add_raster(
                    LUISA_LANDCOVER_PATH,
                    codes=sorted(urban_codes),
                    buffer=urban_distance_onwind,
                    nodata=nodata,
                    allow_no_overlap=True,
                )
    return excluder


def _build_offshore_excluder(
    res: float,
    exclude_natura: bool,
    natura_path: Path,
    exclude_shipdensity: bool,
    shipdensity_path: Path,
    shipdensity_threshold: float,
    gebco_path: Path,
    min_depth: float | None,
    max_depth: float | None,
    min_shore_distance: float | None,
    max_shore_distance: float | None,
    country_shapes_path: Path,
    exclude_wdpa_offshore: bool,
    wdpa_polygons: list[Path],
    wdpa_points: list[Path],
    wdpa_filter: str,
    wdpa_point_buffer_m: float,
    wdpa_iucn_include: tuple[str, ...],
    wdpa_iucn_exclude: tuple[str, ...],
    wdpa_status_include: tuple[str, ...],
    wdpa_status_exclude: tuple[str, ...],
    wdpa_designation_include: tuple[str, ...],
    wdpa_designation_exclude: tuple[str, ...],
    wdpa_designation_type_include: tuple[str, ...],
    wdpa_designation_type_exclude: tuple[str, ...],
) -> atlite.ExclusionContainer:
    LOG.info(
        "building offshore excluder (res=%.0f, natura=%s, shipdensity=%s, bathymetry=%s, shore=%s, wdpa=%s)",
        res,
        exclude_natura,
        exclude_shipdensity,
        min_depth is not None or max_depth is not None,
        min_shore_distance is not None or max_shore_distance is not None,
        exclude_wdpa_offshore,
    )
    excluder = atlite.ExclusionContainer(crs=3035, res=res)
    if exclude_natura:
        LOG.info("adding natura mask to offshore excluder: %s", natura_path)
        nodata = _raster_nodata(natura_path, 0)
        excluder.add_raster(
            natura_path,
            codes=_mask_gt_zero,
            nodata=nodata,
            allow_no_overlap=True,
        )
    if exclude_shipdensity:
        LOG.info(
            "adding shipdensity to offshore excluder (threshold=%s): %s",
            shipdensity_threshold,
            shipdensity_path,
        )
        excluder.add_raster(
            shipdensity_path,
            codes=partial(np.less, shipdensity_threshold),
            nodata=-1,
            allow_no_overlap=True,
            crs="EPSG:4326",
        )
    if min_depth is not None or max_depth is not None:
        LOG.info(
            "adding bathymetry to offshore excluder (min_depth=%s, max_depth=%s): %s",
            "" if min_depth is None else str(min_depth),
            "" if max_depth is None else str(max_depth),
            gebco_path,
        )
        excluder.add_raster(
            gebco_path,
            codes=partial(
                _bathymetry_exclusion, min_depth=min_depth, max_depth=max_depth
            ),
            nodata=-32768,
            allow_no_overlap=True,
            crs="EPSG:4326",
        )
    if min_shore_distance is not None or max_shore_distance is not None:
        LOG.info(
            "adding shore distance to offshore excluder (min=%s, max=%s)",
            "" if min_shore_distance is None else str(min_shore_distance),
            "" if max_shore_distance is None else str(max_shore_distance),
        )
        geometry = _load_union_geometry_cached(_normalize_shapes_paths(country_shapes_path), 3035)
        if not geometry.empty:
            if min_shore_distance is not None:
                excluder.add_geometry(geometry, buffer=min_shore_distance, invert=False)
            if max_shore_distance is not None:
                excluder.add_geometry(geometry, buffer=max_shore_distance, invert=True)
    if exclude_wdpa_offshore:
        LOG.info("adding WDPA geometries to offshore excluder (filter=%s)", wdpa_filter)
        wdpa_geometry = _load_wdpa_geometries_cached(
            tuple(str(p) for p in wdpa_polygons),
            tuple(str(p) for p in wdpa_points),
            wdpa_filter,
            wdpa_point_buffer_m,
            wdpa_iucn_include,
            wdpa_iucn_exclude,
            wdpa_status_include,
            wdpa_status_exclude,
            wdpa_designation_include,
            wdpa_designation_exclude,
            wdpa_designation_type_include,
            wdpa_designation_type_exclude,
            3035,
        )
        if not wdpa_geometry.empty:
            LOG.info("WDPA geometries loaded for offshore (count=%d)", len(wdpa_geometry))
            excluder.add_geometry(wdpa_geometry, invert=False)
    return excluder


def _resource_classify(
    cf_mean: xr.DataArray, availability: xr.DataArray, nbins: int
) -> tuple[xr.DataArray, xr.DataArray]:
    cf_masked = cf_mean.where(availability > 0)
    cf_min = float(cf_masked.min())
    cf_max = float(cf_masked.max())
    if not np.isfinite(cf_min) or not np.isfinite(cf_max):
        class_index = xr.full_like(availability, -1, dtype="int16")
        bins = xr.DataArray(np.full(nbins + 1, np.nan, dtype="float32"), dims=("bin_edge",))
        return class_index, bins

    if cf_min == cf_max:
        class_index = xr.where(availability > 0, 0, -1).astype("int16")
        bins = xr.DataArray(
            np.linspace(cf_min, cf_max, nbins + 1, dtype="float32"), dims=("bin_edge",)
        )
        return class_index, bins

    edges = np.linspace(cf_min, cf_max, nbins + 1, dtype="float32")
    boundaries = edges[1:-1]
    cf_values = np.asarray(cf_masked.values)
    class_values = np.digitize(cf_values, boundaries).astype("int16")
    class_index = xr.DataArray(
        class_values,
        coords=cf_masked.coords,
        dims=cf_masked.dims,
    )
    class_index = class_index.where(availability > 0, other=-1)
    bins = xr.DataArray(edges, dims=("bin_edge",))
    return class_index, bins


def _resource_classify_per_region(
    cf_mean: xr.DataArray, availability: xr.DataArray, nbins: int
) -> tuple[xr.DataArray, xr.DataArray]:
    region_mask = availability > 0
    cf_by_bus = cf_mean.broadcast_like(availability).where(region_mask)
    cf_min = cf_by_bus.min(dim=("y", "x"), skipna=True)
    cf_max = cf_by_bus.max(dim=("y", "x"), skipna=True)

    ratios = xr.DataArray(np.linspace(0, 1, nbins + 1, dtype="float32"), dims=("bin_edge",))
    bins = cf_min + (cf_max - cf_min) * ratios

    if nbins == 1:
        class_index = xr.where(region_mask, 0, -1).astype("int16")
        return class_index, bins

    boundaries = bins.isel(bin_edge=slice(1, -1))
    class_index = xr.apply_ufunc(
        np.digitize,
        cf_by_bus,
        boundaries,
        input_core_dims=[["y", "x"], ["bin_edge"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,
        dask="allowed",
        output_dtypes=[np.int16],
    )
    class_index = class_index.where(region_mask, other=-1).astype("int16")

    has_region = region_mask.any(dim=("y", "x"))
    flat = (cf_max <= cf_min) | (~np.isfinite(cf_min)) | (~np.isfinite(cf_max))
    flat = flat & has_region
    flat_mask = flat.broadcast_like(region_mask)
    class_index = xr.where(flat_mask & region_mask, 0, class_index)
    return class_index, bins


def _p_nom_max_bus_bin(
    availability: xr.DataArray,
    class_index: xr.DataArray,
    area_km2: xr.DataArray,
    capacity_per_sqkm: float,
    nbins: int,
) -> xr.DataArray:
    bin_ids = xr.DataArray(np.arange(nbins), dims=("bin",))
    class_mask = class_index == bin_ids
    p_nom = (availability * class_mask * area_km2).sum(("y", "x")) * capacity_per_sqkm
    if "bus" in p_nom.dims and "bin" in p_nom.dims:
        p_nom = p_nom.transpose("bus", "bin")
    return p_nom.astype("float32")


def _write_availability(ds: xr.Dataset, out_path: Path) -> None:
    for name in ds.data_vars:
        if ds[name].dtype.kind == "f":
            ds[name] = ds[name].astype("float32")
        elif ds[name].dtype.kind in {"i", "u"}:
            ds[name] = ds[name].astype("int16")
    encoding = {}
    for name, da in ds.data_vars.items():
        if da.dtype.kind == "f":
            encoding[name] = {"zlib": True, "complevel": 4, "dtype": "float32"}
        else:
            encoding[name] = {"zlib": True, "complevel": 4, "dtype": "int16"}
    LOG.info("writing %s", out_path)
    ds.to_netcdf(out_path, engine="netcdf4", encoding=encoding)


def _load_hydro_static(plants_path: Path, buses_path: Path) -> tuple[xr.DataArray, xr.DataArray]:
    plants = pd.read_csv(plants_path, sep=";")
    buses = pd.read_csv(buses_path, sep=";")

    hydro = plants[plants["Fueltype"].str.contains("Hydro", case=False, na=False)].copy()
    hydro["hydro_type"] = hydro["Technology"].map(HYDRO_TECH_TO_TYPE)
    hydro = hydro[hydro["hydro_type"].notna()]

    p_inst_bus = (
        hydro.groupby(["bus_id", "hydro_type"])["Capacity"]
        .sum()
        .unstack(fill_value=0.0)
    )
    for hydro_type in HYDRO_TYPES:
        if hydro_type not in p_inst_bus.columns:
            p_inst_bus[hydro_type] = 0.0
    p_inst_bus = p_inst_bus[HYDRO_TYPES]

    bus_ids = p_inst_bus.index.astype(str).rename("bus")
    bus_country = buses.set_index("bus_id")["country"].reindex(bus_ids)

    missing = bus_country.isna()
    if missing.any():
        LOG.warning("missing country for %d buses (examples: %s)", missing.sum(), bus_ids[missing][:5])
        bus_country = bus_country.fillna("XX")

    p_inst = xr.DataArray(
        p_inst_bus.values,
        coords={"bus": bus_ids.values, "hydro_type": HYDRO_TYPES},
        dims=("bus", "hydro_type"),
    )

    bus_country_da = xr.DataArray(
        bus_country.values,
        coords={"bus": bus_ids.values},
        dims=("bus",),
        name="country",
    )

    return p_inst, bus_country_da


def _normalize_shape_labels(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().replace("", pd.NA)


def _select_country_key_column(shapes: gpd.GeoDataFrame, country_shapes_path: Path) -> str:
    for column in ("country_code", "iso2", "iso2_norm", "CNTR_ID", "name", "NAME_ENGL"):
        if column not in shapes.columns:
            continue
        values = _normalize_shape_labels(pd.Series(shapes[column]))
        non_null = values.dropna().astype(str)
        non_empty = non_null[non_null != ""]
        if not non_empty.empty:
            if len(non_empty) < len(values):
                LOG.info(
                    "using hydro country key column %s from %s (%d/%d non-empty rows)",
                    column,
                    country_shapes_path,
                    len(non_empty),
                    len(values),
                )
            return column
    raise ValueError(
        f"no usable hydro country key column found in {country_shapes_path}; "
        "expected one of country_code, iso2, iso2_norm, CNTR_ID, name, NAME_ENGL"
    )


def _parse_member_countries(value) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def _build_hydro_aggregation_shapes(
    country_shapes_path: Path,
    hydro_reductions_path: Path | None,
    bus_country: xr.DataArray | None,
) -> tuple[gpd.GeoDataFrame, dict[str, tuple[str, ...]]]:
    shapes = gpd.read_file(country_shapes_path)
    country_key = _select_country_key_column(shapes, country_shapes_path)
    shape_labels = _normalize_shape_labels(pd.Series(shapes[country_key]))
    keep = shape_labels.notna()
    shapes = shapes.loc[keep].copy()
    if shapes.empty:
        raise ValueError(f"no hydro country shapes with valid labels found in {country_shapes_path}")
    shape_labels = shape_labels.loc[keep].astype(str)
    shapes["country"] = shape_labels.values
    if "NAME_ENGL" in shapes.columns:
        country_label = _normalize_shape_labels(pd.Series(shapes["NAME_ENGL"])).fillna(shape_labels)
    else:
        country_label = shape_labels
    shapes["country_label"] = country_label.astype(str).values
    shapes["member_countries"] = shape_labels.values

    reduction_members: dict[str, tuple[str, ...]] = {}
    active_codes = set()
    if bus_country is not None:
        active_codes = {str(value) for value in bus_country.values if pd.notna(value)}
    if hydro_reductions_path is not None and hydro_reductions_path.exists() and active_codes:
        reductions = gpd.read_file(hydro_reductions_path)
        if not reductions.empty:
            reduction_key = _select_country_key_column(reductions, hydro_reductions_path)
            reduction_codes = _normalize_shape_labels(pd.Series(reductions[reduction_key]))
            red_keep = reduction_codes.notna()
            reductions = reductions.loc[red_keep].copy()
            reduction_codes = reduction_codes.loc[red_keep].astype(str)
            reductions["country"] = reduction_codes.values
            reductions = reductions[reductions["country"].isin(active_codes)].copy()
            if not reductions.empty:
                if reductions.crs is not None and shapes.crs is not None and reductions.crs != shapes.crs:
                    reductions = reductions.to_crs(shapes.crs)
                reduction_codes = reductions["country"].astype(str)
                reduction_labels = (
                    _normalize_shape_labels(pd.Series(reductions["country_label"]))
                    if "country_label" in reductions.columns
                    else pd.Series(pd.NA, index=reductions.index, dtype="string")
                )
                reductions["country_label"] = reduction_labels.fillna(reduction_codes).astype(str)
                member_series = (
                    reductions["member_countries"]
                    if "member_countries" in reductions.columns
                    else pd.Series("", index=reductions.index, dtype=object)
                )
                reductions["member_countries"] = member_series.fillna("").astype(str)
                for code, members_value in zip(reductions["country"], reductions["member_countries"]):
                    members = _parse_member_countries(members_value)
                    if members:
                        reduction_members[str(code)] = members
                reduced_members = {member for members in reduction_members.values() for member in members}
                if reduced_members:
                    shapes = shapes[~shapes["country"].isin(reduced_members)].copy()
                LOG.info(
                    "using hydro reductions from %s for labels: %s",
                    hydro_reductions_path,
                    ", ".join(sorted(reduction_members)),
                )
                shapes = gpd.GeoDataFrame(
                    pd.concat(
                        [
                            shapes[["country", "country_label", "member_countries", "geometry"]],
                            reductions[["country", "country_label", "member_countries", "geometry"]],
                        ],
                        ignore_index=True,
                    ),
                    geometry="geometry",
                    crs=shapes.crs,
                )

    shapes = shapes.set_index(pd.Index(shapes["country"].astype(str).values, name="country"))
    return shapes[["country_label", "member_countries", "geometry"]], reduction_members


def _apply_country_reductions(
    data: xr.DataArray, reduction_members: dict[str, tuple[str, ...]]
) -> xr.DataArray:
    if not reduction_members:
        return data
    base = data
    member_codes = {member for members in reduction_members.values() for member in members}
    keep_codes = [code for code in base.coords["country"].values if code not in member_codes]
    base = base.reindex(country=keep_codes)
    reduced_arrays: list[xr.DataArray] = []
    for code, members in reduction_members.items():
        existing = [member for member in members if member in set(data.coords["country"].values)]
        if existing:
            reduced = data.reindex(country=existing, fill_value=0.0).sum("country")
        else:
            reduced = data.isel(country=0, drop=True) * 0.0
        reduced_arrays.append(reduced.expand_dims(country=[code]))
    if reduced_arrays:
        base = xr.concat([base, *reduced_arrays], dim="country")
    return base


def _compute_country_runoff(
    cutout: atlite.Cutout,
    country_shapes_path: Path,
    hydro_reductions_path: Path | None,
    bus_country: xr.DataArray | None,
) -> xr.DataArray:
    shapes, _ = _build_hydro_aggregation_shapes(
        country_shapes_path, hydro_reductions_path, bus_country
    )
    runoff = cutout.runoff(shapes=shapes.geometry, shapes_crs=shapes.crs)

    country_dim = [d for d in runoff.dims if d != "time"][0]
    runoff = runoff.rename({country_dim: "country"})
    runoff = runoff.assign_coords(country=("country", shapes.index.to_numpy(dtype=object)))
    runoff = runoff.assign_coords(
        country_label=("country", shapes["country_label"].astype(str).to_numpy()),
        member_countries=("country", shapes["member_countries"].astype(str).to_numpy()),
    )
    country_index = pd.Index(runoff.coords["country"].values)
    if country_index.has_duplicates:
        LOG.warning("duplicate hydro country labels found in %s; aggregating duplicates", country_shapes_path)
        runoff = runoff.groupby("country").sum()
    return runoff


def _load_reference_hydro(reference_path: Path) -> xr.DataArray:
    ds = xr.open_dataset(reference_path)
    if "p_avail_total" in ds:
        ref = ds["p_avail_total"]
    elif "e_avail_total" in ds:
        ref = ds["e_avail_total"]
    else:
        ref = ds.to_array().squeeze()

    if "countries" in ref.coords:
        ref = ref.rename({"countries": "country"})
    if "country" not in ref.coords:
        raise ValueError(f"reference hydro file missing 'country' coordinate: {reference_path}")

    return ref


def _compute_hydro_scale(
    cutout_dir: Path,
    reference_year: int,
    country_shapes_path: Path,
    reference_path: Path,
    hydro_reductions_path: Path | None,
    bus_country: xr.DataArray | None,
) -> xr.DataArray:
    cutout_path = _cutout_path(reference_year, cutout_dir)
    cutout = atlite.Cutout(cutout_path)
    try:
        _, reduction_members = _build_hydro_aggregation_shapes(
            country_shapes_path, hydro_reductions_path, bus_country
        )
        runoff = _compute_country_runoff(
            cutout, country_shapes_path, hydro_reductions_path, bus_country
        )
        ref = _load_reference_hydro(reference_path)
        ref = _apply_country_reductions(ref, reduction_members)

        runoff_sum = runoff.sum("time")
        ref_sum = ref.sum("time")
        runoff_sum, ref_sum = xr.align(runoff_sum, ref_sum, join="outer", fill_value=0.0)
        scale = xr.where(runoff_sum > 0, ref_sum / runoff_sum, 0.0)
        return scale.astype("float32")
    finally:
        cutout.data.close()


def _map_country_data_to_buses(
    data: xr.DataArray, bus_country: xr.DataArray, fill_value: float = 0.0
) -> xr.DataArray:
    if "country" not in data.dims:
        raise ValueError("country-to-bus mapping requires a 'country' dimension")

    source_index = pd.Index(data.coords["country"].values)
    if source_index.has_duplicates:
        LOG.warning("country data has duplicate labels; aggregating duplicates before bus mapping")
        data = data.groupby("country").sum()
        source_index = pd.Index(data.coords["country"].values)

    bus_ids = bus_country.coords["bus"].values
    target_index = pd.Index(bus_country.values)
    indexer = source_index.get_indexer(target_index)
    missing = indexer < 0
    if source_index.empty:
        missing[:] = True
        indexer = np.zeros(len(target_index), dtype=int)
    elif missing.any():
        indexer = indexer.copy()
        indexer[missing] = 0

    mapped = data.isel(
        country=xr.DataArray(indexer, dims=("bus",), coords={"bus": bus_ids})
    )
    if "country" in mapped.coords:
        mapped = mapped.drop_vars("country")
    mapped = mapped.assign_coords(bus=bus_ids)

    if missing.any():
        valid = xr.DataArray(~missing, dims=("bus",), coords={"bus": bus_ids})
        mapped = xr.where(valid, mapped, fill_value)
        missing_labels = pd.Index(target_index[missing]).drop_duplicates().tolist()
        LOG.warning(
            "missing hydro country data for %d buses (country labels: %s)",
            int(missing.sum()),
            ", ".join(str(label) for label in missing_labels[:10]),
        )

    target_dims = tuple("bus" if dim == "country" else dim for dim in data.dims)
    mapped = mapped.transpose(*target_dims)
    return mapped


def _compute_hydro_profiles(
    cutout: atlite.Cutout,
    country_shapes_path: Path,
    p_inst: xr.DataArray,
    bus_country: xr.DataArray,
    hydro_scale: xr.DataArray | None,
    hydro_reductions_path: Path | None,
    hydro_write_bus_profiles: bool,
) -> tuple[xr.Dataset | None, xr.Dataset]:
    runoff = _compute_country_runoff(
        cutout, country_shapes_path, hydro_reductions_path, bus_country
    )
    if hydro_scale is not None:
        hydro_scale = hydro_scale.reindex(country=runoff.country, fill_value=0.0)
        runoff = runoff * hydro_scale
    runoff = runoff.rename("p_avail_total")

    p_inst_total = p_inst.sum("hydro_type")

    p_inst_ror_res = p_inst.sel(hydro_type=["ror", "reservoir"]).sum("hydro_type")
    country_cap = p_inst_ror_res.groupby(bus_country).sum("bus")
    country_cap = country_cap.reindex(country=runoff.country, fill_value=0.0)

    avail_pu_country = xr.where(country_cap > 0, runoff / country_cap, 0.0)
    bus_ds = None
    if hydro_write_bus_profiles:
        avail_pu_bus = _map_country_data_to_buses(
            avail_pu_country, bus_country, fill_value=0.0
        )

        avail_pu = xr.DataArray(
            np.zeros(
                (runoff.sizes["time"], p_inst.sizes["bus"], len(HYDRO_TYPES)),
                dtype="float32",
            ),
            coords={
                "time": runoff.coords["time"],
                "bus": p_inst.coords["bus"].values,
                "hydro_type": HYDRO_TYPES,
            },
            dims=("time", "bus", "hydro_type"),
        )
        avail_pu.loc[dict(hydro_type=["ror", "reservoir"])] = avail_pu_bus.astype("float32")

        p_avail = (p_inst * avail_pu).astype("float32")
        e_avail = p_avail

        bus_ds = xr.Dataset(
            {
                "p_inst": p_inst,
                "p_avail": p_avail,
                "e_avail": e_avail,
                "avail_pu": avail_pu,
                "p_inst_total": p_inst_total,
                "p_avail_total": p_avail.sum("hydro_type"),
                "e_avail_total": e_avail.sum("hydro_type"),
            }
        )
        bus_ds["p_inst"].attrs.update(
            units="MW", description="Installed hydro capacity per bus and technology"
        )
        bus_ds["p_avail"].attrs.update(
            units="MW", description="Allocated hydro availability per bus and technology"
        )
        bus_ds["e_avail"].attrs.update(
            units="MWh", description="Hourly hydro energy equivalent per bus and technology"
        )
        bus_ds["avail_pu"].attrs.update(
            units="-", description="Allocated country-level hydro inflow per unit of installed capacity"
        )
        bus_ds["p_inst_total"].attrs.update(
            units="MW", description="Total installed hydro capacity per bus"
        )
        bus_ds["p_avail_total"].attrs.update(
            units="MW", description="Total allocated natural hydro inflow per bus"
        )
        bus_ds["e_avail_total"].attrs.update(
            units="MWh", description="Total hourly hydro energy equivalent per bus"
        )

    p_inst_country = p_inst.groupby(bus_country).sum("bus")
    p_inst_country = p_inst_country.reindex(country=runoff.country, fill_value=0.0)

    avail_pu_country_by_type = xr.DataArray(
        np.zeros((runoff.sizes["time"], runoff.sizes["country"], len(HYDRO_TYPES)), dtype="float32"),
        coords={"time": runoff.coords["time"], "country": runoff.coords["country"], "hydro_type": HYDRO_TYPES},
        dims=("time", "country", "hydro_type"),
    )
    avail_pu_country_by_type.loc[dict(hydro_type=["ror", "reservoir"])] = avail_pu_country.astype(
        "float32"
    )

    p_avail_country = (p_inst_country * avail_pu_country_by_type).astype("float32")
    e_avail_country = p_avail_country

    country_ds = xr.Dataset(
        {
            "p_inst": p_inst_country,
            "p_avail": p_avail_country,
            "e_avail": e_avail_country,
            "p_avail_total": runoff,
            "e_avail_total": runoff,
        }
    )
    country_ds["p_inst"].attrs.update(
        units="MW", description="Installed hydro capacity per country and technology"
    )
    country_ds["p_avail"].attrs.update(
        units="MW", description="Allocated hydro availability per country and technology"
    )
    country_ds["e_avail"].attrs.update(
        units="MWh", description="Hourly hydro energy equivalent per country and technology"
    )
    country_ds["p_avail_total"].attrs.update(
        units="MW", description="Total natural hydro inflow per country"
    )
    country_ds["e_avail_total"].attrs.update(
        units="MWh", description="Total hourly hydro energy equivalent per country"
    )
    for coord_name in ("country_label", "member_countries"):
        if coord_name in runoff.coords:
            country_ds = country_ds.assign_coords(
                {coord_name: ("country", runoff.coords[coord_name].values)}
            )

    return bus_ds, country_ds


def compute_year(
    year: int,
    cutout_dir: Path,
    out_dir: Path,
    do_pv: bool,
    do_onwind: bool,
    do_offwind: bool,
    do_hydro: bool,
    p_inst: xr.DataArray | None,
    bus_country: xr.DataArray | None,
    hydro_scale: xr.DataArray | None,
    hydro_reference: Path,
    hydro_reference_year: int,
    hydro_reductions_path: Path | None,
    hydro_write_bus_profiles: bool,
    mask_onshore: bool,
    mask_offshore: bool,
    onshore_mask_shapes: list[Path],
    offshore_mask_shapes: list[Path],
    onshore_regions_path: Path,
    offshore_regions_path: Path,
    country_shapes_path: Path,
    exclude_natura: bool,
    natura_path: Path,
    exclude_shipdensity: bool,
    shipdensity_path: Path,
    shipdensity_threshold: float,
    exclude_wdpa_offshore: bool,
    wdpa_polygons: list[Path],
    wdpa_points: list[Path],
    wdpa_filter: str,
    wdpa_point_buffer_m: float,
    wdpa_iucn_include: tuple[str, ...],
    wdpa_iucn_exclude: tuple[str, ...],
    wdpa_status_include: tuple[str, ...],
    wdpa_status_exclude: tuple[str, ...],
    wdpa_designation_include: tuple[str, ...],
    wdpa_designation_exclude: tuple[str, ...],
    wdpa_designation_type_include: tuple[str, ...],
    wdpa_designation_type_exclude: tuple[str, ...],
    exclude_wdpa_onshore: bool,
    wdpa_onshore_filter: str,
    wdpa_onshore_iucn_include: tuple[str, ...],
    wdpa_onshore_iucn_exclude: tuple[str, ...],
    wdpa_onshore_status_include: tuple[str, ...],
    wdpa_onshore_status_exclude: tuple[str, ...],
    wdpa_onshore_designation_include: tuple[str, ...],
    wdpa_onshore_designation_exclude: tuple[str, ...],
    wdpa_onshore_designation_type_include: tuple[str, ...],
    wdpa_onshore_designation_type_exclude: tuple[str, ...],
    landuse_dataset: str,
    landuse_fusion: str,
    CORINE_LANDCOVER_PATH: Path,
    landcover_pv: LandcoverConfig,
    landcover_onwind: LandcoverConfig,
    LUISA_LANDCOVER_PATH: Path,
    luisa_legend_path: Path,
    luisa_pv: LandcoverConfig,
    luisa_onwind: LandcoverConfig,
    gebco_path: Path,
    min_depth: float | None,
    max_depth: float | None,
    min_shore_distance: float | None,
    max_shore_distance: float | None,
    urban_distance_onwind: float | None,
    use_excluder_mask: bool,
    excluder_res_onshore: float,
    excluder_res_offshore: float,
    excluder_cache: dict[tuple, atlite.ExclusionContainer],
    resource_classes: int,
    resource_class_year: int,
    capacity_per_sqkm_pv: float | None,
    capacity_per_sqkm_onwind: float | None,
    capacity_per_sqkm_offwind: float | None,
    clip_p_max_pu: float | None,
    availability_mode: str,
    resource_class_mode: str,
    availability_state: dict[str, bool],
    mask_cache: dict[object, xr.DataArray | None],
    overwrite: bool,
    masks_out_only: bool,
) -> None:
    cutout_path = _cutout_path(year, cutout_dir)
    if not cutout_path.exists():
        LOG.warning("missing cutout: %s (skipping)", cutout_path)
        return

    cutout = atlite.Cutout(cutout_path)
    use_corine = landuse_dataset in {"corine", "both"}
    use_luisa = landuse_dataset in {"luisa", "both"}
    use_excluder_onshore = use_excluder_mask and mask_onshore and (do_pv or do_onwind)
    use_excluder_offshore = use_excluder_mask and mask_offshore and do_offwind
    excluder_onshore_pv = None
    excluder_onshore_onwind = None
    excluder_offshore = None
    mask_onshore_base = None
    mask_onshore_pv = None
    mask_onshore_onwind = None
    mask_offshore_da = None
    bathymetry_mask_da = None
    shore_distance_mask_da = None
    wdpa_mask_da = None
    wdpa_onshore_mask = None
    urban_distance_codes_corine = ""
    urban_distance_codes_luisa = ""
    corine_valid = None
    luisa_valid = None
    if (
        mask_onshore
        and (do_pv or do_onwind)
        and
        landuse_dataset == "both"
        and use_corine
        and use_luisa
    ):
        if landuse_fusion == "prefer-corine":
            if mask_cache.get("corine_valid") is None:
                mask_cache["corine_valid"] = _landcover_valid_mask(
                    cutout, CORINE_LANDCOVER_PATH
                )
            corine_valid = mask_cache["corine_valid"]
        elif landuse_fusion == "prefer-luisa":
            if mask_cache.get("luisa_valid") is None:
                mask_cache["luisa_valid"] = _luisa_valid_mask(
                    cutout, LUISA_LANDCOVER_PATH
                )
            luisa_valid = mask_cache["luisa_valid"]

    if mask_onshore and (do_pv or do_onwind):
        if mask_cache.get("onshore_base") is None:
            mask_cache["onshore_base"] = _build_mask(
                cutout,
                onshore_mask_shapes,
                exclude_natura,
                natura_path,
                False,
                shipdensity_path,
                shipdensity_threshold,
                None,
                None,
                None,
            )
        mask_onshore_base = mask_cache["onshore_base"]
        if exclude_wdpa_onshore:
            wdpa_key = (
                "wdpa_onshore",
                excluder_res_onshore,
                ";".join(str(p) for p in wdpa_polygons),
                ";".join(str(p) for p in wdpa_points),
                wdpa_onshore_filter,
                wdpa_point_buffer_m,
                wdpa_onshore_iucn_include,
                wdpa_onshore_iucn_exclude,
                wdpa_onshore_status_include,
                wdpa_onshore_status_exclude,
                wdpa_onshore_designation_include,
                wdpa_onshore_designation_exclude,
                wdpa_onshore_designation_type_include,
                wdpa_onshore_designation_type_exclude,
            )
            if mask_cache.get(wdpa_key) is None:
                mask_cache[wdpa_key] = _wdpa_mask(
                    cutout,
                    wdpa_polygons,
                    wdpa_points,
                    wdpa_onshore_filter,
                    wdpa_point_buffer_m,
                    excluder_res_onshore,
                    wdpa_onshore_iucn_include,
                    wdpa_onshore_iucn_exclude,
                    wdpa_onshore_status_include,
                    wdpa_onshore_status_exclude,
                    wdpa_onshore_designation_include,
                    wdpa_onshore_designation_exclude,
                    wdpa_onshore_designation_type_include,
                    wdpa_onshore_designation_type_exclude,
                )
            wdpa_onshore_mask = mask_cache[wdpa_key]

        if (
            use_excluder_onshore
            and landuse_dataset == "both"
            and landuse_fusion in {"prefer-corine", "prefer-luisa"}
        ):
            empty_cfg = LandcoverConfig(None, None, "", "", "")

            if do_pv:
                corine_mask = None
                luisa_mask = None
                if use_corine and landcover_pv.active:
                    excluder_key = (
                        "landcover_corine_pv",
                        excluder_res_onshore,
                        landcover_pv.mode,
                        landcover_pv.classes_mapped,
                    )
                    excluder = excluder_cache.get(excluder_key)
                    if excluder is None:
                        excluder = _build_onshore_excluder(
                            res=excluder_res_onshore,
                            CORINE_LANDCOVER_PATH=CORINE_LANDCOVER_PATH,
                            landcover_cfg=landcover_pv,
                            LUISA_LANDCOVER_PATH=LUISA_LANDCOVER_PATH,
                            luisa_cfg=empty_cfg,
                            use_corine=True,
                            use_luisa=False,
                            exclude_natura=False,
                            natura_path=natura_path,
                            urban_distance_onwind=None,
                            luisa_legend_path=luisa_legend_path,
                        )
                        excluder_cache[excluder_key] = excluder
                    mask_key = ("mask",) + excluder_key
                    if mask_cache.get(mask_key) is None:
                        mask_cache[mask_key] = _mask_from_excluder(
                            cutout, onshore_mask_shapes, excluder
                        )
                    corine_mask = mask_cache[mask_key]
                if use_luisa and luisa_pv.active:
                    excluder_key = (
                        "landcover_luisa_pv",
                        excluder_res_onshore,
                        luisa_pv.mode,
                        luisa_pv.classes_mapped,
                    )
                    excluder = excluder_cache.get(excluder_key)
                    if excluder is None:
                        excluder = _build_onshore_excluder(
                            res=excluder_res_onshore,
                            CORINE_LANDCOVER_PATH=CORINE_LANDCOVER_PATH,
                            landcover_cfg=empty_cfg,
                            LUISA_LANDCOVER_PATH=LUISA_LANDCOVER_PATH,
                            luisa_cfg=luisa_pv,
                            use_corine=False,
                            use_luisa=True,
                            exclude_natura=False,
                            natura_path=natura_path,
                            urban_distance_onwind=None,
                            luisa_legend_path=luisa_legend_path,
                        )
                        excluder_cache[excluder_key] = excluder
                    mask_key = ("mask",) + excluder_key
                    if mask_cache.get(mask_key) is None:
                        mask_cache[mask_key] = _mask_from_excluder(
                            cutout, onshore_mask_shapes, excluder
                        )
                    luisa_mask = mask_cache[mask_key]

                landuse_mask = _combine_landuse_masks(
                    landuse_dataset,
                    landuse_fusion,
                    corine_mask,
                    luisa_mask,
                    corine_valid,
                    luisa_valid,
                )
                mask_onshore_pv = mask_onshore_base
                if landuse_mask is not None:
                    mask_onshore_pv = mask_onshore_pv * landuse_mask

            if do_onwind:
                corine_mask = None
                luisa_mask = None
                if use_corine and landcover_onwind.active:
                    excluder_key = (
                        "landcover_corine_onwind",
                        excluder_res_onshore,
                        landcover_onwind.mode,
                        landcover_onwind.classes_mapped,
                    )
                    excluder = excluder_cache.get(excluder_key)
                    if excluder is None:
                        excluder = _build_onshore_excluder(
                            res=excluder_res_onshore,
                            CORINE_LANDCOVER_PATH=CORINE_LANDCOVER_PATH,
                            landcover_cfg=landcover_onwind,
                            LUISA_LANDCOVER_PATH=LUISA_LANDCOVER_PATH,
                            luisa_cfg=empty_cfg,
                            use_corine=True,
                            use_luisa=False,
                            exclude_natura=False,
                            natura_path=natura_path,
                            urban_distance_onwind=None,
                            luisa_legend_path=luisa_legend_path,
                        )
                        excluder_cache[excluder_key] = excluder
                    mask_key = ("mask",) + excluder_key
                    if mask_cache.get(mask_key) is None:
                        mask_cache[mask_key] = _mask_from_excluder(
                            cutout, onshore_mask_shapes, excluder
                        )
                    corine_mask = mask_cache[mask_key]
                if use_luisa and luisa_onwind.active:
                    excluder_key = (
                        "landcover_luisa_onwind",
                        excluder_res_onshore,
                        luisa_onwind.mode,
                        luisa_onwind.classes_mapped,
                    )
                    excluder = excluder_cache.get(excluder_key)
                    if excluder is None:
                        excluder = _build_onshore_excluder(
                            res=excluder_res_onshore,
                            CORINE_LANDCOVER_PATH=CORINE_LANDCOVER_PATH,
                            landcover_cfg=empty_cfg,
                            LUISA_LANDCOVER_PATH=LUISA_LANDCOVER_PATH,
                            luisa_cfg=luisa_onwind,
                            use_corine=False,
                            use_luisa=True,
                            exclude_natura=False,
                            natura_path=natura_path,
                            urban_distance_onwind=None,
                            luisa_legend_path=luisa_legend_path,
                        )
                        excluder_cache[excluder_key] = excluder
                    mask_key = ("mask",) + excluder_key
                    if mask_cache.get(mask_key) is None:
                        mask_cache[mask_key] = _mask_from_excluder(
                            cutout, onshore_mask_shapes, excluder
                        )
                    luisa_mask = mask_cache[mask_key]

                landuse_mask = _combine_landuse_masks(
                    landuse_dataset,
                    landuse_fusion,
                    corine_mask,
                    luisa_mask,
                    corine_valid,
                    luisa_valid,
                )
                mask_onshore_onwind = mask_onshore_base
                if landuse_mask is not None:
                    mask_onshore_onwind = mask_onshore_onwind * landuse_mask

                if urban_distance_onwind is not None and urban_distance_onwind > 0:
                    if use_corine:
                        key = f"urban_distance_corine:{urban_distance_onwind}:res={excluder_res_onshore}"
                        if mask_cache.get(key) is None:
                            mask_cache[key] = _urban_distance_mask(
                                cutout,
                                onshore_mask_shapes,
                                CORINE_LANDCOVER_PATH,
                                PYPSA_ONWIND_URBAN_CLC_CODES,
                                urban_distance_onwind,
                                res=excluder_res_onshore,
                                normalize_fn=_normalize_landcover_codes,
                            )
                        mask_onshore_onwind = mask_onshore_onwind * mask_cache[key]
                        urban_distance_codes_corine = _format_class_list(PYPSA_ONWIND_URBAN_CLC_CODES)
                    if use_luisa:
                        key = f"urban_distance_luisa:{urban_distance_onwind}:res={excluder_res_onshore}"
                        if mask_cache.get(key) is None:
                            mask_cache[key] = _urban_distance_mask(
                                cutout,
                                onshore_mask_shapes,
                                LUISA_LANDCOVER_PATH,
                                PYPSA_LUISA_URBAN_CODES,
                                urban_distance_onwind,
                                res=excluder_res_onshore,
                                normalize_fn=_normalize_luisa_codes,
                                legend_path=luisa_legend_path,
                                nodata_default=0,
                            )
                        mask_onshore_onwind = mask_onshore_onwind * mask_cache[key]
                        urban_distance_codes_luisa = _format_class_list(PYPSA_LUISA_URBAN_CODES)

        elif use_excluder_onshore:
            if do_pv:
                excluder_key = (
                    "onshore_pv",
                    excluder_res_onshore,
                    str(CORINE_LANDCOVER_PATH) if use_corine else "",
                    landcover_pv.mode,
                    landcover_pv.classes_mapped,
                    str(LUISA_LANDCOVER_PATH) if use_luisa else "",
                    luisa_pv.mode,
                    luisa_pv.classes_mapped,
                    exclude_natura,
                    str(natura_path) if exclude_natura else "",
                )
                excluder_onshore_pv = excluder_cache.get(excluder_key)
                if excluder_onshore_pv is None:
                    excluder_onshore_pv = _build_onshore_excluder(
                        res=excluder_res_onshore,
                        CORINE_LANDCOVER_PATH=CORINE_LANDCOVER_PATH,
                        landcover_cfg=landcover_pv,
                        LUISA_LANDCOVER_PATH=LUISA_LANDCOVER_PATH,
                        luisa_cfg=luisa_pv,
                        use_corine=use_corine,
                        use_luisa=use_luisa,
                        exclude_natura=exclude_natura,
                        natura_path=natura_path,
                        urban_distance_onwind=None,
                        luisa_legend_path=luisa_legend_path,
                    )
                    excluder_cache[excluder_key] = excluder_onshore_pv
                mask_key = ("mask",) + excluder_key
                if mask_cache.get(mask_key) is None:
                    mask_cache[mask_key] = _mask_from_excluder(
                        cutout, onshore_mask_shapes, excluder_onshore_pv
                    )
                mask_onshore_pv = mask_cache[mask_key]

            if do_onwind:
                excluder_key = (
                    "onshore_onwind",
                    excluder_res_onshore,
                    str(CORINE_LANDCOVER_PATH) if use_corine else "",
                    landcover_onwind.mode,
                    landcover_onwind.classes_mapped,
                    str(LUISA_LANDCOVER_PATH) if use_luisa else "",
                    luisa_onwind.mode,
                    luisa_onwind.classes_mapped,
                    exclude_natura,
                    str(natura_path) if exclude_natura else "",
                    urban_distance_onwind,
                )
                excluder_onshore_onwind = excluder_cache.get(excluder_key)
                if excluder_onshore_onwind is None:
                    excluder_onshore_onwind = _build_onshore_excluder(
                        res=excluder_res_onshore,
                        CORINE_LANDCOVER_PATH=CORINE_LANDCOVER_PATH,
                        landcover_cfg=landcover_onwind,
                        LUISA_LANDCOVER_PATH=LUISA_LANDCOVER_PATH,
                        luisa_cfg=luisa_onwind,
                        use_corine=use_corine,
                        use_luisa=use_luisa,
                        exclude_natura=exclude_natura,
                        natura_path=natura_path,
                        urban_distance_onwind=urban_distance_onwind,
                        luisa_legend_path=luisa_legend_path,
                    )
                    excluder_cache[excluder_key] = excluder_onshore_onwind
                mask_key = ("mask",) + excluder_key
                if mask_cache.get(mask_key) is None:
                    mask_cache[mask_key] = _mask_from_excluder(
                        cutout, onshore_mask_shapes, excluder_onshore_onwind
                    )
                mask_onshore_onwind = mask_cache[mask_key]
                if urban_distance_onwind is not None and urban_distance_onwind > 0:
                    if use_corine:
                        urban_distance_codes_corine = _format_class_list(
                            PYPSA_ONWIND_URBAN_CLC_CODES
                        )
                    if use_luisa:
                        urban_distance_codes_luisa = _format_class_list(
                            PYPSA_LUISA_URBAN_CODES
                        )
        else:
            if do_pv:
                corine_mask = None
                luisa_mask = None
                if use_corine and landcover_pv.active:
                    key = _landuse_cache_key("corine", landcover_pv)
                    if mask_cache.get(key) is None:
                        mask_cache[key] = _landcover_mask(
                            cutout, CORINE_LANDCOVER_PATH, landcover_pv.include, landcover_pv.exclude
                        )
                    corine_mask = mask_cache[key]
                if use_luisa and luisa_pv.active:
                    key = _landuse_cache_key("luisa", luisa_pv)
                    if mask_cache.get(key) is None:
                        mask_cache[key] = _luisa_mask(
                            cutout, LUISA_LANDCOVER_PATH, luisa_pv.include, luisa_pv.exclude
                        )
                    luisa_mask = mask_cache[key]
                landuse_mask = _combine_landuse_masks(
                    landuse_dataset,
                    landuse_fusion,
                    corine_mask,
                    luisa_mask,
                    corine_valid,
                    luisa_valid,
                )
                mask_onshore_pv = mask_onshore_base
                if landuse_mask is not None:
                    mask_onshore_pv = mask_onshore_pv * landuse_mask

            if do_onwind:
                corine_mask = None
                luisa_mask = None
                if use_corine and landcover_onwind.active:
                    key = _landuse_cache_key("corine", landcover_onwind)
                    if mask_cache.get(key) is None:
                        mask_cache[key] = _landcover_mask(
                            cutout, CORINE_LANDCOVER_PATH, landcover_onwind.include, landcover_onwind.exclude
                        )
                    corine_mask = mask_cache[key]
                if use_luisa and luisa_onwind.active:
                    key = _landuse_cache_key("luisa", luisa_onwind)
                    if mask_cache.get(key) is None:
                        mask_cache[key] = _luisa_mask(
                            cutout, LUISA_LANDCOVER_PATH, luisa_onwind.include, luisa_onwind.exclude
                        )
                    luisa_mask = mask_cache[key]
                landuse_mask = _combine_landuse_masks(
                    landuse_dataset,
                    landuse_fusion,
                    corine_mask,
                    luisa_mask,
                    corine_valid,
                    luisa_valid,
                )
                mask_onshore_onwind = mask_onshore_base
                if landuse_mask is not None:
                    mask_onshore_onwind = mask_onshore_onwind * landuse_mask

                if urban_distance_onwind is not None and urban_distance_onwind > 0:
                    if use_corine:
                        key = f"urban_distance_corine:{urban_distance_onwind}"
                        if mask_cache.get(key) is None:
                            mask_cache[key] = _urban_distance_mask(
                                cutout,
                                onshore_mask_shapes,
                                CORINE_LANDCOVER_PATH,
                                PYPSA_ONWIND_URBAN_CLC_CODES,
                                urban_distance_onwind,
                                normalize_fn=_normalize_landcover_codes,
                            )
                        mask_onshore_onwind = mask_onshore_onwind * mask_cache[key]
                        urban_distance_codes_corine = _format_class_list(PYPSA_ONWIND_URBAN_CLC_CODES)
                    if use_luisa:
                        key = f"urban_distance_luisa:{urban_distance_onwind}"
                        if mask_cache.get(key) is None:
                            mask_cache[key] = _urban_distance_mask(
                                cutout,
                                onshore_mask_shapes,
                                LUISA_LANDCOVER_PATH,
                                PYPSA_LUISA_URBAN_CODES,
                                urban_distance_onwind,
                                normalize_fn=_normalize_luisa_codes,
                                legend_path=luisa_legend_path,
                                nodata_default=0,
                            )
                        mask_onshore_onwind = mask_onshore_onwind * mask_cache[key]
                        urban_distance_codes_luisa = _format_class_list(PYPSA_LUISA_URBAN_CODES)

        if wdpa_onshore_mask is not None:
            if mask_onshore_pv is not None:
                mask_onshore_pv = mask_onshore_pv * wdpa_onshore_mask
            if mask_onshore_onwind is not None:
                mask_onshore_onwind = mask_onshore_onwind * wdpa_onshore_mask

    if mask_offshore and do_offwind:
        if use_excluder_offshore:
            excluder_key = (
                "offshore",
                excluder_res_offshore,
                exclude_natura,
                str(natura_path) if exclude_natura else "",
                exclude_shipdensity,
                shipdensity_threshold if exclude_shipdensity else None,
                str(shipdensity_path) if exclude_shipdensity else "",
                str(gebco_path) if min_depth is not None or max_depth is not None else "",
                min_depth,
                max_depth,
                min_shore_distance,
                max_shore_distance,
                str(country_shapes_path),
                exclude_wdpa_offshore,
                ";".join(str(p) for p in wdpa_polygons) if exclude_wdpa_offshore else "",
                ";".join(str(p) for p in wdpa_points) if exclude_wdpa_offshore else "",
                wdpa_filter if exclude_wdpa_offshore else "",
                wdpa_point_buffer_m if exclude_wdpa_offshore else None,
                wdpa_iucn_include if exclude_wdpa_offshore else (),
                wdpa_iucn_exclude if exclude_wdpa_offshore else (),
                wdpa_status_include if exclude_wdpa_offshore else (),
                wdpa_status_exclude if exclude_wdpa_offshore else (),
                wdpa_designation_include if exclude_wdpa_offshore else (),
                wdpa_designation_exclude if exclude_wdpa_offshore else (),
                wdpa_designation_type_include if exclude_wdpa_offshore else (),
                wdpa_designation_type_exclude if exclude_wdpa_offshore else (),
            )
            excluder_offshore = excluder_cache.get(excluder_key)
            if excluder_offshore is None:
                excluder_offshore = _build_offshore_excluder(
                    res=excluder_res_offshore,
                    exclude_natura=exclude_natura,
                    natura_path=natura_path,
                    exclude_shipdensity=exclude_shipdensity,
                    shipdensity_path=shipdensity_path,
                    shipdensity_threshold=shipdensity_threshold,
                    gebco_path=gebco_path,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    min_shore_distance=min_shore_distance,
                    max_shore_distance=max_shore_distance,
                    country_shapes_path=country_shapes_path,
                    exclude_wdpa_offshore=exclude_wdpa_offshore,
                    wdpa_polygons=wdpa_polygons,
                    wdpa_points=wdpa_points,
                    wdpa_filter=wdpa_filter,
                    wdpa_point_buffer_m=wdpa_point_buffer_m,
                    wdpa_iucn_include=wdpa_iucn_include,
                    wdpa_iucn_exclude=wdpa_iucn_exclude,
                    wdpa_status_include=wdpa_status_include,
                    wdpa_status_exclude=wdpa_status_exclude,
                    wdpa_designation_include=wdpa_designation_include,
                    wdpa_designation_exclude=wdpa_designation_exclude,
                    wdpa_designation_type_include=wdpa_designation_type_include,
                    wdpa_designation_type_exclude=wdpa_designation_type_exclude,
                )
                excluder_cache[excluder_key] = excluder_offshore
            mask_key = ("mask",) + excluder_key
            if mask_cache.get(mask_key) is None:
                mask_cache[mask_key] = _mask_from_excluder(
                    cutout, offshore_mask_shapes, excluder_offshore
                )
            mask_offshore_da = mask_cache[mask_key]
        else:
            if min_depth is not None or max_depth is not None:
                if mask_cache.get("bathymetry") is None:
                    mask_cache["bathymetry"] = _bathymetry_mask(
                        cutout, gebco_path, min_depth, max_depth
                    )
                bathymetry_mask_da = mask_cache["bathymetry"]
            if min_shore_distance is not None or max_shore_distance is not None:
                if mask_cache.get("shore_distance") is None:
                    mask_cache["shore_distance"] = _shore_distance_mask(
                        cutout, country_shapes_path, min_shore_distance, max_shore_distance
                    )
                shore_distance_mask_da = mask_cache["shore_distance"]

            if mask_cache.get("offshore_base") is None:
                mask_cache["offshore_base"] = _build_mask(
                    cutout,
                    offshore_mask_shapes,
                    exclude_natura,
                    natura_path,
                    exclude_shipdensity,
                    shipdensity_path,
                    shipdensity_threshold,
                    None,
                    bathymetry_mask_da,
                    shore_distance_mask_da,
                )
            mask_offshore_da = mask_cache["offshore_base"]
            if exclude_wdpa_offshore:
                wdpa_key = (
                    "wdpa",
                    excluder_res_offshore,
                    ";".join(str(p) for p in wdpa_polygons),
                    ";".join(str(p) for p in wdpa_points),
                    wdpa_filter,
                    wdpa_point_buffer_m,
                    wdpa_iucn_include,
                    wdpa_iucn_exclude,
                    wdpa_status_include,
                    wdpa_status_exclude,
                    wdpa_designation_include,
                    wdpa_designation_exclude,
                    wdpa_designation_type_include,
                    wdpa_designation_type_exclude,
                )
                if mask_cache.get(wdpa_key) is None:
                    mask_cache[wdpa_key] = _wdpa_mask(
                        cutout,
                        wdpa_polygons,
                        wdpa_points,
                        wdpa_filter,
                        wdpa_point_buffer_m,
                        excluder_res_offshore,
                        wdpa_iucn_include,
                        wdpa_iucn_exclude,
                        wdpa_status_include,
                        wdpa_status_exclude,
                        wdpa_designation_include,
                        wdpa_designation_exclude,
                        wdpa_designation_type_include,
                        wdpa_designation_type_exclude,
                    )
                wdpa_mask_da = mask_cache[wdpa_key]
                mask_offshore_da = mask_offshore_da * wdpa_mask_da

    if year == resource_class_year:
        if (
            mask_onshore
            and (do_pv or do_onwind)
            and not availability_state.get("onshore", False)
        ):
            out_path = _availability_out_path("onshore", out_dir)
            if overwrite or not out_path.exists():
                area_km2 = _cutout_area_km2(cutout)
                data_vars: dict[str, xr.DataArray] = {"area_km2": area_km2}

                if availability_mode == "raster":
                    if do_pv and mask_onshore_pv is not None:
                        data_vars["availability_pv"] = mask_onshore_pv
                        if capacity_per_sqkm_pv is not None:
                            data_vars["p_nom_max_pv"] = (
                                mask_onshore_pv * area_km2 * capacity_per_sqkm_pv
                            )
                        if resource_classes > 1 and resource_class_mode == "global":
                            cf_mean = cutout.pv(
                                PV_PANEL, PV_ORIENTATION, capacity_factor=True
                            )
                            classes, bins = _resource_classify(
                                cf_mean, mask_onshore_pv, resource_classes
                            )
                            classes.attrs["fill_value"] = "-1"
                            data_vars["resource_class_pv"] = classes
                            data_vars["resource_bins_pv"] = bins

                    if do_onwind and mask_onshore_onwind is not None:
                        data_vars["availability_onwind"] = mask_onshore_onwind
                        if capacity_per_sqkm_onwind is not None:
                            data_vars["p_nom_max_onwind"] = (
                                mask_onshore_onwind * area_km2 * capacity_per_sqkm_onwind
                            )
                        if resource_classes > 1 and resource_class_mode == "global":
                            cf_mean = cutout.wind(ONWIND_TURBINE, capacity_factor=True)
                            classes, bins = _resource_classify(
                                cf_mean, mask_onshore_onwind, resource_classes
                            )
                            classes.attrs["fill_value"] = "-1"
                            data_vars["resource_class_onwind"] = classes
                            data_vars["resource_bins_onwind"] = bins
                else:
                    regions = _load_regions(onshore_regions_path)
                    if regions.empty:
                        LOG.warning("empty onshore regions for availability, skipping")
                    else:
                        if do_pv and mask_onshore_pv is not None:
                            if use_excluder_onshore and excluder_onshore_pv is not None:
                                avail = _availability_by_regions_excluder(
                                    cutout, regions, excluder_onshore_pv
                                )
                                if wdpa_onshore_mask is not None:
                                    avail = avail * wdpa_onshore_mask
                            else:
                                avail = _availability_by_regions(
                                    cutout, regions, mask_onshore_pv
                                )
                            data_vars["availability_pv"] = avail
                            if capacity_per_sqkm_pv is not None:
                                data_vars["p_nom_max_pv_bus"] = (
                                    avail * area_km2
                                ).sum(("y", "x")) * capacity_per_sqkm_pv
                            if resource_classes > 1 and resource_class_mode == "per-region":
                                cf_mean = cutout.pv(
                                    PV_PANEL, PV_ORIENTATION, capacity_factor=True
                                )
                                classes, bins = _resource_classify_per_region(
                                    cf_mean, avail, resource_classes
                                )
                                classes.attrs["fill_value"] = "-1"
                                data_vars["resource_class_pv"] = classes
                                data_vars["resource_bins_pv"] = bins
                                if capacity_per_sqkm_pv is not None:
                                    data_vars["p_nom_max_pv_bus_bin"] = _p_nom_max_bus_bin(
                                        avail,
                                        classes,
                                        area_km2,
                                        capacity_per_sqkm_pv,
                                        resource_classes,
                                    )
                            if resource_classes > 1 and resource_class_mode == "global":
                                cf_mean = cutout.pv(
                                    PV_PANEL, PV_ORIENTATION, capacity_factor=True
                                )
                                classes, bins = _resource_classify(
                                    cf_mean, mask_onshore_pv, resource_classes
                                )
                                classes.attrs["fill_value"] = "-1"
                                data_vars["resource_class_pv"] = classes
                                data_vars["resource_bins_pv"] = bins
                                if capacity_per_sqkm_pv is not None:
                                    data_vars["p_nom_max_pv_bus_bin"] = _p_nom_max_bus_bin(
                                        avail,
                                        classes,
                                        area_km2,
                                        capacity_per_sqkm_pv,
                                        resource_classes,
                                    )
                        if do_onwind and mask_onshore_onwind is not None:
                            if use_excluder_onshore and excluder_onshore_onwind is not None:
                                avail = _availability_by_regions_excluder(
                                    cutout, regions, excluder_onshore_onwind
                                )
                                if wdpa_onshore_mask is not None:
                                    avail = avail * wdpa_onshore_mask
                            else:
                                avail = _availability_by_regions(
                                    cutout, regions, mask_onshore_onwind
                                )
                            data_vars["availability_onwind"] = avail
                            if capacity_per_sqkm_onwind is not None:
                                data_vars["p_nom_max_onwind_bus"] = (
                                    avail * area_km2
                                ).sum(("y", "x")) * capacity_per_sqkm_onwind
                            if resource_classes > 1 and resource_class_mode == "per-region":
                                cf_mean = cutout.wind(ONWIND_TURBINE, capacity_factor=True)
                                classes, bins = _resource_classify_per_region(
                                    cf_mean, avail, resource_classes
                                )
                                classes.attrs["fill_value"] = "-1"
                                data_vars["resource_class_onwind"] = classes
                                data_vars["resource_bins_onwind"] = bins
                                if capacity_per_sqkm_onwind is not None:
                                    data_vars["p_nom_max_onwind_bus_bin"] = _p_nom_max_bus_bin(
                                        avail,
                                        classes,
                                        area_km2,
                                        capacity_per_sqkm_onwind,
                                        resource_classes,
                                    )
                            if resource_classes > 1 and resource_class_mode == "global":
                                cf_mean = cutout.wind(ONWIND_TURBINE, capacity_factor=True)
                                classes, bins = _resource_classify(
                                    cf_mean, mask_onshore_onwind, resource_classes
                                )
                                classes.attrs["fill_value"] = "-1"
                                data_vars["resource_class_onwind"] = classes
                                data_vars["resource_bins_onwind"] = bins
                                if capacity_per_sqkm_onwind is not None:
                                    data_vars["p_nom_max_onwind_bus_bin"] = _p_nom_max_bus_bin(
                                        avail,
                                        classes,
                                        area_km2,
                                        capacity_per_sqkm_onwind,
                                        resource_classes,
                                    )

                ds = xr.Dataset(data_vars)
                ds.attrs.update(
                    {
                        "cutout": str(cutout_path),
                        "mask_onshore_shapes": _format_path_list(onshore_mask_shapes),
                        "availability_regions_onshore": str(onshore_regions_path),
                        "mask_excluder": "true" if use_excluder_onshore else "false",
                        "mask_excluder_res_onshore": ""
                        if not use_excluder_onshore
                        else str(excluder_res_onshore),
                        "mask_natura": "true" if exclude_natura else "false",
                        "mask_landuse_dataset": landuse_dataset,
                        "mask_landuse_fusion": landuse_fusion,
                        "mask_landcover_corine": str(CORINE_LANDCOVER_PATH) if use_corine else "",
                        "mask_landcover_corine_pv_codes": landcover_pv.codes_input,
                        "mask_landcover_corine_onwind_codes": landcover_onwind.codes_input,
                        "mask_landcover_luisa": str(LUISA_LANDCOVER_PATH) if use_luisa else "",
                        "mask_landcover_luisa_pv_codes": luisa_pv.codes_input,
                        "mask_landcover_luisa_onwind_codes": luisa_onwind.codes_input,
                        "mask_urban_distance_m": ""
                        if urban_distance_onwind is None
                        else str(urban_distance_onwind),
                        "mask_urban_codes_corine": urban_distance_codes_corine,
                        "mask_urban_codes_luisa": urban_distance_codes_luisa,
                        "mask_wdpa_onshore": "true" if exclude_wdpa_onshore else "false",
                        "mask_wdpa_onshore_filter": ""
                        if not exclude_wdpa_onshore
                        else wdpa_onshore_filter,
                        "mask_wdpa_onshore_point_buffer_m": ""
                        if not exclude_wdpa_onshore
                        else str(wdpa_point_buffer_m),
                        "mask_wdpa_onshore_iucn_include": ""
                        if not exclude_wdpa_onshore
                        else _format_string_list(wdpa_onshore_iucn_include),
                        "mask_wdpa_onshore_iucn_exclude": ""
                        if not exclude_wdpa_onshore
                        else _format_string_list(wdpa_onshore_iucn_exclude),
                        "mask_wdpa_onshore_status_include": ""
                        if not exclude_wdpa_onshore
                        else _format_string_list(wdpa_onshore_status_include),
                        "mask_wdpa_onshore_status_exclude": ""
                        if not exclude_wdpa_onshore
                        else _format_string_list(wdpa_onshore_status_exclude),
                        "mask_wdpa_onshore_designation_include": ""
                        if not exclude_wdpa_onshore
                        else _format_string_list(wdpa_onshore_designation_include),
                        "mask_wdpa_onshore_designation_exclude": ""
                        if not exclude_wdpa_onshore
                        else _format_string_list(wdpa_onshore_designation_exclude),
                        "mask_wdpa_onshore_designation_type_include": ""
                        if not exclude_wdpa_onshore
                        else _format_string_list(wdpa_onshore_designation_type_include),
                        "mask_wdpa_onshore_designation_type_exclude": ""
                        if not exclude_wdpa_onshore
                        else _format_string_list(wdpa_onshore_designation_type_exclude),
                        "availability_mode": availability_mode,
                        "resource_class_mode": resource_class_mode,
                        "resource_classes": str(resource_classes),
                        "resource_class_year": str(resource_class_year),
                        "capacity_per_sqkm_pv": ""
                        if capacity_per_sqkm_pv is None
                        else str(capacity_per_sqkm_pv),
                        "capacity_per_sqkm_onwind": ""
                        if capacity_per_sqkm_onwind is None
                        else str(capacity_per_sqkm_onwind),
                    }
                )
                _write_availability(ds, out_path)
            else:
                LOG.info("exists, skipping %s", out_path)
            availability_state["onshore"] = True

        if mask_offshore and do_offwind and not availability_state.get("offshore", False):
            out_path = _availability_out_path("offshore", out_dir)
            if overwrite or not out_path.exists():
                area_km2 = _cutout_area_km2(cutout)
                data_vars = {"area_km2": area_km2}
                if availability_mode == "raster":
                    if mask_offshore_da is not None:
                        data_vars["availability_offwind"] = mask_offshore_da
                        if capacity_per_sqkm_offwind is not None:
                            data_vars["p_nom_max_offwind"] = (
                                mask_offshore_da * area_km2 * capacity_per_sqkm_offwind
                            )
                        if resource_classes > 1 and resource_class_mode == "global":
                            cf_mean = cutout.wind(
                                OFFWIND_TURBINE, capacity_factor=True
                            )
                            classes, bins = _resource_classify(
                                cf_mean, mask_offshore_da, resource_classes
                            )
                            classes.attrs["fill_value"] = "-1"
                            data_vars["resource_class_offwind"] = classes
                            data_vars["resource_bins_offwind"] = bins
                elif mask_offshore_da is not None:
                    regions = _load_regions(offshore_regions_path)
                    if regions.empty:
                        LOG.warning("empty offshore regions for availability, skipping")
                    else:
                        if use_excluder_offshore and excluder_offshore is not None:
                            avail = _availability_by_regions_excluder(
                                cutout, regions, excluder_offshore
                            )
                        else:
                            avail = _availability_by_regions(
                                cutout, regions, mask_offshore_da
                            )
                        data_vars["availability_offwind"] = avail
                        if capacity_per_sqkm_offwind is not None:
                            data_vars["p_nom_max_offwind_bus"] = (
                                avail * area_km2
                            ).sum(("y", "x")) * capacity_per_sqkm_offwind
                        if resource_classes > 1 and resource_class_mode == "per-region":
                            cf_mean = cutout.wind(OFFWIND_TURBINE, capacity_factor=True)
                            classes, bins = _resource_classify_per_region(
                                cf_mean, avail, resource_classes
                            )
                            classes.attrs["fill_value"] = "-1"
                            data_vars["resource_class_offwind"] = classes
                            data_vars["resource_bins_offwind"] = bins
                            if capacity_per_sqkm_offwind is not None:
                                data_vars["p_nom_max_offwind_bus_bin"] = _p_nom_max_bus_bin(
                                    avail,
                                    classes,
                                    area_km2,
                                    capacity_per_sqkm_offwind,
                                    resource_classes,
                                )
                        if resource_classes > 1 and resource_class_mode == "global":
                            cf_mean = cutout.wind(OFFWIND_TURBINE, capacity_factor=True)
                            classes, bins = _resource_classify(
                                cf_mean, mask_offshore_da, resource_classes
                            )
                            classes.attrs["fill_value"] = "-1"
                            data_vars["resource_class_offwind"] = classes
                            data_vars["resource_bins_offwind"] = bins
                            if capacity_per_sqkm_offwind is not None:
                                data_vars["p_nom_max_offwind_bus_bin"] = _p_nom_max_bus_bin(
                                    avail,
                                    classes,
                                    area_km2,
                                    capacity_per_sqkm_offwind,
                                    resource_classes,
                                )

                ds = xr.Dataset(data_vars)
                ds.attrs.update(
                    {
                        "cutout": str(cutout_path),
                        "mask_offshore_shapes": _format_path_list(offshore_mask_shapes),
                        "availability_regions_offshore": str(offshore_regions_path),
                        "mask_excluder": "true" if use_excluder_offshore else "false",
                        "mask_excluder_res_offshore": ""
                        if not use_excluder_offshore
                        else str(excluder_res_offshore),
                        "mask_natura": "true" if exclude_natura else "false",
                        "mask_shipdensity": "true" if exclude_shipdensity else "false",
                        "mask_wdpa_offshore": "true" if exclude_wdpa_offshore else "false",
                        "mask_wdpa_offshore_filter": wdpa_filter if exclude_wdpa_offshore else "",
                        "mask_wdpa_offshore_point_buffer_m": ""
                        if not exclude_wdpa_offshore
                        else str(wdpa_point_buffer_m),
                        "mask_wdpa_offshore_iucn_include": ""
                        if not exclude_wdpa_offshore
                        else _format_string_list(wdpa_iucn_include),
                        "mask_wdpa_offshore_iucn_exclude": ""
                        if not exclude_wdpa_offshore
                        else _format_string_list(wdpa_iucn_exclude),
                        "mask_wdpa_offshore_status_include": ""
                        if not exclude_wdpa_offshore
                        else _format_string_list(wdpa_status_include),
                        "mask_wdpa_offshore_status_exclude": ""
                        if not exclude_wdpa_offshore
                        else _format_string_list(wdpa_status_exclude),
                        "mask_wdpa_offshore_designation_include": ""
                        if not exclude_wdpa_offshore
                        else _format_string_list(wdpa_designation_include),
                        "mask_wdpa_offshore_designation_exclude": ""
                        if not exclude_wdpa_offshore
                        else _format_string_list(wdpa_designation_exclude),
                        "mask_wdpa_offshore_designation_type_include": ""
                        if not exclude_wdpa_offshore
                        else _format_string_list(wdpa_designation_type_include),
                        "mask_wdpa_offshore_designation_type_exclude": ""
                        if not exclude_wdpa_offshore
                        else _format_string_list(wdpa_designation_type_exclude),
                        "shipdensity_threshold": str(shipdensity_threshold),
                        "mask_bathymetry": str(gebco_path)
                        if min_depth is not None or max_depth is not None
                        else "",
                        "mask_bathymetry_min_depth": ""
                        if min_depth is None
                        else str(min_depth),
                        "mask_bathymetry_max_depth": ""
                        if max_depth is None
                        else str(max_depth),
                        "mask_shore_distance_min": ""
                        if min_shore_distance is None
                        else str(min_shore_distance),
                        "mask_shore_distance_max": ""
                        if max_shore_distance is None
                        else str(max_shore_distance),
                        "availability_mode": availability_mode,
                        "resource_class_mode": resource_class_mode,
                        "resource_classes": str(resource_classes),
                        "resource_class_year": str(resource_class_year),
                        "capacity_per_sqkm_offwind": ""
                        if capacity_per_sqkm_offwind is None
                        else str(capacity_per_sqkm_offwind),
                    }
                )
                _write_availability(ds, out_path)
            else:
                LOG.info("exists, skipping %s", out_path)
            availability_state["offshore"] = True
    try:
        if not masks_out_only:
            if do_pv:
                out_path = _cf_out_path("pv", year, out_dir)
                if overwrite or not out_path.exists():
                    cf = cutout.pv(PV_PANEL, PV_ORIENTATION, capacity_factor_timeseries=True)
                    cf = _clip_p_max_pu(cf, clip_p_max_pu)
                    cf = _apply_mask(cf, mask_onshore_pv)
                    _write_cf(
                        cf,
                        out_path,
                        {
                            "resource": "pv",
                            "panel": PV_PANEL,
                            "orientation": PV_ORIENTATION,
                            "clip_p_max_pu": "" if clip_p_max_pu is None else str(clip_p_max_pu),
                            "mask": _format_path_list(onshore_mask_shapes)
                            if mask_onshore_pv is not None
                            else "",
                            "mask_excluder": "true" if use_excluder_onshore else "false",
                            "mask_excluder_res_onshore": ""
                            if not use_excluder_onshore
                            else str(excluder_res_onshore),
                            "mask_natura": "true" if exclude_natura else "false",
                            "mask_landuse_dataset": landuse_dataset,
                            "mask_landuse_fusion": landuse_fusion,
                            "mask_landcover": str(CORINE_LANDCOVER_PATH) if landcover_pv.active else "",
                            "mask_landcover_mode": landcover_pv.mode,
                            "mask_landcover_codes": landcover_pv.codes_input,
                            "mask_landcover_classes": landcover_pv.classes_mapped,
                            "mask_landcover_luisa": str(LUISA_LANDCOVER_PATH) if luisa_pv.active else "",
                            "mask_landcover_luisa_mode": luisa_pv.mode,
                            "mask_landcover_luisa_codes": luisa_pv.codes_input,
                            "mask_wdpa_onshore": "true" if exclude_wdpa_onshore else "false",
                            "mask_wdpa_onshore_filter": ""
                            if not exclude_wdpa_onshore
                            else wdpa_onshore_filter,
                            "mask_wdpa_onshore_point_buffer_m": ""
                            if not exclude_wdpa_onshore
                            else str(wdpa_point_buffer_m),
                            "mask_wdpa_onshore_iucn_include": ""
                            if not exclude_wdpa_onshore
                            else _format_string_list(wdpa_onshore_iucn_include),
                            "mask_wdpa_onshore_iucn_exclude": ""
                            if not exclude_wdpa_onshore
                            else _format_string_list(wdpa_onshore_iucn_exclude),
                            "mask_wdpa_onshore_status_include": ""
                            if not exclude_wdpa_onshore
                            else _format_string_list(wdpa_onshore_status_include),
                            "mask_wdpa_onshore_status_exclude": ""
                            if not exclude_wdpa_onshore
                            else _format_string_list(wdpa_onshore_status_exclude),
                            "mask_wdpa_onshore_designation_include": ""
                            if not exclude_wdpa_onshore
                            else _format_string_list(wdpa_onshore_designation_include),
                            "mask_wdpa_onshore_designation_exclude": ""
                            if not exclude_wdpa_onshore
                            else _format_string_list(wdpa_onshore_designation_exclude),
                            "mask_wdpa_onshore_designation_type_include": ""
                            if not exclude_wdpa_onshore
                            else _format_string_list(wdpa_onshore_designation_type_include),
                            "mask_wdpa_onshore_designation_type_exclude": ""
                            if not exclude_wdpa_onshore
                            else _format_string_list(wdpa_onshore_designation_type_exclude),
                            "cutout": str(cutout_path),
                        },
                    )
                else:
                    LOG.info("exists, skipping %s", out_path)

            if do_onwind:
                out_path = _cf_out_path("onwind", year, out_dir)
                if overwrite or not out_path.exists():
                    cf = cutout.wind(ONWIND_TURBINE, capacity_factor_timeseries=True)
                    cf = _clip_p_max_pu(cf, clip_p_max_pu)
                    cf = _apply_mask(cf, mask_onshore_onwind)
                    _write_cf(
                        cf,
                        out_path,
                        {
                            "resource": "onwind",
                            "turbine": ONWIND_TURBINE,
                            "clip_p_max_pu": "" if clip_p_max_pu is None else str(clip_p_max_pu),
                            "mask": _format_path_list(onshore_mask_shapes)
                            if mask_onshore_onwind is not None
                            else "",
                            "mask_excluder": "true" if use_excluder_onshore else "false",
                            "mask_excluder_res_onshore": ""
                            if not use_excluder_onshore
                            else str(excluder_res_onshore),
                            "mask_natura": "true" if exclude_natura else "false",
                            "mask_landuse_dataset": landuse_dataset,
                            "mask_landuse_fusion": landuse_fusion,
                            "mask_landcover": str(CORINE_LANDCOVER_PATH) if landcover_onwind.active else "",
                            "mask_landcover_mode": landcover_onwind.mode,
                            "mask_landcover_codes": landcover_onwind.codes_input,
                            "mask_landcover_classes": landcover_onwind.classes_mapped,
                            "mask_landcover_luisa": str(LUISA_LANDCOVER_PATH) if luisa_onwind.active else "",
                            "mask_landcover_luisa_mode": luisa_onwind.mode,
                            "mask_landcover_luisa_codes": luisa_onwind.codes_input,
                            "mask_urban_distance_m": ""
                            if urban_distance_onwind is None
                            else str(urban_distance_onwind),
                            "mask_urban_codes_corine": urban_distance_codes_corine,
                            "mask_urban_codes_luisa": urban_distance_codes_luisa,
                            "mask_wdpa_onshore": "true" if exclude_wdpa_onshore else "false",
                            "mask_wdpa_onshore_filter": ""
                            if not exclude_wdpa_onshore
                            else wdpa_onshore_filter,
                            "mask_wdpa_onshore_point_buffer_m": ""
                            if not exclude_wdpa_onshore
                            else str(wdpa_point_buffer_m),
                            "mask_wdpa_onshore_iucn_include": ""
                            if not exclude_wdpa_onshore
                            else _format_string_list(wdpa_onshore_iucn_include),
                            "mask_wdpa_onshore_iucn_exclude": ""
                            if not exclude_wdpa_onshore
                            else _format_string_list(wdpa_onshore_iucn_exclude),
                            "mask_wdpa_onshore_status_include": ""
                            if not exclude_wdpa_onshore
                            else _format_string_list(wdpa_onshore_status_include),
                            "mask_wdpa_onshore_status_exclude": ""
                            if not exclude_wdpa_onshore
                            else _format_string_list(wdpa_onshore_status_exclude),
                            "mask_wdpa_onshore_designation_include": ""
                            if not exclude_wdpa_onshore
                            else _format_string_list(wdpa_onshore_designation_include),
                            "mask_wdpa_onshore_designation_exclude": ""
                            if not exclude_wdpa_onshore
                            else _format_string_list(wdpa_onshore_designation_exclude),
                            "mask_wdpa_onshore_designation_type_include": ""
                            if not exclude_wdpa_onshore
                            else _format_string_list(wdpa_onshore_designation_type_include),
                            "mask_wdpa_onshore_designation_type_exclude": ""
                            if not exclude_wdpa_onshore
                            else _format_string_list(wdpa_onshore_designation_type_exclude),
                            "cutout": str(cutout_path),
                        },
                    )
                else:
                    LOG.info("exists, skipping %s", out_path)

            if do_offwind:
                out_path = _cf_out_path("offwind", year, out_dir)
                if overwrite or not out_path.exists():
                    cf = cutout.wind(OFFWIND_TURBINE, capacity_factor_timeseries=True)
                    cf = _clip_p_max_pu(cf, clip_p_max_pu)
                    cf = _apply_mask(cf, mask_offshore_da)
                    _write_cf(
                        cf,
                        out_path,
                        {
                            "resource": "offwind",
                            "turbine": OFFWIND_TURBINE,
                            "connection_type": OFFWIND_CONNECTION_TYPE,
                            "clip_p_max_pu": "" if clip_p_max_pu is None else str(clip_p_max_pu),
                            "mask": _format_path_list(offshore_mask_shapes)
                            if mask_offshore_da is not None
                            else "",
                            "mask_excluder": "true" if use_excluder_offshore else "false",
                            "mask_excluder_res_offshore": ""
                            if not use_excluder_offshore
                            else str(excluder_res_offshore),
                            "mask_natura": "true" if exclude_natura else "false",
                            "mask_shipdensity": "true" if exclude_shipdensity else "false",
                            "mask_wdpa_offshore": "true" if exclude_wdpa_offshore else "false",
                            "mask_wdpa_offshore_filter": wdpa_filter if exclude_wdpa_offshore else "",
                            "mask_wdpa_offshore_point_buffer_m": ""
                            if not exclude_wdpa_offshore
                            else str(wdpa_point_buffer_m),
                            "mask_wdpa_offshore_iucn_include": ""
                            if not exclude_wdpa_offshore
                            else _format_string_list(wdpa_iucn_include),
                            "mask_wdpa_offshore_iucn_exclude": ""
                            if not exclude_wdpa_offshore
                            else _format_string_list(wdpa_iucn_exclude),
                            "mask_wdpa_offshore_status_include": ""
                            if not exclude_wdpa_offshore
                            else _format_string_list(wdpa_status_include),
                            "mask_wdpa_offshore_status_exclude": ""
                            if not exclude_wdpa_offshore
                            else _format_string_list(wdpa_status_exclude),
                            "mask_wdpa_offshore_designation_include": ""
                            if not exclude_wdpa_offshore
                            else _format_string_list(wdpa_designation_include),
                            "mask_wdpa_offshore_designation_exclude": ""
                            if not exclude_wdpa_offshore
                            else _format_string_list(wdpa_designation_exclude),
                            "mask_wdpa_offshore_designation_type_include": ""
                            if not exclude_wdpa_offshore
                            else _format_string_list(wdpa_designation_type_include),
                            "mask_wdpa_offshore_designation_type_exclude": ""
                            if not exclude_wdpa_offshore
                            else _format_string_list(wdpa_designation_type_exclude),
                            "mask_bathymetry": str(gebco_path)
                            if min_depth is not None or max_depth is not None
                            else "",
                            "mask_bathymetry_min_depth": ""
                            if min_depth is None
                            else str(min_depth),
                            "mask_bathymetry_max_depth": ""
                            if max_depth is None
                            else str(max_depth),
                            "mask_shore_distance_min": ""
                            if min_shore_distance is None
                            else str(min_shore_distance),
                            "mask_shore_distance_max": ""
                            if max_shore_distance is None
                            else str(max_shore_distance),
                            "cutout": str(cutout_path),
                        },
                    )
                else:
                    LOG.info("exists, skipping %s", out_path)

            if do_hydro:
                if p_inst is None or bus_country is None:
                    raise RuntimeError("hydro needs p_inst and bus_country")
                out_bus, out_country = _hydro_out_paths(year, out_dir)
                needs_write = overwrite or not out_country.exists()
                if hydro_write_bus_profiles:
                    needs_write = needs_write or not out_bus.exists()
                if needs_write:
                    bus_ds, country_ds = _compute_hydro_profiles(
                        cutout,
                        country_shapes_path,
                        p_inst,
                        bus_country,
                        hydro_scale,
                        hydro_reductions_path,
                        hydro_write_bus_profiles,
                    )
                    if hydro_scale is not None:
                        if bus_ds is not None:
                            bus_ds.attrs["hydro_normalized"] = "true"
                            bus_ds.attrs["hydro_reference"] = str(hydro_reference)
                            bus_ds.attrs["hydro_reference_year"] = str(hydro_reference_year)
                        country_ds.attrs["hydro_normalized"] = "true"
                        country_ds.attrs["hydro_reference"] = str(hydro_reference)
                        country_ds.attrs["hydro_reference_year"] = str(hydro_reference_year)
                    if bus_ds is not None:
                        LOG.info("writing %s", out_bus)
                        bus_ds.to_netcdf(out_bus, engine="netcdf4")
                    LOG.info("writing %s", out_country)
                    country_ds.to_netcdf(out_country, engine="netcdf4")
                else:
                    if hydro_write_bus_profiles:
                        LOG.info("exists, skipping hydro %s / %s", out_bus, out_country)
                    else:
                        LOG.info("exists, skipping hydro %s", out_country)
        else:
            LOG.info("masks-out-only enabled: skipping CF/hydro outputs for %s", year)
    finally:
        cutout.data.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute atlite capacity factor profiles.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--start-year", type=int, default=2013)
    parser.add_argument("--end-year", type=int, default=2013)
    parser.add_argument("--cutout-dir", type=Path, default=CUTOUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--resources",
        nargs="+",
        choices=["pv", "onwind", "offwind", "hydro", "all"],
        default=["all"],
    )
    parser.add_argument("--mask-onshore", action="store_true")
    parser.add_argument("--mask-offshore", action="store_true")
    parser.add_argument("--use-excluder-mask", action="store_true")
    parser.add_argument("--excluder-res-onshore", type=float, default=100.0)
    parser.add_argument("--excluder-res-offshore", type=float, default=200.0)
    parser.add_argument("--onshore-mask-shapes", type=Path, nargs="+")
    parser.add_argument("--offshore-mask-shapes", type=Path, nargs="+")
    parser.add_argument("--onshore-regions", type=Path)
    parser.add_argument("--offshore-regions", type=Path)
    parser.add_argument("--onshore-shapes", type=Path)
    parser.add_argument("--offshore-shapes", type=Path)
    parser.add_argument("--country-shapes", type=Path, default=COUNTRY_SHAPES)
    parser.add_argument("--exclude-natura", action="store_true")
    parser.add_argument("--natura-path", type=Path, default=NATURA_PATH)
    parser.add_argument("--exclude-shipdensity", action="store_true")
    parser.add_argument("--shipdensity-path", type=Path, default=SHIPDENSITY_PATH)
    parser.add_argument("--shipdensity-threshold", type=float)
    parser.add_argument("--exclude-wdpa-offshore", action="store_true")
    parser.add_argument("--exclude-wdpa-onshore", action="store_true")
    parser.add_argument("--wdpa-polygons", type=Path, nargs="+", default=WDPA_POLYGON_PATHS)
    parser.add_argument("--wdpa-points", type=Path, nargs="+", default=WDPA_POINT_PATHS)
    parser.add_argument(
        "--wdpa-filter",
        choices=["marine", "all"],
        default="marine",
    )
    parser.add_argument(
        "--wdpa-point-buffer-m",
        type=float,
        default=WDPA_POINT_BUFFER_M,
    )
    parser.add_argument("--wdpa-iucn-include", nargs="+")
    parser.add_argument("--wdpa-iucn-exclude", nargs="+")
    parser.add_argument("--wdpa-status-include", nargs="+")
    parser.add_argument("--wdpa-status-exclude", nargs="+")
    parser.add_argument("--wdpa-designation-include", nargs="+")
    parser.add_argument("--wdpa-designation-exclude", nargs="+")
    parser.add_argument("--wdpa-designation-type-include", nargs="+")
    parser.add_argument("--wdpa-designation-type-exclude", nargs="+")
    parser.add_argument(
        "--landuse-dataset",
        choices=["corine", "luisa", "both"],
        default="corine",
    )
    parser.add_argument(
        "--landuse-fusion",
        choices=["intersection", "prefer-corine", "prefer-luisa"],
        default="prefer-luisa",
    )
    parser.add_argument(
        "--corine-landcover-path", type=Path, default=CORINE_LANDCOVER_PATH
    )
    parser.add_argument("--landcover-include", type=int, nargs="+")
    parser.add_argument("--landcover-exclude", type=int, nargs="+")
    parser.add_argument("--landcover-include-pv", type=int, nargs="+")
    parser.add_argument("--landcover-exclude-pv", type=int, nargs="+")
    parser.add_argument("--landcover-include-onwind", type=int, nargs="+")
    parser.add_argument("--landcover-exclude-onwind", type=int, nargs="+")
    parser.add_argument("--luisa-landcover-path", type=Path, default=LUISA_LANDCOVER_PATH)
    parser.add_argument("--luisa-legend", type=Path, default=LUISA_LEGEND_PATH)
    parser.add_argument("--luisa-include", type=int, nargs="+")
    parser.add_argument("--luisa-exclude", type=int, nargs="+")
    parser.add_argument("--luisa-include-pv", type=int, nargs="+")
    parser.add_argument("--luisa-exclude-pv", type=int, nargs="+")
    parser.add_argument("--luisa-include-onwind", type=int, nargs="+")
    parser.add_argument("--luisa-exclude-onwind", type=int, nargs="+")
    parser.add_argument("--gebco-path", type=Path, default=GEBCO_PATH)
    parser.add_argument("--min-depth", type=float)
    parser.add_argument("--max-depth", type=float)
    parser.add_argument("--min-shore-distance", type=float)
    parser.add_argument("--max-shore-distance", type=float)
    parser.add_argument("--urban-distance-onwind", type=float)
    parser.add_argument(
        "--availability-mode", choices=["raster", "regions"], default="raster"
    )
    parser.add_argument("--resource-classes", type=int, default=1)
    parser.add_argument(
        "--resource-class-mode",
        choices=["global", "per-region"],
        default="global",
    )
    parser.add_argument("--resource-class-year", type=int)
    parser.add_argument("--clip-p-max-pu", type=float)
    parser.add_argument(
        "--capacity-per-sqkm-pv", type=float, default=PYPSA_CAPACITY_PER_SQKM_PV
    )
    parser.add_argument(
        "--capacity-per-sqkm-onwind", type=float, default=PYPSA_CAPACITY_PER_SQKM_ONWIND
    )
    parser.add_argument(
        "--capacity-per-sqkm-offwind", type=float, default=PYPSA_CAPACITY_PER_SQKM_OFFWIND
    )
    parser.add_argument("--onwind-turbine", type=str, default=ONWIND_TURBINE)
    parser.add_argument("--offwind-turbine", type=str, default=OFFWIND_TURBINE)
    parser.add_argument("--pypsa-defaults", action="store_true")
    parser.add_argument(
        "--pypsa-offwind-variant",
        choices=["ac", "dc", "acdc", "float"],
        default="acdc",
    )
    parser.add_argument("--hydro-normalize", action="store_true")
    parser.add_argument("--hydro-normalize-year", type=int, default=HYDRO_REFERENCE_YEAR)
    parser.add_argument("--hydro-reference", type=Path, default=HYDRO_REFERENCE)
    parser.add_argument("--hydro-reductions-path", type=Path, default=COUNTRY_REDUCTIONS)
    parser.add_argument(
        "--hydro-write-bus-profiles",
        action="store_true",
        default=HYDRO_WRITE_BUS_PROFILES,
    )
    parser.add_argument("--masks-out-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args()
    parser = _build_parser()
    config_args: list[str] = []
    if pre_args.config:
        config = _load_yaml_config(pre_args.config)
        config = _normalize_config_keys(config)
        config_args = _config_to_argv(config, parser)
    return parser.parse_args(config_args + sys.argv[1:])


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.config:
        LOG.info("using config: %s", args.config)
    global ONWIND_TURBINE, OFFWIND_TURBINE
    ONWIND_TURBINE = args.onwind_turbine
    OFFWIND_TURBINE = args.offwind_turbine
    LOG.info("using turbines: onwind=%s offwind=%s", ONWIND_TURBINE, OFFWIND_TURBINE)
    if args.onshore_shapes or args.offshore_shapes:
        LOG.warning(
            "--onshore-shapes/--offshore-shapes are deprecated; use "
            "--onshore-mask-shapes/--offshore-mask-shapes for mask clipping and "
            "--onshore-regions/--offshore-regions for availability regions."
        )
    if args.onshore_mask_shapes is None:
        if args.onshore_shapes is not None:
            args.onshore_mask_shapes = [args.onshore_shapes]
        else:
            args.onshore_mask_shapes = list(ONSHORE_MASK_SHAPES)
    if args.offshore_mask_shapes is None:
        if args.offshore_shapes is not None:
            args.offshore_mask_shapes = [args.offshore_shapes]
        else:
            args.offshore_mask_shapes = list(OFFSHORE_MASK_SHAPES)
    if args.onshore_regions is None:
        args.onshore_regions = (
            args.onshore_shapes if args.onshore_shapes is not None else ONSHORE_SHAPES
        )
    if args.offshore_regions is None:
        args.offshore_regions = (
            args.offshore_shapes if args.offshore_shapes is not None else OFFSHORE_SHAPES
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # CLI overview:
    # --start-year/--end-year: inclusive year range of cutouts to process.
    # --cutout-dir/--out-dir: input cutout folder and output base folder.
    # --resources: subset of ["pv", "onwind", "offwind", "hydro", "all"].
    # --mask-onshore: apply onshore shape mask (PV + onwind).
    # --mask-offshore: apply offshore shape mask (offwind).
    # --onshore-mask-shapes/--offshore-mask-shapes: base mask shapes (intersected).
    # --onshore-regions/--offshore-regions: region polygons for availability_mode=regions.
    # --use-excluder-mask: build masks via high-resolution ExclusionContainer.
    # --excluder-res-onshore/--excluder-res-offshore: base resolution in meters
    #   for onshore/offshore excluder masks.
    # --config: load arguments from a YAML config file (keys use underscores).
    #   CLI options override config values; true flags in config can only be
    #   disabled by editing the config.
    # --exclude-natura: multiply Natura2000 exclusions into onshore/offshore masks.
    # --exclude-shipdensity: exclude shipping lanes for offshore (thresholded).
    # --exclude-wdpa-offshore: exclude WDPA protected areas for offshore; use
    #   --wdpa-filter marine/all and --wdpa-point-buffer-m for points.
    #   Optional filters: --wdpa-iucn-*, --wdpa-status-*, --wdpa-designation-*,
    #   --wdpa-designation-type-* (include or exclude values).
    # --exclude-wdpa-onshore: exclude WDPA protected areas for onshore using a
    #   strict filter (terrestrial only, IUCN Ia/Ib/II/III, STATUS Adopted/Designated/Inscribed).
    # --shipdensity-threshold: explicit shipdensity threshold; if omitted and
    #   --pypsa-defaults is set, uses PyPSA's 400 * 8760 * 6; otherwise uses the
    #   script default.
    # - --landuse-dataset: choose landuse source(s) for onshore masks
    #   ("corine", "luisa", or "both" for intersection).
    # - --landuse-fusion: when --landuse-dataset both, choose "intersection"
    #   (AND), "prefer-corine" (use LUISA only where CORINE has nodata), or
    #   "prefer-luisa" (use CORINE only where LUISA has nodata).
    # - --corine-landcover-path: CORINE landcover raster path.
    # - --landcover-include/--landcover-exclude: global CLC mask (3-digit codes
    #   like 111, 211, 312 or raster classes 1-44); applies to PV + onwind unless
    #   overridden per technology.
    # - --landcover-include-pv/--landcover-exclude-pv: PV-only landcover mask.
    # - --landcover-include-onwind/--landcover-exclude-onwind: onwind-only mask.
    # - --luisa-landcover-path: LUISA landuse raster path.
    # - --luisa-include/--luisa-exclude: global LUISA mask (4-digit codes).
    # - --luisa-include-pv/--luisa-exclude-pv: PV-only LUISA mask.
    # - --luisa-include-onwind/--luisa-exclude-onwind: onwind-only LUISA mask.
    # - --gebco-path: bathymetry dataset (GEBCO netCDF, variable "elevation").
    # - --min-depth/--max-depth: offshore depth window in meters (positive
    #   numbers; depth is -elevation). Applies only with --mask-offshore.
    # - --min-shore-distance/--max-shore-distance: offshore distance to coast
    #   window in meters (buffer of country shapes). Applies only with
    #   --mask-offshore.
    # - --urban-distance-onwind: exclusion buffer around urban classes in
    #   meters (applies only to onwind). If --pypsa-defaults is set, defaults
    #   to 1000 m.
    # - --pypsa-defaults: applies PyPSA-style defaults: landcover lists for PV
    #   and onwind (CORINE + LUISA if selected), Natura+ship exclusions, and
    #   offwind depth/shore defaults.
    # - --pypsa-offwind-variant: "ac"/"dc"/"float" use PyPSA-specific shore/depth
    #   rules; "acdc" uses shared constraints (max depth only).
    # - --availability-mode: write availability as raster masks ("raster") or
    #   per-region availability matrices ("regions").
    # - --resource-classes: number of CF bins for availability outputs (default 1).
    # - --resource-class-mode: "global" uses one bin set for all regions;
    #   "per-region" computes bins per region (requires --availability-mode regions).
    # - --resource-class-year: year used to compute resource classes (default start-year).
    # - --clip-p-max-pu: set CF values below this threshold to zero (PV/onwind/offwind CFs).
    # - --capacity-per-sqkm-*: MW/km^2 for p_nom_max in availability outputs.
    # Availability outputs: writes availability_onshore.nc / availability_offshore.nc
    # with masks, optional resource classes, and p_nom_max.
    # - --masks-out-only: skip CF/hydro time series outputs; only masks and
    #   availability outputs are written.
    # - --overwrite: recompute outputs even if they exist.
    #
    # Example (Git Bash, raster/global bins):
    # python ./compute_profiles.py --start-year 1982 --end-year 2024 \
    #   --cutout-dir "//IIP-COMP103/endata/MA_Lisa/atlite_cutouts/cutouts" \
    #   --out-dir "//IIP-COMP103/endata/MA_Eric/res_masked" \
    #   --resources pv onwind offwind \
    #   --mask-onshore --mask-offshore \
    #   --use-excluder-mask --excluder-res-onshore 100 --excluder-res-offshore 200 \
    #   --landuse-dataset both --landuse-fusion prefer-luisa \
    #   --pypsa-defaults --pypsa-offwind-variant acdc \
    #   --exclude-wdpa-offshore --exclude-wdpa-onshore \
    #   --shipdensity-threshold 1e7 \
    #   --availability-mode raster --resource-class-mode global \
    #   --resource-classes 8 --resource-class-year 2024 \
    #   --capacity-per-sqkm-pv 6 --capacity-per-sqkm-onwind 3.5 \
    #   --capacity-per-sqkm-offwind 2.5
    '''
    python ./compute_profiles.py --start-year 1982 --end-year 2024 \
        --cutout-dir "//IIP-COMP103/endata/MA_Lisa/atlite_cutouts/cutouts" \
        --out-dir "//IIP-COMP103/endata/MA_Eric/res_masked" \
        --resources pv onwind offwind \
        --mask-onshore --mask-offshore \
        --use-excluder-mask --excluder-res-onshore 100 --excluder-res-offshore 200 \
        --landuse-dataset both --landuse-fusion prefer-luisa \
        --pypsa-defaults --pypsa-offwind-variant acdc \
        --exclude-wdpa-offshore --exclude-wdpa-onshore \
        --shipdensity-threshold 1e7 \
        --availability-mode raster --resource-class-mode global \
        --resource-classes 8 --resource-class-year 2024 \
        --capacity-per-sqkm-pv 6 --capacity-per-sqkm-onwind 3.5 \
        --capacity-per-sqkm-offwind 2.5
    '''
    
    # Example (Git Bash, regions/per-region bins):
    # python ./compute_profiles.py --start-year 1982 --end-year 2024 \
    #   --cutout-dir "//IIP-COMP103/endata/MA_Lisa/atlite_cutouts/cutouts" \
    #   --out-dir "//IIP-COMP103/endata/MA_Eric/res_masked_regions" \
    #   --resources pv onwind offwind \
    #   --mask-onshore --mask-offshore \
    #   --use-excluder-mask --excluder-res-onshore 100 --excluder-res-offshore 200 \
    #   --landuse-dataset both --landuse-fusion prefer-luisa \
    #   --pypsa-defaults --pypsa-offwind-variant acdc \
    #   --exclude-wdpa-offshore --exclude-wdpa-onshore \
    #   --shipdensity-threshold 1e7 \
    #   --availability-mode regions --resource-class-mode per-region \
    #   --resource-classes 4 --resource-class-year 2024 \
    #   --capacity-per-sqkm-pv 6 --capacity-per-sqkm-onwind 3.5 \
    #   --capacity-per-sqkm-offwind 2.5
    '''
    python ./compute_profiles.py --start-year 1982 --end-year 2024 \
        --cutout-dir "//IIP-COMP103/endata/MA_Lisa/atlite_cutouts/cutouts" \
        --out-dir "//IIP-COMP103/endata/MA_Eric/res_masked_regions" \
        --resources pv onwind offwind \
        --mask-onshore --mask-offshore \
        --use-excluder-mask --excluder-res-onshore 100 --excluder-res-offshore 200 \
        --landuse-dataset both --landuse-fusion prefer-luisa \
        --pypsa-defaults --pypsa-offwind-variant acdc \
        --exclude-wdpa-offshore --exclude-wdpa-onshore \
        --shipdensity-threshold 1e7 \
        --availability-mode regions --resource-class-mode per-region \
        --resource-classes 4 --resource-class-year 2024 \
        --capacity-per-sqkm-pv 6 --capacity-per-sqkm-onwind 3.5 \
        --capacity-per-sqkm-offwind 2.5
    '''

    if args.landcover_include and args.landcover_exclude:
        raise SystemExit("use either --landcover-include or --landcover-exclude (not both)")
    if args.landcover_include_pv and args.landcover_exclude_pv:
        raise SystemExit("use either --landcover-include-pv or --landcover-exclude-pv (not both)")
    if args.landcover_include_onwind and args.landcover_exclude_onwind:
        raise SystemExit("use either --landcover-include-onwind or --landcover-exclude-onwind (not both)")
    if args.luisa_include and args.luisa_exclude:
        raise SystemExit("use either --luisa-include or --luisa-exclude (not both)")
    if args.luisa_include_pv and args.luisa_exclude_pv:
        raise SystemExit("use either --luisa-include-pv or --luisa-exclude-pv (not both)")
    if args.luisa_include_onwind and args.luisa_exclude_onwind:
        raise SystemExit("use either --luisa-include-onwind or --luisa-exclude-onwind (not both)")
    if args.wdpa_iucn_include and args.wdpa_iucn_exclude:
        raise SystemExit("use either --wdpa-iucn-include or --wdpa-iucn-exclude (not both)")
    if args.wdpa_status_include and args.wdpa_status_exclude:
        raise SystemExit("use either --wdpa-status-include or --wdpa-status-exclude (not both)")
    if args.wdpa_designation_include and args.wdpa_designation_exclude:
        raise SystemExit("use either --wdpa-designation-include or --wdpa-designation-exclude (not both)")
    if args.wdpa_designation_type_include and args.wdpa_designation_type_exclude:
        raise SystemExit(
            "use either --wdpa-designation-type-include or --wdpa-designation-type-exclude (not both)"
        )
    if args.resource_classes < 1:
        raise SystemExit("--resource-classes must be >= 1")
    if args.resource_class_mode == "per-region" and args.availability_mode != "regions":
        raise SystemExit("--resource-class-mode per-region requires --availability-mode regions")
    if args.use_excluder_mask:
        if args.excluder_res_onshore <= 0 or args.excluder_res_offshore <= 0:
            raise SystemExit("--excluder-res-onshore/offshore must be > 0")
    if args.wdpa_point_buffer_m < 0:
        raise SystemExit("--wdpa-point-buffer-m must be >= 0")
    if args.clip_p_max_pu is not None and not (0 <= args.clip_p_max_pu <= 1):
        raise SystemExit("--clip-p-max-pu must be between 0 and 1")

    resources = set(args.resources)
    do_all = "all" in resources
    do_pv = do_all or "pv" in resources
    do_onwind = do_all or "onwind" in resources
    do_offwind = do_all or "offwind" in resources
    do_hydro = do_all or "hydro" in resources
    masks_out_only = args.masks_out_only
    if not masks_out_only:
        if do_onwind:
            _validate_windturbine(ONWIND_TURBINE, "onwind")
        if do_offwind:
            _validate_windturbine(OFFWIND_TURBINE, "offwind")
    if masks_out_only:
        LOG.info("masks-out-only enabled: skipping CF/hydro outputs")
        if do_hydro:
            LOG.info("masks-out-only: hydro outputs will be skipped")
            do_hydro = False

    use_corine = args.landuse_dataset in {"corine", "both"}
    use_luisa = args.landuse_dataset in {"luisa", "both"}
    if args.landuse_fusion in {"prefer-corine", "prefer-luisa"} and args.landuse_dataset != "both":
        LOG.warning("--landuse-fusion=%s ignored for --landuse-dataset=%s", args.landuse_fusion, args.landuse_dataset)

    resource_classes = args.resource_classes
    resource_class_year = args.resource_class_year or args.start_year
    if not (args.start_year <= resource_class_year <= args.end_year):
        raise SystemExit("--resource-class-year must be within the selected year range")
    clip_p_max_pu = args.clip_p_max_pu

    capacity_per_sqkm_pv = args.capacity_per_sqkm_pv if do_pv else None
    capacity_per_sqkm_onwind = args.capacity_per_sqkm_onwind if do_onwind else None
    capacity_per_sqkm_offwind = args.capacity_per_sqkm_offwind if do_offwind else None
    if capacity_per_sqkm_pv is not None and capacity_per_sqkm_pv <= 0:
        LOG.warning("disabling PV capacity per sqkm (non-positive value)")
        capacity_per_sqkm_pv = None
    if capacity_per_sqkm_onwind is not None and capacity_per_sqkm_onwind <= 0:
        LOG.warning("disabling onwind capacity per sqkm (non-positive value)")
        capacity_per_sqkm_onwind = None
    if capacity_per_sqkm_offwind is not None and capacity_per_sqkm_offwind <= 0:
        LOG.warning("disabling offwind capacity per sqkm (non-positive value)")
        capacity_per_sqkm_offwind = None
    if args.availability_mode == "regions" and resource_classes > 1:
        LOG.info("resource classes will be computed per %s in regions mode", args.resource_class_mode)

    exclude_natura = args.exclude_natura or args.pypsa_defaults
    exclude_shipdensity = args.exclude_shipdensity or args.pypsa_defaults
    exclude_wdpa_offshore = args.exclude_wdpa_offshore
    exclude_wdpa_onshore = args.exclude_wdpa_onshore

    wdpa_iucn_include = tuple(_normalize_wdpa_values(args.wdpa_iucn_include))
    wdpa_iucn_exclude = tuple(_normalize_wdpa_values(args.wdpa_iucn_exclude))
    wdpa_status_include = tuple(_normalize_wdpa_values(args.wdpa_status_include))
    wdpa_status_exclude = tuple(_normalize_wdpa_values(args.wdpa_status_exclude))
    wdpa_designation_include = tuple(_normalize_wdpa_values(args.wdpa_designation_include))
    wdpa_designation_exclude = tuple(_normalize_wdpa_values(args.wdpa_designation_exclude))
    wdpa_designation_type_include = tuple(
        _normalize_wdpa_values(args.wdpa_designation_type_include)
    )
    wdpa_designation_type_exclude = tuple(
        _normalize_wdpa_values(args.wdpa_designation_type_exclude)
    )

    wdpa_onshore_filter = WDPA_ONSHORE_STRICT_FILTER if exclude_wdpa_onshore else "all"
    wdpa_onshore_iucn_include = WDPA_ONSHORE_STRICT_IUCN if exclude_wdpa_onshore else ()
    wdpa_onshore_iucn_exclude = ()
    wdpa_onshore_status_include = WDPA_ONSHORE_STRICT_STATUS if exclude_wdpa_onshore else ()
    wdpa_onshore_status_exclude = ()
    wdpa_onshore_designation_include = ()
    wdpa_onshore_designation_exclude = ()
    wdpa_onshore_designation_type_include = ()
    wdpa_onshore_designation_type_exclude = ()

    shipdensity_threshold = args.shipdensity_threshold
    if shipdensity_threshold is None:
        shipdensity_threshold = (
            PYPSA_SHIP_THRESHOLD * PYPSA_SHIP_HOURS
            if args.pypsa_defaults
            else SHIPDENSITY_THRESHOLD
        )

    min_depth = args.min_depth
    max_depth = args.max_depth
    min_shore_distance = args.min_shore_distance
    max_shore_distance = args.max_shore_distance
    urban_distance_onwind = args.urban_distance_onwind
    if args.pypsa_defaults and do_offwind:
        defaults = PYPSA_OFFWIND_DEFAULTS.get(args.pypsa_offwind_variant, {})
        if min_depth is None:
            min_depth = defaults.get("min_depth")
        if max_depth is None:
            max_depth = defaults.get("max_depth")
        if min_shore_distance is None:
            min_shore_distance = defaults.get("min_shore_distance")
        if max_shore_distance is None:
            max_shore_distance = defaults.get("max_shore_distance")
    if urban_distance_onwind is None and args.pypsa_defaults and do_onwind:
        urban_distance_onwind = PYPSA_ONWIND_URBAN_DISTANCE_M
    if urban_distance_onwind is not None and urban_distance_onwind <= 0:
        urban_distance_onwind = None

    if min_depth is not None and max_depth is not None and min_depth > max_depth:
        raise SystemExit("min depth cannot be greater than max depth")
    if (
        min_shore_distance is not None
        and max_shore_distance is not None
        and min_shore_distance > max_shore_distance
    ):
        raise SystemExit("min shore distance cannot be greater than max shore distance")

    landcover_pv = LandcoverConfig(None, None, "", "", "")
    landcover_onwind = LandcoverConfig(None, None, "", "", "")
    luisa_pv = LandcoverConfig(None, None, "", "", "")
    luisa_onwind = LandcoverConfig(None, None, "", "", "")

    corine_args_used = any(
        [
            args.landcover_include,
            args.landcover_exclude,
            args.landcover_include_pv,
            args.landcover_exclude_pv,
            args.landcover_include_onwind,
            args.landcover_exclude_onwind,
        ]
    )
    luisa_args_used = any(
        [
            args.luisa_include,
            args.luisa_exclude,
            args.luisa_include_pv,
            args.luisa_exclude_pv,
            args.luisa_include_onwind,
            args.luisa_exclude_onwind,
        ]
    )

    if use_corine:
        global_include = set(args.landcover_include) if args.landcover_include else None
        global_exclude = set(args.landcover_exclude) if args.landcover_exclude else None
        pv_include_raw, pv_exclude_raw = _select_landcover_raw(
            set(args.landcover_include_pv) if args.landcover_include_pv else None,
            set(args.landcover_exclude_pv) if args.landcover_exclude_pv else None,
            global_include,
            global_exclude,
            PYPSA_SOLAR_CLC_CODES if args.pypsa_defaults and do_pv else None,
        )
        onwind_include_raw, onwind_exclude_raw = _select_landcover_raw(
            set(args.landcover_include_onwind) if args.landcover_include_onwind else None,
            set(args.landcover_exclude_onwind) if args.landcover_exclude_onwind else None,
            global_include,
            global_exclude,
            PYPSA_ONWIND_CLC_CODES if args.pypsa_defaults and do_onwind else None,
        )
        landcover_pv = _make_landcover_config(pv_include_raw, pv_exclude_raw)
        landcover_onwind = _make_landcover_config(onwind_include_raw, onwind_exclude_raw)
    elif corine_args_used:
        LOG.warning("CORINE landcover options ignored for --landuse-dataset=%s", args.landuse_dataset)

    if use_luisa:
        global_include = set(args.luisa_include) if args.luisa_include else None
        global_exclude = set(args.luisa_exclude) if args.luisa_exclude else None
        pv_include_raw, pv_exclude_raw = _select_landcover_raw(
            set(args.luisa_include_pv) if args.luisa_include_pv else None,
            set(args.luisa_exclude_pv) if args.luisa_exclude_pv else None,
            global_include,
            global_exclude,
            PYPSA_LUISA_SOLAR_CODES if args.pypsa_defaults and do_pv else None,
        )
        onwind_include_raw, onwind_exclude_raw = _select_landcover_raw(
            set(args.luisa_include_onwind) if args.luisa_include_onwind else None,
            set(args.luisa_exclude_onwind) if args.luisa_exclude_onwind else None,
            global_include,
            global_exclude,
            PYPSA_LUISA_ONWIND_CODES if args.pypsa_defaults and do_onwind else None,
        )
        luisa_pv = _make_luisa_config(pv_include_raw, pv_exclude_raw, args.luisa_legend)
        luisa_onwind = _make_luisa_config(
            onwind_include_raw, onwind_exclude_raw, args.luisa_legend
        )
    elif luisa_args_used:
        LOG.warning("LUISA landuse options ignored for --landuse-dataset=%s", args.landuse_dataset)

    if (
        landcover_pv.active
        or landcover_onwind.active
        or luisa_pv.active
        or luisa_onwind.active
    ) and not args.mask_onshore:
        LOG.warning("landcover mask requested without --mask-onshore; mask will be ignored")
    if urban_distance_onwind is not None and not args.mask_onshore:
        LOG.warning("urban distance mask requested without --mask-onshore; mask will be ignored")
    if (
        min_depth is not None
        or max_depth is not None
        or min_shore_distance is not None
        or max_shore_distance is not None
    ) and not args.mask_offshore:
        LOG.warning("offshore distance/depth mask requested without --mask-offshore; mask will be ignored")
    if exclude_shipdensity and not args.mask_offshore:
        LOG.warning("shipdensity mask requested without --mask-offshore; mask will be ignored")
    if exclude_wdpa_offshore and not args.mask_offshore:
        LOG.warning("WDPA mask requested without --mask-offshore; mask will be ignored")
    if exclude_wdpa_onshore and not args.mask_onshore:
        LOG.warning("WDPA onshore mask requested without --mask-onshore; mask will be ignored")
    wdpa_filters_requested = any(
        [
            wdpa_iucn_include,
            wdpa_iucn_exclude,
            wdpa_status_include,
            wdpa_status_exclude,
            wdpa_designation_include,
            wdpa_designation_exclude,
            wdpa_designation_type_include,
            wdpa_designation_type_exclude,
        ]
    )
    if wdpa_filters_requested and not exclude_wdpa_offshore:
        LOG.warning("WDPA filters provided without --exclude-wdpa-offshore; filters ignored")
    if exclude_natura and not (args.mask_onshore or args.mask_offshore):
        LOG.warning("natura mask requested without --mask-onshore/--mask-offshore; mask will be ignored")
    if args.use_excluder_mask and not (args.mask_onshore or args.mask_offshore):
        LOG.warning("excluder mask requested without --mask-onshore/--mask-offshore; mask will be ignored")

    p_inst = bus_country = None
    hydro_scale = None
    if do_hydro:
        p_inst, bus_country = _load_hydro_static(
            NETWORK_DIR / "plants.csv", NETWORK_DIR / "buses.csv"
        )
        if args.hydro_normalize:
            hydro_scale = _compute_hydro_scale(
                cutout_dir=args.cutout_dir,
                reference_year=args.hydro_normalize_year,
                country_shapes_path=args.country_shapes,
                reference_path=args.hydro_reference,
                hydro_reductions_path=args.hydro_reductions_path,
                bus_country=bus_country,
            )

    mask_cache: dict[object, xr.DataArray | None] = {
        "onshore_base": None,
        "offshore_base": None,
        "bathymetry": None,
        "shore_distance": None,
    }
    excluder_cache: dict[tuple, atlite.ExclusionContainer] = {}
    availability_state = {"onshore": False, "offshore": False}
    for year in range(args.start_year, args.end_year + 1):
        LOG.info("processing %d", year)
        compute_year(
            year=year,
            cutout_dir=args.cutout_dir,
            out_dir=args.out_dir,
            do_pv=do_pv,
            do_onwind=do_onwind,
            do_offwind=do_offwind,
            do_hydro=do_hydro,
            p_inst=p_inst,
            bus_country=bus_country,
            hydro_scale=hydro_scale,
            hydro_reference=args.hydro_reference,
            hydro_reference_year=args.hydro_normalize_year,
            hydro_reductions_path=args.hydro_reductions_path,
            hydro_write_bus_profiles=args.hydro_write_bus_profiles,
            mask_onshore=args.mask_onshore,
            mask_offshore=args.mask_offshore,
            onshore_mask_shapes=list(args.onshore_mask_shapes),
            offshore_mask_shapes=list(args.offshore_mask_shapes),
            onshore_regions_path=args.onshore_regions,
            offshore_regions_path=args.offshore_regions,
            country_shapes_path=args.country_shapes,
            exclude_natura=exclude_natura,
            natura_path=args.natura_path,
            exclude_shipdensity=exclude_shipdensity,
            shipdensity_path=args.shipdensity_path,
            shipdensity_threshold=shipdensity_threshold,
            exclude_wdpa_offshore=exclude_wdpa_offshore,
            wdpa_polygons=list(args.wdpa_polygons),
            wdpa_points=list(args.wdpa_points),
            wdpa_filter=args.wdpa_filter,
            wdpa_point_buffer_m=args.wdpa_point_buffer_m,
            wdpa_iucn_include=wdpa_iucn_include,
            wdpa_iucn_exclude=wdpa_iucn_exclude,
            wdpa_status_include=wdpa_status_include,
            wdpa_status_exclude=wdpa_status_exclude,
            wdpa_designation_include=wdpa_designation_include,
            wdpa_designation_exclude=wdpa_designation_exclude,
            wdpa_designation_type_include=wdpa_designation_type_include,
            wdpa_designation_type_exclude=wdpa_designation_type_exclude,
            exclude_wdpa_onshore=exclude_wdpa_onshore,
            wdpa_onshore_filter=wdpa_onshore_filter,
            wdpa_onshore_iucn_include=wdpa_onshore_iucn_include,
            wdpa_onshore_iucn_exclude=wdpa_onshore_iucn_exclude,
            wdpa_onshore_status_include=wdpa_onshore_status_include,
            wdpa_onshore_status_exclude=wdpa_onshore_status_exclude,
            wdpa_onshore_designation_include=wdpa_onshore_designation_include,
            wdpa_onshore_designation_exclude=wdpa_onshore_designation_exclude,
            wdpa_onshore_designation_type_include=wdpa_onshore_designation_type_include,
            wdpa_onshore_designation_type_exclude=wdpa_onshore_designation_type_exclude,
            landuse_dataset=args.landuse_dataset,
            landuse_fusion=args.landuse_fusion,
            CORINE_LANDCOVER_PATH=args.corine_landcover_path,
            landcover_pv=landcover_pv,
            landcover_onwind=landcover_onwind,
            LUISA_LANDCOVER_PATH=args.luisa_landcover_path,
            luisa_legend_path=args.luisa_legend,
            luisa_pv=luisa_pv,
            luisa_onwind=luisa_onwind,
            gebco_path=args.gebco_path,
            min_depth=min_depth,
            max_depth=max_depth,
            min_shore_distance=min_shore_distance,
            max_shore_distance=max_shore_distance,
            urban_distance_onwind=urban_distance_onwind,
            use_excluder_mask=args.use_excluder_mask,
            excluder_res_onshore=args.excluder_res_onshore,
            excluder_res_offshore=args.excluder_res_offshore,
            excluder_cache=excluder_cache,
            resource_classes=resource_classes,
            resource_class_year=resource_class_year,
            capacity_per_sqkm_pv=capacity_per_sqkm_pv,
            capacity_per_sqkm_onwind=capacity_per_sqkm_onwind,
            capacity_per_sqkm_offwind=capacity_per_sqkm_offwind,
            clip_p_max_pu=clip_p_max_pu,
            availability_mode=args.availability_mode,
            resource_class_mode=args.resource_class_mode,
            availability_state=availability_state,
            mask_cache=mask_cache,
            overwrite=args.overwrite,
            masks_out_only=masks_out_only,
        )


if __name__ == "__main__":
    main()
