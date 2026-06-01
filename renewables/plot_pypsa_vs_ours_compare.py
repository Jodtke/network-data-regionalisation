from __future__ import annotations

"""Compare generated renewable profiles and potentials with PyPSA-style inputs.

The plots produced here are diagnostic rather than part of the optimisation
input. They help identify whether differences between the local Atlite-like
workflow and the reference PyPSA-Eur preparation come from masks, aggregation
regions, capacity densities, or weather-resource treatment.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from res_common import (
    DEFAULT_END_YEAR,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_START_YEAR,
    build_simulation_case_dir,
    compute_distance_to_shapes_km,
    ensure_dir,
    parse_optional_float,
    parse_threshold_value,
    resolve_cli_path,
    resolve_path,
)

LOG = logging.getLogger(__name__)

SEA_COLOR = "#cfe8f6"
LAND_COLOR = "#c4be92"
COUNTRY_EDGE_COLOR = "#8a8164"
DEFAULT_PLOT_BOUNDS = (-11.5, 34.0, 41.5, 72.5)

TECH_CONFIG = {
    "pv": {
        "label": "PV",
        "ours_capacity_relpath": Path("availability_onshore.nc"),
        "ours_capacity_var": "p_nom_max_pv",
        "ours_availability_var": "availability_pv",
        "ours_cf_relpath_tpl": Path("pv") / "pv_cf_{year}.nc",
        "pypsa_capacity_relpath": Path("renewables") / "pypsa_extracted" / "pv" / "pv_raster_capacities.nc",
        "pypsa_cf_relpath": Path("renewables") / "pypsa_extracted" / "pv" / "pv_raster_profiles.nc",
        "capacity_cmap": "YlOrRd",
        "cf_cmap": "viridis",
    },
    "onwind": {
        "label": "Wind Onshore",
        "ours_capacity_relpath": Path("availability_onshore.nc"),
        "ours_capacity_var": "p_nom_max_onwind",
        "ours_availability_var": "availability_onwind",
        "ours_cf_relpath_tpl": Path("onwind") / "onwind_cf_{year}.nc",
        "pypsa_capacity_relpath": Path("renewables") / "pypsa_extracted" / "onwind" / "onwind_raster_capacities.nc",
        "pypsa_cf_relpath": Path("renewables") / "pypsa_extracted" / "onwind" / "onwind_raster_profiles.nc",
        "capacity_cmap": "PuRd",
        "cf_cmap": "viridis",
    },
    "offwind": {
        "label": "Wind Offshore",
        "ours_capacity_relpath": Path("availability_offshore.nc"),
        "ours_capacity_var": "p_nom_max_offwind",
        "ours_availability_var": "availability_offwind",
        "ours_cf_relpath_tpl": Path("offwind") / "offwind_cf_{year}.nc",
        "pypsa_capacity_relpath": Path("renewables") / "pypsa_extracted" / "offwind" / "offwind_raster_capacities.nc",
        "pypsa_cf_relpath": Path("renewables") / "pypsa_extracted" / "offwind" / "offwind_raster_profiles.nc",
        "capacity_cmap": "Blues",
        "cf_cmap": "cividis",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyPSA extracted raster maps against our raster maps."
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--atlite-case-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--simulation-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--network-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--weather-year", type=int, default=2013)
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--min-p-max-pu", default=None)
    parser.add_argument("--min-p-nom-max", default=None)
    parser.add_argument("--min-distance-offshore-km", type=float, default=None)
    parser.add_argument("--max-distance-offshore-km", type=float, default=None)
    parser.add_argument(
        "--offshore-distance-geojson",
        type=Path,
        action="append",
        default=None,
        help="Land/coastline shape for offshore distance filtering. Defaults to europe_shape_geojson.",
    )
    parser.add_argument(
        "--plot-bounds",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=DEFAULT_PLOT_BOUNDS,
        help="Map extent in lon/lat. Default clips remote European islands.",
    )
    parser.add_argument(
        "--technologies",
        nargs="+",
        choices=sorted(TECH_CONFIG),
        default=["pv", "onwind", "offwind"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--country-shapes-geojson",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--europe-shape-geojson",
        type=Path,
        default=None,
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def threshold_for_technology(value: Any, technology: str) -> float:
    parsed = parse_threshold_value(value)
    raw_value = parsed
    if isinstance(parsed, dict):
        raw_value = parsed.get(technology, parsed.get("default", 0.0))
    if raw_value in (None, ""):
        return 0.0
    return float(raw_value)


def load_background(country_shapes: Path, europe_shape: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    countries = gpd.read_file(country_shapes).to_crs("EPSG:4326")
    europe = gpd.read_file(europe_shape).to_crs("EPSG:4326")
    return countries, europe


def raster_coords(ds: xr.Dataset | xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    lat_values = np.asarray(ds["lat"].values if "lat" in ds.coords else ds["y"].values)
    lon_values = np.asarray(ds["lon"].values if "lon" in ds.coords else ds["x"].values)
    if lat_values.ndim == 1 and lon_values.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
    else:
        lon_grid, lat_grid = lon_values, lat_values
    return lon_grid, lat_grid


def setup_ax(
    ax: plt.Axes,
    countries: gpd.GeoDataFrame,
    europe: gpd.GeoDataFrame,
    plot_bounds: tuple[float, float, float, float] | None,
) -> None:
    ax.set_facecolor(SEA_COLOR)
    countries.plot(ax=ax, color=LAND_COLOR, edgecolor=COUNTRY_EDGE_COLOR, linewidth=0.35, zorder=1)
    if plot_bounds is None:
        minx, miny, maxx, maxy = europe.total_bounds
        dx = (maxx - minx) * 0.03
        dy = (maxy - miny) * 0.03
        ax.set_xlim(minx - dx, maxx + dx)
        ax.set_ylim(miny - dy, maxy + dy)
    else:
        min_lon, min_lat, max_lon, max_lat = plot_bounds
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def normalize_cf_dataarray(cf: xr.DataArray) -> xr.DataArray:
    if {"time", "y", "x"}.issubset(set(cf.dims)):
        return cf.transpose("time", "y", "x")
    raise ValueError(f"Unsupported CF dimensions: {cf.dims}")


def load_filter_mean_cf(
    atlite_case_dir: Path,
    technology: str,
    start_year: int,
    end_year: int,
) -> np.ndarray:
    cfg = TECH_CONFIG[technology]
    sums: np.ndarray | None = None
    counts: np.ndarray | None = None
    found_any = False
    for weather_year in range(start_year, end_year + 1):
        cf_path = atlite_case_dir / Path(str(cfg["ours_cf_relpath_tpl"]).format(year=weather_year))
        if not cf_path.exists():
            LOG.warning("missing CF file for %s %s during plot filtering", technology, weather_year)
            continue
        found_any = True
        with xr.open_dataset(cf_path, engine="netcdf4") as ds:
            mean_cf = np.asarray(normalize_cf_dataarray(ds["cf"]).mean(dim="time", skipna=True).values, dtype=float)
        if sums is None:
            sums = np.zeros_like(mean_cf, dtype=float)
            counts = np.zeros_like(mean_cf, dtype=np.int64)
        valid = np.isfinite(mean_cf)
        sums += np.where(valid, mean_cf, 0.0)
        counts += valid.astype(np.int64)
    if not found_any or sums is None or counts is None:
        raise FileNotFoundError(
            f"No CF files found for {technology} in {atlite_case_dir} for years {start_year}-{end_year}."
        )
    return np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=counts > 0)


def build_ours_filter_mask(
    atlite_case_dir: Path,
    technology: str,
    p_nom_max: np.ndarray,
    availability: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    *,
    min_p_nom_max: float,
    min_p_max_pu: float,
    min_distance_offshore_km: float | None,
    max_distance_offshore_km: float | None,
    offshore_distance_shape_paths: list[Path],
    start_year: int,
    end_year: int,
) -> np.ndarray:
    mask = np.isfinite(availability) & (availability > 0.0) & np.isfinite(p_nom_max) & (p_nom_max > 0.0)
    base_cells = int(mask.sum())
    if min_p_nom_max > 0.0:
        mask &= p_nom_max >= min_p_nom_max
    if min_p_max_pu > 0.0:
        mean_cf = load_filter_mean_cf(atlite_case_dir, technology, start_year, end_year)
        mask &= np.isfinite(mean_cf) & (mean_cf >= min_p_max_pu)
    if technology == "offwind" and (
        min_distance_offshore_km is not None or max_distance_offshore_km is not None
    ):
        distance_cells = pd.DataFrame(
            {
                "lon": np.asarray(lon_grid, dtype=float).reshape(-1),
                "lat": np.asarray(lat_grid, dtype=float).reshape(-1),
            }
        )
        distances = compute_distance_to_shapes_km(distance_cells, offshore_distance_shape_paths).to_numpy().reshape(
            p_nom_max.shape
        )
        if min_distance_offshore_km is not None:
            mask &= distances >= min_distance_offshore_km
        if max_distance_offshore_km is not None:
            mask &= distances <= max_distance_offshore_km
    LOG.info(
        (
            "plot filter for %s kept %d/%d raw available cells "
            "(min_p_nom_max=%g, min_p_max_pu=%g, min_distance_offshore_km=%s, max_distance_offshore_km=%s)"
        ),
        technology,
        int(mask.sum()),
        base_cells,
        min_p_nom_max,
        min_p_max_pu,
        min_distance_offshore_km,
        max_distance_offshore_km,
    )
    return mask


def load_ours_capacity(
    atlite_case_dir: Path,
    technology: str,
    *,
    min_p_nom_max: float,
    min_p_max_pu: float,
    min_distance_offshore_km: float | None,
    max_distance_offshore_km: float | None,
    offshore_distance_shape_paths: list[Path],
    start_year: int,
    end_year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = TECH_CONFIG[technology]
    with xr.open_dataset(atlite_case_dir / cfg["ours_capacity_relpath"], engine="netcdf4") as ds:
        values = np.asarray(ds[cfg["ours_capacity_var"]].values, dtype=float)
        availability = np.asarray(ds[cfg["ours_availability_var"]].values, dtype=float)
        lon_grid, lat_grid = raster_coords(ds)
    mask = build_ours_filter_mask(
        atlite_case_dir,
        technology,
        values,
        availability,
        lon_grid,
        lat_grid,
        min_p_nom_max=min_p_nom_max,
        min_p_max_pu=min_p_max_pu,
        min_distance_offshore_km=min_distance_offshore_km,
        max_distance_offshore_km=max_distance_offshore_km,
        offshore_distance_shape_paths=offshore_distance_shape_paths,
        start_year=start_year,
        end_year=end_year,
    )
    return lon_grid, lat_grid, np.where(mask, values, np.nan), mask


def load_ours_cf_mean(
    atlite_case_dir: Path,
    technology: str,
    weather_year: int,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = TECH_CONFIG[technology]
    with xr.open_dataset(
        atlite_case_dir / Path(str(cfg["ours_cf_relpath_tpl"]).format(year=weather_year)),
        engine="netcdf4",
    ) as ds:
        mean_cf = normalize_cf_dataarray(ds["cf"]).mean(dim="time", skipna=True)
        lon_grid, lat_grid = raster_coords(mean_cf)
        values = np.asarray(mean_cf.values, dtype=float)
    if mask is not None:
        values = np.where(mask, values, np.nan)
    return lon_grid, lat_grid, np.where(np.isfinite(values), values, np.nan)


def load_pypsa_capacity(project_root: Path, technology: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = TECH_CONFIG[technology]
    with xr.open_dataset(project_root / cfg["pypsa_capacity_relpath"], engine="netcdf4") as ds:
        values = np.asarray(ds["p_nom_max"].values, dtype=float)
        lon_grid, lat_grid = raster_coords(ds)
    return lon_grid, lat_grid, np.where(np.isfinite(values) & (values > 0.0), values, np.nan)


def load_pypsa_cf_mean(project_root: Path, technology: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = TECH_CONFIG[technology]
    with xr.open_dataset(project_root / cfg["pypsa_cf_relpath"], engine="netcdf4") as ds:
        mean_cf = normalize_cf_dataarray(ds["cf"]).mean(dim="time", skipna=True)
        lon_grid, lat_grid = raster_coords(mean_cf)
        values = np.asarray(mean_cf.values, dtype=float)
    return lon_grid, lat_grid, np.where(np.isfinite(values), values, np.nan)


def combined_vmax(*arrays: np.ndarray) -> float:
    finite_max = [float(np.nanmax(arr)) for arr in arrays if np.isfinite(arr).any()]
    return max(finite_max) if finite_max else 1.0


def crop_to_bounds(
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    values: np.ndarray,
    plot_bounds: tuple[float, float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if plot_bounds is None:
        return lon_grid, lat_grid, values

    min_lon, min_lat, max_lon, max_lat = plot_bounds
    inside = (lon_grid >= min_lon) & (lon_grid <= max_lon) & (lat_grid >= min_lat) & (lat_grid <= max_lat)
    if not inside.any():
        return lon_grid, lat_grid, values

    rows = np.where(inside.any(axis=1))[0]
    cols = np.where(inside.any(axis=0))[0]
    row_slice = slice(int(rows.min()), int(rows.max()) + 1)
    col_slice = slice(int(cols.min()), int(cols.max()) + 1)
    return lon_grid[row_slice, col_slice], lat_grid[row_slice, col_slice], values[row_slice, col_slice]


def plot_technology_compare(
    project_root: Path,
    atlite_case_dir: Path,
    technology: str,
    weather_year: int,
    countries: gpd.GeoDataFrame,
    europe: gpd.GeoDataFrame,
    output_dir: Path,
    dpi: int,
    start_year: int,
    end_year: int,
    min_p_max_pu: Any,
    min_p_nom_max: Any,
    min_distance_offshore_km: float | None,
    max_distance_offshore_km: float | None,
    offshore_distance_shape_paths: list[Path],
    plot_bounds: tuple[float, float, float, float] | None,
) -> Path:
    cfg = TECH_CONFIG[technology]
    min_p_nom_max_value = threshold_for_technology(min_p_nom_max, technology)
    min_p_max_pu_value = threshold_for_technology(min_p_max_pu, technology)
    pypsa_lon_cap, pypsa_lat_cap, pypsa_cap = load_pypsa_capacity(project_root, technology)
    ours_lon_cap, ours_lat_cap, ours_cap, ours_mask = load_ours_capacity(
        atlite_case_dir,
        technology,
        min_p_nom_max=min_p_nom_max_value,
        min_p_max_pu=min_p_max_pu_value,
        min_distance_offshore_km=min_distance_offshore_km,
        max_distance_offshore_km=max_distance_offshore_km,
        offshore_distance_shape_paths=offshore_distance_shape_paths,
        start_year=start_year,
        end_year=end_year,
    )
    pypsa_lon_cf, pypsa_lat_cf, pypsa_cf = load_pypsa_cf_mean(project_root, technology)
    ours_lon_cf, ours_lat_cf, ours_cf = load_ours_cf_mean(atlite_case_dir, technology, weather_year, ours_mask)

    pypsa_lon_cap, pypsa_lat_cap, pypsa_cap = crop_to_bounds(
        pypsa_lon_cap,
        pypsa_lat_cap,
        pypsa_cap,
        plot_bounds,
    )
    ours_lon_cap, ours_lat_cap, ours_cap = crop_to_bounds(
        ours_lon_cap,
        ours_lat_cap,
        ours_cap,
        plot_bounds,
    )
    pypsa_lon_cf, pypsa_lat_cf, pypsa_cf = crop_to_bounds(
        pypsa_lon_cf,
        pypsa_lat_cf,
        pypsa_cf,
        plot_bounds,
    )
    ours_lon_cf, ours_lat_cf, ours_cf = crop_to_bounds(
        ours_lon_cf,
        ours_lat_cf,
        ours_cf,
        plot_bounds,
    )

    cap_vmax = combined_vmax(pypsa_cap, ours_cap)
    cf_vmax = combined_vmax(pypsa_cf, ours_cf)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    for ax in axes.flat:
        setup_ax(ax, countries, europe, plot_bounds)

    mesh_cap_left = axes[0, 0].pcolormesh(
        pypsa_lon_cap,
        pypsa_lat_cap,
        pypsa_cap,
        shading="auto",
        cmap=cfg["capacity_cmap"],
        vmin=0.0,
        vmax=cap_vmax,
        zorder=3,
    )
    axes[0, 0].set_title(f"PyPSA extracted {cfg['label']} p_nom_max")

    mesh_cap_right = axes[0, 1].pcolormesh(
        ours_lon_cap,
        ours_lat_cap,
        ours_cap,
        shading="auto",
        cmap=cfg["capacity_cmap"],
        vmin=0.0,
        vmax=cap_vmax,
        zorder=3,
    )
    axes[0, 1].set_title(f"Our filtered raster {cfg['label']} p_nom_max")

    mesh_cf_left = axes[1, 0].pcolormesh(
        pypsa_lon_cf,
        pypsa_lat_cf,
        pypsa_cf,
        shading="auto",
        cmap=cfg["cf_cmap"],
        vmin=0.0,
        vmax=cf_vmax,
        zorder=3,
    )
    axes[1, 0].set_title(f"PyPSA extracted {cfg['label']} mean CF {weather_year}")

    mesh_cf_right = axes[1, 1].pcolormesh(
        ours_lon_cf,
        ours_lat_cf,
        ours_cf,
        shading="auto",
        cmap=cfg["cf_cmap"],
        vmin=0.0,
        vmax=cf_vmax,
        zorder=3,
    )
    axes[1, 1].set_title(f"Our filtered raster {cfg['label']} mean CF {weather_year}")

    cbar_cap = fig.colorbar(mesh_cap_right, ax=axes[0, :], shrink=0.92, pad=0.02)
    cbar_cap.set_label("Max installable capacity [MW per cell]")
    cbar_cf = fig.colorbar(mesh_cf_right, ax=axes[1, :], shrink=0.92, pad=0.02)
    cbar_cf.set_label("Mean annual capacity factor [-]")

    filter_label = (
        f"filters: min_p_nom_max={min_p_nom_max_value:g}, "
        f"min_p_max_pu={min_p_max_pu_value:g}, "
        f"offshore_distance={min_distance_offshore_km}-{max_distance_offshore_km} km ({start_year}-{end_year})"
    )
    fig.suptitle(
        f"{cfg['label']} comparison: PyPSA extracted vs our filtered raster ({weather_year}; {filter_label})",
        fontsize=15,
    )
    out_path = output_dir / f"{technology}_pypsa_vs_ours_compare_{weather_year}.svg"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    project_root = resolve_cli_path(args.project_root) or DEFAULT_PROJECT_ROOT
    atlite_case_dir = (resolve_path(args.atlite_case_dir, base_dir=project_root) or args.atlite_case_dir).resolve()
    simulation_dir = (
        (resolve_path(args.simulation_dir, base_dir=project_root) or args.simulation_dir).resolve()
        if args.simulation_dir is not None
        else (project_root / "renewables")
    )
    network_dir = (
        (resolve_path(args.network_dir, base_dir=project_root) or args.network_dir).resolve()
        if args.network_dir is not None
        else None
    )
    if args.output_dir is not None:
        output_dir = ensure_dir((resolve_path(args.output_dir, base_dir=project_root) or args.output_dir).resolve())
    elif network_dir is not None:
        output_dir = ensure_dir(
            build_simulation_case_dir(simulation_dir, network_dir, atlite_case_dir) / "plot_comparison"
        )
    else:
        output_dir = ensure_dir(simulation_dir / f"res_{atlite_case_dir.name}" / "plot_comparison")
    country_shapes_geojson = (
        resolve_path(args.country_shapes_geojson, base_dir=project_root)
        or (project_root / "datashapes" / "country_shapes.geojson")
    )
    europe_shape_geojson = (
        resolve_path(args.europe_shape_geojson, base_dir=project_root)
        or (project_root / "datashapes" / "europe_shape.geojson")
    )
    countries, europe = load_background(country_shapes_geojson, europe_shape_geojson)
    plot_bounds = tuple(args.plot_bounds) if args.plot_bounds is not None else None
    min_p_max_pu = parse_threshold_value(args.min_p_max_pu)
    min_p_nom_max = parse_threshold_value(args.min_p_nom_max)
    min_distance_offshore_km = parse_optional_float(
        args.min_distance_offshore_km,
        "min_distance_offshore_km",
    )
    max_distance_offshore_km = parse_optional_float(
        args.max_distance_offshore_km,
        "max_distance_offshore_km",
    )
    if (
        min_distance_offshore_km is not None
        and max_distance_offshore_km is not None
        and min_distance_offshore_km > max_distance_offshore_km
    ):
        raise ValueError("min_distance_offshore_km must be <= max_distance_offshore_km.")
    offshore_distance_shape_paths = (
        [
            (resolve_path(path, base_dir=project_root) or resolve_cli_path(path) or path).resolve()
            for path in args.offshore_distance_geojson
        ]
        if args.offshore_distance_geojson
        else [europe_shape_geojson]
    )

    for technology in args.technologies:
        LOG.info("plotting comparison for %s", technology)
        out_path = plot_technology_compare(
            project_root,
            atlite_case_dir,
            technology,
            args.weather_year,
            countries,
            europe,
            output_dir,
            args.dpi,
            args.start_year,
            args.end_year,
            min_p_max_pu,
            min_p_nom_max,
            min_distance_offshore_km,
            max_distance_offshore_km,
            offshore_distance_shape_paths,
            plot_bounds,
        )
        LOG.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
