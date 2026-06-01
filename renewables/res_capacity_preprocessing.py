from __future__ import annotations

"""Prepare renewable capacity potentials and bus-level scenario capacities.

The script links three data layers that are kept separate upstream: raster
potentials from the Atlite-like preprocessing, current renewable sites from the
plant database, and national TYNDP capacity targets. Current capacity is first
placed back into eligible raster cells assigned to each bus. Additional scenario
capacity is then filled by investment resource class, which gives preference to
better weather-resource cells without hard-coding a single best-cell ranking.

The resulting files contain cell and bus capacities, diagnostics on saturated
potentials, and the capacity basis later used to scale renewable generation
profiles to TYNDP weather-year generation.
"""

import argparse
import logging
import math
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from res_common import (
    DEFAULT_END_YEAR,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_START_YEAR,
    CountryClusterMap,
    aggregate_current_capacity_by_bus,
    apply_offshore_distance_filter,
    assign_points_to_polygons,
    build_bus_lookup,
    build_onshore_voronoi,
    build_simulation_case_dir,
    convert_capacity_series_to_mw,
    derive_network_country_map,
    detect_delimiter,
    ensure_dir,
    label_for_country,
    load_plants,
    load_reduced_buses,
    load_yaml_config,
    map_country_code,
    merge_country_cluster_maps,
    normalize_country_name_or_code,
    parse_optional_float,
    parse_threshold_value,
    read_country_cluster_map,
    read_excluded_countries,
    resolve_cli_path,
    resolve_path,
    write_json,
    write_onshore_voronoi,
)

LOG = logging.getLogger(__name__)
DEFAULT_ATLITE_CASE_DIR = (
    DEFAULT_PROJECT_ROOT / "renewables" / "atlite_copy" / "corine_luisa_wdpa_onoff_acdc"
)
TYNDP_TARGET_YEARS = (2030, 2040, 2050)
TYNDP_YEAR_PATTERN = re.compile(r"(?<!\d)(?:2030|2040|2050)(?!\d)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build scenario-specific RES raster and bus capacities from global Atlite outputs."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--network-dir", type=Path, default=None)
    parser.add_argument("--simulation-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--atlite-case-dir", type=Path, default=None)
    parser.add_argument("--availability-onshore-nc", type=Path, default=None)
    parser.add_argument("--availability-offshore-nc", type=Path, default=None)
    parser.add_argument("--target-capacity-csv", type=Path, default=None)
    parser.add_argument("--pv-power-csv", type=Path, default=None)
    parser.add_argument("--onwind-power-csv", type=Path, default=None)
    parser.add_argument("--offwind-power-csv", type=Path, default=None)
    parser.add_argument("--weather-root", type=Path, default=None)
    parser.add_argument("--buses-csv", type=Path, default=None)
    parser.add_argument("--plants-csv", type=Path, default=None)
    parser.add_argument("--country-clusters-csv", type=Path, default=None)
    parser.add_argument("--excluded-countries-csv", type=Path, default=None)
    parser.add_argument("--onshore-voronoi-geojson", type=Path, default=None)
    parser.add_argument("--generated-onshore-voronoi-geojson", type=Path, default=None)
    parser.add_argument("--offshore-eez-geojson", type=Path, default=None)
    parser.add_argument("--onshore-mask-geojson", action="append", default=None)
    parser.add_argument("--target-year", type=int, default=None)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--tyndp-scenario", default=None)
    parser.add_argument("--tyndp-capacity-unit", default=None)
    parser.add_argument("--min-p-max-pu", default=None)
    parser.add_argument("--min-p-nom-max", default=None)
    parser.add_argument("--min-distance-offshore-km", type=float, default=None)
    parser.add_argument("--max-distance-offshore-km", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_settings() -> dict[str, Any]:
    return {
        "scenario_name": "res_capacity_preprocessing",
        "project_root": str(DEFAULT_PROJECT_ROOT),
        "network_dir": None,
        "simulation_dir": None,
        "output_dir": None,
        "atlite_case_dir": str(DEFAULT_ATLITE_CASE_DIR),
        "availability_onshore_nc": None,
        "availability_offshore_nc": None,
        "target_capacity_csv": None,
        "pv_power_csv": None,
        "onwind_power_csv": None,
        "offwind_power_csv": None,
        "weather_root": None,
        "buses_csv": None,
        "plants_csv": None,
        "country_clusters_csv": None,
        "excluded_countries_csv": None,
        "onshore_voronoi_geojson": None,
        "generated_onshore_voronoi_geojson": None,
        "offshore_eez_geojson": str(
            DEFAULT_PROJECT_ROOT / "datashapes" / "eez_offshore_eu27_uk_no_europe_only.geojson"
        ),
        "onshore_mask_geojsons": [str(DEFAULT_PROJECT_ROOT / "datashapes" / "europe_shape.geojson")],
        "target_year": None,
        "start_year": DEFAULT_START_YEAR,
        "end_year": DEFAULT_END_YEAR,
        "tyndp_scenario": "NationalTrends",
        "tyndp_capacity_unit": "GW",
        "min_p_max_pu": 0.0,
        "min_p_nom_max": 0.0,
        "min_distance_offshore_km": None,
        "max_distance_offshore_km": None,
    }


def infer_target_year(network_dir: Path | None, explicit: int | None) -> int | None:
    if explicit is not None:
        return explicit
    if network_dir is None:
        return None
    match = re.search(r"target_year_(\d{4})", str(network_dir))
    if match:
        return int(match.group(1))
    return None


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = default_settings()
    config_dir = None
    if args.config is not None:
        cfg_path = resolve_cli_path(args.config)
        assert cfg_path is not None
        settings.update(load_yaml_config(cfg_path))
        config_dir = cfg_path.parent

    overrides = {
        "project_root": resolve_cli_path(args.project_root),
        "network_dir": resolve_cli_path(args.network_dir),
        "simulation_dir": resolve_cli_path(args.simulation_dir),
        "output_dir": resolve_cli_path(args.output_dir),
        "atlite_case_dir": resolve_cli_path(args.atlite_case_dir),
        "availability_onshore_nc": resolve_cli_path(args.availability_onshore_nc),
        "availability_offshore_nc": resolve_cli_path(args.availability_offshore_nc),
        "target_capacity_csv": resolve_cli_path(args.target_capacity_csv),
        "pv_power_csv": resolve_cli_path(args.pv_power_csv),
        "onwind_power_csv": resolve_cli_path(args.onwind_power_csv),
        "offwind_power_csv": resolve_cli_path(args.offwind_power_csv),
        "weather_root": resolve_cli_path(args.weather_root),
        "buses_csv": resolve_cli_path(args.buses_csv),
        "plants_csv": resolve_cli_path(args.plants_csv),
        "country_clusters_csv": resolve_cli_path(args.country_clusters_csv),
        "excluded_countries_csv": resolve_cli_path(args.excluded_countries_csv),
        "onshore_voronoi_geojson": resolve_cli_path(args.onshore_voronoi_geojson),
        "generated_onshore_voronoi_geojson": resolve_cli_path(args.generated_onshore_voronoi_geojson),
        "offshore_eez_geojson": resolve_cli_path(args.offshore_eez_geojson),
        "target_year": args.target_year,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "tyndp_scenario": args.tyndp_scenario,
        "tyndp_capacity_unit": args.tyndp_capacity_unit,
        "min_p_max_pu": args.min_p_max_pu,
        "min_p_nom_max": args.min_p_nom_max,
        "min_distance_offshore_km": args.min_distance_offshore_km,
        "max_distance_offshore_km": args.max_distance_offshore_km,
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    if args.onshore_mask_geojson:
        settings["onshore_mask_geojsons"] = [resolve_cli_path(path) for path in args.onshore_mask_geojson]
    for threshold_key in ("min_p_max_pu", "min_p_nom_max"):
        settings[threshold_key] = parse_threshold_value(settings.get(threshold_key))
    settings["min_distance_offshore_km"] = parse_optional_float(
        settings.get("min_distance_offshore_km"),
        "min_distance_offshore_km",
    )
    settings["max_distance_offshore_km"] = parse_optional_float(
        settings.get("max_distance_offshore_km"),
        "max_distance_offshore_km",
    )
    if (
        settings["min_distance_offshore_km"] is not None
        and settings["max_distance_offshore_km"] is not None
        and settings["min_distance_offshore_km"] > settings["max_distance_offshore_km"]
    ):
        raise ValueError("min_distance_offshore_km must be <= max_distance_offshore_km.")
    settings["overwrite"] = args.overwrite or bool(settings.get("overwrite", False))

    project_root = resolve_path(settings.get("project_root"), base_dir=config_dir) or DEFAULT_PROJECT_ROOT
    settings["project_root"] = project_root

    network_dir = resolve_path(settings.get("network_dir"), base_dir=project_root)
    if network_dir is None:
        raise ValueError("network_dir is required.")
    settings["network_dir"] = network_dir
    settings["target_year"] = infer_target_year(network_dir, settings.get("target_year"))
    if settings["target_year"] is None:
        raise ValueError("Could not infer target year. Provide target_year explicitly.")
    settings["target_year"] = validate_tyndp_target_year(settings["target_year"])
    settings["atlite_case_dir"] = resolve_path(settings.get("atlite_case_dir"), base_dir=project_root)
    if settings.get("simulation_dir") is not None:
        settings["simulation_dir"] = resolve_path(settings["simulation_dir"], base_dir=project_root)
    elif settings["atlite_case_dir"] is not None:
        settings["simulation_dir"] = settings["project_root"] / "renewables"
    else:
        settings["simulation_dir"] = settings["project_root"] / "renewables"

    settings["buses_csv"] = resolve_path(settings.get("buses_csv"), base_dir=project_root) or (network_dir / "buses.csv")
    settings["plants_csv"] = resolve_path(settings.get("plants_csv"), base_dir=project_root) or (network_dir / "plants.csv")
    settings["country_clusters_csv"] = resolve_path(
        settings.get("country_clusters_csv"), base_dir=project_root
    ) or (network_dir / "cesa_country_clusters.csv")
    settings["excluded_countries_csv"] = resolve_path(
        settings.get("excluded_countries_csv"), base_dir=project_root
    )
    if settings["excluded_countries_csv"] is None:
        candidate = network_dir / "excluded_countries.csv"
        settings["excluded_countries_csv"] = candidate if candidate.exists() else None
    if settings["availability_onshore_nc"] is not None:
        settings["availability_onshore_nc"] = resolve_path(settings["availability_onshore_nc"], base_dir=project_root)
    elif settings["atlite_case_dir"] is not None:
        settings["availability_onshore_nc"] = settings["atlite_case_dir"] / "availability_onshore.nc"
    else:
        settings["availability_onshore_nc"] = DEFAULT_ATLITE_CASE_DIR / "availability_onshore.nc"
    if settings["availability_offshore_nc"] is not None:
        settings["availability_offshore_nc"] = resolve_path(settings["availability_offshore_nc"], base_dir=project_root)
    elif settings["atlite_case_dir"] is not None:
        settings["availability_offshore_nc"] = settings["atlite_case_dir"] / "availability_offshore.nc"
    else:
        settings["availability_offshore_nc"] = DEFAULT_ATLITE_CASE_DIR / "availability_offshore.nc"
    if settings.get("weather_root") is not None:
        settings["weather_root"] = resolve_path(settings["weather_root"], base_dir=project_root)
    else:
        settings["weather_root"] = settings["atlite_case_dir"]
    settings["offshore_eez_geojson"] = resolve_path(settings["offshore_eez_geojson"], base_dir=project_root)
    settings["onshore_voronoi_geojson"] = resolve_path(
        settings.get("onshore_voronoi_geojson"), base_dir=project_root
    )
    generated_default = (
        settings["project_root"]
        / "datashapes"
        / "generated_regions"
        / f"{settings['network_dir'].parent.name}_{settings['network_dir'].name}_onshore_voronoi.geojson"
    )
    settings["generated_onshore_voronoi_geojson"] = resolve_path(
        settings.get("generated_onshore_voronoi_geojson"),
        base_dir=project_root,
    ) or generated_default

    if settings.get("output_dir") is None:
        settings["output_dir"] = (
            build_simulation_case_dir(
                settings["simulation_dir"],
                settings["network_dir"],
                settings["atlite_case_dir"],
            )
            / "res_bus_cap_preprocessed"
        )
    else:
        settings["output_dir"] = resolve_path(settings["output_dir"], base_dir=project_root)

    target_year = settings["target_year"]
    project_root = settings["project_root"]
    if settings.get("target_capacity_csv") is None:
        settings["target_capacity_csv"] = (
            project_root / "renewables" / f"res_generation_mapping_diag_{target_year}_tyndp2024.csv"
        )
    else:
        settings["target_capacity_csv"] = resolve_path(settings["target_capacity_csv"], base_dir=project_root)
    if settings.get("pv_power_csv") is None:
        settings["pv_power_csv"] = project_root / "renewables" / f"pv_power_{target_year}_tyndp2024.csv"
    else:
        settings["pv_power_csv"] = resolve_path(settings["pv_power_csv"], base_dir=project_root)
    if settings.get("onwind_power_csv") is None:
        settings["onwind_power_csv"] = project_root / "renewables" / f"won_power_{target_year}_tyndp2024.csv"
    else:
        settings["onwind_power_csv"] = resolve_path(settings["onwind_power_csv"], base_dir=project_root)
    if settings.get("offwind_power_csv") is None:
        settings["offwind_power_csv"] = project_root / "renewables" / f"woff_power_{target_year}_tyndp2024.csv"
    else:
        settings["offwind_power_csv"] = resolve_path(settings["offwind_power_csv"], base_dir=project_root)

    settings["onshore_mask_geojsons"] = [
        resolve_path(path, base_dir=project_root) for path in settings.get("onshore_mask_geojsons", [])
    ]
    for required in ("availability_onshore_nc", "availability_offshore_nc", "offshore_eez_geojson"):
        if settings.get(required) is None:
            raise ValueError(f"{required} is required.")
    return settings


def read_cluster_map(settings: dict[str, Any]) -> CountryClusterMap:
    network_map = derive_network_country_map(settings["buses_csv"])
    external_map = read_country_cluster_map(settings["country_clusters_csv"])
    return merge_country_cluster_maps(network_map, external_map)


TARGET_CAPACITY_TECH_MAP = {
    "pv": {"pv", "solar_pv", "solar_rooftop"},
    "onwind": {"onwind", "wind_onshore"},
    "offwind": {"offwind", "wind_offshore", "offwind_ac", "offwind_dc", "wind_offshore_ac", "wind_offshore_dc"},
}


def parse_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def validate_tyndp_target_year(target_year: int) -> int:
    year = int(target_year)
    if year not in TYNDP_TARGET_YEARS:
        raise ValueError(
            f"Unsupported TYNDP target year {year}. Expected one of: "
            + ", ".join(str(value) for value in TYNDP_TARGET_YEARS)
        )
    return year


def detect_tyndp_years_in_path(path: Path) -> set[int]:
    return {int(match.group(0)) for match in TYNDP_YEAR_PATTERN.finditer(str(path))}


def target_year_column(lower_cols: dict[str, str]) -> str | None:
    for key in ("target_year", "ref_year", "reference_year", "year"):
        column = lower_cols.get(key)
        if column is not None:
            return column
    return None


def filter_to_tyndp_target_year(
    df: pd.DataFrame,
    path: Path,
    settings: dict[str, Any],
    lower_cols: dict[str, str],
    *,
    require_column: bool,
) -> pd.DataFrame:
    year = validate_tyndp_target_year(int(settings["target_year"]))
    year_col = target_year_column(lower_cols)
    if year_col is None:
        path_years = detect_tyndp_years_in_path(path)
        if not path_years:
            hint = (
                " Regenerate the file with year/target_year/ref_year metadata, or include the "
                "target year in the file name."
                if require_column else ""
            )
            raise ValueError(
                f"{path} has no target year column and its path does not contain one of "
                f"{list(TYNDP_TARGET_YEARS)}.{hint}"
            )
        if path_years != {year}:
            raise ValueError(
                f"{path} points to TYNDP year(s) {sorted(path_years)}, "
                f"but target_year is {year}."
            )
        return df.copy()

    years = parse_numeric_series(df[year_col]).round()
    mask = years.eq(year)
    if not bool(mask.any()):
        available_years = sorted(
            {
                int(value)
                for value in years.dropna().astype(int).tolist()
                if int(value) in TYNDP_TARGET_YEARS
            }
        )
        raise ValueError(
            f"{path} has column '{year_col}' but no rows for target_year={year}. "
            f"Available TYNDP years: {available_years or 'none'}."
        )
    return df[mask].copy()


def filter_to_tyndp_scenario(df: pd.DataFrame, settings: dict[str, Any], lower_cols: dict[str, str]) -> pd.DataFrame:
    scenario_col = lower_cols.get("scenario")
    if scenario_col is None:
        return df
    requested = str(settings["tyndp_scenario"]).strip()
    if not requested:
        return df
    return df[df[scenario_col].astype(str).str.strip() == requested].copy()


TECH_WEATHER_CONFIG = {
    "pv": {"weather_subdir": "pv", "prefix": "pv_cf_"},
    "onwind": {"weather_subdir": "onwind", "prefix": "onwind_cf_"},
    "offwind": {"weather_subdir": "offwind", "prefix": "offwind_cf_"},
}


def threshold_for_technology(value: Any, technology: str) -> float:
    raw_value = value
    if isinstance(value, dict):
        raw_value = value.get(technology, value.get("default", 0.0))
    if raw_value in (None, ""):
        return 0.0
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid threshold value for {technology}: {raw_value!r}") from exc


def normalize_cf_dataarray(cf: xr.DataArray, source: Path) -> xr.DataArray:
    if set(cf.dims) != {"time", "y", "x"}:
        raise ValueError(f"Unexpected CF dims in {source}: {list(cf.dims)}")
    if tuple(cf.dims) != ("time", "y", "x"):
        cf = cf.transpose("time", "y", "x")
    return cf


def load_cf_block(cf: xr.DataArray, positions: list[int]) -> np.ndarray:
    if not positions:
        return np.empty((0, int(cf.sizes["y"]), int(cf.sizes["x"])), dtype=np.float64)
    return np.asarray(cf.isel(time=np.asarray(positions, dtype=int)).values, dtype=np.float64)


def compute_mean_cell_cf(
    cells: pd.DataFrame,
    weather_root: Path,
    technology: str,
    start_year: int,
    end_year: int,
) -> np.ndarray:
    if cells.empty:
        return np.array([], dtype=float)

    row_ix = cells["row_ix"].to_numpy(int)
    col_ix = cells["col_ix"].to_numpy(int)
    sums = np.zeros(len(cells), dtype=float)
    counts = np.zeros(len(cells), dtype=np.int64)
    found_any = False
    cfg = TECH_WEATHER_CONFIG[technology]

    for weather_year in range(start_year, end_year + 1):
        cf_path = weather_root / cfg["weather_subdir"] / f"{cfg['prefix']}{weather_year}.nc"
        if not cf_path.exists():
            LOG.warning("missing CF file for %s %s during cell filtering", technology, weather_year)
            continue
        found_any = True
        with xr.open_dataset(cf_path, engine="netcdf4") as ds:
            cf = normalize_cf_dataarray(ds["cf"], cf_path)
            mean_cf = np.asarray(cf.mean(dim="time", skipna=True).values, dtype=np.float64)
            cell_values = mean_cf[row_ix, col_ix]
            valid = np.isfinite(cell_values)
            sums += np.where(valid, cell_values, 0.0)
            counts += valid.astype(np.int64)

    if not found_any:
        raise FileNotFoundError(
            f"No CF files found for {technology} in {weather_root} for years {start_year}-{end_year}."
        )

    return np.divide(
        sums,
        counts,
        out=np.full(len(cells), np.nan, dtype=float),
        where=counts > 0,
    )


def apply_cell_threshold_filters(
    cells: pd.DataFrame,
    technology: str,
    settings: dict[str, Any],
) -> pd.DataFrame:
    if cells.empty:
        return cells

    min_p_nom_max = threshold_for_technology(settings.get("min_p_nom_max"), technology)
    min_p_max_pu = threshold_for_technology(settings.get("min_p_max_pu"), technology)
    filtered = cells.copy()

    if min_p_nom_max > 0.0:
        before_cells = len(filtered)
        before_buses = filtered["bus_id"].nunique()
        filtered = filtered[filtered["p_nom_max_mw"] >= min_p_nom_max].copy()
        after_cells = len(filtered)
        after_buses = filtered["bus_id"].nunique()
        LOG.info(
            "applied min_p_nom_max=%.4f for %s at cell level: kept %d/%d cells across %d/%d buses",
            min_p_nom_max,
            technology,
            after_cells,
            before_cells,
            after_buses,
            before_buses,
        )

    if min_p_max_pu > 0.0 and not filtered.empty:
        weather_root = settings.get("weather_root")
        if weather_root is None:
            raise ValueError("min_p_max_pu requires weather_root or atlite_case_dir.")
        filtered["mean_cell_cf"] = compute_mean_cell_cf(
            filtered,
            weather_root,
            technology,
            int(settings["start_year"]),
            int(settings["end_year"]),
        )
        valid_cf = filtered[np.isfinite(filtered["mean_cell_cf"])].copy()
        if valid_cf.empty:
            LOG.warning("no valid mean CF values found for %s during cell filtering", technology)
            return filtered.iloc[0:0].copy()
        before_cells = len(filtered)
        before_buses = filtered["bus_id"].nunique()
        filtered = valid_cf[valid_cf["mean_cell_cf"] >= min_p_max_pu].copy()
        after_cells = len(filtered)
        after_buses = filtered["bus_id"].nunique()
        LOG.info(
            "applied min_p_max_pu=%.4f for %s at cell level: kept %d/%d cells across %d/%d buses",
            min_p_max_pu,
            technology,
            after_cells,
            before_cells,
            after_buses,
            before_buses,
        )
        filtered = filtered.drop(columns=["mean_cell_cf"], errors="ignore")

    return filtered


def load_target_capacity(
    path: Path,
    settings: dict[str, Any],
    cluster_map: CountryClusterMap,
    technology: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path, sep=detect_delimiter(path))
    lower_cols = {column.lower(): column for column in df.columns}
    excluded_source_countries = set(settings.get("excluded_source_countries") or [])

    if {"tech", "total_cap_mw"}.issubset(lower_cols):
        if technology is None:
            raise ValueError("technology is required when loading target-capacity mapping diagnostics.")
        country_col = lower_cols.get("country")
        tech_col = lower_cols["tech"]
        value_col = lower_cols["total_cap_mw"]
        if country_col is None:
            raise ValueError(f"{path} is missing a country column.")

        try:
            df = filter_to_tyndp_target_year(df, path, settings, lower_cols, require_column=True)
        except ValueError as exc:
            fallback_key = {
                "pv": "pv_power_csv",
                "onwind": "onwind_power_csv",
                "offwind": "offwind_power_csv",
            }.get(technology)
            fallback_path = Path(settings[fallback_key]) if fallback_key and settings.get(fallback_key) else None
            if fallback_path is not None and fallback_path.exists():
                LOG.warning(
                    "target_capacity_csv %s cannot be filtered to target_year=%s; using %s instead",
                    path,
                    settings["target_year"],
                    fallback_path,
                )
                return load_target_capacity(fallback_path, settings, cluster_map)
            raise ValueError(
                f"{path} cannot be filtered to target_year={settings['target_year']}. "
                "Regenerate the mapping diagnostics with year metadata or provide pv_power_csv/onwind_power_csv/offwind_power_csv."
            ) from exc
        df = filter_to_tyndp_scenario(df, settings, lower_cols)
        df["technology_key"] = df[tech_col].astype(str).str.strip().str.lower()
        allowed = TARGET_CAPACITY_TECH_MAP[technology]
        ignored = sorted(set(df["technology_key"].dropna()) - set().union(*TARGET_CAPACITY_TECH_MAP.values()))
        if ignored:
            LOG.warning("ignoring unsupported target-capacity technologies in %s: %s", path.name, ", ".join(ignored))
        df = df[df["technology_key"].isin(allowed)].copy()
        df["source_country"] = df[country_col].map(normalize_country_name_or_code)
        if excluded_source_countries:
            df = df[~df["source_country"].isin(excluded_source_countries)].copy()
        df["model_country"] = df["source_country"].map(lambda value: map_country_code(value, cluster_map))
        df["target_capacity_mw"] = parse_numeric_series(df[value_col]).fillna(0.0)
        df = df[df["model_country"].astype(bool)]
        return df.groupby("model_country", as_index=False)["target_capacity_mw"].sum()

    expected = {"country", "cap"}
    missing = sorted(expected - set(lower_cols))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    country_col = lower_cols["country"]
    cap_col = lower_cols["cap"]
    df = filter_to_tyndp_target_year(df, path, settings, lower_cols, require_column=False)
    df = filter_to_tyndp_scenario(df, settings, lower_cols)
    df["source_country"] = df[country_col].map(normalize_country_name_or_code)
    if excluded_source_countries:
        df = df[~df["source_country"].isin(excluded_source_countries)].copy()
    df["model_country"] = df["source_country"].map(lambda value: map_country_code(value, cluster_map))
    df["target_capacity_mw"] = convert_capacity_series_to_mw(
        parse_numeric_series(df[cap_col]).fillna(0.0),
        settings["tyndp_capacity_unit"],
    )
    df = df[df["model_country"].astype(bool)]
    return df.groupby("model_country", as_index=False)["target_capacity_mw"].sum()


def base_cell_table(ds: xr.Dataset, mask: np.ndarray) -> pd.DataFrame:
    row_idx, col_idx = np.where(mask)
    lat_values = np.asarray(ds["lat"].values)
    lon_values = np.asarray(ds["lon"].values)
    if lat_values.ndim == 1 and lon_values.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
    else:
        lat_grid = lat_values
        lon_grid = lon_values
    lat = lat_grid[row_idx, col_idx]
    lon = lon_grid[row_idx, col_idx]
    area = np.asarray(ds["area_km2"].values)[row_idx, col_idx]
    return pd.DataFrame(
        {
            "row_ix": row_idx.astype(int),
            "col_ix": col_idx.astype(int),
            "lat": lat.astype(float),
            "lon": lon.astype(float),
            "area_km2": area.astype(float),
        }
    )


def build_onshore_assignment(ds_onshore: xr.Dataset, voronoi_path: Path) -> pd.DataFrame:
    mask = (
        np.isfinite(np.asarray(ds_onshore["p_nom_max_pv"].values))
        & (np.asarray(ds_onshore["p_nom_max_pv"].values) > 0)
    ) | (
        np.isfinite(np.asarray(ds_onshore["p_nom_max_onwind"].values))
        & (np.asarray(ds_onshore["p_nom_max_onwind"].values) > 0)
    )
    cells = base_cell_table(ds_onshore, mask)
    points = gpd.GeoDataFrame(
        cells,
        geometry=gpd.points_from_xy(cells["lon"], cells["lat"]),
        crs="EPSG:4326",
    )
    regions = gpd.read_file(voronoi_path).to_crs("EPSG:4326")
    assigned = assign_points_to_polygons(points, regions, ["name", "country"])
    assigned = assigned.rename(columns={"name": "bus_id", "country": "model_country"})
    return pd.DataFrame(assigned.drop(columns="geometry"))


def extract_eez_country(row: pd.Series) -> str:
    for field in ("country", "Country", "ISO2", "iso2", "ISO_SOV1", "ISO_TER1", "SOVEREIGN1", "TERRITORY1"):
        value = row.get(field)
        if value not in (None, "", np.nan):
            country = normalize_country_name_or_code(value)
            if country:
                return country
    return ""


def haversine_matrix(lat: np.ndarray, lon: np.ndarray, ref_lat: np.ndarray, ref_lon: np.ndarray) -> np.ndarray:
    lat1 = np.radians(lat)[:, None]
    lon1 = np.radians(lon)[:, None]
    lat2 = np.radians(ref_lat)[None, :]
    lon2 = np.radians(ref_lon)[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def assign_nearest_buses_by_country(cells: pd.DataFrame, bus_lookup: pd.DataFrame) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for country, group in cells.groupby("model_country", sort=False):
        candidates = bus_lookup[bus_lookup["country"] == country]
        group = group.copy()
        if candidates.empty:
            group["bus_id"] = ""
            outputs.append(group)
            continue
        distances = haversine_matrix(
            group["lat"].to_numpy(float),
            group["lon"].to_numpy(float),
            candidates["lat"].to_numpy(float),
            candidates["lon"].to_numpy(float),
        )
        nearest = distances.argmin(axis=1)
        group["bus_id"] = candidates.iloc[nearest]["bus_id"].to_numpy()
        outputs.append(group)
    return pd.concat(outputs, ignore_index=True) if outputs else cells


def build_offshore_assignment(
    ds_offshore: xr.Dataset,
    eez_path: Path,
    bus_lookup: pd.DataFrame,
    cluster_map: CountryClusterMap,
    offshore_distance_shape_paths: list[Path] | None = None,
    min_distance_offshore_km: float | None = None,
    max_distance_offshore_km: float | None = None,
) -> pd.DataFrame:
    mask = np.isfinite(np.asarray(ds_offshore["p_nom_max_offwind"].values)) & (
        np.asarray(ds_offshore["p_nom_max_offwind"].values) > 0
    )
    cells = base_cell_table(ds_offshore, mask)
    cells = apply_offshore_distance_filter(
        cells,
        offshore_distance_shape_paths or [],
        min_distance_km=min_distance_offshore_km,
        max_distance_km=max_distance_offshore_km,
    )
    if cells.empty:
        cells["source_country"] = []
        cells["model_country"] = []
        cells["bus_id"] = []
        return cells
    points = gpd.GeoDataFrame(
        cells,
        geometry=gpd.points_from_xy(cells["lon"], cells["lat"]),
        crs="EPSG:4326",
    )
    eez = gpd.read_file(eez_path).to_crs("EPSG:4326")
    country_columns = [
        column
        for column in ("country", "Country", "ISO2", "iso2", "ISO_SOV1", "ISO_TER1", "SOVEREIGN1", "TERRITORY1")
        if column in eez.columns
    ]
    if not country_columns:
        raise ValueError(
            f"Offshore EEZ file {eez_path} must contain a country column "
            "(for example country, ISO2, ISO_SOV1, ISO_TER1, SOVEREIGN1, or TERRITORY1)."
        )
    assigned = assign_points_to_polygons(points, eez, country_columns)
    assigned = pd.DataFrame(assigned.drop(columns="geometry"))
    assigned["source_country"] = assigned.apply(extract_eez_country, axis=1)
    assigned["model_country"] = assigned["source_country"].map(lambda value: map_country_code(value, cluster_map))
    assigned = assign_nearest_buses_by_country(assigned, bus_lookup)
    return assigned


def allocate_current_capacity_to_cells(cells: pd.DataFrame, current_by_bus: pd.DataFrame) -> pd.DataFrame:
    if cells.empty:
        cells["current_capacity_mw"] = []
        return cells
    # Existing installations reserve part of the technical potential before new
    # TYNDP capacity is added. Within a bus region the split follows cell
    # potential, with an equal split only when the raster potential is absent.
    merged = cells.merge(
        current_by_bus.rename(columns={"Capacity": "bus_current_capacity_mw"})[
            ["bus_id", "bus_current_capacity_mw"]
        ],
        on="bus_id",
        how="left",
    )
    merged["bus_current_capacity_mw"] = merged["bus_current_capacity_mw"].fillna(0.0)
    merged["weight"] = merged["p_nom_max_mw"].clip(lower=0.0).fillna(0.0)
    group_weight = merged.groupby("bus_id")["weight"].transform("sum")
    group_count = merged.groupby("bus_id")["bus_id"].transform("count").clip(lower=1)
    merged["alloc_share"] = np.where(group_weight > 0.0, merged["weight"] / group_weight, 1.0 / group_count)
    merged["current_capacity_mw"] = merged["bus_current_capacity_mw"] * merged["alloc_share"]
    return merged.drop(columns=["alloc_share", "weight", "bus_current_capacity_mw"])


def allocate_delta_by_class(group: pd.DataFrame, delta_mw: float) -> tuple[np.ndarray, float]:
    additions = np.zeros(len(group), dtype=float)
    remaining = float(max(delta_mw, 0.0))
    if remaining <= 0.0:
        return additions, 0.0

    headroom = np.maximum(group["p_nom_max_mw"].to_numpy(float) - group["current_capacity_mw"].to_numpy(float), 0.0)
    resource_class = group["resource_class"].to_numpy(float)

    valid_class_mask = np.isfinite(resource_class) & (resource_class >= 0.0)
    class_values = sorted(set(resource_class[valid_class_mask].astype(int)), reverse=True)
    # New capacity fills the best resource classes first. Cells in the same class
    # receive equal increments until their local headroom is exhausted.
    for class_value in class_values:
        class_mask = valid_class_mask & (resource_class.astype(int) == class_value) & (headroom > 0.0)
        if not class_mask.any():
            continue
        while remaining > 1e-9 and class_mask.any():
            share = remaining / class_mask.sum()
            increment = np.minimum(np.full(class_mask.sum(), share), headroom[class_mask])
            if float(increment.sum()) <= 1e-12:
                break
            additions[class_mask] += increment
            headroom[class_mask] -= increment
            remaining -= float(increment.sum())
            class_mask = valid_class_mask & (resource_class.astype(int) == class_value) & (headroom > 0.0)
    return additions, remaining


def build_technology_frame(
    base_assignment: pd.DataFrame,
    ds: xr.Dataset,
    *,
    technology: str,
    availability_var: str,
    p_nom_var: str,
    resource_class_var: str,
    current_by_bus: pd.DataFrame,
    target_by_country: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = base_assignment.copy()
    availability = np.asarray(ds[availability_var].values)
    p_nom_max = np.asarray(ds[p_nom_var].values)
    resource_class = np.asarray(ds[resource_class_var].values)

    cells["availability"] = availability[cells["row_ix"], cells["col_ix"]]
    cells["p_nom_max_mw"] = p_nom_max[cells["row_ix"], cells["col_ix"]]
    cells["resource_class"] = resource_class[cells["row_ix"], cells["col_ix"]]
    cells = cells[np.isfinite(cells["availability"]) & np.isfinite(cells["p_nom_max_mw"])].copy()
    cells = cells[cells["availability"] > 0.0].copy()
    cells["technology"] = technology
    cells = apply_cell_threshold_filters(cells, technology, settings)

    current_rows = current_by_bus[current_by_bus["technology"] == technology]
    cells = allocate_current_capacity_to_cells(cells, current_rows)

    targets = target_by_country.rename(columns={"target_capacity_mw": "country_target_capacity_mw"})
    cells = cells.merge(targets, on="model_country", how="left")
    cells["country_target_capacity_mw"] = cells["country_target_capacity_mw"].fillna(0.0)

    additions = np.zeros(len(cells), dtype=float)
    summary_rows: list[dict[str, Any]] = []
    for country, group in cells.groupby("model_country", sort=False):
        country_current = float(group["current_capacity_mw"].sum())
        target_capacity = float(group["country_target_capacity_mw"].iloc[0])
        positive_delta = max(target_capacity - country_current, 0.0)
        # The scenario target is interpreted as total capacity. If current sites
        # already exceed the target, no artificial decommissioning is imposed here.
        group_additions, unallocated = allocate_delta_by_class(group, positive_delta)
        additions[group.index.to_numpy()] = group_additions
        headroom = float(np.maximum(group["p_nom_max_mw"] - group["current_capacity_mw"], 0.0).sum())
        summary_rows.append(
            {
                "technology": technology,
                "country_model": country,
                "current_capacity_mw": country_current,
                "target_capacity_mw": target_capacity,
                "positive_delta_mw": positive_delta,
                "added_capacity_mw": float(group_additions.sum()),
                "unallocated_delta_mw": float(unallocated),
                "available_headroom_mw": headroom,
            }
        )

    cells["added_capacity_mw"] = additions
    cells["scenario_capacity_mw"] = cells["current_capacity_mw"] + cells["added_capacity_mw"]
    cells["relative_installed"] = np.where(
        cells["p_nom_max_mw"] > 0.0,
        cells["current_capacity_mw"] / cells["p_nom_max_mw"],
        np.nan,
    )
    cells["current_exceeds_p_nom_max"] = cells["current_capacity_mw"] > cells["p_nom_max_mw"]
    return cells, pd.DataFrame(summary_rows)


def fill_array(shape: tuple[int, int], frame: pd.DataFrame, value_column: str, *, dtype=float, fill_value=np.nan) -> np.ndarray:
    array = np.full(shape, fill_value, dtype=dtype)
    if frame.empty:
        return array
    array[frame["row_ix"].to_numpy(int), frame["col_ix"].to_numpy(int)] = frame[value_column].to_numpy(dtype)
    return array


def output_paths(output_dir: Path) -> dict[str, Path]:
    ensure_dir(output_dir)
    return {
        "cells_nc": output_dir / "res_capacity_cells.nc",
        "bus_lookup_csv": output_dir / "res_bus_lookup.csv",
        "bus_capacity_csv": output_dir / "res_capacity_bus.csv",
        "country_summary_csv": output_dir / "res_capacity_country_summary.csv",
        "manifest_json": output_dir / "res_capacity_preprocessing_manifest.json",
    }


def write_dataset_atomic(dataset: xr.Dataset, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    dataset.to_netcdf(tmp_path)
    tmp_path.replace(path)


def build_output_dataset(
    ds_onshore: xr.Dataset,
    bus_lookup: pd.DataFrame,
    onshore_assignment: pd.DataFrame,
    offshore_assignment: pd.DataFrame,
    tech_frames: dict[str, pd.DataFrame],
) -> xr.Dataset:
    shape = np.asarray(ds_onshore["area_km2"].values).shape
    onshore_idx = onshore_assignment.merge(bus_lookup[["bus_id", "bus_index"]], on="bus_id", how="left")
    offshore_idx = offshore_assignment.merge(bus_lookup[["bus_id", "bus_index"]], on="bus_id", how="left")

    ds = xr.Dataset(
        coords={
            "y": ds_onshore["y"],
            "x": ds_onshore["x"],
            "lat": ds_onshore["lat"],
            "lon": ds_onshore["lon"],
        }
    )
    ds["area_km2"] = ds_onshore["area_km2"]
    ds["onshore_bus_index"] = (("y", "x"), fill_array(shape, onshore_idx, "bus_index", dtype=int, fill_value=-1))
    ds["offshore_bus_index"] = (("y", "x"), fill_array(shape, offshore_idx, "bus_index", dtype=int, fill_value=-1))

    for technology, frame in tech_frames.items():
        ds[f"{technology}_availability"] = (("y", "x"), fill_array(shape, frame, "availability"))
        ds[f"{technology}_p_nom_max_mw"] = (("y", "x"), fill_array(shape, frame, "p_nom_max_mw"))
        ds[f"{technology}_resource_class"] = (("y", "x"), fill_array(shape, frame, "resource_class"))
        ds[f"{technology}_current_capacity_mw"] = (("y", "x"), fill_array(shape, frame, "current_capacity_mw"))
        ds[f"{technology}_added_capacity_mw"] = (("y", "x"), fill_array(shape, frame, "added_capacity_mw"))
        ds[f"{technology}_scenario_capacity_mw"] = (("y", "x"), fill_array(shape, frame, "scenario_capacity_mw"))
        ds[f"{technology}_relative_installed"] = (("y", "x"), fill_array(shape, frame, "relative_installed"))
        if "offshore_distance_km" in frame.columns:
            ds[f"{technology}_offshore_distance_km"] = (
                ("y", "x"),
                fill_array(shape, frame, "offshore_distance_km"),
            )
        ds[f"{technology}_current_exceeds_p_nom_max"] = (
            ("y", "x"),
            fill_array(shape, frame, "current_exceeds_p_nom_max", dtype=bool, fill_value=False),
        )
    return ds


def bus_capacity_summary(tech_frames: dict[str, pd.DataFrame], bus_lookup: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for technology, frame in tech_frames.items():
        if frame.empty:
            continue
        summary = (
            frame.groupby(["bus_id", "model_country"], as_index=False)[
                ["current_capacity_mw", "added_capacity_mw", "scenario_capacity_mw"]
            ]
            .sum()
            .rename(columns={"model_country": "country"})
        )
        summary["technology"] = technology
        frames.append(summary)
    if not frames:
        return pd.DataFrame(
            columns=["bus_id", "country", "technology", "current_capacity_mw", "added_capacity_mw", "scenario_capacity_mw"]
        )
    out = pd.concat(frames, ignore_index=True)
    out = out[out["bus_id"].astype(str).str.strip().astype(bool)].copy()
    return out.merge(bus_lookup[["bus_id", "country_label"]], on="bus_id", how="left")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = resolve_settings(parse_args())
    out_paths = output_paths(settings["output_dir"])

    LOG.info("loading network inputs")
    buses = load_reduced_buses(settings["buses_csv"])
    plants = load_plants(settings["plants_csv"])
    cluster_map = read_cluster_map(settings)
    settings["excluded_source_countries"] = sorted(read_excluded_countries(settings.get("excluded_countries_csv")))
    bus_lookup = build_bus_lookup(buses)

    voronoi_path = settings["onshore_voronoi_geojson"]
    if settings.get("overwrite") or voronoi_path is None or not voronoi_path.exists():
        LOG.info("building onshore Voronoi for %s", settings["network_dir"])
        regions = build_onshore_voronoi(buses, settings["onshore_mask_geojsons"])
        voronoi_path = settings["generated_onshore_voronoi_geojson"]
        write_onshore_voronoi(voronoi_path, regions)
    settings["onshore_voronoi_geojson"] = voronoi_path

    LOG.info("loading global availability datasets")
    ds_onshore = xr.open_dataset(settings["availability_onshore_nc"], engine="netcdf4")
    ds_offshore = xr.open_dataset(settings["availability_offshore_nc"], engine="netcdf4")

    LOG.info("building cell-to-bus assignments")
    onshore_assignment = build_onshore_assignment(ds_onshore, settings["onshore_voronoi_geojson"])
    offshore_assignment = build_offshore_assignment(
        ds_offshore,
        settings["offshore_eez_geojson"],
        bus_lookup,
        cluster_map,
        settings["onshore_mask_geojsons"],
        settings.get("min_distance_offshore_km"),
        settings.get("max_distance_offshore_km"),
    )

    current_by_bus = aggregate_current_capacity_by_bus(plants)

    LOG.info("loading TYNDP target capacities")
    target_capacity_csv = settings.get("target_capacity_csv")
    if target_capacity_csv is not None and Path(target_capacity_csv).exists():
        pv_target = load_target_capacity(Path(target_capacity_csv), settings, cluster_map, technology="pv")
        onwind_target = load_target_capacity(Path(target_capacity_csv), settings, cluster_map, technology="onwind")
        offwind_target = load_target_capacity(Path(target_capacity_csv), settings, cluster_map, technology="offwind")
    else:
        pv_target = load_target_capacity(settings["pv_power_csv"], settings, cluster_map)
        onwind_target = load_target_capacity(settings["onwind_power_csv"], settings, cluster_map)
        offwind_target = load_target_capacity(settings["offwind_power_csv"], settings, cluster_map)

    LOG.info("allocating current and target capacity to raster cells")
    pv_frame, pv_summary = build_technology_frame(
        onshore_assignment,
        ds_onshore,
        technology="pv",
        availability_var="availability_pv",
        p_nom_var="p_nom_max_pv",
        resource_class_var="resource_class_pv",
        current_by_bus=current_by_bus,
        target_by_country=pv_target,
        settings=settings,
    )
    onwind_frame, onwind_summary = build_technology_frame(
        onshore_assignment,
        ds_onshore,
        technology="onwind",
        availability_var="availability_onwind",
        p_nom_var="p_nom_max_onwind",
        resource_class_var="resource_class_onwind",
        current_by_bus=current_by_bus,
        target_by_country=onwind_target,
        settings=settings,
    )
    offwind_frame, offwind_summary = build_technology_frame(
        offshore_assignment,
        ds_offshore,
        technology="offwind",
        availability_var="availability_offwind",
        p_nom_var="p_nom_max_offwind",
        resource_class_var="resource_class_offwind",
        current_by_bus=current_by_bus,
        target_by_country=offwind_target,
        settings=settings,
    )

    tech_frames = {"pv": pv_frame, "onwind": onwind_frame, "offwind": offwind_frame}

    LOG.info("writing outputs")
    dataset = build_output_dataset(ds_onshore, bus_lookup, onshore_assignment, offshore_assignment, tech_frames)
    write_dataset_atomic(dataset, out_paths["cells_nc"])
    bus_lookup.to_csv(out_paths["bus_lookup_csv"], index=False)
    bus_capacity_summary(tech_frames, bus_lookup).to_csv(out_paths["bus_capacity_csv"], index=False)

    country_summary = pd.concat([pv_summary, onwind_summary, offwind_summary], ignore_index=True)
    country_summary["country_label"] = country_summary["country_model"].map(
        lambda value: label_for_country(value, cluster_map)
    )
    country_summary.to_csv(out_paths["country_summary_csv"], index=False)

    manifest = {
        "scenario_name": settings["scenario_name"],
        "project_root": str(settings["project_root"]),
        "network_dir": str(settings["network_dir"]),
        "simulation_dir": str(settings["simulation_dir"]) if settings.get("simulation_dir") is not None else None,
        "target_year": settings["target_year"],
        "start_year": settings["start_year"],
        "end_year": settings["end_year"],
        "atlite_case_dir": str(settings["atlite_case_dir"]) if settings.get("atlite_case_dir") is not None else None,
        "availability_onshore_nc": str(settings["availability_onshore_nc"]),
        "availability_offshore_nc": str(settings["availability_offshore_nc"]),
        "target_capacity_csv": str(settings["target_capacity_csv"]) if settings.get("target_capacity_csv") else None,
        "excluded_countries_csv": str(settings["excluded_countries_csv"]) if settings.get("excluded_countries_csv") else None,
        "excluded_source_countries": settings.get("excluded_source_countries", []),
        "pv_power_csv": str(settings["pv_power_csv"]),
        "onwind_power_csv": str(settings["onwind_power_csv"]),
        "offwind_power_csv": str(settings["offwind_power_csv"]),
        "weather_root": str(settings["weather_root"]) if settings.get("weather_root") is not None else None,
        "min_p_max_pu": settings.get("min_p_max_pu"),
        "min_p_nom_max": settings.get("min_p_nom_max"),
        "min_distance_offshore_km": settings.get("min_distance_offshore_km"),
        "max_distance_offshore_km": settings.get("max_distance_offshore_km"),
        "onshore_voronoi_geojson": str(settings["onshore_voronoi_geojson"]),
        "offshore_eez_geojson": str(settings["offshore_eez_geojson"]),
        "output_dir": str(settings["output_dir"]),
        "outputs": {key: str(value) for key, value in out_paths.items()},
    }
    write_json(out_paths["manifest_json"], manifest)
    LOG.info("done")


if __name__ == "__main__":
    main()
