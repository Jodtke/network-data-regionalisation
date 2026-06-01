from __future__ import annotations

"""Disaggregate national TYNDP renewable generation to reduced-grid buses.

Bus-level renewable generation is derived from raster capacity factors and the
scenario capacity allocation created in ``res_capacity_preprocessing``. For each
weather year and hour, the local raw output is scaled to the corresponding
national TYNDP generation total. This preserves the scenario energy balance
while retaining the spatial and temporal heterogeneity of the self-generated
resource profiles.

Fallbacks are intentionally conservative: if a valid weather profile produces
zero output, the zero is kept; if the profile is missing but scenario capacity
exists, generation is distributed by installed bus capacity and flagged in the
diagnostics.
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from res_common import (
    DEFAULT_END_YEAR,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_START_YEAR,
    build_simulation_case_dir,
    detect_delimiter,
    derive_network_country_map,
    ensure_dir,
    label_for_country,
    load_yaml_config,
    map_country_code,
    merge_country_cluster_maps,
    normalize_country_name_or_code,
    read_country_cluster_map,
    read_excluded_countries,
    resolve_cli_path,
    resolve_path,
    sources_for_country,
    write_json,
)

LOG = logging.getLogger(__name__)
DEFAULT_ATLITE_CASE_DIR = (
    DEFAULT_PROJECT_ROOT / "renewables" / "atlite_copy" / "corine_luisa_wdpa_onoff_acdc"
)
TYNDP_TARGET_YEARS = (2030, 2040, 2050)
TYNDP_YEAR_PATTERN = re.compile(r"(?<!\d)(?:2030|2040|2050)(?!\d)")

TECH_CONFIG = {
    "pv": {
        "column": "solar_gen",
        "weather_subdir": "pv",
        "prefix": "pv_cf_",
        "bus_index_var": "onshore_bus_index",
        "weight_var": "pv_scenario_capacity_mw",
    },
    "onwind": {
        "column": "wind_onshore_gen",
        "weather_subdir": "onwind",
        "prefix": "onwind_cf_",
        "bus_index_var": "onshore_bus_index",
        "weight_var": "onwind_scenario_capacity_mw",
    },
    "offwind": {
        "column": "wind_offshore_gen",
        "weather_subdir": "offwind",
        "prefix": "offwind_cf_",
        "bus_index_var": "offshore_bus_index",
        "weight_var": "offwind_scenario_capacity_mw",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Disaggregate national RES generation to reduced network buses."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--network-dir", type=Path, default=None)
    parser.add_argument("--simulation-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--atlite-case-dir", type=Path, default=None)
    parser.add_argument("--capacity-output-dir", type=Path, default=None)
    parser.add_argument("--cells-nc", type=Path, default=None)
    parser.add_argument("--bus-lookup-csv", type=Path, default=None)
    parser.add_argument("--bus-capacity-csv", type=Path, default=None)
    parser.add_argument("--country-clusters-csv", type=Path, default=None)
    parser.add_argument("--excluded-countries-csv", type=Path, default=None)
    parser.add_argument("--generation-long-csv", type=Path, default=None)
    parser.add_argument("--weather-root", type=Path, default=None)
    parser.add_argument("--buses-csv", type=Path, default=None)
    parser.add_argument("--target-year", type=int, default=None)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    return parser.parse_args()


def default_settings() -> dict[str, Any]:
    return {
        "scenario_name": "res_generation_disaggregation",
        "project_root": str(DEFAULT_PROJECT_ROOT),
        "network_dir": None,
        "simulation_dir": None,
        "output_dir": None,
        "atlite_case_dir": str(DEFAULT_ATLITE_CASE_DIR),
        "capacity_output_dir": None,
        "cells_nc": None,
        "bus_lookup_csv": None,
        "bus_capacity_csv": None,
        "country_clusters_csv": None,
        "excluded_countries_csv": None,
        "generation_long_csv": None,
        "weather_root": None,
        "buses_csv": None,
        "target_year": None,
        "start_year": DEFAULT_START_YEAR,
        "end_year": DEFAULT_END_YEAR,
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
        "capacity_output_dir": resolve_cli_path(args.capacity_output_dir),
        "cells_nc": resolve_cli_path(args.cells_nc),
        "bus_lookup_csv": resolve_cli_path(args.bus_lookup_csv),
        "bus_capacity_csv": resolve_cli_path(args.bus_capacity_csv),
        "country_clusters_csv": resolve_cli_path(args.country_clusters_csv),
        "excluded_countries_csv": resolve_cli_path(args.excluded_countries_csv),
        "generation_long_csv": resolve_cli_path(args.generation_long_csv),
        "weather_root": resolve_cli_path(args.weather_root),
        "buses_csv": resolve_cli_path(args.buses_csv),
        "target_year": args.target_year,
        "start_year": args.start_year,
        "end_year": args.end_year,
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value

    project_root = resolve_path(settings["project_root"], base_dir=config_dir) or DEFAULT_PROJECT_ROOT
    settings["project_root"] = project_root
    settings["network_dir"] = resolve_path(settings["network_dir"], base_dir=project_root)
    if settings["network_dir"] is None:
        raise ValueError("network_dir is required.")
    settings["target_year"] = infer_target_year(settings["network_dir"], settings.get("target_year"))
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

    capacity_output_dir = resolve_path(settings.get("capacity_output_dir"), base_dir=project_root)
    if capacity_output_dir is None:
        capacity_output_dir = (
            build_simulation_case_dir(
                settings["simulation_dir"],
                settings["network_dir"],
                settings["atlite_case_dir"],
            )
            / "res_bus_cap_preprocessed"
        )
    settings["capacity_output_dir"] = capacity_output_dir
    settings["cells_nc"] = resolve_path(settings.get("cells_nc"), base_dir=project_root) or (
        capacity_output_dir / "res_capacity_cells.nc"
    )
    settings["bus_lookup_csv"] = resolve_path(settings.get("bus_lookup_csv"), base_dir=project_root) or (
        capacity_output_dir / "res_bus_lookup.csv"
    )
    settings["bus_capacity_csv"] = resolve_path(settings.get("bus_capacity_csv"), base_dir=project_root) or (
        capacity_output_dir / "res_capacity_bus.csv"
    )
    settings["country_clusters_csv"] = resolve_path(
        settings.get("country_clusters_csv"), base_dir=project_root
    ) or (settings["network_dir"] / "cesa_country_clusters.csv")
    settings["excluded_countries_csv"] = resolve_path(
        settings.get("excluded_countries_csv"), base_dir=project_root
    )
    if settings["excluded_countries_csv"] is None:
        candidate = settings["network_dir"] / "excluded_countries.csv"
        settings["excluded_countries_csv"] = candidate if candidate.exists() else None
    settings["buses_csv"] = resolve_path(settings.get("buses_csv"), base_dir=project_root) or (
        settings["network_dir"] / "buses.csv"
    )
    if settings["weather_root"] is not None:
        settings["weather_root"] = resolve_path(settings["weather_root"], base_dir=project_root)
    else:
        settings["weather_root"] = settings["atlite_case_dir"]
    if settings.get("generation_long_csv") is None:
        settings["generation_long_csv"] = (
            settings["project_root"] / "renewables" / f"res_load_country_long_{settings['target_year']}_tyndp2024.csv"
        )
    else:
        settings["generation_long_csv"] = resolve_path(settings["generation_long_csv"], base_dir=project_root)

    if settings.get("output_dir") is None:
        settings["output_dir"] = (
            build_simulation_case_dir(
                settings["simulation_dir"],
                settings["network_dir"],
                settings["atlite_case_dir"],
            )
            / "disaggregated"
        )
    else:
        settings["output_dir"] = resolve_path(settings["output_dir"], base_dir=project_root)
    if settings["weather_root"] is None:
        raise ValueError("weather_root or atlite_case_dir is required.")
    return settings


def output_paths(output_dir: Path) -> dict[str, Path]:
    ensure_dir(output_dir)
    return {
        "generation_csv": output_dir / "disaggregated_res_country_bus.csv",
        "diagnostics_csv": output_dir / "res_generation_scaling_diagnostics.csv",
        "manifest_json": output_dir / "res_generation_disaggregation_manifest.json",
    }


def read_cluster_map(settings: dict[str, Any]):
    network_map = derive_network_country_map(settings["buses_csv"])
    external_map = read_country_cluster_map(settings["country_clusters_csv"])
    return merge_country_cluster_maps(network_map, external_map)


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


def target_year_column(columns: list[str]) -> str | None:
    by_lower = {column.lower(): column for column in columns}
    for key in ("target_year", "ref_year", "reference_year", "scenario_year"):
        column = by_lower.get(key)
        if column is not None:
            return column
    return None


def filter_generation_to_target_year(df: pd.DataFrame, path: Path, target_year: int) -> pd.DataFrame:
    year = validate_tyndp_target_year(target_year)
    year_col = target_year_column(list(df.columns))
    if year_col is None:
        path_years = detect_tyndp_years_in_path(path)
        if not path_years:
            raise ValueError(
                f"{path} has no target year column and its path does not contain one of "
                f"{list(TYNDP_TARGET_YEARS)}."
            )
        if path_years != {year}:
            raise ValueError(
                f"{path} points to TYNDP year(s) {sorted(path_years)}, "
                f"but target_year is {year}."
            )
        return df.copy()

    years = pd.to_numeric(df[year_col], errors="coerce").round()
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


def load_generation_rows(
    path: Path,
    cluster_map,
    start_year: int,
    end_year: int,
    target_year: int,
    excluded_source_countries: set[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path, sep=detect_delimiter(path))
    df = filter_generation_to_target_year(df, path, target_year)
    country_col = "Country" if "Country" in df.columns else "country"
    required = {country_col, "weather_year", "week", "peak_timestamp"}
    required.update(cfg["column"] for cfg in TECH_CONFIG.values())
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.copy()
    df["source_country"] = df[country_col].map(normalize_country_name_or_code)
    if excluded_source_countries:
        df = df[~df["source_country"].isin(excluded_source_countries)].copy()
    df["country_model"] = df["source_country"].map(lambda value: map_country_code(value, cluster_map))
    df["weather_year"] = pd.to_numeric(df["weather_year"], errors="coerce").astype("Int64")
    df = df[df["weather_year"].between(start_year, end_year, inclusive="both")]
    df = df[df["country_model"].astype(bool)].copy()
    df["week"] = df["week"].astype(str)
    df["peak_timestamp"] = pd.to_datetime(df["peak_timestamp"], utc=True).dt.tz_convert(None)

    grouped = (
        df.groupby(["country_model", "weather_year", "week", "peak_timestamp"], as_index=False)[
            [cfg["column"] for cfg in TECH_CONFIG.values()]
        ]
        .sum()
    )
    source_map = (
        df.groupby(["country_model", "weather_year", "week", "peak_timestamp"])["source_country"]
        .agg(lambda values: ",".join(sorted(set(values))))
        .reset_index()
    )
    return grouped.merge(source_map, on=["country_model", "weather_year", "week", "peak_timestamp"], how="left")


def load_capacity_outputs(settings: dict[str, Any]) -> tuple[xr.Dataset, pd.DataFrame, pd.DataFrame]:
    ds = xr.open_dataset(settings["cells_nc"], engine="netcdf4")
    bus_lookup = pd.read_csv(settings["bus_lookup_csv"])
    bus_capacity = pd.read_csv(settings["bus_capacity_csv"])
    return ds, bus_lookup, bus_capacity


def precompute_cell_weights(ds: xr.Dataset, bus_lookup: pd.DataFrame) -> dict[str, dict[str, Any]]:
    n_buses = len(bus_lookup)
    outputs: dict[str, dict[str, Any]] = {}
    for technology, cfg in TECH_CONFIG.items():
        # Store only valid raster cells once. The hourly profile loop can then use
        # vectorised bincounts instead of repeatedly scanning the full grid.
        bus_index = np.asarray(ds[cfg["bus_index_var"]].values)
        weights = np.asarray(ds[cfg["weight_var"]].values)
        valid = np.isfinite(weights) & (weights > 0.0) & (bus_index >= 0)
        row_ix, col_ix = np.where(valid)
        bus_ids = bus_index[row_ix, col_ix].astype(int)
        weight_values = weights[row_ix, col_ix].astype(float)
        denom = np.bincount(bus_ids, weights=weight_values, minlength=n_buses)
        outputs[technology] = {
            "row_ix": row_ix.astype(int),
            "col_ix": col_ix.astype(int),
            "bus_index": bus_ids,
            "cell_capacity_weights": weight_values,
            "bus_capacity_sum": denom,
        }
    return outputs


def normalize_cf_dataarray(cf: xr.DataArray, source: Path) -> xr.DataArray:
    if set(cf.dims) != {"time", "y", "x"}:
        raise ValueError(f"Unexpected CF dims in {source}: {list(cf.dims)}")
    if tuple(cf.dims) != ("time", "y", "x"):
        cf = cf.transpose("time", "y", "x")
    return cf


def timestamp_to_hour_index(timestamp: pd.Timestamp, weather_year: int) -> int:
    year_start = pd.Timestamp(year=weather_year, month=1, day=1)
    delta = timestamp - year_start
    return int(delta.total_seconds() // 3600)


def load_cf_block(cf: xr.DataArray, positions: list[int]) -> np.ndarray:
    if not positions:
        return np.empty((0, int(cf.sizes["y"]), int(cf.sizes["x"])), dtype=np.float64)
    return np.asarray(cf.isel(time=np.asarray(positions, dtype=int)).values, dtype=np.float64)


def compute_bus_cf(
    cf_slice: np.ndarray,
    tech_data: dict[str, Any],
    n_buses: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = cf_slice[tech_data["row_ix"], tech_data["col_ix"]]
    finite = np.isfinite(values)
    # Bus capacity factors are technical-potential-weighted means over all
    # eligible raster cells assigned to the reduced bus.
    numer = np.bincount(
        tech_data["bus_index"][finite],
        weights=values[finite] * tech_data["cell_capacity_weights"][finite],
        minlength=n_buses,
    )
    valid_weight_sum = np.bincount(
        tech_data["bus_index"][finite],
        weights=tech_data["cell_capacity_weights"][finite],
        minlength=n_buses,
    )
    cf_bus = np.full(n_buses, np.nan, dtype=float)
    valid = valid_weight_sum > 0.0
    cf_bus[valid] = numer[valid] / valid_weight_sum[valid]
    return cf_bus, valid_weight_sum


def capacity_fallback_generation(
    national_generation: float,
    country_indices: np.ndarray,
    scenario_capacity: np.ndarray,
) -> tuple[np.ndarray, bool]:
    bus_generation = np.zeros(len(country_indices), dtype=float)
    if national_generation <= 0.0 or len(country_indices) == 0:
        return bus_generation, False

    country_capacity = scenario_capacity[country_indices]
    active = country_capacity > 0.0
    if not active.any():
        return bus_generation, False

    active_capacity = country_capacity[active]
    total_capacity = float(active_capacity.sum())
    if total_capacity <= 0.0:
        return bus_generation, False

    bus_generation[active] = national_generation * (active_capacity / total_capacity)
    return bus_generation, True


def scenario_bus_capacity(bus_capacity: pd.DataFrame, bus_lookup: pd.DataFrame) -> pd.DataFrame:
    merged = bus_capacity.merge(bus_lookup[["bus_id", "bus_index"]], on="bus_id", how="left")
    merged["scenario_capacity_mw"] = pd.to_numeric(merged["scenario_capacity_mw"], errors="coerce").fillna(0.0)
    merged = merged[merged["bus_index"].notna()].copy()
    merged["bus_index"] = merged["bus_index"].astype(int)
    return merged


def generation_for_technology(
    technology: str,
    generation_rows: pd.DataFrame,
    cf_root: Path,
    tech_data: dict[str, Any],
    bus_capacity: pd.DataFrame,
    bus_lookup: pd.DataFrame,
    cluster_map,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = TECH_CONFIG[technology]
    n_buses = len(bus_lookup)
    if generation_rows.empty:
        return [], []

    scenario_bus = bus_capacity[bus_capacity["technology"] == technology].copy()
    scenario_bus = scenario_bus.set_index("bus_index")
    bus_country = bus_lookup.set_index("bus_index")["country"]
    bus_ids = bus_lookup.set_index("bus_index")["bus_id"]
    scenario_capacity = np.zeros(n_buses, dtype=float)
    if not scenario_bus.empty:
        scenario_capacity[scenario_bus.index.to_numpy(int)] = scenario_bus["scenario_capacity_mw"].to_numpy(float)

    generation_records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for weather_year, group in generation_rows.groupby("weather_year", sort=True):
        cf_path = cf_root / cfg["weather_subdir"] / f"{cfg['prefix']}{int(weather_year)}.nc"
        if not cf_path.exists():
            LOG.warning("missing CF file for %s %s", technology, weather_year)
            continue
        cached_raw_generation: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        with xr.open_dataset(cf_path, engine="netcdf4") as ds:
            cf = normalize_cf_dataarray(ds["cf"], cf_path)
            max_time_index = int(cf.sizes["time"])
            positions_by_timestamp: dict[pd.Timestamp, int] = {}
            for timestamp in group["peak_timestamp"]:
                ts = pd.Timestamp(timestamp)
                time_pos = timestamp_to_hour_index(ts, int(weather_year))
                if 0 <= time_pos < max_time_index:
                    positions_by_timestamp[ts] = time_pos
            unique_positions = sorted(set(positions_by_timestamp.values()))
            if not unique_positions:
                continue
            cf_block = load_cf_block(cf, unique_positions)
            for idx, time_pos in enumerate(unique_positions):
                cf_bus, valid_weight_sum = compute_bus_cf(cf_block[idx], tech_data, n_buses)
                raw_generation = np.zeros(n_buses, dtype=float)
                valid_profile = valid_weight_sum > 0.0
                raw_generation[valid_profile] = cf_bus[valid_profile] * tech_data["bus_capacity_sum"][valid_profile]
                cached_raw_generation[time_pos] = (cf_bus, raw_generation, valid_profile)

        for row in group.itertuples(index=False):
            timestamp = pd.Timestamp(row.peak_timestamp)
            time_pos = positions_by_timestamp.get(timestamp)
            if time_pos is None:
                LOG.warning("timestamp %s not found in %s", timestamp, cf_path)
                continue
            cf_bus, raw_generation, valid_profile = cached_raw_generation[time_pos]

            model_country = row.country_model
            country_indices = bus_country[bus_country == model_country].index.to_numpy(int)
            national_generation = float(getattr(row, cfg["column"]))
            country_profile_available = bool(valid_profile[country_indices].any())
            country_raw_total = float(raw_generation[country_indices].sum()) if country_profile_available else 0.0
            if country_profile_available and country_raw_total > 0.0:
                # Preserve the TYNDP national generation level and use the
                # self-generated raster profile only for the nodal distribution.
                scale_factor = national_generation / country_raw_total
                fallback_required = False
                distribution_mode = "cf_scaled"
                scaled_country_generation = raw_generation[country_indices] * scale_factor
            elif national_generation == 0.0:
                scale_factor = 0.0
                fallback_required = False
                distribution_mode = "zero_national_generation"
                scaled_country_generation = np.zeros(len(country_indices), dtype=float)
            elif country_profile_available:
                scale_factor = np.nan
                fallback_required = False
                distribution_mode = "profile_zero_no_fallback"
                scaled_country_generation = np.zeros(len(country_indices), dtype=float)
            else:
                scale_factor = np.nan
                # This branch is deliberately conservative: without a usable
                # profile, generation is spread by installed capacity rather than
                # inventing weather structure.
                scaled_country_generation, fallback_required = capacity_fallback_generation(
                    national_generation,
                    country_indices,
                    scenario_capacity,
                )
                distribution_mode = "capacity_fallback" if fallback_required else "no_profile_no_capacity"

            country_label = label_for_country(model_country, cluster_map)
            source_countries = ",".join(sources_for_country(model_country, cluster_map))
            active_bus_count = len(country_indices)
            for local_index, bus_index in enumerate(country_indices):
                bus_generation_raw = float(raw_generation[bus_index])
                scaled_generation = float(scaled_country_generation[local_index])
                generation_records.append(
                    {
                        "technology": technology,
                        "country": model_country,
                        "country_label": country_label,
                        "source_countries": source_countries,
                        "bus": bus_ids[bus_index],
                        "timestamp": timestamp.isoformat(),
                        "weather_year": int(weather_year),
                        "week": row.week,
                        "national_generation_mw": national_generation,
                        "bus_installed_capacity_mw": float(scenario_capacity[bus_index]),
                        "bus_cf": float(cf_bus[bus_index]) if np.isfinite(cf_bus[bus_index]) else np.nan,
                        "raw_bus_generation_mw": bus_generation_raw,
                        "scale_factor": float(scale_factor) if np.isfinite(scale_factor) else np.nan,
                        "scaled_bus_generation_mw": scaled_generation,
                        "fallback_required": fallback_required,
                        "distribution_mode": distribution_mode,
                    }
                )

            diagnostics.append(
                {
                    "technology": technology,
                    "country": model_country,
                    "country_label": country_label,
                    "source_countries": source_countries,
                    "timestamp": timestamp.isoformat(),
                    "weather_year": int(weather_year),
                    "week": row.week,
                    "national_generation_mw": national_generation,
                    "raw_country_generation_mw": country_raw_total,
                    "scale_factor": float(scale_factor) if np.isfinite(scale_factor) else np.nan,
                    "fallback_required": fallback_required,
                    "country_profile_available": country_profile_available,
                    "distribution_mode": distribution_mode,
                    "active_buses": active_bus_count,
                    "total_bus_capacity_mw": float(scenario_capacity[country_indices].sum()),
                }
            )

    return generation_records, diagnostics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = resolve_settings(parse_args())
    out_paths = output_paths(settings["output_dir"])

    cluster_map = read_cluster_map(settings)
    excluded_source_countries = read_excluded_countries(settings.get("excluded_countries_csv"))
    generation_rows = load_generation_rows(
        settings["generation_long_csv"],
        cluster_map,
        int(settings["start_year"]),
        int(settings["end_year"]),
        int(settings["target_year"]),
        excluded_source_countries,
    )
    ds, bus_lookup, bus_capacity = load_capacity_outputs(settings)
    tech_data = precompute_cell_weights(ds, bus_lookup)
    bus_capacity = scenario_bus_capacity(bus_capacity, bus_lookup)

    generation_records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for technology in ("pv", "onwind", "offwind"):
        tech_records, tech_diag = generation_for_technology(
            technology,
            generation_rows,
            settings["weather_root"],
            tech_data[technology],
            bus_capacity,
            bus_lookup,
            cluster_map,
        )
        generation_records.extend(tech_records)
        diagnostics.extend(tech_diag)

    pd.DataFrame(generation_records).to_csv(out_paths["generation_csv"], index=False)
    pd.DataFrame(diagnostics).to_csv(out_paths["diagnostics_csv"], index=False)
    manifest = {
        "scenario_name": settings["scenario_name"],
        "project_root": str(settings["project_root"]),
        "network_dir": str(settings["network_dir"]),
        "simulation_dir": str(settings["simulation_dir"]) if settings.get("simulation_dir") is not None else None,
        "target_year": settings["target_year"],
        "start_year": settings["start_year"],
        "end_year": settings["end_year"],
        "atlite_case_dir": str(settings["atlite_case_dir"]) if settings.get("atlite_case_dir") is not None else None,
        "generation_long_csv": str(settings["generation_long_csv"]),
        "excluded_countries_csv": str(settings["excluded_countries_csv"]) if settings.get("excluded_countries_csv") else None,
        "excluded_source_countries": sorted(excluded_source_countries),
        "weather_root": str(settings["weather_root"]),
        "capacity_output_dir": str(settings["capacity_output_dir"]),
        "cells_nc": str(settings["cells_nc"]),
        "bus_lookup_csv": str(settings["bus_lookup_csv"]),
        "bus_capacity_csv": str(settings["bus_capacity_csv"]),
        "outputs": {key: str(value) for key, value in out_paths.items()},
    }
    write_json(out_paths["manifest_json"], manifest)
    LOG.info("done")


if __name__ == "__main__":
    main()
