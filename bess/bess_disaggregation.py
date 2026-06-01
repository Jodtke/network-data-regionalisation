from __future__ import annotations

"""Regionalise TYNDP battery storage targets to reduced-grid buses.

Battery storage is allocated by a combined siting basis rather than by a hard
split between existing and new capacity. Existing battery sites represent
revealed siting information, while renewable scenario capacity identifies buses
where storage is likely to be useful for balancing local variable generation.

If neither basis exists for a country, the module falls back to static load
shares and finally to a uniform split over eligible buses. Charging capacity,
discharging capacity, storage energy, and effective capacity are scaled from the
same bus shares to keep the storage parametrisation internally consistent.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
COMMON_DIR = ROOT_DIR / "powerplants"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from powerplants_common import (
    DEFAULT_PROJECT_ROOT,
    DEFAULT_SCENARIO,
    DEFAULT_TYNDP_INPUT_DIR,
    build_manifest,
    filter_to_tyndp_target_year,
    find_column,
    infer_target_year,
    load_bus_country_membership,
    load_load_shares,
    load_network_plant_rows,
    load_res_bus_capacity,
    load_yaml_config,
    norm_country,
    numeric_column,
    read_csv_auto,
    resolve_path,
    source_country_mapping_from_load_shares,
    validate_tyndp_target_year,
    write_csv,
    write_json,
)

LOG = logging.getLogger(__name__)
DEFAULT_RES_TECHNOLOGIES = ("pv", "onwind")
BATTERY_GROUP = "battery"


def require_existing_file(path: Path | None, setting_name: str, hint: str) -> Path:
    if path is None:
        raise ValueError(f"{setting_name} is required. {hint}")
    try:
        exists = path.exists()
    except OSError as exc:
        raise FileNotFoundError(f"Could not access configured {setting_name}: {path}. {hint}") from exc
    if not exists:
        raise FileNotFoundError(f"Configured {setting_name} does not exist: {path}. {hint}")
    return path


def default_output_dir(project_root: Path, network_dir: Path) -> Path:
    return project_root / "bess" / network_dir.parent.name / network_dir.name


def normalize_res_technology(value: Any) -> str:
    tech = str(value or "").strip().lower()
    if not tech:
        return ""
    if "offshore" in tech and "wind" in tech:
        return "offwind"
    if "offwind" in tech:
        return "offwind"
    if "onshore" in tech and "wind" in tech:
        return "onwind"
    if "onwind" in tech:
        return "onwind"
    if "solar" in tech or "photovoltaic" in tech:
        return "pv"
    return tech


def normalize_scenario(value: Any) -> str:
    return str(value or "").strip().casefold()


def parse_technology_list(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return DEFAULT_RES_TECHNOLOGIES
    if isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        raw = str(value).replace(";", ",").split(",")
    normalized = []
    for entry in raw:
        tech = normalize_res_technology(entry)
        if tech and tech not in normalized:
            normalized.append(tech)
    return tuple(normalized) or DEFAULT_RES_TECHNOLOGIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Disaggregate national BESS targets to buses.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--network-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-year", type=int, default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--countries", nargs="*", default=None)
    parser.add_argument("--bess-csv", type=Path, default=None)
    parser.add_argument("--tyndp-input-dir", type=Path, default=None)
    parser.add_argument("--buses-csv", type=Path, default=None)
    parser.add_argument("--plants-csv", type=Path, default=None)
    parser.add_argument("--buses-with-clusters-csv", type=Path, default=None)
    parser.add_argument("--load-shares-csv", type=Path, default=None)
    parser.add_argument("--res-bus-capacity-csv", type=Path, default=None)
    parser.add_argument("--country-allocation-mode", default=None)
    parser.add_argument("--res-weight-column", default=None)
    parser.add_argument("--res-technologies", default=None)
    return parser.parse_args()


def default_settings() -> dict[str, Any]:
    return {
        "scenario_name": "bess_disaggregation",
        "project_root": str(DEFAULT_PROJECT_ROOT),
        "network_dir": None,
        "output_dir": None,
        "target_year": None,
        "scenario": DEFAULT_SCENARIO,
        "countries": None,
        "tyndp_input_dir": str(DEFAULT_TYNDP_INPUT_DIR),
        "bess_csv": None,
        "buses_csv": None,
        "plants_csv": None,
        "buses_with_clusters_csv": None,
        "load_shares_csv": None,
        "res_bus_capacity_csv": None,
        "country_allocation_mode": "bus_country",
        "res_weight_column": "scenario_capacity_mw",
        "res_technologies": list(DEFAULT_RES_TECHNOLOGIES),
    }


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = default_settings()
    config_dir = None
    if args.config is not None:
        cfg_path = resolve_path(args.config, Path.cwd())
        assert cfg_path is not None
        settings.update(load_yaml_config(cfg_path))
        config_dir = cfg_path.parent
    for key, value in {
        "project_root": args.project_root,
        "network_dir": args.network_dir,
        "output_dir": args.output_dir,
        "target_year": args.target_year,
        "scenario": args.scenario,
        "countries": args.countries,
        "bess_csv": args.bess_csv,
        "tyndp_input_dir": args.tyndp_input_dir,
        "buses_csv": args.buses_csv,
        "plants_csv": args.plants_csv,
        "buses_with_clusters_csv": args.buses_with_clusters_csv,
        "load_shares_csv": args.load_shares_csv,
        "res_bus_capacity_csv": args.res_bus_capacity_csv,
        "country_allocation_mode": args.country_allocation_mode,
        "res_weight_column": args.res_weight_column,
        "res_technologies": args.res_technologies,
    }.items():
        if value is not None:
            settings[key] = value

    project_root = resolve_path(settings["project_root"], config_dir) or DEFAULT_PROJECT_ROOT
    settings["project_root"] = project_root
    network_dir = resolve_path(settings["network_dir"], project_root)
    if network_dir is None:
        raise ValueError("network_dir is required.")
    settings["network_dir"] = network_dir
    target_year = infer_target_year(network_dir, settings.get("target_year"))
    if target_year is None:
        raise ValueError("Could not infer target_year from network_dir. Set target_year explicitly.")
    settings["target_year"] = validate_tyndp_target_year(target_year)

    tyndp_input_dir = resolve_path(settings.get("tyndp_input_dir"), project_root) or DEFAULT_TYNDP_INPUT_DIR
    settings["tyndp_input_dir"] = tyndp_input_dir
    settings["bess_csv"] = resolve_path(
        settings.get("bess_csv") or (tyndp_input_dir / f"bess_power_{int(target_year)}_tyndp2024.csv"),
        project_root,
    )
    settings["buses_csv"] = resolve_path(settings.get("buses_csv") or (network_dir / "buses.csv"), project_root)
    settings["plants_csv"] = resolve_path(settings.get("plants_csv") or (network_dir / "plants.csv"), project_root)
    settings["buses_with_clusters_csv"] = resolve_path(
        settings.get("buses_with_clusters_csv") or (network_dir / "buses_with_clusters.csv"),
        project_root,
    )
    settings["load_shares_csv"] = resolve_path(settings.get("load_shares_csv"), project_root)
    settings["res_bus_capacity_csv"] = resolve_path(settings.get("res_bus_capacity_csv"), project_root)
    settings["res_technologies"] = list(parse_technology_list(settings.get("res_technologies")))
    settings["output_dir"] = resolve_path(settings.get("output_dir"), project_root) or default_output_dir(project_root, network_dir)
    return settings


def load_bess_country_targets(
    csv_path: Path,
    *,
    ref_year: int,
    scenario: str,
    country_map: Mapping[str, str],
    target_countries: list[str] | None,
) -> pd.DataFrame:
    df = read_csv_auto(csv_path)
    country_col = find_column(df, ("country", "country_code", "country_model", "area", "zone"))
    if country_col is None:
        raise KeyError(f"{csv_path.name} missing a country column.")
    year_col = find_column(df, ("year", "target_year", "ref_year", "reference_year"))
    scenario_col = find_column(df, ("scenario", "storyline", "scenario_name"))
    discharge_col = find_column(df, ("discharging_power_mw", "discharge_power_mw", "power_mw", "capacity_mw"))
    if discharge_col is None:
        raise KeyError(f"{csv_path.name} missing discharging power column.")
    charge_col = find_column(df, ("charging_power_mw", "charge_power_mw", "charging_mw"))
    energy_col = find_column(df, ("capacity_mwh", "energy_mwh", "storage_capacity_mwh"))
    eff_col = find_column(df, ("eff", "efficiency", "roundtrip_efficiency", "round_trip_efficiency"))

    work = filter_to_tyndp_target_year(
        df,
        csv_path,
        ref_year=ref_year,
        year_col=year_col,
        context="BESS country target",
    )
    work["source_country"] = work[country_col].map(norm_country)
    if scenario_col is not None:
        requested = normalize_scenario(scenario)
        work["_scenario_key"] = work[scenario_col].map(normalize_scenario)
        if requested and work["_scenario_key"].eq(requested).any():
            work = work[work["_scenario_key"].eq(requested)].copy()
    work["country"] = work["source_country"].map(lambda c: norm_country(country_map.get(c, c)))
    work["discharging_power_mw"] = numeric_column(work[discharge_col]).fillna(0.0)
    work["charging_power_mw"] = numeric_column(work[charge_col]).fillna(0.0) if charge_col is not None else work["discharging_power_mw"]
    work["capacity_mwh"] = numeric_column(work[energy_col]).fillna(0.0) if energy_col is not None else 0.0
    work["eff"] = numeric_column(work[eff_col]).fillna(1.0) if eff_col is not None else 1.0
    work["eff"] = work["eff"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.0)
    work["effective_capacity_mw"] = work["discharging_power_mw"] * work["eff"]
    work = work[work["country"].astype(str).str.strip().ne("")].copy()
    if target_countries:
        work = work[work["country"].isin(set(target_countries))].copy()

    if work.empty:
        return pd.DataFrame(
            columns=[
                "country",
                "target_year",
                "scenario",
                "discharging_power_mw",
                "charging_power_mw",
                "capacity_mwh",
                "eff",
                "effective_capacity_mw",
                "source_countries",
            ]
        )

    grouped = (
        work.groupby("country", as_index=False)
        .agg(
            discharging_power_mw=("discharging_power_mw", "sum"),
            charging_power_mw=("charging_power_mw", "sum"),
            capacity_mwh=("capacity_mwh", "sum"),
            effective_capacity_mw=("effective_capacity_mw", "sum"),
            source_countries=("source_country", lambda values: ",".join(sorted(set(str(value) for value in values if str(value).strip())))),
        )
        .sort_values("country")
        .reset_index(drop=True)
    )
    grouped["eff"] = np.divide(
        grouped["effective_capacity_mw"],
        grouped["discharging_power_mw"],
        out=np.ones(len(grouped), dtype=float),
        where=grouped["discharging_power_mw"].to_numpy(dtype=float) > 0.0,
    )
    grouped["target_year"] = int(ref_year)
    grouped["scenario"] = str(scenario)
    return grouped[
        [
            "country",
            "target_year",
            "scenario",
            "discharging_power_mw",
            "charging_power_mw",
            "capacity_mwh",
            "eff",
            "effective_capacity_mw",
            "source_countries",
        ]
    ]


def build_current_battery_basis(plant_rows: pd.DataFrame) -> pd.DataFrame:
    if plant_rows.empty:
        return pd.DataFrame(columns=["country", "bus_id", "current_battery_capacity_mw", "current_battery_share"])
    basis = plant_rows[plant_rows["fueltype"].astype(str).str.upper().eq("BATTERY")].copy()
    if basis.empty:
        return pd.DataFrame(columns=["country", "bus_id", "current_battery_capacity_mw", "current_battery_share"])
    basis = (
        basis.groupby(["country", "bus_id"], as_index=False)["capacity_mw"]
        .sum()
        .rename(columns={"capacity_mw": "current_battery_capacity_mw"})
    )
    total = basis.groupby("country")["current_battery_capacity_mw"].transform("sum")
    basis["current_battery_share"] = np.divide(
        basis["current_battery_capacity_mw"],
        total,
        out=np.zeros(len(basis), dtype=float),
        where=total.to_numpy(dtype=float) > 0.0,
    )
    return basis


def build_res_weight_basis(
    res_bus_capacity: pd.DataFrame,
    *,
    weight_column: str,
    technologies: tuple[str, ...],
) -> pd.DataFrame:
    if weight_column not in res_bus_capacity.columns:
        raise KeyError(f"res_bus_capacity_csv missing configured weight column: {weight_column}")
    basis = res_bus_capacity.copy()
    basis["technology"] = basis["technology"].map(normalize_res_technology)
    basis = basis[basis["technology"].isin(set(technologies))].copy()
    if basis.empty:
        return pd.DataFrame(columns=["country", "bus_id", "res_weight_mw", "res_weight_share", "res_technologies"])
    basis["res_weight_mw"] = pd.to_numeric(basis[weight_column], errors="coerce").fillna(0.0)
    basis = basis[basis["res_weight_mw"] > 0.0].copy()
    if basis.empty:
        return pd.DataFrame(columns=["country", "bus_id", "res_weight_mw", "res_weight_share", "res_technologies"])
    grouped = (
        basis.groupby(["country", "bus_id"], as_index=False)
        .agg(
            res_weight_mw=("res_weight_mw", "sum"),
            res_technologies=("technology", lambda values: ",".join(sorted(set(str(value) for value in values if str(value).strip())))),
        )
        .sort_values(["country", "bus_id"])
        .reset_index(drop=True)
    )
    total = grouped.groupby("country")["res_weight_mw"].transform("sum")
    grouped["res_weight_share"] = np.divide(
        grouped["res_weight_mw"],
        total,
        out=np.zeros(len(grouped), dtype=float),
        where=total.to_numpy(dtype=float) > 0.0,
    )
    return grouped


def build_load_share_basis(load_shares: pd.DataFrame) -> pd.DataFrame:
    if load_shares.empty:
        return pd.DataFrame(columns=["country", "bus_id", "load_share"])
    basis = load_shares[["country", "bus_id", "load_share"]].copy()
    basis["country"] = basis["country"].map(norm_country)
    basis["bus_id"] = basis["bus_id"].astype(str)
    basis["load_share"] = pd.to_numeric(basis["load_share"], errors="coerce").fillna(0.0)
    basis = basis[basis["load_share"] > 0.0].copy()
    if basis.empty:
        return pd.DataFrame(columns=["country", "bus_id", "load_share"])
    basis = basis.groupby(["country", "bus_id"], as_index=False)["load_share"].sum()
    total = basis.groupby("country")["load_share"].transform("sum")
    basis["load_share"] = np.divide(
        basis["load_share"],
        total,
        out=np.zeros(len(basis), dtype=float),
        where=total.to_numpy(dtype=float) > 0.0,
    )
    return basis


def restrict_to_eligible_buses(frame: pd.DataFrame, eligible_buses: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["country"] = out["country"].map(norm_country)
    out["bus_id"] = out["bus_id"].astype(str)
    allowed = eligible_buses[["country", "bus_id"]].drop_duplicates().copy()
    allowed["country"] = allowed["country"].map(norm_country)
    allowed["bus_id"] = allowed["bus_id"].astype(str)
    return out.merge(allowed, how="inner", on=["country", "bus_id"])


def renormalize_country_shares(frame: pd.DataFrame, *, value_col: str, share_col: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce").fillna(0.0)
    total = out.groupby("country")[value_col].transform("sum")
    out[share_col] = np.divide(
        out[value_col],
        total,
        out=np.zeros(len(out), dtype=float),
        where=total.to_numpy(dtype=float) > 0.0,
    )
    return out


def positive_share_rows(frame: pd.DataFrame, value_col: str) -> list[tuple[str, float]]:
    if frame.empty:
        return []
    work = frame[["bus_id", value_col]].copy()
    work["bus_id"] = work["bus_id"].astype(str)
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)
    work = work[work[value_col] > 0.0].copy()
    if work.empty:
        return []
    total = float(work[value_col].sum())
    if total <= 0.0:
        return []
    work["share"] = work[value_col] / total
    return [(str(row.bus_id), float(row.share)) for row in work.itertuples(index=False)]


def uniform_share_rows(bus_ids: list[str]) -> list[tuple[str, float]]:
    unique_bus_ids = sorted({str(bus_id) for bus_id in bus_ids if str(bus_id).strip()})
    if not unique_bus_ids:
        return []
    share = 1.0 / float(len(unique_bus_ids))
    return [(bus_id, share) for bus_id in unique_bus_ids]


def scale_share_rows(target_mw: float, share_rows: list[tuple[str, float]]) -> dict[str, float]:
    if target_mw <= 0.0 or not share_rows:
        return {}
    return {str(bus_id): float(target_mw) * float(share) for bus_id, share in share_rows}


def choose_bess_share_rows(
    *,
    country: str,
    basis_rows: list[tuple[str, float]],
    load_rows: list[tuple[str, float]],
    uniform_rows: list[tuple[str, float]],
) -> tuple[list[tuple[str, float]], str]:
    # Battery siting is intentionally tied to today's batteries and RES-heavy
    # buses first. Load and uniform shares are only fallbacks for countries where
    # no such siting signal is available.
    if basis_rows:
        return basis_rows, "battery_res_capacity"
    if load_rows:
        return load_rows, "load_share_fallback"
    if uniform_rows:
        return uniform_rows, "uniform_fallback"
    LOG.warning("No eligible buses found for BESS allocation in %s", country)
    return [], "unallocated"


def allocate_country(
    *,
    target_row: Mapping[str, Any],
    country_bus_ids: list[str],
    current_basis: pd.DataFrame,
    res_basis: pd.DataFrame,
    load_basis: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    country = str(target_row["country"])
    target_discharge = float(target_row.get("discharging_power_mw", 0.0) or 0.0)
    target_charge = float(target_row.get("charging_power_mw", 0.0) or 0.0)
    target_energy = float(target_row.get("capacity_mwh", 0.0) or 0.0)
    eff = float(target_row.get("eff", 1.0) or 1.0)
    target_effective = float(target_row.get("effective_capacity_mw", target_discharge * eff) or 0.0)

    current_rows_df = current_basis[current_basis["country"].eq(country)].copy()
    res_rows_df = res_basis[res_basis["country"].eq(country)].copy()
    load_rows_df = load_basis[load_basis["country"].eq(country)].copy()

    load_share_rows = positive_share_rows(load_rows_df, "load_share")
    uniform_rows = uniform_share_rows(country_bus_ids)

    current_total = float(current_rows_df["current_battery_capacity_mw"].sum()) if not current_rows_df.empty else 0.0
    res_total = float(res_rows_df["res_weight_mw"].sum()) if not res_rows_df.empty else 0.0

    basis = pd.DataFrame({"bus_id": sorted(set(country_bus_ids))})
    current_values = current_rows_df[["bus_id", "current_battery_capacity_mw"]] if not current_rows_df.empty else pd.DataFrame(columns=["bus_id", "current_battery_capacity_mw"])
    res_values = res_rows_df[["bus_id", "res_weight_mw"]] if not res_rows_df.empty else pd.DataFrame(columns=["bus_id", "res_weight_mw"])
    basis = basis.merge(current_values, how="left", on="bus_id").merge(res_values, how="left", on="bus_id")
    basis[["current_battery_capacity_mw", "res_weight_mw"]] = basis[["current_battery_capacity_mw", "res_weight_mw"]].fillna(0.0)
    basis["combined_bess_weight_mw"] = basis["current_battery_capacity_mw"] + basis["res_weight_mw"]
    basis_share_rows = positive_share_rows(basis, "combined_bess_weight_mw")
    share_rows, allocation_source = choose_bess_share_rows(
        country=country,
        basis_rows=basis_share_rows,
        load_rows=load_share_rows,
        uniform_rows=uniform_rows,
    )
    allocation = scale_share_rows(target_discharge, share_rows)

    bus_ids = sorted(set(country_bus_ids) | set(allocation))
    if not bus_ids and target_discharge > 0.0:
        diag = {
            "country": country,
            "target_discharging_power_mw": target_discharge,
            "target_effective_capacity_mw": target_effective,
            "allocated_discharging_power_mw": 0.0,
            "allocated_effective_capacity_mw": 0.0,
            "unallocated_discharging_power_mw": target_discharge,
            "current_battery_basis_mw": current_total,
            "res_weight_basis_mw": res_total,
            "current_allocated_mw": 0.0,
            "added_allocated_mw": 0.0,
            "added_allocation_source": allocation_source,
            "n_allocated_buses": 0,
        }
        return pd.DataFrame(), diag

    current_lookup = current_rows_df.set_index("bus_id")["current_battery_capacity_mw"].to_dict() if not current_rows_df.empty else {}
    current_share_lookup = current_rows_df.set_index("bus_id")["current_battery_share"].to_dict() if not current_rows_df.empty else {}
    res_lookup = res_rows_df.set_index("bus_id")["res_weight_mw"].to_dict() if not res_rows_df.empty else {}
    res_share_lookup = res_rows_df.set_index("bus_id")["res_weight_share"].to_dict() if not res_rows_df.empty else {}
    load_share_lookup = load_rows_df.set_index("bus_id")["load_share"].to_dict() if not load_rows_df.empty else {}

    weight_lookup = basis.set_index("bus_id")["combined_bess_weight_mw"].to_dict() if not basis.empty else {}
    total_allocated_discharge = float(sum(allocation.values()))
    country_share_denominator = target_discharge if target_discharge > 0.0 else total_allocated_discharge
    rows: list[dict[str, Any]] = []
    for bus_id in bus_ids:
        discharging_power_mw = float(allocation.get(bus_id, 0.0))
        if discharging_power_mw <= 0.0:
            continue
        weight = float(weight_lookup.get(bus_id, 0.0))
        current_weight = float(current_lookup.get(bus_id, 0.0))
        res_weight = float(res_lookup.get(bus_id, 0.0))
        # Power, energy, and effective capacity use the same country share. This
        # avoids creating artificial bus-level duration or efficiency differences.
        current_part = discharging_power_mw * current_weight / weight if weight > 0.0 else 0.0
        added_part = discharging_power_mw * res_weight / weight if weight > 0.0 else discharging_power_mw
        country_share = discharging_power_mw / country_share_denominator if country_share_denominator > 0.0 else 0.0
        charging_power_mw = float(target_charge) * country_share
        capacity_mwh = float(target_energy) * country_share
        effective_capacity_mw = discharging_power_mw * eff
        rows.append(
            {
                "target_year": int(target_row["target_year"]),
                "scenario": str(target_row["scenario"]),
                "country": country,
                "bus_id": str(bus_id),
                "capacity_share": country_share,
                "discharging_power_mw": discharging_power_mw,
                "charging_power_mw": charging_power_mw,
                "capacity_mwh": capacity_mwh,
                "eff": eff,
                "effective_capacity_mw": effective_capacity_mw,
                "capacity_mw": effective_capacity_mw,
                "current_battery_capacity_mw": float(current_lookup.get(bus_id, 0.0)),
                "current_battery_share": float(current_share_lookup.get(bus_id, 0.0)),
                "res_weight_mw": float(res_lookup.get(bus_id, 0.0)),
                "res_weight_share": float(res_share_lookup.get(bus_id, 0.0)),
                "load_share_fallback": float(load_share_lookup.get(bus_id, 0.0)),
                "current_allocated_mw": current_part,
                "added_allocated_mw": added_part,
                "current_allocation_source": "current_battery" if current_part > 0.0 else "",
                "added_allocation_source": allocation_source if added_part > 0.0 else "",
                "technology": BATTERY_GROUP,
                "n_units": 1,
                "unit_capacity_mw": effective_capacity_mw,
                "unit_id": f"battery|{country}|{bus_id}",
            }
        )

    out = pd.DataFrame(rows).sort_values(["country", "bus_id"]).reset_index(drop=True)
    diag = {
        "country": country,
        "target_discharging_power_mw": target_discharge,
        "target_effective_capacity_mw": target_effective,
        "allocated_discharging_power_mw": float(out["discharging_power_mw"].sum()) if not out.empty else 0.0,
        "allocated_effective_capacity_mw": float(out["effective_capacity_mw"].sum()) if not out.empty else 0.0,
        "unallocated_discharging_power_mw": float(max(0.0, target_discharge - (float(out["discharging_power_mw"].sum()) if not out.empty else 0.0))),
        "current_battery_basis_mw": current_total,
        "res_weight_basis_mw": res_total,
        "current_allocated_mw": float(out["current_allocated_mw"].sum()) if not out.empty else 0.0,
        "added_allocated_mw": float(out["added_allocated_mw"].sum()) if not out.empty else 0.0,
        "added_allocation_source": allocation_source,
        "n_allocated_buses": int(len(out)),
    }
    return out, diag


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = resolve_settings(parse_args())
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    buses_csv = require_existing_file(
        settings.get("buses_csv"),
        "buses_csv",
        "Provide the reduced-grid buses.csv for the target network case.",
    )
    plants_csv = require_existing_file(
        settings.get("plants_csv"),
        "plants_csv",
        "Provide the reduced-grid plants.csv for the target network case.",
    )
    bess_csv = require_existing_file(
        settings.get("bess_csv"),
        "bess_csv",
        "Prepare bess_power_<year>_tyndp2024.csv first.",
    )
    res_bus_capacity_csv = require_existing_file(
        settings.get("res_bus_capacity_csv"),
        "res_bus_capacity_csv",
        "Run the RES bus capacity preprocessing for the same network case.",
    )
    buses_with_clusters_csv = settings.get("buses_with_clusters_csv")
    load_shares_csv = require_existing_file(
        settings.get("load_shares_csv"),
        "load_shares_csv",
        "Provide the load-share output so loadless DC converter terminals can be excluded from BESS allocation.",
    )

    bus_country_membership = load_bus_country_membership(
        buses_csv=buses_csv,
        buses_with_clusters_csv=buses_with_clusters_csv,
        country_allocation_mode=str(settings.get("country_allocation_mode", "bus_country")),
    )
    plant_rows = load_network_plant_rows(
        plants_csv=plants_csv,
        buses_csv=buses_csv,
        bus_country_membership=bus_country_membership,
    )
    load_basis = build_load_share_basis(load_load_shares(load_shares_csv))
    allowed_buses = load_basis[["country", "bus_id"]].drop_duplicates().copy()

    current_battery_basis = build_current_battery_basis(plant_rows)
    current_battery_basis = restrict_to_eligible_buses(current_battery_basis, allowed_buses)
    current_battery_basis = renormalize_country_shares(
        current_battery_basis,
        value_col="current_battery_capacity_mw",
        share_col="current_battery_share",
    )

    res_bus_capacity = load_res_bus_capacity(res_bus_capacity_csv)
    res_weight_basis = build_res_weight_basis(
        res_bus_capacity,
        weight_column=str(settings.get("res_weight_column", "scenario_capacity_mw")),
        technologies=tuple(settings.get("res_technologies") or DEFAULT_RES_TECHNOLOGIES),
    )
    res_weight_basis = restrict_to_eligible_buses(res_weight_basis, allowed_buses)
    res_weight_basis = renormalize_country_shares(
        res_weight_basis,
        value_col="res_weight_mw",
        share_col="res_weight_share",
    )
    country_map = source_country_mapping_from_load_shares(load_shares_csv)

    countries = settings.get("countries") or sorted(bus_country_membership["country"].dropna().astype(str).unique().tolist())
    countries = [norm_country(country) for country in countries]
    targets = load_bess_country_targets(
        bess_csv,
        ref_year=int(settings["target_year"]),
        scenario=str(settings["scenario"]),
        country_map=country_map,
        target_countries=countries,
    )
    targets = targets.sort_values("country").reset_index(drop=True)

    bus_ids_by_country = {
        country: sorted(group["bus_id"].astype(str).unique().tolist())
        for country, group in load_basis.groupby("country")
    }
    allocation_frames: list[pd.DataFrame] = []
    diag_rows: list[dict[str, Any]] = []
    for target_row in targets.itertuples(index=False):
        country = str(target_row.country)
        allocation, diag = allocate_country(
            target_row=target_row._asdict(),
            country_bus_ids=bus_ids_by_country.get(country, []),
            current_basis=current_battery_basis,
            res_basis=res_weight_basis,
            load_basis=load_basis,
        )
        allocation_frames.append(allocation)
        diag_rows.append(diag)

    bus_out = pd.concat(allocation_frames, ignore_index=True) if allocation_frames else pd.DataFrame(
        columns=[
            "target_year",
            "scenario",
            "country",
            "bus_id",
            "capacity_share",
            "discharging_power_mw",
            "charging_power_mw",
            "capacity_mwh",
            "eff",
            "effective_capacity_mw",
            "capacity_mw",
            "current_battery_capacity_mw",
            "current_battery_share",
            "res_weight_mw",
            "res_weight_share",
            "load_share_fallback",
            "current_allocated_mw",
            "added_allocated_mw",
            "current_allocation_source",
            "added_allocation_source",
            "technology",
            "n_units",
            "unit_capacity_mw",
            "unit_id",
        ]
    )
    diag_df = pd.DataFrame(
        diag_rows,
        columns=[
            "country",
            "target_discharging_power_mw",
            "target_effective_capacity_mw",
            "allocated_discharging_power_mw",
            "allocated_effective_capacity_mw",
            "unallocated_discharging_power_mw",
            "current_battery_basis_mw",
            "res_weight_basis_mw",
            "current_allocated_mw",
            "added_allocated_mw",
            "added_allocation_source",
            "n_allocated_buses",
        ],
    )

    outputs = {
        "bess_capacity_country_bus": output_dir / "bess_capacity_country_bus.csv",
        "bess_country_targets": output_dir / "bess_country_targets.csv",
        "bess_current_battery_basis_bus": output_dir / "bess_current_battery_basis_bus.csv",
        "bess_res_weight_basis_bus": output_dir / "bess_res_weight_basis_bus.csv",
        "bess_load_share_basis_bus": output_dir / "bess_load_share_basis_bus.csv",
        "bess_allocation_diagnostics": output_dir / "bess_allocation_diagnostics.csv",
        "manifest": output_dir / "bess_disaggregation_manifest.json",
    }
    write_csv(outputs["bess_capacity_country_bus"], bus_out)
    write_csv(outputs["bess_country_targets"], targets)
    write_csv(outputs["bess_current_battery_basis_bus"], current_battery_basis)
    write_csv(outputs["bess_res_weight_basis_bus"], res_weight_basis)
    write_csv(outputs["bess_load_share_basis_bus"], load_basis)
    write_csv(outputs["bess_allocation_diagnostics"], diag_df)
    write_json(
        outputs["manifest"],
        build_manifest(
            settings,
            outputs,
            {
                "allocated_discharging_power_mw": float(bus_out["discharging_power_mw"].sum()) if not bus_out.empty else 0.0,
                "allocated_effective_capacity_mw": float(bus_out["effective_capacity_mw"].sum()) if not bus_out.empty else 0.0,
                "n_countries": int(targets["country"].nunique()) if not targets.empty else 0,
            },
        ),
    )
    LOG.info("wrote BESS disaggregation outputs to %s", output_dir)


if __name__ == "__main__":
    main()
