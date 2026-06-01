from __future__ import annotations

"""Place generic TYNDP thermal units on reduced-grid buses.

TYNDP thermal targets are often reported as generic national capacities or unit
counts rather than plant-level decisions. This module reconstructs a plausible
nodal fleet by matching targets to current plant sites by fuel and technology,
then by fuel only, and finally by recently decommissioned sites or load-bearing
buses. The staged allocation is designed to preserve national capacity totals
while keeping the siting rules transparent and reproducible.
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from powerplants_common import (
    DEFAULT_PROJECT_ROOT,
    DEFAULT_SCENARIO,
    DEFAULT_TYNDP_INPUT_DIR,
    allocate_units_to_decommissioned_sites,
    LONG_REV_DUR_BY_TECH,
    STD_REV_DUR_BY_TECH,
    build_manifest,
    default_decommissioned_plants_csv,
    default_output_dir,
    fuel_group_from_row,
    has_chp_flag,
    infer_target_year,
    inertia_h,
    is_thermal_row,
    load_bus_country_membership,
    load_decommissioned_site_basis,
    load_load_shares,
    load_network_plant_rows,
    load_yaml_config,
    map_fuel_code,
    map_thermal_tech,
    norm_country,
    read_csv_auto,
    require_positive_load_share_buses,
    restrict_to_load_buses,
    resolve_path,
    safe_int,
    source_country_mapping_from_load_shares,
    validate_tyndp_target_year,
    write_csv,
    write_json,
)

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Disaggregate TYNDP thermal power plants to reduced network buses.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--network-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-year", type=int, default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--countries", nargs="*", default=None)
    parser.add_argument("--tyndp-thermal-csv", type=Path, default=None)
    parser.add_argument("--revision-duration-std-csv", type=Path, default=None)
    parser.add_argument("--revision-duration-long-csv", type=Path, default=None)
    parser.add_argument("--fuel-prices-csv", type=Path, default=None)
    parser.add_argument("--thermal-params-csv", type=Path, default=None)
    parser.add_argument("--buses-csv", type=Path, default=None)
    parser.add_argument("--plants-csv", type=Path, default=None)
    parser.add_argument("--buses-with-clusters-csv", type=Path, default=None)
    parser.add_argument("--load-shares-csv", type=Path, default=None)
    parser.add_argument("--decommissioned-plants-csv", type=Path, default=None)
    return parser.parse_args()


def default_settings() -> dict[str, Any]:
    return {
        "scenario_name": "thermal_powerplants_disaggregation",
        "project_root": str(DEFAULT_PROJECT_ROOT),
        "network_dir": None,
        "output_dir": None,
        "target_year": None,
        "scenario": DEFAULT_SCENARIO,
        "countries": None,
        "tyndp_input_dir": str(DEFAULT_TYNDP_INPUT_DIR),
        "tyndp_thermal_csv": None,
        "revision_duration_std_csv": None,
        "revision_duration_long_csv": None,
        "fuel_prices_csv": None,
        "thermal_params_csv": None,
        "buses_csv": None,
        "plants_csv": None,
        "buses_with_clusters_csv": None,
        "load_shares_csv": None,
        "decommissioned_plants_csv": None,
        "decommissioning_lookback_years": 10,
        "country_allocation_mode": "bus_country",
        "min_unit_mw_tyndp": 100.0,
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
        "tyndp_thermal_csv": args.tyndp_thermal_csv,
        "revision_duration_std_csv": args.revision_duration_std_csv,
        "revision_duration_long_csv": args.revision_duration_long_csv,
        "fuel_prices_csv": args.fuel_prices_csv,
        "thermal_params_csv": args.thermal_params_csv,
        "buses_csv": args.buses_csv,
        "plants_csv": args.plants_csv,
        "buses_with_clusters_csv": args.buses_with_clusters_csv,
        "load_shares_csv": args.load_shares_csv,
        "decommissioned_plants_csv": args.decommissioned_plants_csv,
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

    tyndp_dir = resolve_path(settings.get("tyndp_input_dir"), project_root) or DEFAULT_TYNDP_INPUT_DIR
    settings["tyndp_input_dir"] = tyndp_dir
    settings["tyndp_thermal_csv"] = resolve_path(settings.get("tyndp_thermal_csv"), tyndp_dir) or tyndp_dir / f"thermal_{target_year}_tyndp2024.csv"
    settings["revision_duration_std_csv"] = (
        resolve_path(settings.get("revision_duration_std_csv"), tyndp_dir)
        or tyndp_dir / "plants_median_revision_duration_weeks_country_2015-2025_planned.csv"
    )
    settings["revision_duration_long_csv"] = (
        resolve_path(settings.get("revision_duration_long_csv"), tyndp_dir)
        or tyndp_dir / "plants_max_revision_duration_weeks_country_2015-2025_planned.csv"
    )
    settings["fuel_prices_csv"] = resolve_path(settings.get("fuel_prices_csv"), tyndp_dir) or tyndp_dir / f"fuel_prices_{target_year}_tyndp2024.csv"
    settings["thermal_params_csv"] = resolve_path(settings.get("thermal_params_csv"), tyndp_dir) or tyndp_dir / "thermal_params_eraa2024.csv"
    settings["buses_csv"] = resolve_path(settings.get("buses_csv"), project_root) or network_dir / "buses.csv"
    settings["plants_csv"] = resolve_path(settings.get("plants_csv"), project_root) or network_dir / "plants.csv"
    settings["buses_with_clusters_csv"] = (
        resolve_path(settings.get("buses_with_clusters_csv"), project_root) or network_dir / "buses_with_clusters.csv"
    )
    settings["load_shares_csv"] = resolve_path(settings.get("load_shares_csv"), project_root)
    settings["decommissioned_plants_csv"] = (
        resolve_path(settings.get("decommissioned_plants_csv"), project_root)
        or default_decommissioned_plants_csv(project_root, network_dir)
    )
    settings["output_dir"] = resolve_path(settings.get("output_dir"), project_root) or default_output_dir(project_root, network_dir, "thermal")
    return settings


def first_mode(series: pd.Series) -> str:
    clean = series.dropna().astype(str).str.strip()
    clean = clean[clean != ""]
    if clean.empty:
        return ""
    mode = clean.mode(dropna=True)
    return str(mode.iloc[0] if not mode.empty else clean.iloc[0]).strip()


def load_tyndp_thermal_groups(settings: dict[str, Any], countries: list[str]) -> pd.DataFrame:
    df = read_csv_auto(settings["tyndp_thermal_csv"])
    req = {"country", "year", "scenario", "fuel_type", "plant_type", "installed_capacity", "n_units"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"{settings['tyndp_thermal_csv']} missing columns: {sorted(missing)}")
    country_map = source_country_mapping_from_load_shares(settings.get("load_shares_csv"))
    df["country"] = df["country"].map(norm_country).map(lambda country: country_map.get(country, country))
    df["year"] = pd.to_numeric(df["year"], errors="coerce").round().astype("Int64")
    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["installed_capacity"] = pd.to_numeric(df["installed_capacity"], errors="coerce")
    df["n_units"] = pd.to_numeric(df["n_units"], errors="coerce").round().astype("Int64")
    df = df[
        df["year"].eq(int(settings["target_year"]))
        & df["scenario"].eq(str(settings["scenario"]))
        & df["country"].isin(countries)
    ].dropna(subset=["installed_capacity", "n_units", "fuel_type", "plant_type", "country"]).copy()
    df = df[(df["installed_capacity"] > 0.0) & (df["n_units"] > 0)].copy()
    df["fuel_code"] = df["fuel_type"].map(map_fuel_code)
    df["tech_norm"] = df.apply(lambda r: map_thermal_tech(r["plant_type"], r["fuel_code"]), axis=1)
    grouped = (
        df.groupby(["country", "fuel_code", "tech_norm"], as_index=False)
        .agg(
            cap_total_mw=("installed_capacity", "sum"),
            n_units=("n_units", "sum"),
            raw_fuel_type=("fuel_type", first_mode),
            raw_plant_type=("plant_type", first_mode),
        )
    )
    grouped["n_units"] = grouped["n_units"].astype(int)
    min_unit = float(settings.get("min_unit_mw_tyndp", 0.0) or 0.0)
    if min_unit > 0.0:
        cap_unit = grouped["cap_total_mw"] / grouped["n_units"].clip(lower=1)
        mask = (grouped["cap_total_mw"] >= min_unit) & (cap_unit < min_unit)
        grouped.loc[mask, "n_units"] = np.maximum(1, np.floor(grouped.loc[mask, "cap_total_mw"] / min_unit)).astype(int)
    return grouped


def load_revision_durations(path: Path, countries: list[str], country_map: dict[str, str] | None = None) -> dict[tuple[str, str, str], int]:
    df = read_csv_auto(path)
    req = {"country_final", "fuel_type_code", "technology", "revision_duration_weeks"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"{path.name} missing columns: {sorted(missing)}")
    mapping = country_map or {}
    df["country_final"] = df["country_final"].map(norm_country).map(lambda country: mapping.get(country, country))
    df["fuel_type_code"] = df["fuel_type_code"].astype(str).str.strip().str.upper()
    df["technology_norm"] = df.apply(lambda r: map_thermal_tech(r["technology"], r["fuel_type_code"]), axis=1)
    df["revision_duration_weeks"] = pd.to_numeric(df["revision_duration_weeks"], errors="coerce")
    df = df[df["country_final"].isin(countries)].dropna(subset=["revision_duration_weeks"]).copy()
    grouped = df.groupby(["country_final", "fuel_type_code", "technology_norm"], as_index=False)["revision_duration_weeks"].median()
    return {
        (str(r.country_final), str(r.fuel_type_code), str(r.technology_norm)): safe_int(r.revision_duration_weeks, 1)
        for r in grouped.itertuples(index=False)
    }


def lookup_duration(country: str, fuel: str, tech: str, mapping: dict[tuple[str, str, str], int], defaults: dict[str, int], fallback: int) -> int:
    key = (norm_country(country), str(fuel).upper(), str(tech).upper())
    return int(mapping.get(key, defaults.get(str(tech).upper(), defaults.get("OTHERS", fallback))))


def prepare_network_thermal_rows(plant_rows: pd.DataFrame) -> pd.DataFrame:
    rows = plant_rows[plant_rows.apply(lambda r: is_thermal_row(r["fueltype"], r["set_name"]), axis=1)].copy()
    rows["fuel_code"] = rows["fueltype"].map(map_fuel_code)
    rows["tech_norm"] = rows.apply(lambda r: map_thermal_tech(r["technology"], r["fuel_code"]), axis=1)
    rows["chp_flag"] = rows.apply(lambda r: has_chp_flag(r["set_name"], r["technology"]), axis=1)
    rows["capacity_basis_mw"] = rows["capacity_mw"].astype(float).clip(lower=0.0)
    rows["n_plants_basis"] = rows["n_plants"].astype(float).clip(lower=0.0)
    return rows


def build_country_bus_priority(bus_country_membership: pd.DataFrame, plant_rows: pd.DataFrame, load_shares: pd.DataFrame) -> pd.DataFrame:
    basis = plant_rows.copy()
    basis["is_thermal"] = basis.apply(lambda r: is_thermal_row(r["fueltype"], r["set_name"]), axis=1)
    basis["unit_capacity_mw"] = np.divide(
        pd.to_numeric(basis["capacity_mw"], errors="coerce").fillna(0.0),
        pd.to_numeric(basis["n_plants"], errors="coerce").fillna(1.0).replace(0.0, 1.0),
    )
    thermal_caps = (
        basis.groupby(["bus_id", "country"], as_index=False)
        .agg(
            thermal_cap_mw=("capacity_mw", lambda s: float(s[basis.loc[s.index, "is_thermal"]].sum())),
            total_cap_mw=("capacity_mw", "sum"),
            max_unit_capacity_mw=("unit_capacity_mw", "max"),
        )
    )
    priority = bus_country_membership.merge(thermal_caps, how="left", on=["bus_id", "country"], validate="one_to_one")
    shares = load_shares[["country", "bus_id", "load_share"]].copy()
    shares["country"] = shares["country"].map(norm_country)
    shares["bus_id"] = shares["bus_id"].astype(str)
    priority = priority.merge(shares, how="left", on=["country", "bus_id"])
    priority[["thermal_cap_mw", "total_cap_mw", "max_unit_capacity_mw", "load_share"]] = priority[
        ["thermal_cap_mw", "total_cap_mw", "max_unit_capacity_mw", "load_share"]
    ].fillna(0.0)
    # The rank is the last fallback in the thermal siting chain. It favours load
    # centres with large historical units, but remains deterministic for audits.
    priority = priority.sort_values(
        ["country", "load_share", "max_unit_capacity_mw", "thermal_cap_mw", "total_cap_mw", "bus_id"],
        ascending=[True, False, False, False, False, True],
    )
    priority["bus_rank"] = priority.groupby("country").cumcount() + 1
    return priority.reset_index(drop=True)


def candidate_mask(candidates: pd.DataFrame, target: pd.Series, stage: str) -> pd.Series:
    mask = candidates["country"].eq(target["country"])
    if stage in {"fuel", "fuel_tech"}:
        mask &= candidates["fuel_code"].eq(target["fuel_code"])
    if stage == "fuel_tech":
        mask &= candidates["tech_norm"].eq(target["tech_norm"])
    return mask


def compute_weights(candidates: pd.DataFrame) -> pd.Series:
    cap = candidates["capacity_basis_mw"].clip(lower=0.0)
    units = candidates["n_plants_basis"].clip(lower=0.0)
    cap_sum = float(cap.sum())
    unit_sum = float(units.sum())
    if cap_sum > 0.0 and unit_sum > 0.0:
        # Capacity captures siting scale; unit counts keep multi-unit sites from
        # being hidden behind one very large block.
        return 0.5 * cap / cap_sum + 0.5 * units / unit_sum
    if cap_sum > 0.0:
        return cap / cap_sum
    if unit_sum > 0.0:
        return units / unit_sum
    return pd.Series(dtype=float)


def integer_bus_assignments(target: pd.Series, bus_weights: pd.DataFrame) -> pd.DataFrame:
    n_units = max(int(round(float(target["n_units"]))), 1)
    cap_unit = float(target["cap_total_mw"]) / n_units
    work = bus_weights.sort_values(["weight", "thermal_cap_mw", "total_cap_mw", "bus_id"], ascending=[False, False, False, True]).reset_index(drop=True)
    work["expected_units"] = work["weight"] * n_units
    work["assigned_units"] = np.floor(work["expected_units"]).astype(int)
    remainder = n_units - int(work["assigned_units"].sum())
    if remainder > 0:
        # Largest-remainder rounding keeps the national unit count exact without
        # moving capacity away from the weighted siting basis.
        work["fractional"] = work["expected_units"] - work["assigned_units"]
        order = work.sort_values(["fractional", "weight", "bus_id"], ascending=[False, False, True]).index.tolist()
        for idx in order[:remainder]:
            work.loc[idx, "assigned_units"] += 1
    work = work[work["assigned_units"] > 0].copy()
    work["assigned_cap_mw"] = work["assigned_units"] * cap_unit
    return work[["bus_id", "assigned_units", "assigned_cap_mw", "weight", "bus_rank"]].copy()


def fallback_bus_sequence(target: pd.Series, priority: pd.DataFrame, start_offset: int) -> tuple[pd.DataFrame, int]:
    buses = priority.loc[priority["country"].eq(target["country"])].sort_values("bus_rank").reset_index(drop=True)
    if buses.empty:
        return pd.DataFrame(), start_offset
    n_units = max(int(round(float(target["n_units"]))), 1)
    cap_unit = float(target["cap_total_mw"]) / n_units
    counts: dict[str, int] = defaultdict(int)
    for idx in range(n_units):
        bus = buses.iloc[(start_offset + idx) % len(buses)]
        counts[str(bus["bus_id"])] += 1
    rows = [
        {
            "bus_id": str(bus.bus_id),
            "assigned_units": int(counts.get(str(bus.bus_id), 0)),
            "assigned_cap_mw": float(counts.get(str(bus.bus_id), 0)) * cap_unit,
            "weight": np.nan,
            "bus_rank": int(bus.bus_rank),
        }
        for bus in buses.itertuples(index=False)
        if counts.get(str(bus.bus_id), 0) > 0
    ]
    return pd.DataFrame(rows), (start_offset + n_units) % len(buses)


def map_thermal_units(
    *,
    network_rows: pd.DataFrame,
    tyndp_groups: pd.DataFrame,
    priority: pd.DataFrame,
    decommissioned_sites: pd.DataFrame,
    dur_std: dict[tuple[str, str, str], int],
    dur_long: dict[tuple[str, str, str], int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    units: list[dict[str, Any]] = []
    alloc_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    offsets: dict[str, int] = defaultdict(int)
    decommissioning_usage: dict[tuple[str, str, str], float] = defaultdict(float)
    used_site_fuels: dict[tuple[str, str], set[str]] = defaultdict(set)
    rank_map = priority.set_index(["country", "bus_id"])["bus_rank"].to_dict()

    for target in tyndp_groups.sort_values(["country", "fuel_code", "tech_norm"]).itertuples(index=False):
        row = pd.Series(target._asdict())
        selected_stage = ""
        alloc = pd.DataFrame()
        candidates_selected = pd.DataFrame()
        for stage in ("fuel_tech", "fuel"):
            candidates = network_rows.loc[candidate_mask(network_rows, row, stage)].copy()
            weights = compute_weights(candidates) if not candidates.empty else pd.Series(dtype=float)
            if candidates.empty or weights.empty or float(weights.sum()) <= 0.0:
                continue
            candidates = candidates.assign(weight=weights)
            bus_weights = (
                candidates.groupby(["bus_id", "country"], as_index=False)
                .agg(weight=("weight", "sum"), thermal_cap_mw=("capacity_basis_mw", "sum"), total_cap_mw=("capacity_mw", "sum"))
                .sort_values(["weight", "bus_id"], ascending=[False, True])
                .reset_index(drop=True)
            )
            bus_weights["bus_rank"] = [int(rank_map.get((row["country"], bus_id), 0)) for bus_id in bus_weights["bus_id"]]
            alloc = integer_bus_assignments(row, bus_weights)
            alloc["fallback_stage"] = stage
            selected_stage = stage
            candidates_selected = candidates
            break
        if not selected_stage:
            n_units = max(int(round(float(row["n_units"]))), 1)
            cap_unit = float(row["cap_total_mw"]) / n_units
            target_group = fuel_group_from_row(row.get("raw_fuel_type", ""), row["fuel_code"])
            # If no active plant basis remains, recent decommissioning locations
            # are treated as plausible brownfield sites before the load-rank
            # fallback is used.
            decom_alloc, decom_diag = allocate_units_to_decommissioned_sites(
                country=str(row["country"]),
                target_fuel_group=target_group,
                unit_sizes_mw=[cap_unit] * n_units,
                decommissioned_sites=decommissioned_sites,
                usage_mw=decommissioning_usage,
                used_site_fuel_groups=used_site_fuels,
                context="thermal",
            )
            remaining_units = n_units - int(decom_alloc["assigned_units"].sum()) if not decom_alloc.empty else n_units
            parts = []
            if not decom_alloc.empty:
                decom_alloc["bus_rank"] = [int(rank_map.get((row["country"], bus_id), 0)) for bus_id in decom_alloc["bus_id"]]
                parts.append(decom_alloc)
            if remaining_units > 0:
                fb_target = row.copy()
                fb_target["n_units"] = remaining_units
                fb_target["cap_total_mw"] = remaining_units * cap_unit
                fallback_alloc, offsets[row["country"]] = fallback_bus_sequence(fb_target, priority, offsets[row["country"]])
                if not fallback_alloc.empty:
                    fallback_alloc["fallback_stage"] = "bus_sequence"
                    parts.append(fallback_alloc)
            alloc = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
            if not decom_alloc.empty and remaining_units > 0:
                selected_stage = "decommissioned_then_bus_sequence"
            elif not decom_alloc.empty:
                selected_stage = "decommissioned"
            else:
                selected_stage = "bus_sequence" if not alloc.empty else "unmatched"

        chp_share = 0.0
        if not candidates_selected.empty and float(candidates_selected["n_plants_basis"].sum()) > 0.0:
            chp_share = float(candidates_selected.loc[candidates_selected["chp_flag"], "n_plants_basis"].sum()) / float(candidates_selected["n_plants_basis"].sum())

        diag_rows.append(
            {
                "country": row["country"],
                "fuel_code": row["fuel_code"],
                "tech_norm": row["tech_norm"],
                "target_cap_mw": float(row["cap_total_mw"]),
                "target_n_units": int(row["n_units"]),
                "fallback_stage": selected_stage,
                "candidate_rows": int(len(candidates_selected)),
                "basis_cap_mw": float(candidates_selected["capacity_basis_mw"].sum()) if not candidates_selected.empty else 0.0,
                "basis_n_units": float(candidates_selected["n_plants_basis"].sum()) if not candidates_selected.empty else 0.0,
                "matched": bool(not alloc.empty),
            }
        )
        if alloc.empty:
            continue
        cap_unit = float(row["cap_total_mw"]) / max(1, int(row["n_units"]))
        n_chp_total = int(round(int(row["n_units"]) * chp_share))
        chp_remaining = n_chp_total
        for item in alloc.itertuples(index=False):
            n_bus = int(item.assigned_units)
            n_chp_bus = min(chp_remaining, n_bus)
            chp_remaining -= n_chp_bus
            item_stage = str(getattr(item, "fallback_stage", selected_stage))
            alloc_rows.append(
                {
                    "country": row["country"],
                    "fuel_code": row["fuel_code"],
                    "tech_norm": row["tech_norm"],
                    "bus_id": str(item.bus_id),
                    "assigned_units": n_bus,
                    "assigned_cap_mw": float(item.assigned_cap_mw),
                    "fallback_stage": item_stage,
                    "bus_rank": int(item.bus_rank),
                    "chp_units": int(n_chp_bus),
                }
            )
            for unit_idx in range(n_bus):
                plant_id = f"th|{row['country']}|{row['fuel_code']}|{row['tech_norm']}|{len(units) + 1:06d}"
                is_chp = unit_idx < n_chp_bus
                units.append(
                    {
                        "plant_id": plant_id,
                        "country": row["country"],
                        "bus_id": str(item.bus_id),
                        "fuel_code": row["fuel_code"],
                        "tech_norm": row["tech_norm"],
                        "raw_fuel_type": str(row.get("raw_fuel_type", "")),
                        "raw_plant_type": str(row.get("raw_plant_type", "")),
                        "installed_capacity_mw": cap_unit,
                        "chp": bool(is_chp),
                        "dur_rev_std_weeks": lookup_duration(row["country"], row["fuel_code"], row["tech_norm"], dur_std, STD_REV_DUR_BY_TECH, 2),
                        "dur_rev_long_weeks": lookup_duration(row["country"], row["fuel_code"], row["tech_norm"], dur_long, LONG_REV_DUR_BY_TECH, 4),
                        "fallback_stage": item_stage,
                        "inertia_h": inertia_h(row["fuel_code"], row["tech_norm"]),
                    }
                )
    return pd.DataFrame(units), pd.DataFrame(alloc_rows), pd.DataFrame(diag_rows)


def build_groups(units: pd.DataFrame) -> pd.DataFrame:
    if units.empty:
        return pd.DataFrame()
    grouped = (
        units.groupby(
            [
                "country",
                "bus_id",
                "fuel_code",
                "tech_norm",
                "chp",
                "dur_rev_std_weeks",
                "dur_rev_long_weeks",
                "installed_capacity_mw",
            ],
            as_index=False,
        )
        .agg(
            n_units=("plant_id", "count"),
            cap_total_mw=("installed_capacity_mw", "sum"),
            raw_fuel_type=("raw_fuel_type", "first"),
            raw_plant_type=("raw_plant_type", "first"),
            fallback_stage=("fallback_stage", "first"),
            inertia_h=("inertia_h", "first"),
        )
        .sort_values(["country", "bus_id", "fuel_code", "tech_norm", "chp", "installed_capacity_mw"])
        .reset_index(drop=True)
    )
    grouped["group_id"] = [
        f"grp|{row.country}|{row.bus_id}|{row.fuel_code}|{row.tech_norm}|{int(bool(row.chp))}|{idx + 1:06d}"
        for idx, row in enumerate(grouped.itertuples(index=False))
    ]
    return grouped[
        [
            "group_id",
            "country",
            "bus_id",
            "fuel_code",
            "tech_norm",
            "chp",
            "n_units",
            "installed_capacity_mw",
            "cap_total_mw",
            "dur_rev_std_weeks",
            "dur_rev_long_weeks",
            "raw_fuel_type",
            "raw_plant_type",
            "fallback_stage",
            "inertia_h",
        ]
    ].copy()


def attach_group_ids(units: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    if units.empty or groups.empty:
        return units
    out = units.merge(
        groups[
            [
                "group_id",
                "country",
                "bus_id",
                "fuel_code",
                "tech_norm",
                "chp",
                "dur_rev_std_weeks",
                "dur_rev_long_weeks",
                "installed_capacity_mw",
            ]
        ],
        how="left",
        on=["country", "bus_id", "fuel_code", "tech_norm", "chp", "dur_rev_std_weeks", "dur_rev_long_weeks", "installed_capacity_mw"],
        validate="many_to_one",
    )
    return out


def load_fuel_price_inputs(path: Path, ref_year: int, countries: list[str]) -> tuple[dict[tuple[str, str], float], dict[str, float], float]:
    if path is None or not Path(path).exists():
        return {}, {}, 0.0
    df = read_csv_auto(path)
    if not {"year", "country", "plant_type_code"}.issubset(df.columns):
        return {}, {}, 0.0
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["country"] = df["country"].map(norm_country)
    df["plant_type_code"] = df["plant_type_code"].astype(str).str.strip().str.upper()
    df = df[df["year"].eq(int(ref_year)) & df["country"].isin(set(countries))].copy()
    for col in ("price_eur_mwh", "price_eur_gj", "price_eur_ton"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        else:
            df[col] = np.nan
    df["fuel_price_eur_mwh"] = df["price_eur_mwh"]
    df.loc[df["fuel_price_eur_mwh"].isna() & df["price_eur_gj"].notna(), "fuel_price_eur_mwh"] = df["price_eur_gj"] * 3.6
    fuel = df[df["plant_type_code"].ne("B100")].dropna(subset=["fuel_price_eur_mwh"])
    fuel_lookup = {
        (str(r.country), str(r.plant_type_code)): float(r.fuel_price_eur_mwh)
        for r in fuel.groupby(["country", "plant_type_code"], as_index=False)["fuel_price_eur_mwh"].mean().itertuples(index=False)
    }
    co2 = df[df["plant_type_code"].eq("B100")].dropna(subset=["price_eur_ton"])
    co2_lookup = {
        str(r.country): float(r.price_eur_ton)
        for r in co2.groupby("country", as_index=False)["price_eur_ton"].mean().itertuples(index=False)
    }
    co2_default = float(np.nanmean(list(co2_lookup.values()))) if co2_lookup else 0.0
    return fuel_lookup, co2_lookup, co2_default


def load_thermal_params(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    if path is None or not Path(path).exists():
        return {}
    df = read_csv_auto(path)
    if not {"fuel_type", "plant_type"}.issubset(df.columns):
        return {}
    for col in ("efficiency", "co2_em_factor_kg_gj", "var_cost_om_eur_mwh"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        else:
            df[col] = np.nan
    out: dict[tuple[str, str], dict[str, float]] = {}
    grouped = df.groupby(["fuel_type", "plant_type"], as_index=False)[["efficiency", "co2_em_factor_kg_gj", "var_cost_om_eur_mwh"]].mean()
    for r in grouped.itertuples(index=False):
        out[(str(r.fuel_type).strip(), str(r.plant_type).strip())] = {
            "efficiency": float(r.efficiency) if pd.notna(r.efficiency) else np.nan,
            "co2_em_factor_kg_gj": float(r.co2_em_factor_kg_gj) if pd.notna(r.co2_em_factor_kg_gj) else np.nan,
            "var_cost_om_eur_mwh": float(r.var_cost_om_eur_mwh) if pd.notna(r.var_cost_om_eur_mwh) else 0.0,
        }
    return out


def fuel_label(fuel_code: str, raw_fuel: str) -> str | None:
    if "HYDROGEN" in str(raw_fuel).upper() or str(fuel_code).upper() == "B101":
        return "Hydrogen"
    return {
        "B01": "Biomass",
        "B02": "Lignite",
        "B04": "Gas",
        "B05": "Hard coal",
        "B06": "Oil",
        "B07": "Oil shale",
        "B14": "Nuclear",
        "B17": "Waste",
    }.get(str(fuel_code).upper())


def plant_param_type(tech_norm: str, raw_plant: str) -> str:
    tech = str(tech_norm).upper()
    raw = str(raw_plant or "").upper()
    if tech == "CCGT" or "CCGT" in raw:
        return "CCGT"
    if tech == "OCGT" or "OCGT" in raw:
        return "OCGT"
    if tech == "STEAM" or "STEAM" in raw:
        return "Steam turbine"
    if tech == "NUCLEAR":
        return "Nuclear"
    return "Other"


def attach_marginal_costs(groups: pd.DataFrame, units: pd.DataFrame, settings: dict[str, Any], countries: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if groups.empty:
        return groups, units, pd.DataFrame()
    fuel_lookup, co2_lookup, co2_default = load_fuel_price_inputs(settings["fuel_prices_csv"], int(settings["target_year"]), countries)
    params_lookup = load_thermal_params(settings["thermal_params_csv"])
    fuel_price_fallback = {
        fuel_code: float(np.nanmean([v for (_country, fuel), v in fuel_lookup.items() if fuel == fuel_code]))
        for fuel_code in sorted({fuel for _country, fuel in fuel_lookup.keys()})
    }
    rows: list[dict[str, Any]] = []
    mc_by_group: dict[str, float] = {}
    price_code_by_label = {"Biomass": "B01", "Lignite": "B02", "Gas": "B04", "Hard coal": "B05", "Oil": "B06", "Oil shale": "B07", "Nuclear": "B14", "Waste": "B17", "Hydrogen": "B101"}
    for r in groups.itertuples(index=False):
        label = fuel_label(str(r.fuel_code), str(r.raw_fuel_type))
        plant_type = plant_param_type(str(r.tech_norm), str(r.raw_plant_type))
        price_code = price_code_by_label.get(str(label), str(r.fuel_code).upper())
        fuel_price = fuel_lookup.get((str(r.country), price_code), fuel_price_fallback.get(price_code, np.nan))
        co2_price = co2_lookup.get(str(r.country), co2_default)
        params = params_lookup.get((str(label), plant_type)) or params_lookup.get((str(label), "Other"))
        if str(label).upper() in {"BIOMASS", "WASTE"}:
            mc = 0.0
            status = "fixed_zero"
            efficiency = np.nan
            co2_factor = 0.0
            var_om = 0.0
        elif params is None or pd.isna(fuel_price) or pd.isna(params["efficiency"]) or float(params["efficiency"]) <= 0.0:
            mc = 1.0e6
            status = "high_fallback"
            efficiency = np.nan if params is None else params["efficiency"]
            co2_factor = np.nan if params is None else params["co2_em_factor_kg_gj"]
            var_om = np.nan if params is None else params["var_cost_om_eur_mwh"]
        else:
            efficiency = float(params["efficiency"])
            co2_factor = float(params["co2_em_factor_kg_gj"])
            var_om = float(params["var_cost_om_eur_mwh"])
            mc = float(fuel_price) / efficiency + float(co2_price) * ((co2_factor * 3.6 / 1000.0) / efficiency) + var_om
            status = "formula"
        mc_by_group[str(r.group_id)] = float(mc)
        rows.append({"group_id": str(r.group_id), "country": str(r.country), "fuel_code": str(r.fuel_code), "tech_norm": str(r.tech_norm), "param_fuel": label or "", "param_plant_type": plant_type, "price_code": price_code, "marginal_cost_eur_mwh": float(mc), "cost_status": status})
    groups = groups.copy()
    groups["marginal_cost_eur_mwh"] = groups["group_id"].map(mc_by_group).astype(float)
    units = units.copy()
    units["marginal_cost_eur_mwh"] = units["group_id"].map(mc_by_group).astype(float) if "group_id" in units.columns else np.nan
    return groups, units, pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = resolve_settings(parse_args())
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    bus_membership = load_bus_country_membership(
        buses_csv=settings["buses_csv"],
        buses_with_clusters_csv=settings["buses_with_clusters_csv"],
        country_allocation_mode=str(settings.get("country_allocation_mode", "split_cluster_members")),
    )
    if settings.get("load_shares_csv") is None or not Path(settings["load_shares_csv"]).exists():
        raise ValueError("load_shares_csv is required so thermal capacity is assigned only to buses with positive load share.")
    load_shares = require_positive_load_share_buses(load_load_shares(settings["load_shares_csv"]))
    bus_membership = restrict_to_load_buses(bus_membership, load_shares)
    plant_rows = load_network_plant_rows(
        plants_csv=settings["plants_csv"],
        buses_csv=settings["buses_csv"],
        bus_country_membership=bus_membership,
    )
    plant_rows = restrict_to_load_buses(plant_rows, load_shares)
    countries = settings.get("countries") or sorted(bus_membership["country"].dropna().astype(str).unique().tolist())
    countries = [norm_country(c) for c in countries]
    network_rows = prepare_network_thermal_rows(plant_rows)
    tyndp_groups = load_tyndp_thermal_groups(settings, countries)
    priority = build_country_bus_priority(bus_membership, plant_rows, load_shares)
    decommissioned_sites = load_decommissioned_site_basis(
        plants_with_bus_csv=settings.get("decommissioned_plants_csv"),
        buses_with_clusters_csv=settings.get("buses_with_clusters_csv"),
        bus_country_membership=bus_membership,
        load_shares=load_shares,
        target_year=int(settings["target_year"]),
        lookback_years=int(settings.get("decommissioning_lookback_years", 10) or 10),
        allowed_fuel_groups={"nuclear", "lignite", "hard_coal", "gas", "oil", "oil_shale"},
    )
    country_map = source_country_mapping_from_load_shares(settings.get("load_shares_csv"))
    dur_std = load_revision_durations(settings["revision_duration_std_csv"], countries, country_map)
    dur_long = load_revision_durations(settings["revision_duration_long_csv"], countries, country_map)

    units, allocations, diagnostics = map_thermal_units(
        network_rows=network_rows,
        tyndp_groups=tyndp_groups,
        priority=priority,
        decommissioned_sites=decommissioned_sites,
        dur_std=dur_std,
        dur_long=dur_long,
    )
    groups = build_groups(units)
    units = attach_group_ids(units, groups)
    groups, units, cost_diag = attach_marginal_costs(groups, units, settings, countries)

    outputs = {
        "thermal_units": output_dir / "thermal_units.csv",
        "thermal_groups": output_dir / "thermal_groups.csv",
        "thermal_bus_allocations": output_dir / "thermal_bus_allocations.csv",
        "thermal_mapping_diagnostics": output_dir / "thermal_mapping_diagnostics.csv",
        "thermal_group_marginal_costs": output_dir / "thermal_group_marginal_costs.csv",
        "network_thermal_basis": output_dir / "network_thermal_basis.csv",
        "decommissioned_site_basis": output_dir / "decommissioned_site_basis.csv",
        "bus_country_membership": output_dir / "bus_country_membership.csv",
        "manifest": output_dir / "thermal_disaggregation_manifest.json",
    }
    write_csv(outputs["thermal_units"], units)
    write_csv(outputs["thermal_groups"], groups)
    write_csv(outputs["thermal_bus_allocations"], allocations)
    write_csv(outputs["thermal_mapping_diagnostics"], diagnostics)
    write_csv(outputs["thermal_group_marginal_costs"], cost_diag)
    write_csv(outputs["network_thermal_basis"], network_rows)
    write_csv(outputs["decommissioned_site_basis"], decommissioned_sites)
    write_csv(outputs["bus_country_membership"], bus_membership)
    write_json(
        outputs["manifest"],
        build_manifest(
            settings,
            outputs,
            {
                "target_capacity_mw": float(tyndp_groups["cap_total_mw"].sum()),
                "assigned_capacity_mw": float(units["installed_capacity_mw"].sum()) if not units.empty else 0.0,
                "n_units": int(len(units)),
                "n_groups": int(len(groups)),
            },
        ),
    )
    LOG.info("wrote thermal disaggregation outputs to %s", output_dir)


if __name__ == "__main__":
    main()
