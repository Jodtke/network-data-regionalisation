from __future__ import annotations

"""Regionalise TYNDP ``other non-RES`` capacity to reduced-grid buses.

The TYNDP category is not a plant list; it mainly represents residual gas and
oil-like peaking capacity. The module therefore uses already allocated thermal
capacity as the first siting basis, then decommissioned sites, and only then a
load-share fallback. This ordering keeps the heuristic close to observed
thermal infrastructure while still producing complete nodal inputs when the
plant database is sparse.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from powerplants_common import (
    DEFAULT_OTHERS_INPUT_DIR,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_SCENARIO,
    DEFAULT_TYNDP_INPUT_DIR,
    build_manifest,
    decommissioning_fuel_priority,
    allocate_units_to_decommissioned_sites,
    default_decommissioned_plants_csv,
    default_output_dir,
    discover_others_csv,
    find_column,
    fuel_group_from_fuel_code,
    fuel_group_from_raw_fuel,
    infer_target_year,
    is_thermal_row,
    load_bus_country_membership,
    load_decommissioned_site_basis,
    load_load_shares,
    load_network_plant_rows,
    load_yaml_config,
    numeric_column,
    norm_country,
    read_csv_auto,
    require_positive_load_share_buses,
    restrict_to_load_buses,
    resolve_path,
    source_country_mapping_from_load_shares,
    validate_tyndp_target_year,
    write_csv,
    write_json,
)

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Disaggregate Other non-RES capacity to reduced network buses.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--network-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-year", type=int, default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--countries", nargs="*", default=None)
    parser.add_argument("--other-nonres-csv", type=Path, default=None)
    parser.add_argument("--others-input-dir", type=Path, default=None)
    parser.add_argument("--thermal-allocations-csv", type=Path, default=None)
    parser.add_argument("--load-shares-csv", type=Path, default=None)
    parser.add_argument("--buses-csv", type=Path, default=None)
    parser.add_argument("--plants-csv", type=Path, default=None)
    parser.add_argument("--buses-with-clusters-csv", type=Path, default=None)
    parser.add_argument("--decommissioned-plants-csv", type=Path, default=None)
    return parser.parse_args()


def default_settings() -> dict[str, Any]:
    return {
        "scenario_name": "other_nonres_disaggregation",
        "project_root": str(DEFAULT_PROJECT_ROOT),
        "network_dir": None,
        "output_dir": None,
        "target_year": None,
        "scenario": DEFAULT_SCENARIO,
        "countries": None,
        "tyndp_input_dir": str(DEFAULT_TYNDP_INPUT_DIR),
        "others_input_dir": str(DEFAULT_OTHERS_INPUT_DIR),
        "other_nonres_csv": None,
        "thermal_allocations_csv": None,
        "load_shares_csv": None,
        "buses_csv": None,
        "plants_csv": None,
        "buses_with_clusters_csv": None,
        "decommissioned_plants_csv": None,
        "decommissioning_lookback_years": 10,
        "country_allocation_mode": "bus_country",
        "other_nonres_include_fixed_eur_mwh_in_marginal": True,
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
        "other_nonres_csv": args.other_nonres_csv,
        "others_input_dir": args.others_input_dir,
        "thermal_allocations_csv": args.thermal_allocations_csv,
        "load_shares_csv": args.load_shares_csv,
        "buses_csv": args.buses_csv,
        "plants_csv": args.plants_csv,
        "buses_with_clusters_csv": args.buses_with_clusters_csv,
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
    others_input_dir = resolve_path(settings.get("others_input_dir"), project_root) or tyndp_dir
    settings["others_input_dir"] = others_input_dir
    settings["other_nonres_csv"] = (
        resolve_path(settings.get("other_nonres_csv"), others_input_dir)
        or discover_others_csv(others_input_dir, kind="other_nonres", ref_year=target_year)
    )
    thermal_dir = default_output_dir(project_root, network_dir, "thermal")
    settings["thermal_allocations_csv"] = resolve_path(settings.get("thermal_allocations_csv"), project_root) or thermal_dir / "thermal_bus_allocations.csv"
    settings["load_shares_csv"] = resolve_path(settings.get("load_shares_csv"), project_root)
    settings["buses_csv"] = resolve_path(settings.get("buses_csv"), project_root) or network_dir / "buses.csv"
    settings["plants_csv"] = resolve_path(settings.get("plants_csv"), project_root) or network_dir / "plants.csv"
    settings["buses_with_clusters_csv"] = resolve_path(settings.get("buses_with_clusters_csv"), project_root) or network_dir / "buses_with_clusters.csv"
    settings["decommissioned_plants_csv"] = (
        resolve_path(settings.get("decommissioned_plants_csv"), project_root)
        or default_decommissioned_plants_csv(project_root, network_dir)
    )
    settings["output_dir"] = resolve_path(settings.get("output_dir"), project_root) or default_output_dir(project_root, network_dir, "other_nonres")
    return settings


def build_thermal_residual_basis(plant_rows: pd.DataFrame, allocations: pd.DataFrame) -> pd.DataFrame:
    basis = plant_rows[plant_rows.apply(lambda r: is_thermal_row(r["fueltype"], r["set_name"]), axis=1)].copy()
    basis["fuel_group"] = basis["fueltype"].map(fuel_group_from_raw_fuel)
    thermal_basis = (
        basis.groupby(["country", "bus_id", "fuel_group"], as_index=False)
        .agg(thermal_basis_capacity_mw=("capacity_mw", "sum"), thermal_basis_n_plants=("n_plants", "sum"))
    )
    if not allocations.empty:
        assigned = allocations.copy()
        if "fuel_group" not in assigned.columns:
            assigned["fuel_group"] = assigned["fuel_code"].map(fuel_group_from_fuel_code) if "fuel_code" in assigned.columns else "other"
        assigned = assigned.groupby(["country", "bus_id", "fuel_group"], as_index=False).agg(
            thermal_assigned_capacity_mw=("assigned_cap_mw", "sum"),
            thermal_assigned_units=("assigned_units", "sum"),
        )
    else:
        assigned = pd.DataFrame(columns=["country", "bus_id", "fuel_group", "thermal_assigned_capacity_mw", "thermal_assigned_units"])
    out = thermal_basis.merge(assigned, how="outer", on=["country", "bus_id", "fuel_group"]).fillna(0.0)
    # Other non-RES is interpreted as residual peaking capacity. The active
    # thermal allocation is deducted first so that these units do not duplicate
    # already represented TYNDP thermal blocks at the same sites.
    out["thermal_residual_capacity_mw"] = (out["thermal_basis_capacity_mw"] - out["thermal_assigned_capacity_mw"]).clip(lower=0.0)
    out["thermal_residual_n_plants"] = (out["thermal_basis_n_plants"] - out["thermal_assigned_units"]).clip(lower=0.0)
    return out


def classify_other_nonres_fuel_group(value: Any) -> str:
    group = fuel_group_from_raw_fuel(value)
    return "oil" if group in {"oil", "oil_shale"} else "gas"


def load_other_nonres_targets(path: Path, *, countries: list[str], country_map: dict[str, str], ref_year: int, scenario: str) -> pd.DataFrame:
    df = read_csv_auto(path)
    country_col = find_column(df, ("country", "country_code", "country_iso", "area", "region", "zone"))
    year_col = find_column(df, ("year", "target_year", "ref_year", "reference_year"))
    scenario_col = find_column(df, ("scenario", "storyline", "scenario_name"))
    capacity_col = find_column(df, ("installed_capacity_MW", "installed_capacity_mw", "capacity_mw", "capacity_MW", "capacity", "mw"))
    units_col = find_column(df, ("units", "n_units", "unit_count"))
    type_col = find_column(df, ("pemmdb_types", "technology", "type", "fuel_type", "fuel"))
    if country_col is None or capacity_col is None:
        raise KeyError(f"{path.name} missing country or capacity column.")
    work = df.copy()
    if year_col is not None:
        work["_year"] = pd.to_numeric(work[year_col], errors="coerce").round().astype("Int64")
        work = work[work["_year"].eq(int(ref_year))].copy()
    if scenario_col is not None:
        requested = str(scenario).strip()
        work["_scenario"] = work[scenario_col].astype(str).str.strip()
        if requested and work["_scenario"].str.casefold().eq(requested.casefold()).any():
            work = work[work["_scenario"].str.casefold().eq(requested.casefold())].copy()
    work["country_source"] = work[country_col].map(norm_country)
    work["country"] = work["country_source"].map(lambda c: norm_country(country_map.get(c, c)))
    work = work[work["country"].isin(set(countries))].copy()
    work["target_capacity_mw"] = numeric_column(work[capacity_col]).fillna(0.0).clip(lower=0.0)
    work["units"] = numeric_column(work[units_col]).fillna(0.0).clip(lower=0.0) if units_col is not None else 0.0
    work["fuel_group"] = work[type_col].map(classify_other_nonres_fuel_group) if type_col is not None else "gas"
    work = work[work["target_capacity_mw"] > 0.0].copy()
    passthrough = [
        col
        for col in work.columns
        if col
        not in {
            country_col,
            year_col,
            scenario_col,
            capacity_col,
            units_col,
            "country",
            "country_source",
            "target_capacity_mw",
            "units",
            "fuel_group",
            "_year",
            "_scenario",
        }
    ]
    rows: list[dict[str, Any]] = []
    for (country, fuel_group), group in work.groupby(["country", "fuel_group"], sort=True):
        weights = group["target_capacity_mw"].clip(lower=0.0)
        weight_sum = float(weights.sum())
        row: dict[str, Any] = {
            "country": str(country),
            "fuel_group": str(fuel_group),
            "target_capacity_mw": float(group["target_capacity_mw"].sum()),
            "units": float(group["units"].sum()),
            "country_source": ",".join(sorted(set(str(v) for v in group["country_source"] if str(v).strip()))),
        }
        for col in passthrough:
            numeric = numeric_column(group[col])
            if numeric.notna().any():
                row[col] = float((numeric.fillna(0.0) * weights).sum() / weight_sum) if weight_sum > 0.0 else float(numeric.mean())
            else:
                vals = sorted(set(str(v).strip() for v in group[col].dropna() if str(v).strip()))
                if vals:
                    row[col] = ",".join(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def allocate_units_to_capacity_basis(
    *,
    country: str,
    target_fuel_group: str,
    unit_sizes_mw: list[float],
    basis: pd.DataFrame,
    usage_mw: dict[tuple[str, str, str], float],
    context: str,
    allocation_prefix: str,
) -> tuple[pd.DataFrame, list[float]]:
    rows: list[dict[str, Any]] = []
    unassigned: list[float] = []
    country = norm_country(country)
    sites = basis[basis["country"].eq(country)].copy()
    if sites.empty:
        return pd.DataFrame(), list(unit_sizes_mw)
    sites = sites.sort_values(["capacity_mw", "bus_id"], ascending=[False, True]).reset_index(drop=True)
    for unit_size in sorted([float(v) for v in unit_sizes_mw if float(v) > 0.0], reverse=True):
        assigned = False
        # Match same fuel-group brownfield sites first, then broaden the search
        # according to the context-specific fuel priority.
        for stage_idx, fuel_groups in enumerate(decommissioning_fuel_priority(target_fuel_group, context=context), start=1):
            candidates = sites[sites["fuel_group"].isin(set(fuel_groups))]
            for site in candidates.itertuples(index=False):
                key = (country, str(site.bus_id), str(site.fuel_group))
                used = float(usage_mw.get(key, 0.0))
                cap = float(site.capacity_mw)
                if used >= cap:
                    continue
                usage_mw[key] = used + unit_size
                rows.append(
                    {
                        "country": country,
                        "bus_id": str(site.bus_id),
                        "fuel_group": str(target_fuel_group),
                        "source_fuel_group": str(site.fuel_group),
                        "capacity_mw": unit_size,
                        "n_units": 1,
                        "allocation_mode": f"{allocation_prefix}_{str(site.fuel_group)}",
                        "priority_stage": int(stage_idx),
                    }
                )
                assigned = True
                break
            if assigned:
                break
        if not assigned:
            unassigned.append(unit_size)
    return pd.DataFrame(rows), unassigned


def allocate_units_round_robin_load(
    *,
    country: str,
    target_fuel_group: str,
    unit_sizes_mw: list[float],
    load_shares: pd.DataFrame,
    ascending_load_share: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    buses = load_shares[load_shares["country"].eq(country)].copy()
    if buses.empty:
        return pd.DataFrame()
    buses = buses.sort_values(["load_share", "bus_id"], ascending=[ascending_load_share, True]).reset_index(drop=True)
    # The residual fallback spreads units over the demand geography instead of
    # piling all unmapped capacity onto one bus.
    for idx, unit_size in enumerate(sorted([float(v) for v in unit_sizes_mw if float(v) > 0.0], reverse=True)):
        bus = buses.iloc[idx % len(buses)]
        rows.append(
            {
                "country": country,
                "bus_id": str(bus.bus_id),
                "fuel_group": str(target_fuel_group),
                "source_fuel_group": "",
                "capacity_mw": unit_size,
                "n_units": 1,
                "allocation_mode": "round_robin_low_load_share",
                "priority_stage": 999,
            }
        )
    return pd.DataFrame(rows)


def unit_sizes_from_target(capacity_mw: float, units: float) -> list[float]:
    cap = float(max(capacity_mw, 0.0))
    unit_count_raw = float(units or 0.0)
    if cap <= 0.0:
        return []
    if unit_count_raw <= 0.0:
        return [cap]
    n_units = max(1, int(round(unit_count_raw)))
    unit_size = cap / n_units
    return [unit_size] * n_units


def copy_numeric_alias(frame: pd.DataFrame, canonical: str, candidates: tuple[str, ...]) -> pd.Series:
    column = find_column(frame, candidates)
    if column is None:
        return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="Float64")
    values = numeric_column(frame[column]).astype("Float64")
    if canonical == "efficiency":
        values = values.mask(values > 1.5, values / 100.0)
    if canonical == "fixed_cost_eur_mw_a" and "kw" in str(column).lower():
        values = values * 1000.0
    return values


def build_other_nonres_attributes(targets_df: pd.DataFrame, include_fixed_eur_mwh: bool) -> pd.DataFrame:
    if targets_df.empty:
        return pd.DataFrame(columns=["country"])
    out = targets_df.copy()
    aliases = {
        "efficiency": ("efficiency", "efficiency_percent", "net_efficiency", "eta"),
        "co2_em_factor_kg_gj": ("co2_em_factor_kg_gj", "co2_factor_kg_gj", "co2_kg_gj", "co2_emissions_kg_gj"),
        "co2_intensity_t_mwh": (
            "co2_intensity_t_mwh",
            "co2_factor_t_per_mwh",
            "co2_t_per_mwh",
            "co2_t_mwh",
            "emission_factor_t_mwh",
            "emission_factor_tco2_mwh",
        ),
        "co2_price_eur_t": ("co2_price_eur_t", "co2_price_eur_ton", "co2_eur_t", "co2_eur_ton"),
        "fuel_price_eur_mwh": ("fuel_price_eur_mwh", "fuel_cost_eur_mwh", "fuel_price_eur_mwh_th"),
        "fuel_price_eur_gj": ("fuel_price_eur_gj", "price_eur_gj", "fuel_cost_eur_gj"),
        "variable_cost_eur_mwh": (
            "variable_cost_eur_mwh",
            "variable_costs_eur_mwh",
            "var_cost_eur_mwh",
            "var_cost_om_eur_mwh",
            "vom_eur_mwh",
            "variable_om_eur_mwh",
        ),
        "fixed_cost_eur_mw_a": (
            "fixed_cost_eur_mw_a",
            "fixed_cost_eur_mw_year",
            "fixed_cost_eur_mw_per_year",
            "fixed_costs_eur_mw_a",
            "fixed_cost_eur_kw_a",
            "fixed_costs_eur_kw_a",
            "fixed_cost_eur_kw_year",
            "fixed_cost_eur_kw_per_year",
            "fixed_cost_eur_kw",
            "fixed_costs_eur_kw",
            "fom_eur_mw_a",
            "fom_eur_kw_a",
        ),
        "fixed_cost_eur_mwh": ("fixed_cost_eur_mwh", "fixed_costs_eur_mwh", "fixed_om_eur_mwh"),
        "total_cost_eur_mwh": (
            "total_cost_eur_mwh",
            "cost_eur_mwh",
            "costs_eur_mwh",
            "price_eur_mwh",
            "generation_cost_eur_mwh",
            "variable_and_fixed_cost_eur_mwh",
            "variable_plus_fixed_cost_eur_mwh",
        ),
        "marginal_cost_eur_mwh": ("marginal_cost_eur_mwh", "marginal_costs_eur_mwh", "mc_eur_mwh"),
    }
    for canonical, candidates in aliases.items():
        if canonical not in out.columns:
            out[canonical] = copy_numeric_alias(out, canonical, candidates)

    direct = out["marginal_cost_eur_mwh"].copy()
    direct = direct.mask(direct.isna(), out["total_cost_eur_mwh"])

    calculated = pd.Series(0.0, index=out.index, dtype="Float64")
    has_component = pd.Series(False, index=out.index)

    variable = out["variable_cost_eur_mwh"]
    calculated = calculated + variable.fillna(0.0)
    has_component = has_component | variable.notna()

    efficiency = out["efficiency"]
    fuel_mwh = out["fuel_price_eur_mwh"]
    fuel_gj = out["fuel_price_eur_gj"]
    fuel_component = pd.Series(pd.NA, index=out.index, dtype="Float64")
    valid_fuel_mwh = fuel_mwh.notna() & efficiency.notna() & (efficiency > 0.0)
    fuel_component.loc[valid_fuel_mwh] = fuel_mwh.loc[valid_fuel_mwh] / efficiency.loc[valid_fuel_mwh]
    valid_fuel_gj = fuel_component.isna() & fuel_gj.notna() & efficiency.notna() & (efficiency > 0.0)
    fuel_component.loc[valid_fuel_gj] = fuel_gj.loc[valid_fuel_gj] * 3.6 / efficiency.loc[valid_fuel_gj]
    calculated = calculated + fuel_component.fillna(0.0)
    has_component = has_component | fuel_component.notna()

    co2_component = pd.Series(pd.NA, index=out.index, dtype="Float64")
    valid_co2_mwh = out["co2_price_eur_t"].notna() & out["co2_intensity_t_mwh"].notna()
    co2_component.loc[valid_co2_mwh] = out.loc[valid_co2_mwh, "co2_price_eur_t"] * out.loc[valid_co2_mwh, "co2_intensity_t_mwh"]
    valid_co2_gj = (
        co2_component.isna()
        & out["co2_price_eur_t"].notna()
        & out["co2_em_factor_kg_gj"].notna()
        & efficiency.notna()
        & (efficiency > 0.0)
    )
    co2_component.loc[valid_co2_gj] = out.loc[valid_co2_gj, "co2_price_eur_t"] * (
        (out.loc[valid_co2_gj, "co2_em_factor_kg_gj"] * 3.6 / 1000.0) / efficiency.loc[valid_co2_gj]
    )
    calculated = calculated + co2_component.fillna(0.0)
    has_component = has_component | co2_component.notna()

    if include_fixed_eur_mwh:
        fixed_eur_mwh = out["fixed_cost_eur_mwh"]
        calculated = calculated + fixed_eur_mwh.fillna(0.0)
        has_component = has_component | fixed_eur_mwh.notna()

    out["marginal_cost_eur_mwh"] = direct.mask(direct.isna(), calculated.where(has_component))
    out["marginal_cost_status"] = "missing"
    out.loc[direct.notna(), "marginal_cost_status"] = "direct_eur_mwh"
    out.loc[direct.isna() & has_component, "marginal_cost_status"] = "calculated_components"

    keep = [
        column
        for column in [
            "country",
            "fuel_group",
            "country_source",
            "target_capacity_mw",
            "units",
            "efficiency",
            "co2_em_factor_kg_gj",
            "co2_intensity_t_mwh",
            "co2_price_eur_t",
            "fuel_price_eur_mwh",
            "fuel_price_eur_gj",
            "variable_cost_eur_mwh",
            "fixed_cost_eur_mw_a",
            "fixed_cost_eur_mwh",
            "total_cost_eur_mwh",
            "marginal_cost_eur_mwh",
            "marginal_cost_status",
        ]
        if column in out.columns
    ]
    return out[keep].copy()


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
        raise ValueError("load_shares_csv is required so Other non-RES is assigned only to buses with positive load share.")
    load_shares = require_positive_load_share_buses(load_load_shares(settings["load_shares_csv"]))
    bus_membership = restrict_to_load_buses(bus_membership, load_shares)
    countries = settings.get("countries") or sorted(bus_membership["country"].dropna().astype(str).unique().tolist())
    countries = [norm_country(c) for c in countries]
    plant_rows = load_network_plant_rows(
        plants_csv=settings["plants_csv"],
        buses_csv=settings["buses_csv"],
        bus_country_membership=bus_membership,
    )
    plant_rows = restrict_to_load_buses(plant_rows, load_shares)
    allocations = read_csv_auto(settings["thermal_allocations_csv"]) if Path(settings["thermal_allocations_csv"]).exists() else pd.DataFrame()
    if not allocations.empty:
        allocations = restrict_to_load_buses(allocations, load_shares)
    residual = build_thermal_residual_basis(plant_rows, allocations)
    residual_basis = residual.loc[
        residual["thermal_residual_capacity_mw"] > 0.0,
        ["country", "bus_id", "fuel_group", "thermal_residual_capacity_mw"],
    ].rename(columns={"thermal_residual_capacity_mw": "capacity_mw"})

    country_map = source_country_mapping_from_load_shares(settings.get("load_shares_csv"))
    targets_df = load_other_nonres_targets(
        settings["other_nonres_csv"],
        countries=countries,
        country_map=country_map,
        ref_year=int(settings["target_year"]),
        scenario=str(settings["scenario"]),
    )
    attributes = build_other_nonres_attributes(
        targets_df,
        include_fixed_eur_mwh=bool(settings.get("other_nonres_include_fixed_eur_mwh_in_marginal", True)),
    )
    decommissioned_sites = load_decommissioned_site_basis(
        plants_with_bus_csv=settings.get("decommissioned_plants_csv"),
        buses_with_clusters_csv=settings.get("buses_with_clusters_csv"),
        bus_country_membership=bus_membership,
        load_shares=load_shares,
        target_year=int(settings["target_year"]),
        lookback_years=int(settings.get("decommissioning_lookback_years", 10) or 10),
        allowed_fuel_groups={"lignite", "hard_coal", "gas", "oil", "oil_shale"},
    )

    residual_usage: dict[tuple[str, str, str], float] = {}
    decom_usage: dict[tuple[str, str, str], float] = {}
    used_site_fuels: dict[tuple[str, str], set[str]] = {}
    allocation_frames: list[pd.DataFrame] = []
    diag_rows: list[dict[str, Any]] = []
    for target_row in targets_df.sort_values(["country", "fuel_group"]).itertuples(index=False):
        country = str(target_row.country)
        fuel_group = str(target_row.fuel_group)
        target_cap = float(target_row.target_capacity_mw)
        unit_sizes = unit_sizes_from_target(target_cap, float(getattr(target_row, "units", 0.0) or 0.0))
        residual_alloc, remaining = allocate_units_to_capacity_basis(
            country=country,
            target_fuel_group=fuel_group,
            unit_sizes_mw=unit_sizes,
            basis=residual_basis,
            usage_mw=residual_usage,
            context="other_nonres",
            allocation_prefix="thermal_residual",
        )
        if not residual_alloc.empty:
            allocation_frames.append(residual_alloc)
        decom_alloc, decom_unassigned = allocate_units_to_decommissioned_sites(
            country=country,
            target_fuel_group=fuel_group,
            unit_sizes_mw=remaining,
            decommissioned_sites=decommissioned_sites,
            usage_mw=decom_usage,
            used_site_fuel_groups=used_site_fuels,
            context="other_nonres",
        )
        remaining_after_decom = [float(item["unassigned_unit_mw"]) for item in decom_unassigned]
        if not decom_alloc.empty:
            converted = decom_alloc.rename(columns={"assigned_cap_mw": "capacity_mw"}).copy()
            converted["fuel_group"] = fuel_group
            converted["source_fuel_group"] = converted["decommissioned_fuel_group"]
            converted["n_units"] = converted["assigned_units"]
            converted["allocation_mode"] = converted["fallback_stage"]
            converted["priority_stage"] = converted["decommissioning_priority_stage"]
            allocation_frames.append(converted[["country", "bus_id", "fuel_group", "source_fuel_group", "capacity_mw", "n_units", "allocation_mode", "priority_stage"]])
        rr_alloc = allocate_units_round_robin_load(
            country=country,
            target_fuel_group=fuel_group,
            unit_sizes_mw=remaining_after_decom,
            load_shares=load_shares,
            ascending_load_share=True,
        )
        if not rr_alloc.empty:
            allocation_frames.append(rr_alloc)
        allocated = (
            (float(residual_alloc["capacity_mw"].sum()) if not residual_alloc.empty else 0.0)
            + (float(decom_alloc["assigned_cap_mw"].sum()) if not decom_alloc.empty else 0.0)
            + (float(rr_alloc["capacity_mw"].sum()) if not rr_alloc.empty else 0.0)
        )
        diag_rows.append(
            {
                "country": country,
                "fuel_group": fuel_group,
                "target_capacity_mw": target_cap,
                "target_units": float(getattr(target_row, "units", 0.0) or 0.0),
                "avg_unit_capacity_mw": target_cap / max(1.0, float(getattr(target_row, "units", 0.0) or 0.0)),
                "allocated_capacity_mw": allocated,
                "unallocated_capacity_mw": max(0.0, target_cap - allocated),
                "n_allocated_buses": 0,
            }
        )

    tech_out = pd.concat(allocation_frames, ignore_index=True, sort=False) if allocation_frames else pd.DataFrame()
    if not tech_out.empty:
        tech_out = tech_out.groupby(["country", "bus_id", "fuel_group", "source_fuel_group", "allocation_mode"], as_index=False).agg(
            capacity_mw=("capacity_mw", "sum"),
            source_n_units=("n_units", "sum"),
            priority_stage=("priority_stage", "min"),
        )
        tech_out["target_year"] = int(settings["target_year"])
        tech_out["scenario"] = str(settings["scenario"])
        if not attributes.empty:
            merge_cols = [column for column in attributes.columns if column not in {"target_capacity_mw", "units"}]
            tech_out = tech_out.merge(attributes[merge_cols], how="left", on=["country", "fuel_group"], validate="many_to_one")

    def weighted_cost(group: pd.DataFrame) -> float:
        if "marginal_cost_eur_mwh" not in group.columns:
            return float("nan")
        cost = pd.to_numeric(group["marginal_cost_eur_mwh"], errors="coerce")
        weight = pd.to_numeric(group["capacity_mw"], errors="coerce").fillna(0.0)
        mask = cost.notna() & weight.gt(0.0)
        return float((cost[mask] * weight[mask]).sum() / weight[mask].sum()) if mask.any() and float(weight[mask].sum()) > 0.0 else float("nan")

    if not tech_out.empty:
        bus_rows: list[dict[str, Any]] = []
        for (target_year, scenario, country, bus_id), group in tech_out.groupby(["target_year", "scenario", "country", "bus_id"], sort=True):
            bus_rows.append(
                {
                    "target_year": target_year,
                    "scenario": scenario,
                    "country": country,
                    "bus_id": bus_id,
                    "capacity_mw": float(group["capacity_mw"].sum()),
                    "source_n_units": float(group["source_n_units"].sum()),
                    "source_fuel_groups": ",".join(sorted(set(str(v) for v in group["fuel_group"] if str(v).strip()))),
                    "source_allocation_modes": ",".join(sorted(set(str(v) for v in group["allocation_mode"] if str(v).strip()))),
                    "marginal_cost_eur_mwh": weighted_cost(group),
                }
            )
        out = pd.DataFrame(bus_rows)
        out["technology"] = "other_nonres"
        out["n_units"] = 1
        out["unit_capacity_mw"] = out["capacity_mw"]
        out["unit_id"] = [f"other_nonres|{row.country}|{row.bus_id}" for row in out[["country", "bus_id"]].itertuples(index=False)]
    else:
        out = pd.DataFrame()
    diag = pd.DataFrame(diag_rows)
    if not diag.empty and not tech_out.empty:
        bus_counts = tech_out.groupby(["country", "fuel_group"])["bus_id"].nunique()
        diag["n_allocated_buses"] = [
            int(bus_counts.get((row.country, row.fuel_group), 0))
            for row in diag.itertuples(index=False)
        ]

    outputs = {
        "other_nonres_capacity_country_bus": output_dir / "other_nonres_capacity_country_bus.csv",
        "other_nonres_capacity_country_bus_fuel": output_dir / "other_nonres_capacity_country_bus_fuel.csv",
        "other_nonres_bus_scaling": output_dir / "other_nonres_bus_scaling.csv",
        "other_nonres_country_targets": output_dir / "other_nonres_country_targets.csv",
        "other_nonres_cost_parameters": output_dir / "other_nonres_cost_parameters.csv",
        "thermal_residual_basis": output_dir / "thermal_residual_basis_bus.csv",
        "decommissioned_site_basis": output_dir / "decommissioned_site_basis.csv",
        "manifest": output_dir / "other_nonres_disaggregation_manifest.json",
    }
    write_csv(outputs["other_nonres_capacity_country_bus"], out)
    write_csv(outputs["other_nonres_capacity_country_bus_fuel"], tech_out)
    write_csv(outputs["other_nonres_bus_scaling"], diag)
    write_csv(outputs["other_nonres_country_targets"], targets_df)
    write_csv(outputs["other_nonres_cost_parameters"], attributes)
    write_csv(outputs["thermal_residual_basis"], residual)
    write_csv(outputs["decommissioned_site_basis"], decommissioned_sites)
    write_json(outputs["manifest"], build_manifest(settings, outputs, {"assigned_capacity_mw": float(out["capacity_mw"].sum()) if not out.empty else 0.0}))
    LOG.info("wrote Other non-RES outputs to %s", output_dir)


if __name__ == "__main__":
    main()
