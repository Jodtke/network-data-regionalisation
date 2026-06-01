from __future__ import annotations

"""Regionalise non-standard renewable technologies from TYNDP to buses.

Technologies outside PV, onshore wind, and offshore wind are allocated with
technology-specific proxies. Marine capacity is restricted to buses with
offshore-wind eligibility, while bio-based, waste, geothermal, and undefined
renewables use current sites, decommissioned sites, renewable headroom, and
load-share priority rules. The aim is not to optimise siting, but to document a
deterministic and auditable allocation rule for residual RES categories.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from powerplants_common import (
    DEFAULT_OTHERS_INPUT_DIR,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_SCENARIO,
    DEFAULT_TYNDP_INPUT_DIR,
    allocate_capacity_by_capped_basis,
    allocate_capacity_to_decommissioned_sites,
    build_manifest,
    default_decommissioned_plants_csv,
    default_output_dir,
    infer_target_year,
    aggregate_targets_with_country_map,
    discover_others_csv,
    fuel_group_from_raw_fuel,
    is_other_res_pypsa_fuel,
    load_bus_country_membership,
    load_country_capacity_targets,
    load_decommissioned_site_basis,
    load_load_shares,
    load_network_plant_rows,
    load_res_bus_capacity,
    load_res_potential,
    load_yaml_config,
    normalize_column_key,
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
ALLOWED_TECHS = ("pv", "onwind")
POTENTIAL_TECHS = ("pv", "onwind", "offwind")
DECOMMISSIONING_OTHER_RES_KINDS = {"bio", "waste", "geothermal"}
OTHER_RES_TECH_GROUP = "other_res_technology"


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


def normalize_res_technology(value: Any) -> str:
    key = normalize_column_key(value)
    if "offshore" in key and "wind" in key:
        return "offwind"
    if "offwind" in key or key in {"windoffshore", "windoff"}:
        return "offwind"
    if "onshore" in key and "wind" in key:
        return "onwind"
    if "onwind" in key or key in {"windonshore", "windon"}:
        return "onwind"
    if "solar" in key or "photovoltaic" in key or key == "pv":
        return "pv"
    return str(value or "").strip().lower()


def normalize_other_res_technology(value: Any) -> str:
    key = normalize_column_key(value)
    if any(token in key for token in ("marine", "tidal", "wave", "ocean")):
        return "marine"
    if not key or key == "nan":
        return "unspecified"
    return key


def is_undefined_other_res(value: Any) -> bool:
    key = normalize_other_res_technology(value)
    return (
        not key
        or key == "nan"
        or "notdefined" in key
        or "splittingnotknown" in key
        or "undefined" in key
        or "unspecified" in key
    )


def is_marine_other_res(value: Any) -> bool:
    return normalize_other_res_technology(value) == "marine"


def other_res_kind(value: Any) -> str:
    key = normalize_other_res_technology(value)
    if key == "marine":
        return "marine"
    if is_undefined_other_res(value):
        return "undefined"
    if "waste" in key or "muell" in key or "mull" in key:
        return "waste"
    if "geotherm" in key:
        return "geothermal"
    if "bio" in key or "biomass" in key:
        return "bio"
    return "undefined"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Balanced headroom-based Other RES disaggregation.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--network-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-year", type=int, default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--countries", nargs="*", default=None)
    parser.add_argument("--other-res-csv", type=Path, default=None)
    parser.add_argument("--others-input-dir", type=Path, default=None)
    parser.add_argument("--res-bus-capacity-csv", type=Path, default=None)
    parser.add_argument("--res-potential-csv", type=Path, default=None)
    parser.add_argument("--res-capacity-cells-nc", type=Path, default=None)
    parser.add_argument("--res-bus-lookup-csv", type=Path, default=None)
    parser.add_argument("--load-shares-csv", type=Path, default=None)
    parser.add_argument("--buses-csv", type=Path, default=None)
    parser.add_argument("--plants-csv", type=Path, default=None)
    parser.add_argument("--buses-with-clusters-csv", type=Path, default=None)
    parser.add_argument("--decommissioned-plants-csv", type=Path, default=None)
    return parser.parse_args()


def default_settings() -> dict[str, Any]:
    return {
        "scenario_name": "other_res_disaggregation",
        "project_root": str(DEFAULT_PROJECT_ROOT),
        "network_dir": None,
        "output_dir": None,
        "target_year": None,
        "scenario": DEFAULT_SCENARIO,
        "countries": None,
        "tyndp_input_dir": str(DEFAULT_TYNDP_INPUT_DIR),
        "others_input_dir": str(DEFAULT_OTHERS_INPUT_DIR),
        "other_res_csv": None,
        "res_bus_capacity_csv": None,
        "res_potential_csv": None,
        "res_capacity_cells_nc": None,
        "res_bus_lookup_csv": None,
        "load_shares_csv": None,
        "buses_csv": None,
        "plants_csv": None,
        "buses_with_clusters_csv": None,
        "decommissioned_plants_csv": None,
        "decommissioning_lookback_years": 10,
        "country_allocation_mode": "bus_country",
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
        "other_res_csv": args.other_res_csv,
        "others_input_dir": args.others_input_dir,
        "res_bus_capacity_csv": args.res_bus_capacity_csv,
        "res_potential_csv": args.res_potential_csv,
        "res_capacity_cells_nc": args.res_capacity_cells_nc,
        "res_bus_lookup_csv": args.res_bus_lookup_csv,
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
    settings["other_res_csv"] = (
        resolve_path(settings.get("other_res_csv"), others_input_dir)
        or discover_others_csv(others_input_dir, kind="other_res", ref_year=target_year)
    )
    settings["res_bus_capacity_csv"] = resolve_path(settings.get("res_bus_capacity_csv"), project_root)
    settings["res_potential_csv"] = resolve_path(settings.get("res_potential_csv"), project_root)
    settings["res_capacity_cells_nc"] = resolve_path(settings.get("res_capacity_cells_nc"), project_root)
    settings["res_bus_lookup_csv"] = resolve_path(settings.get("res_bus_lookup_csv"), project_root)
    settings["load_shares_csv"] = resolve_path(settings.get("load_shares_csv"), project_root)
    settings["buses_csv"] = resolve_path(settings.get("buses_csv"), project_root) or network_dir / "buses.csv"
    settings["plants_csv"] = resolve_path(settings.get("plants_csv"), project_root) or network_dir / "plants.csv"
    settings["buses_with_clusters_csv"] = resolve_path(settings.get("buses_with_clusters_csv"), project_root) or network_dir / "buses_with_clusters.csv"
    settings["decommissioned_plants_csv"] = (
        resolve_path(settings.get("decommissioned_plants_csv"), project_root)
        or default_decommissioned_plants_csv(project_root, network_dir)
    )
    settings["output_dir"] = resolve_path(settings.get("output_dir"), project_root) or default_output_dir(project_root, network_dir, "other_res")
    return settings


def normalize_potential_from_capacity(cap: pd.DataFrame) -> pd.DataFrame:
    # Supports long files with p_nom_max_mw already present and the standard
    # RES capacity summary if future runs add the potential column there.
    if "p_nom_max_mw" in cap.columns:
        return cap[["country", "bus_id", "technology", "p_nom_max_mw"]].copy()
    wide_cols = {
        "pv": ["pv_p_nom_max_mw", "p_nom_max_pv_mw", "pv_installable_capacity_mw"],
        "onwind": ["onwind_p_nom_max_mw", "p_nom_max_onwind_mw", "onwind_installable_capacity_mw"],
        "offwind": ["offwind_p_nom_max_mw", "p_nom_max_offwind_mw", "offwind_installable_capacity_mw"],
    }
    rows: list[pd.DataFrame] = []
    for tech, candidates in wide_cols.items():
        col = next((candidate for candidate in candidates if candidate in cap.columns), None)
        if col is not None:
            rows.append(cap[["country", "bus_id", col]].rename(columns={col: "p_nom_max_mw"}).assign(technology=tech))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["country", "bus_id", "technology", "p_nom_max_mw"])


def load_potential_from_cells_nc(cells_nc: Path | None, bus_lookup_csv: Path | None) -> pd.DataFrame:
    if cells_nc is None or bus_lookup_csv is None or not cells_nc.exists() or not bus_lookup_csv.exists():
        return pd.DataFrame(columns=["country", "bus_id", "technology", "p_nom_max_mw"])
    try:
        import xarray as xr
    except Exception as exc:
        raise RuntimeError("xarray is required to aggregate potential from res_capacity_cells.nc") from exc
    lookup = read_csv_auto(bus_lookup_csv)
    missing = {"bus_index", "bus_id", "country"} - set(lookup.columns)
    if missing:
        raise KeyError(f"{bus_lookup_csv} missing columns: {sorted(missing)}")
    lookup = lookup[["bus_index", "bus_id", "country"]].copy()
    lookup["bus_index"] = pd.to_numeric(lookup["bus_index"], errors="coerce").astype("Int64")
    lookup["country"] = lookup["country"].map(norm_country)
    ds = xr.open_dataset(cells_nc)
    if "onshore_bus_index" not in ds:
        return pd.DataFrame(columns=["country", "bus_id", "technology", "p_nom_max_mw"])
    rows = []
    bus_index_vars = {"pv": "onshore_bus_index", "onwind": "onshore_bus_index", "offwind": "offshore_bus_index"}
    for tech in POTENTIAL_TECHS:
        var = f"{tech}_p_nom_max_mw"
        bus_index_var = bus_index_vars[tech]
        if var not in ds or bus_index_var not in ds:
            continue
        bus_index = np.asarray(ds[bus_index_var].values).ravel()
        vals = np.asarray(ds[var].values).ravel()
        frame = pd.DataFrame({"bus_index": bus_index, "technology": tech, "p_nom_max_mw": vals})
        frame = frame[(frame["bus_index"] >= 0) & np.isfinite(frame["p_nom_max_mw"])].copy()
        frame = frame.groupby(["bus_index", "technology"], as_index=False)["p_nom_max_mw"].sum()
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["country", "bus_id", "technology", "p_nom_max_mw"])
    out = pd.concat(rows, ignore_index=True).merge(lookup, how="left", on="bus_index")
    return out.dropna(subset=["bus_id"])[["country", "bus_id", "technology", "p_nom_max_mw"]].copy()


def build_headroom(res_capacity: pd.DataFrame, potential: pd.DataFrame) -> pd.DataFrame:
    cap = res_capacity[res_capacity["technology"].isin(ALLOWED_TECHS)].copy()
    cap = cap.groupby(["country", "bus_id", "technology"], as_index=False)["scenario_capacity_mw"].sum()
    pot = potential[potential["technology"].isin(ALLOWED_TECHS)].copy()
    if pot.empty:
        raise ValueError("No PV/onwind potential data found. Provide res_potential_csv or p_nom_max_mw columns in res_bus_capacity_csv.")
    pot = pot.groupby(["country", "bus_id", "technology"], as_index=False)["p_nom_max_mw"].sum()
    out = pot.merge(cap, how="left", on=["country", "bus_id", "technology"]).fillna({"scenario_capacity_mw": 0.0})
    # Free PV/onshore-wind headroom is used as a weak siting signal for residual
    # renewable categories when no technology-specific location data exist.
    out["free_capacity_mw"] = (out["p_nom_max_mw"] - out["scenario_capacity_mw"]).clip(lower=0.0)
    return out


def build_offwind_eligibility(res_capacity: pd.DataFrame, potential: pd.DataFrame) -> pd.DataFrame:
    cap = res_capacity[res_capacity["technology"].eq("offwind")].copy()
    cap = (
        cap.groupby(["country", "bus_id"], as_index=False)["scenario_capacity_mw"].sum()
        if not cap.empty
        else pd.DataFrame(columns=["country", "bus_id", "scenario_capacity_mw"])
    )
    pot = potential[potential["technology"].eq("offwind")].copy()
    pot = (
        pot.groupby(["country", "bus_id"], as_index=False)["p_nom_max_mw"].sum()
        if not pot.empty
        else pd.DataFrame(columns=["country", "bus_id", "p_nom_max_mw"])
    )
    out = pot.merge(cap, how="outer", on=["country", "bus_id"]).fillna(0.0)
    out["offwind_free_capacity_mw"] = (out["p_nom_max_mw"] - out["scenario_capacity_mw"]).clip(lower=0.0)
    out["offwind_installed_capacity_mw"] = out["scenario_capacity_mw"].clip(lower=0.0)
    out["marine_eligible"] = (out["offwind_free_capacity_mw"] > 0.0) | (out["offwind_installed_capacity_mw"] > 0.0)
    return out[
        [
            "country",
            "bus_id",
            "offwind_free_capacity_mw",
            "offwind_installed_capacity_mw",
            "marine_eligible",
        ]
    ].copy()


def build_pypsa_other_res_basis(plant_rows: pd.DataFrame) -> pd.DataFrame:
    if plant_rows.empty:
        return pd.DataFrame(columns=["country", "bus_id", "other_res_kind", "pypsa_basis_capacity_mw"])
    basis = plant_rows[plant_rows["fueltype"].map(is_other_res_pypsa_fuel)].copy()
    if basis.empty:
        return pd.DataFrame(columns=["country", "bus_id", "other_res_kind", "pypsa_basis_capacity_mw"])
    basis["other_res_kind"] = basis["fueltype"].map(fuel_group_from_raw_fuel)
    return (
        basis.groupby(["country", "bus_id", "other_res_kind"], as_index=False)["capacity_mw"]
        .sum()
        .rename(columns={"capacity_mw": "pypsa_basis_capacity_mw"})
    )


def load_bus_resource_class(cells_nc: Path | None, bus_lookup_csv: Path | None) -> pd.DataFrame:
    if cells_nc is None or bus_lookup_csv is None or not Path(cells_nc).exists() or not Path(bus_lookup_csv).exists():
        return pd.DataFrame(columns=["country", "bus_id", "resource_class"])
    try:
        import xarray as xr
    except Exception:
        return pd.DataFrame(columns=["country", "bus_id", "resource_class"])
    lookup = read_csv_auto(Path(bus_lookup_csv))
    if not {"bus_index", "bus_id", "country"}.issubset(lookup.columns):
        return pd.DataFrame(columns=["country", "bus_id", "resource_class"])
    lookup = lookup[["bus_index", "bus_id", "country"]].copy()
    lookup["bus_index"] = pd.to_numeric(lookup["bus_index"], errors="coerce").astype("Int64")
    lookup["country"] = lookup["country"].map(norm_country)
    ds = xr.open_dataset(cells_nc)
    if "onwind_resource_class" not in ds or "onshore_bus_index" not in ds:
        return pd.DataFrame(columns=["country", "bus_id", "resource_class"])
    bus_index = np.asarray(ds["onshore_bus_index"].values).ravel()
    resource_class = np.asarray(ds["onwind_resource_class"].values).ravel()
    out = pd.DataFrame({"bus_index": bus_index, "resource_class": resource_class})
    out = out[(out["bus_index"] >= 0) & np.isfinite(out["resource_class"])].copy()
    if out.empty:
        return pd.DataFrame(columns=["country", "bus_id", "resource_class"])
    out = out.groupby("bus_index", as_index=False)["resource_class"].max()
    out = out.merge(lookup, how="left", on="bus_index")
    return out.dropna(subset=["bus_id"])[["country", "bus_id", "resource_class"]].copy()


def allocate_marine_country(
    *,
    country: str,
    target_mw: float,
    other_res_technology: str,
    offwind_eligibility: pd.DataFrame,
    load_shares: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = offwind_eligibility[
        offwind_eligibility["country"].eq(country) & offwind_eligibility["marine_eligible"]
    ].copy()
    eligible = eligible.merge(load_shares[load_shares["country"].eq(country)][["bus_id", "load_share"]], how="inner", on="bus_id")
    eligible = eligible.sort_values("bus_id").reset_index(drop=True)
    rows = []
    if not eligible.empty and target_mw > 0.0:
        # Marine technologies are restricted to offshore-wind candidate buses but
        # are not ranked by the wind resource itself; the data usually describe a
        # national marine target rather than site-specific projects.
        amount = float(target_mw) / len(eligible)
        for idx, bus in enumerate(eligible.itertuples(index=False), start=1):
            rows.append(
                {
                    "country": country,
                    "bus_id": str(bus.bus_id),
                    "other_res_technology": str(other_res_technology),
                    "other_res_technology_norm": "marine",
                    "other_res_kind": "marine",
                    "capacity_mw": amount,
                    "load_share": float(bus.load_share),
                    "resource_class": 0.0,
                    "allocation_mode": "marine_equal_offwind_eligible",
                    "priority_stage": 1,
                }
            )
    allocated = float(sum(row["capacity_mw"] for row in rows))
    return pd.DataFrame(rows), {
        "country": country,
        "other_res_technology": str(other_res_technology),
        "other_res_technology_norm": "marine",
        "target_capacity_mw": float(target_mw),
        "allocated_capacity_mw": allocated,
        "unallocated_capacity_mw": float(max(0.0, target_mw - allocated)),
        "n_candidate_buses": int(len(eligible)),
        "allocation_mode": "marine_equal_offwind_eligible",
    }


def allocate_round_robin_other_res(
    *,
    country: str,
    other_res_kind_value: str,
    other_res_technology: str,
    remaining_mw: float,
    load_shares: pd.DataFrame,
    resource_class: pd.DataFrame,
) -> pd.DataFrame:
    if remaining_mw <= 0.0:
        return pd.DataFrame()
    candidates = load_shares[load_shares["country"].eq(country)][["country", "bus_id", "load_share"]].copy()
    if candidates.empty:
        return pd.DataFrame()
    candidates = candidates.merge(resource_class[["country", "bus_id", "resource_class"]], how="left", on=["country", "bus_id"])
    candidates["resource_class"] = pd.to_numeric(candidates["resource_class"], errors="coerce").fillna(0.0)
    ascending_load = False if other_res_kind_value == "waste" else True
    # Waste follows load centres; bio/geothermal-like residuals are pushed toward
    # lower load shares and better generic resource classes to avoid reusing the
    # same urban buses for every missing technology.
    candidates = candidates.sort_values(["load_share", "resource_class", "bus_id"], ascending=[ascending_load, False, True]).reset_index(drop=True)
    amount = float(remaining_mw) / len(candidates)
    rows = []
    for idx, bus in enumerate(candidates.itertuples(index=False), start=1):
        rows.append(
            {
                "country": country,
                "bus_id": str(bus.bus_id),
                "other_res_technology": str(other_res_technology),
                "other_res_technology_norm": normalize_other_res_technology(other_res_technology),
                "other_res_kind": other_res_kind_value,
                "capacity_mw": amount,
                "load_share": float(bus.load_share),
                "resource_class": float(bus.resource_class),
                "allocation_mode": "round_robin_high_load_resource_class" if other_res_kind_value == "waste" else "round_robin_low_load_resource_class",
                "priority_stage": 999,
            }
        )
    return pd.DataFrame(rows)


def allocate_other_res_country(
    *,
    country: str,
    target_mw: float,
    other_res_technology: str,
    pypsa_basis: pd.DataFrame,
    decommissioned_sites: pd.DataFrame,
    load_shares: pd.DataFrame,
    resource_class: pd.DataFrame,
    decommissioning_usage: dict[tuple[str, str, str], float],
    used_site_fuels: dict[tuple[str, str], set[str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    kind = other_res_kind(other_res_technology)
    # Allocation order: explicit PyPSA residual sites, then recent brownfield
    # sites, then a transparent load/resource-class fallback.
    basis = pypsa_basis[
        pypsa_basis["country"].eq(country) & pypsa_basis["other_res_kind"].eq(kind)
    ].copy() if kind in DECOMMISSIONING_OTHER_RES_KINDS else pd.DataFrame()
    basis = basis.rename(columns={"pypsa_basis_capacity_mw": "basis_capacity_mw"})
    basis_alloc, remaining = allocate_capacity_by_capped_basis(
        basis,
        target_capacity_mw=float(target_mw),
        capacity_col="basis_capacity_mw",
        allocation_mode="pypsa_other_res_basis",
    )
    rows = []
    if not basis_alloc.empty:
        for row in basis_alloc.itertuples(index=False):
            rows.append(
                {
                    "country": country,
                    "bus_id": str(row.bus_id),
                    "other_res_technology": str(other_res_technology),
                    "other_res_technology_norm": normalize_other_res_technology(other_res_technology),
                    "other_res_kind": kind,
                    "capacity_mw": float(row.capacity_mw),
                    "load_share": 0.0,
                    "resource_class": 0.0,
                    "allocation_mode": "pypsa_other_res_basis",
                    "priority_stage": 1,
                }
            )
    if kind in DECOMMISSIONING_OTHER_RES_KINDS:
        decom_alloc, remaining = allocate_capacity_to_decommissioned_sites(
            country=country,
            target_fuel_group=kind,
            target_capacity_mw=remaining,
            decommissioned_sites=decommissioned_sites,
            usage_mw=decommissioning_usage,
            used_site_fuel_groups=used_site_fuels,
            context="other_res",
        )
    else:
        decom_alloc = pd.DataFrame()
    if not decom_alloc.empty:
        for row in decom_alloc.itertuples(index=False):
            rows.append(
                {
                    "country": country,
                    "bus_id": str(row.bus_id),
                    "other_res_technology": str(other_res_technology),
                    "other_res_technology_norm": normalize_other_res_technology(other_res_technology),
                    "other_res_kind": kind,
                    "capacity_mw": float(row.capacity_mw),
                    "load_share": 0.0,
                    "resource_class": 0.0,
                    "allocation_mode": str(row.allocation_mode),
                    "priority_stage": 2,
                }
            )
    rr_alloc = allocate_round_robin_other_res(
        country=country,
        other_res_kind_value=kind,
        other_res_technology=other_res_technology,
        remaining_mw=remaining,
        load_shares=load_shares,
        resource_class=resource_class,
    )
    if not rr_alloc.empty:
        rows.extend(rr_alloc.to_dict("records"))
        remaining = 0.0
    allocated = float(sum(float(row["capacity_mw"]) for row in rows))
    return pd.DataFrame(rows), {
        "country": country,
        "other_res_technology": str(other_res_technology),
        "other_res_technology_norm": normalize_other_res_technology(other_res_technology),
        "target_capacity_mw": float(target_mw),
        "allocated_capacity_mw": allocated,
        "unallocated_capacity_mw": float(max(0.0, target_mw - allocated)),
        "n_candidate_buses": int(len({str(row["bus_id"]) for row in rows})),
        "allocation_mode": "pypsa_decommissioning_round_robin" if kind in DECOMMISSIONING_OTHER_RES_KINDS else "round_robin_only",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = resolve_settings(parse_args())
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    res_bus_capacity_csv = require_existing_file(
        settings.get("res_bus_capacity_csv"),
        "res_bus_capacity_csv",
        "Run the RES capacity preprocessing for the same network case or update the powerplants YAML.",
    )
    load_shares_csv = require_existing_file(
        settings.get("load_shares_csv"),
        "load_shares_csv",
        "Run the load disaggregation for the same network case or update the powerplants YAML.",
    )
    other_res_csv = require_existing_file(
        settings.get("other_res_csv"),
        "other_res_csv",
        "Check the TYNDP input directory and target year.",
    )
    buses_csv = require_existing_file(settings.get("buses_csv"), "buses_csv", "Provide the reduced-grid buses.csv.")
    plants_csv = require_existing_file(settings.get("plants_csv"), "plants_csv", "Provide the reduced-grid plants.csv.")

    res_capacity = load_res_bus_capacity(res_bus_capacity_csv)
    potential = load_res_potential(settings.get("res_potential_csv"))
    if potential.empty:
        potential = normalize_potential_from_capacity(res_capacity)
    if potential.empty:
        res_capacity_cells_nc = require_existing_file(
            settings.get("res_capacity_cells_nc"),
            "res_capacity_cells_nc",
            "Provide res_potential_csv or keep the RES capacity preprocessing outputs together.",
        )
        res_bus_lookup_csv = require_existing_file(
            settings.get("res_bus_lookup_csv"),
            "res_bus_lookup_csv",
            "Provide res_potential_csv or keep the RES capacity preprocessing outputs together.",
        )
        potential = load_potential_from_cells_nc(res_capacity_cells_nc, res_bus_lookup_csv)
    for frame in (res_capacity, potential):
        if "country" in frame.columns:
            frame["country"] = frame["country"].map(norm_country)
        if "technology" in frame.columns:
            frame["technology"] = frame["technology"].map(normalize_res_technology)
    load_shares = require_positive_load_share_buses(load_load_shares(load_shares_csv))
    headroom = restrict_to_load_buses(build_headroom(res_capacity, potential), load_shares)
    offwind_eligibility = restrict_to_load_buses(build_offwind_eligibility(res_capacity, potential), load_shares)
    bus_membership = load_bus_country_membership(
        buses_csv=buses_csv,
        buses_with_clusters_csv=settings.get("buses_with_clusters_csv"),
        country_allocation_mode=str(settings.get("country_allocation_mode", "bus_country")),
    )
    bus_membership = restrict_to_load_buses(bus_membership, load_shares)
    plant_rows = load_network_plant_rows(
        plants_csv=plants_csv,
        buses_csv=buses_csv,
        bus_country_membership=bus_membership,
    )
    plant_rows = restrict_to_load_buses(plant_rows, load_shares)
    pypsa_basis = build_pypsa_other_res_basis(plant_rows)
    decommissioned_sites = load_decommissioned_site_basis(
        plants_with_bus_csv=settings.get("decommissioned_plants_csv"),
        buses_with_clusters_csv=settings.get("buses_with_clusters_csv"),
        bus_country_membership=bus_membership,
        load_shares=load_shares,
        target_year=int(settings["target_year"]),
        lookback_years=int(settings.get("decommissioning_lookback_years", 10) or 10),
        allowed_fuel_groups={"bio", "waste", "geothermal"},
    )
    resource_class = load_bus_resource_class(settings.get("res_capacity_cells_nc"), settings.get("res_bus_lookup_csv"))
    countries = settings.get("countries") or sorted(load_shares["country"].dropna().astype(str).unique().tolist())
    countries = [norm_country(c) for c in countries]
    country_map = source_country_mapping_from_load_shares(load_shares_csv)
    target_filter_countries = sorted(set(countries) | set(country_map.keys()))
    targets, targets_df = load_country_capacity_targets(
        other_res_csv,
        countries=target_filter_countries,
        ref_year=int(settings["target_year"]),
        scenario=str(settings["scenario"]),
        group_col_candidates=("technology", "tech", "source_technology", "res_technology", "category", "type"),
        group_col_output=OTHER_RES_TECH_GROUP,
    )
    if OTHER_RES_TECH_GROUP not in targets_df.columns:
        targets_df[OTHER_RES_TECH_GROUP] = "unspecified"
    targets, targets_df = aggregate_targets_with_country_map(
        targets_df,
        country_map=country_map,
        target_countries=countries,
        group_cols=(OTHER_RES_TECH_GROUP,),
    )
    if OTHER_RES_TECH_GROUP not in targets_df.columns:
        targets_df[OTHER_RES_TECH_GROUP] = "unspecified"

    allocation_frames: list[pd.DataFrame] = []
    diag_rows: list[dict[str, Any]] = []
    decommissioning_usage: dict[tuple[str, str, str], float] = {}
    used_site_fuels: dict[tuple[str, str], set[str]] = {}
    for target_row in targets_df.sort_values(["country", OTHER_RES_TECH_GROUP]).itertuples(index=False):
        country = str(target_row.country)
        target = float(target_row.target_capacity_mw)
        other_res_technology = str(getattr(target_row, OTHER_RES_TECH_GROUP, "unspecified") or "unspecified")
        if float(target) <= 0.0:
            continue
        if is_marine_other_res(other_res_technology):
            allocation, diag = allocate_marine_country(
                country=country,
                target_mw=float(target),
                other_res_technology=other_res_technology,
                offwind_eligibility=offwind_eligibility,
                load_shares=load_shares,
            )
        else:
            allocation, diag = allocate_other_res_country(
                country=country,
                target_mw=float(target),
                other_res_technology=other_res_technology,
                pypsa_basis=pypsa_basis,
                decommissioned_sites=decommissioned_sites,
                load_shares=load_shares,
                resource_class=resource_class,
                decommissioning_usage=decommissioning_usage,
                used_site_fuels=used_site_fuels,
            )
        allocation_frames.append(allocation)
        diag_rows.append(diag)
    tech_out = pd.concat(allocation_frames, ignore_index=True) if allocation_frames else pd.DataFrame()
    if not tech_out.empty:
        tech_out["target_year"] = int(settings["target_year"])
        tech_out["scenario"] = str(settings["scenario"])
        share_lookup = load_shares[["country", "bus_id", "load_share"]].drop_duplicates()
        tech_out = tech_out.drop(columns=["load_share"], errors="ignore").merge(share_lookup, how="left", on=["country", "bus_id"])
        tech_out["load_share"] = pd.to_numeric(tech_out["load_share"], errors="coerce").fillna(0.0)
        if "resource_class" not in tech_out.columns:
            tech_out["resource_class"] = 0.0
    bus_out = (
        tech_out.groupby(["target_year", "scenario", "country", "bus_id"], as_index=False)
        .agg(
            capacity_mw=("capacity_mw", "sum"),
            load_share=("load_share", "first"),
            resource_class=("resource_class", "max"),
            source_other_res_technologies=(
                "other_res_technology",
                lambda values: ",".join(sorted(set(str(value) for value in values if str(value).strip()))),
            ),
            source_other_res_technology_norms=(
                "other_res_technology_norm",
                lambda values: ",".join(sorted(set(str(value) for value in values if str(value).strip()))),
            ),
            source_other_res_kinds=(
                "other_res_kind",
                lambda values: ",".join(sorted(set(str(value) for value in values if str(value).strip()))),
            ),
            source_allocation_modes=(
                "allocation_mode",
                lambda values: ",".join(sorted(set(str(value) for value in values if str(value).strip()))),
            ),
        )
        if not tech_out.empty
        else pd.DataFrame(
            columns=[
                "target_year",
                "scenario",
                "country",
                "bus_id",
                "capacity_mw",
                "load_share",
                "resource_class",
                "source_other_res_technologies",
                "source_other_res_technology_norms",
                "source_other_res_kinds",
                "source_allocation_modes",
            ]
        )
    )
    if not bus_out.empty:
        bus_out["technology"] = "other_res"
        bus_out["n_units"] = 1
        bus_out["unit_capacity_mw"] = bus_out["capacity_mw"]
        bus_out["unit_id"] = [
            f"other_res|{row.country}|{row.bus_id}"
            for row in bus_out[["country", "bus_id"]].itertuples(index=False)
        ]
    diag_df = pd.DataFrame(diag_rows)

    outputs = {
        "other_res_capacity_country_bus": output_dir / "other_res_capacity_country_bus.csv",
        "other_res_capacity_country_bus_tech": output_dir / "other_res_capacity_country_bus_tech.csv",
        "other_res_headroom": output_dir / "other_res_headroom_pv_onwind.csv",
        "other_res_pypsa_basis": output_dir / "other_res_pypsa_basis_bus.csv",
        "other_res_decommissioned_site_basis": output_dir / "other_res_decommissioned_site_basis.csv",
        "other_res_resource_class": output_dir / "other_res_resource_class_bus.csv",
        "other_res_marine_offwind_eligibility": output_dir / "other_res_marine_offwind_eligibility.csv",
        "other_res_country_targets": output_dir / "other_res_country_targets.csv",
        "other_res_allocation_diagnostics": output_dir / "other_res_allocation_diagnostics.csv",
        "manifest": output_dir / "other_res_disaggregation_manifest.json",
    }
    write_csv(outputs["other_res_capacity_country_bus"], bus_out)
    write_csv(outputs["other_res_capacity_country_bus_tech"], tech_out)
    write_csv(outputs["other_res_headroom"], headroom)
    write_csv(outputs["other_res_pypsa_basis"], pypsa_basis)
    write_csv(outputs["other_res_decommissioned_site_basis"], decommissioned_sites)
    write_csv(outputs["other_res_resource_class"], resource_class)
    write_csv(outputs["other_res_marine_offwind_eligibility"], offwind_eligibility)
    write_csv(outputs["other_res_country_targets"], targets_df)
    write_csv(outputs["other_res_allocation_diagnostics"], diag_df)
    write_json(outputs["manifest"], build_manifest(settings, outputs, {"allocated_capacity_mw": float(bus_out["capacity_mw"].sum()) if not bus_out.empty else 0.0}))
    LOG.info("wrote Other RES outputs to %s", output_dir)


if __name__ == "__main__":
    main()
