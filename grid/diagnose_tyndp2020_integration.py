from __future__ import annotations

"""Diagnose how selected TYNDP 2020 expansion projects enter the grid model.

This script is used before and after grid reduction to check whether planned
lines, upgrades, and links can be matched to the base grid and whether they
survive the chosen aggregation. The diagnostics are separate from the reduction
itself so that data-integration assumptions can be inspected without changing
the reduced network.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


EARTH_R_KM = 6371.0088

BASE_DATA_DEFAULT = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf\grid")
GRID_FOLDER_DEFAULT = Path(r"xiong2025 v07")
OUTPUT_FOLDER_DEFAULT = "modified"

TYNDP_DATASETS = {
    "new_lines": Path(r"tyndp2020_nt_transmission_projects\tyndp2020_new_lines.csv"),
    "new_links": Path(r"tyndp2020_nt_transmission_projects\tyndp2020_new_links.csv"),
    "upgraded_lines": Path(r"tyndp2020_nt_transmission_projects\tyndp2020_upgraded_lines.csv"),
    "upgraded_links": Path(r"tyndp2020_nt_transmission_projects\tyndp2020_upgraded_links.csv"),
}

BOOL_TRUE = {"t", "true", "1", "yes", "y"}


@dataclass(frozen=True)
class BusSnapper:
    tree: BallTree
    bus_ids: np.ndarray
    countries: np.ndarray
    voltages: np.ndarray
    dc_flags: np.ndarray
    lats: np.ndarray
    lons: np.ndarray


def _detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        header = fh.readline()

    counts = {sep: header.count(sep) for sep in ("\t", ";", ",")}
    sep, count = max(counts.items(), key=lambda kv: kv[1])
    return sep if count > 0 else ","


def _to_numeric_loose(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce")

    out = values.astype(str).str.strip()
    out = out.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
    out = out.str.replace(" ", "", regex=False)

    comma_decimal = out.str.contains(",", regex=False, na=False) & ~out.str.contains(".", regex=False, na=False)
    out = out.where(~comma_decimal, out.str.replace(",", ".", regex=False))
    return pd.to_numeric(out, errors="coerce")


def _to_bool(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin(BOOL_TRUE)


def _mode_first(values: pd.Series) -> Any:
    s = values.dropna()
    if s.empty:
        return np.nan
    mode = s.mode(dropna=True)
    if mode.empty:
        return np.nan
    return mode.iloc[0]


def _unique_sorted_join(values: pd.Series, sep: str = "|") -> str:
    vals = sorted(pd.unique(values.dropna().astype(str)))
    return sep.join(vals) if vals else ""


def _normalize_voltage_class(value: Any) -> str:
    txt = str(value).strip()
    if txt in {"380", "400"}:
        return "380_400"
    return txt


def _pair_key_from_arrays(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=str)
    right = np.asarray(right, dtype=str)
    left_right = np.char.add(np.char.add(left, "||"), right)
    right_left = np.char.add(np.char.add(right, "||"), left)
    return np.where(left <= right, left_right, right_left)


def _pair_key_series(left: pd.Series, right: pd.Series) -> pd.Series:
    return pd.Series(
        _pair_key_from_arrays(
            left.fillna("").astype(str).to_numpy(dtype=object),
            right.fillna("").astype(str).to_numpy(dtype=object),
        ),
        index=left.index,
        dtype="object",
    )


def _country_pair_key(left: pd.Series, right: pd.Series) -> pd.Series:
    left_u = left.fillna("").astype(str).str.upper()
    right_u = right.fillna("").astype(str).str.upper()
    return _pair_key_series(left_u, right_u)


def _snap_quality(max_distance_km: pd.Series) -> pd.Series:
    out = pd.Series("poor", index=max_distance_km.index, dtype="object")
    out.loc[max_distance_km.le(20.0)] = "review"
    out.loc[max_distance_km.le(10.0)] = "good"
    out.loc[max_distance_km.isna()] = "missing"
    return out


def read_net_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=",", quotechar="'", engine="python")


def read_tyndp_csv(path: Path) -> pd.DataFrame:
    sep = _detect_delimiter(path)
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig", engine="python")


def prep_buses(buses: pd.DataFrame) -> pd.DataFrame:
    b = buses.copy()
    b["bus_id"] = b["bus_id"].astype(str)
    b["lat"] = _to_numeric_loose(b["y"] if "y" in b.columns else b["lat"])
    b["lon"] = _to_numeric_loose(b["x"] if "x" in b.columns else b["lon"])
    b["voltage"] = _to_numeric_loose(b["voltage"])
    b["dc_bool"] = _to_bool(b["dc"])
    b["country"] = b["country"].astype(str).str.upper()
    b = b.dropna(subset=["lat", "lon", "voltage"]).copy()
    b["voltage"] = b["voltage"].astype(int)
    return b


def prep_lines(lines: pd.DataFrame) -> pd.DataFrame:
    out = lines.copy()
    out["line_id"] = out["line_id"].astype(str)
    out["bus0"] = out["bus0"].astype(str)
    out["bus1"] = out["bus1"].astype(str)
    out["voltage"] = _to_numeric_loose(out["voltage"])
    out["i_nom"] = _to_numeric_loose(out["i_nom"])
    out["circuits"] = _to_numeric_loose(out["circuits"]).fillna(1.0)
    out["s_nom"] = _to_numeric_loose(out["s_nom"])
    out["r"] = _to_numeric_loose(out["r"])
    out["x"] = _to_numeric_loose(out["x"])
    out["b"] = _to_numeric_loose(out["b"])
    out["length"] = _to_numeric_loose(out["length"])
    out["type"] = out["type"].astype(str).str.strip()
    return out


def prep_links(links: pd.DataFrame) -> pd.DataFrame:
    out = links.copy()
    out["link_id"] = out["link_id"].astype(str)
    out["bus0"] = out["bus0"].astype(str)
    out["bus1"] = out["bus1"].astype(str)
    out["voltage"] = _to_numeric_loose(out["voltage"])
    out["p_nom"] = _to_numeric_loose(out["p_nom"])
    out["length"] = _to_numeric_loose(out["length"])
    return out


def build_bus_snapper(buses: pd.DataFrame) -> BusSnapper:
    coords = np.deg2rad(np.c_[buses["lat"].to_numpy(dtype=float), buses["lon"].to_numpy(dtype=float)])
    return BusSnapper(
        tree=BallTree(coords, metric="haversine"),
        bus_ids=buses["bus_id"].astype(str).to_numpy(dtype=object),
        countries=buses["country"].astype(str).to_numpy(dtype=object),
        voltages=buses["voltage"].to_numpy(dtype=float),
        dc_flags=buses["dc_bool"].to_numpy(dtype=bool),
        lats=buses["lat"].to_numpy(dtype=float),
        lons=buses["lon"].to_numpy(dtype=float),
    )


def snap_project_endpoints(projects: pd.DataFrame, snapper: BusSnapper, *, prefix: str) -> pd.DataFrame:
    out = projects.copy()
    for endpoint in ("0", "1"):
        x_col = f"x{endpoint}"
        y_col = f"y{endpoint}"
        valid = out[x_col].notna() & out[y_col].notna()

        idx = np.full(len(out), -1, dtype=int)
        dist_km = np.full(len(out), np.nan, dtype=float)

        if valid.any():
            coords = np.deg2rad(
                np.c_[
                    out.loc[valid, y_col].to_numpy(dtype=float),
                    out.loc[valid, x_col].to_numpy(dtype=float),
                ]
            )
            d, ind = snapper.tree.query(coords, k=1)
            idx_valid = valid.to_numpy()
            idx[idx_valid] = ind[:, 0]
            dist_km[idx_valid] = d[:, 0] * EARTH_R_KM

        snapped = idx >= 0

        bus_values = np.full(len(out), pd.NA, dtype=object)
        country_values = np.full(len(out), pd.NA, dtype=object)
        voltage_values = np.full(len(out), np.nan, dtype=float)
        dc_values = np.full(len(out), pd.NA, dtype=object)
        lat_values = np.full(len(out), np.nan, dtype=float)
        lon_values = np.full(len(out), np.nan, dtype=float)

        bus_values[snapped] = snapper.bus_ids[idx[snapped]]
        country_values[snapped] = snapper.countries[idx[snapped]]
        voltage_values[snapped] = snapper.voltages[idx[snapped]]
        dc_values[snapped] = snapper.dc_flags[idx[snapped]]
        lat_values[snapped] = snapper.lats[idx[snapped]]
        lon_values[snapped] = snapper.lons[idx[snapped]]

        out[f"{prefix}_bus{endpoint}"] = bus_values
        out[f"{prefix}_country{endpoint}"] = country_values
        out[f"{prefix}_voltage{endpoint}_kv"] = voltage_values
        out[f"{prefix}_dc{endpoint}"] = dc_values
        out[f"{prefix}_lat{endpoint}"] = lat_values
        out[f"{prefix}_lon{endpoint}"] = lon_values
        out[f"{prefix}_distance{endpoint}_km"] = dist_km

    out[f"{prefix}_pair_key"] = _pair_key_series(out[f"{prefix}_bus0"], out[f"{prefix}_bus1"])
    out[f"{prefix}_country_pair_key"] = _country_pair_key(out[f"{prefix}_country0"], out[f"{prefix}_country1"])
    out[f"{prefix}_max_distance_km"] = out[[f"{prefix}_distance0_km", f"{prefix}_distance1_km"]].max(axis=1)
    out[f"{prefix}_quality"] = _snap_quality(out[f"{prefix}_max_distance_km"])
    return out


def build_line_lookup(lines: pd.DataFrame) -> pd.DataFrame:
    out = lines.copy()
    out["pair_key"] = _pair_key_series(out["bus0"], out["bus1"])
    out["voltage_class"] = out["voltage"].map(_normalize_voltage_class)
    out["s_nom_per_circuit_mva"] = out["s_nom"] / out["circuits"].replace(0.0, np.nan)
    out["r_per_km"] = out["r"] / out["length"].replace(0.0, np.nan)
    out["x_per_km"] = out["x"] / out["length"].replace(0.0, np.nan)
    out["b_per_km"] = out["b"] / out["length"].replace(0.0, np.nan)

    grouped = (
        out.groupby("pair_key", as_index=False)
        .agg(
            matched_line_count=("line_id", "size"),
            matched_line_ids=("line_id", _unique_sorted_join),
            matched_line_voltages_kv=("voltage", _unique_sorted_join),
            matched_line_voltage_classes=("voltage_class", _unique_sorted_join),
            matched_line_types=("type", _unique_sorted_join),
        )
    )
    return grouped


def build_link_lookup(links: pd.DataFrame, buses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    bus_country = buses.set_index("bus_id")["country"]

    out = links.copy()
    out["pair_key"] = _pair_key_series(out["bus0"], out["bus1"])
    out["country0"] = out["bus0"].map(bus_country)
    out["country1"] = out["bus1"].map(bus_country)
    out["country_pair_key"] = _country_pair_key(out["country0"], out["country1"])

    pair_lookup = (
        out.groupby("pair_key", as_index=False)
        .agg(
            matched_link_count=("link_id", "size"),
            matched_link_ids=("link_id", _unique_sorted_join),
            matched_link_voltages_kv=("voltage", _unique_sorted_join),
            inferred_voltage_from_pair_kv=("voltage", _mode_first),
        )
    )

    country_pair_lookup = (
        out.groupby("country_pair_key", as_index=False)
        .agg(
            country_pair_existing_link_count=("link_id", "size"),
            country_pair_link_ids=("link_id", _unique_sorted_join),
            country_pair_link_voltages_kv=("voltage", _unique_sorted_join),
            inferred_voltage_from_country_pair_kv=("voltage", _mode_first),
        )
    )

    global_mode = float(_mode_first(out["voltage"]))
    return pair_lookup, country_pair_lookup, global_mode


def build_line_parameter_templates(lines: pd.DataFrame) -> pd.DataFrame:
    out = lines.copy()
    out["voltage_class"] = out["voltage"].map(_normalize_voltage_class)
    out["type"] = out["type"].astype(str).str.strip()
    out["s_nom_per_circuit_mva"] = out["s_nom"] / out["circuits"].replace(0.0, np.nan)
    out["r_per_km"] = out["r"] / out["length"].replace(0.0, np.nan)
    out["x_per_km"] = out["x"] / out["length"].replace(0.0, np.nan)
    out["b_per_km"] = out["b"] / out["length"].replace(0.0, np.nan)

    templates = (
        out.groupby(["voltage_class", "type"], as_index=False)
        .agg(
            template_asset_count=("line_id", "size"),
            template_i_nom_mode_ka=("i_nom", _mode_first),
            template_circuits_mode=("circuits", _mode_first),
            template_s_nom_per_circuit_mva_median=("s_nom_per_circuit_mva", "median"),
            template_r_per_km_median=("r_per_km", "median"),
            template_x_per_km_median=("x_per_km", "median"),
            template_b_per_km_median=("b_per_km", "median"),
        )
        .sort_values(["voltage_class", "type"])
        .reset_index(drop=True)
    )
    return templates


def _project_id_column(columns: list[str]) -> str:
    candidates = [
        "project_id",
        "H1",
        "Unnamed: 0",
        "",
    ]
    by_lower = {str(col).strip().lower(): str(col) for col in columns}
    for cand in candidates:
        match = by_lower.get(cand.strip().lower())
        if match is not None:
            return match
    return str(columns[0])


def read_tyndp_line_projects(path: Path, *, dataset: str) -> pd.DataFrame:
    df = read_tyndp_csv(path)
    project_col = _project_id_column(list(df.columns))

    out = df.rename(
        columns={
            project_col: "project_id",
            "v_nom": "project_voltage_kv",
            "num_parallel": "project_num_parallel",
            "type": "project_line_type",
        }
    ).copy()

    out["dataset"] = dataset
    out["project_variant"] = "new" if dataset.startswith("new_") else "upgraded"
    out["project_kind"] = "line"
    out["project_id"] = out["project_id"].astype(str).str.strip()
    out["project_status"] = out["project_status"].astype(str).str.strip()
    out["build_year"] = _to_numeric_loose(out["build_year"]).astype("Int64")
    out["length_km"] = _to_numeric_loose(out["length"])
    out["length_m"] = out["length_km"] * 1000.0
    out["underground_bool"] = _to_bool(out["underground"])
    out["project_voltage_kv"] = _to_numeric_loose(out["project_voltage_kv"])
    out["project_voltage_class"] = out["project_voltage_kv"].map(_normalize_voltage_class)
    out["project_num_parallel"] = _to_numeric_loose(out["project_num_parallel"]).fillna(1.0)
    out["project_line_type"] = out["project_line_type"].astype(str).str.strip()
    out["x0"] = _to_numeric_loose(out["x0"])
    out["y0"] = _to_numeric_loose(out["y0"])
    out["x1"] = _to_numeric_loose(out["x1"])
    out["y1"] = _to_numeric_loose(out["y1"])
    return out


def read_tyndp_link_projects(path: Path, *, dataset: str) -> pd.DataFrame:
    df = read_tyndp_csv(path)
    project_col = _project_id_column(list(df.columns))

    out = df.rename(
        columns={
            project_col: "project_id",
            "p_nom": "project_p_nom_mw",
        }
    ).copy()

    out["dataset"] = dataset
    out["project_variant"] = "new" if dataset.startswith("new_") else "upgraded"
    out["project_kind"] = "link"
    out["project_id"] = out["project_id"].astype(str).str.strip()
    out["project_status"] = out["project_status"].astype(str).str.strip()
    out["build_year"] = _to_numeric_loose(out["build_year"]).astype("Int64")
    out["length_km"] = _to_numeric_loose(out["length"])
    out["length_m"] = out["length_km"] * 1000.0
    out["underground_bool"] = _to_bool(out["underground"])
    out["project_p_nom_mw"] = _to_numeric_loose(out["project_p_nom_mw"])
    out["x0"] = _to_numeric_loose(out["x0"])
    out["y0"] = _to_numeric_loose(out["y0"])
    out["x1"] = _to_numeric_loose(out["x1"])
    out["y1"] = _to_numeric_loose(out["y1"])
    return out


def classify_line_actions(lines_diag: pd.DataFrame) -> pd.Series:
    action = pd.Series("manual_review", index=lines_diag.index, dtype="object")

    in_target = lines_diag["included_in_target_year"]
    in_snapshot = lines_diag["included_in_base_snapshot_year"]
    snap_good = lines_diag["snap_ac_quality"].eq("good")
    snap_ok = lines_diag["snap_ac_quality"].isin(["good", "review"])
    has_match = lines_diag["matched_line_count"].fillna(0).gt(0)
    relaxed_match = lines_diag["matched_line_voltage_relaxed"].fillna(False)
    upgraded = lines_diag["project_variant"].eq("upgraded")

    action.loc[~in_target] = "skip_after_target_year"
    action.loc[in_target & ~snap_ok] = "manual_review_snap_far"
    action.loc[in_target & snap_ok & ~has_match & ~upgraded] = "candidate_add_new_line"
    action.loc[in_target & snap_ok & ~has_match & upgraded] = "candidate_add_or_reroute_upgraded_line"
    action.loc[in_target & snap_ok & has_match & relaxed_match & in_snapshot & ~upgraded] = "likely_already_in_base"
    action.loc[in_target & snap_ok & has_match & relaxed_match & in_snapshot & upgraded] = "likely_already_in_base"
    action.loc[in_target & snap_ok & has_match & relaxed_match & ~in_snapshot & ~upgraded] = "review_existing_corridor_or_parallel_line"
    action.loc[in_target & snap_ok & has_match & relaxed_match & ~in_snapshot & upgraded] = "candidate_update_existing_line"
    action.loc[in_target & snap_ok & has_match & ~relaxed_match & upgraded] = "review_voltage_upgrade_existing_corridor"
    action.loc[in_target & snap_ok & has_match & ~relaxed_match & ~upgraded] = "review_existing_corridor_voltage_mismatch"
    action.loc[in_target & snap_good & ~lines_diag["template_available"].fillna(False)] = "manual_review_no_line_template"

    return action


def classify_link_actions(links_diag: pd.DataFrame) -> pd.Series:
    action = pd.Series("manual_review", index=links_diag.index, dtype="object")

    in_target = links_diag["included_in_target_year"]
    in_snapshot = links_diag["included_in_base_snapshot_year"]
    has_match = links_diag["matched_link_count"].fillna(0).gt(0)
    good_dc = links_diag["snap_dc_quality"].eq("good")
    good_or_review_dc = links_diag["snap_dc_quality"].isin(["good", "review"])
    good_ac = links_diag["snap_ac_quality"].isin(["good", "review"])
    pair_source = links_diag["inferred_voltage_source"].eq("existing_pair")
    country_source = links_diag["inferred_voltage_source"].eq("country_pair_mode")
    global_source = links_diag["inferred_voltage_source"].eq("global_mode")
    upgraded = links_diag["project_variant"].eq("upgraded")

    action.loc[~in_target] = "skip_after_target_year"
    action.loc[in_target & has_match & in_snapshot] = "likely_already_in_base"
    action.loc[in_target & has_match & ~in_snapshot & upgraded] = "candidate_update_existing_link"
    action.loc[in_target & has_match & ~in_snapshot & ~upgraded] = "review_existing_link_corridor"
    action.loc[in_target & ~has_match & good_or_review_dc & pair_source] = "candidate_add_link_reusing_dc_terminals"
    action.loc[in_target & ~has_match & good_or_review_dc & country_source] = "candidate_add_link_reusing_nearby_dc_terminals"
    action.loc[in_target & ~has_match & ~good_or_review_dc & good_ac & country_source] = "review_add_link_with_new_dc_terminals"
    action.loc[in_target & ~has_match & ~good_or_review_dc & good_ac & global_source] = "manual_review_voltage_and_terminal_topology"
    action.loc[in_target & ~has_match & good_or_review_dc & global_source] = "manual_review_voltage_assumption"
    action.loc[in_target & ~has_match & ~good_ac] = "manual_review_snap_far"
    action.loc[in_target & links_diag["inferred_voltage_selected_kv"].isna()] = "manual_review_no_voltage_inference"
    action.loc[in_target & good_dc & ~has_match & ~pair_source & ~country_source & ~global_source] = "manual_review_no_voltage_inference"

    return action


def simplify_recommendations(detailed_actions: pd.Series) -> tuple[pd.Series, pd.Series]:
    simple = pd.Series("inspect", index=detailed_actions.index, dtype="object")
    note = pd.Series("inspect manually", index=detailed_actions.index, dtype="object")

    ignore_actions = {
        "skip_after_target_year": "ignore: build_year > target_year",
        "likely_already_in_base": "ignore: likely already represented in base grid",
        "manual_review_snap_far": "ignore: endpoint snap too far from plausible bus",
        "manual_review_voltage_and_terminal_topology": "ignore: link voltage and terminal topology too uncertain",
        "manual_review_voltage_assumption": "ignore: link voltage inference too weak",
        "manual_review_no_voltage_inference": "ignore: no reliable link voltage inference",
    }
    for detailed, text in ignore_actions.items():
        mask = detailed_actions.eq(detailed)
        simple.loc[mask] = "ignore"
        note.loc[mask] = text

    new_line_actions = {
        "candidate_add_new_line": "add as new AC line",
    }
    for detailed, text in new_line_actions.items():
        mask = detailed_actions.eq(detailed)
        simple.loc[mask] = "include_new_line"
        note.loc[mask] = text

    link_reuse_actions = {
        "candidate_add_link_reusing_dc_terminals": "add HVDC link and reuse matched DC terminals",
        "candidate_add_link_reusing_nearby_dc_terminals": "add HVDC link and reuse nearby DC terminals",
    }
    for detailed, text in link_reuse_actions.items():
        mask = detailed_actions.eq(detailed)
        simple.loc[mask] = "include_new_link_reuse_dc"
        note.loc[mask] = text

    mask = detailed_actions.eq("review_add_link_with_new_dc_terminals")
    simple.loc[mask] = "include_new_link_to_nearest_ac"
    note.loc[mask] = "add HVDC link and attach directly to nearest AC buses"

    upgrade_actions = {
        "candidate_add_or_reroute_upgraded_line": "treat as upgrade",
        "review_voltage_upgrade_existing_corridor": "treat as voltage upgrade on existing corridor",
        "review_existing_corridor_voltage_mismatch": "treat as upgrade / corridor adaptation",
        "candidate_update_existing_line": "treat as upgrade of existing AC line",
        "candidate_update_existing_link": "treat as upgrade of existing HVDC link",
        "review_existing_link_corridor": "treat as upgrade of existing HVDC corridor",
        "review_existing_corridor_or_parallel_line": "treat as upgrade / existing corridor adaptation",
    }
    for detailed, text in upgrade_actions.items():
        mask = detailed_actions.eq(detailed)
        simple.loc[mask] = "include_as_upgrade"
        note.loc[mask] = text

    mask = detailed_actions.eq("manual_review_no_line_template")
    simple.loc[mask] = "inspect"
    note.loc[mask] = "inspect manually: no matching AC line parameter template"

    return simple, note


def diagnose_line_projects(
    projects: pd.DataFrame,
    *,
    ac_snapper: BusSnapper,
    line_lookup: pd.DataFrame,
    line_templates: pd.DataFrame,
    target_year: int,
    base_snapshot_year: int,
) -> pd.DataFrame:
    out = snap_project_endpoints(projects, ac_snapper, prefix="snap_ac")
    out["included_in_target_year"] = out["build_year"].le(target_year).fillna(False)
    out["included_in_base_snapshot_year"] = out["build_year"].le(base_snapshot_year).fillna(False)

    out = out.merge(
        line_lookup,
        left_on="snap_ac_pair_key",
        right_on="pair_key",
        how="left",
    ).drop(columns=["pair_key"], errors="ignore")

    out = out.merge(
        line_templates,
        left_on=["project_voltage_class", "project_line_type"],
        right_on=["voltage_class", "type"],
        how="left",
    ).drop(columns=["voltage_class", "type"], errors="ignore")

    matched_voltage_classes = out["matched_line_voltage_classes"].fillna("").astype(str).str.split("|")
    out["matched_line_voltage_relaxed"] = [
        (voltage_class in set(classes)) if voltage_class and classes != [""] else False
        for voltage_class, classes in zip(out["project_voltage_class"], matched_voltage_classes, strict=False)
    ]

    out["template_available"] = out["template_asset_count"].fillna(0).gt(0)
    out["detailed_recommendation"] = classify_line_actions(out)
    out["recommended_action"], out["recommended_note"] = simplify_recommendations(out["detailed_recommendation"])
    return out


def diagnose_link_projects(
    projects: pd.DataFrame,
    *,
    dc_snapper: BusSnapper,
    ac_snapper: BusSnapper,
    link_pair_lookup: pd.DataFrame,
    link_country_pair_lookup: pd.DataFrame,
    global_link_voltage_mode: float,
    target_year: int,
    base_snapshot_year: int,
) -> pd.DataFrame:
    out = snap_project_endpoints(projects, dc_snapper, prefix="snap_dc")
    out = snap_project_endpoints(out, ac_snapper, prefix="snap_ac")

    out["included_in_target_year"] = out["build_year"].le(target_year).fillna(False)
    out["included_in_base_snapshot_year"] = out["build_year"].le(base_snapshot_year).fillna(False)

    out = out.merge(
        link_pair_lookup,
        left_on="snap_dc_pair_key",
        right_on="pair_key",
        how="left",
    ).drop(columns=["pair_key"], errors="ignore")

    out = out.merge(
        link_country_pair_lookup,
        left_on="snap_dc_country_pair_key",
        right_on="country_pair_key",
        how="left",
    ).drop(columns=["country_pair_key"], errors="ignore")

    out["inferred_voltage_global_mode_kv"] = float(global_link_voltage_mode)
    out["inferred_voltage_selected_kv"] = out["inferred_voltage_from_pair_kv"]
    out["inferred_voltage_source"] = np.where(
        out["inferred_voltage_from_pair_kv"].notna(),
        "existing_pair",
        "",
    )

    mask_country = out["inferred_voltage_selected_kv"].isna() & out["inferred_voltage_from_country_pair_kv"].notna()
    out.loc[mask_country, "inferred_voltage_selected_kv"] = out.loc[mask_country, "inferred_voltage_from_country_pair_kv"]
    out.loc[mask_country, "inferred_voltage_source"] = "country_pair_mode"

    mask_global = out["inferred_voltage_selected_kv"].isna()
    out.loc[mask_global, "inferred_voltage_selected_kv"] = out.loc[mask_global, "inferred_voltage_global_mode_kv"]
    out.loc[mask_global, "inferred_voltage_source"] = "global_mode"

    out["detailed_recommendation"] = classify_link_actions(out)
    out["recommended_action"], out["recommended_note"] = simplify_recommendations(out["detailed_recommendation"])
    return out


def summarize_projects(diag: pd.DataFrame, *, snap_quality_col: str, match_count_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, group in diag.groupby("dataset", sort=False):
        action_counts = (
            group["recommended_action"]
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        )
        rows.append(
            {
                "dataset": dataset,
                "n_projects": int(len(group)),
                "n_included_in_target_year": int(group["included_in_target_year"].sum()),
                "n_included_in_base_snapshot_year": int(group["included_in_base_snapshot_year"].sum()),
                "n_snap_good": int(group[snap_quality_col].eq("good").sum()),
                "n_snap_review": int(group[snap_quality_col].eq("review").sum()),
                "n_snap_poor": int(group[snap_quality_col].eq("poor").sum()),
                "n_matched_existing_pair": int(group[match_count_col].fillna(0).gt(0).sum()),
                "recommended_actions": "|".join(f"{k}:{v}" for k, v in action_counts.items()),
            }
        )
    return pd.DataFrame(rows)


def write_assumptions_file(path: Path, *, target_year: int, base_snapshot_year: int, global_link_voltage_mode: float) -> None:
    text = f"""TYNDP 2020 integration diagnostics

target_year={target_year}
base_snapshot_year={base_snapshot_year}

Assumptions:
- AC line projects are snapped to existing AC buses only.
- HVDC link projects are snapped to existing DC buses for pair matching and to AC buses for terminal plausibility review.
- For AC line pair compatibility, 380 kV and 400 kV are treated as one voltage class.
- project_status is retained in outputs but not used as a hard filter.
- build_year <= target_year defines whether a project is in-scope for the selected study year.
- build_year <= base_snapshot_year is used only as a hint that a project may already be represented in the base grid.
- For HVDC link voltage inference, the priority is:
  1. voltage of an existing matched DC link pair
  2. mode voltage of existing links for the same country pair
  3. global mode voltage of all existing links

Current global mode voltage of existing links: {global_link_voltage_mode:g} kV

Simplified recommendation classes:
- ignore
- include_new_line
- include_new_link_reuse_dc
- include_new_link_to_nearest_ac
- include_as_upgrade
- inspect
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose how TYNDP 2020 projects can be mapped to the current grid.")
    parser.add_argument("--base-data", type=Path, default=BASE_DATA_DEFAULT)
    parser.add_argument("--grid-folder", type=Path, default=GRID_FOLDER_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--target-year", type=int, default=2025)
    parser.add_argument("--base-snapshot-year", type=int, default=2025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_data = Path(args.base_data)
    grid_folder = Path(args.grid_folder)

    if args.output_root is None:
        output_dir = base_data.parent / OUTPUT_FOLDER_DEFAULT / f"tyndp2020_diagnostics_target_year_{args.target_year}"
    else:
        output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    buses = prep_buses(read_net_csv(base_data / grid_folder / "buses.csv"))
    lines = prep_lines(read_net_csv(base_data / grid_folder / "lines.csv"))
    links = prep_links(read_net_csv(base_data / grid_folder / "links.csv"))

    ac_buses = buses.loc[~buses["dc_bool"]].copy()
    dc_buses = buses.loc[buses["dc_bool"]].copy()

    ac_snapper = build_bus_snapper(ac_buses)
    dc_snapper = build_bus_snapper(dc_buses)

    line_lookup = build_line_lookup(lines)
    line_templates = build_line_parameter_templates(lines)
    link_pair_lookup, link_country_pair_lookup, global_link_voltage_mode = build_link_lookup(links, buses)

    new_lines = read_tyndp_line_projects(base_data / TYNDP_DATASETS["new_lines"], dataset="new_lines")
    upgraded_lines = read_tyndp_line_projects(base_data / TYNDP_DATASETS["upgraded_lines"], dataset="upgraded_lines")
    new_links = read_tyndp_link_projects(base_data / TYNDP_DATASETS["new_links"], dataset="new_links")
    upgraded_links = read_tyndp_link_projects(base_data / TYNDP_DATASETS["upgraded_links"], dataset="upgraded_links")

    diag_new_lines = diagnose_line_projects(
        new_lines,
        ac_snapper=ac_snapper,
        line_lookup=line_lookup,
        line_templates=line_templates,
        target_year=args.target_year,
        base_snapshot_year=args.base_snapshot_year,
    )
    diag_upgraded_lines = diagnose_line_projects(
        upgraded_lines,
        ac_snapper=ac_snapper,
        line_lookup=line_lookup,
        line_templates=line_templates,
        target_year=args.target_year,
        base_snapshot_year=args.base_snapshot_year,
    )
    diag_new_links = diagnose_link_projects(
        new_links,
        dc_snapper=dc_snapper,
        ac_snapper=ac_snapper,
        link_pair_lookup=link_pair_lookup,
        link_country_pair_lookup=link_country_pair_lookup,
        global_link_voltage_mode=global_link_voltage_mode,
        target_year=args.target_year,
        base_snapshot_year=args.base_snapshot_year,
    )
    diag_upgraded_links = diagnose_link_projects(
        upgraded_links,
        dc_snapper=dc_snapper,
        ac_snapper=ac_snapper,
        link_pair_lookup=link_pair_lookup,
        link_country_pair_lookup=link_country_pair_lookup,
        global_link_voltage_mode=global_link_voltage_mode,
        target_year=args.target_year,
        base_snapshot_year=args.base_snapshot_year,
    )

    line_diag = pd.concat([diag_new_lines, diag_upgraded_lines], ignore_index=True, sort=False)
    link_diag = pd.concat([diag_new_links, diag_upgraded_links], ignore_index=True, sort=False)

    summary_lines = summarize_projects(line_diag, snap_quality_col="snap_ac_quality", match_count_col="matched_line_count")
    summary_links = summarize_projects(link_diag, snap_quality_col="snap_dc_quality", match_count_col="matched_link_count")
    summary = pd.concat([summary_lines, summary_links], ignore_index=True, sort=False)

    action_summary = (
        pd.concat([line_diag, link_diag], ignore_index=True, sort=False)
        .groupby(["dataset", "recommended_action"], as_index=False)
        .size()
        .rename(columns={"size": "n_projects"})
        .sort_values(["dataset", "recommended_action"])
        .reset_index(drop=True)
    )
    detailed_action_summary = (
        pd.concat([line_diag, link_diag], ignore_index=True, sort=False)
        .groupby(["dataset", "detailed_recommendation"], as_index=False)
        .size()
        .rename(columns={"size": "n_projects"})
        .sort_values(["dataset", "detailed_recommendation"])
        .reset_index(drop=True)
    )

    diag_new_lines.to_csv(output_dir / "diagnostic_new_lines.csv", sep=";", index=False)
    diag_upgraded_lines.to_csv(output_dir / "diagnostic_upgraded_lines.csv", sep=";", index=False)
    diag_new_links.to_csv(output_dir / "diagnostic_new_links.csv", sep=";", index=False)
    diag_upgraded_links.to_csv(output_dir / "diagnostic_upgraded_links.csv", sep=";", index=False)
    summary.to_csv(output_dir / "summary_by_dataset.csv", sep=";", index=False)
    action_summary.to_csv(output_dir / "summary_actions.csv", sep=";", index=False)
    detailed_action_summary.to_csv(output_dir / "summary_detailed_actions.csv", sep=";", index=False)
    line_templates.to_csv(output_dir / "line_parameter_templates.csv", sep=";", index=False)
    link_country_pair_lookup.to_csv(output_dir / "link_voltage_country_pair_modes.csv", sep=";", index=False)
    write_assumptions_file(
        output_dir / "assumptions.txt",
        target_year=args.target_year,
        base_snapshot_year=args.base_snapshot_year,
        global_link_voltage_mode=global_link_voltage_mode,
    )

    print(f"Wrote diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
