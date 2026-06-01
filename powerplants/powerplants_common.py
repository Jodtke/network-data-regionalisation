from __future__ import annotations

"""Shared data cleaning and allocation helpers for plant-based modules.

The plant disaggregation scripts use a common representation of reduced buses,
country aggregates, plant technologies, and TYNDP target tables. This module
contains the normalisation rules and small allocation utilities that keep
thermal, other non-RES, other RES, and BESS regionalisation comparable. The
helpers avoid optimisation assumptions; they encode deterministic siting
heuristics and diagnostics for transparent scenario construction.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - used in minimal Python environments.
    yaml = None


DEFAULT_PROJECT_ROOT = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf")
DEFAULT_TYNDP_INPUT_DIR = Path(
    r"Y:\Group_SEM\MA_Eric\Dissertation\revision_outage_optimisation\input\single_year"
)
DEFAULT_OTHERS_INPUT_DIR = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\DATA\raw\others\tyndp2024")
DEFAULT_SCENARIO = "NationalTrends"
TYNDP_TARGET_YEARS = (2030, 2040, 2050)
TYNDP_YEAR_PATTERN = re.compile(r"(?<!\d)(?:2030|2040|2050)(?!\d)")

COUNTRY_ALIASES = {
    "UK": "GB",
    "GBR": "GB",
    "UNITED KINGDOM": "GB",
    "UNITEDKINGDOM": "GB",
    "GREAT BRITAIN": "GB",
    "GREATBRITAIN": "GB",
    "NORTHERN IRELAND": "NI",
    "NORTHERNIRELAND": "NI",
    "NORTH IRELAND": "NI",
    "NORTHIRELAND": "NI",
    "NORDIRLAND": "NI",
    "NORD IRLAND": "NI",
    "NIR": "NI",
    "UKRAINE": "UA",
    "UKR": "UA",
    "MOLDOVA": "MD",
    "REPUBLIC OF MOLDOVA": "MD",
    "MOLDOVA REPUBLIC OF": "MD",
    "MOLDAU": "MD",
    "MOLDAWIEN": "MD",
    "MOLDAVIEN": "MD",
    "MOLDAVIA": "MD",
    "MDA": "MD",
    "EL": "GR",
}
PROTECTED_COUNTRY_CODES = {"NI", "UA", "MD"}
NON_THERMAL_FUEL_TOKENS = ("BATTERY", "HYDRO", "SOLAR", "WIND", "HEAT STORAGE", "MECHANICAL STORAGE")
OTHER_RES_FUEL_GROUPS = {"bio", "waste", "geothermal"}
OTHER_RES_FUEL_TOKENS = (
    "BIO",
    "BIOGAS",
    "BIOMASS",
    "BIOFUEL",
    "WASTE",
    "MÜLL",
    "MUELL",
    "GEOTHERM",
)
FUEL_MAP = {
    "GAS": "B04",
    "NATURAL GAS": "B04",
    "HARD COAL": "B05",
    "COAL": "B05",
    "LIGNITE": "B02",
    "BROWN COAL": "B02",
    "OIL": "B06",
    "OIL SHALE": "B07",
    "NUCLEAR": "B14",
    "BIOMASS": "B01",
    "BIOFUEL": "B01",
    "BIOGAS": "B01",
    "WASTE": "B17",
    "GEOTHERMAL": "B09",
    "OTHER": "B20",
    "OTHERS": "B20",
}
TECH_MAP = {
    "CCGT": "CCGT",
    "OCGT": "OCGT",
    "STEAM TURBINE": "STEAM",
    "STEAM": "STEAM",
    "COMBUSTION ENGINE": "OTHERS",
    "OTHER OR UNSPECIFIED TECHNOLOGY": "OTHERS",
}
STD_REV_DUR_BY_TECH = {"CCGT": 3, "OCGT": 2, "STEAM": 4, "NUCLEAR": 4, "OTHERS": 2}
LONG_REV_DUR_BY_TECH = {"CCGT": 6, "OCGT": 4, "STEAM": 8, "NUCLEAR": 8, "OTHERS": 4}
THERMAL_INERTIA_H_BY_FUEL = {
    "B01": 4.0,
    "B02": 5.5,
    "B04": 4.5,
    "B05": 5.5,
    "B06": 5.0,
    "B07": 5.0,
    "B09": 4.0,
    "B14": 6.0,
    "B17": 4.0,
    "B20": 4.0,
}
GAS_INERTIA_H_BY_TECH = {"CCGT": 5.0, "OCGT": 4.0, "OTHERS": 4.5}


def detect_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    if not sample:
        return ";"
    return ";" if sample[0].count(";") >= sample[0].count(",") else ","


def read_csv_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=detect_delimiter(path), low_memory=False).rename(columns=str.strip)


def should_round_zero_decimals(column: Any) -> bool:
    name = str(column or "").strip().lower()
    return name.endswith("_mw") or "eur_mwh" in name or "eur_per_mwh" in name or "eur/mwh" in name


def round_zero_decimal_output_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if not should_round_zero_decimals(column):
            continue
        numeric = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[column] = numeric.round(0).astype("Int64")
    return out


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    round_zero_decimal_output_columns(frame).to_csv(path, index=False, sep=";")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load_simple_yaml_config(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("-") and current_key is not None:
            if not isinstance(data.get(current_key), list):
                data[current_key] = []
            data[current_key].append(line.lstrip()[1:].strip())
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        current_key = key.strip()
        value = value.strip()
        if value == "":
            data[current_key] = None
        elif "," in value and current_key in {"countries", "res_technologies", "steps"}:
            data[current_key] = [item.strip() for item in value.split(",") if item.strip()]
        else:
            data[current_key] = value
    return data


def load_yaml_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if yaml is not None else _load_simple_yaml_config(path)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def resolve_path(path_value: Any, base_dir: Path | None = None) -> Path | None:
    if path_value in (None, ""):
        return None
    text = str(path_value)
    if base_dir is not None:
        text = text.format(project_root=str(base_dir))
    path = Path(text)
    if path.is_absolute():
        return path
    return (base_dir or Path.cwd()) / path


def norm_country(value: Any) -> str:
    code = re.sub(r"[\s_\-]+", " ", str(value or "").strip().upper())
    if code in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[code]
    compact = re.sub(r"[^A-Z0-9]+", "", code)
    return COUNTRY_ALIASES.get(compact, code)


def normalize_column_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("€", "eur").replace("co₂", "co2").replace("co\u2082", "co2")
    text = text.replace("%", "percent")
    return re.sub(r"[^a-z0-9]+", "", text)


def find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    wanted = {normalize_column_key(candidate) for candidate in candidates}
    for column in df.columns:
        if normalize_column_key(column) in wanted:
            return column
    return None


def numeric_column(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def validate_tyndp_target_year(ref_year: int) -> int:
    year = int(ref_year)
    if year not in TYNDP_TARGET_YEARS:
        raise ValueError(
            f"Unsupported TYNDP target year {year}. Expected one of: "
            + ", ".join(str(value) for value in TYNDP_TARGET_YEARS)
        )
    return year


def detect_tyndp_years_in_path(path: Path) -> set[int]:
    return {int(match.group(0)) for match in TYNDP_YEAR_PATTERN.finditer(str(path))}


def validate_path_target_year(path: Path, ref_year: int, *, context: str) -> None:
    year = validate_tyndp_target_year(ref_year)
    path_years = detect_tyndp_years_in_path(path)
    if not path_years:
        raise ValueError(
            f"{context} file has no target-year column and its path does not contain one of "
            f"{list(TYNDP_TARGET_YEARS)}: {path}"
        )
    if path_years and path_years != {year}:
        raise ValueError(
            f"{context} path points to TYNDP year(s) {sorted(path_years)}, "
            f"but the selected target_year is {year}: {path}"
        )


def filter_to_tyndp_target_year(
    frame: pd.DataFrame,
    path: Path,
    *,
    ref_year: int,
    year_col: str | None,
    context: str,
) -> pd.DataFrame:
    year = validate_tyndp_target_year(ref_year)
    if year_col is None:
        validate_path_target_year(path, year, context=context)
        return frame.copy()

    years = numeric_column(frame[year_col]).round()
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
            f"{context} file has column '{year_col}' but no rows for target_year={year}: {path}. "
            f"Available TYNDP years: {available_years or 'none'}."
        )
    return frame[mask].copy()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return float(default)
        return float(str(value).replace(",", "."))
    except Exception:
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return int(default)
        return int(round(float(value)))
    except Exception:
        return int(default)


def map_fuel_code(raw_fuel: Any) -> str:
    fuel = str(raw_fuel or "").strip().upper()
    if fuel in FUEL_MAP:
        return FUEL_MAP[fuel]
    if "HYDROGEN" in fuel:
        return "B101"
    if "NUCLEAR" in fuel:
        return "B14"
    if "LIGNITE" in fuel or "BROWN COAL" in fuel:
        return "B02"
    if "COAL" in fuel:
        return "B05"
    if "GAS" in fuel:
        return "B04"
    if "OIL SHALE" in fuel:
        return "B07"
    if "OIL" in fuel:
        return "B06"
    if "BIO" in fuel:
        return "B01"
    if "WASTE" in fuel:
        return "B17"
    if "GEOTHERM" in fuel:
        return "B09"
    return "B20"


def fuel_group_from_fuel_code(fuel_code: Any) -> str:
    fuel = str(fuel_code or "").strip().upper()
    return {
        "B01": "bio",
        "B02": "lignite",
        "B04": "gas",
        "B05": "hard_coal",
        "B06": "oil",
        "B07": "oil_shale",
        "B09": "geothermal",
        "B14": "nuclear",
        "B17": "waste",
    }.get(fuel, "other")


def fuel_group_from_raw_fuel(raw_fuel: Any) -> str:
    fuel = normalize_column_key(raw_fuel)
    if not fuel or fuel == "nan":
        return "other"
    if "nuclear" in fuel or fuel in {"b14"}:
        return "nuclear"
    if "lignite" in fuel or "browncoal" in fuel or fuel in {"b02"}:
        return "lignite"
    if "hardcoal" in fuel or ("coal" in fuel and "lignite" not in fuel) or fuel in {"b05"}:
        return "hard_coal"
    if "oilshale" in fuel or fuel in {"b07"}:
        return "oil_shale"
    if "gas" in fuel or fuel in {"b04"}:
        return "gas"
    if "oil" in fuel or fuel in {"b06"}:
        return "oil"
    if "geotherm" in fuel or fuel in {"b09"}:
        return "geothermal"
    if "waste" in fuel or "muell" in fuel or "mull" in fuel or fuel in {"b17"}:
        return "waste"
    if "bio" in fuel or fuel in {"b01"}:
        return "bio"
    return "other"


def fuel_group_from_row(raw_fuel: Any, fuel_code: Any | None = None) -> str:
    if fuel_code is not None:
        group = fuel_group_from_fuel_code(fuel_code)
        if group != "other":
            return group
    return fuel_group_from_raw_fuel(raw_fuel)


def is_other_res_pypsa_fuel(raw_fuel: Any) -> bool:
    return fuel_group_from_raw_fuel(raw_fuel) in OTHER_RES_FUEL_GROUPS


def map_thermal_tech(raw_tech: Any, fuel_code: str) -> str:
    fuel = str(fuel_code or "").strip().upper()
    if fuel == "B14":
        return "NUCLEAR"
    tech = str(raw_tech or "").strip().upper()
    if "CCGT" in tech:
        return "CCGT"
    if "OCGT" in tech or "OPEN CYCLE" in tech:
        return "OCGT"
    if "STEAM" in tech:
        return "STEAM"
    if "CCS" in tech:
        return "OTHERS"
    return TECH_MAP.get(tech, "OTHERS")


def is_thermal_row(raw_fuel: Any, raw_set: Any) -> bool:
    fuel = str(raw_fuel or "").strip().upper()
    set_name = str(raw_set or "").strip().upper()
    if "STORE" in set_name:
        return False
    if any(token in fuel for token in NON_THERMAL_FUEL_TOKENS):
        return False
    return fuel_group_from_raw_fuel(raw_fuel) not in OTHER_RES_FUEL_GROUPS


def has_chp_flag(raw_set: Any, raw_tech: Any) -> bool:
    set_name = str(raw_set or "").strip().upper()
    tech = str(raw_tech or "").strip().upper()
    return "CHP" in set_name or "COGEN" in set_name or "CHP" in tech


def inertia_h(fuel_code: Any, tech_norm: Any) -> float:
    fuel = str(fuel_code or "").strip().upper()
    tech = str(tech_norm or "").strip().upper()
    if fuel == "B04":
        return float(GAS_INERTIA_H_BY_TECH.get(tech, THERMAL_INERTIA_H_BY_FUEL["B04"]))
    return float(THERMAL_INERTIA_H_BY_FUEL.get(fuel, 0.0))


def infer_target_year(network_dir: Path | None, explicit: int | None = None) -> int | None:
    if explicit is not None:
        return int(explicit)
    if network_dir is None:
        return None
    import re

    match = re.search(r"target_year_(\d{4})", str(network_dir))
    return int(match.group(1)) if match else None


def default_output_dir(project_root: Path, network_dir: Path, subdir: str) -> Path:
    return project_root / "powerplants" / network_dir.parent.name / network_dir.name / subdir


def default_decommissioned_plants_csv(project_root: Path, network_dir: Path) -> Path:
    return project_root / "grid" / "target_year_2025" / network_dir.name / "plants_with_bus.csv"


def build_manifest(settings: Mapping[str, Any], outputs: Mapping[str, Path], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {str(k): str(v) if isinstance(v, Path) else v for k, v in settings.items()},
        "outputs": {str(k): str(v) for k, v in outputs.items()},
    }
    if extra:
        payload.update(dict(extra))
    return payload


def discover_others_csv(base_dir: Path, *, kind: str, ref_year: int) -> Path:
    preferred_names = (
        f"{kind}_capacity_{ref_year}_tyndp2024.csv",
        f"{kind}_power_{ref_year}_tyndp2024.csv",
    )
    fallback = base_dir / preferred_names[0]
    for name in preferred_names:
        candidate = base_dir / name
        try:
            if candidate.exists():
                return candidate
        except OSError:
            return candidate

    include_tokens = {
        "other_res": ("other", "res"),
        "other_nonres": ("other", "non", "res"),
    }.get(kind)
    if include_tokens is None:
        raise ValueError(f"Unsupported others input kind: {kind}")
    exclude_tokens = ("nonres", "non_res", "non-res", "non") if kind == "other_res" else ()

    try:
        candidates = list(base_dir.rglob("*.csv")) if base_dir.exists() else []
    except OSError:
        return fallback

    scored: list[tuple[int, Path]] = []
    for path in candidates:
        name = path.name.lower()
        compact = name.replace("_", "").replace("-", "")
        if kind == "other_res" and any(token in name for token in exclude_tokens):
            continue
        if not all(token in name or token in compact for token in include_tokens):
            continue
        score = 0
        if str(ref_year) in name:
            score += 10
        if "tyndp2024" in name or "tyndp" in name:
            score += 3
        if "power" in name or "capacity" in name:
            score += 2
        scored.append((score, path))
    if not scored:
        return fallback
    scored.sort(key=lambda item: (-item[0], len(str(item[1])), str(item[1]).lower()))
    return scored[0][1]


def load_bus_country_membership(
    *,
    buses_csv: Path,
    buses_with_clusters_csv: Path | None,
    country_allocation_mode: str = "split_cluster_members",
) -> pd.DataFrame:
    buses = read_csv_auto(buses_csv)
    missing = {"bus_id", "country"} - set(buses.columns)
    if missing:
        raise KeyError(f"Missing columns in {buses_csv}: {sorted(missing)}")

    cluster_weights = pd.DataFrame(columns=["cluster_id", "country", "country_share"])
    if buses_with_clusters_csv is not None and buses_with_clusters_csv.exists():
        raw = read_csv_auto(buses_with_clusters_csv)
        if {"cluster_id", "country"}.issubset(raw.columns):
            raw["cluster_id"] = raw["cluster_id"].astype(str)
            raw["country"] = raw["country"].map(norm_country)
            cluster_weights = raw.groupby(["cluster_id", "country"], as_index=False).size().rename(columns={"size": "member_buses"})
            cluster_weights["cluster_total_buses"] = cluster_weights.groupby("cluster_id")["member_buses"].transform("sum")
            cluster_weights["country_share"] = cluster_weights["member_buses"] / cluster_weights["cluster_total_buses"]

    cluster_map = {
        str(cluster_id): group[["country", "country_share"]].to_dict("records")
        for cluster_id, group in cluster_weights.groupby("cluster_id")
    }

    rows: list[dict[str, Any]] = []
    for row in buses.itertuples(index=False):
        bus_id = str(row.bus_id)
        physical_country = norm_country(getattr(row, "country", ""))
        allocations = [{"country": physical_country, "share": 1.0, "country_source": "bus_country"}]
        if country_allocation_mode == "split_cluster_members" and bus_id in cluster_map:
            allocations = [
                {
                    "country": norm_country(item["country"]),
                    "share": float(item["country_share"]),
                    "country_source": "split_cluster_members",
                }
                for item in cluster_map[bus_id]
            ]
        for alloc in allocations:
            rows.append(
                {
                    "bus_id": bus_id,
                    "country": norm_country(alloc["country"]),
                    "membership_share": float(alloc["share"]),
                    "physical_country": physical_country,
                    "country_source": alloc["country_source"],
                    "lat": safe_float(getattr(row, "lat", np.nan), np.nan),
                    "lon": safe_float(getattr(row, "lon", np.nan), np.nan),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["bus_id", "country"])


def load_country_capacity_targets(
    csv_path: Path | None,
    *,
    countries: list[str],
    ref_year: int,
    scenario: str,
    capacity_col_candidates: tuple[str, ...] = (
        "capacity_mw",
        "capacity_MW",
        "capacity",
        "installed_capacity",
        "installed_capacity_mw",
        "power_mw",
        "value_mw",
        "value",
        "mw",
    ),
    group_col_candidates: tuple[str, ...] = (),
    group_col_output: str = "target_group",
) -> tuple[dict[str, float], pd.DataFrame]:
    if csv_path is None or not csv_path.exists():
        return {}, pd.DataFrame()
    df = read_csv_auto(csv_path)
    country_col = find_column(
        df,
        (
            "country",
            "country_code",
            "country_iso",
            "country_name",
            "area",
            "region",
            "zone",
        ),
    )
    if country_col is None:
        raise KeyError(f"{csv_path.name} missing a country column.")

    year_col = find_column(df, ("year", "target_year", "ref_year", "reference_year"))
    scenario_col = find_column(df, ("scenario", "storyline", "scenario_name"))
    capacity_col = next((col for col in capacity_col_candidates if col in df.columns), None)
    if capacity_col is None:
        capacity_col = find_column(df, capacity_col_candidates)
    if capacity_col is None and str(ref_year) in df.columns:
        capacity_col = str(ref_year)
    if capacity_col is None:
        year_text = str(ref_year)
        for column in df.columns:
            key = normalize_column_key(column)
            if year_text in key and ("capacity" in key or "power" in key or "mw" in key):
                capacity_col = column
                break
    if capacity_col is None:
        raise KeyError(f"{csv_path.name} missing capacity column. Expected one of {list(capacity_col_candidates)} or a {ref_year} MW column.")
    group_col = find_column(df, group_col_candidates) if group_col_candidates else None

    work = filter_to_tyndp_target_year(
        df,
        csv_path,
        ref_year=ref_year,
        year_col=year_col,
        context="country capacity target",
    )
    work["country"] = work[country_col].map(norm_country)
    group_cols: list[str] = []
    if group_col is not None:
        work[group_col_output] = work[group_col].astype(str).str.strip().replace({"": "unspecified"})
        group_cols.append(group_col_output)
    if scenario_col is not None:
        requested = str(scenario).strip()
        work["_scenario"] = work[scenario_col].astype(str).str.strip()
        requested_key = normalize_column_key(requested)
        scenario_key = work["_scenario"].map(normalize_column_key)
        if requested_key and scenario_key.eq(requested_key).any():
            work = work[scenario_key.eq(requested_key)].copy()
    work["target_capacity_mw"] = numeric_column(work[capacity_col]).fillna(0.0)
    work = work[work["country"].isin(countries)].copy()

    passthrough_columns = [
        column
        for column in work.columns
        if column not in {country_col, year_col, scenario_col, capacity_col, group_col, "country", "_year", "_scenario", *group_cols}
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in work.groupby(["country", *group_cols], sort=True):
        if group_cols:
            key_values = keys if isinstance(keys, tuple) else (keys,)
            country = str(key_values[0])
            group_values = key_values[1:]
        else:
            country = str(keys[0] if isinstance(keys, tuple) else keys)
            group_values = ()
        target = float(group["target_capacity_mw"].sum())
        row: dict[str, Any] = {"country": country, "target_capacity_mw": target}
        for column, value in zip(group_cols, group_values):
            row[column] = value
        weights = group["target_capacity_mw"].clip(lower=0.0)
        weight_sum = float(weights.sum())
        for column in passthrough_columns:
            if column == "target_capacity_mw":
                continue
            numeric = numeric_column(group[column])
            if numeric.notna().any():
                if weight_sum > 0.0:
                    row[column] = float((numeric.fillna(0.0) * weights).sum() / weight_sum)
                else:
                    row[column] = float(numeric.mean())
            else:
                values = sorted(set(str(value).strip() for value in group[column].dropna() if str(value).strip()))
                if values:
                    row[column] = ",".join(values)
        rows.append(row)
    grouped = pd.DataFrame(rows)
    if grouped.empty:
        return {}, pd.DataFrame(columns=["country", *group_cols, "target_capacity_mw"])
    if group_cols:
        target_dict = {
            tuple(getattr(row, column) for column in ["country", *group_cols]): float(row.target_capacity_mw)
            for row in grouped.itertuples(index=False)
        }
    else:
        target_dict = dict(zip(grouped["country"], grouped["target_capacity_mw"].astype(float)))
    return target_dict, grouped


def load_load_shares(path: Path) -> pd.DataFrame:
    df = read_csv_auto(path)
    if {"country", "bus", "load_share"}.issubset(df.columns):
        out = df[["country", "bus", "load_share"]].rename(columns={"bus": "bus_id"}).copy()
    elif {"country", "bus_id", "load_share"}.issubset(df.columns):
        out = df[["country", "bus_id", "load_share"]].copy()
    elif {"country", "bus", "share"}.issubset(df.columns):
        out = df[["country", "bus", "share"]].rename(columns={"bus": "bus_id", "share": "load_share"}).copy()
    elif {"country", "bus_id", "share"}.issubset(df.columns):
        out = df[["country", "bus_id", "share"]].rename(columns={"share": "load_share"}).copy()
    elif {"country", "bus", "allocated_load_mw"}.issubset(df.columns):
        work = df[["country", "bus", "allocated_load_mw"]].rename(columns={"bus": "bus_id"}).copy()
        work["allocated_load_mw"] = pd.to_numeric(work["allocated_load_mw"], errors="coerce").fillna(0.0)
        out = work.groupby(["country", "bus_id"], as_index=False)["allocated_load_mw"].mean()
        out["load_share"] = out["allocated_load_mw"] / out.groupby("country")["allocated_load_mw"].transform("sum").replace(0.0, np.nan)
        out = out.drop(columns=["allocated_load_mw"])
    else:
        raise KeyError(f"{path} does not contain a recognized load share schema.")
    out["country"] = out["country"].map(norm_country)
    out["bus_id"] = out["bus_id"].astype(str)
    out["load_share"] = pd.to_numeric(out["load_share"], errors="coerce").fillna(0.0)
    out = out.groupby(["country", "bus_id"], as_index=False)["load_share"].sum()
    total = out.groupby("country")["load_share"].transform("sum")
    out["load_share"] = np.divide(out["load_share"], total, out=np.zeros(len(out), dtype=float), where=total > 0.0)
    return out


def require_positive_load_share_buses(load_shares: pd.DataFrame) -> pd.DataFrame:
    missing = {"country", "bus_id", "load_share"} - set(load_shares.columns)
    if missing:
        raise KeyError(f"load_shares missing columns: {sorted(missing)}")
    allowed = load_shares[["country", "bus_id", "load_share"]].copy()
    allowed["country"] = allowed["country"].map(norm_country)
    allowed["bus_id"] = allowed["bus_id"].astype(str)
    allowed["load_share"] = pd.to_numeric(allowed["load_share"], errors="coerce").fillna(0.0)
    allowed = allowed[allowed["load_share"] > 0.0].copy()
    allowed = allowed.groupby(["country", "bus_id"], as_index=False)["load_share"].sum()
    if allowed.empty:
        raise ValueError("No buses with positive load_share found. Capacity disaggregation requires load-bearing buses.")
    return allowed


def restrict_to_load_buses(frame: pd.DataFrame, load_shares: pd.DataFrame, *, keep_load_share: bool = False) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    missing = {"country", "bus_id"} - set(frame.columns)
    if missing:
        raise KeyError(f"frame missing columns for load-bus filtering: {sorted(missing)}")
    allowed = require_positive_load_share_buses(load_shares)
    out = frame.copy()
    out["country"] = out["country"].map(norm_country)
    out["bus_id"] = out["bus_id"].astype(str)
    merge_cols = ["country", "bus_id", "load_share"] if keep_load_share else ["country", "bus_id"]
    return out.merge(allowed[merge_cols], how="inner", on=["country", "bus_id"])


def decommissioning_fuel_priority(target_group: Any, context: str = "thermal") -> tuple[tuple[str, ...], ...]:
    group = str(target_group or "").strip().lower()
    ctx = str(context or "").strip().lower()
    if ctx == "other_res":
        if group == "waste":
            return (("waste",),)
        if group == "geothermal":
            return (("geothermal",),)
        if group == "bio":
            return (("bio",),)
        return ()
    if group == "nuclear":
        return (("nuclear",),)
    if group == "hard_coal":
        return (("hard_coal",), ("lignite",))
    if group == "lignite":
        return (("lignite",), ("hard_coal",))
    if group == "gas":
        return (("gas",), ("oil",), ("lignite", "hard_coal"))
    if group == "oil":
        return (("oil",), ("gas",), ("lignite", "hard_coal"))
    if group == "oil_shale":
        return (("oil_shale",), ("oil",), ("gas",), ("lignite", "hard_coal"))
    return ((group,),) if group else ()


def load_decommissioned_site_basis(
    *,
    plants_with_bus_csv: Path | None,
    buses_with_clusters_csv: Path | None,
    bus_country_membership: pd.DataFrame,
    load_shares: pd.DataFrame,
    target_year: int,
    lookback_years: int = 10,
    allowed_fuel_groups: set[str] | None = None,
) -> pd.DataFrame:
    columns = [
        "country",
        "bus_id",
        "fuel_group",
        "decommissioned_capacity_mw",
        "decommissioned_date_out",
        "decommissioned_units",
        "source_fuels",
        "source_technologies",
        "source_names",
    ]
    if plants_with_bus_csv is None or not Path(plants_with_bus_csv).exists():
        return pd.DataFrame(columns=columns)
    raw = read_csv_auto(Path(plants_with_bus_csv))
    bus_col = find_column(raw, ("assigned_bus", "bus_id", "bus", "Bus"))
    fuel_col = find_column(raw, ("Fueltype", "fueltype", "fuel_type", "fuel"))
    cap_col = find_column(raw, ("Capacity", "capacity_mw", "installed_capacity_mw", "installed_capacity", "NetCAP", "GrossCAP"))
    date_out_col = find_column(raw, ("DateOut", "date_out", "retirement_year", "decommissioning_year", "Retired"))
    if bus_col is None or fuel_col is None or cap_col is None or date_out_col is None:
        return pd.DataFrame(columns=columns)
    work = raw.copy()
    work["source_bus_id"] = work[bus_col].astype(str).str.strip()
    work["decommissioned_date_out"] = numeric_column(work[date_out_col])
    work["decommissioned_capacity_mw"] = numeric_column(work[cap_col]).fillna(0.0).clip(lower=0.0)
    work["fuel_group"] = work[fuel_col].map(fuel_group_from_raw_fuel)
    if allowed_fuel_groups is not None:
        work = work[work["fuel_group"].isin(set(allowed_fuel_groups))].copy()
    start_year = int(target_year) - int(lookback_years)
    work = work[
        work["source_bus_id"].ne("")
        & work["decommissioned_date_out"].ge(start_year)
        & work["decommissioned_date_out"].lt(int(target_year))
        & work["decommissioned_capacity_mw"].gt(0.0)
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)

    cluster_map: dict[str, str] = {}
    if buses_with_clusters_csv is not None and Path(buses_with_clusters_csv).exists():
        clusters = read_csv_auto(Path(buses_with_clusters_csv))
        if {"bus_id", "cluster_id"}.issubset(clusters.columns):
            cluster_map = dict(zip(clusters["bus_id"].astype(str), clusters["cluster_id"].astype(str)))
    work["bus_id"] = work["source_bus_id"].map(cluster_map).fillna(work["source_bus_id"]).astype(str)

    membership = bus_country_membership[["bus_id", "country", "membership_share"]].drop_duplicates().copy()
    membership["bus_id"] = membership["bus_id"].astype(str)
    membership["country"] = membership["country"].map(norm_country)
    membership["membership_share"] = pd.to_numeric(membership["membership_share"], errors="coerce").fillna(1.0)
    work = work.merge(membership, how="inner", on="bus_id", validate="many_to_many")
    if work.empty:
        return pd.DataFrame(columns=columns)
    work["decommissioned_capacity_mw"] = work["decommissioned_capacity_mw"] * work["membership_share"].clip(lower=0.0)
    work["country"] = work["country"].map(norm_country)
    name_col = find_column(work, ("Name", "name", "plant_name", "unit_name"))
    tech_col = find_column(work, ("Technology", "technology", "plant_tech", "Tech-ID"))
    work["_source_name"] = work[name_col].astype(str).str.strip() if name_col is not None else ""
    work["_source_technology"] = work[tech_col].astype(str).str.strip() if tech_col is not None else ""
    work["_source_fuel"] = work[fuel_col].astype(str).str.strip()

    grouped = (
        work.groupby(["country", "bus_id", "fuel_group"], as_index=False)
        .agg(
            decommissioned_capacity_mw=("decommissioned_capacity_mw", "sum"),
            decommissioned_date_out=("decommissioned_date_out", "max"),
            decommissioned_units=("source_bus_id", "count"),
            source_fuels=("_source_fuel", lambda values: ",".join(sorted(set(str(v) for v in values if str(v).strip())))),
            source_technologies=("_source_technology", lambda values: ",".join(sorted(set(str(v) for v in values if str(v).strip())))),
            source_names=("_source_name", lambda values: ",".join(sorted(set(str(v) for v in values if str(v).strip()))[:10])),
        )
    )
    grouped = restrict_to_load_buses(grouped, load_shares)
    return grouped.sort_values(
        ["country", "decommissioned_date_out", "decommissioned_capacity_mw", "bus_id"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def _site_key(country: Any, bus_id: Any) -> tuple[str, str]:
    return (norm_country(country), str(bus_id))


def _decom_key(country: Any, bus_id: Any, fuel_group: Any) -> tuple[str, str, str]:
    return (norm_country(country), str(bus_id), str(fuel_group or "").strip().lower())


def allocate_units_to_decommissioned_sites(
    *,
    country: str,
    target_fuel_group: str,
    unit_sizes_mw: list[float],
    decommissioned_sites: pd.DataFrame,
    usage_mw: dict[tuple[str, str, str], float],
    used_site_fuel_groups: dict[tuple[str, str], set[str]],
    context: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    target_group = str(target_fuel_group or "").strip().lower()
    country = norm_country(country)
    positive_unit_sizes = [float(v) for v in unit_sizes_mw if float(v) > 0.0]
    if not positive_unit_sizes:
        return pd.DataFrame(), diagnostics
    unassigned = [
        {"country": country, "target_fuel_group": target_group, "unassigned_unit_mw": unit_size}
        for unit_size in positive_unit_sizes
    ]
    if decommissioned_sites.empty:
        return pd.DataFrame(), unassigned
    country_sites = decommissioned_sites[decommissioned_sites["country"].eq(country)].copy()
    if country_sites.empty:
        return pd.DataFrame(), unassigned
    country_sites = country_sites.sort_values(
        ["decommissioned_date_out", "decommissioned_capacity_mw", "bus_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    coal_groups = {"hard_coal", "lignite"}
    priority = decommissioning_fuel_priority(target_group, context=context)
    for unit_size in sorted(positive_unit_sizes, reverse=True):
        assigned = False
        for stage_idx, fuel_groups in enumerate(priority, start=1):
            candidates = country_sites[country_sites["fuel_group"].isin(set(fuel_groups))].copy()
            for site in candidates.itertuples(index=False):
                key = _decom_key(country, site.bus_id, site.fuel_group)
                site_key = _site_key(country, site.bus_id)
                used = float(usage_mw.get(key, 0.0))
                cap = float(site.decommissioned_capacity_mw)
                if used >= cap:
                    continue
                prior_site_groups = used_site_fuel_groups.get(site_key, set())
                if target_group in {"gas", "oil", "oil_shale"} and str(site.fuel_group) in coal_groups and prior_site_groups.intersection(coal_groups):
                    continue
                usage_mw[key] = used + unit_size
                used_site_fuel_groups.setdefault(site_key, set()).add(target_group)
                rows.append(
                    {
                        "country": country,
                        "bus_id": str(site.bus_id),
                        "assigned_units": 1,
                        "assigned_cap_mw": unit_size,
                        "weight": np.nan,
                        "bus_rank": 0,
                        "fallback_stage": f"decommissioned_{str(site.fuel_group)}",
                        "decommissioned_fuel_group": str(site.fuel_group),
                        "decommissioned_capacity_mw": cap,
                        "decommissioned_used_before_mw": used,
                        "decommissioned_date_out": float(site.decommissioned_date_out),
                        "decommissioning_priority_stage": int(stage_idx),
                    }
                )
                assigned = True
                break
            if assigned:
                break
        if not assigned:
            diagnostics.append({"country": country, "target_fuel_group": target_group, "unassigned_unit_mw": unit_size})
    if not rows:
        return pd.DataFrame(), diagnostics
    out = (
        pd.DataFrame(rows)
        .groupby(["country", "bus_id", "fallback_stage", "decommissioned_fuel_group"], as_index=False)
        .agg(
            assigned_units=("assigned_units", "sum"),
            assigned_cap_mw=("assigned_cap_mw", "sum"),
            weight=("weight", "first"),
            bus_rank=("bus_rank", "first"),
            decommissioned_capacity_mw=("decommissioned_capacity_mw", "first"),
            decommissioned_used_before_mw=("decommissioned_used_before_mw", "min"),
            decommissioned_date_out=("decommissioned_date_out", "max"),
            decommissioning_priority_stage=("decommissioning_priority_stage", "min"),
        )
    )
    return out, diagnostics


def allocate_capacity_to_decommissioned_sites(
    *,
    country: str,
    target_fuel_group: str,
    target_capacity_mw: float,
    decommissioned_sites: pd.DataFrame,
    usage_mw: dict[tuple[str, str, str], float],
    used_site_fuel_groups: dict[tuple[str, str], set[str]],
    context: str,
) -> tuple[pd.DataFrame, float]:
    rows: list[dict[str, Any]] = []
    remaining = float(max(target_capacity_mw, 0.0))
    if remaining <= 0.0 or decommissioned_sites.empty:
        return pd.DataFrame(), remaining
    country = norm_country(country)
    target_group = str(target_fuel_group or "").strip().lower()
    sites = decommissioned_sites[decommissioned_sites["country"].eq(country)].copy()
    if sites.empty:
        return pd.DataFrame(), remaining
    sites = sites.sort_values(["decommissioned_date_out", "decommissioned_capacity_mw", "bus_id"], ascending=[False, False, True])
    coal_groups = {"hard_coal", "lignite"}
    for stage_idx, fuel_groups in enumerate(decommissioning_fuel_priority(target_group, context=context), start=1):
        candidates = sites[sites["fuel_group"].isin(set(fuel_groups))].copy()
        for site in candidates.itertuples(index=False):
            if remaining <= 1e-9:
                break
            key = _decom_key(country, site.bus_id, site.fuel_group)
            site_key = _site_key(country, site.bus_id)
            used = float(usage_mw.get(key, 0.0))
            cap = float(site.decommissioned_capacity_mw)
            if used >= cap:
                continue
            prior_site_groups = used_site_fuel_groups.get(site_key, set())
            if target_group in {"gas", "oil", "oil_shale"} and str(site.fuel_group) in coal_groups and prior_site_groups.intersection(coal_groups):
                continue
            amount = min(remaining, cap - used)
            if amount <= 1e-9:
                continue
            usage_mw[key] = used + amount
            used_site_fuel_groups.setdefault(site_key, set()).add(target_group)
            rows.append(
                {
                    "country": country,
                    "bus_id": str(site.bus_id),
                    "capacity_mw": amount,
                    "allocation_mode": f"decommissioned_{str(site.fuel_group)}",
                    "decommissioned_fuel_group": str(site.fuel_group),
                    "decommissioned_capacity_mw": cap,
                    "decommissioned_used_before_mw": used,
                    "decommissioned_date_out": float(site.decommissioned_date_out),
                    "decommissioning_priority_stage": int(stage_idx),
                }
            )
            remaining -= amount
        if remaining <= 1e-9:
            break
    return pd.DataFrame(rows), float(max(0.0, remaining))


def allocate_capacity_by_capped_basis(
    basis: pd.DataFrame,
    *,
    target_capacity_mw: float,
    capacity_col: str,
    allocation_mode: str,
) -> tuple[pd.DataFrame, float]:
    remaining = float(max(target_capacity_mw, 0.0))
    if remaining <= 0.0 or basis.empty:
        return pd.DataFrame(), remaining
    work = basis.copy()
    work[capacity_col] = pd.to_numeric(work[capacity_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    work = work[work[capacity_col] > 0.0].copy()
    if work.empty:
        return pd.DataFrame(), remaining
    total = float(work[capacity_col].sum())
    if total <= 0.0:
        return pd.DataFrame(), remaining
    if total >= remaining:
        work["capacity_mw"] = work[capacity_col] / total * remaining
        remaining = 0.0
    else:
        work["capacity_mw"] = work[capacity_col]
        remaining -= total
    work["allocation_mode"] = allocation_mode
    return work, float(max(0.0, remaining))


def source_country_mapping_from_load_shares(path: Path | None) -> dict[str, str]:
    if path is None or not Path(path).exists():
        return {}
    df = read_csv_auto(Path(path))
    if "country" not in df.columns or "source_countries" not in df.columns:
        return {}
    mapping: dict[str, str] = {}
    for row in df[["country", "source_countries"]].drop_duplicates().itertuples(index=False):
        model_country = norm_country(row.country)
        for raw_source in str(row.source_countries or "").split(","):
            source = norm_country(raw_source)
            if source:
                mapping[source] = source if source in PROTECTED_COUNTRY_CODES else model_country
    return mapping


def aggregate_targets_with_country_map(
    targets_df: pd.DataFrame,
    *,
    country_map: Mapping[str, str],
    target_countries: list[str],
    group_cols: tuple[str, ...] = (),
) -> tuple[dict[str, float], pd.DataFrame]:
    if targets_df.empty:
        return {}, targets_df
    out = targets_df.copy()
    out["country_source"] = out["country"].map(norm_country)
    out["country"] = out["country_source"].map(
        lambda country: country if country in PROTECTED_COUNTRY_CODES else country_map.get(country, country)
    )
    out = out[out["country"].isin(set(target_countries))].copy()
    rows: list[dict[str, Any]] = []
    passthrough_columns = [
        column
        for column in out.columns
        if column not in {"country", "country_source", "target_capacity_mw", *group_cols}
    ]
    group_keys = ["country", *group_cols]
    for keys, group in out.groupby(group_keys, sort=True):
        if group_cols:
            key_values = keys if isinstance(keys, tuple) else (keys,)
            country = str(key_values[0])
            group_values = key_values[1:]
        else:
            country = str(keys[0] if isinstance(keys, tuple) else keys)
            group_values = ()
        target = float(pd.to_numeric(group["target_capacity_mw"], errors="coerce").fillna(0.0).sum())
        row: dict[str, Any] = {"country": country, "target_capacity_mw": target}
        for column, value in zip(group_cols, group_values):
            row[column] = value
        weights = pd.to_numeric(group["target_capacity_mw"], errors="coerce").fillna(0.0).clip(lower=0.0)
        weight_sum = float(weights.sum())
        for column in passthrough_columns:
            numeric = numeric_column(group[column])
            if numeric.notna().any():
                if weight_sum > 0.0:
                    row[column] = float((numeric.fillna(0.0) * weights).sum() / weight_sum)
                else:
                    row[column] = float(numeric.mean())
            else:
                values = sorted(set(str(value).strip() for value in group[column].dropna() if str(value).strip()))
                if values:
                    row[column] = ",".join(values)
        rows.append(row)
    grouped = pd.DataFrame(rows)
    if grouped.empty:
        return {}, pd.DataFrame(columns=["country", *group_cols, "target_capacity_mw", "country_source"])
    sources = (
        out.groupby(group_keys)["country_source"]
        .apply(lambda values: ",".join(sorted(set(str(value) for value in values))))
        .reset_index()
    )
    grouped = grouped.merge(sources, how="left", on=group_keys)
    if group_cols:
        target_dict = {
            tuple(getattr(row, column) for column in group_keys): float(row.target_capacity_mw)
            for row in grouped.itertuples(index=False)
        }
    else:
        target_dict = dict(zip(grouped["country"], grouped["target_capacity_mw"].astype(float)))
    return target_dict, grouped


def load_network_plant_rows(
    *,
    plants_csv: Path,
    buses_csv: Path,
    bus_country_membership: pd.DataFrame,
) -> pd.DataFrame:
    plants = read_csv_auto(plants_csv)
    buses = read_csv_auto(buses_csv)
    bus_col = "bus_id" if "bus_id" in plants.columns else ("assigned_bus" if "assigned_bus" in plants.columns else None)
    missing_required = {"Fueltype", "Technology", "Set", "Capacity"} - set(plants.columns)
    missing_plants = set(missing_required)
    if bus_col is None:
        missing_plants.add("bus_id/assigned_bus")
    missing_buses = {"bus_id", "country"} - set(buses.columns)
    if missing_plants:
        raise KeyError(f"Missing columns in {plants_csv}: {sorted(missing_plants)}")
    if missing_buses:
        raise KeyError(f"Missing columns in {buses_csv}: {sorted(missing_buses)}")

    if bus_col != "bus_id":
        plants = plants.rename(columns={bus_col: "bus_id"})
    if "n_plants" not in plants.columns:
        plants = plants.copy()
        plants["n_plants"] = 1.0

    buses = buses[["bus_id", "country"]].copy()
    buses["bus_id"] = buses["bus_id"].astype(str)
    buses["physical_country"] = buses["country"].map(norm_country)
    plants = plants.rename(columns={"country": "plant_country_raw", "country_label": "plant_country_label_raw"})
    merged = plants.merge(buses[["bus_id", "physical_country"]], how="left", on="bus_id", validate="many_to_one")
    merged = merged.merge(
        bus_country_membership[["bus_id", "country", "membership_share", "country_source"]],
        how="left",
        on="bus_id",
        validate="many_to_many",
    )
    if "country" not in merged.columns and "country_y" in merged.columns:
        merged["country"] = merged["country_y"]
    if "country" not in merged.columns and "country_x" in merged.columns:
        merged["country"] = merged["country_x"]
    merged["country"] = merged["country"].map(norm_country)
    merged["membership_share"] = pd.to_numeric(merged["membership_share"], errors="coerce").fillna(1.0)

    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        cap = safe_float(row.Capacity)
        units = safe_float(row.n_plants)
        if cap <= 0.0:
            continue
        share = max(0.0, safe_float(getattr(row, "membership_share", 1.0), 1.0))
        rows.append(
            {
                "bus_id": str(row.bus_id),
                "country": norm_country(getattr(row, "country", "")),
                "physical_country": norm_country(getattr(row, "physical_country", "")),
                "country_source": getattr(row, "country_source", "bus_country"),
                "fueltype": str(row.Fueltype or "").strip(),
                "technology": str(row.Technology or "").strip(),
                "set_name": str(getattr(row, "Set", "") or "").strip(),
                "capacity_mw": cap * share,
                "n_plants": units * share,
            }
        )
    return pd.DataFrame(rows)


def build_bus_shares(
    *,
    basis_df: pd.DataFrame,
    bus_country_membership: pd.DataFrame,
    target_by_group: dict[tuple[str, str], float],
    group_col: str,
    fallback_mode: str = "uniform",
    fallback_rows_by_country: Mapping[str, list[tuple[str, float]]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = bus_country_membership[["bus_id", "country"]].drop_duplicates().copy()
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    for (country, group_key), target in sorted(target_by_group.items()):
        target = float(target)
        if target <= 0.0:
            continue
        group_basis = basis_df[(basis_df["country"] == country) & (basis_df[group_col] == group_key)].copy()
        by_bus = (
            group_basis.groupby(["country", "bus_id"], as_index=False)["capacity_mw"]
            .sum()
            .rename(columns={"capacity_mw": "basis_capacity_mw"})
        )
        basis_total = float(by_bus["basis_capacity_mw"].sum()) if not by_bus.empty else 0.0
        if basis_total > 0.0:
            by_bus["share"] = by_bus["basis_capacity_mw"] / basis_total
            fallback = "basis_capacity"
        else:
            fallback_rows = (fallback_rows_by_country or {}).get(country, [])
            if fallback_rows:
                by_bus = pd.DataFrame(
                    {
                        "country": country,
                        "bus_id": [str(bus_id) for bus_id, _share in fallback_rows],
                        "basis_capacity_mw": 0.0,
                        "share": [float(share) for _bus_id, share in fallback_rows],
                    }
                )
                fallback = fallback_mode
            else:
                country_buses = membership[membership["country"] == country][["bus_id"]].drop_duplicates().copy()
                if country_buses.empty:
                    continue
                country_buses["basis_capacity_mw"] = 0.0
                country_buses["share"] = 1.0 / float(len(country_buses))
                by_bus = country_buses.assign(country=country)
                fallback = fallback_mode
        by_bus["scaled_capacity_mw"] = by_bus["share"] * target
        by_bus[group_col] = group_key
        by_bus["fallback_mode"] = fallback
        rows.extend(by_bus.to_dict("records"))
        diag_rows.append(
            {
                "country": country,
                group_col: group_key,
                "target_capacity_mw": target,
                "basis_capacity_mw": basis_total,
                "fallback_mode": fallback,
                "n_buses": int(len(by_bus)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(diag_rows)


def load_res_bus_capacity(path: Path) -> pd.DataFrame:
    df = read_csv_auto(path)
    missing = {"bus_id", "country", "technology", "scenario_capacity_mw"} - set(df.columns)
    if missing:
        raise KeyError(f"{path} missing required columns: {sorted(missing)}")
    out = df.copy()
    out["country"] = out["country"].map(norm_country)
    out["bus_id"] = out["bus_id"].astype(str)
    out["technology"] = out["technology"].astype(str).str.strip().str.lower()
    for col in ("current_capacity_mw", "added_capacity_mw", "scenario_capacity_mw"):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def load_res_potential(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["bus_id", "country", "technology", "p_nom_max_mw"])
    df = read_csv_auto(path)
    if {"bus_id", "country", "technology", "p_nom_max_mw"}.issubset(df.columns):
        out = df[["bus_id", "country", "technology", "p_nom_max_mw"]].copy()
    elif {"bus_id", "country", "technology", "installable_capacity_mw"}.issubset(df.columns):
        out = df[["bus_id", "country", "technology", "installable_capacity_mw"]].rename(
            columns={"installable_capacity_mw": "p_nom_max_mw"}
        )
    else:
        raise KeyError(
            f"{path} must contain bus_id,country,technology,p_nom_max_mw "
            "or bus_id,country,technology,installable_capacity_mw."
        )
    out["country"] = out["country"].map(norm_country)
    out["bus_id"] = out["bus_id"].astype(str)
    out["technology"] = out["technology"].astype(str).str.strip().str.lower()
    out["p_nom_max_mw"] = pd.to_numeric(out["p_nom_max_mw"], errors="coerce").fillna(0.0)
    return out


def build_res_headroom_from_bus_capacity(res_bus_capacity: pd.DataFrame, potential: pd.DataFrame | None = None) -> pd.DataFrame:
    allowed = {"pv", "onwind"}
    cap = res_bus_capacity[res_bus_capacity["technology"].isin(allowed)].copy()
    base = (
        cap.groupby(["country", "bus_id", "technology"], as_index=False)
        .agg(
            current_capacity_mw=("current_capacity_mw", "sum"),
            added_capacity_mw=("added_capacity_mw", "sum"),
            scenario_capacity_mw=("scenario_capacity_mw", "sum"),
        )
    )
    if potential is not None and not potential.empty:
        pot = potential[potential["technology"].isin(allowed)].copy()
        pot = pot.groupby(["country", "bus_id", "technology"], as_index=False)["p_nom_max_mw"].sum()
        base = base.merge(pot, how="outer", on=["country", "bus_id", "technology"]).fillna(0.0)
    else:
        # The RES capacity preprocessing already applies p_nom_max constraints; in
        # that common case use target scenario capacity as the conservative cap.
        base["p_nom_max_mw"] = base["scenario_capacity_mw"]
    base["used_capacity_mw"] = base["scenario_capacity_mw"]
    base["free_capacity_mw"] = (base["p_nom_max_mw"] - base["used_capacity_mw"]).clip(lower=0.0)
    return base


def summarize_by_country(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["country", value_col])
    return frame.groupby("country", as_index=False)[value_col].sum()
