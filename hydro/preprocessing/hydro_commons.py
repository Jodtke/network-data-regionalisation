from __future__ import annotations

"""Common readers and normalisation rules for hydro preprocessing.

Hydro source data combine plant inventories, storage estimates, and country or
zone labels from different providers. The helpers in this file keep the
normalisation of country codes, plant types, reference years, and lightweight
configuration parsing consistent between inflow preparation, constraint
preparation, and the final bus disaggregation.
"""

import csv
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

EPSILON = 1e-9
LEGACY_CAPACITY_PATTERN = "hydro_capacities_country_*_tyndp2024.csv"
POWER_CAPACITY_PATTERN = "hydro_power_capacity_*_tyndp2024.csv"
CONSTRAINT_PATTERN_BY_REGION_LEVEL = {
    "country": "hydro_uniform_constraints_weekly_country_*_tyndp2024.csv",
    "market_node": "hydro_uniform_constraints_weekly_market_node_*_tyndp2024.csv",
}
INFLOW_PATTERN_BY_REGION_LEVEL = {
    "country": "hydro_inflow_profiles_weekly_country_*_tyndp2024.csv",
    "market_node": "hydro_inflow_profiles_weekly_market_node_*_tyndp2024.csv",
}
REGION_LEVELS = ("country", "market_node")
CONFIG_SECTIONS = frozenset({"common", "constraints", "inflows"})
TYNDP_TARGET_YEARS = (2030, 2040, 2050)

CAPACITY_FIELDS = (
    "country", "ref_year", "plant_type", "technology", "turb_mw", "pump_mw", "storage_gwh"
)
CONSTRAINT_FIELDS = (
    "country", "ref_year", "week", "plant_type", "technology",
    "min_turb_en_gwh_day", "max_turb_en_gwh_day", "min_pump_en_gwh_day", "max_pump_en_gwh_day",
    "min_turb_mw", "max_turb_mw", "min_pump_mw", "max_pump_mw",
    "min_res_hist", "max_res_hist", "min_res_tech", "max_res_tech"
)
INFLOW_FIELDS = (
    "country", "ref_year", "weather_year", "week", "plant_type", "technology", "inflow_gwh_week"
)
POWER_FIELDS = ("min_turb_mw", "max_turb_mw", "min_pump_mw", "max_pump_mw")
ENERGY_FIELDS = (
    "min_turb_en_gwh_day", "max_turb_en_gwh_day", "min_pump_en_gwh_day", "max_pump_en_gwh_day"
)
RESERVOIR_FIELDS = ("min_res_hist", "max_res_hist", "min_res_tech", "max_res_tech")
INTEGER_PATTERN = re.compile(r"[+-]?\d+")
FLOAT_PATTERN = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+(?:\.\d*)?[eE][+-]?\d+)")
POWER_CAPACITY_PLANT_TYPE_MAP = {
    "run of river": ("ror", "open_loop", "ror"),
    "pondage": ("ror", "open_loop", "pondage"),
    "reservoir": ("wr", "open_loop", "reservoir"),
    "pump storage open loop": ("phs", "open_loop", "phs_open_loop"),
    "pump storage closed loop": ("phs", "closed_loop", "phs_closed_loop"),
}
COUNTRY_NAME_TO_ISO_A2 = {
    "algeria": "DZ",
    "egypt": "EG",
    "israel": "IL",
    "libya": "LY",
    "moldova": "MD",
    "morocco": "MA",
    "palestine": "PS",
    "tunisia": "TN",
    "turkey": "TR",
    "ukraine": "UA",
}

@dataclass(frozen=True)
class DayInfo:
    week: int
    day_of_week: int
    day_of_year: int
    date: date


def _strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double and (index == 0 or line[index - 1].isspace()):
            return line[:index].rstrip()
    return line.rstrip()


def _parse_yaml_scalar(text: str, path: Path, lineno: int) -> object:
    if text in {"null", "Null", "NULL", "~"}:
        return None
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if len(text) >= 2 and text[0] == text[-1] == "'":
        return text[1:-1].replace("''", "'")
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return text[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if INTEGER_PATTERN.fullmatch(text):
        return int(text)
    if FLOAT_PATTERN.fullmatch(text):
        return float(text)
    if text.startswith(("'", '"')) or text.endswith(("'", '"')):
        raise ValueError(f"Unterminated quoted string in '{path}' at line {lineno}.")
    return text


def load_simple_yaml(path: Path) -> dict[str, object]:
    root: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object]]] = [(0, root)]
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = _strip_yaml_comment(raw_line.rstrip("\n\r"))
            if not line or not line.strip():
                continue
            stripped = line.lstrip(" ")
            if "\t" in line[:len(line) - len(stripped)]:
                raise ValueError(f"Tabs are not supported in YAML indentation ('{path}', line {lineno}).")
            if stripped in {"---", "..."}:
                continue
            indent = len(line) - len(stripped)
            if indent % 2 != 0:
                raise ValueError(f"Indentation must use multiples of two spaces ('{path}', line {lineno}).")
            while stack and stack[-1][0] > indent:
                stack.pop()
            if not stack or stack[-1][0] != indent:
                raise ValueError(f"Unexpected indentation in '{path}' at line {lineno}.")
            if ":" not in stripped:
                raise ValueError(f"Expected 'key: value' in '{path}' at line {lineno}.")
            key, raw_value = stripped.split(":", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Blank keys are not supported in '{path}' at line {lineno}.")
            current = stack[-1][1]
            if key in current:
                raise ValueError(f"Duplicate key '{key}' in '{path}' at line {lineno}.")
            value_text = raw_value.strip()
            if not value_text:
                child: dict[str, object] = {}
                current[key] = child
                stack.append((indent + 2, child))
                continue
            current[key] = _parse_yaml_scalar(value_text, path, lineno)
    return root


def load_workflow_config(path: Path, workflow_name: str) -> dict[str, object]:
    config = load_simple_yaml(path)
    unknown_sections = sorted(
        key for key, value in config.items()
        if isinstance(value, dict) and key not in CONFIG_SECTIONS
    )
    if unknown_sections:
        raise ValueError(
            f"Unsupported YAML sections in '{path}': {', '.join(unknown_sections)}. "
            f"Supported sections are: {', '.join(sorted(CONFIG_SECTIONS))}."
        )
    merged: dict[str, object] = {}
    for key, value in config.items():
        if key in CONFIG_SECTIONS:
            continue
        if isinstance(value, dict):
            raise ValueError(f"Top-level key '{key}' in '{path}' must not contain a nested mapping.")
        merged[key] = value
    for section_name in ("common", workflow_name):
        section = config.get(section_name)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ValueError(f"Section '{section_name}' in '{path}' must be a mapping.")
        merged.update(section)
    return merged


def validate_config_keys(config: Mapping[str, object], allowed_keys: set[str], workflow_name: str, config_path: Path) -> None:
    unknown_keys = sorted(set(config) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unsupported config keys for workflow '{workflow_name}' in '{config_path}': {', '.join(unknown_keys)}."
        )


def coerce_config_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"'{field_name}' must be an integer, not a boolean.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and INTEGER_PATTERN.fullmatch(value.strip()):
        return int(value.strip())
    raise ValueError(f"'{field_name}' must be an integer.")


def validate_tyndp_ref_year(ref_year: int) -> int:
    year = int(ref_year)
    if year not in TYNDP_TARGET_YEARS:
        raise ValueError(
            f"Unsupported TYNDP ref_year {year}. Expected one of: "
            + ", ".join(str(value) for value in TYNDP_TARGET_YEARS)
        )
    return year


def coerce_config_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "on", "1"}:
            return True
        if lowered in {"false", "no", "off", "0"}:
            return False
    raise ValueError(f"'{field_name}' must be a boolean.")


def coerce_config_choice(value: object, field_name: str, allowed_values: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise ValueError(f"'{field_name}' must be one of: {', '.join(allowed_values)}.")
    normalized = value.strip().lower()
    if normalized not in allowed_values:
        raise ValueError(f"'{field_name}' must be one of: {', '.join(allowed_values)}.")
    return normalized


def coerce_path_value(value: object, field_name: str) -> Path:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"'{field_name}' must not be blank.")
        path = Path(text)
    else:
        raise ValueError(f"'{field_name}' must be a filesystem path.")
    return path


def coerce_config_path(value: object, field_name: str, relative_to: Path) -> Path:
    path = coerce_path_value(value, field_name)
    return path if path.is_absolute() else (relative_to / path).resolve()


def coerce_optional_config_path(value: object | None, field_name: str, relative_to: Path) -> Path | None:
    return None if value is None else coerce_config_path(value, field_name, relative_to)


def parse_int(value: str | None) -> int:
    text = (value or "").strip()
    if not text:
        raise ValueError("Expected integer value, got blank.")
    return int(float(text))


def parse_optional_float(value: str | None) -> float | None:
    text = (value or "").strip()
    return None if not text else float(text)


def normalize_text(field: str, value: str | None) -> str:
    text = (value or "").strip()
    if field == "country":
        return COUNTRY_NAME_TO_ISO_A2.get(text.lower(), text.upper())
    if field in {"market_node", "zone"}:
        return text.upper()
    if field in {"plant_type", "technology"}:
        return text.lower()
    return text


def validate_region_level(region_level: str) -> str:
    normalized = region_level.strip().lower()
    if normalized not in REGION_LEVELS:
        raise ValueError(f"Unsupported region_level '{region_level}'. Expected one of: {', '.join(REGION_LEVELS)}.")
    return normalized


def region_fields(region_level: str) -> tuple[str, ...]:
    region_level = validate_region_level(region_level)
    if region_level == "country":
        return ("country",)
    return ("country", "market_node")


def region_values(row: dict[str, object], region_level: str) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in region_fields(region_level))


def region_dict(values: tuple[str, ...], region_level: str) -> dict[str, object]:
    return dict(zip(region_fields(region_level), values, strict=True))


def technology_key(row: dict[str, object], region_level: str) -> tuple[str, ...]:
    return region_values(row, region_level) + (str(row["plant_type"]), str(row["technology"]))


def copy_region_fields(row: dict[str, object], region_level: str) -> dict[str, object]:
    return {field: row[field] for field in region_fields(region_level)}


def floats_close(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def rows_equal(left: dict[str, object], right: dict[str, object], fields: Iterable[str]) -> bool:
    for field in fields:
        lv = left.get(field)
        rv = right.get(field)
        if isinstance(lv, float) or isinstance(rv, float) or lv is None or rv is None:
            if not floats_close(lv, rv):  # type: ignore[arg-type]
                return False
        elif lv != rv:
            return False
    return True


def load_rows(base_dir: Path, pattern: str, ref_year: int, fields: tuple[str, ...], int_fields: tuple[str, ...], float_fields: tuple[str, ...], key_fields: tuple[str, ...]) -> list[dict[str, object]]:
    files = sorted(base_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {base_dir}.")
    rows_by_key: dict[tuple[object, ...], dict[str, object]] = {}
    source_by_key: dict[tuple[object, ...], Path] = {}
    for path in files:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                row_ref_year = parse_int(raw.get("ref_year"))
                if row_ref_year != ref_year:
                    continue
                row: dict[str, object] = {}
                for field in fields:
                    if field in float_fields:
                        row[field] = parse_optional_float(raw.get(field))
                    elif field in int_fields:
                        row[field] = parse_int(raw.get(field))
                    else:
                        row[field] = normalize_text(field, raw.get(field))
                key = tuple(row[field] for field in key_fields)
                existing = rows_by_key.get(key)
                if existing is None:
                    rows_by_key[key] = row
                    source_by_key[key] = path
                elif not rows_equal(existing, row, fields):
                    raise ValueError(f"Conflicting duplicate row for key {key} in '{source_by_key[key].name}' and '{path.name}'.")
    if not rows_by_key:
        raise ValueError(f"No rows found for ref_year={ref_year} in files matching '{pattern}'.")
    return list(rows_by_key.values())


def resolve_power_capacity_plant_type(raw_plant_type: str) -> tuple[str, str, str]:
    normalized = raw_plant_type.lower()
    for pattern, mapped in POWER_CAPACITY_PLANT_TYPE_MAP.items():
        if pattern in normalized:
            return mapped
    raise ValueError(f"Unsupported hydro power capacity plant_type '{raw_plant_type}'.")


def normalize_power_capacity_country(raw_country: str | None, market_node: str) -> str:
    raw_text = (raw_country or "").strip()
    country = normalize_text("country", raw_text)
    if len(country) == 2 and country.isalpha():
        return country
    mapped = COUNTRY_NAME_TO_ISO_A2.get(raw_text.lower())
    if mapped is not None:
        return mapped
    zone_prefix = market_node[:2].upper()
    if len(zone_prefix) == 2 and zone_prefix.isalpha():
        return zone_prefix
    raise ValueError(f"Cannot infer ISO-A2 country code from country='{raw_text}' and zone='{market_node}'.")


def load_power_capacity_rows(base_dir: Path, ref_year: int, region_level: str) -> list[dict[str, object]]:
    region_level = validate_region_level(region_level)
    files = sorted(base_dir.glob(POWER_CAPACITY_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files matching '{POWER_CAPACITY_PATTERN}' found in {base_dir}.")
    rows_by_key: dict[tuple[str, ...], dict[str, object]] = {}
    storage_mwh_by_key_and_component: dict[tuple[str, ...], dict[tuple[str, str], float]] = {}
    for path in files:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            for raw in reader:
                row_ref_year = parse_int(raw.get("reference_year"))
                if row_ref_year != ref_year:
                    continue
                market_node = normalize_text("market_node", raw.get("zone"))
                country = normalize_power_capacity_country(raw.get("country"), market_node)
                plant_type, technology, storage_component = resolve_power_capacity_plant_type(raw.get("plant_type") or "")
                region_key = (country,) if region_level == "country" else (country, market_node)
                key = region_key + (plant_type, technology)
                row = rows_by_key.setdefault(key, {
                    **region_dict(region_key, region_level),
                    "ref_year": ref_year,
                    "plant_type": plant_type,
                    "technology": technology,
                    "turb_mw": 0.0,
                    "pump_mw": 0.0,
                    "storage_gwh": 0.0,
                })
                power_mw = parse_optional_float(raw.get("power_mw")) or 0.0
                power_type = (raw.get("power_type") or "").strip().lower()
                if power_type == "turbine":
                    row["turb_mw"] = float(row["turb_mw"]) + power_mw
                elif power_type == "pumping":
                    row["pump_mw"] = float(row["pump_mw"]) + abs(power_mw)
                else:
                    raise ValueError(f"Unsupported hydro power capacity power_type '{raw.get('power_type')}' in '{path.name}'.")
                storage_mwh = parse_optional_float(raw.get("capacity_mwh")) or 0.0
                component_storage = storage_mwh_by_key_and_component.setdefault(key, {})
                storage_component_key = (market_node, storage_component)
                component_storage[storage_component_key] = max(component_storage.get(storage_component_key, 0.0), storage_mwh)
    if not rows_by_key:
        raise ValueError(f"No rows found for reference_year={ref_year} in files matching '{POWER_CAPACITY_PATTERN}'.")
    for key, component_storage in storage_mwh_by_key_and_component.items():
        rows_by_key[key]["storage_gwh"] = sum(component_storage.values()) / 1000.0
    return list(rows_by_key.values())


def load_capacity_rows(base_dir: Path, ref_year: int, region_level: str = "country") -> list[dict[str, object]]:
    region_level = validate_region_level(region_level)
    if list(base_dir.glob(POWER_CAPACITY_PATTERN)):
        return load_power_capacity_rows(base_dir, ref_year, region_level)
    if region_level != "country":
        raise FileNotFoundError(
            f"Market-node capacity loading requires files matching '{POWER_CAPACITY_PATTERN}' in {base_dir}."
        )
    return load_rows(base_dir, LEGACY_CAPACITY_PATTERN, ref_year, CAPACITY_FIELDS, ("ref_year",), ("turb_mw", "pump_mw", "storage_gwh"), ("country", "ref_year", "plant_type", "technology"))


def load_constraint_rows(base_dir: Path, ref_year: int, region_level: str = "country") -> list[dict[str, object]]:
    region_level = validate_region_level(region_level)
    row_fields = region_fields(region_level) + tuple(field for field in CONSTRAINT_FIELDS if field != "country")
    return load_rows(base_dir, CONSTRAINT_PATTERN_BY_REGION_LEVEL[region_level], ref_year, row_fields, ("ref_year", "week"), (
        "min_turb_en_gwh_day", "max_turb_en_gwh_day", "min_pump_en_gwh_day", "max_pump_en_gwh_day",
        "min_turb_mw", "max_turb_mw", "min_pump_mw", "max_pump_mw",
        "min_res_hist", "max_res_hist", "min_res_tech", "max_res_tech",
    ), region_fields(region_level) + ("ref_year", "week", "plant_type", "technology"))


def load_inflow_rows(base_dir: Path, ref_year: int, region_level: str = "country") -> list[dict[str, object]]:
    region_level = validate_region_level(region_level)
    row_fields = region_fields(region_level) + tuple(field for field in INFLOW_FIELDS if field != "country")
    return load_rows(base_dir, INFLOW_PATTERN_BY_REGION_LEVEL[region_level], ref_year, row_fields, ("ref_year", "weather_year", "week"), ("inflow_gwh_week",), region_fields(region_level) + ("ref_year", "weather_year", "week", "plant_type", "technology"))


def row_has_capacity(row: dict[str, object]) -> bool:
    return any(isinstance(row.get(field), float) and abs(float(row.get(field) or 0.0)) > EPSILON for field in ("turb_mw", "pump_mw", "storage_gwh"))


def build_capacity_index(capacity_rows: Iterable[dict[str, object]], region_level: str = "country") -> dict[tuple[str, ...], dict[str, object]]:
    return {technology_key(row, region_level): row for row in capacity_rows}


def active_capacity_index(capacity_rows: Iterable[dict[str, object]], region_level: str = "country") -> dict[tuple[str, ...], dict[str, object]]:
    return {key: row for key, row in build_capacity_index(capacity_rows, region_level).items() if row_has_capacity(row)}


def resolve_phs_target(region_key: tuple[str, ...], technology: str, has_wr: bool, has_ror: bool) -> tuple[str, ...]:
    if technology == "closed_loop":
        if not has_wr and has_ror:
            return region_key + ("ror", "open_loop")
        if not has_wr and not has_ror:
            return region_key + ("wr", "closed_loop")
    return region_key + ("wr", "open_loop")


def resolve_phs_component_map(capacity_rows: list[dict[str, object]], presence_rows: list[dict[str, object]] | None = None, region_level: str = "country") -> dict[tuple[str, ...], list[tuple[str, ...]]]:
    region_level = validate_region_level(region_level)
    if presence_rows is None:
        presence_rows = capacity_rows
    by_region: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in capacity_rows:
        if not row_has_capacity(row):
            continue
        by_region.setdefault(region_values(row, region_level), []).append(row)
    presence_by_region: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in presence_rows:
        if not row_has_capacity(row):
            continue
        presence_by_region.setdefault(region_values(row, region_level), []).append(row)
    component_map: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
    for region_key, rows in by_region.items():
        presence_region_rows = presence_by_region.get(region_key, [])
        has_wr = any(row["plant_type"] == "wr" and row["technology"] == "open_loop" for row in presence_region_rows)
        has_ror = any(row["plant_type"] == "ror" and row["technology"] == "open_loop" for row in presence_region_rows)
        for row in rows:
            source_key = technology_key(row, region_level)
            if row["plant_type"] == "phs":
                target_key = resolve_phs_target(region_key, str(row["technology"]), has_wr, has_ror)
            else:
                target_key = source_key
            component_map.setdefault(target_key, []).append(source_key)
    return component_map


def resolve_phs_capacities(capacity_rows: list[dict[str, object]], presence_rows: list[dict[str, object]] | None = None, region_level: str = "country") -> list[dict[str, object]]:
    region_level = validate_region_level(region_level)
    capacity_index = build_capacity_index(capacity_rows, region_level)
    component_map = resolve_phs_component_map(capacity_rows, presence_rows, region_level)
    merged_rows: list[dict[str, object]] = []
    for target_key, source_keys in sorted(component_map.items()):
        region_key = target_key[:-2]
        plant_type, technology = target_key[-2:]
        sample_row = capacity_index[source_keys[0]]
        combined = {
            **region_dict(region_key, region_level),
            "ref_year": sample_row["ref_year"],
            "plant_type": plant_type,
            "technology": technology,
            "turb_mw": 0.0,
            "pump_mw": 0.0,
            "storage_gwh": 0.0,
        }
        receives_phs = False
        for source_key in source_keys:
            row = capacity_index[source_key]
            combined["turb_mw"] = float(combined["turb_mw"]) + float(row.get("turb_mw") or 0.0)
            combined["pump_mw"] = float(combined["pump_mw"]) + float(row.get("pump_mw") or 0.0)
            combined["storage_gwh"] = float(combined["storage_gwh"]) + float(row.get("storage_gwh") or 0.0)
            receives_phs = receives_phs or source_key[-2] == "phs"
        if receives_phs:
            combined["pump_mw"] = 0.0
        merged_rows.append(combined)
    return merged_rows


def positive_magnitude(value: float | None) -> float | None:
    return None if value is None else abs(value)


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None:
        return None
    if denominator is None or abs(denominator) <= EPSILON:
        return 0.0 if abs(numerator) <= EPSILON else None
    return numerator / denominator


def sum_optional(values: Iterable[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    return None if not filtered else sum(filtered)


def weighted_average(pairs: Iterable[tuple[float | None, float]]) -> float | None:
    total = 0.0
    total_weight = 0.0
    for value, weight in pairs:
        if value is None or abs(weight) <= EPSILON:
            continue
        total += value * weight
        total_weight += weight
    return None if total_weight <= EPSILON else total / total_weight


def days_in_year(year: int) -> int:
    return 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365


def expand_week_to_days(ref_year: int, week: int) -> list[DayInfo]:
    year_days = days_in_year(ref_year)
    start_day = (week - 1) * 7 + 1
    if start_day > year_days:
        return []
    end_day = min(start_day + 6, year_days)
    first_date = date(ref_year, 1, 1)
    return [DayInfo(week=week, day_of_week=(day_of_year - start_day + 1), day_of_year=day_of_year, date=first_date + timedelta(days=day_of_year - 1)) for day_of_year in range(start_day, end_day + 1)]


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
