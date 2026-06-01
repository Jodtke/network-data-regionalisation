from __future__ import annotations

"""Prepare weekly hydro operating constraints for TYNDP target years.

The script converts national or market-node hydro capacity assumptions into a
uniform constraint table for run-of-river, reservoir, and pumped-storage units.
Storage-energy overrides and plant-type mappings are applied before the bus
disaggregation step, so that turbine limits, pump limits, reservoir bounds, and
energy constraints are scaled from one consistent country-level basis.
"""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from hydro_commons import (
    ENERGY_FIELDS,
    EPSILON,
    POWER_FIELDS,
    RESERVOIR_FIELDS,
    active_capacity_index,
    coerce_config_bool,
    coerce_config_choice,
    coerce_config_int,
    coerce_config_path,
    coerce_optional_config_path,
    coerce_path_value,
    copy_region_fields,
    expand_week_to_days,
    load_capacity_rows,
    load_constraint_rows,
    load_workflow_config,
    positive_magnitude,
    region_dict,
    region_values,
    resolve_phs_capacities,
    resolve_phs_component_map,
    safe_ratio,
    technology_key,
    validate_config_keys,
    validate_tyndp_ref_year,
    weighted_average,
    write_csv,
)

DEFAULT_QUARANTA_OVERRIDE_PATH = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\DATA\hydro\quaranta2024\hydro_capacity_2025_quaranta.csv")
ALLOWED_OVERRIDE_SOURCE = "Quaranta et al. 2024"
COUNTRY_MAP = {"GB": "UK"}
PLANT_TYPE_MAP = {"Hydro Pumped Storage": "phs", "Hydro Water Reservoir": "wr"}

DAILY_OUTPUT_FIELDS = [
    "country", "ref_year", "plant_type", "technology", "temporal_resolution", "week", "day_of_week", "day_of_year", "date",
    "installed_turb_mw", "installed_pump_mw", "installed_storage_mwh",
    "min_turb_mw", "max_turb_mw", "min_turb_pu", "max_turb_pu",
    "min_pump_mw", "max_pump_mw", "min_pump_pu", "max_pump_pu",
    "min_turb_en_mwh_day", "max_turb_en_mwh_day", "min_pump_en_mwh_day", "max_pump_en_mwh_day",
    "min_res_hist_pu", "max_res_hist_pu", "min_res_tech_pu", "max_res_tech_pu",
    "imputed",
]
WEEKLY_OUTPUT_FIELDS = [
    "country", "ref_year", "plant_type", "technology", "temporal_resolution", "week", "period_start_date", "period_end_date", "days_in_period",
    "installed_turb_mw", "installed_pump_mw", "installed_storage_mwh",
    "min_turb_mw", "max_turb_mw", "min_turb_pu", "max_turb_pu",
    "min_pump_mw", "max_pump_mw", "min_pump_pu", "max_pump_pu",
    "min_turb_en_mwh_day", "max_turb_en_mwh_day", "min_pump_en_mwh_day", "max_pump_en_mwh_day",
    "min_turb_en_mwh_period", "max_turb_en_mwh_period", "min_pump_en_mwh_period", "max_pump_en_mwh_period",
    "min_res_hist_pu", "max_res_hist_pu", "min_res_tech_pu", "max_res_tech_pu",
    "imputed",
]
CONSTRAINT_ROUNDING = {
    "installed_turb_mw": 0,
    "installed_pump_mw": 0,
    "installed_storage_mwh": 0,
    "min_turb_mw": 2,
    "max_turb_mw": 2,
    "min_turb_pu": 2,
    "max_turb_pu": 2,
    "min_pump_mw": 2,
    "max_pump_mw": 2,
    "min_pump_pu": 2,
    "max_pump_pu": 2,
    "min_turb_en_mwh_day": 0,
    "max_turb_en_mwh_day": 0,
    "min_pump_en_mwh_day": 0,
    "max_pump_en_mwh_day": 0,
    "min_turb_en_mwh_period": 0,
    "max_turb_en_mwh_period": 0,
    "min_pump_en_mwh_period": 0,
    "max_pump_en_mwh_period": 0,
    "min_res_hist_pu": 2,
    "max_res_hist_pu": 2,
    "min_res_tech_pu": 2,
    "max_res_tech_pu": 2,
}
CONFIG_KEYS = {
    "year", "base_dir", "output", "region_level", "temporal_resolution",
    "resolve_phs_to_wr", "impute_missing_constraint_fields",
    "use_default_storage_overrides", "storage_override_csv",
}


@dataclass(frozen=True)
class ConstraintOptions:
    year: int
    base_dir: Path
    output: Path | None
    region_level: str
    temporal_resolution: str
    resolve_phs_to_wr: bool
    impute_missing_constraint_fields: bool
    use_default_storage_overrides: bool
    storage_override_csv: Path | None


def fields_for_region_level(fields: list[str], region_level: str) -> list[str]:
    if region_level == "country":
        return fields
    return ["country", "market_node", *fields[1:]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bereitet TYNDP2024-Hydro-Constraints auf.")
    parser.add_argument("--config", type=Path, default=None, help="Optionale YAML-Konfiguration mit Werten aus 'common' und 'constraints'.")
    parser.add_argument("--year", type=int, default=argparse.SUPPRESS, help="Zieljahr / ref_year.")
    parser.add_argument("--base-dir", type=Path, default=argparse.SUPPRESS, help="Verzeichnis mit den TYNDP2024-Country-CSV-Dateien.")
    parser.add_argument("--output", type=Path, default=argparse.SUPPRESS, help="Optionaler Ausgabepfad.")
    parser.add_argument("--region-level", choices=("country", "market_node"), default=argparse.SUPPRESS, help="Raeumliche Granularitaet: country oder market_node.")
    parser.add_argument("--temporal-resolution", choices=("daily", "weekly"), default=argparse.SUPPRESS, help="daily oder weekly.")
    parser.add_argument("--resolve-phs-to-wr", dest="resolve_phs_to_wr", action="store_true", default=argparse.SUPPRESS, help="Aggregiert PHS in WR und setzt Pumpen auf 0.")
    parser.add_argument("--no-resolve-phs-to-wr", dest="resolve_phs_to_wr", action="store_false", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--impute-missing-constraint-fields", dest="impute_missing_constraint_fields", action="store_true", default=argparse.SUPPRESS, help="Fuellt fehlende Felder vorhandener Constraint-Zeilen ueber Technologie/Wochen-Mittelwerte.")
    parser.add_argument("--no-impute-missing-constraint-fields", dest="impute_missing_constraint_fields", action="store_false", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--use-default-storage-overrides", dest="use_default_storage_overrides", action="store_true", default=argparse.SUPPRESS, help="Verwendet die bekannte Quaranta-Datei und ignoriert alle Nicht-Quaranta-Quellen.")
    parser.add_argument("--no-default-storage-overrides", dest="use_default_storage_overrides", action="store_false", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--storage-override-csv", type=Path, default=argparse.SUPPRESS, help="Alternative Override-Datei im Quaranta-Format.")
    return parser


def load_options(args: argparse.Namespace) -> ConstraintOptions:
    script_dir = Path(__file__).resolve().parent
    config_path = args.config.resolve() if args.config is not None else None
    config_values: dict[str, object] = {}
    if config_path is not None:
        config_values = load_workflow_config(config_path, "constraints")
        validate_config_keys(config_values, CONFIG_KEYS, "constraints", config_path)
    cli_values = vars(args).copy()
    cli_values.pop("config", None)
    config_dir = config_path.parent if config_path is not None else script_dir
    year_value = cli_values.get("year", config_values.get("year"))
    if year_value is None:
        raise ValueError("Missing required value 'year'. Provide --year or set year: in the YAML config.")
    year = validate_tyndp_ref_year(coerce_config_int(year_value, "year"))
    if "base_dir" in cli_values:
        base_dir = cli_values["base_dir"].resolve()
    elif "base_dir" in config_values:
        base_dir = coerce_config_path(config_values["base_dir"], "base_dir", config_dir)
    else:
        base_dir = script_dir
    output_value = cli_values.get("output", config_values.get("output"))
    output = None if output_value is None else coerce_path_value(output_value, "output")
    region_level = coerce_config_choice(
        cli_values.get("region_level", config_values.get("region_level", "country")),
        "region_level",
        ("country", "market_node"),
    )
    temporal_resolution = coerce_config_choice(
        cli_values.get("temporal_resolution", config_values.get("temporal_resolution", "daily")),
        "temporal_resolution",
        ("daily", "weekly"),
    )
    resolve_phs_to_wr = coerce_config_bool(
        cli_values.get("resolve_phs_to_wr", config_values.get("resolve_phs_to_wr", False)),
        "resolve_phs_to_wr",
    )
    impute_missing_constraint_fields = coerce_config_bool(
        cli_values.get("impute_missing_constraint_fields", config_values.get("impute_missing_constraint_fields", False)),
        "impute_missing_constraint_fields",
    )
    use_default_storage_overrides = coerce_config_bool(
        cli_values.get("use_default_storage_overrides", config_values.get("use_default_storage_overrides", False)),
        "use_default_storage_overrides",
    )
    if "storage_override_csv" in cli_values:
        storage_override_csv = cli_values["storage_override_csv"].resolve()
    elif "storage_override_csv" in config_values:
        storage_override_csv = coerce_optional_config_path(config_values["storage_override_csv"], "storage_override_csv", config_dir)
    else:
        storage_override_csv = None
    return ConstraintOptions(
        year=year,
        base_dir=base_dir,
        output=output,
        region_level=region_level,
        temporal_resolution=temporal_resolution,
        resolve_phs_to_wr=resolve_phs_to_wr,
        impute_missing_constraint_fields=impute_missing_constraint_fields,
        use_default_storage_overrides=use_default_storage_overrides,
        storage_override_csv=storage_override_csv,
    )


def parse_args(argv: list[str] | None = None) -> ConstraintOptions:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return load_options(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


def normalize_override_country(country: str) -> str:
    return COUNTRY_MAP.get(country.strip().upper(), country.strip().upper())


def load_storage_overrides(path: Path) -> dict[tuple[str, str], float]:
    overrides: dict[tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            if (row.get("source") or "").strip() != ALLOWED_OVERRIDE_SOURCE:
                continue
            plant_type = PLANT_TYPE_MAP.get((row.get("plant_type") or "").strip())
            if plant_type is None:
                continue
            key = (normalize_override_country(row.get("country") or ""), plant_type)
            storage_gwh = float((row.get("capacity_gwh") or "0").strip() or 0.0)
            existing = overrides.get(key)
            if existing is not None and not math.isclose(existing, storage_gwh, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"Conflicting storage override for {key} in '{path}'.")
            overrides[key] = storage_gwh
    return overrides


def apply_storage_overrides(capacity_rows: list[dict[str, object]], overrides: dict[tuple[str, str], float]) -> tuple[list[dict[str, object]], dict[str, int]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in capacity_rows:
        grouped.setdefault((str(row["country"]), str(row["plant_type"])), []).append(row)
    override_keys = set(overrides)
    matched_keys = override_keys & set(grouped)
    unmatched_override_keys = override_keys - set(grouped)
    result_rows: list[dict[str, object]] = []
    changed_keys = 0
    applied_keys = 0
    for key, rows in grouped.items():
        override_total = overrides.get(key)
        if override_total is None:
            result_rows.extend(dict(row) for row in rows)
            continue
        applied_keys += 1
        current_values = [float(row.get("storage_gwh") or 0.0) for row in rows]
        current_total = sum(current_values)
        row_copies = [dict(row) for row in rows]
        if current_total > 0:
            for row_copy, current_value in zip(row_copies, current_values):
                row_copy["storage_gwh"] = override_total * (current_value / current_total)
        else:
            target_index = next((
                idx for idx, row_copy in enumerate(row_copies)
                if str(row_copy.get("technology")) == "open_loop" and (
                    abs(float(row_copy.get("turb_mw") or 0.0)) > 1e-9 or abs(float(row_copy.get("pump_mw") or 0.0)) > 1e-9
                )
            ), None)
            if target_index is None:
                target_index = next((
                    idx for idx, row_copy in enumerate(row_copies)
                    if abs(float(row_copy.get("turb_mw") or 0.0)) > 1e-9 or abs(float(row_copy.get("pump_mw") or 0.0)) > 1e-9
                ), None)
            if target_index is None:
                result_rows.extend(row_copies)
                continue
            for idx, row_copy in enumerate(row_copies):
                row_copy["storage_gwh"] = override_total if idx == target_index else 0.0
        new_total = sum(float(row.get("storage_gwh") or 0.0) for row in row_copies)
        if not math.isclose(new_total, current_total, rel_tol=1e-9, abs_tol=1e-9):
            changed_keys += 1
        result_rows.extend(row_copies)
    return result_rows, {
        "override_keys_total": len(override_keys), "matched_keys": len(matched_keys), "unmatched_override_keys": len(unmatched_override_keys), "applied_keys": applied_keys, "changed_keys": changed_keys,
    }


def resolve_storage_override_path(options: ConstraintOptions) -> Path | None:
    if options.storage_override_csv is not None:
        return options.storage_override_csv.resolve()
    if options.use_default_storage_overrides:
        return DEFAULT_QUARANTA_OVERRIDE_PATH
    return None


def format_rounded(value: object, decimals: int) -> str | None:
    if value is None or value == "":
        return None
    numeric = float(value)
    if abs(numeric) < 5e-13:
        numeric = 0.0
    if decimals == 0:
        return str(int(round(numeric)))
    return f"{numeric:.{decimals}f}"


def apply_constraint_rounding(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rounded_rows: list[dict[str, object]] = []
    for row in rows:
        rounded = dict(row)
        for field, decimals in CONSTRAINT_ROUNDING.items():
            if field in rounded:
                rounded[field] = format_rounded(rounded[field], decimals)
        rounded_rows.append(rounded)
    return rounded_rows


def default_constraint_value(field: str, capacity_row: dict[str, object]) -> float:
    installed_turb_mw = float(capacity_row.get("turb_mw") or 0.0)
    installed_pump_mw = abs(float(capacity_row.get("pump_mw") or 0.0))
    if field in {"min_turb_mw", "min_pump_mw", "min_turb_en_gwh_day", "min_pump_en_gwh_day", "min_res_hist", "min_res_tech"}:
        return 0.0
    if field == "max_turb_mw":
        return installed_turb_mw
    if field == "max_pump_mw":
        return installed_pump_mw
    if field == "max_turb_en_gwh_day":
        return installed_turb_mw * 24.0 / 1000.0
    if field == "max_pump_en_gwh_day":
        return installed_pump_mw * 24.0 / 1000.0
    if field in {"max_res_hist", "max_res_tech"}:
        return 1.0
    raise KeyError(f"Unsupported constraint field '{field}'.")


def constraint_value_with_default(field: str, constraint_row: dict[str, object], capacity_row: dict[str, object]) -> float:
    value = constraint_row.get(field)
    return default_constraint_value(field, capacity_row) if value is None else float(value)


def fill_missing_constraint_defaults(row: dict[str, object]) -> dict[str, object]:
    filled = dict(row)
    installed_turb_mw = float(filled.get("installed_turb_mw") or 0.0)
    installed_pump_mw = float(filled.get("installed_pump_mw") or 0.0)
    days_in_period = int(filled.get("days_in_period") or 0)
    default_values: dict[str, float] = {
        "min_turb_mw": 0.0,
        "max_turb_mw": installed_turb_mw,
        "min_turb_pu": 0.0,
        "max_turb_pu": 1.0 if installed_turb_mw > 0.0 else 0.0,
        "min_pump_mw": 0.0,
        "max_pump_mw": installed_pump_mw,
        "min_pump_pu": 0.0,
        "max_pump_pu": 1.0 if installed_pump_mw > 0.0 else 0.0,
        "min_turb_en_mwh_day": 0.0,
        "max_turb_en_mwh_day": installed_turb_mw * 24.0,
        "min_pump_en_mwh_day": 0.0,
        "max_pump_en_mwh_day": installed_pump_mw * 24.0,
        "min_turb_en_mwh_period": 0.0,
        "max_turb_en_mwh_period": installed_turb_mw * 24.0 * days_in_period,
        "min_pump_en_mwh_period": 0.0,
        "max_pump_en_mwh_period": installed_pump_mw * 24.0 * days_in_period,
        "min_res_hist_pu": 0.0,
        "max_res_hist_pu": 1.0,
        "min_res_tech_pu": 0.0,
        "max_res_tech_pu": 1.0,
    }
    for field, default_value in default_values.items():
        if field in filled and filled.get(field) is None:
            filled[field] = default_value
    return filled


def constraint_denominator(field: str, capacity_row: dict[str, object]) -> float:
    installed_turb_mw = float(capacity_row.get("turb_mw") or 0.0)
    installed_pump_mw = abs(float(capacity_row.get("pump_mw") or 0.0))
    if field in {"min_turb_mw", "max_turb_mw"}:
        return installed_turb_mw
    if field in {"min_pump_mw", "max_pump_mw"}:
        return installed_pump_mw
    if field in {"min_turb_en_gwh_day", "max_turb_en_gwh_day"}:
        return installed_turb_mw * 24.0 / 1000.0
    if field in {"min_pump_en_gwh_day", "max_pump_en_gwh_day"}:
        return installed_pump_mw * 24.0 / 1000.0
    raise KeyError(f"Unsupported constraint field '{field}'.")


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def bounded_constraint_value(field: str, value: object, capacity_row: dict[str, object]) -> tuple[float, bool]:
    raw_value = float(value)
    if field in {"min_pump_mw", "max_pump_mw", "min_pump_en_gwh_day", "max_pump_en_gwh_day"}:
        raw_value = abs(raw_value)
    if field in POWER_FIELDS + ENERGY_FIELDS:
        upper = max(constraint_denominator(field, capacity_row), 0.0)
        bounded_value = clamp(raw_value, 0.0, upper)
    elif field in RESERVOIR_FIELDS:
        bounded_value = clamp(raw_value, 0.0, 1.0)
    else:
        raise KeyError(f"Unsupported constraint field '{field}'.")
    return bounded_value, not math.isclose(raw_value, bounded_value, rel_tol=1e-9, abs_tol=1e-9)


def cap_daily_energy_by_power(energy_mwh_day: float | None, max_power_mw: float | None) -> tuple[float | None, bool]:
    if energy_mwh_day is None or max_power_mw is None:
        return energy_mwh_day, False
    upper = max(float(max_power_mw), 0.0) * 24.0
    bounded_value = clamp(float(energy_mwh_day), 0.0, upper)
    return bounded_value, not math.isclose(float(energy_mwh_day), bounded_value, rel_tol=1e-9, abs_tol=1e-9)


def mean_by_key(values: dict[tuple[object, ...], list[float]]) -> dict[tuple[object, ...], float]:
    return {key: sum(items) / len(items) for key, items in values.items() if items}


def build_constraint_imputation_profiles(constraint_rows: list[dict[str, object]], component_capacities: dict[tuple[str, ...], dict[str, object]], region_level: str) -> tuple[dict[tuple[object, ...], float], dict[tuple[object, ...], float]]:
    availability_values: dict[tuple[object, ...], list[float]] = {}
    reservoir_values: dict[tuple[object, ...], list[float]] = {}
    for row in constraint_rows:
        capacity_row = component_capacities.get(technology_key(row, region_level))
        if capacity_row is None:
            continue
        plant_type = str(row["plant_type"])
        technology = str(row["technology"])
        week = int(row["week"])
        for field in POWER_FIELDS + ENERGY_FIELDS:
            raw_value = row.get(field)
            if raw_value is None:
                continue
            denominator = constraint_denominator(field, capacity_row)
            if denominator > EPSILON:
                value, _ = bounded_constraint_value(field, raw_value, capacity_row)
                availability_values.setdefault((plant_type, technology, week, field), []).append(value / denominator)
        for field in RESERVOIR_FIELDS:
            raw_value = row.get(field)
            if raw_value is None:
                continue
            if float(capacity_row.get("storage_gwh") or 0.0) <= EPSILON:
                continue
            value, _ = bounded_constraint_value(field, raw_value, capacity_row)
            reservoir_values.setdefault((plant_type, technology, week, field), []).append(value)
    return mean_by_key(availability_values), mean_by_key(reservoir_values)


def impute_constraint_row(region_key: tuple[str, ...], region_level: str, ref_year: int, week: int, plant_type: str, technology: str, capacity_row: dict[str, object], availability_profile: dict[tuple[object, ...], float], reservoir_profile: dict[tuple[object, ...], float]) -> dict[str, object]:
    imputed: dict[str, object] = {**region_dict(region_key, region_level), "ref_year": ref_year, "week": week, "plant_type": plant_type, "technology": technology, "imputed": "other"}
    for field in POWER_FIELDS + ENERGY_FIELDS:
        denominator = constraint_denominator(field, capacity_row)
        mean_availability = availability_profile.get((plant_type, technology, week, field))
        imputed[field] = default_constraint_value(field, capacity_row) if mean_availability is None else mean_availability * denominator
    for field in RESERVOIR_FIELDS:
        mean_reservoir_level = reservoir_profile.get((plant_type, technology, week, field))
        imputed[field] = default_constraint_value(field, capacity_row) if mean_reservoir_level is None else mean_reservoir_level
    return imputed


def constraint_value_with_optional_imputation(field: str, constraint_row: dict[str, object], capacity_row: dict[str, object], availability_profile: dict[tuple[object, ...], float], reservoir_profile: dict[tuple[object, ...], float], impute_missing_constraint_fields: bool) -> tuple[float, bool]:
    value = constraint_row.get(field)
    if value is not None:
        bounded_value, was_bounded = bounded_constraint_value(field, value, capacity_row)
        return bounded_value, was_bounded
    if not impute_missing_constraint_fields:
        return default_constraint_value(field, capacity_row), False
    plant_type = str(constraint_row["plant_type"])
    technology = str(constraint_row["technology"])
    week = int(constraint_row["week"])
    if field in POWER_FIELDS + ENERGY_FIELDS:
        denominator = constraint_denominator(field, capacity_row)
        if denominator <= EPSILON:
            return default_constraint_value(field, capacity_row), False
        mean_availability = availability_profile.get((plant_type, technology, week, field))
        if mean_availability is not None:
            return mean_availability * denominator, True
    elif field in RESERVOIR_FIELDS:
        if float(capacity_row.get("storage_gwh") or 0.0) <= EPSILON:
            return default_constraint_value(field, capacity_row), False
        mean_reservoir_level = reservoir_profile.get((plant_type, technology, week, field))
        if mean_reservoir_level is not None:
            return mean_reservoir_level, True
    return default_constraint_value(field, capacity_row), True


def aggregate_constraint_row(region_key: tuple[str, ...], region_level: str, ref_year: int, week: int, plant_type: str, technology: str, component_capacities: dict[tuple[str, ...], dict[str, object]], constraint_map: dict[tuple[str, ...], dict[str, object]], resolved_component_map: dict[tuple[str, ...], list[tuple[str, ...]]], availability_profile: dict[tuple[object, ...], float], reservoir_profile: dict[tuple[object, ...], float], impute_missing_constraint_fields: bool) -> dict[str, object] | None:
    target_key = region_key + (plant_type, technology)
    source_keys = resolved_component_map.get(target_key, [target_key])
    components = [(source_key, constraint_map[source_key + (week,)]) for source_key in source_keys if source_key + (week,) in constraint_map]
    if not components:
        return None
    aggregated: dict[str, object] = {**region_dict(region_key, region_level), "ref_year": ref_year, "week": week, "plant_type": plant_type, "technology": technology}
    partly_imputed = False
    for field in POWER_FIELDS + ENERGY_FIELDS:
        field_values = []
        for key, row in components:
            value, was_imputed = constraint_value_with_optional_imputation(field, row, component_capacities.get(key, {}), availability_profile, reservoir_profile, impute_missing_constraint_fields)
            field_values.append(value)
            partly_imputed = partly_imputed or was_imputed
        aggregated[field] = sum(field_values)
    receives_phs = any(source_key[-2] == "phs" for source_key in source_keys)
    if plant_type != "phs" and receives_phs:
        aggregated["min_pump_mw"] = 0.0
        aggregated["max_pump_mw"] = 0.0
        aggregated["min_pump_en_gwh_day"] = 0.0
        aggregated["max_pump_en_gwh_day"] = 0.0
    for field in RESERVOIR_FIELDS:
        pairs = []
        for key, row in components:
            value, was_imputed = constraint_value_with_optional_imputation(field, row, component_capacities.get(key, {}), availability_profile, reservoir_profile, impute_missing_constraint_fields)
            pairs.append((value, float(component_capacities.get(key, {}).get("storage_gwh") or 0.0)))
            partly_imputed = partly_imputed or was_imputed
        aggregated[field] = weighted_average(pairs)
    aggregated["imputed"] = "yes" if partly_imputed else "no"
    return aggregated


def build_weekly_base_rows(base_dir: Path, ref_year: int, region_level: str, resolve_phs_to_wr: bool, impute_missing_constraint_fields: bool, storage_override_path: Path | None) -> tuple[list[dict[str, object]], dict[str, int] | None]:
    original_capacity_rows = load_capacity_rows(base_dir, ref_year, region_level)
    routing_capacity_rows = [dict(row) for row in original_capacity_rows]
    override_summary = None
    if storage_override_path is not None:
        overrides = load_storage_overrides(storage_override_path)
        original_capacity_rows, override_summary = apply_storage_overrides(original_capacity_rows, overrides)
    component_capacities = {technology_key(row, region_level): row for row in original_capacity_rows}
    resolved_component_map = resolve_phs_component_map(original_capacity_rows, routing_capacity_rows, region_level) if resolve_phs_to_wr else {
        key: [key] for key in component_capacities
    }
    output_capacity_rows = resolve_phs_capacities(original_capacity_rows, routing_capacity_rows, region_level) if resolve_phs_to_wr else original_capacity_rows
    active_capacities = active_capacity_index(output_capacity_rows, region_level)
    constraint_rows = load_constraint_rows(base_dir, ref_year, region_level)
    constraint_map = {technology_key(row, region_level) + (int(row["week"]),): row for row in constraint_rows}
    availability_profile, reservoir_profile = build_constraint_imputation_profiles(constraint_rows, component_capacities, region_level)
    weeks = sorted({int(row["week"]) for row in constraint_rows})
    output_rows: list[dict[str, object]] = []
    for capacity_key, capacity_row in sorted(active_capacities.items(), key=lambda item: item[0]):
        region_key = capacity_key[:-2]
        plant_type, technology = capacity_key[-2:]
        installed_turb_mw = float(capacity_row.get("turb_mw") or 0.0)
        installed_pump_mw = float(capacity_row.get("pump_mw") or 0.0)
        installed_storage_mwh = float(capacity_row.get("storage_gwh") or 0.0) * 1000.0
        for week in weeks:
            aggregated = aggregate_constraint_row(region_key, region_level, ref_year, week, plant_type, technology, component_capacities, constraint_map, resolved_component_map, availability_profile, reservoir_profile, impute_missing_constraint_fields)
            if aggregated is None:
                aggregated = impute_constraint_row(region_key, region_level, ref_year, week, plant_type, technology, capacity_row, availability_profile, reservoir_profile)
            days = expand_week_to_days(ref_year, week)
            if not days:
                continue
            min_turb_mw = aggregated.get("min_turb_mw")
            max_turb_mw = aggregated.get("max_turb_mw")
            min_pump_mw = positive_magnitude(aggregated.get("min_pump_mw"))  # type: ignore[arg-type]
            max_pump_mw = positive_magnitude(aggregated.get("max_pump_mw"))  # type: ignore[arg-type]
            min_turb_en_mwh_day = None if aggregated.get("min_turb_en_gwh_day") is None else float(aggregated["min_turb_en_gwh_day"]) * 1000.0
            max_turb_en_mwh_day = None if aggregated.get("max_turb_en_gwh_day") is None else float(aggregated["max_turb_en_gwh_day"]) * 1000.0
            min_pump_en_mwh_day = None if aggregated.get("min_pump_en_gwh_day") is None else abs(float(aggregated["min_pump_en_gwh_day"])) * 1000.0
            max_pump_en_mwh_day = None if aggregated.get("max_pump_en_gwh_day") is None else abs(float(aggregated["max_pump_en_gwh_day"])) * 1000.0
            max_turb_en_mwh_day, turb_energy_was_bounded = cap_daily_energy_by_power(max_turb_en_mwh_day, max_turb_mw)
            max_pump_en_mwh_day, pump_energy_was_bounded = cap_daily_energy_by_power(max_pump_en_mwh_day, max_pump_mw)
            if min_turb_en_mwh_day is not None and max_turb_en_mwh_day is not None:
                min_turb_en_mwh_day = min(min_turb_en_mwh_day, max_turb_en_mwh_day)
            if min_pump_en_mwh_day is not None and max_pump_en_mwh_day is not None:
                min_pump_en_mwh_day = min(min_pump_en_mwh_day, max_pump_en_mwh_day)
            imputed = str(aggregated["imputed"])
            if imputed == "no" and (turb_energy_was_bounded or pump_energy_was_bounded):
                imputed = "yes"
            weekly_row = fill_missing_constraint_defaults({
                **region_dict(region_key, region_level), "ref_year": ref_year, "plant_type": plant_type, "technology": technology, "week": week,
                "period_start_date": days[0].date.isoformat(), "period_end_date": days[-1].date.isoformat(), "days_in_period": len(days),
                "installed_turb_mw": installed_turb_mw, "installed_pump_mw": installed_pump_mw, "installed_storage_mwh": installed_storage_mwh,
                "min_turb_mw": min_turb_mw, "max_turb_mw": max_turb_mw, "min_turb_pu": safe_ratio(min_turb_mw, installed_turb_mw), "max_turb_pu": safe_ratio(max_turb_mw, installed_turb_mw),
                "min_pump_mw": min_pump_mw, "max_pump_mw": max_pump_mw, "min_pump_pu": safe_ratio(min_pump_mw, installed_pump_mw), "max_pump_pu": safe_ratio(max_pump_mw, installed_pump_mw),
                "min_turb_en_mwh_day": min_turb_en_mwh_day, "max_turb_en_mwh_day": max_turb_en_mwh_day,
                "min_pump_en_mwh_day": min_pump_en_mwh_day, "max_pump_en_mwh_day": max_pump_en_mwh_day,
                "min_turb_en_mwh_period": None if min_turb_en_mwh_day is None else min_turb_en_mwh_day * len(days),
                "max_turb_en_mwh_period": None if max_turb_en_mwh_day is None else max_turb_en_mwh_day * len(days),
                "min_pump_en_mwh_period": None if min_pump_en_mwh_day is None else min_pump_en_mwh_day * len(days),
                "max_pump_en_mwh_period": None if max_pump_en_mwh_day is None else max_pump_en_mwh_day * len(days),
                "min_res_hist_pu": aggregated.get("min_res_hist"), "max_res_hist_pu": aggregated.get("max_res_hist"), "min_res_tech_pu": aggregated.get("min_res_tech"), "max_res_tech_pu": aggregated.get("max_res_tech"),
                "imputed": imputed,
            })
            output_rows.append(weekly_row)
    output_rows.sort(key=lambda row: region_values(row, region_level) + (str(row["plant_type"]), str(row["technology"]), int(row["week"])))
    return output_rows, override_summary


def build_daily_rows(weekly_rows: list[dict[str, object]], ref_year: int, region_level: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for weekly_row in weekly_rows:
        week = int(weekly_row["week"])
        for day in expand_week_to_days(ref_year, week):
            rows.append({
                **copy_region_fields(weekly_row, region_level), "ref_year": weekly_row["ref_year"], "plant_type": weekly_row["plant_type"], "technology": weekly_row["technology"],
                "temporal_resolution": "daily", "week": week, "day_of_week": day.day_of_week, "day_of_year": day.day_of_year, "date": day.date.isoformat(),
                "installed_turb_mw": weekly_row["installed_turb_mw"], "installed_pump_mw": weekly_row["installed_pump_mw"], "installed_storage_mwh": weekly_row["installed_storage_mwh"],
                "min_turb_mw": weekly_row["min_turb_mw"], "max_turb_mw": weekly_row["max_turb_mw"], "min_turb_pu": weekly_row["min_turb_pu"], "max_turb_pu": weekly_row["max_turb_pu"],
                "min_pump_mw": weekly_row["min_pump_mw"], "max_pump_mw": weekly_row["max_pump_mw"], "min_pump_pu": weekly_row["min_pump_pu"], "max_pump_pu": weekly_row["max_pump_pu"],
                "min_turb_en_mwh_day": weekly_row["min_turb_en_mwh_day"], "max_turb_en_mwh_day": weekly_row["max_turb_en_mwh_day"], "min_pump_en_mwh_day": weekly_row["min_pump_en_mwh_day"], "max_pump_en_mwh_day": weekly_row["max_pump_en_mwh_day"],
                "min_res_hist_pu": weekly_row["min_res_hist_pu"], "max_res_hist_pu": weekly_row["max_res_hist_pu"], "min_res_tech_pu": weekly_row["min_res_tech_pu"], "max_res_tech_pu": weekly_row["max_res_tech_pu"],
                "imputed": weekly_row["imputed"],
            })
    rows.sort(key=lambda row: region_values(row, region_level) + (str(row["plant_type"]), str(row["technology"]), int(row["week"]), int(row["day_of_year"])))
    return rows


def build_weekly_rows(weekly_base_rows: list[dict[str, object]], region_level: str) -> list[dict[str, object]]:
    return [{
        **copy_region_fields(row, region_level), "ref_year": row["ref_year"], "plant_type": row["plant_type"], "technology": row["technology"], "temporal_resolution": "weekly",
        "week": row["week"], "period_start_date": row["period_start_date"], "period_end_date": row["period_end_date"], "days_in_period": row["days_in_period"],
        "installed_turb_mw": row["installed_turb_mw"], "installed_pump_mw": row["installed_pump_mw"], "installed_storage_mwh": row["installed_storage_mwh"],
        "min_turb_mw": row["min_turb_mw"], "max_turb_mw": row["max_turb_mw"], "min_turb_pu": row["min_turb_pu"], "max_turb_pu": row["max_turb_pu"],
        "min_pump_mw": row["min_pump_mw"], "max_pump_mw": row["max_pump_mw"], "min_pump_pu": row["min_pump_pu"], "max_pump_pu": row["max_pump_pu"],
        "min_turb_en_mwh_day": row["min_turb_en_mwh_day"], "max_turb_en_mwh_day": row["max_turb_en_mwh_day"], "min_pump_en_mwh_day": row["min_pump_en_mwh_day"], "max_pump_en_mwh_day": row["max_pump_en_mwh_day"],
        "min_turb_en_mwh_period": row["min_turb_en_mwh_period"], "max_turb_en_mwh_period": row["max_turb_en_mwh_period"], "min_pump_en_mwh_period": row["min_pump_en_mwh_period"], "max_pump_en_mwh_period": row["max_pump_en_mwh_period"],
        "min_res_hist_pu": row["min_res_hist_pu"], "max_res_hist_pu": row["max_res_hist_pu"], "min_res_tech_pu": row["min_res_tech_pu"], "max_res_tech_pu": row["max_res_tech_pu"],
        "imputed": row["imputed"],
    } for row in weekly_base_rows]


def main() -> None:
    options = parse_args()
    base_dir = options.base_dir.resolve()
    storage_override_path = resolve_storage_override_path(options)
    if options.output is None:
        storage_suffix = "_quaranta_only" if storage_override_path is not None else ""
        phs_suffix = "_phs_as_wr" if options.resolve_phs_to_wr else ""
        output_path = base_dir / f"hydro_constraints_{options.region_level}_{options.temporal_resolution}_{options.year}{storage_suffix}{phs_suffix}.csv"
    else:
        output_path = options.output if options.output.is_absolute() else (base_dir / options.output)
        output_path = output_path.resolve()
    weekly_base_rows, override_summary = build_weekly_base_rows(base_dir, options.year, options.region_level, options.resolve_phs_to_wr, options.impute_missing_constraint_fields, storage_override_path)
    if override_summary is not None:
        print(f"Applied Quaranta-only storage overrides: matched_keys={override_summary['matched_keys']}, changed_keys={override_summary['changed_keys']}, unmatched_override_keys={override_summary['unmatched_override_keys']}")
    if options.temporal_resolution == "daily":
        rows = build_daily_rows(weekly_base_rows, options.year, options.region_level)
        fieldnames = fields_for_region_level(DAILY_OUTPUT_FIELDS, options.region_level)
    else:
        rows = build_weekly_rows(weekly_base_rows, options.region_level)
        fieldnames = fields_for_region_level(WEEKLY_OUTPUT_FIELDS, options.region_level)
    rows = apply_constraint_rounding(rows)
    write_csv(output_path, rows, fieldnames)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
