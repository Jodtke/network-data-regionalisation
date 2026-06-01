from __future__ import annotations

"""Prepare weekly national hydro inflow profiles for bus allocation.

The preprocessing harmonises TYNDP hydro inflows with the plant-type and country
aggregation used by the hydro capacity workflow. Missing years can later be
filled from the self-generated Atlite-like hydro profiles, but this file keeps
the original TYNDP source values and any storage overrides explicit for audit
purposes.
"""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from hydro_commons import (
    active_capacity_index,
    coerce_config_bool,
    coerce_config_choice,
    coerce_config_int,
    coerce_config_path,
    coerce_optional_config_path,
    coerce_path_value,
    copy_region_fields,
    load_capacity_rows,
    load_inflow_rows,
    load_workflow_config,
    region_values,
    resolve_phs_capacities,
    resolve_phs_component_map,
    sum_optional,
    technology_key,
    validate_config_keys,
    validate_tyndp_ref_year,
    write_csv,
)

DEFAULT_QUARANTA_OVERRIDE_PATH = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\DATA\hydro\quaranta2024\hydro_capacity_2025_quaranta.csv")
ALLOWED_OVERRIDE_SOURCE = "Quaranta et al. 2024"
COUNTRY_MAP = {"GB": "UK"}
PLANT_TYPE_MAP = {"Hydro Pumped Storage": "phs", "Hydro Water Reservoir": "wr"}

OUTPUT_FIELDS = [
    "country", "ref_year", "weather_year", "week", "plant_type", "technology",
    "installed_turb_mw", "installed_pump_mw", "installed_storage_mwh", "inflow_mwh_week",
]
INFLOW_ROUNDING = {
    "installed_turb_mw": 0,
    "installed_pump_mw": 0,
    "installed_storage_mwh": 0,
    "inflow_mwh_week": 0,
}
CONFIG_KEYS = {
    "year", "base_dir", "output", "region_level", "resolve_phs_to_wr",
    "weather_year_start", "weather_year_end",
    "use_default_storage_overrides", "storage_override_csv",
}


@dataclass(frozen=True)
class InflowOptions:
    year: int
    base_dir: Path
    output: Path | None
    region_level: str
    resolve_phs_to_wr: bool
    weather_year_start: int
    weather_year_end: int
    use_default_storage_overrides: bool
    storage_override_csv: Path | None


def fields_for_region_level(fields: list[str], region_level: str) -> list[str]:
    if region_level == "country":
        return fields
    return ["country", "market_node", *fields[1:]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bereitet TYNDP2024-Hydro-Zufluesse je Land, Technologie, Wetterjahr und Woche auf.")
    parser.add_argument("--config", type=Path, default=None, help="Optionale YAML-Konfiguration mit Werten aus 'common' und 'inflows'.")
    parser.add_argument("--year", type=int, default=argparse.SUPPRESS, help="Zieljahr / ref_year.")
    parser.add_argument("--base-dir", type=Path, default=argparse.SUPPRESS, help="Verzeichnis mit den TYNDP2024-Country-CSV-Dateien.")
    parser.add_argument("--output", type=Path, default=argparse.SUPPRESS, help="Optionaler Ausgabepfad.")
    parser.add_argument("--region-level", choices=("country", "market_node"), default=argparse.SUPPRESS, help="Raeumliche Granularitaet: country oder market_node.")
    parser.add_argument("--resolve-phs-to-wr", dest="resolve_phs_to_wr", action="store_true", default=argparse.SUPPRESS, help="Aggregiert PHS-Zufluesse in WR und entfernt PHS aus dem Output.")
    parser.add_argument("--no-resolve-phs-to-wr", dest="resolve_phs_to_wr", action="store_false", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--weather-year-start", type=int, default=argparse.SUPPRESS, help="Erstes Wetterjahr.")
    parser.add_argument("--weather-year-end", type=int, default=argparse.SUPPRESS, help="Letztes Wetterjahr.")
    parser.add_argument("--use-default-storage-overrides", dest="use_default_storage_overrides", action="store_true", default=argparse.SUPPRESS, help="Verwendet die bekannte Quaranta-Datei und ignoriert alle Nicht-Quaranta-Quellen.")
    parser.add_argument("--no-default-storage-overrides", dest="use_default_storage_overrides", action="store_false", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--storage-override-csv", type=Path, default=argparse.SUPPRESS, help="Alternative Override-Datei im Quaranta-Format.")
    return parser


def load_options(args: argparse.Namespace) -> InflowOptions:
    script_dir = Path(__file__).resolve().parent
    config_path = args.config.resolve() if args.config is not None else None
    config_values: dict[str, object] = {}
    if config_path is not None:
        config_values = load_workflow_config(config_path, "inflows")
        validate_config_keys(config_values, CONFIG_KEYS, "inflows", config_path)
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
    resolve_phs_to_wr = coerce_config_bool(
        cli_values.get("resolve_phs_to_wr", config_values.get("resolve_phs_to_wr", False)),
        "resolve_phs_to_wr",
    )
    weather_year_start = coerce_config_int(
        cli_values.get("weather_year_start", config_values.get("weather_year_start", 1982)),
        "weather_year_start",
    )
    weather_year_end = coerce_config_int(
        cli_values.get("weather_year_end", config_values.get("weather_year_end", 2016)),
        "weather_year_end",
    )
    if weather_year_start > weather_year_end:
        raise ValueError("'weather_year_start' must be less than or equal to 'weather_year_end'.")
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
    return InflowOptions(
        year=year,
        base_dir=base_dir,
        output=output,
        region_level=region_level,
        resolve_phs_to_wr=resolve_phs_to_wr,
        weather_year_start=weather_year_start,
        weather_year_end=weather_year_end,
        use_default_storage_overrides=use_default_storage_overrides,
        storage_override_csv=storage_override_csv,
    )


def parse_args(argv: list[str] | None = None) -> InflowOptions:
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


def resolve_storage_override_path(options: InflowOptions) -> Path | None:
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


def apply_inflow_rounding(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rounded_rows: list[dict[str, object]] = []
    for row in rows:
        rounded = dict(row)
        for field, decimals in INFLOW_ROUNDING.items():
            if field in rounded:
                rounded[field] = format_rounded(rounded[field], decimals)
        rounded_rows.append(rounded)
    return rounded_rows


def build_rows(base_dir: Path, ref_year: int, region_level: str, resolve_phs_to_wr: bool, weather_year_start: int, weather_year_end: int, storage_override_path: Path | None) -> tuple[list[dict[str, object]], dict[str, int] | None]:
    capacity_rows = load_capacity_rows(base_dir, ref_year, region_level)
    routing_capacity_rows = [dict(row) for row in capacity_rows]
    override_summary = None
    if storage_override_path is not None:
        overrides = load_storage_overrides(storage_override_path)
        capacity_rows, override_summary = apply_storage_overrides(capacity_rows, overrides)
    source_capacity_index = {technology_key(row, region_level): row for row in capacity_rows}
    if resolve_phs_to_wr:
        resolved_component_map = resolve_phs_component_map(capacity_rows, routing_capacity_rows, region_level)
        capacity_rows = resolve_phs_capacities(capacity_rows, routing_capacity_rows, region_level)
    else:
        resolved_component_map = {key: [key] for key in source_capacity_index}
    active_capacities = active_capacity_index(capacity_rows, region_level)
    inflow_rows = [row for row in load_inflow_rows(base_dir, ref_year, region_level) if weather_year_start <= int(row["weather_year"]) <= weather_year_end]
    inflow_map = {technology_key(row, region_level) + (int(row["weather_year"]), int(row["week"])): row for row in inflow_rows}
    weather_years = sorted({int(row["weather_year"]) for row in inflow_rows})
    weeks = sorted({int(row["week"]) for row in inflow_rows})
    output_rows: list[dict[str, object]] = []
    for capacity_key, capacity_row in sorted(active_capacities.items(), key=lambda item: item[0]):
        plant_type, technology = capacity_key[-2:]
        installed_turb_mw = float(capacity_row.get("turb_mw") or 0.0)
        installed_pump_mw = float(capacity_row.get("pump_mw") or 0.0)
        installed_storage_mwh = float(capacity_row.get("storage_gwh") or 0.0) * 1000.0
        source_keys = resolved_component_map.get(capacity_key, [capacity_key])
        only_closed_loop_phs = all(source_key[-2] == "phs" and source_key[-1] == "closed_loop" for source_key in source_keys)
        for weather_year in weather_years:
            for week in weeks:
                component_keys = [source_key + (weather_year, week) for source_key in source_keys]
                component_rows = [inflow_map[key] for key in component_keys if key in inflow_map]
                if component_rows:
                    inflow_gwh_week = sum_optional(row.get("inflow_gwh_week") for row in component_rows)  # type: ignore[arg-type]
                elif only_closed_loop_phs:
                    inflow_gwh_week = 0.0
                else:
                    continue
                if inflow_gwh_week is None:
                    continue
                output_rows.append({
                    **copy_region_fields(capacity_row, region_level), "ref_year": ref_year, "weather_year": weather_year, "week": week,
                    "plant_type": plant_type, "technology": technology,
                    "installed_turb_mw": installed_turb_mw, "installed_pump_mw": installed_pump_mw,
                    "installed_storage_mwh": installed_storage_mwh, "inflow_mwh_week": inflow_gwh_week * 1000.0,
                })
    output_rows.sort(key=lambda row: region_values(row, region_level) + (str(row["plant_type"]), str(row["technology"]), int(row["weather_year"]), int(row["week"])))
    return output_rows, override_summary


def main() -> None:
    options = parse_args()
    base_dir = options.base_dir.resolve()
    storage_override_path = resolve_storage_override_path(options)
    if options.output is None:
        storage_suffix = "_quaranta_only" if storage_override_path is not None else ""
        phs_suffix = "_phs_as_wr" if options.resolve_phs_to_wr else ""
        output_path = base_dir / f"hydro_inflows_{options.region_level}_weekly_{options.year}{storage_suffix}{phs_suffix}.csv"
    else:
        output_path = options.output if options.output.is_absolute() else (base_dir / options.output)
        output_path = output_path.resolve()
    rows, override_summary = build_rows(
        base_dir,
        options.year,
        options.region_level,
        options.resolve_phs_to_wr,
        options.weather_year_start,
        options.weather_year_end,
        storage_override_path,
    )
    if override_summary is not None:
        print(f"Applied Quaranta-only storage overrides: matched_keys={override_summary['matched_keys']}, changed_keys={override_summary['changed_keys']}, unmatched_override_keys={override_summary['unmatched_override_keys']}")
    rows = apply_inflow_rounding(rows)
    write_csv(output_path, rows, fields_for_region_level(OUTPUT_FIELDS, options.region_level))
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
