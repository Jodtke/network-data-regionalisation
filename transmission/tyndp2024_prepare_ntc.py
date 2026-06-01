"""Prepare TYNDP 2024 net transfer capacities for the reduced topology.

The network reduction can merge countries or exclude special model regions from
capacity allocation. This script translates TYNDP border-level NTC data to the
same model-country representation, aggregates directional capacities where
needed, and writes diagnostics for borders that are lost, merged, or not covered
by the reduced topology.

Example:
python transmission/tyndp2024_prepare_ntc.py ^
  --raw-ntc "Y:\\...\\ntcs_2030_tyndp2024_RAW.csv" ^
  --grid-dir "Y:\\...\\grid\\target_year_2030\\electrical_spectral_line_equivalent" ^
  --output-dir "Y:\\...\\input\\single_year" ^
  --target-year 2030
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grid.pipeline_config import deep_merge, load_yaml_like, resolve_path


TYNDP_TARGET_YEARS = (2030, 2040, 2050)
TYNDP_YEAR_PATTERN = re.compile(r"(?<!\d)(2030|2040|2050)(?!\d)")
TYNDP_YEAR_COLUMNS = (
    "target_year",
    "ref_year",
    "reference_year",
    "scenario_year",
    "year",
)
NTC_FORWARD_COLUMNS = ("from->to", "ntc->", "ntc_to", "ntc_to_mw")
NTC_BACKWARD_COLUMNS = ("to->from", "ntc<-", "ntc_from", "ntc_from_mw")
DEFAULT_OUTPUT_NAME = "ntc_tyndp2024.csv"
PATH_SETTING_NAMES = {"raw_ntc", "grid_dir", "output_dir", "excluded_countries_csv"}


COUNTRY_ALIASES = {
    "UK": "GB",
    "EL": "GR",
    "KV": "XK",
    "KO": "XK",
}

# Raw country pairs that are represented by HVDC interconnectors in the
# TYNDP2024 NTC input. Everything else defaults to HVAC.
HVDC_RAW_PAIRS = {
    tuple(sorted(pair))
    for pair in [
        ("BE", "GB"),
        ("CY", "GR"),
        ("CY", "IL"),
        ("DE", "DK"),
        ("DE", "GB"),
        ("DE", "NO"),
        ("DE", "SE"),
        ("DK", "GB"),
        ("DK", "NL"),
        ("DK", "NO"),
        ("DK", "SE"),
        ("EE", "FI"),
        ("ES", "MA"),
        ("FI", "SE"),
        ("FR", "GB"),
        ("FR", "IE"),
        ("GB", "IE"),
        ("GB", "NL"),
        ("GB", "NO"),
        ("GR", "IT"),
        ("IT", "ME"),
        ("IT", "MT"),
        ("IT", "TN"),
        ("LT", "SE"),
        ("NL", "NO"),
        ("PL", "SE"),
    ]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare aggregated TYNDP2024 NTC inputs.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML scenario config.")
    parser.add_argument("--raw-ntc", type=Path, default=None, help="Raw TYNDP2024 NTC CSV.")
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=None,
        help="Reduced-grid directory with cesa_country_clusters.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="single_year output directory.")
    parser.add_argument(
        "--target-year",
        type=int,
        default=None,
        help="TYNDP target year. Defaults to target_year_YYYY in --grid-dir and is used to filter/validate raw rows.",
    )
    parser.add_argument("--output-name", default=None, help="Name of the aggregated NTC output CSV.")
    parser.add_argument(
        "--excluded-countries-csv",
        type=Path,
        default=None,
        help="Optional excluded_countries.csv from the reduced-grid directory.",
    )
    return parser.parse_args()


def default_settings() -> dict[str, Any]:
    return {
        "raw_ntc": None,
        "grid_dir": None,
        "output_dir": None,
        "target_year": None,
        "output_name": DEFAULT_OUTPUT_NAME,
        "excluded_countries_csv": None,
    }


def resolve_settings(args: argparse.Namespace) -> argparse.Namespace:
    settings = default_settings()
    config_base = Path.cwd()
    if args.config is not None:
        config_path = args.config.resolve()
        config_base = config_path.parent
        settings = deep_merge(settings, load_yaml_like(config_path))

    for key in list(settings):
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                settings[key] = value

    for key in PATH_SETTING_NAMES:
        settings[key] = resolve_path(settings.get(key), base_dir=config_base)

    missing = [key for key in ("raw_ntc", "grid_dir", "output_dir") if settings.get(key) is None]
    if missing:
        raise ValueError(f"Missing required transmission setting(s): {', '.join(missing)}")

    settings["target_year"] = infer_target_year(settings["grid_dir"], settings.get("target_year"))
    settings["output_name"] = str(settings.get("output_name") or DEFAULT_OUTPUT_NAME)
    return argparse.Namespace(**settings)


def validate_tyndp_target_year(year: int) -> int:
    target_year = int(year)
    if target_year not in TYNDP_TARGET_YEARS:
        allowed = ", ".join(str(item) for item in TYNDP_TARGET_YEARS)
        raise ValueError(f"TYNDP target year must be one of {allowed}; got {target_year}.")
    return target_year


def detect_tyndp_years_in_path(path: Path) -> list[int]:
    return sorted({int(match.group(1)) for match in TYNDP_YEAR_PATTERN.finditer(str(path))})


def validate_path_target_year(
    path: Path,
    target_year: int,
    *,
    context: str,
    allow_unlabelled: bool = False,
) -> None:
    years = detect_tyndp_years_in_path(path)
    if years == [target_year]:
        return
    if not years:
        if allow_unlabelled:
            return
        allowed = ", ".join(str(item) for item in TYNDP_TARGET_YEARS)
        raise ValueError(
            f"{context} has no target-year column and its path does not identify one of {allowed}: {path}"
        )
    raise ValueError(
        f"{context} path identifies TYNDP year(s) {years}, but --target-year is {target_year}: {path}"
    )


def normalized_field_name(value: Any) -> str:
    return str(value or "").strip().lower()


def find_field(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {normalized_field_name(field): field for field in fieldnames}
    for candidate in candidates:
        field = lookup.get(normalized_field_name(candidate))
        if field is not None:
            return field
    return None


def row_matches_target_year(row: dict[str, Any], year_field: str, target_year: int) -> bool:
    value = row.get(year_field)
    text = str(value or "").strip()
    if not text:
        return False
    try:
        return int(float(text.replace(",", "."))) == target_year
    except ValueError as exc:
        raise ValueError(f"Invalid TYNDP year value {value!r} in column {year_field!r}.") from exc


def norm_country(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return COUNTRY_ALIASES.get(text, text)


def zone_to_country(value: Any) -> str:
    text = str(value or "").strip().upper()
    match = re.match(r"([A-Z]{2})", text)
    if match is None:
        return norm_country(text)
    return norm_country(match.group(1))


def infer_target_year(grid_dir: Path, explicit: int | None) -> int:
    if explicit is not None:
        return validate_tyndp_target_year(int(explicit))
    match = re.search(r"target_year_(\d{4})", str(grid_dir))
    if match is None:
        raise ValueError("Could not infer target_year from grid_dir. Pass --target-year explicitly.")
    return validate_tyndp_target_year(int(match.group(1)))


def parse_number(value: Any) -> float:
    text = str(value or "").strip().replace(" ", "")
    if not text:
        return 0.0
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return 0.0


def load_country_cluster_map(grid_dir: Path) -> list[dict[str, str]]:
    path = grid_dir / "cesa_country_clusters.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing cluster mapping file: {path}")
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            source = norm_country(row.get("source_country", ""))
            target = norm_country(row.get("target_country", ""))
            label = str(row.get("target_label", "") or "").strip()
            if not source or not target:
                continue
            rows.append(
                {
                    "source_country": source,
                    "target_country": target,
                    "target_label": label,
                }
            )
    if not rows:
        raise ValueError(f"No country cluster rows found in {path}")
    return rows


def load_excluded_countries(path: Path | None, grid_dir: Path) -> set[str]:
    candidate = path if path is not None else (grid_dir / "excluded_countries.csv")
    if candidate is None or not candidate.exists():
        return set()
    rows: set[str] = set()
    with candidate.open("r", newline="", encoding="utf-8-sig") as handle:
        sample = handle.readline()
        handle.seek(0)
        delimiter = ";" if sample.count(";") >= sample.count(",") else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            return set()
        country_col = "source_country" if "source_country" in reader.fieldnames else (
            "country" if "country" in reader.fieldnames else None
        )
        if country_col is None:
            return set()
        for row in reader:
            country = norm_country(row.get(country_col))
            if country:
                rows.add(country)
    return rows


def collapse_country(country: str, source_to_target: dict[str, str]) -> str:
    normalized = norm_country(country)
    return source_to_target.get(normalized, normalized)


def classify_raw_pair_type(country_a: str, country_b: str) -> str:
    pair = tuple(sorted((norm_country(country_a), norm_country(country_b))))
    return "hvdc" if pair in HVDC_RAW_PAIRS else "hvac"


def aggregate_ntc(
    raw_ntc_path: Path,
    *,
    source_to_target: dict[str, str],
    target_year: int,
    excluded_countries: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arc_values: dict[tuple[str, str], float] = defaultdict(float)
    pair_types: dict[tuple[str, str], set[str]] = defaultdict(set)
    diag_rows: list[dict[str, Any]] = []
    raw_row_count = 0
    used_row_count = 0
    excluded_countries = excluded_countries or set()

    with raw_ntc_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        fieldnames = list(reader.fieldnames or [])
        required = {"Border"}
        missing = required - set(fieldnames)
        if missing:
            raise KeyError(f"{raw_ntc_path.name} missing columns: {sorted(missing)}")
        ntc_forward_field = find_field(fieldnames, NTC_FORWARD_COLUMNS)
        ntc_backward_field = find_field(fieldnames, NTC_BACKWARD_COLUMNS)
        if ntc_forward_field is None or ntc_backward_field is None:
            raise KeyError(
                f"{raw_ntc_path.name} must contain one forward NTC column from {NTC_FORWARD_COLUMNS} "
                f"and one backward NTC column from {NTC_BACKWARD_COLUMNS}; got {fieldnames}"
            )
        year_field = find_field(fieldnames, TYNDP_YEAR_COLUMNS)
        if year_field is None:
            validate_path_target_year(
                raw_ntc_path,
                target_year,
                context="Raw NTC file",
                allow_unlabelled=True,
            )

        for row in reader:
            raw_row_count += 1
            if year_field is not None and not row_matches_target_year(row, year_field, target_year):
                continue
            used_row_count += 1
            border = str(row.get("Border", "") or "").strip()
            if "-" not in border:
                continue
            zone_from, zone_to = border.split("-", 1)
            country_from_raw = zone_to_country(zone_from)
            country_to_raw = zone_to_country(zone_to)
            if country_from_raw in excluded_countries or country_to_raw in excluded_countries:
                diag_rows.append(
                    {
                        "target_year": target_year,
                        "border": border,
                        "zone_from": zone_from,
                        "zone_to": zone_to,
                        "country_from_raw": country_from_raw,
                        "country_to_raw": country_to_raw,
                        "country_from_model": "",
                        "country_to_model": "",
                        "ntc_to_raw_mw": int(round(max(0.0, parse_number(row.get(ntc_forward_field, 0.0))))),
                        "ntc_from_raw_mw": int(round(max(0.0, parse_number(row.get(ntc_backward_field, 0.0))))),
                        "type": "excluded_country",
                        "becomes_internal": "excluded_country",
                    }
                )
                continue
            country_from_model = collapse_country(country_from_raw, source_to_target)
            country_to_model = collapse_country(country_to_raw, source_to_target)
            ntc_to = max(0.0, parse_number(row.get(ntc_forward_field, 0.0)))
            ntc_from = max(0.0, parse_number(row.get(ntc_backward_field, 0.0)))
            pair_type = classify_raw_pair_type(country_from_raw, country_to_raw)
            becomes_internal = country_from_model == country_to_model

            diag_rows.append(
                {
                    "target_year": target_year,
                    "border": border,
                    "zone_from": zone_from,
                    "zone_to": zone_to,
                    "country_from_raw": country_from_raw,
                    "country_to_raw": country_to_raw,
                    "country_from_model": country_from_model,
                    "country_to_model": country_to_model,
                    "ntc_to_raw_mw": int(round(ntc_to)),
                    "ntc_from_raw_mw": int(round(ntc_from)),
                    "type": pair_type,
                    "becomes_internal": str(becomes_internal),
                }
            )

            if becomes_internal:
                continue
            if ntc_to > 0.0:
                arc_values[(country_from_model, country_to_model)] += ntc_to
            if ntc_from > 0.0:
                arc_values[(country_to_model, country_from_model)] += ntc_from
            pair_types[tuple(sorted((country_from_model, country_to_model)))].add(pair_type)

    if year_field is not None and raw_row_count > 0 and used_row_count == 0:
        raise ValueError(f"No raw NTC rows for target_year={target_year} in {raw_ntc_path}.")

    pair_rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for country_from, country_to in sorted(arc_values):
        pair = tuple(sorted((country_from, country_to)))
        if pair in seen_pairs or pair[0] == pair[1]:
            continue
        seen_pairs.add(pair)
        ntc_to = arc_values.get((pair[0], pair[1]), 0.0)
        ntc_from = arc_values.get((pair[1], pair[0]), 0.0)
        if ntc_to <= 0.0 and ntc_from <= 0.0:
            continue
        types = sorted(pair_types.get(pair, {"hvac"}))
        pair_rows.append(
            {
                "target_year": target_year,
                "country_from": pair[0],
                "country_to": pair[1],
                "ntc_to": int(round(ntc_to)),
                "ntc_from": int(round(ntc_from)),
                "type": types[0] if len(types) == 1 else "mixed",
            }
        )

    pair_rows.sort(key=lambda row: (row["country_from"], row["country_to"]))
    diag_rows.sort(key=lambda row: row["border"])
    return pair_rows, diag_rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = resolve_settings(parse_args())
    target_year = args.target_year

    country_map_rows = load_country_cluster_map(args.grid_dir)
    excluded_countries = load_excluded_countries(args.excluded_countries_csv, args.grid_dir)
    source_to_target = {
        row["source_country"]: row["target_country"]
        for row in country_map_rows
    }
    ntc_rows, diag_rows = aggregate_ntc(
        args.raw_ntc,
        source_to_target=source_to_target,
        target_year=target_year,
        excluded_countries=excluded_countries,
    )

    out_ntc = args.output_dir / args.output_name
    out_map = args.output_dir / f"country_aggregation_map_{target_year}_tyndp2024.csv"
    out_diag = args.output_dir / f"ntc_aggregation_diag_{target_year}_tyndp2024.csv"

    write_csv(
        out_ntc,
        ["target_year", "country_from", "country_to", "ntc_to", "ntc_from", "type"],
        ntc_rows,
    )
    write_csv(
        out_map,
        ["source_country", "target_country", "target_label"],
        country_map_rows,
    )
    write_csv(
        out_diag,
        [
            "target_year",
            "border",
            "zone_from",
            "zone_to",
            "country_from_raw",
            "country_to_raw",
            "country_from_model",
            "country_to_model",
            "ntc_to_raw_mw",
            "ntc_from_raw_mw",
            "type",
            "becomes_internal",
        ],
        diag_rows,
    )

    print(f"NTC written to: {out_ntc}")
    print(f"Country map written to: {out_map}")
    print(f"Diagnostics written to: {out_diag}")
    print(f"NTC rows: {len(ntc_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
