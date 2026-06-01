from __future__ import annotations

"""Extract TYNDP 2024 profiles for residual RES and non-RES technologies.

The TYNDP workbooks contain profile-like assumptions in a layout that is not
directly usable by the nodal disaggregation modules. This script reads the
workbook sheets, normalises technology and country labels, and writes tidy
tables that can be joined to the allocated bus capacities.
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

from tyndp2024_excel import (
    format_number,
    iter_hourly_rows,
    normalize_label,
    read_sheet,
    safe_float,
    safe_int,
)


LOG = logging.getLogger(__name__)
DEFAULT_PEMMDB_DIR = Path(r"Y:\Data\TYNDP2024\raw\PEMMDB2")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\jr8037\bwSyncShare\Dissertation\DATA\raw\others\tyndp2024")
TARGET_YEARS = (2030, 2040, 2050)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract TYNDP 2024 Other RES and Other non-RES capacity and hourly availability from PEMMDB2."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_PEMMDB_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--years", nargs="*", type=int, default=list(TARGET_YEARS))
    parser.add_argument("--scenario", default="NationalTrends")
    parser.add_argument(
        "--market-nodes",
        nargs="*",
        default=None,
        help="Optional PEMMDB market-node filter such as DE00 AT00 for test runs.",
    )
    parser.add_argument("--skip-other-res", action="store_true")
    parser.add_argument("--skip-other-nonres", action="store_true")
    parser.add_argument(
        "--write-other-nonres-band-profiles",
        action="store_true",
        help="Also write hourly Other non-RES profiles per price band before summing to the national profile.",
    )
    return parser.parse_args()


def discover_workbooks(input_dir: Path, year: int, scenario: str) -> list[Path]:
    year_dir = input_dir / str(year)
    pattern = f"PEMMDB_*_{scenario}_{year}.xlsx"
    return sorted(
        path
        for path in year_dir.glob(pattern)
        if path.is_file() and not path.name.startswith("~$")
    )


def mean_numeric(values: list[Any], default: float = 0.0) -> float:
    parsed = [safe_float(value) for value in values]
    nums = [float(value) for value in parsed if value is not None]
    if not nums:
        return default
    return sum(nums) / len(nums)


def join_unique(values: list[Any]) -> str:
    clean = sorted({str(value).strip() for value in values if str(value).strip()})
    return ",".join(clean)


def metadata(sheet: Any, scenario_row: int) -> dict[str, Any]:
    return {
        "country": sheet.cell(3, 2).strip(),
        "market_node": sheet.cell(4, 2).strip(),
        "year": safe_int(sheet.cell(5, 2)),
        "scenario": sheet.cell(scenario_row, 2).strip(),
    }


def other_res_technologies(sheet: Any, source_file: Path) -> list[dict[str, Any]]:
    meta = metadata(sheet, scenario_row=6)
    columns = sheet.nonempty_columns((8, 9), min_column=5)
    rows: list[dict[str, Any]] = []
    for column in columns:
        raw_technology = sheet.cell(8, column).strip()
        if not raw_technology:
            continue
        installed = safe_float(sheet.cell(9, column), 0.0) or 0.0
        rows.append(
            {
                **meta,
                "technology": normalize_label(raw_technology),
                "technology_label": raw_technology,
                "technology_column": column,
                "installed_capacity_MW": installed,
                "source_file": str(source_file),
            }
        )
    return rows


def other_nonres_band_groups(sheet: Any, source_file: Path) -> list[dict[str, Any]]:
    meta = metadata(sheet, scenario_row=7)
    columns = sheet.nonempty_columns(range(8, 18), min_column=3)
    groups: list[dict[str, Any]] = []
    last_label = ""
    for column in columns:
        label = sheet.cell(8, column).strip() or last_label
        if label:
            last_label = label
        installed = safe_float(sheet.cell(9, column))
        price = safe_float(sheet.cell(13, column))
        if not label and installed is None and price is None:
            continue
        if groups and groups[-1]["price_band_label"] == label:
            groups[-1]["columns"].append(column)
            continue
        groups.append({"price_band_label": label or f"Price Band {len(groups) + 1}", "columns": [column]})

    out: list[dict[str, Any]] = []
    for ordinal, group in enumerate(groups, start=1):
        cols = list(group["columns"])
        capacities = [sheet.cell(9, col) for col in cols]
        units = [sheet.cell(10, col) for col in cols]
        prices = [sheet.cell(13, col) for col in cols]
        efficiencies = [sheet.cell(14, col) for col in cols]
        co2 = [sheet.cell(15, col) for col in cols]
        starts = [sheet.cell(16, col) for col in cols]
        ends = [sheet.cell(17, col) for col in cols]
        out.append(
            {
                **meta,
                "price_band_id": f"pb{ordinal:03d}",
                "price_band_label": group["price_band_label"],
                "band_columns": ",".join(str(col) for col in cols),
                "installed_capacity_MW": mean_numeric(capacities),
                "units": mean_numeric(units),
                "pemmdb_types": join_unique([sheet.cell(11, col) for col in cols]),
                "purpose": join_unique([sheet.cell(12, col) for col in cols]),
                "price_EUR_MWh": mean_numeric(prices),
                "efficiency": mean_numeric(efficiencies),
                "co2_factor_t_per_MWh": mean_numeric(co2),
                "source_climate_year_start": join_unique(starts),
                "source_climate_year_end": join_unique(ends),
                "source_file": str(source_file),
            }
        )
    return out


def write_other_res(
    *,
    workbook: Path,
    profile_writer: csv.DictWriter,
    capacity_writer: csv.DictWriter,
) -> tuple[int, int]:
    sheet = read_sheet(workbook, "Other RES")
    techs = other_res_technologies(sheet, workbook)
    for row in techs:
        capacity_writer.writerow({key: format_number(value) for key, value in row.items()})
    profile_rows = 0
    for timestep, date, hour, values in iter_hourly_rows(sheet):
        for tech in techs:
            installed = float(tech["installed_capacity_MW"] or 0.0)
            available = safe_float(values.get(int(tech["technology_column"])), 0.0) or 0.0
            if installed <= 0.0 and available == 0.0:
                continue
            factor = available / installed if installed > 0.0 else 0.0
            profile_writer.writerow(
                {
                    "country": tech["country"],
                    "market_node": tech["market_node"],
                    "year": tech["year"],
                    "scenario": tech["scenario"],
                    "technology": tech["technology"],
                    "technology_label": tech["technology_label"],
                    "date": date,
                    "hour": hour,
                    "timestep": timestep,
                    "installed_capacity_mw": format_number(installed),
                    "available_capacity_mw": format_number(available),
                    "availability_factor": format_number(factor),
                }
            )
            profile_rows += 1
    return len(techs), profile_rows


def write_other_nonres(
    *,
    workbook: Path,
    profile_writer: csv.DictWriter,
    capacity_writer: csv.DictWriter,
    band_profile_writer: csv.DictWriter | None,
) -> tuple[int, int, int]:
    sheet = read_sheet(workbook, "Other Non-RES")
    bands = other_nonres_band_groups(sheet, workbook)
    for row in bands:
        capacity_writer.writerow({key: format_number(value) for key, value in row.items()})
    profile_rows = 0
    band_profile_rows = 0
    installed_total = sum(float(row["installed_capacity_MW"] or 0.0) for row in bands)
    source_bands = ",".join(row["price_band_id"] for row in bands)
    source_climate_years = ",".join(
        sorted(
            {
                str(year)
                for row in bands
                for year in str(row["source_climate_year_start"]).split(",")
                if str(year).strip()
            }
        )
    )
    if not bands:
        return 0, 0, 0
    first = bands[0]
    for timestep, date, hour, values in iter_hourly_rows(sheet):
        available_total = 0.0
        for band in bands:
            columns = [int(col) for col in str(band["band_columns"]).split(",") if str(col).strip()]
            available = mean_numeric([values.get(column) for column in columns])
            available_total += available
            if band_profile_writer is not None:
                installed = float(band["installed_capacity_MW"] or 0.0)
                factor = available / installed if installed > 0.0 else 0.0
                band_profile_writer.writerow(
                    {
                        "country": band["country"],
                        "market_node": band["market_node"],
                        "year": band["year"],
                        "scenario": band["scenario"],
                        "price_band_id": band["price_band_id"],
                        "price_band_label": band["price_band_label"],
                        "price_EUR_MWh": format_number(band["price_EUR_MWh"]),
                        "date": date,
                        "hour": hour,
                        "timestep": timestep,
                        "installed_capacity_mw": format_number(installed),
                        "available_capacity_mw": format_number(available),
                        "availability_factor": format_number(factor),
                    }
                )
                band_profile_rows += 1
        profile_writer.writerow(
            {
                "country": first["country"],
                "market_node": first["market_node"],
                "year": first["year"],
                "scenario": first["scenario"],
                "technology": "other_nonres",
                "date": date,
                "hour": hour,
                "timestep": timestep,
                "installed_capacity_mw": format_number(installed_total),
                "available_capacity_mw": format_number(available_total),
                "availability_factor": format_number(available_total / installed_total if installed_total > 0.0 else 0.0),
                "source_price_bands": source_bands,
                "source_climate_years": source_climate_years,
            }
        )
        profile_rows += 1
    return len(bands), profile_rows, band_profile_rows


def process_year(
    *,
    year: int,
    workbooks: list[Path],
    output_dir: Path,
    market_nodes: set[str] | None,
    skip_other_res: bool,
    skip_other_nonres: bool,
    write_other_nonres_band_profiles: bool,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "other_res_capacity_rows": 0,
        "other_res_profile_rows": 0,
        "other_nonres_capacity_rows": 0,
        "other_nonres_profile_rows": 0,
        "other_nonres_band_profile_rows": 0,
    }

    other_res_capacity_header = [
        "country",
        "market_node",
        "year",
        "scenario",
        "technology",
        "technology_label",
        "technology_column",
        "installed_capacity_MW",
        "source_file",
    ]
    other_res_profile_header = [
        "country",
        "market_node",
        "year",
        "scenario",
        "technology",
        "technology_label",
        "date",
        "hour",
        "timestep",
        "installed_capacity_mw",
        "available_capacity_mw",
        "availability_factor",
    ]
    other_nonres_capacity_header = [
        "country",
        "market_node",
        "year",
        "scenario",
        "price_band_id",
        "price_band_label",
        "band_columns",
        "installed_capacity_MW",
        "units",
        "pemmdb_types",
        "purpose",
        "price_EUR_MWh",
        "efficiency",
        "co2_factor_t_per_MWh",
        "source_climate_year_start",
        "source_climate_year_end",
        "source_file",
    ]
    other_nonres_profile_header = [
        "country",
        "market_node",
        "year",
        "scenario",
        "technology",
        "date",
        "hour",
        "timestep",
        "installed_capacity_mw",
        "available_capacity_mw",
        "availability_factor",
        "source_price_bands",
        "source_climate_years",
    ]
    other_nonres_band_profile_header = [
        "country",
        "market_node",
        "year",
        "scenario",
        "price_band_id",
        "price_band_label",
        "price_EUR_MWh",
        "date",
        "hour",
        "timestep",
        "installed_capacity_mw",
        "available_capacity_mw",
        "availability_factor",
    ]

    files: list[Any] = []
    try:
        other_res_capacity_writer = other_res_profile_writer = None
        other_nonres_capacity_writer = other_nonres_profile_writer = other_nonres_band_profile_writer = None
        if not skip_other_res:
            f = (output_dir / f"other_res_capacity_{year}_tyndp2024.csv").open("w", newline="", encoding="utf-8")
            files.append(f)
            other_res_capacity_writer = csv.DictWriter(f, fieldnames=other_res_capacity_header, delimiter=";", lineterminator="\n")
            other_res_capacity_writer.writeheader()
            f = (output_dir / f"other_res_availability_{year}_tyndp2024.csv").open("w", newline="", encoding="utf-8")
            files.append(f)
            other_res_profile_writer = csv.DictWriter(f, fieldnames=other_res_profile_header, delimiter=";", lineterminator="\n")
            other_res_profile_writer.writeheader()
        if not skip_other_nonres:
            f = (output_dir / f"other_nonres_capacity_{year}_tyndp2024.csv").open("w", newline="", encoding="utf-8")
            files.append(f)
            other_nonres_capacity_writer = csv.DictWriter(f, fieldnames=other_nonres_capacity_header, delimiter=";", lineterminator="\n")
            other_nonres_capacity_writer.writeheader()
            f = (output_dir / f"other_nonres_availability_{year}_tyndp2024.csv").open("w", newline="", encoding="utf-8")
            files.append(f)
            other_nonres_profile_writer = csv.DictWriter(f, fieldnames=other_nonres_profile_header, delimiter=";", lineterminator="\n")
            other_nonres_profile_writer.writeheader()
            if write_other_nonres_band_profiles:
                f = (output_dir / f"other_nonres_band_availability_{year}_tyndp2024.csv").open("w", newline="", encoding="utf-8")
                files.append(f)
                other_nonres_band_profile_writer = csv.DictWriter(
                    f,
                    fieldnames=other_nonres_band_profile_header,
                    delimiter=";",
                    lineterminator="\n",
                )
                other_nonres_band_profile_writer.writeheader()

        for workbook in workbooks:
            market_node = workbook.name.split("_")[1] if "_" in workbook.name else ""
            if market_nodes is not None and market_node not in market_nodes:
                continue
            if other_res_capacity_writer is not None and other_res_profile_writer is not None:
                try:
                    capacity_rows, profile_rows = write_other_res(
                        workbook=workbook,
                        profile_writer=other_res_profile_writer,
                        capacity_writer=other_res_capacity_writer,
                    )
                    counts["other_res_capacity_rows"] += capacity_rows
                    counts["other_res_profile_rows"] += profile_rows
                except Exception as exc:
                    LOG.warning("Skipping %s Other RES sheet: %s", workbook.name, exc)
            if other_nonres_capacity_writer is not None and other_nonres_profile_writer is not None:
                try:
                    capacity_rows, profile_rows, band_profile_rows = write_other_nonres(
                        workbook=workbook,
                        profile_writer=other_nonres_profile_writer,
                        capacity_writer=other_nonres_capacity_writer,
                        band_profile_writer=other_nonres_band_profile_writer,
                    )
                    counts["other_nonres_capacity_rows"] += capacity_rows
                    counts["other_nonres_profile_rows"] += profile_rows
                    counts["other_nonres_band_profile_rows"] += band_profile_rows
                except Exception as exc:
                    LOG.warning("Skipping %s Other Non-RES sheet: %s", workbook.name, exc)
    finally:
        for file_obj in files:
            file_obj.close()
    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    market_nodes = {str(node).strip() for node in args.market_nodes} if args.market_nodes else None
    for year in args.years:
        workbooks = discover_workbooks(args.input_dir, int(year), str(args.scenario))
        if not workbooks:
            LOG.warning("No workbooks found for %s in %s", year, args.input_dir / str(year))
            continue
        counts = process_year(
            year=int(year),
            workbooks=workbooks,
            output_dir=args.output_dir,
            market_nodes=market_nodes,
            skip_other_res=bool(args.skip_other_res),
            skip_other_nonres=bool(args.skip_other_nonres),
            write_other_nonres_band_profiles=bool(args.write_other_nonres_band_profiles),
        )
        LOG.info("wrote Other profiles %s: %s", year, counts)


if __name__ == "__main__":
    main()
