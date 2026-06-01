from __future__ import annotations

"""Extract TYNDP 2024 demand-side-response tables into tidy CSV files.

The downstream DSR regionalisation expects one row per country, product, price
band, and time step. This script isolates the workbook-specific parsing from
the allocation logic, making later changes in TYNDP sheet layout easier to
audit.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
COMMON_DIR = ROOT_DIR / "powerplants"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from tyndp2024_excel import format_number, iter_hourly_rows, read_sheet, safe_float, safe_int


LOG = logging.getLogger(__name__)
DEFAULT_PEMMDB_DIR = Path(r"Y:\Data\TYNDP2024\raw\PEMMDB2")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\jr8037\bwSyncShare\Dissertation\DATA\raw\dsr\tyndp2024")
TARGET_YEARS = (2030, 2040, 2050)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract TYNDP 2024 DSR price bands and hourly availability from PEMMDB2 workbooks."
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
    parser.add_argument(
        "--skip-timeseries",
        action="store_true",
        help="Only write price-band metadata, not hourly availability.",
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


def parse_bands(sheet: Any, *, source_file: Path) -> list[dict[str, Any]]:
    country = sheet.cell(3, 2).strip()
    market_node = sheet.cell(4, 2).strip()
    year = safe_int(sheet.cell(5, 2))
    scenario = sheet.cell(7, 2).strip()
    candidate_columns = sheet.nonempty_columns(range(8, 15), min_column=3)
    bands: list[dict[str, Any]] = []
    for ordinal, column in enumerate(candidate_columns, start=1):
        capacity = safe_float(sheet.cell(9, column), 0.0) or 0.0
        price = safe_float(sheet.cell(12, column))
        band_label = sheet.cell(8, column).strip() or f"Price Band {ordinal}"
        if not band_label and capacity <= 0.0 and price is None:
            continue
        bands.append(
            {
                "country": country,
                "market_node": market_node,
                "year": year,
                "scenario": scenario,
                "band_id": f"pb{ordinal:03d}",
                "band_label": band_label,
                "band_column": column,
                "installed_capacity_mw": capacity,
                "units": safe_float(sheet.cell(10, column)),
                "activation_hours": safe_float(sheet.cell(11, column)),
                "price_eur_mwh": price,
                "climate_year_start": safe_int(sheet.cell(13, column)),
                "climate_year_end": safe_int(sheet.cell(14, column)),
                "source_file": str(source_file),
            }
        )
    return bands


def write_dsr_year(
    *,
    year: int,
    workbooks: list[Path],
    output_dir: Path,
    market_nodes: set[str] | None,
    skip_timeseries: bool,
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    band_path = output_dir / f"dsr_price_bands_{year}_tyndp2024.csv"
    profile_path = output_dir / f"dsr_available_capacity_{year}_tyndp2024.csv"

    band_header = [
        "country",
        "market_node",
        "year",
        "scenario",
        "band_id",
        "band_label",
        "band_column",
        "installed_capacity_mw",
        "units",
        "activation_hours",
        "price_eur_mwh",
        "climate_year_start",
        "climate_year_end",
        "source_file",
    ]
    profile_header = [
        "country",
        "market_node",
        "year",
        "scenario",
        "band_id",
        "band_label",
        "price_eur_mwh",
        "installed_capacity_mw",
        "date",
        "hour",
        "timestep",
        "available_capacity_mw",
        "availability_factor",
    ]

    band_count = 0
    profile_count = 0
    with band_path.open("w", newline="", encoding="utf-8") as band_file:
        band_writer = csv.DictWriter(band_file, fieldnames=band_header, delimiter=";", lineterminator="\n")
        band_writer.writeheader()
        profile_file = None
        profile_writer = None
        if not skip_timeseries:
            profile_file = profile_path.open("w", newline="", encoding="utf-8")
            profile_writer = csv.DictWriter(profile_file, fieldnames=profile_header, delimiter=";", lineterminator="\n")
            profile_writer.writeheader()
        try:
            for workbook in workbooks:
                filename_market_node = workbook.name.split("_")[1] if "_" in workbook.name else ""
                if market_nodes is not None and filename_market_node not in market_nodes:
                    continue
                try:
                    sheet = read_sheet(workbook, "DSR")
                except Exception as exc:
                    LOG.warning("Skipping %s DSR sheet: %s", workbook.name, exc)
                    continue
                market_node = sheet.cell(4, 2).strip()
                if market_nodes is not None and market_node not in market_nodes:
                    continue
                bands = parse_bands(sheet, source_file=workbook)
                if not bands:
                    continue
                for band in bands:
                    band_writer.writerow({key: format_number(value) for key, value in band.items()})
                band_count += len(bands)
                if skip_timeseries or profile_writer is None:
                    continue

                for timestep, date, hour, values in iter_hourly_rows(sheet):
                    for band in bands:
                        available = safe_float(values.get(int(band["band_column"])), 0.0) or 0.0
                        installed = float(band["installed_capacity_mw"] or 0.0)
                        availability_factor = available / installed if installed > 0.0 else 0.0
                        profile_writer.writerow(
                            {
                                "country": band["country"],
                                "market_node": band["market_node"],
                                "year": year,
                                "scenario": band["scenario"],
                                "band_id": band["band_id"],
                                "band_label": band["band_label"],
                                "price_eur_mwh": format_number(band["price_eur_mwh"]),
                                "installed_capacity_mw": format_number(installed),
                                "date": date,
                                "hour": hour,
                                "timestep": timestep,
                                "available_capacity_mw": format_number(available),
                                "availability_factor": format_number(availability_factor),
                            }
                        )
                        profile_count += 1
        finally:
            if profile_file is not None:
                profile_file.close()
    return band_count, profile_count


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    market_nodes = {str(node).strip() for node in args.market_nodes} if args.market_nodes else None
    for year in args.years:
        workbooks = discover_workbooks(args.input_dir, int(year), str(args.scenario))
        if not workbooks:
            LOG.warning("No workbooks found for %s in %s", year, args.input_dir / str(year))
            continue
        band_count, profile_count = write_dsr_year(
            year=int(year),
            workbooks=workbooks,
            output_dir=args.output_dir,
            market_nodes=market_nodes,
            skip_timeseries=bool(args.skip_timeseries),
        )
        LOG.info(
            "wrote DSR %s: %s price-band rows, %s hourly profile rows",
            year,
            band_count,
            profile_count,
        )


if __name__ == "__main__":
    main()
