from __future__ import annotations

"""Allocate TYNDP demand-side-response capacities to reduced-grid buses.

DSR is linked to demand rather than to plant locations. The module therefore
uses the static load shares from the load disaggregation as the spatial basis
for both installed DSR capacity and hourly available capacity. Price-band
structure is preserved at the national level and copied to the bus level after
scaling, so that downstream optimisation can retain the TYNDP flexibility
cost structure.
"""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


LOG = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_ROOT = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf")
DEFAULT_DSR_INPUT_DIR = Path(r"C:\Users\jr8037\bwSyncShare\Dissertation\DATA\raw\dsr\tyndp2024")
DEFAULT_SCENARIO = "NationalTrends"
TARGET_YEARS = (2030, 2040, 2050)
COUNTRY_ALIASES = {
    "UK": "GB",
    "GBR": "GB",
    "UNITED KINGDOM": "GB",
    "GREAT BRITAIN": "GB",
    "NORTHERN IRELAND": "NI",
    "NORTH IRELAND": "NI",
    "NIR": "NI",
    "UKRAINE": "UA",
    "UKR": "UA",
    "MOLDOVA": "MD",
    "MDA": "MD",
    "EL": "GR",
}
PROTECTED_COUNTRY_CODES = {"NI", "UA", "MD"}


def norm_country(value: Any) -> str:
    code = re.sub(r"[\s_\-]+", " ", str(value or "").strip().upper())
    if code in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[code]
    compact = re.sub(r"[^A-Z0-9]+", "", code)
    return COUNTRY_ALIASES.get(compact, code)


def detect_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    if not sample:
        return ";"
    return ";" if sample[0].count(";") >= sample[0].count(",") else ","


def read_csv_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=detect_delimiter(path), low_memory=False).rename(columns=str.strip)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, sep=";")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def simple_yaml_config(path: Path) -> dict[str, Any]:
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
        elif "," in value and current_key in {"countries"}:
            data[current_key] = [item.strip() for item in value.split(",") if item.strip()]
        else:
            data[current_key] = value
    return data


def resolve_path(value: Any, base_dir: Path | None = None) -> Path | None:
    if value in (None, ""):
        return None
    text = str(value)
    if base_dir is not None:
        text = text.format(project_root=str(base_dir))
    path = Path(text)
    if path.is_absolute():
        return path
    return (base_dir or Path.cwd()) / path


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
    else:
        raise KeyError(f"{path} does not contain a recognized load-share schema.")
    out["country"] = out["country"].map(norm_country)
    out["bus_id"] = out["bus_id"].astype(str)
    out["load_share"] = pd.to_numeric(out["load_share"], errors="coerce").fillna(0.0)
    out = out[out["load_share"] > 0.0].groupby(["country", "bus_id"], as_index=False)["load_share"].sum()
    total = out.groupby("country")["load_share"].transform("sum")
    out["load_share"] = np.divide(
        out["load_share"],
        total,
        out=np.zeros(len(out), dtype=float),
        where=total.to_numpy(dtype=float) > 0.0,
    )
    return out


def source_country_mapping_from_load_shares(path: Path) -> dict[str, str]:
    df = read_csv_auto(path)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Disaggregate TYNDP 2024 DSR price bands to reduced-grid buses.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--network-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-year", type=int, default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--countries", nargs="*", default=None)
    parser.add_argument("--dsr-input-dir", type=Path, default=None)
    parser.add_argument("--price-bands-csv", type=Path, default=None)
    parser.add_argument("--availability-csv", type=Path, default=None)
    parser.add_argument("--load-shares-csv", type=Path, default=None)
    parser.add_argument("--skip-timeseries", action="store_true")
    parser.add_argument("--chunksize", type=int, default=250_000)
    return parser.parse_args()


def default_settings() -> dict[str, Any]:
    return {
        "project_root": str(DEFAULT_PROJECT_ROOT),
        "network_dir": None,
        "output_dir": None,
        "target_year": None,
        "scenario": DEFAULT_SCENARIO,
        "countries": None,
        "dsr_input_dir": str(DEFAULT_DSR_INPUT_DIR),
        "price_bands_csv": None,
        "availability_csv": None,
        "load_shares_csv": None,
        "skip_timeseries": False,
        "chunksize": 250_000,
    }


def infer_target_year(network_dir: Path | None, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    if network_dir is not None:
        match = re.search(r"target_year_(\d{4})", str(network_dir))
        if match:
            return int(match.group(1))
    raise ValueError("target_year is required or must be inferable from network_dir.")


def default_output_dir(project_root: Path, network_dir: Path | None, target_year: int) -> Path:
    if network_dir is None:
        return ROOT_DIR / "dsr" / f"target_year_{target_year}" / "dsr_disaggregation"
    return project_root / "dsr" / network_dir.parent.name / network_dir.name


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = default_settings()
    config_dir = None
    if args.config is not None:
        config_path = resolve_path(args.config, Path.cwd())
        assert config_path is not None
        settings.update(simple_yaml_config(config_path))
        config_dir = config_path.parent

    for key, value in {
        "project_root": args.project_root,
        "network_dir": args.network_dir,
        "output_dir": args.output_dir,
        "target_year": args.target_year,
        "scenario": args.scenario,
        "countries": args.countries,
        "dsr_input_dir": args.dsr_input_dir,
        "price_bands_csv": args.price_bands_csv,
        "availability_csv": args.availability_csv,
        "load_shares_csv": args.load_shares_csv,
        "skip_timeseries": args.skip_timeseries if args.skip_timeseries else None,
        "chunksize": args.chunksize,
    }.items():
        if value is not None:
            settings[key] = value

    project_root = resolve_path(settings.get("project_root"), config_dir) or DEFAULT_PROJECT_ROOT
    network_dir = resolve_path(settings.get("network_dir"), project_root)
    target_year = infer_target_year(network_dir, int(settings["target_year"]) if settings.get("target_year") else None)
    if target_year not in TARGET_YEARS:
        raise ValueError(f"Unsupported target_year {target_year}; expected one of {TARGET_YEARS}.")
    dsr_input_dir = resolve_path(settings.get("dsr_input_dir"), project_root) or DEFAULT_DSR_INPUT_DIR

    settings["project_root"] = project_root
    settings["network_dir"] = network_dir
    settings["target_year"] = target_year
    settings["dsr_input_dir"] = dsr_input_dir
    settings["price_bands_csv"] = (
        resolve_path(settings.get("price_bands_csv"), dsr_input_dir)
        or dsr_input_dir / f"dsr_price_bands_{target_year}_tyndp2024.csv"
    )
    settings["availability_csv"] = (
        resolve_path(settings.get("availability_csv"), dsr_input_dir)
        or dsr_input_dir / f"dsr_available_capacity_{target_year}_tyndp2024.csv"
    )
    settings["load_shares_csv"] = resolve_path(settings.get("load_shares_csv"), project_root)
    settings["output_dir"] = (
        resolve_path(settings.get("output_dir"), project_root)
        or default_output_dir(project_root, network_dir, target_year)
    )
    return settings


def map_source_to_model_country(frame: pd.DataFrame, country_map: Mapping[str, str]) -> pd.DataFrame:
    out = frame.copy()
    out["source_country"] = out["country"].map(norm_country)
    out["country"] = out["source_country"].map(
        lambda country: country if country in PROTECTED_COUNTRY_CODES else country_map.get(country, country)
    )
    return out


def load_price_bands(settings: Mapping[str, Any], load_shares: pd.DataFrame, country_map: Mapping[str, str]) -> pd.DataFrame:
    path = Path(settings["price_bands_csv"])
    if not path.exists():
        raise FileNotFoundError(f"Missing DSR price-band file: {path}")
    bands = read_csv_auto(path)
    bands = map_source_to_model_country(bands, country_map)
    bands["year"] = pd.to_numeric(bands["year"], errors="coerce").astype("Int64")
    bands = bands[bands["year"].eq(int(settings["target_year"]))].copy()
    if "scenario" in bands.columns:
        requested = str(settings.get("scenario") or "").strip().casefold()
        scenario_key = bands["scenario"].astype(str).str.strip().str.casefold()
        if requested and scenario_key.eq(requested).any():
            bands = bands[scenario_key.eq(requested)].copy()
    countries = settings.get("countries") or sorted(load_shares["country"].unique().tolist())
    countries = [norm_country(country) for country in countries]
    bands = bands[bands["country"].isin(set(countries))].copy()
    for column in ["installed_capacity_mw", "units", "activation_hours", "price_eur_mwh"]:
        if column in bands.columns:
            bands[column] = pd.to_numeric(bands[column], errors="coerce").fillna(0.0)
    bands["source_band_uid"] = bands["market_node"].astype(str) + "|" + bands["band_id"].astype(str)
    return bands.sort_values(["country", "market_node", "band_id"]).reset_index(drop=True)


def build_bus_price_bands(bands: pd.DataFrame, load_shares: pd.DataFrame) -> pd.DataFrame:
    bus = bands.merge(load_shares, how="inner", on="country", validate="many_to_many")
    bus["capacity_share"] = bus["load_share"]
    # Price bands stay intact; only their capacity is split spatially. This keeps
    # the economic merit order from the TYNDP DSR data unchanged.
    bus["source_installed_capacity_mw"] = pd.to_numeric(bus["installed_capacity_mw"], errors="coerce").fillna(0.0)
    bus["installed_capacity_mw"] = bus["source_installed_capacity_mw"] * bus["capacity_share"]
    if "units" in bus.columns:
        bus["source_units"] = pd.to_numeric(bus["units"], errors="coerce").fillna(0.0)
        bus["units"] = bus["source_units"] * bus["capacity_share"]
    bus["technology"] = "dsr"
    bus["unit_id"] = [
        f"dsr|{row.country}|{row.bus_id}|{row.market_node}|{row.band_id}"
        for row in bus[["country", "bus_id", "market_node", "band_id"]].itertuples(index=False)
    ]
    keep = [
        column
        for column in [
            "year",
            "scenario",
            "country",
            "source_country",
            "market_node",
            "bus_id",
            "source_band_uid",
            "band_id",
            "band_label",
            "price_eur_mwh",
            "activation_hours",
            "climate_year_start",
            "climate_year_end",
            "capacity_share",
            "installed_capacity_mw",
            "source_installed_capacity_mw",
            "units",
            "source_units",
            "technology",
            "unit_id",
        ]
        if column in bus.columns
    ]
    return bus[keep].sort_values(["country", "bus_id", "market_node", "band_id"]).reset_index(drop=True)


def write_timeseries(settings: Mapping[str, Any], load_shares: pd.DataFrame, country_map: Mapping[str, str], output_path: Path) -> int:
    source_path = Path(settings["availability_csv"])
    if not source_path.exists():
        LOG.warning("Skipping DSR time-series disaggregation; missing %s", source_path)
        return 0
    countries = set(norm_country(country) for country in (settings.get("countries") or load_shares["country"].unique().tolist()))
    delimiter = detect_delimiter(source_path)
    wrote_header = False
    total_rows = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Availability can be large, so the national profiles are streamed and
    # multiplied by load shares chunk by chunk.
    for chunk in pd.read_csv(source_path, sep=delimiter, chunksize=int(settings.get("chunksize") or 250_000), low_memory=False):
        chunk = chunk.rename(columns=str.strip)
        chunk = map_source_to_model_country(chunk, country_map)
        chunk["year"] = pd.to_numeric(chunk["year"], errors="coerce").astype("Int64")
        chunk = chunk[chunk["year"].eq(int(settings["target_year"])) & chunk["country"].isin(countries)].copy()
        if chunk.empty:
            continue
        if "scenario" in chunk.columns:
            requested = str(settings.get("scenario") or "").strip().casefold()
            scenario_key = chunk["scenario"].astype(str).str.strip().str.casefold()
            if requested and scenario_key.eq(requested).any():
                chunk = chunk[scenario_key.eq(requested)].copy()
        if chunk.empty:
            continue
        for column in ["installed_capacity_mw", "available_capacity_mw", "availability_factor", "price_eur_mwh"]:
            if column in chunk.columns:
                chunk[column] = pd.to_numeric(chunk[column], errors="coerce").fillna(0.0)
        chunk["source_available_capacity_mw"] = chunk["available_capacity_mw"]
        chunk["source_installed_capacity_mw"] = chunk["installed_capacity_mw"]
        chunk = chunk.merge(load_shares, how="inner", on="country", validate="many_to_many")
        if chunk.empty:
            continue
        chunk["capacity_share"] = chunk["load_share"]
        chunk["available_capacity_mw"] = chunk["source_available_capacity_mw"] * chunk["capacity_share"]
        chunk["installed_capacity_mw"] = chunk["source_installed_capacity_mw"] * chunk["capacity_share"]
        chunk["source_band_uid"] = chunk["market_node"].astype(str) + "|" + chunk["band_id"].astype(str)
        keep = [
            "year",
            "scenario",
            "country",
            "source_country",
            "market_node",
            "bus_id",
            "source_band_uid",
            "band_id",
            "band_label",
            "price_eur_mwh",
            "date",
            "hour",
            "timestep",
            "capacity_share",
            "installed_capacity_mw",
            "available_capacity_mw",
            "availability_factor",
            "source_installed_capacity_mw",
            "source_available_capacity_mw",
        ]
        out = chunk[[column for column in keep if column in chunk.columns]].copy()
        out.to_csv(output_path, mode="a", index=False, sep=";", header=not wrote_header)
        wrote_header = True
        total_rows += len(out)
    return total_rows


def build_diagnostics(bands: pd.DataFrame, bus_bands: pd.DataFrame) -> pd.DataFrame:
    source = (
        bands.groupby("country", as_index=False)["installed_capacity_mw"]
        .sum()
        .rename(columns={"installed_capacity_mw": "source_installed_capacity_mw"})
    )
    bus = (
        bus_bands.groupby("country", as_index=False)["installed_capacity_mw"]
        .sum()
        .rename(columns={"installed_capacity_mw": "bus_installed_capacity_mw"})
    )
    out = source.merge(bus, how="outer", on="country").fillna(0.0)
    out["difference_mw"] = out["source_installed_capacity_mw"] - out["bus_installed_capacity_mw"]
    out["n_source_bands"] = bands.groupby("country")["source_band_uid"].nunique().reindex(out["country"]).to_numpy()
    out["n_bus_rows"] = bus_bands.groupby("country").size().reindex(out["country"]).fillna(0).astype(int).to_numpy()
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = resolve_settings(parse_args())
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if settings.get("load_shares_csv") is None or not Path(settings["load_shares_csv"]).exists():
        raise FileNotFoundError("load_shares_csv is required for DSR disaggregation.")

    load_shares = load_load_shares(Path(settings["load_shares_csv"]))
    country_map = source_country_mapping_from_load_shares(Path(settings["load_shares_csv"]))
    bands = load_price_bands(settings, load_shares, country_map)
    bus_bands = build_bus_price_bands(bands, load_shares)
    diagnostics = build_diagnostics(bands, bus_bands)

    outputs = {
        "dsr_price_bands_model_country": output_dir / "dsr_price_bands_model_country.csv",
        "dsr_capacity_country_bus": output_dir / "dsr_capacity_country_bus.csv",
        "dsr_allocation_diagnostics": output_dir / "dsr_allocation_diagnostics.csv",
        "dsr_available_capacity_country_bus": output_dir / "dsr_available_capacity_country_bus.csv",
        "manifest": output_dir / "dsr_disaggregation_manifest.json",
    }
    write_csv(outputs["dsr_price_bands_model_country"], bands)
    write_csv(outputs["dsr_capacity_country_bus"], bus_bands)
    write_csv(outputs["dsr_allocation_diagnostics"], diagnostics)

    timeseries_rows = 0
    if not bool(settings.get("skip_timeseries", False)):
        timeseries_rows = write_timeseries(settings, load_shares, country_map, outputs["dsr_available_capacity_country_bus"])

    write_json(
        outputs["manifest"],
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "settings": {str(key): str(value) if isinstance(value, Path) else value for key, value in settings.items()},
            "outputs": {key: str(value) for key, value in outputs.items()},
            "source_installed_capacity_mw": float(bands["installed_capacity_mw"].sum()) if not bands.empty else 0.0,
            "bus_installed_capacity_mw": float(bus_bands["installed_capacity_mw"].sum()) if not bus_bands.empty else 0.0,
            "timeseries_rows": int(timeseries_rows),
        },
    )
    LOG.info("wrote DSR disaggregation outputs to %s", output_dir)


if __name__ == "__main__":
    main()
