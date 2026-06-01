from __future__ import annotations

"""Convert TYNDP availability profiles for residual technologies to buses.

Other RES and other non-RES capacities may carry national availability or
profile information rather than site-specific time series. This script maps the
extracted TYNDP profiles to the bus capacities created by the disaggregation
modules, preserving technology labels and target-year diagnostics for later OPF
input assembly.
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
DEFAULT_OTHERS_INPUT_DIR = Path(r"C:\Users\jr8037\bwSyncShare\Dissertation\DATA\raw\others\tyndp2024")
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
        elif "," in value and current_key in {"countries", "technologies"}:
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
    parser = argparse.ArgumentParser(
        description="Map Other RES and Other non-RES national hourly availability to reduced-grid buses."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--network-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-year", type=int, default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--countries", nargs="*", default=None)
    parser.add_argument("--others-input-dir", type=Path, default=None)
    parser.add_argument("--other-res-availability-csv", type=Path, default=None)
    parser.add_argument("--other-nonres-availability-csv", type=Path, default=None)
    parser.add_argument("--load-shares-csv", type=Path, default=None)
    parser.add_argument("--skip-other-res", action="store_true")
    parser.add_argument("--skip-other-nonres", action="store_true")
    return parser.parse_args()


def default_settings() -> dict[str, Any]:
    return {
        "project_root": str(DEFAULT_PROJECT_ROOT),
        "network_dir": None,
        "output_dir": None,
        "target_year": None,
        "scenario": DEFAULT_SCENARIO,
        "countries": None,
        "others_input_dir": str(DEFAULT_OTHERS_INPUT_DIR),
        "other_res_availability_csv": None,
        "other_nonres_availability_csv": None,
        "load_shares_csv": None,
        "skip_other_res": False,
        "skip_other_nonres": False,
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
        return ROOT_DIR / "powerplants" / f"target_year_{target_year}" / "other_availability"
    return project_root / "powerplants" / network_dir.parent.name / network_dir.name / "other_availability"


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
        "others_input_dir": args.others_input_dir,
        "other_res_availability_csv": args.other_res_availability_csv,
        "other_nonres_availability_csv": args.other_nonres_availability_csv,
        "load_shares_csv": args.load_shares_csv,
        "skip_other_res": args.skip_other_res if args.skip_other_res else None,
        "skip_other_nonres": args.skip_other_nonres if args.skip_other_nonres else None,
    }.items():
        if value is not None:
            settings[key] = value

    project_root = resolve_path(settings.get("project_root"), config_dir) or DEFAULT_PROJECT_ROOT
    network_dir = resolve_path(settings.get("network_dir"), project_root)
    target_year = infer_target_year(network_dir, int(settings["target_year"]) if settings.get("target_year") else None)
    if target_year not in TARGET_YEARS:
        raise ValueError(f"Unsupported target_year {target_year}; expected one of {TARGET_YEARS}.")
    others_input_dir = resolve_path(settings.get("others_input_dir"), project_root) or DEFAULT_OTHERS_INPUT_DIR

    settings["project_root"] = project_root
    settings["network_dir"] = network_dir
    settings["target_year"] = target_year
    settings["others_input_dir"] = others_input_dir
    settings["other_res_availability_csv"] = (
        resolve_path(settings.get("other_res_availability_csv"), others_input_dir)
        or others_input_dir / f"other_res_availability_{target_year}_tyndp2024.csv"
    )
    settings["other_nonres_availability_csv"] = (
        resolve_path(settings.get("other_nonres_availability_csv"), others_input_dir)
        or others_input_dir / f"other_nonres_availability_{target_year}_tyndp2024.csv"
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


def load_and_aggregate_profile(
    path: Path,
    *,
    settings: Mapping[str, Any],
    load_shares: pd.DataFrame,
    country_map: Mapping[str, str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing availability profile: {path}")
    profile = read_csv_auto(path)
    required = {"country", "market_node", "year", "scenario", "technology", "date", "hour", "timestep", "installed_capacity_mw", "available_capacity_mw"}
    missing = required - set(profile.columns)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")
    profile = map_source_to_model_country(profile, country_map)
    profile["year"] = pd.to_numeric(profile["year"], errors="coerce").astype("Int64")
    profile = profile[profile["year"].eq(int(settings["target_year"]))].copy()
    requested = str(settings.get("scenario") or "").strip().casefold()
    scenario_key = profile["scenario"].astype(str).str.strip().str.casefold()
    if requested and scenario_key.eq(requested).any():
        profile = profile[scenario_key.eq(requested)].copy()
    countries = set(norm_country(country) for country in (settings.get("countries") or load_shares["country"].unique().tolist()))
    profile = profile[profile["country"].isin(countries)].copy()
    if profile.empty:
        return pd.DataFrame(
            columns=[
                "country",
                "year",
                "scenario",
                "technology",
                "date",
                "hour",
                "timestep",
                "country_installed_capacity_mw",
                "country_available_capacity_mw",
                "availability_factor",
                "source_market_nodes",
            ]
        )

    profile["installed_capacity_mw"] = pd.to_numeric(profile["installed_capacity_mw"], errors="coerce").fillna(0.0)
    profile["available_capacity_mw"] = pd.to_numeric(profile["available_capacity_mw"], errors="coerce").fillna(0.0)
    profile = profile[profile["installed_capacity_mw"] > 0.0].copy()
    group_cols = ["country", "year", "scenario", "technology", "date", "hour", "timestep"]
    grouped = (
        profile.groupby(group_cols, as_index=False)
        .agg(
            country_installed_capacity_mw=("installed_capacity_mw", "sum"),
            country_available_capacity_mw=("available_capacity_mw", "sum"),
            source_market_nodes=("market_node", lambda values: ",".join(sorted(set(str(value) for value in values)))),
        )
        .sort_values(["country", "technology", "timestep"])
        .reset_index(drop=True)
    )
    grouped["availability_factor"] = np.divide(
        grouped["country_available_capacity_mw"],
        grouped["country_installed_capacity_mw"],
        out=np.zeros(len(grouped), dtype=float),
        where=grouped["country_installed_capacity_mw"].to_numpy(dtype=float) > 0.0,
    )
    return grouped


def map_profile_to_buses(profile: pd.DataFrame, load_shares: pd.DataFrame) -> pd.DataFrame:
    if profile.empty:
        return pd.DataFrame()
    out = profile.merge(load_shares, how="inner", on="country", validate="many_to_many")
    out = out[
        [
            "year",
            "scenario",
            "country",
            "bus_id",
            "technology",
            "date",
            "hour",
            "timestep",
            "availability_factor",
            "load_share",
            "country_installed_capacity_mw",
            "country_available_capacity_mw",
            "source_market_nodes",
        ]
    ].copy()
    return out.sort_values(["country", "bus_id", "technology", "timestep"]).reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = resolve_settings(parse_args())
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if settings.get("load_shares_csv") is None or not Path(settings["load_shares_csv"]).exists():
        raise FileNotFoundError("load_shares_csv is required for bus availability mapping.")

    load_shares = load_load_shares(Path(settings["load_shares_csv"]))
    country_map = source_country_mapping_from_load_shares(Path(settings["load_shares_csv"]))
    outputs: dict[str, Path] = {}
    frames: list[pd.DataFrame] = []
    row_counts: dict[str, int] = {}

    if not bool(settings.get("skip_other_res", False)):
        res_profile = load_and_aggregate_profile(
            Path(settings["other_res_availability_csv"]),
            settings=settings,
            load_shares=load_shares,
            country_map=country_map,
        )
        res_bus = map_profile_to_buses(res_profile, load_shares)
        outputs["other_res_availability_country_bus"] = output_dir / "other_res_availability_country_bus.csv"
        write_csv(outputs["other_res_availability_country_bus"], res_bus)
        row_counts["other_res_availability_country_bus"] = int(len(res_bus))
        frames.append(res_bus)
        LOG.info("wrote Other RES bus availability rows: %s", len(res_bus))

    if not bool(settings.get("skip_other_nonres", False)):
        nonres_profile = load_and_aggregate_profile(
            Path(settings["other_nonres_availability_csv"]),
            settings=settings,
            load_shares=load_shares,
            country_map=country_map,
        )
        nonres_bus = map_profile_to_buses(nonres_profile, load_shares)
        outputs["other_nonres_availability_country_bus"] = output_dir / "other_nonres_availability_country_bus.csv"
        write_csv(outputs["other_nonres_availability_country_bus"], nonres_bus)
        row_counts["other_nonres_availability_country_bus"] = int(len(nonres_bus))
        frames.append(nonres_bus)
        LOG.info("wrote Other non-RES bus availability rows: %s", len(nonres_bus))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    outputs["other_availability_country_bus"] = output_dir / "other_availability_country_bus.csv"
    write_csv(outputs["other_availability_country_bus"], combined)
    row_counts["other_availability_country_bus"] = int(len(combined))
    outputs["manifest"] = output_dir / "other_availability_bus_profiles_manifest.json"
    write_json(
        outputs["manifest"],
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "settings": {str(key): str(value) if isinstance(value, Path) else value for key, value in settings.items()},
            "outputs": {key: str(value) for key, value in outputs.items()},
            "rows": row_counts,
        },
    )
    LOG.info("wrote Other availability bus outputs to %s", output_dir)


if __name__ == "__main__":
    main()
