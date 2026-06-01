from __future__ import annotations

"""Disaggregate national load time series to reduced network buses.

The load workflow follows the static regionalisation idea used in PyPSA-Eur, but
updates the socioeconomic input data and keeps the country aggregation logic used
elsewhere in the preprocessing pipeline. Population and GDP proxy regions are
weighted by distance to eligible AC buses; the resulting static bus shares are
then applied to national TYNDP load time series for each weather year.

The shares are also written as a reusable siting basis for downstream modules
such as DSR, thermal fallback allocation, and other non-RES fallback allocation.
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from configs.pipeline_config import deep_merge, load_yaml_like, resolve_path
except ModuleNotFoundError:
    from grid.pipeline_config import deep_merge, load_yaml_like, resolve_path


EARTH_RADIUS_KM = 6371.0
EPSILON_KM = 1e-6
SHARE_DECIMALS = 3
COUNTRY_ALIASES = {
    "UK": "GB",
    "NIR": "NI",
}
DEFAULT_PROJECT_ROOT = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf")
DEFAULT_NETWORK_DIR = (
    DEFAULT_PROJECT_ROOT
    / "grid"
    / "target_year_2030"
    / "electrical_spectral_line_equivalent_dc_effective_reactance"
)
TYNDP_TARGET_YEARS = (2030, 2040, 2050)
TYNDP_YEAR_PATTERN = re.compile(r"(?<!\d)(?:2030|2040|2050)(?!\d)")


@dataclass(frozen=True)
class OriginalBus:
    bus_id: str
    cluster_id: str
    country: str
    lat: float
    lon: float
    lat_rad: float
    lon_rad: float


@dataclass(frozen=True)
class RegionProxy:
    region_id: str
    country: str
    lat: float
    lon: float
    lat_rad: float
    lon_rad: float
    pop: float
    gdp: float
    source: str


@dataclass(frozen=True)
class CountryClusterMap:
    source_to_target: dict[str, str]
    target_to_sources: dict[str, tuple[str, ...]]
    target_to_label: dict[str, str]


@dataclass(frozen=True)
class AggregatedLoadRow:
    country_model: str
    source_countries: tuple[str, ...]
    timestamp: str
    weather_year: str
    week: str
    national_load_mw: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Disaggregate national load time series to reduced network buses."
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML scenario config.")
    parser.add_argument("--network-dir", type=Path, default=None)
    parser.add_argument("--target-year", type=int, default=None)
    parser.add_argument("--load-csv", type=Path, default=None)
    parser.add_argument("--load-column", default=None)
    parser.add_argument("--reduced-buses-csv", type=Path, default=None)
    parser.add_argument("--buses-with-clusters-csv", type=Path, default=None)
    parser.add_argument("--country-clusters-csv", type=Path, default=None)
    parser.add_argument("--excluded-countries-csv", type=Path, default=None)
    parser.add_argument("--nuts3-geojson", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--shares-output-csv", type=Path, default=None)
    parser.add_argument("--population-weight", type=float, default=None)
    parser.add_argument("--gdp-weight", type=float, default=None)
    parser.add_argument("--distance-alpha", type=float, default=None)
    parser.add_argument(
        "--skip-missing-countries",
        dest="skip_missing_countries",
        action="store_true",
        help="Skip model-country rows without bus shares.",
    )
    parser.add_argument(
        "--no-skip-missing-countries",
        dest="skip_missing_countries",
        action="store_false",
        help="Raise on missing model-country shares.",
    )
    parser.set_defaults(skip_missing_countries=None)
    return parser.parse_args()


def default_settings() -> dict[str, Any]:
    return {
        "scenario_name": "2030_load_disaggregation",
        "project_root": str(DEFAULT_PROJECT_ROOT),
        "network_dir": str(DEFAULT_NETWORK_DIR),
        "target_year": None,
        "load_csv": str(DEFAULT_PROJECT_ROOT / "load" / "res_load_country_long_2030_tyndp2024.csv"),
        "load_column": "load",
        "reduced_buses_csv": None,
        "buses_with_clusters_csv": None,
        "country_clusters_csv": None,
        "excluded_countries_csv": None,
        "nuts3_geojson": str(DEFAULT_PROJECT_ROOT / "datashapes" / "nuts3_shapes_pop2021_gdp2024.geojson"),
        "output_dir": None,
        "output_csv": None,
        "shares_output_csv": None,
        "population_weight": 0.4,
        "gdp_weight": 0.6,
        "distance_alpha": 1.0,
        "skip_missing_countries": False,
    }


def normalize_country_code(country: str) -> str:
    code = str(country).strip().upper()
    return COUNTRY_ALIASES.get(code, code)


def parse_float(value: object, default: float = 0.0) -> float:
    if value in (None, "", "NA"):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace(",", "."))


def parse_optional_int(value: object) -> int | None:
    if value in (None, "", "NA"):
        return None
    try:
        return int(round(float(str(value).strip().replace(",", "."))))
    except ValueError:
        return None


def validate_tyndp_target_year(target_year: int) -> int:
    year = int(target_year)
    if year not in TYNDP_TARGET_YEARS:
        raise ValueError(
            f"Unsupported TYNDP target year {year}. Expected one of: "
            + ", ".join(str(value) for value in TYNDP_TARGET_YEARS)
        )
    return year


def infer_target_year(network_dir: Path | None, explicit: int | None = None) -> int | None:
    if explicit is not None:
        return int(explicit)
    if network_dir is None:
        return None
    match = re.search(r"target_year_(\d{4})", str(network_dir))
    return int(match.group(1)) if match else None


def detect_tyndp_years_in_path(path: Path) -> set[int]:
    return {int(match.group(0)) for match in TYNDP_YEAR_PATTERN.finditer(str(path))}


def validate_path_target_year(path: Path, target_year: int) -> None:
    year = validate_tyndp_target_year(target_year)
    path_years = detect_tyndp_years_in_path(path)
    if not path_years:
        raise ValueError(
            f"Load input has no target-year column and its path does not contain one of "
            f"{list(TYNDP_TARGET_YEARS)}: {path}"
        )
    if path_years != {year}:
        raise ValueError(
            f"Load input path points to TYNDP year(s) {sorted(path_years)}, "
            f"but target_year is {year}: {path}"
        )


def find_tyndp_target_year_column(
    by_lower: dict[str, str],
    *,
    weather_year_col: str | None,
) -> str | None:
    for key in ("target_year", "ref_year", "reference_year", "scenario_year"):
        column = by_lower.get(key)
        if column is not None:
            return column
    generic_year = by_lower.get("year")
    if generic_year is not None and generic_year != weather_year_col:
        return generic_year
    return None


def parse_timestamp(value: str) -> datetime:
    timestamp = str(value).strip()
    if not timestamp:
        raise ValueError("Timestamp must not be empty.")

    normalized = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(timestamp, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unsupported timestamp format: {timestamp}")


def derive_weather_year_and_week(timestamp: str) -> tuple[str, str]:
    ts = parse_timestamp(timestamp)
    weather_year = ts.year
    week = ((ts.timetuple().tm_yday - 1) // 7) + 1
    return str(weather_year), str(week)


def rounded_share_row(row: dict[str, object]) -> dict[str, object]:
    out = dict(row)
    for key in ("load_share", "gdp_share", "population_share"):
        if key in out:
            out[key] = round(float(out[key]), SHARE_DECIMALS)
    return out


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()

    counts = {sep: header.count(sep) for sep in (";", "\t", ",")}
    delimiter, count = max(counts.items(), key=lambda item: item[1])
    return delimiter if count > 0 else ";"


def build_default_output_dir(project_root: Path, network_dir: Path) -> Path:
    return project_root / "load" / network_dir.parent.name / network_dir.name


def weight_suffix(value: float) -> str:
    return str(int(round(value * 100)))


def build_output_names(settings: dict[str, Any]) -> tuple[str, str]:
    pop_tag = weight_suffix(settings["population_weight"])
    gdp_tag = weight_suffix(settings["gdp_weight"])
    stem = f"{settings['load_column']}_pop{pop_tag}_gdp{gdp_tag}"
    return (
        f"disaggregated_load_country_bus_{stem}.csv",
        f"disaggregated_load_country_bus_shares_{stem}.csv",
    )


def resolve_runtime_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = default_settings()
    config_path = Path(args.config).resolve() if args.config is not None else None

    if config_path is not None:
        settings = deep_merge(settings, load_yaml_like(config_path))

    cli_overrides = {
        key: value
        for key, value in {
            "network_dir": args.network_dir,
            "target_year": args.target_year,
            "load_csv": args.load_csv,
            "load_column": args.load_column,
            "reduced_buses_csv": args.reduced_buses_csv,
            "buses_with_clusters_csv": args.buses_with_clusters_csv,
            "country_clusters_csv": args.country_clusters_csv,
            "excluded_countries_csv": getattr(args, "excluded_countries_csv", None),
            "nuts3_geojson": args.nuts3_geojson,
            "output_dir": args.output_dir,
            "output_csv": args.output_csv,
            "shares_output_csv": args.shares_output_csv,
            "population_weight": args.population_weight,
            "gdp_weight": args.gdp_weight,
            "distance_alpha": args.distance_alpha,
            "skip_missing_countries": args.skip_missing_countries,
        }.items()
        if value is not None
    }
    settings = deep_merge(settings, cli_overrides)

    base_dir = config_path.parent if config_path is not None else None
    project_root = resolve_path(settings["project_root"], base_dir=base_dir)
    if project_root is None:
        raise ValueError("project_root could not be resolved.")

    settings["project_root"] = project_root
    settings["config_path"] = config_path
    settings["scenario_name"] = str(
        settings.get("scenario_name")
        or (config_path.stem if config_path is not None else "load_disaggregation")
    )

    network_dir = resolve_path(settings.get("network_dir"), base_dir=base_dir)
    if network_dir is None:
        raise ValueError("network_dir must be set either via config or CLI.")
    settings["network_dir"] = network_dir
    target_year = infer_target_year(network_dir, settings.get("target_year"))
    if target_year is None:
        raise ValueError("Could not infer target_year from network_dir. Set target_year explicitly.")
    settings["target_year"] = validate_tyndp_target_year(target_year)

    settings["load_csv"] = resolve_path(settings["load_csv"], base_dir=base_dir)
    settings["nuts3_geojson"] = resolve_path(settings["nuts3_geojson"], base_dir=base_dir)

    settings["reduced_buses_csv"] = resolve_path(settings.get("reduced_buses_csv"), base_dir=base_dir)
    if settings["reduced_buses_csv"] is None:
        settings["reduced_buses_csv"] = network_dir / "buses.csv"

    settings["buses_with_clusters_csv"] = resolve_path(
        settings.get("buses_with_clusters_csv"),
        base_dir=base_dir,
    )
    if settings["buses_with_clusters_csv"] is None:
        settings["buses_with_clusters_csv"] = network_dir / "buses_with_clusters.csv"

    settings["country_clusters_csv"] = resolve_path(
        settings.get("country_clusters_csv"),
        base_dir=base_dir,
    )
    if settings["country_clusters_csv"] is None:
        candidate = network_dir / "cesa_country_clusters.csv"
        settings["country_clusters_csv"] = candidate if candidate.exists() else None

    settings["excluded_countries_csv"] = resolve_path(
        settings.get("excluded_countries_csv"),
        base_dir=base_dir,
    )
    if settings["excluded_countries_csv"] is None:
        candidate = network_dir / "excluded_countries.csv"
        settings["excluded_countries_csv"] = candidate if candidate.exists() else None

    settings["output_dir"] = resolve_path(settings.get("output_dir"), base_dir=base_dir)
    if settings["output_dir"] is None:
        settings["output_dir"] = build_default_output_dir(project_root, network_dir)

    output_name, shares_name = build_output_names(settings)
    settings["output_csv"] = resolve_path(settings.get("output_csv"), base_dir=base_dir)
    if settings["output_csv"] is None:
        settings["output_csv"] = settings["output_dir"] / output_name

    settings["shares_output_csv"] = resolve_path(settings.get("shares_output_csv"), base_dir=base_dir)
    if settings["shares_output_csv"] is None:
        settings["shares_output_csv"] = settings["output_dir"] / shares_name

    return settings


def empty_country_cluster_map() -> CountryClusterMap:
    return CountryClusterMap(source_to_target={}, target_to_sources={}, target_to_label={})


def map_country_code(country: str, cluster_map: CountryClusterMap) -> str:
    country_norm = normalize_country_code(country)
    return cluster_map.source_to_target.get(country_norm, country_norm)


def northern_ireland_model_country(cluster_map: CountryClusterMap) -> str:
    if "NI" in cluster_map.target_to_label:
        return "NI"
    if "IE_NOIE" in cluster_map.target_to_label:
        return "IE_NOIE"
    return "NI"


def sources_for_country(country_model: str, cluster_map: CountryClusterMap) -> tuple[str, ...]:
    return cluster_map.target_to_sources.get(country_model, (country_model,))


def label_for_country(country_model: str, cluster_map: CountryClusterMap) -> str:
    return cluster_map.target_to_label.get(country_model, country_model)


def read_country_cluster_map(path: Path | None) -> CountryClusterMap:
    if path is None or not path.exists():
        return empty_country_cluster_map()

    delimiter = detect_delimiter(path)
    source_to_target: dict[str, str] = {}
    target_to_sources: dict[str, set[str]] = defaultdict(set)
    target_to_label: dict[str, str] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        required = {"source_country", "target_country"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Country clusters CSV must contain source_country and target_country columns."
            )

        for row in reader:
            source_country = normalize_country_code(row["source_country"])
            target_country = normalize_country_code(row["target_country"])
            target_label = str(row.get("target_label") or target_country).strip() or target_country

            if source_country in source_to_target and source_to_target[source_country] != target_country:
                raise ValueError(
                    f"Source country {source_country} maps to multiple targets in {path}."
                )

            source_to_target[source_country] = target_country
            target_to_sources[target_country].add(source_country)
            target_to_label[target_country] = target_label

    return CountryClusterMap(
        source_to_target=source_to_target,
        target_to_sources={key: tuple(sorted(value)) for key, value in target_to_sources.items()},
        target_to_label=target_to_label,
    )


def read_excluded_countries(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            return set()
        country_col = "source_country" if "source_country" in reader.fieldnames else (
            "country" if "country" in reader.fieldnames else None
        )
        if country_col is None:
            return set()
        return {
            normalize_country_code(row.get(country_col, ""))
            for row in reader
            if normalize_country_code(row.get(country_col, ""))
        }


def split_country_members(value: object) -> list[str]:
    if value in (None, "", "NA"):
        return []
    return [
        normalize_country_code(part)
        for part in str(value).split(",")
        if str(part).strip()
    ]


def derive_network_country_map(path: Path) -> CountryClusterMap:
    delimiter = detect_delimiter(path)
    source_targets: dict[str, set[str]] = defaultdict(set)
    target_to_label: dict[str, str] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None or "country" not in reader.fieldnames:
            return empty_country_cluster_map()

        for row in reader:
            target_country = str(row.get("country") or "").strip().upper()
            if not target_country:
                continue
            target_label = str(row.get("country_label") or target_country).strip() or target_country
            target_to_label[target_country] = target_label
            original_countries = split_country_members(row.get("original_country") or target_country)
            if not original_countries:
                original_countries = [target_country]
            for source_country in original_countries:
                source_targets[source_country].add(target_country)

    source_to_target: dict[str, str] = {}
    target_to_sources: dict[str, set[str]] = defaultdict(set)
    for source_country, targets in source_targets.items():
        if len(targets) != 1:
            continue
        target_country = next(iter(targets))
        source_to_target[source_country] = target_country
        target_to_sources[target_country].add(source_country)

    return CountryClusterMap(
        source_to_target=source_to_target,
        target_to_sources={key: tuple(sorted(value)) for key, value in target_to_sources.items()},
        target_to_label=target_to_label,
    )


def merge_country_cluster_maps(
    primary: CountryClusterMap,
    secondary: CountryClusterMap,
) -> CountryClusterMap:
    source_to_target = dict(secondary.source_to_target)
    source_to_target.update(primary.source_to_target)

    target_to_label = dict(secondary.target_to_label)
    target_to_label.update(primary.target_to_label)

    ni_target = "NI" if "NI" in target_to_label else ("IE_NOIE" if "IE_NOIE" in target_to_label else "NI")
    source_to_target["NI"] = ni_target
    source_to_target["NIR"] = ni_target
    target_to_label.setdefault(ni_target, ni_target)

    target_to_sources: dict[str, set[str]] = defaultdict(set)
    for source_country, target_country in source_to_target.items():
        target_to_sources[target_country].add(source_country)
        target_to_label.setdefault(target_country, target_country)

    return CountryClusterMap(
        source_to_target=source_to_target,
        target_to_sources={key: tuple(sorted(value)) for key, value in target_to_sources.items()},
        target_to_label=target_to_label,
    )


def infer_bus_model_country(row: dict[str, str], cluster_map: CountryClusterMap) -> str:
    source_country = normalize_country_code(str(row["country"]).strip())
    cluster_id = str(row.get("cluster_id") or "").strip().upper()
    sync_area = str(row.get("sync_area") or "").strip().upper()
    if source_country == "GB" and cluster_id == "SYNC_IE":
        return "IE_NOIE"
    if source_country == "GB" and sync_area == "IE_NOIE":
        return northern_ireland_model_country(cluster_map)
    return map_country_code(source_country, cluster_map)


def infer_proxy_model_country(props: dict[str, Any], cluster_map: CountryClusterMap) -> str:
    source_country = normalize_country_code(str(props["country"]))
    region_id = str(props.get("index") or "").strip().upper()
    if source_country == "GB" and region_id.startswith("UKN"):
        return northern_ireland_model_country(cluster_map)
    return map_country_code(source_country, cluster_map)


def load_target_bus_ids(path: Path) -> set[str]:
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None or "bus_id" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a bus_id column.")
        return {str(row["bus_id"]).strip() for row in reader if str(row["bus_id"]).strip()}


def is_ac_load_bus(row: dict[str, str]) -> bool:
    dc_bool = str(row.get("dc_bool") or "").strip().lower()
    if dc_bool in {"true", "1", "yes"}:
        return False
    if dc_bool in {"false", "0", "no"}:
        return True

    carrier = str(row.get("carrier") or "").strip().upper()
    if carrier == "HVDC":
        return False
    if carrier == "AC":
        return True

    cluster_id = str(row.get("cluster_id") or "").strip().lower()
    if cluster_id.startswith("cl_dc_"):
        return False

    return True


def load_original_buses(
    path: Path,
    valid_clusters: set[str],
    cluster_map: CountryClusterMap,
) -> dict[str, list[OriginalBus]]:
    buses_by_country: dict[str, list[OriginalBus]] = defaultdict(list)
    delimiter = detect_delimiter(path)

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        required = {"bus_id", "cluster_id", "lat", "lon", "country"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} is missing required columns: {sorted(required)}")

        for row in reader:
            cluster_id = str(row["cluster_id"]).strip()
            if cluster_id not in valid_clusters:
                continue
            if not is_ac_load_bus(row):
                continue

            country = infer_bus_model_country(row, cluster_map)
            lat = float(row["lat"])
            lon = float(row["lon"])
            buses_by_country[country].append(
                OriginalBus(
                    bus_id=str(row["bus_id"]).strip(),
                    cluster_id=cluster_id,
                    country=country,
                    lat=lat,
                    lon=lon,
                    lat_rad=math.radians(lat),
                    lon_rad=math.radians(lon),
                )
            )

    return dict(buses_by_country)


def ring_centroid(ring: list[list[float]]) -> tuple[float, float, float]:
    if len(ring) < 4:
        xs = [point[0] for point in ring]
        ys = [point[1] for point in ring]
        return 0.0, sum(xs) / len(xs), sum(ys) / len(ys)

    double_area = 0.0
    centroid_x = 0.0
    centroid_y = 0.0

    for first, second in zip(ring, ring[1:]):
        x0, y0 = first[:2]
        x1, y1 = second[:2]
        cross = x0 * y1 - x1 * y0
        double_area += cross
        centroid_x += (x0 + x1) * cross
        centroid_y += (y0 + y1) * cross

    area = 0.5 * double_area
    if abs(area) < 1e-12:
        xs = [point[0] for point in ring]
        ys = [point[1] for point in ring]
        return 0.0, sum(xs) / len(xs), sum(ys) / len(ys)

    centroid_x /= 6.0 * area
    centroid_y /= 6.0 * area
    return area, centroid_x, centroid_y


def geometry_centroid(geometry: dict[str, Any]) -> tuple[float, float]:
    geometry_type = geometry["type"]
    polygons: Iterable[list[list[list[float]]]]

    if geometry_type == "Polygon":
        polygons = [geometry["coordinates"]]
    elif geometry_type == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        raise ValueError(f"Unsupported geometry type: {geometry_type}")

    total_area = 0.0
    centroid_x_sum = 0.0
    centroid_y_sum = 0.0
    fallback_points: list[tuple[float, float]] = []

    for polygon in polygons:
        for ring in polygon:
            area, centroid_x, centroid_y = ring_centroid(ring)
            total_area += area
            centroid_x_sum += centroid_x * area
            centroid_y_sum += centroid_y * area
            fallback_points.extend((point[0], point[1]) for point in ring)

    if abs(total_area) >= 1e-12:
        lon = centroid_x_sum / total_area
        lat = centroid_y_sum / total_area
        return lat, lon

    lon = sum(point[0] for point in fallback_points) / len(fallback_points)
    lat = sum(point[1] for point in fallback_points) / len(fallback_points)
    return lat, lon


def load_nuts3_proxies(
    path: Path,
    cluster_map: CountryClusterMap,
) -> dict[str, list[RegionProxy]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    proxies_by_country: dict[str, list[RegionProxy]] = defaultdict(list)
    for feature in data["features"]:
        props = feature["properties"]
        country = infer_proxy_model_country(props, cluster_map)
        lat, lon = geometry_centroid(feature["geometry"])
        proxy_source = str(props.get("proxy_source") or props.get("source") or "nuts3")
        proxies_by_country[country].append(
            RegionProxy(
                region_id=str(props["index"]),
                country=country,
                lat=lat,
                lon=lon,
                lat_rad=math.radians(lat),
                lon_rad=math.radians(lon),
                pop=max(parse_float(props.get("pop")), 0.0),
                gdp=max(parse_float(props.get("gdp")), 0.0),
                source=proxy_source,
            )
        )

    return dict(proxies_by_country)


def build_fallback_proxy(country: str, buses: list[OriginalBus]) -> list[RegionProxy]:
    # If no external socioeconomic proxy is available, use the country bus
    # centroid as a neutral placeholder instead of silently dropping the load.
    lat = sum(bus.lat for bus in buses) / len(buses)
    lon = sum(bus.lon for bus in buses) / len(buses)
    return [
        RegionProxy(
            region_id=f"{country}_fallback",
            country=country,
            lat=lat,
            lon=lon,
            lat_rad=math.radians(lat),
            lon_rad=math.radians(lon),
            pop=1.0,
            gdp=1.0,
            source="fallback_bus_mean",
        )
    ]


def equirectangular_distance_km(
    lat1_rad: float, lon1_rad: float, lat2_rad: float, lon2_rad: float
) -> float:
    x = (lon2_rad - lon1_rad) * math.cos((lat1_rad + lat2_rad) / 2.0)
    y = lat2_rad - lat1_rad
    return EARTH_RADIUS_KM * math.sqrt(x * x + y * y)


def distance_weights(
    proxy: RegionProxy,
    buses: list[OriginalBus],
    distance_alpha: float,
) -> list[tuple[OriginalBus, float]]:
    distances: list[tuple[OriginalBus, float]] = []
    min_distance = float("inf")

    for bus in buses:
        distance = equirectangular_distance_km(
            proxy.lat_rad,
            proxy.lon_rad,
            bus.lat_rad,
            bus.lon_rad,
        )
        distances.append((bus, distance))
        min_distance = min(min_distance, distance)

    if min_distance <= EPSILON_KM:
        same_location = [(bus, dist) for bus, dist in distances if dist <= EPSILON_KM]
        weight = 1.0 / len(same_location)
        return [(bus, weight) for bus, _ in same_location]

    # Inverse-distance weights spread each proxy region over nearby reduced buses
    # while retaining a smooth response to the chosen distance exponent.
    inverse_weights = []
    total_weight = 0.0
    for bus, distance in distances:
        weight = distance ** (-distance_alpha)
        inverse_weights.append((bus, weight))
        total_weight += weight

    return [(bus, weight / total_weight) for bus, weight in inverse_weights]


def normalize_share_dict(shares: dict[str, float]) -> dict[str, float]:
    total = sum(shares.values())
    if total <= 0.0:
        raise ValueError("Encountered non-positive share sum during normalization.")
    return {key: value / total for key, value in shares.items()}


def compute_cluster_shares(
    buses_by_country: dict[str, list[OriginalBus]],
    proxies_by_country: dict[str, list[RegionProxy]],
    population_weight: float,
    gdp_weight: float,
    distance_alpha: float,
) -> dict[str, list[dict[str, object]]]:
    shares_by_country: dict[str, list[dict[str, object]]] = {}

    for country, buses in sorted(buses_by_country.items()):
        proxies = proxies_by_country.get(country)
        if not proxies:
            proxies = build_fallback_proxy(country, buses)

        pop_total = sum(proxy.pop for proxy in proxies)
        gdp_total = sum(proxy.gdp for proxy in proxies)

        if pop_total <= 0.0:
            pop_base = {proxy.region_id: 1.0 / len(proxies) for proxy in proxies}
        else:
            pop_base = {proxy.region_id: proxy.pop / pop_total for proxy in proxies}

        if gdp_total <= 0.0:
            gdp_base = {proxy.region_id: 1.0 / len(proxies) for proxy in proxies}
        else:
            gdp_base = {proxy.region_id: proxy.gdp / gdp_total for proxy in proxies}

        cluster_pop: dict[str, float] = defaultdict(float)
        cluster_gdp: dict[str, float] = defaultdict(float)
        proxy_sources = sorted({proxy.source for proxy in proxies})

        for proxy in proxies:
            for bus, dist_weight in distance_weights(proxy, buses, distance_alpha):
                cluster_pop[bus.cluster_id] += pop_base[proxy.region_id] * dist_weight
                cluster_gdp[bus.cluster_id] += gdp_base[proxy.region_id] * dist_weight

        cluster_pop = normalize_share_dict(dict(cluster_pop))
        cluster_gdp = normalize_share_dict(dict(cluster_gdp))

        # Population and GDP are kept as separate diagnostics before the final
        # weighted blend. This makes later sensitivity checks straightforward.
        combined = {
            cluster_id: population_weight * cluster_pop.get(cluster_id, 0.0)
            + gdp_weight * cluster_gdp.get(cluster_id, 0.0)
            for cluster_id in set(cluster_pop) | set(cluster_gdp)
        }
        combined = normalize_share_dict(combined)

        shares_by_country[country] = [
            {
                "country": country,
                "bus": cluster_id,
                "population_share": cluster_pop.get(cluster_id, 0.0),
                "gdp_share": cluster_gdp.get(cluster_id, 0.0),
                "load_share": combined[cluster_id],
                "proxy_region_count": len(proxies),
                "proxy_sources": ",".join(proxy_sources),
            }
            for cluster_id in sorted(combined)
        ]

    return shares_by_country


def aggregate_model_load_rows(
    path: Path,
    load_column: str,
    cluster_map: CountryClusterMap,
    target_year: int,
    excluded_source_countries: set[str] | None = None,
) -> tuple[list[AggregatedLoadRow], int]:
    validate_tyndp_target_year(target_year)
    delimiter = detect_delimiter(path)
    aggregated: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    skipped_rows = 0
    excluded_source_countries = excluded_source_countries or set()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Load CSV has no header: {path}")

        fieldnames = list(reader.fieldnames)
        by_lower = {str(name).strip().lower(): str(name) for name in fieldnames}

        country_col = by_lower.get("country")
        if country_col is None:
            raise ValueError("Load CSV is missing required column: Country")

        timestamp_col = (
            by_lower.get("timestamp")
            or by_lower.get("peak_timestamp")
        )
        weather_year_col = by_lower.get("weather_year")
        week_col = by_lower.get("week")
        target_year_col = find_tyndp_target_year_column(
            by_lower,
            weather_year_col=weather_year_col,
        )
        if target_year_col is None:
            validate_path_target_year(path, target_year)

        if timestamp_col is None and (weather_year_col is None or week_col is None):
            raise ValueError(
                "Load CSV must contain either Timestamp/peak_timestamp "
                "or weather_year and week columns."
            )

        if load_column not in fieldnames:
            raise ValueError(f"Load CSV does not contain requested column: {load_column}")

        for row in reader:
            if target_year_col is not None and parse_optional_int(row.get(target_year_col)) != int(target_year):
                continue

            raw_load = row.get(load_column)
            if raw_load in ("", "NA", None):
                skipped_rows += 1
                continue

            country_source = normalize_country_code(str(row[country_col]))
            if country_source in excluded_source_countries:
                skipped_rows += 1
                continue
            country_model = map_country_code(country_source, cluster_map)

            timestamp = str(row[timestamp_col]).strip() if timestamp_col is not None else ""
            if weather_year_col is not None and week_col is not None:
                weather_year = str(row[weather_year_col]).strip()
                week = str(row[week_col]).strip()
                if not timestamp:
                    timestamp = f"{weather_year}-W{week}"
            else:
                weather_year, week = derive_weather_year_and_week(timestamp)

            key = (country_model, timestamp, weather_year, week)

            entry = aggregated.setdefault(
                key,
                {
                    "national_load_mw": 0.0,
                    "source_countries": set(),
                },
            )
            entry["national_load_mw"] += parse_float(raw_load)
            entry["source_countries"].add(country_source)

    rows = [
        AggregatedLoadRow(
            country_model=country_model,
            source_countries=tuple(sorted(entry["source_countries"])),
            timestamp=timestamp,
            weather_year=weather_year,
            week=week,
            national_load_mw=float(entry["national_load_mw"]),
        )
        for (country_model, timestamp, weather_year, week), entry in sorted(aggregated.items())
    ]
    if not rows:
        raise ValueError(f"No load rows remain after filtering {path} to target_year={target_year}.")
    return rows, skipped_rows


def find_missing_countries(
    load_rows: list[AggregatedLoadRow],
    shares_by_country: dict[str, list[dict[str, object]]],
) -> list[str]:
    countries = {row.country_model for row in load_rows}
    return sorted(country for country in countries if country not in shares_by_country)


def write_static_shares(
    path: Path,
    shares_by_country: dict[str, list[dict[str, object]]],
    cluster_map: CountryClusterMap,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "country",
                "country_label",
                "source_countries",
                "bus",
                "load_share",
                "gdp_share",
                "population_share",
                "proxy_region_count",
                "proxy_sources",
            ],
        )
        writer.writeheader()
        for country in sorted(shares_by_country):
            source_countries = ",".join(sources_for_country(country, cluster_map))
            country_label = label_for_country(country, cluster_map)
            for row in shares_by_country[country]:
                rounded_row = rounded_share_row(row)
                writer.writerow(
                    {
                        "country": country,
                        "country_label": country_label,
                        "source_countries": source_countries,
                        **rounded_row,
                    }
                )


def write_disaggregated_load(
    load_rows: list[AggregatedLoadRow],
    output_csv: Path,
    shares_by_country: dict[str, list[dict[str, object]]],
    cluster_map: CountryClusterMap,
    skip_missing_countries: bool = False,
) -> list[str]:
    skipped_countries: set[str] = set()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "country",
                "country_model",
                "country_label",
                "source_countries",
                "bus",
                "timestamp",
                "weather_year",
                "week",
                "national_peak_load_mw",
                "allocated_load_mw",
                "load_share",
                "gdp_share",
                "population_share",
            ],
        )
        writer.writeheader()

        for row in load_rows:
            country_shares = shares_by_country.get(row.country_model)
            if country_shares is None:
                if skip_missing_countries:
                    skipped_countries.add(row.country_model)
                    continue
                raise ValueError(
                    "No bus-share definition available for model country: "
                    f"{row.country_model}"
                )

            country_label = label_for_country(row.country_model, cluster_map)
            source_countries = ",".join(row.source_countries)
            for share in country_shares:
                rounded_share = rounded_share_row(share)
                writer.writerow(
                    {
                        "country": row.country_model,
                        "country_model": row.country_model,
                        "country_label": country_label,
                        "source_countries": source_countries,
                        "bus": share["bus"],
                        "timestamp": row.timestamp,
                        "weather_year": row.weather_year,
                        "week": row.week,
                        "national_peak_load_mw": row.national_load_mw,
                        "allocated_load_mw": int(
                            round(row.national_load_mw * float(share["load_share"]))
                        ),
                        "load_share": rounded_share["load_share"],
                        "gdp_share": rounded_share["gdp_share"],
                        "population_share": rounded_share["population_share"],
                    }
                )

    return sorted(skipped_countries)


def write_manifest(
    settings: dict[str, Any],
    cluster_map: CountryClusterMap,
    shares_by_country: dict[str, list[dict[str, object]]],
    load_rows: list[AggregatedLoadRow],
) -> None:
    manifest_path = settings["output_dir"] / "disaggregation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scenario_name": settings["scenario_name"],
        "config_path": str(settings["config_path"]) if settings["config_path"] is not None else None,
        "network_dir": str(settings["network_dir"]),
        "target_year": settings["target_year"],
        "load_csv": str(settings["load_csv"]),
        "nuts3_geojson": str(settings["nuts3_geojson"]),
        "country_clusters_csv": (
            str(settings["country_clusters_csv"]) if settings["country_clusters_csv"] is not None else None
        ),
        "excluded_countries_csv": (
            str(settings["excluded_countries_csv"]) if settings.get("excluded_countries_csv") is not None else None
        ),
        "output_csv": str(settings["output_csv"]),
        "shares_output_csv": str(settings["shares_output_csv"]),
        "population_weight": settings["population_weight"],
        "gdp_weight": settings["gdp_weight"],
        "distance_alpha": settings["distance_alpha"],
        "model_countries_with_shares": sorted(shares_by_country),
        "model_countries_in_load": sorted({row.country_model for row in load_rows}),
        "country_cluster_targets": sorted(cluster_map.target_to_sources),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    settings = resolve_runtime_settings(args)

    if not math.isclose(
        settings["population_weight"] + settings["gdp_weight"],
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("Population and GDP weights must sum to 1.0.")
    if settings["distance_alpha"] <= 0.0:
        raise ValueError("distance-alpha must be positive.")

    network_cluster_map = derive_network_country_map(settings["reduced_buses_csv"])
    external_cluster_map = read_country_cluster_map(settings["country_clusters_csv"])
    cluster_map = merge_country_cluster_maps(network_cluster_map, external_cluster_map)
    excluded_source_countries = read_excluded_countries(settings.get("excluded_countries_csv"))
    target_bus_ids = load_target_bus_ids(settings["reduced_buses_csv"])
    buses_by_country = load_original_buses(
        settings["buses_with_clusters_csv"],
        target_bus_ids,
        cluster_map,
    )
    proxies_by_country = load_nuts3_proxies(settings["nuts3_geojson"], cluster_map)
    shares_by_country = compute_cluster_shares(
        buses_by_country=buses_by_country,
        proxies_by_country=proxies_by_country,
        population_weight=settings["population_weight"],
        gdp_weight=settings["gdp_weight"],
        distance_alpha=settings["distance_alpha"],
    )

    load_rows, skipped_input_rows = aggregate_model_load_rows(
        settings["load_csv"],
        settings["load_column"],
        cluster_map,
        int(settings["target_year"]),
        excluded_source_countries,
    )
    missing_countries = find_missing_countries(load_rows, shares_by_country)
    if missing_countries and not settings["skip_missing_countries"]:
        raise ValueError(
            "No bus-share definition available for model countries: "
            + ", ".join(missing_countries)
        )
    if missing_countries and settings["skip_missing_countries"]:
        print(
            "Skipping model countries without bus-share definition: "
            + ", ".join(missing_countries)
        )

    write_static_shares(settings["shares_output_csv"], shares_by_country, cluster_map)
    skipped_countries = write_disaggregated_load(
        load_rows=load_rows,
        output_csv=settings["output_csv"],
        shares_by_country=shares_by_country,
        cluster_map=cluster_map,
        skip_missing_countries=settings["skip_missing_countries"],
    )
    write_manifest(settings, cluster_map, shares_by_country, load_rows)

    if skipped_input_rows:
        print(
            f"Skipped {skipped_input_rows} rows with missing `{settings['load_column']}` "
            "values or excluded source countries."
        )
    if skipped_countries:
        print(
            "Skipped model-country rows without bus-share definition: "
            + ", ".join(skipped_countries)
        )


if __name__ == "__main__":
    main()
