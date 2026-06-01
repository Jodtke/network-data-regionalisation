from __future__ import annotations

"""Shared utilities for renewable capacity and profile regionalisation.

The renewable modules use the same country aggregation, bus lookup, offshore
filtering, unit conversion, and lightweight YAML parsing routines. Keeping these
operations here avoids small but consequential differences between capacity
allocation, generation scaling, and diagnostic scripts. Most helpers are kept
side-effect free so that they can be reused in publication checks and sensitivity
runs.
"""

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry import MultiPoint
from shapely.ops import voronoi_diagram

LOG = logging.getLogger(__name__)

DEFAULT_PROJECT_ROOT = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf")
DEFAULT_START_YEAR = 1982
DEFAULT_END_YEAR = 2016
SRC_CRS = "EPSG:4326"
WORK_CRS = "EPSG:3035"

COUNTRY_ALIASES = {
    "UK": "GB",
    "NIR": "NI",
}

COUNTRY_NAME_TO_ISO2 = {
    "ALBANIA": "AL",
    "ALGERIA": "DZ",
    "AUSTRIA": "AT",
    "BELGIUM": "BE",
    "BOSNIA AND HERZEGOVINA": "BA",
    "BULGARIA": "BG",
    "CROATIA": "HR",
    "CYPRUS": "CY",
    "CZECH REPUBLIC": "CZ",
    "CZECHIA": "CZ",
    "DENMARK": "DK",
    "EGYPT": "EG",
    "ESTONIA": "EE",
    "FINLAND": "FI",
    "FRANCE": "FR",
    "GERMANY": "DE",
    "GREECE": "GR",
    "HUNGARY": "HU",
    "IRELAND": "IE",
    "ISRAEL": "IL",
    "ITALY": "IT",
    "KOSOVO": "XK",
    "LATVIA": "LV",
    "LIBYA": "LY",
    "LITHUANIA": "LT",
    "LUXEMBOURG": "LU",
    "MOLDOVA": "MD",
    "MONTENEGRO": "ME",
    "MOROCCO": "MA",
    "NETHERLANDS": "NL",
    "NORTH MACEDONIA": "MK",
    "NORWAY": "NO",
    "PALESTINE": "PS",
    "POLAND": "PL",
    "PORTUGAL": "PT",
    "ROMANIA": "RO",
    "SERBIA": "RS",
    "SLOVAKIA": "SK",
    "SLOVENIA": "SI",
    "SPAIN": "ES",
    "SWEDEN": "SE",
    "SWITZERLAND": "CH",
    "TUNISIA": "TN",
    "TURKEY": "TR",
    "UKRAINE": "UA",
    "UNITED KINGDOM": "GB",
}

ISO3_TO_ISO2 = {
    "ALB": "AL",
    "AUT": "AT",
    "BEL": "BE",
    "BGR": "BG",
    "BIH": "BA",
    "CHE": "CH",
    "CYP": "CY",
    "CZE": "CZ",
    "DEU": "DE",
    "DNK": "DK",
    "DZA": "DZ",
    "EGY": "EG",
    "ESP": "ES",
    "EST": "EE",
    "FIN": "FI",
    "FRA": "FR",
    "GBR": "GB",
    "GRC": "GR",
    "HRV": "HR",
    "HUN": "HU",
    "IRL": "IE",
    "ISR": "IL",
    "ITA": "IT",
    "LBY": "LY",
    "LTU": "LT",
    "LUX": "LU",
    "LVA": "LV",
    "MAR": "MA",
    "MDA": "MD",
    "MKD": "MK",
    "MNE": "ME",
    "NLD": "NL",
    "NOR": "NO",
    "POL": "PL",
    "PRT": "PT",
    "PSE": "PS",
    "ROU": "RO",
    "SRB": "RS",
    "SVK": "SK",
    "SVN": "SI",
    "SWE": "SE",
    "TUN": "TN",
    "TUR": "TR",
    "UKR": "UA",
    "XKX": "XK",
}


@dataclass(frozen=True)
class CountryClusterMap:
    source_to_target: dict[str, str]
    target_to_sources: dict[str, tuple[str, ...]]
    target_to_label: dict[str, str]


def detect_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    if not sample:
        return ","
    return ";" if sample[0].count(";") > sample[0].count(",") else ","


def resolve_path(path_value: object, base_dir: Path | None = None) -> Path | None:
    if path_value in (None, ""):
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return (base_dir or DEFAULT_PROJECT_ROOT) / path


def resolve_cli_path(path_value: object, base_dir: Path | None = None) -> Path | None:
    if path_value in (None, ""):
        return None
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir or Path.cwd()) / path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_optional_float(value: Any, name: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric value, got {value!r}.") from exc
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative, got {parsed}.")
    return parsed


def offshore_distance_filter_enabled(
    min_distance_km: float | None,
    max_distance_km: float | None,
) -> bool:
    return min_distance_km is not None or max_distance_km is not None


def compute_distance_to_shapes_km(cells: pd.DataFrame, shape_paths: list[Path]) -> pd.Series:
    if cells.empty:
        return pd.Series([], dtype=float, index=cells.index, name="offshore_distance_km")
    if not shape_paths:
        raise ValueError("At least one shape path is required for offshore distance filtering.")

    shapes = [gpd.read_file(path).to_crs(SRC_CRS) for path in shape_paths]
    land = gpd.GeoDataFrame(pd.concat(shapes, ignore_index=True), crs=SRC_CRS)
    if land.empty:
        raise ValueError(f"No geometries found in offshore distance shape files: {shape_paths}")

    land_metric = land.to_crs(WORK_CRS)
    merged_land = (
        land_metric.geometry.union_all()
        if hasattr(land_metric.geometry, "union_all")
        else land_metric.geometry.unary_union
    )
    shoreline = merged_land.boundary
    points = gpd.GeoDataFrame(
        cells[["lon", "lat"]].copy(),
        geometry=gpd.points_from_xy(cells["lon"], cells["lat"]),
        crs=SRC_CRS,
    ).to_crs(WORK_CRS)
    distances = points.geometry.distance(shoreline) / 1000.0
    return pd.Series(distances.to_numpy(float), index=cells.index, name="offshore_distance_km")


def apply_offshore_distance_filter(
    cells: pd.DataFrame,
    shape_paths: list[Path],
    *,
    min_distance_km: float | None = None,
    max_distance_km: float | None = None,
) -> pd.DataFrame:
    if not offshore_distance_filter_enabled(min_distance_km, max_distance_km):
        return cells
    if min_distance_km is not None and max_distance_km is not None and min_distance_km > max_distance_km:
        raise ValueError(
            f"min_distance_offshore_km ({min_distance_km}) must be <= max_distance_offshore_km ({max_distance_km})."
        )

    filtered = cells.copy()
    filtered["offshore_distance_km"] = compute_distance_to_shapes_km(filtered, shape_paths)
    keep = filtered["offshore_distance_km"].notna()
    if min_distance_km is not None:
        keep &= filtered["offshore_distance_km"] >= min_distance_km
    if max_distance_km is not None:
        keep &= filtered["offshore_distance_km"] <= max_distance_km

    before_cells = len(filtered)
    filtered = filtered[keep].copy()
    LOG.info(
        "applied offshore distance filter min=%s km max=%s km: kept %d/%d cells",
        min_distance_km,
        max_distance_km,
        len(filtered),
        before_cells,
    )
    return filtered


def build_simulation_case_dir(
    simulation_dir: Path,
    network_dir: Path,
    atlite_case_dir: Path | None = None,
) -> Path:
    if atlite_case_dir is not None:
        atlite_label = f"res_{Path(atlite_case_dir).name}"
        base_dir = simulation_dir.parent if simulation_dir.name.startswith("res_") else simulation_dir
        return base_dir / network_dir.parent.name / network_dir.name / atlite_label
    if simulation_dir.name.startswith("res_"):
        return simulation_dir.parent / network_dir.parent.name / network_dir.name / simulation_dir.name
    return simulation_dir / network_dir.parent.name / network_dir.name


def load_yaml_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def parse_threshold_value(value: Any) -> float | dict[str, float] | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        parsed: dict[str, float] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key or "").strip()
            if not key:
                raise ValueError(f"Threshold mapping contains an empty key: {value!r}")
            try:
                parsed[key] = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid threshold value for {key}: {raw_value!r}") from exc
        return parsed

    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass

    if "=" in text:
        parsed_assignments: dict[str, float] = {}
        for item in text.replace(";", ",").split(","):
            part = item.strip()
            if not part:
                continue
            key, sep, raw_value = part.partition("=")
            if not sep:
                parsed_assignments = {}
                break
            key = key.strip()
            if not key:
                raise ValueError(f"Threshold mapping contains an empty key: {text!r}")
            try:
                parsed_assignments[key] = float(raw_value.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid threshold value for {key}: {raw_value.strip()!r}") from exc
        if parsed_assignments:
            return parsed_assignments

    try:
        parsed_yaml = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse threshold value: {text!r}") from exc

    if isinstance(parsed_yaml, (int, float, dict)) and not isinstance(parsed_yaml, bool):
        return parse_threshold_value(parsed_yaml)
    raise ValueError(
        "Threshold value must be a number or a mapping such as "
        "'pv=0.05,onwind=0.10,offwind=0.15'."
    )


def format_threshold_value(value: Any) -> str:
    parsed = parse_threshold_value(value)
    if parsed is None:
        return ""
    if isinstance(parsed, dict):
        return ",".join(f"{key}={raw_value:g}" for key, raw_value in parsed.items())
    return f"{parsed:g}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def normalize_country_code(country: object) -> str:
    code = str(country or "").strip().upper()
    return COUNTRY_ALIASES.get(code, code)


def normalize_country_name_or_code(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    upper = text.upper()
    if len(upper) == 2:
        return normalize_country_code(upper)
    if len(upper) == 3 and upper.isalpha():
        return normalize_country_code(ISO3_TO_ISO2.get(upper, upper))

    mapped = COUNTRY_NAME_TO_ISO2.get(upper)
    if mapped:
        return normalize_country_code(mapped)

    try:
        import pycountry  # type: ignore

        match = pycountry.countries.search_fuzzy(text)[0]
        alpha2 = getattr(match, "alpha_2", "")
        if alpha2:
            return normalize_country_code(alpha2)
    except Exception:
        pass

    return normalize_country_code(text)


def empty_country_cluster_map() -> CountryClusterMap:
    return CountryClusterMap(source_to_target={}, target_to_sources={}, target_to_label={})


def split_country_members(value: object) -> list[str]:
    if value in (None, "", "NA"):
        return []
    return [normalize_country_code(part) for part in str(value).split(",") if str(part).strip()]


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
    df = pd.read_csv(path, sep=delimiter)
    country_col = "source_country" if "source_country" in df.columns else (
        "country" if "country" in df.columns else None
    )
    if country_col is None:
        return set()
    return {
        normalize_country_name_or_code(value)
        for value in df[country_col].dropna().tolist()
        if normalize_country_name_or_code(value)
    }


def derive_network_country_map(path: Path) -> CountryClusterMap:
    delimiter = detect_delimiter(path)
    source_targets: dict[str, set[str]] = defaultdict(set)
    target_to_label: dict[str, str] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None or "country" not in reader.fieldnames:
            return empty_country_cluster_map()
        for row in reader:
            target_country = normalize_country_code(row.get("country"))
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

    ni_target = (
        "NI"
        if "NI" in target_to_label
        else ("IE_NOIE" if "IE_NOIE" in target_to_label else "NI")
    )
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


def map_country_code(country: object, cluster_map: CountryClusterMap) -> str:
    country_norm = normalize_country_name_or_code(country)
    return cluster_map.source_to_target.get(country_norm, country_norm)


def sources_for_country(country_model: str, cluster_map: CountryClusterMap) -> tuple[str, ...]:
    return cluster_map.target_to_sources.get(country_model, (country_model,))


def label_for_country(country_model: str, cluster_map: CountryClusterMap) -> str:
    return cluster_map.target_to_label.get(country_model, country_model)


def load_reduced_buses(path: Path) -> pd.DataFrame:
    buses = pd.read_csv(path, sep=detect_delimiter(path))
    required = {"bus_id", "lat", "lon", "country", "country_label"}
    missing = sorted(required - set(buses.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    buses = buses.copy()
    buses["bus_id"] = buses["bus_id"].astype(str).str.strip()
    buses["country"] = buses["country"].map(normalize_country_code)
    buses["country_label"] = buses["country_label"].astype(str).str.strip()
    return buses


def first_existing_column(frame: pd.DataFrame, candidates: list[str], *, context: str) -> str:
    by_lower = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        match = by_lower.get(candidate.lower())
        if match is not None:
            return match
    raise ValueError(f"Could not find {context} column. Available columns: {list(frame.columns)}")


def load_plants(path: Path) -> pd.DataFrame:
    plants = pd.read_csv(path, sep=detect_delimiter(path), low_memory=False)
    bus_col = first_existing_column(plants, ["bus_id", "assigned_bus", "bus"], context="plant bus")
    country_col = first_existing_column(plants, ["country", "country_code", "Country"], context="plant country")
    fuel_col = first_existing_column(plants, ["Fueltype", "fuel", "carrier"], context="plant fuel")
    technology_col = first_existing_column(plants, ["Technology", "technology"], context="plant technology")
    capacity_col = first_existing_column(plants, ["Capacity", "capacity_mw", "p_nom"], context="plant capacity")
    plants = plants.copy()
    plants["bus_id"] = plants[bus_col].astype(str).str.strip()
    plants["country"] = plants[country_col].map(normalize_country_name_or_code)
    plants["Fueltype"] = plants[fuel_col].astype(str).str.strip()
    plants["Technology"] = plants[technology_col].astype(str).str.strip()
    plants["Capacity"] = pd.to_numeric(plants[capacity_col], errors="coerce").fillna(0.0)
    return plants


def aggregate_current_capacity_by_bus(plants: pd.DataFrame) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []

    solar = plants[plants["Fueltype"].str.casefold() == "solar"]
    if not solar.empty:
        outputs.append(
            solar.groupby(["bus_id", "country"], as_index=False)["Capacity"]
            .sum()
            .assign(technology="pv")
        )

    wind = plants[plants["Fueltype"].str.casefold() == "wind"].copy()
    if not wind.empty:
        wind["technology"] = wind["Technology"].str.casefold()
        onshore = wind[wind["technology"] == "onshore"]
        offshore = wind[wind["technology"] == "offshore"]
        if not onshore.empty:
            outputs.append(
                onshore.groupby(["bus_id", "country"], as_index=False)["Capacity"]
                .sum()
                .assign(technology="onwind")
            )
        if not offshore.empty:
            outputs.append(
                offshore.groupby(["bus_id", "country"], as_index=False)["Capacity"]
                .sum()
                .assign(technology="offwind")
            )

    if not outputs:
        return pd.DataFrame(columns=["bus_id", "country", "Capacity", "technology"])
    return pd.concat(outputs, ignore_index=True)


def convert_capacity_series_to_mw(series: pd.Series, unit: str) -> pd.Series:
    unit_norm = str(unit or "MW").strip().upper()
    if unit_norm == "MW":
        return series.astype(float)
    if unit_norm == "GW":
        return series.astype(float) * 1000.0
    raise ValueError(f"Unsupported capacity unit: {unit}")


def _make_valid(geoms: gpd.GeoSeries) -> gpd.GeoSeries:
    try:
        from shapely import make_valid as _make_valid_geom
    except Exception:
        _make_valid_geom = None
    if _make_valid_geom is None:
        return geoms.buffer(0)
    return geoms.apply(_make_valid_geom)


def union_shapes(paths: list[Path]) -> object:
    pieces: list[gpd.GeoSeries] = []
    for path in paths:
        gdf = gpd.read_file(path)
        if gdf.empty:
            LOG.warning("empty shape file: %s", path)
            continue
        if gdf.crs is None:
            raise ValueError(f"shapes missing CRS: {path}")
        gdf = gdf.to_crs(WORK_CRS)
        valid = _make_valid(gdf.geometry)
        valid = valid[~valid.is_empty]
        if not valid.empty:
            pieces.append(valid)
    if not pieces:
        raise ValueError("no valid geometries available for union")
    merged = gpd.GeoSeries(pd.concat(pieces, ignore_index=True), crs=WORK_CRS)
    if hasattr(merged, "union_all"):
        return merged.union_all()
    return merged.unary_union


def assign_points_to_polygons(
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    polygon_columns: list[str],
) -> gpd.GeoDataFrame:
    polygon_view = polygons[polygon_columns + ["geometry"]].copy()
    joined = gpd.sjoin(points, polygon_view, how="left", predicate="within")
    joined = joined.drop(columns=["index_right"], errors="ignore")

    missing_mask = joined[polygon_columns[0]].isna()
    if missing_mask.any():
        try:
            points_proj = joined.loc[missing_mask, ["geometry"]].to_crs(WORK_CRS)
            polygons_proj = polygon_view.to_crs(WORK_CRS)
            nearest = gpd.sjoin_nearest(
                points_proj,
                polygons_proj,
                how="left",
                distance_col="dist_m",
            )
            nearest = nearest.drop(columns=["index_right", "dist_m"], errors="ignore")
            for column in polygon_columns:
                joined.loc[missing_mask, column] = nearest[column].values
        except Exception:
            points_work = joined.loc[missing_mask, ["geometry"]].to_crs(WORK_CRS)
            polygons_work = polygon_view.to_crs(WORK_CRS)
            for idx, geom in zip(points_work.index, points_work.geometry):
                nearest_idx = polygons_work.distance(geom).idxmin()
                for column in polygon_columns:
                    joined.loc[idx, column] = polygons.iloc[nearest_idx][column]

    return joined


def assign_polygons_to_points(
    polygons: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    point_columns: list[str],
) -> gpd.GeoDataFrame:
    polygon_view = polygons.copy()
    point_view = points[point_columns + ["geometry"]].copy()
    try:
        joined = gpd.sjoin_nearest(
            polygon_view.to_crs(WORK_CRS),
            point_view.to_crs(WORK_CRS),
            how="left",
            distance_col="dist_m",
        )
        joined = joined.drop(columns=["index_right", "dist_m"], errors="ignore")
        joined = joined.to_crs(polygons.crs)
        return joined
    except Exception:
        polygons_work = polygon_view.to_crs(WORK_CRS)
        points_work = point_view.to_crs(WORK_CRS)
        for idx, geom in zip(polygons_work.index, polygons_work.geometry):
            nearest_idx = points_work.distance(geom).idxmin()
            for column in point_columns:
                polygon_view.loc[idx, column] = point_view.loc[nearest_idx, column]
        return polygon_view


def build_onshore_voronoi(buses: pd.DataFrame, mask_shapes: list[Path]) -> gpd.GeoDataFrame:
    required = {"bus_id", "lat", "lon", "country"}
    missing = sorted(required - set(buses.columns))
    if missing:
        raise ValueError(f"Buses DataFrame is missing required columns: {missing}")

    points = gpd.GeoDataFrame(
        buses[["bus_id", "lat", "lon", "country"]].copy(),
        geometry=gpd.points_from_xy(buses["lon"], buses["lat"]),
        crs=SRC_CRS,
    ).to_crs(WORK_CRS)

    mask_union = union_shapes(mask_shapes)
    within = points.geometry.intersects(mask_union)
    points = points.loc[within].reset_index(drop=True)
    if points.empty:
        raise ValueError("no buses intersect the onshore mask")

    multi = MultiPoint(points.geometry.values)
    vor = voronoi_diagram(multi, envelope=mask_union, edges=False)
    polygons = gpd.GeoDataFrame(geometry=list(vor.geoms), crs=WORK_CRS)
    polygons["geometry"] = polygons.geometry.intersection(mask_union)
    polygons = polygons[~polygons.geometry.is_empty].reset_index(drop=True)

    assigned = assign_polygons_to_points(
        polygons,
        points.rename(columns={"bus_id": "name", "lon": "x", "lat": "y"}),
        ["name", "x", "y", "country"],
    )
    assigned = assigned[["name", "x", "y", "country", "geometry"]].copy()
    return assigned.to_crs(SRC_CRS)


def write_onshore_voronoi(path: Path, gdf: gpd.GeoDataFrame) -> None:
    ensure_dir(path.parent)
    gdf.to_file(path, driver="GeoJSON")


def build_bus_lookup(buses: pd.DataFrame) -> pd.DataFrame:
    lookup = buses[["bus_id", "country", "country_label", "lat", "lon"]].copy()
    lookup = lookup.sort_values("bus_id").reset_index(drop=True)
    lookup["bus_index"] = lookup.index.astype(int)
    return lookup[["bus_index", "bus_id", "country", "country_label", "lat", "lon"]]
