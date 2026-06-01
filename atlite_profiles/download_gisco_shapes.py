from __future__ import annotations

"""Download and normalise GISCO administrative shapes for the pipeline.

The raster and load workflows need a stable set of country and regional
geometries. This helper retrieves GISCO data, applies the naming conventions
used by the reduced-grid country aggregation, and writes local shape files that
can be versioned or archived with the preprocessing case.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
import requests
from pandas.api.types import is_string_dtype
from requests import RequestException
import yaml

GISCO_INDEX_BASE = "https://gisco-services.ec.europa.eu/distribution/v1"
VALID_EPSG = {3035, 3857, 4326}
VALID_SCALES = {"01M", "03M", "10M", "20M", "60M"}
ISO2_ALIASES = {
    "EL": "GR",  # Greece
    "UK": "GB",  # United Kingdom
}


# Default ISO3 -> ISO2 mapping (your target list)
COUNTRIES_ISO3_TO_ISO2 = {
    "ALB": "AL",
    "AUT": "AT",
    "BEL": "BE",
    "BGR": "BG",
    "BIH": "BA",
    "CHE": "CH",
    "CZE": "CZ",
    "DEU": "DE",
    "DNK": "DK",
    "ESP": "ES",
    "EST": "EE",
    "FIN": "FI",
    "FRA": "FR",
    "GBR": "GB",
    "GRC": "GR",
    "HRV": "HR",
    "HUN": "HU",
    "IRL": "IE",
    "ITA": "IT",
    "LTU": "LT",
    "LUX": "LU",
    "LVA": "LV",
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
    "UKR": "UA",
    "XKX": "XK",
}


def _normalize_scale(scale: str) -> str:
    scale = scale.strip().upper()
    if not scale.endswith("M"):
        scale = f"{scale}M"
    if scale not in VALID_SCALES:
        raise SystemExit(f"scale must be one of {sorted(VALID_SCALES)}")
    return scale


def _load_yaml_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit("config YAML must be a mapping of keys to values")
    return data


def _normalize_config_keys(data: dict) -> dict:
    normalized: dict[str, object] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise SystemExit("config keys must be strings")
        dest = key.strip().replace("-", "_")
        if dest in normalized:
            raise SystemExit(f"duplicate config key: {dest}")
        normalized[dest] = value
    return normalized


def _config_to_argv(config: dict, parser: argparse.ArgumentParser) -> list[str]:
    action_by_dest = {
        action.dest: action
        for action in parser._actions
        if action.option_strings and action.dest != "help"
    }
    argv: list[str] = []
    for key, value in config.items():
        if key == "config":
            continue
        action = action_by_dest.get(key)
        if action is None:
            raise SystemExit(f"unknown config key: {key}")
        if isinstance(action, argparse._StoreTrueAction):
            if bool(value):
                argv.append(action.option_strings[-1])
            continue
        if isinstance(action, argparse._StoreFalseAction):
            if not bool(value):
                argv.append(action.option_strings[-1])
            continue
        if value is None:
            continue
        option = action.option_strings[-1]
        if action.nargs is None or action.nargs == 1 or action.nargs == "?":
            if isinstance(value, (list, tuple)):
                if len(value) != 1:
                    raise SystemExit(f"config key {key} expects a single value")
                value = value[0]
            argv.extend([option, str(value)])
        else:
            if isinstance(value, (list, tuple)):
                values = list(value)
            else:
                values = [value]
            if not values:
                continue
            argv.append(option)
            argv.extend(str(v) for v in values)
    return argv


def _iter_strings(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_strings(value)
    elif isinstance(obj, str):
        yield obj


def _resolve_gisco_urls(year: int, scale: str, epsg: int) -> list[str]:
    filename = f"CNTR_RG_{scale}_{year}_{epsg}.geojson"
    urls: list[str] = []

    index_url = f"{GISCO_INDEX_BASE}/countries-{year}.json"
    try:
        resp = requests.get(index_url, timeout=60)
        resp.raise_for_status()
        index = resp.json()
        for value in _iter_strings(index):
            if filename in value:
                url = value
                if not url.startswith("http"):
                    url = f"{GISCO_INDEX_BASE}/" + url.lstrip("/")
                if url not in urls:
                    urls.append(url)
    except Exception as exc:
        print(f"Warning: failed to load GISCO index {index_url}: {exc}", file=sys.stderr)

    # Fallback guesses
    fallback_bases = [
        f"{GISCO_INDEX_BASE}/countries/geojson",
        "https://gisco-services.ec.europa.eu/distribution/v2/countries/geojson",
        GISCO_INDEX_BASE,
    ]
    for base in fallback_bases:
        url = f"{base}/{filename}"
        if url not in urls:
            urls.append(url)

    return urls


def _download(urls: str | list[str], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    if isinstance(urls, str):
        urls = [urls]
    last_error: Exception | None = None
    for url in urls:
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code in {404, 410}:
                last_error = RequestException(f"{resp.status_code} for {url}")
                continue
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return dest
        except RequestException as exc:
            last_error = exc
            continue
    raise SystemExit(f"Failed to download GISCO file. Tried: {urls}. Last error: {last_error}")


def _read_geojson(path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


def _pick_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {c.lower(): c for c in columns}
    for cand in candidates:
        key = cand.lower()
        if key in lookup:
            return lookup[key]
    return None


def _load_country_map(path: Path) -> dict[str, str]:
    # Expect columns iso3, iso2 (comma or semicolon separated)
    df = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower(): c for c in df.columns}
    if "iso3" not in cols or "iso2" not in cols:
        raise SystemExit("country map csv must have columns: iso3, iso2")
    return dict(zip(df[cols["iso3"]].str.upper(), df[cols["iso2"]].str.upper()))


def _load_reductions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if not {"country_code", "country_label", "member_countries"}.issubset(df.columns):
        raise SystemExit(
            "country reductions csv must have columns: country_code, country_label, member_countries"
        )
    return df


def _ensure_crs(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    if gdf.crs.to_epsg() != epsg:
        gdf = gdf.to_crs(epsg=epsg)
    return gdf


def _add_northern_ireland(
    countries: gpd.GeoDataFrame,
    ni_geom,
    iso2_col: str,
    iso3_col: str | None,
) -> gpd.GeoDataFrame:
    if ni_geom.is_empty:
        return countries

    # Add NI row
    ni_row = {
        iso2_col: "NI",
        "name": "Northern Ireland",
        "geometry": ni_geom,
    }
    if iso3_col:
        ni_row[iso3_col] = "NI"

    countries = countries.copy()
    countries = pd.concat(
        [countries, gpd.GeoDataFrame([ni_row], crs=countries.crs)], ignore_index=True
    )

    # Subtract NI from GB if present
    if iso2_col in countries.columns:
        gb_mask = countries[iso2_col] == "GB"
        if gb_mask.any():
            countries.loc[gb_mask, "geometry"] = countries.loc[gb_mask, "geometry"].difference(
                ni_geom
            )

    return countries


def _load_ni_geometry_from_nuts3(
    path: Path, epsg: int, field: str, prefixes: list[str]
):
    nuts3 = _read_geojson(path)
    nuts3 = _ensure_crs(nuts3, epsg)
    if field not in nuts3.columns:
        raise SystemExit(f"NUTS3 field '{field}' not found in {path}")
    values = nuts3[field].astype(str)
    mask = values.str.startswith(tuple(prefixes))
    subset = nuts3[mask]
    if subset.empty:
        raise SystemExit(
            f"No NUTS3 features matched prefixes {prefixes} in field '{field}'."
        )
    return _union_all(subset)


def _load_kosovo_geometry(
    path: Path, epsg: int, field: str | None, value: str | None
):
    gdf = _read_geojson(path)
    gdf = _ensure_crs(gdf, epsg)

    tokens = [value] if value else ["KOSOVO", "KOS", "XKX", "XK"]
    tokens = [t for t in tokens if t]

    if field:
        if field not in gdf.columns:
            raise SystemExit(f"Kosovo field '{field}' not found in {path}")
        series = gdf[field].astype(str)
        mask = pd.Series(False, index=gdf.index)
        for token in tokens:
            mask |= series.str.upper().str.contains(token.upper())
    else:
        mask = pd.Series(False, index=gdf.index)
        for col in gdf.columns:
            if is_string_dtype(gdf[col]):
                series = gdf[col].astype(str)
                for token in tokens:
                    mask |= series.str.upper().str.contains(token.upper())

    subset = gdf[mask]
    if subset.empty:
        raise SystemExit("No Kosovo features found in provided dataset")
    return _union_all(subset)


def _normalize_iso2(series: pd.Series) -> pd.Series:
    series = series.astype(str).str.upper()
    return series.replace(ISO2_ALIASES)


def _union_all(gdf: gpd.GeoDataFrame):
    if hasattr(gdf, "union_all"):
        return gdf.union_all()
    return gdf.unary_union


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and build GISCO country shapes.")
    parser.add_argument("--config", type=Path, help="YAML config file with options.")
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--scale", type=str, default="10M")
    parser.add_argument("--epsg", type=int, default=3035)
    parser.add_argument(
        "--country-map-csv",
        type=Path,
        help="Optional CSV with iso3, iso2 columns to override defaults.",
    )
    parser.add_argument(
        "--reductions-csv",
        type=Path,
        default=Path("country_reductions.csv"),
        help="Optional reductions CSV (country_code,country_label,member_countries).",
    )
    parser.add_argument(
        "--include-northern-ireland",
        action="store_true",
        help="If set, add NI as separate shape and subtract from GB.",
    )
    parser.add_argument(
        "--ni-nuts3-geojson",
        type=Path,
        help="Optional NUTS3 GeoJSON to derive NI (e.g. via UKN* codes).",
    )
    parser.add_argument(
        "--ni-nuts3-field",
        type=str,
        default="index",
        help="Field in NUTS3 data used for prefix matching (default: index).",
    )
    parser.add_argument(
        "--ni-nuts3-prefix",
        type=str,
        default="UKN",
        help="Comma-separated prefix list for NI extraction (default: UKN).",
    )
    parser.add_argument(
        "--include-kosovo",
        action="store_true",
        help="If set, add Kosovo (XK) from an external dataset and optionally subtract from RS.",
    )
    parser.add_argument(
        "--kosovo-source",
        type=Path,
        help="Path to vector dataset containing Kosovo geometry (e.g. Natural Earth .shp).",
    )
    parser.add_argument(
        "--kosovo-field",
        type=str,
        help="Optional field name to select Kosovo from kosovo_source.",
    )
    parser.add_argument(
        "--kosovo-value",
        type=str,
        help="Optional field value (substring match) for Kosovo selection.",
    )
    parser.add_argument(
        "--kosovo-subtract-from",
        type=str,
        default="RS",
        help="ISO2 code to subtract Kosovo from (default: RS).",
    )
    parser.add_argument(
        "--countries-out",
        type=Path,
        default=Path("country_shapes.geojson"),
    )
    parser.add_argument(
        "--europe-out",
        type=Path,
        default=Path("europe_shape.geojson"),
    )
    parser.add_argument(
        "--reductions-out",
        type=Path,
        default=Path("country_reductions.geojson"),
    )
    return parser


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args()
    parser = _build_parser()
    config_args: list[str] = []
    if pre_args.config:
        config = _load_yaml_config(pre_args.config)
        config = _normalize_config_keys(config)
        config_args = _config_to_argv(config, parser)
    return parser.parse_args(config_args + sys.argv[1:])


def main() -> None:
    args = parse_args()

    scale = _normalize_scale(args.scale)
    epsg = args.epsg
    if epsg <= 0:
        raise SystemExit("epsg must be a positive integer")

    if args.country_map_csv and args.country_map_csv.exists():
        iso3_to_iso2 = _load_country_map(args.country_map_csv)
    else:
        iso3_to_iso2 = dict(COUNTRIES_ISO3_TO_ISO2)

    iso2_target = sorted(set(iso3_to_iso2.values()))

    if epsg in VALID_EPSG:
        urls = _resolve_gisco_urls(args.year, scale, epsg)
        cache_path = args.out_dir / "cache" / Path(urls[0]).name
        source_path = _download(urls, cache_path)
        gdf = _read_geojson(source_path)
    else:
        # Fallback: download EPSG:4326 then reproject
        urls = _resolve_gisco_urls(args.year, scale, 4326)
        cache_path = args.out_dir / "cache" / Path(urls[0]).name
        source_path = _download(urls, cache_path)
        gdf = _read_geojson(source_path)
        gdf = _ensure_crs(gdf, 4326).to_crs(epsg=epsg)

    iso2_col = _pick_column(gdf.columns, ["CNTR_ID", "ISO2", "ISO_2", "ISO2_CODE"])
    iso3_col = _pick_column(gdf.columns, ["ISO3", "ISO_3", "ISO_A3", "ISO3_CODE"])

    if iso2_col is None and iso3_col is None:
        raise SystemExit(
            "Could not find ISO columns in GISCO data. Available columns: "
            + ", ".join(map(str, gdf.columns))
        )

    if iso2_col:
        gdf[iso2_col] = gdf[iso2_col].str.upper()
        gdf["iso2_norm"] = _normalize_iso2(gdf[iso2_col])
        countries = gdf[gdf["iso2_norm"].isin(iso2_target)].copy()
    else:
        gdf[iso3_col] = gdf[iso3_col].str.upper()
        iso3_target = sorted(iso3_to_iso2.keys())
        countries = gdf[gdf[iso3_col].isin(iso3_target)].copy()

    if countries.empty:
        raise SystemExit("No countries matched the target list.")

    # Add explicit iso2/iso3 columns for convenience
    if "iso2" not in countries.columns:
        if iso2_col:
            countries["iso2"] = countries["iso2_norm"]
        else:
            countries["iso2"] = countries[iso3_col].map(iso3_to_iso2)
    if "iso3" not in countries.columns:
        if iso3_col:
            countries["iso3"] = countries[iso3_col]
        else:
            iso2_to_iso3 = {v: k for k, v in iso3_to_iso2.items()}
            countries["iso3"] = countries["iso2"].map(iso2_to_iso3)

    if args.include_northern_ireland:
        if not (args.ni_nuts3_geojson and args.ni_nuts3_geojson.exists()):
            raise SystemExit(
                "include_northern_ireland is set, but ni_nuts3_geojson is missing."
            )
        prefixes = [
            p.strip().upper()
            for p in str(args.ni_nuts3_prefix).split(",")
            if p.strip()
        ]
        ni_geom = _load_ni_geometry_from_nuts3(
            args.ni_nuts3_geojson, epsg, args.ni_nuts3_field, prefixes
        )
        countries = _add_northern_ireland(countries, ni_geom, "iso2", "iso3")

    if args.include_kosovo:
        if not (args.kosovo_source and args.kosovo_source.exists()):
            raise SystemExit(
                "include_kosovo is set, but kosovo_source is missing."
            )
        kosovo_geom = _load_kosovo_geometry(
            args.kosovo_source, epsg, args.kosovo_field, args.kosovo_value
        )
        # Add XK row
        kosovo_row = {
            "iso2": "XK",
            "iso3": "XKX",
            "name": "Kosovo",
            "geometry": kosovo_geom,
        }
        countries = pd.concat(
            [countries, gpd.GeoDataFrame([kosovo_row], crs=countries.crs)],
            ignore_index=True,
        )
        # Subtract from RS if requested
        if args.kosovo_subtract_from:
            target = args.kosovo_subtract_from.upper()
            if "iso2" in countries.columns:
                mask = countries["iso2"] == target
                if mask.any():
                    countries.loc[mask, "geometry"] = countries.loc[mask, "geometry"].difference(
                        kosovo_geom
                    )

    # Warn about missing targets
    present = set(countries["iso2"].unique())
    missing = sorted(set(iso2_target) - present)
    if missing:
        print(f"Warning: missing ISO2 in GISCO data: {missing}", file=sys.stderr)

    # Ensure CRS
    countries = _ensure_crs(countries, epsg)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    countries_out = out_dir / args.countries_out
    countries.to_file(countries_out, driver="GeoJSON")

    # Europe shape (dissolve)
    europe_geom = _union_all(countries)
    europe = gpd.GeoDataFrame(
        [{"name": "Europe", "geometry": europe_geom}],
        crs=countries.crs,
    )
    europe_out = out_dir / args.europe_out
    europe.to_file(europe_out, driver="GeoJSON")

    # Reductions
    if args.reductions_csv and args.reductions_csv.exists():
        red = _load_reductions(args.reductions_csv)
        records = []
        for _, row in red.iterrows():
            members = [c.strip() for c in str(row.member_countries).split(",") if c.strip()]
            if not members:
                continue
            subset = countries[countries["iso2"].isin(members)]
            if subset.empty:
                continue
            geom = _union_all(subset)
            records.append(
                {
                    "country_code": row.country_code,
                    "country_label": row.country_label,
                    "member_countries": row.member_countries,
                    "geometry": geom,
                }
            )
        if records:
            red_gdf = gpd.GeoDataFrame(records, crs=countries.crs)
            reductions_out = out_dir / args.reductions_out
            red_gdf.to_file(reductions_out, driver="GeoJSON")


if __name__ == "__main__":
    main()
