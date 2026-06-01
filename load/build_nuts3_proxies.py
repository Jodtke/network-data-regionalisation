#!/usr/bin/env python3
"""Refresh NUTS3 proxy attributes from newer population and GDP datasets.

The script keeps the existing NUTS3 geometries as spatial template and updates
their `pop` and `gdp` attributes:

- population is aggregated from the Eurostat 1 km census grid (2021)
- GDP is aggregated from a selected band of the GDP raster

If a region does not receive a new value from one of the two datasets, the
legacy value from the base GeoJSON is retained for that attribute.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
from rasterio.mask import mask


POPULATION_GRID_CRS = "EPSG:3035"
GRID_ID_PATTERN = re.compile(r"N(?P<north>\d+)E(?P<east>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh NUTS3 population and GDP proxy attributes."
    )
    parser.add_argument(
        "--base-nuts3-geojson",
        type=Path,
        default=Path("pypsa-eur/resources/nuts3_shapes_pypsa.geojson"),
        help="Base NUTS3 GeoJSON whose geometries are reused.",
    )
    parser.add_argument(
        "--population-parquet",
        type=Path,
        default=Path("datashapes/ESTAT_Census_2021_V2.parquet"),
        help="Eurostat census grid parquet.",
    )
    parser.add_argument(
        "--population-column",
        default="T",
        help="Population column from the census parquet.",
    )
    parser.add_argument(
        "--gdp-raster",
        type=Path,
        default=Path("datashapes/rast_gdpTot_1990_2024_5arcmin.tif"),
        help="GDP raster with yearly bands named like `gdp_tot_2024`.",
    )
    parser.add_argument(
        "--gdp-year",
        type=int,
        default=2024,
        help="GDP year to extract from the raster.",
    )
    parser.add_argument(
        "--output-geojson",
        type=Path,
        default=None,
        help="Output GeoJSON. Defaults to a year-tagged file in pypsa-eur/resources.",
    )
    parser.add_argument(
        "--population-batch-size",
        type=int,
        default=250_000,
        help="Batch size for streaming census-grid rows from parquet.",
    )
    parser.add_argument(
        "--all-touched",
        dest="all_touched",
        action="store_true",
        help="Include all raster cells touched by a NUTS3 polygon for GDP aggregation.",
    )
    parser.add_argument(
        "--no-all-touched",
        dest="all_touched",
        action="store_false",
        help="Restrict GDP aggregation to raster cells whose centers fall inside the polygon.",
    )
    parser.set_defaults(all_touched=True)
    return parser.parse_args()


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output_geojson is not None:
        return args.output_geojson
    return Path(
        f"pypsa-eur/resources/nuts3_shapes_pop2021_gdp{args.gdp_year}.geojson"
    )


def load_base_nuts3(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    required = {"index", "country", "pop", "gdp", "geometry"}
    missing = sorted(required - set(gdf.columns))
    if missing:
        raise ValueError(
            f"Base GeoJSON is missing required columns: {', '.join(missing)}"
        )
    return gdf


def decode_grid_centroids(grid_ids: pd.Series) -> pd.DataFrame:
    coords = grid_ids.str.extract(GRID_ID_PATTERN)
    if coords.isna().any().any():
        raise ValueError("Failed to parse at least one GRD_ID from the census parquet.")
    coords = coords.astype("int64")
    coords["east"] += 500
    coords["north"] += 500
    return coords


def aggregate_population_to_nuts3(
    census_path: Path,
    nuts3_gdf: gpd.GeoDataFrame,
    population_column: str,
    batch_size: int,
) -> tuple[pd.Series, dict[str, int]]:
    parquet_file = pq.ParquetFile(census_path)
    nuts3_3035 = nuts3_gdf[["index", "geometry"]].to_crs(POPULATION_GRID_CRS)

    totals: defaultdict[str, float] = defaultdict(float)
    stats = {
        "batches": 0,
        "rows_total": 0,
        "rows_kept": 0,
        "rows_matched": 0,
        "rows_unmatched": 0,
    }

    columns = ["GRD_ID", "POPULATED", population_column]
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        stats["batches"] += 1
        df = batch.to_pandas()
        stats["rows_total"] += len(df)

        df[population_column] = pd.to_numeric(df[population_column], errors="coerce").fillna(0.0)
        populated_mask = df["POPULATED"].fillna(0).astype("int64") == 1
        df = df.loc[populated_mask, ["GRD_ID", population_column]].copy()
        if df.empty:
            continue

        stats["rows_kept"] += len(df)
        coords = decode_grid_centroids(df["GRD_ID"])
        points = gpd.GeoDataFrame(
            df[[population_column]].rename(columns={population_column: "population"}),
            geometry=gpd.points_from_xy(coords["east"], coords["north"]),
            crs=POPULATION_GRID_CRS,
        )
        joined = gpd.sjoin(points, nuts3_3035, how="left", predicate="within")
        matched = joined["index"].notna()
        stats["rows_matched"] += int(matched.sum())
        stats["rows_unmatched"] += int((~matched).sum())

        grouped = joined.loc[matched].groupby("index")["population"].sum()
        for region_id, value in grouped.items():
            totals[str(region_id)] += float(value)

    return pd.Series(totals, name="pop_new"), stats


def resolve_gdp_band(src: rasterio.DatasetReader, gdp_year: int) -> tuple[int, str]:
    band_name = f"gdp_tot_{gdp_year}"
    try:
        band_index = list(src.descriptions).index(band_name) + 1
    except ValueError as exc:
        available = ", ".join(desc for desc in src.descriptions if desc)
        raise ValueError(
            f"GDP year {gdp_year} not found in raster. Available bands: {available}"
        ) from exc
    return band_index, band_name


def aggregate_gdp_to_nuts3(
    gdp_raster: Path,
    nuts3_gdf: gpd.GeoDataFrame,
    gdp_year: int,
    all_touched: bool,
) -> tuple[pd.Series, str]:
    totals: dict[str, float] = {}

    with rasterio.open(gdp_raster) as src:
        band_index, band_name = resolve_gdp_band(src, gdp_year)
        nuts3_raster_crs = nuts3_gdf[["index", "geometry"]].to_crs(src.crs)

        for row in nuts3_raster_crs.itertuples(index=False):
            try:
                clipped, _ = mask(
                    src,
                    [row.geometry],
                    crop=True,
                    filled=False,
                    indexes=band_index,
                    all_touched=all_touched,
                )
            except ValueError:
                totals[str(row.index)] = np.nan
                continue

            total = np.ma.sum(clipped[0])
            totals[str(row.index)] = np.nan if np.ma.is_masked(total) else float(total)

    return pd.Series(totals, name="gdp_new"), band_name


def merge_proxy_values(
    base_gdf: gpd.GeoDataFrame,
    population_series: pd.Series,
    gdp_series: pd.Series,
    gdp_band_name: str,
    population_column: str,
) -> gpd.GeoDataFrame:
    refreshed = base_gdf.copy()
    refreshed = refreshed.merge(
        population_series.rename_axis("index").reset_index(),
        on="index",
        how="left",
    )
    refreshed = refreshed.merge(
        gdp_series.rename_axis("index").reset_index(),
        on="index",
        how="left",
    )

    refreshed["legacy_pop"] = pd.to_numeric(refreshed["pop"], errors="coerce")
    refreshed["legacy_gdp"] = pd.to_numeric(refreshed["gdp"], errors="coerce")

    refreshed["used_legacy_pop"] = refreshed["pop_new"].isna()
    refreshed["used_legacy_gdp"] = refreshed["gdp_new"].isna()
    refreshed["population_source"] = np.where(
        refreshed["used_legacy_pop"],
        "legacy_nuts3",
        f"ESTAT_Census_2021:{population_column}",
    )
    refreshed["gdp_source"] = np.where(
        refreshed["used_legacy_gdp"],
        "legacy_nuts3",
        gdp_band_name,
    )
    refreshed["proxy_source"] = (
        "pop="
        + refreshed["population_source"].astype(str)
        + ";gdp="
        + refreshed["gdp_source"].astype(str)
    )

    refreshed["pop"] = refreshed["pop_new"].where(
        ~refreshed["used_legacy_pop"], refreshed["legacy_pop"]
    )
    refreshed["gdp"] = refreshed["gdp_new"].where(
        ~refreshed["used_legacy_gdp"], refreshed["legacy_gdp"]
    )

    refreshed["pop"] = refreshed["pop"].astype(float)
    refreshed["gdp"] = refreshed["gdp"].astype(float)
    return refreshed


def print_summary(
    refreshed: gpd.GeoDataFrame,
    population_stats: dict[str, int],
    output_path: Path,
) -> None:
    pop_fallback_regions = int(refreshed["used_legacy_pop"].sum())
    gdp_fallback_regions = int(refreshed["used_legacy_gdp"].sum())
    pop_fallback_countries = sorted(
        refreshed.loc[refreshed["used_legacy_pop"], "country"].drop_duplicates().tolist()
    )
    gdp_fallback_countries = sorted(
        refreshed.loc[refreshed["used_legacy_gdp"], "country"].drop_duplicates().tolist()
    )

    print(f"Wrote refreshed NUTS3 proxy file to {output_path}")
    print(f"Regions: {len(refreshed)}")
    print(
        "Population batches/rows kept/matched/unmatched: "
        f"{population_stats['batches']}/"
        f"{population_stats['rows_kept']}/"
        f"{population_stats['rows_matched']}/"
        f"{population_stats['rows_unmatched']}"
    )
    print(
        "Regions using legacy population fallback: "
        f"{pop_fallback_regions}"
        + (
            f" ({', '.join(pop_fallback_countries)})"
            if pop_fallback_countries
            else ""
        )
    )
    print(
        "Regions using legacy GDP fallback: "
        f"{gdp_fallback_regions}"
        + (
            f" ({', '.join(gdp_fallback_countries)})"
            if gdp_fallback_countries
            else ""
        )
    )


def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args)

    base_gdf = load_base_nuts3(args.base_nuts3_geojson)
    population_series, population_stats = aggregate_population_to_nuts3(
        census_path=args.population_parquet,
        nuts3_gdf=base_gdf,
        population_column=args.population_column,
        batch_size=args.population_batch_size,
    )
    gdp_series, gdp_band_name = aggregate_gdp_to_nuts3(
        gdp_raster=args.gdp_raster,
        nuts3_gdf=base_gdf,
        gdp_year=args.gdp_year,
        all_touched=args.all_touched,
    )

    refreshed = merge_proxy_values(
        base_gdf=base_gdf,
        population_series=population_series,
        gdp_series=gdp_series,
        gdp_band_name=gdp_band_name,
        population_column=args.population_column,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    refreshed.to_file(output_path, driver="GeoJSON")
    print_summary(refreshed, population_stats, output_path)


if __name__ == "__main__":
    main()
