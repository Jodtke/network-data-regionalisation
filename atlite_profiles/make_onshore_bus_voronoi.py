from __future__ import annotations

"""Create onshore bus regions for raster-to-bus aggregation.

Onshore renewable potential is aggregated to the reduced-grid buses using
Voronoi regions clipped to the land mask and country assignment. The generated
geometries define which raster cells can contribute to each bus before the
scenario-capacity allocation step is applied.
"""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPoint
from shapely.ops import voronoi_diagram

LOG = logging.getLogger(__name__)

BUS_CSV = Path(
    r"Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf\grid\target_year_2030\electrical_spectral_line_equivalent_dc_effective_reactance\buses.csv"
)
MASK_SHAPES = [
    Path(r"Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf\datashapes\europe_shape.geojson"),
]
OUT_PATH = Path(
    r"Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf\datashapes\regions_onshore_bus_2030_voronoi.geojson"
)

SRC_CRS = "EPSG:4326"
WORK_CRS = "EPSG:3035"


def _make_valid(geoms: gpd.GeoSeries) -> gpd.GeoSeries:
    try:
        from shapely import make_valid as _make_valid_geom
    except Exception:
        _make_valid_geom = None
    if _make_valid_geom is None:
        return geoms.buffer(0)
    return geoms.apply(_make_valid_geom)


def _union_shapes(paths: list[Path]) -> object:
    pieces: list[gpd.GeoSeries] = []
    for path in paths:
        gdf = gpd.read_file(path)
        if gdf.empty:
            LOG.warning("empty shapes file: %s", path)
            continue
        if gdf.crs is None:
            raise ValueError(f"shapes missing CRS: {path}")
        gdf = gdf.to_crs(WORK_CRS)
        valid = _make_valid(gdf.geometry)
        valid = valid[~valid.is_empty]
        if not valid.empty:
            pieces.append(valid)
    if not pieces:
        raise SystemExit("no shapes available for mask union")
    merged = gpd.GeoSeries(pd.concat(pieces, ignore_index=True), crs=WORK_CRS)
    if hasattr(merged, "union_all"):
        return merged.union_all()
    return merged.unary_union


def _assign_polygons(
    polygons: gpd.GeoDataFrame, points: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    if hasattr(gpd, "sjoin_nearest"):
        try:
            joined = gpd.sjoin_nearest(polygons, points, how="left", distance_col="dist_m")
            return joined.drop(columns=["dist_m", "index_right"])
        except Exception as exc:
            LOG.warning("sjoin_nearest failed (%s), falling back to STRtree", exc)

    from shapely.strtree import STRtree

    point_geoms = list(points.geometry.values)
    tree = STRtree(point_geoms)
    geom_index = {id(geom): idx for idx, geom in enumerate(point_geoms)}
    rows = []
    for geom in polygons.geometry.values:
        nearest = tree.nearest(geom)
        rows.append(points.iloc[geom_index[id(nearest)]])
    joined = polygons.join(pd.DataFrame(rows).reset_index(drop=True))
    return joined


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOG.info("loading buses: %s", BUS_CSV)
    buses = pd.read_csv(BUS_CSV, sep=";")
    for col in ("bus_id", "lat", "lon", "country"):
        if col not in buses.columns:
            raise ValueError(f"missing column in buses.csv: {col}")

    buses = buses[["bus_id", "lat", "lon", "country"]].copy()
    points = gpd.GeoDataFrame(
        buses,
        geometry=gpd.points_from_xy(buses["lon"], buses["lat"]),
        crs=SRC_CRS,
    ).to_crs(WORK_CRS)

    LOG.info("building mask union")
    mask_union = _union_shapes(MASK_SHAPES)
    within = points.geometry.intersects(mask_union)
    if not within.any():
        raise SystemExit("no buses intersect onshore mask")
    dropped = (~within).sum()
    if dropped:
        LOG.warning("dropping %d buses outside mask", dropped)
    points = points.loc[within].reset_index(drop=True)

    LOG.info("building Voronoi diagram (%d buses)", len(points))
    multi = MultiPoint(points.geometry.values)
    vor = voronoi_diagram(multi, envelope=mask_union, edges=False)
    polygons = gpd.GeoDataFrame(geometry=list(vor.geoms), crs=WORK_CRS)
    polygons["geometry"] = polygons.geometry.intersection(mask_union)
    polygons = polygons[~polygons.geometry.is_empty].reset_index(drop=True)

    LOG.info("assigning polygons to buses")
    assigned = _assign_polygons(polygons, points)
    if assigned["bus_id"].isna().any():
        raise SystemExit("some polygons were not assigned to buses")

    assigned = assigned.rename(columns={"bus_id": "name", "lon": "x", "lat": "y"})
    assigned = assigned[["name", "x", "y", "country", "geometry"]]
    assigned = assigned.to_crs(SRC_CRS)

    LOG.info("writing %s", OUT_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    assigned.to_file(OUT_PATH, driver="GeoJSON")
    LOG.info("done: %d regions", len(assigned))


if __name__ == "__main__":
    main()
