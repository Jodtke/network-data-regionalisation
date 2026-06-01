from __future__ import annotations

"""Inspect generated raster profiles and availability masks.

The diagnostics in this script are intended for plausibility checks before the
raster data are used for nodal allocation. They compare spatial coverage, mask
overlaps, and capacity-factor patterns for selected countries or technologies,
which helps distinguish data-source problems from later disaggregation effects.
"""

import argparse
import logging
from pathlib import Path

import atlite
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

LOG = logging.getLogger(__name__)

DEFAULT_CUTOUT_DIR = Path(r"\\IIP-COMP103\endata\MA_Lisa\atlite_cutouts\cutouts")
DEFAULT_OUT_DIR = Path(r"\\IIP-COMP103\endata\MA_Eric")
DEFAULT_COUNTRY_SHAPES = Path(
    r"Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf\pypsa-eur\resources\country_shapes.geojson"
)
DEFAULT_PLOT_OUT_DIR = Path("diagnose_plots")

RESOURCE_CHOICES = ["pv", "onwind", "offwind", "hydro"]
PLOT_FORMAT_CHOICES = ["svg", "png", "pdf"]
PLOT_MODE_CHOICES = ["overlay", "per-year"]

COUNTRY_ALIASES = {
    "albania": "AL",
    "austria": "AT",
    "bosnia": "BA",
    "bosnia and herzegovina": "BA",
    "belgium": "BE",
    "bulgaria": "BG",
    "switzerland": "CH",
    "czechia": "CZ",
    "czech republic": "CZ",
    "germany": "DE",
    "denmark": "DK",
    "estonia": "EE",
    "spain": "ES",
    "finland": "FI",
    "france": "FR",
    "united kingdom": "GB",
    "uk": "GB",
    "great britain": "GB",
    "britain": "GB",
    "greece": "GR",
    "croatia": "HR",
    "hungary": "HU",
    "ireland": "IE",
    "italy": "IT",
    "lithuania": "LT",
    "luxembourg": "LU",
    "latvia": "LV",
    "montenegro": "ME",
    "north macedonia": "MK",
    "macedonia": "MK",
    "netherlands": "NL",
    "holland": "NL",
    "norway": "NO",
    "poland": "PL",
    "portugal": "PT",
    "romania": "RO",
    "serbia": "RS",
    "sweden": "SE",
    "slovenia": "SI",
    "slovakia": "SK",
}


def _find_cutout_path(cutout_dir: Path, start_year: int, end_year: int) -> Path:
    for year in range(start_year, end_year + 1):
        path = cutout_dir / f"europe-{year}.nc"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"no cutout files found in {cutout_dir} for {start_year}-{end_year}"
    )


def _normalize_country(country: str | None) -> str | None:
    if not country:
        return None
    normalized = country.strip().lower()
    if normalized in {"eu", "europe", "all"}:
        return None
    return country.strip()


def _resolve_country_code(country: str, available: set[str]) -> str | None:
    normalized = country.strip().lower()
    for code in available:
        if code.lower() == normalized:
            return code.upper()
    alias = COUNTRY_ALIASES.get(normalized)
    if alias in available:
        return alias
    try:
        import pycountry  # type: ignore
    except ImportError:
        return None
    if len(normalized) == 3:
        entry = pycountry.countries.get(alpha_3=normalized.upper())
        if entry and entry.alpha_2 in available:
            return entry.alpha_2
    try:
        entry = pycountry.countries.lookup(country)
    except LookupError:
        return None
    if entry and entry.alpha_2 in available:
        return entry.alpha_2
    return None


def _load_country_shapes(country_shapes_path: Path) -> gpd.GeoDataFrame:
    shapes = gpd.read_file(country_shapes_path)
    if shapes.empty:
        raise ValueError(f"no shapes found in {country_shapes_path}")
    if "name" not in shapes.columns:
        raise ValueError(f"country shapes missing 'name' column: {country_shapes_path}")
    shapes["name"] = shapes["name"].astype(str)
    return shapes


def _available_country_codes(shapes: gpd.GeoDataFrame) -> list[str]:
    codes = {name.upper() for name in shapes["name"] if name}
    return sorted(codes)


def _resolve_countries(requested: list[str], available: set[str]) -> list[str]:
    resolved: list[str] = []
    for country in requested:
        code = _resolve_country_code(country, available)
        if not code:
            raise ValueError(
                f"country '{country}' not found; available codes: {', '.join(sorted(available))}"
            )
        if code not in resolved:
            resolved.append(code)
    return resolved


def _build_country_masks(
    cutout_path: Path, shapes: gpd.GeoDataFrame, countries: list[str]
) -> dict[str, xr.DataArray]:
    cutout = atlite.Cutout(cutout_path)
    try:
        name_col = "name"
        geoms = []
        codes = []
        for country in countries:
            subset = shapes[shapes[name_col].str.upper() == country.upper()]
            if subset.empty:
                raise ValueError(f"country '{country}' not found in {name_col}")
            geom_series = subset.geometry
            geom = (
                geom_series.union_all()
                if hasattr(geom_series, "union_all")
                else geom_series.unary_union
            )
            geoms.append(geom)
            codes.append(country.upper())
        matrix = cutout.indicatormatrix(geoms, shapes_crs=shapes.crs)
        if hasattr(matrix, "toarray"):
            coverage = matrix.toarray()
        else:
            coverage = np.asarray(matrix)
        coverage = coverage.reshape(len(codes), *cutout.shape).astype("float32")
        masks: dict[str, xr.DataArray] = {}
        for idx, code in enumerate(codes):
            mask = xr.DataArray(
                coverage[idx],
                coords=[cutout.data.coords["y"], cutout.data.coords["x"]],
                dims=("y", "x"),
            )
            masks[code] = mask.clip(0.0, 1.0)
        return masks
    finally:
        cutout.data.close()


def _spatial_dims(cf: xr.DataArray) -> tuple[str, str]:
    for dims in (("y", "x"), ("lat", "lon"), ("latitude", "longitude")):
        if all(d in cf.dims for d in dims):
            return dims
    raise ValueError(f"could not find spatial dims in {cf.dims}")


def _drop_leap_day(da: xr.DataArray) -> xr.DataArray:
    if "time" not in da.coords:
        return da
    try:
        month = da["time"].dt.month
        day = da["time"].dt.day
    except Exception:
        return da
    mask = ~((month == 2) & (day == 29))
    return da.sel(time=mask)


def _mean_cf_timeseries_by_country(
    cf_path: Path, masks: dict[str, xr.DataArray], drop_leap_day: bool
) -> dict[str, xr.DataArray]:
    ds = xr.open_dataset(cf_path, chunks="auto")
    try:
        cf = ds["cf"]
        if drop_leap_day:
            cf = _drop_leap_day(cf)
        spatial_dims = _spatial_dims(cf)
        if not masks:
            return {"Europe": cf.mean(dim=spatial_dims).compute()}
        results: dict[str, xr.DataArray] = {}
        for code, mask in masks.items():
            weights = mask.reindex(
                {spatial_dims[0]: cf.coords[spatial_dims[0]], spatial_dims[1]: cf.coords[spatial_dims[1]]}
            )
            denom = weights.sum()
            if float(denom) == 0.0:
                raise ValueError(f"mask has zero coverage for {code}")
            ts = (cf * weights).sum(dim=spatial_dims) / denom
            results[code] = ts.compute()
        return results
    finally:
        ds.close()


def _hydro_timeseries_by_country(
    hydro_path: Path, countries: list[str] | None, drop_leap_day: bool, per_unit: bool
) -> dict[str, xr.DataArray]:
    ds = xr.open_dataset(hydro_path, chunks="auto")
    try:
        inflow = ds["p_avail_total"]
        if drop_leap_day:
            inflow = _drop_leap_day(inflow)

        if per_unit:
            p_inst_total = ds["p_inst"].sum("hydro_type")

        if not countries:
            if per_unit:
                total_inst = p_inst_total.sum("country")
                ts = xr.where(total_inst > 0, inflow.sum("country") / total_inst, 0.0)
            else:
                ts = inflow.sum("country")
            return {"Europe": ts.compute()}

        available = {str(c) for c in inflow.coords["country"].values}
        results: dict[str, xr.DataArray] = {}
        for code in countries:
            if code not in available:
                LOG.warning("missing hydro country %s in %s", code, hydro_path)
                continue
            series = inflow.sel(country=code)
            if per_unit:
                inst = p_inst_total.sel(country=code)
                series = xr.where(inst > 0, series / inst, 0.0)
            results[code] = series.compute()
        return results
    finally:
        ds.close()


def _plot_title(
    resource: str, region_label: str, per_unit: bool, custom_title: str | None
) -> str:
    if custom_title:
        base = custom_title
    elif resource == "hydro":
        base = "hydro inflow per unit" if per_unit else "hydro inflow"
    else:
        base = f"{resource} mean CF"
    return f"{base} ({region_label})"


def _plot_ylabel(resource: str, per_unit: bool) -> str:
    if resource == "hydro":
        return "hydro inflow per unit" if per_unit else "hydro inflow (MW)"
    return "capacity factor"


def _sanitize_label(value: str) -> str:
    return value.replace(" ", "-")


def _plot_path(
    out_dir: Path,
    resource: str,
    region_label: str,
    start_year: int,
    end_year: int,
    plot_format: str,
    per_unit: bool,
    year: int | None,
) -> Path:
    region_tag = "EU" if region_label.lower() == "europe" else _sanitize_label(region_label)
    if resource == "hydro":
        resource_tag = "hydro_pu" if per_unit else "hydro_inflow"
    else:
        resource_tag = resource
    if year is None:
        filename = f"{resource_tag}_{region_tag}_{start_year}_{end_year}.{plot_format}"
    else:
        filename = f"{resource_tag}_{region_tag}_{year}.{plot_format}"
    return out_dir / filename


def _save_plot(fig: plt.Figure, plot_path: Path, plot_format: str) -> None:
    if not plot_path.suffix:
        plot_path = plot_path.with_suffix(f".{plot_format}")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = plot_path.suffix.lstrip(".") if plot_path.suffix else plot_format
    fig.savefig(plot_path, dpi=150, format=fmt)
    LOG.info("wrote %s", plot_path)


def _plot_overlay_series(
    series_by_year: list[tuple[int, xr.DataArray]],
    title: str,
    ylabel: str,
    alpha: float,
    linewidth: float,
    plot_path: Path | None,
    plot_format: str,
) -> None:
    if not series_by_year:
        return
    series_by_year.sort(key=lambda item: item[0])
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("viridis", len(series_by_year))
    for idx, (year, ts) in enumerate(series_by_year):
        x = np.arange(ts.sizes["time"])
        ax.plot(
            x,
            ts.values,
            label=str(year),
            color=cmap(idx),
            alpha=alpha,
            linewidth=linewidth,
        )
    ax.set_title(title)
    ax.set_xlabel("hour of year")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    legend_cols = 4 if len(series_by_year) > 30 else 2
    ax.legend(ncol=legend_cols, fontsize=8)
    fig.tight_layout()

    if plot_path:
        _save_plot(fig, plot_path, plot_format)
        plt.close(fig)
    else:
        plt.show()


def _plot_single_series(
    year: int,
    ts: xr.DataArray,
    title: str,
    ylabel: str,
    alpha: float,
    linewidth: float,
    plot_path: Path | None,
    plot_format: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(ts.sizes["time"])
    ax.plot(x, ts.values, color="#1f77b4", alpha=alpha, linewidth=linewidth)
    ax.set_title(title)
    ax.set_xlabel("hour of year")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if plot_path:
        _save_plot(fig, plot_path, plot_format)
        plt.close(fig)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean capacity factor time series across weather years."
    )
    parser.add_argument("--resource", choices=RESOURCE_CHOICES, default=None)
    parser.add_argument(
        "--resources", nargs="+", choices=RESOURCE_CHOICES + ["all"], default=None
    )
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cutout-dir", type=Path, default=DEFAULT_CUTOUT_DIR)
    parser.add_argument("--country", type=str, default=None)
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--all-countries", action="store_true")
    parser.add_argument("--country-shapes", type=Path, default=DEFAULT_COUNTRY_SHAPES)
    parser.add_argument("--drop-leap-day", action="store_true")
    parser.add_argument("--hydro-per-unit", action="store_true")
    parser.add_argument("--plot-out", type=Path, default=None)
    parser.add_argument("--plot-out-dir", type=Path, default=None)
    parser.add_argument("--plot-format", choices=PLOT_FORMAT_CHOICES, default="svg")
    parser.add_argument("--plot-mode", choices=PLOT_MODE_CHOICES, default="overlay")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--linewidth", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.resource and args.resources:
        raise SystemExit("use either --resource or --resources")
    if not args.resource and not args.resources:
        raise SystemExit("missing --resource or --resources")

    resources = args.resources or [args.resource]
    if "all" in resources:
        resources = RESOURCE_CHOICES
    resources = list(dict.fromkeys(resources))

    requested_countries: list[str] = []
    if args.country:
        normalized = _normalize_country(args.country)
        if normalized:
            requested_countries.append(normalized)
    if args.countries:
        for entry in args.countries:
            normalized = _normalize_country(entry)
            if normalized:
                requested_countries.append(normalized)

    shapes = None
    countries: list[str] | None = None
    if args.all_countries or requested_countries:
        shapes = _load_country_shapes(args.country_shapes)
        available = set(_available_country_codes(shapes))
        if args.all_countries:
            if requested_countries:
                LOG.info("ignoring --country/--countries because --all-countries is set")
            countries = sorted(available)
        else:
            countries = _resolve_countries(requested_countries, available)

    multiple_resources = len(resources) > 1
    multiple_countries = countries is not None and len(countries) > 1
    if args.plot_out and (multiple_resources or multiple_countries):
        raise SystemExit("--plot-out only supports a single resource/country")
    if args.plot_out and args.plot_mode == "per-year" and args.start_year != args.end_year:
        raise SystemExit("--plot-out with --plot-mode per-year needs start-year == end-year")

    plot_out_dir = args.plot_out_dir
    needs_dir = (
        multiple_resources
        or multiple_countries
        or args.all_countries
        or (args.plot_mode == "per-year" and args.start_year != args.end_year)
    )
    if plot_out_dir is None and needs_dir:
        plot_out_dir = DEFAULT_PLOT_OUT_DIR

    masks: dict[str, xr.DataArray] = {}
    if countries and any(r in {"pv", "onwind", "offwind"} for r in resources):
        if shapes is None:
            shapes = _load_country_shapes(args.country_shapes)
        cutout_path = _find_cutout_path(args.cutout_dir, args.start_year, args.end_year)
        LOG.info("building masks using %s", cutout_path)
        masks = _build_country_masks(cutout_path, shapes, countries)

    for resource in resources:
        regions = countries if countries else ["Europe"]
        series_by_region: dict[str, list[tuple[int, xr.DataArray]]] = {
            region: [] for region in regions
        }

        for year in range(args.start_year, args.end_year + 1):
            if resource == "hydro":
                hydro_path = args.out_dir / "hydro" / f"hydro_country_profiles_{year}.nc"
                if not hydro_path.exists():
                    LOG.warning("missing %s (skipping)", hydro_path)
                    continue
                LOG.info("loading %s", hydro_path)
                year_series = _hydro_timeseries_by_country(
                    hydro_path, countries, args.drop_leap_day, args.hydro_per_unit
                )
            else:
                cf_path = args.out_dir / resource / f"{resource}_cf_{year}.nc"
                if not cf_path.exists():
                    LOG.warning("missing %s (skipping)", cf_path)
                    continue
                LOG.info("loading %s", cf_path)
                year_series = _mean_cf_timeseries_by_country(
                    cf_path, masks if countries else {}, args.drop_leap_day
                )

            for region, ts in year_series.items():
                if region not in series_by_region:
                    series_by_region[region] = []
                series_by_region[region].append((year, ts))

        for region, series in series_by_region.items():
            if not series:
                LOG.warning("no data for %s (%s)", region, resource)
                continue
            region_label = region
            ylabel = _plot_ylabel(resource, args.hydro_per_unit)

            if args.plot_mode == "per-year":
                for year, ts in series:
                    base_title = _plot_title(resource, region_label, args.hydro_per_unit, args.title)
                    title = f"{base_title} {year}"
                    if args.plot_out:
                        plot_path = args.plot_out
                    elif plot_out_dir:
                        plot_path = _plot_path(
                            plot_out_dir,
                            resource,
                            region_label,
                            args.start_year,
                            args.end_year,
                            args.plot_format,
                            args.hydro_per_unit,
                            year=year,
                        )
                    else:
                        plot_path = None
                    _plot_single_series(
                        year=year,
                        ts=ts,
                        title=title,
                        ylabel=ylabel,
                        alpha=args.alpha,
                        linewidth=args.linewidth,
                        plot_path=plot_path,
                        plot_format=args.plot_format,
                    )
            else:
                title = _plot_title(resource, region_label, args.hydro_per_unit, args.title)
                if args.plot_out:
                    plot_path = args.plot_out
                elif plot_out_dir:
                    plot_path = _plot_path(
                        plot_out_dir,
                        resource,
                        region_label,
                        args.start_year,
                        args.end_year,
                        args.plot_format,
                        args.hydro_per_unit,
                        year=None,
                    )
                else:
                    plot_path = None
                _plot_overlay_series(
                    series,
                    title=title,
                    ylabel=ylabel,
                    alpha=args.alpha,
                    linewidth=args.linewidth,
                    plot_path=plot_path,
                    plot_format=args.plot_format,
                )


if __name__ == "__main__":
    main()
