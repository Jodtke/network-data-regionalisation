from __future__ import annotations

"""Analyse raster-cell filtering thresholds for renewable potentials.

Minimum potential and capacity-factor filters remove negligible cells before
capacity is allocated. This diagnostic script quantifies how much area,
potential capacity, and expected generation are lost under alternative
thresholds, which helps justify the thresholds used in publication scenarios.
"""

import argparse
import logging
from argparse import Namespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from res_capacity_preprocessing import (
    build_offshore_assignment,
    build_onshore_assignment,
    compute_mean_cell_cf,
    default_settings as capacity_default_settings,
    infer_target_year,
    resolve_settings as resolve_capacity_settings,
)
from res_common import (
    build_onshore_voronoi,
    build_bus_lookup,
    derive_network_country_map,
    ensure_dir,
    load_plants,
    load_reduced_buses,
    merge_country_cluster_maps,
    read_country_cluster_map,
    resolve_cli_path,
    write_onshore_voronoi,
)

LOG = logging.getLogger(__name__)

P_NOM_SWEEPS = {
    "pv": np.array([0, 1, 2, 5, 10, 20, 50, 100, 250, 500, 1000, 1500, 2000, 2500], dtype=float),
    "onwind": np.array([0, 1, 2, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000], dtype=float),
    "offwind": np.array([0, 1, 2, 5, 10, 20, 50, 100, 200, 400, 600], dtype=float),
}

CF_SWEEPS = {
    "pv": np.round(np.arange(0.00, 0.151, 0.01), 2),
    "onwind": np.round(np.arange(0.00, 0.201, 0.01), 2),
    "offwind": np.round(np.arange(0.00, 0.301, 0.01), 2),
}

TECH_STYLE = {
    "pv": {"label": "PV", "color": "#e28743"},
    "onwind": {"label": "Onshore Wind", "color": "#2a6f97"},
    "offwind": {"label": "Offshore Wind", "color": "#5b8c5a"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze cell-level RES filter thresholds and export CSV/PNG diagnostics."
    )
    parser.add_argument("--config", type=Path, default=Path("res_capacity_template.yaml"))
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--network-dir", type=Path, default=None)
    parser.add_argument("--atlite-case-dir", type=Path, default=None)
    parser.add_argument("--weather-root", type=Path, default=None)
    parser.add_argument("--buses-csv", type=Path, default=None)
    parser.add_argument("--plants-csv", type=Path, default=None)
    parser.add_argument("--country-clusters-csv", type=Path, default=None)
    parser.add_argument("--onshore-voronoi-geojson", type=Path, default=None)
    parser.add_argument("--generated-onshore-voronoi-geojson", type=Path, default=None)
    parser.add_argument("--offshore-eez-geojson", type=Path, default=None)
    parser.add_argument("--onshore-mask-geojson", action="append", default=None)
    parser.add_argument("--target-year", type=int, default=None)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--min-p-max-pu", default=None)
    parser.add_argument("--min-p-nom-max", default=None)
    parser.add_argument("--min-distance-offshore-km", type=float, default=None)
    parser.add_argument("--max-distance-offshore-km", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite-voronoi", action="store_true")
    return parser.parse_args()


def build_capacity_namespace(args: argparse.Namespace) -> Namespace:
    defaults = capacity_default_settings()
    return Namespace(
        config=args.config,
        project_root=args.project_root,
        network_dir=args.network_dir,
        simulation_dir=None,
        output_dir=args.output_dir,
        atlite_case_dir=args.atlite_case_dir,
        availability_onshore_nc=None,
        availability_offshore_nc=None,
        target_capacity_csv=None,
        pv_power_csv=None,
        onwind_power_csv=None,
        offwind_power_csv=None,
        weather_root=args.weather_root,
        buses_csv=args.buses_csv,
        plants_csv=args.plants_csv,
        country_clusters_csv=args.country_clusters_csv,
        onshore_voronoi_geojson=args.onshore_voronoi_geojson,
        generated_onshore_voronoi_geojson=args.generated_onshore_voronoi_geojson,
        offshore_eez_geojson=args.offshore_eez_geojson,
        onshore_mask_geojson=args.onshore_mask_geojson,
        target_year=args.target_year,
        start_year=args.start_year,
        end_year=args.end_year,
        tyndp_scenario=defaults["tyndp_scenario"],
        tyndp_capacity_unit=defaults["tyndp_capacity_unit"],
        min_p_max_pu=args.min_p_max_pu,
        min_p_nom_max=args.min_p_nom_max,
        min_distance_offshore_km=args.min_distance_offshore_km,
        max_distance_offshore_km=args.max_distance_offshore_km,
        overwrite=False,
    )


def resolve_analysis_settings(args: argparse.Namespace) -> dict[str, object]:
    settings = resolve_capacity_settings(build_capacity_namespace(args))
    network_dir = Path(settings["network_dir"])
    target_year = infer_target_year(network_dir, settings.get("target_year"))
    if target_year is None:
        raise ValueError("Could not infer target year for analysis output layout.")
    project_root = Path(settings["project_root"])
    if args.output_dir is not None:
        output_dir = resolve_cli_path(args.output_dir)
        assert output_dir is not None
    else:
        output_dir = (
            Path.cwd()
            / "cell_filter_threshold_analysis"
            / network_dir.parent.name
            / network_dir.name
            / f"res_{Path(settings['atlite_case_dir']).name}"
        )
    settings["analysis_output_dir"] = output_dir
    settings["target_year"] = target_year
    settings["overwrite_voronoi"] = args.overwrite_voronoi
    return settings


def ensure_voronoi(settings: dict[str, object], buses: pd.DataFrame) -> Path:
    output_dir = ensure_dir(Path(settings["analysis_output_dir"]))
    voronoi_path = settings.get("onshore_voronoi_geojson")
    if isinstance(voronoi_path, Path) and voronoi_path.exists() and not settings.get("overwrite_voronoi", False):
        return voronoi_path

    generated = output_dir / "generated_regions" / (
        f"{Path(settings['network_dir']).parent.name}_{Path(settings['network_dir']).name}_onshore_voronoi.geojson"
    )
    if generated.exists() and not settings.get("overwrite_voronoi", False):
        return generated

    LOG.info("building analysis Voronoi for %s", settings["network_dir"])
    regions = build_onshore_voronoi(buses, list(settings["onshore_mask_geojsons"]))
    write_onshore_voronoi(generated, regions)
    return generated


def build_cluster_map(settings: dict[str, object]):
    network_map = derive_network_country_map(Path(settings["buses_csv"]))
    external_map = read_country_cluster_map(Path(settings["country_clusters_csv"]))
    return merge_country_cluster_maps(network_map, external_map)


def prepare_cell_frame(base_assignment: pd.DataFrame, ds: xr.Dataset, technology: str) -> pd.DataFrame:
    cells = base_assignment.copy()
    if technology == "pv":
        availability_var = "availability_pv"
        p_nom_var = "p_nom_max_pv"
    elif technology == "onwind":
        availability_var = "availability_onwind"
        p_nom_var = "p_nom_max_onwind"
    elif technology == "offwind":
        availability_var = "availability_offwind"
        p_nom_var = "p_nom_max_offwind"
    else:
        raise ValueError(f"Unsupported technology: {technology}")

    availability = np.asarray(ds[availability_var].values)
    p_nom_max = np.asarray(ds[p_nom_var].values)
    cells["availability"] = availability[cells["row_ix"].to_numpy(int), cells["col_ix"].to_numpy(int)]
    cells["p_nom_max_mw"] = p_nom_max[cells["row_ix"].to_numpy(int), cells["col_ix"].to_numpy(int)]
    cells = cells[np.isfinite(cells["availability"]) & np.isfinite(cells["p_nom_max_mw"])].copy()
    cells = cells[cells["availability"] > 0.0].copy()
    return cells


def quantile_rows(values: np.ndarray, metric: str, technology: str) -> list[dict[str, object]]:
    probs = {
        "min": 0.0,
        "p05": 5.0,
        "p10": 10.0,
        "p25": 25.0,
        "p50": 50.0,
        "p75": 75.0,
        "p90": 90.0,
        "p95": 95.0,
        "max": 100.0,
    }
    rows: list[dict[str, object]] = []
    for label, pct in probs.items():
        rows.append(
            {
                "technology": technology,
                "metric": metric,
                "quantile": label,
                "value": float(np.nanpercentile(values, pct)),
            }
        )
    rows.append(
        {
            "technology": technology,
            "metric": metric,
            "quantile": "mean",
            "value": float(np.nanmean(values)),
        }
    )
    return rows


def retention_rows(
    cells: pd.DataFrame,
    metric: str,
    threshold_values: np.ndarray,
    value_column: str,
    technology: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total_cells = len(cells)
    total_buses = cells["bus_id"].nunique()
    values = cells[value_column].to_numpy(float)
    for threshold in threshold_values:
        keep = values >= float(threshold)
        rows.append(
            {
                "technology": technology,
                "metric": metric,
                "threshold": float(threshold),
                "cells_kept": int(keep.sum()),
                "cells_total": int(total_cells),
                "cells_share": float(keep.mean()) if total_cells else np.nan,
                "buses_kept": int(cells.loc[keep, "bus_id"].nunique()),
                "buses_total": int(total_buses),
                "buses_share": float(cells.loc[keep, "bus_id"].nunique() / total_buses) if total_buses else np.nan,
            }
        )
    return rows


def current_threshold_rows(cells: pd.DataFrame, technology: str, threshold_nom: float, threshold_cf: float) -> list[dict[str, object]]:
    total_cells = len(cells)
    total_buses = cells["bus_id"].nunique()
    keep_nom = cells["p_nom_max_mw"].to_numpy(float) >= threshold_nom
    keep_cf = cells["mean_cell_cf"].to_numpy(float) >= threshold_cf
    keep_both = keep_nom & keep_cf
    rows = []
    for label, mask, threshold in (
        ("min_p_nom_max", keep_nom, threshold_nom),
        ("min_p_max_pu", keep_cf, threshold_cf),
        ("combined", keep_both, np.nan),
    ):
        rows.append(
            {
                "technology": technology,
                "filter_stage": label,
                "threshold": float(threshold) if np.isfinite(threshold) else np.nan,
                "cells_kept": int(mask.sum()),
                "cells_total": int(total_cells),
                "cells_share": float(mask.mean()) if total_cells else np.nan,
                "buses_kept": int(cells.loc[mask, "bus_id"].nunique()),
                "buses_total": int(total_buses),
                "buses_share": float(cells.loc[mask, "bus_id"].nunique() / total_buses) if total_buses else np.nan,
            }
        )
    return rows


def save_distribution_plot(cell_stats: pd.DataFrame, value_column: str, threshold_column: str, title: str, x_label: str, output_path: Path, *, log_x: bool = False) -> None:
    technologies = ["pv", "onwind", "offwind"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, technology in zip(axes, technologies):
        subset = cell_stats[cell_stats["technology"] == technology].copy()
        values = subset[value_column].to_numpy(float)
        threshold = float(subset[threshold_column].iloc[0])
        style = TECH_STYLE[technology]
        valid = values[np.isfinite(values)]
        if log_x:
            valid = valid[valid > 0.0]
            bins = np.geomspace(valid.min(), valid.max(), 40) if len(valid) else np.array([1.0, 10.0])
            ax.hist(valid, bins=bins, color=style["color"], alpha=0.75)
            ax.set_xscale("log")
        else:
            ax.hist(valid, bins=40, color=style["color"], alpha=0.75)
        ax.axvline(threshold, color="#222222", linestyle="--", linewidth=1.5, label=f"threshold={threshold:g}")
        ax.set_title(style["label"])
        ax.set_xlabel(x_label)
        ax.set_ylabel("Cell count")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2, linewidth=0.5)
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_retention_plot(retention: pd.DataFrame, output_path: Path) -> None:
    metrics = [("p_nom_max_mw", "P_nom_max threshold [MW/cell]"), ("mean_cell_cf", "Mean cell CF threshold [-]")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for row_index, (metric, x_label) in enumerate(metrics):
        subset = retention[retention["metric"] == metric]
        for technology in ["pv", "onwind", "offwind"]:
            tech = subset[subset["technology"] == technology]
            style = TECH_STYLE[technology]
            axes[row_index, 0].plot(tech["threshold"], tech["cells_share"], color=style["color"], label=style["label"])
            axes[row_index, 1].plot(tech["threshold"], tech["buses_share"], color=style["color"], label=style["label"])
        axes[row_index, 0].set_ylabel("Retained cells share")
        axes[row_index, 1].set_ylabel("Retained buses share")
        axes[row_index, 0].set_xlabel(x_label)
        axes[row_index, 1].set_xlabel(x_label)
        axes[row_index, 0].grid(alpha=0.2, linewidth=0.5)
        axes[row_index, 1].grid(alpha=0.2, linewidth=0.5)
        axes[row_index, 0].set_ylim(0.0, 1.02)
        axes[row_index, 1].set_ylim(0.0, 1.02)
    axes[0, 0].set_title("P_nom_max threshold retention")
    axes[0, 1].set_title("P_nom_max threshold bus coverage")
    axes[1, 0].set_title("Mean cell CF threshold retention")
    axes[1, 1].set_title("Mean cell CF threshold bus coverage")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_current_threshold_bar_plot(current_thresholds: pd.DataFrame, output_path: Path) -> None:
    plot_data = current_thresholds[current_thresholds["filter_stage"] == "combined"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(len(plot_data))
    colors = [TECH_STYLE[tech]["color"] for tech in plot_data["technology"]]
    axes[0].bar(x, plot_data["cells_share"], color=colors)
    axes[1].bar(x, plot_data["buses_share"], color=colors)
    for ax, column, title in (
        (axes[0], "cells_share", "Current combined thresholds: retained cells"),
        (axes[1], "buses_share", "Current combined thresholds: buses with remaining cells"),
    ):
        ax.set_xticks(x, [TECH_STYLE[tech]["label"] for tech in plot_data["technology"]])
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("Share")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        for xpos, value in zip(x, plot_data[column]):
            ax.text(xpos, float(value) + 0.02, f"{float(value) * 100:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    settings = resolve_analysis_settings(args)
    output_dir = ensure_dir(Path(settings["analysis_output_dir"]))

    LOG.info("loading network inputs for threshold analysis")
    buses = load_reduced_buses(Path(settings["buses_csv"]))
    load_plants(Path(settings["plants_csv"]))  # validates the plants input used by the workflow
    cluster_map = build_cluster_map(settings)
    bus_lookup = build_bus_lookup(buses)
    voronoi_path = ensure_voronoi(settings, buses)

    LOG.info("loading availability datasets")
    with xr.open_dataset(Path(settings["availability_onshore_nc"]), engine="netcdf4") as ds_onshore, xr.open_dataset(
        Path(settings["availability_offshore_nc"]), engine="netcdf4"
    ) as ds_offshore:
        LOG.info("building cell assignments")
        onshore_assignment = build_onshore_assignment(ds_onshore, voronoi_path)
        offshore_assignment = build_offshore_assignment(
            ds_offshore,
            Path(settings["offshore_eez_geojson"]),
            bus_lookup,
            cluster_map,
            settings["onshore_mask_geojsons"],
            settings.get("min_distance_offshore_km"),
            settings.get("max_distance_offshore_km"),
        )

        technology_inputs = {
            "pv": (onshore_assignment, ds_onshore),
            "onwind": (onshore_assignment, ds_onshore),
            "offwind": (offshore_assignment, ds_offshore),
        }

        all_cell_stats: list[pd.DataFrame] = []
        quantile_output: list[dict[str, object]] = []
        retention_output: list[dict[str, object]] = []
        current_threshold_output: list[dict[str, object]] = []

        for technology, (assignment, dataset) in technology_inputs.items():
            LOG.info("analyzing %s cell thresholds", technology)
            threshold_nom = float(settings["min_p_nom_max"].get(technology, settings["min_p_nom_max"].get("default", 0.0))) if isinstance(settings["min_p_nom_max"], dict) else float(settings["min_p_nom_max"])
            threshold_cf = float(settings["min_p_max_pu"].get(technology, settings["min_p_max_pu"].get("default", 0.0))) if isinstance(settings["min_p_max_pu"], dict) else float(settings["min_p_max_pu"])
            cells = prepare_cell_frame(assignment, dataset, technology)
            cells["technology"] = technology
            cells["mean_cell_cf"] = compute_mean_cell_cf(
                cells,
                Path(settings["weather_root"]),
                technology,
                int(settings["start_year"]),
                int(settings["end_year"]),
            )
            cells["current_min_p_nom_max"] = threshold_nom
            cells["current_min_p_max_pu"] = threshold_cf
            cells["keep_nom_current"] = cells["p_nom_max_mw"] >= threshold_nom
            cells["keep_cf_current"] = cells["mean_cell_cf"] >= threshold_cf
            cells["keep_both_current"] = cells["keep_nom_current"] & cells["keep_cf_current"]
            all_cell_stats.append(
                cells[
                    [
                        "technology",
                        "bus_id",
                        "model_country",
                        "row_ix",
                        "col_ix",
                        "availability",
                        "p_nom_max_mw",
                        "mean_cell_cf",
                        "current_min_p_nom_max",
                        "current_min_p_max_pu",
                        "keep_nom_current",
                        "keep_cf_current",
                        "keep_both_current",
                    ]
                ].copy()
            )
            quantile_output.extend(quantile_rows(cells["p_nom_max_mw"].to_numpy(float), "p_nom_max_mw", technology))
            quantile_output.extend(quantile_rows(cells["mean_cell_cf"].to_numpy(float), "mean_cell_cf", technology))
            retention_output.extend(retention_rows(cells, "p_nom_max_mw", P_NOM_SWEEPS[technology], "p_nom_max_mw", technology))
            retention_output.extend(retention_rows(cells, "mean_cell_cf", CF_SWEEPS[technology], "mean_cell_cf", technology))
            current_threshold_output.extend(current_threshold_rows(cells, technology, threshold_nom, threshold_cf))

    cell_stats = pd.concat(all_cell_stats, ignore_index=True)
    quantiles = pd.DataFrame(quantile_output)
    retention = pd.DataFrame(retention_output)
    current_thresholds = pd.DataFrame(current_threshold_output)

    cell_stats_path = output_dir / "cell_filter_cell_stats.csv"
    quantiles_path = output_dir / "cell_filter_quantiles.csv"
    retention_path = output_dir / "cell_filter_threshold_retention.csv"
    current_thresholds_path = output_dir / "cell_filter_current_thresholds.csv"

    LOG.info("writing CSV outputs")
    cell_stats.to_csv(cell_stats_path, index=False)
    quantiles.to_csv(quantiles_path, index=False)
    retention.to_csv(retention_path, index=False)
    current_thresholds.to_csv(current_thresholds_path, index=False)

    LOG.info("writing plots")
    save_distribution_plot(
        cell_stats,
        value_column="p_nom_max_mw",
        threshold_column="current_min_p_nom_max",
        title="Installable capacity distribution per cell",
        x_label="P_nom_max [MW per cell]",
        output_path=output_dir / "p_nom_max_distribution_by_technology.png",
        log_x=True,
    )
    save_distribution_plot(
        cell_stats,
        value_column="mean_cell_cf",
        threshold_column="current_min_p_max_pu",
        title="Mean cell capacity factor distribution",
        x_label="Mean cell capacity factor [-]",
        output_path=output_dir / "mean_cell_cf_distribution_by_technology.png",
        log_x=False,
    )
    save_retention_plot(retention, output_dir / "cell_filter_threshold_retention.png")
    save_current_threshold_bar_plot(current_thresholds, output_dir / "cell_filter_current_threshold_effect.png")
    LOG.info("done")


if __name__ == "__main__":
    main()
