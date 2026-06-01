from __future__ import annotations

"""Run the renewable regionalisation workflow for configured scenario cases.

The runner coordinates capacity preprocessing and generation-profile scaling for
one or more Atlite-like raster cases. It is intentionally thin: methodological
choices live in the called modules, while this file resolves paths, applies case
names, and keeps the scenario directory layout reproducible.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from res_common import (
    DEFAULT_END_YEAR,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_START_YEAR,
    build_simulation_case_dir,
    format_threshold_value,
    load_yaml_config,
    parse_threshold_value,
    parse_optional_float,
    resolve_cli_path,
    resolve_path,
    write_json,
)

LOG = logging.getLogger(__name__)

DEFAULT_CASES = [
    "corine_luisa_wdpa_onoff_acdc",
    "corine_luisa_wdpa_on_acdc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RES capacity preprocessing and generation disaggregation for multiple Atlite cases."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--network-dir", type=Path, default=None)
    parser.add_argument("--target-year", type=int, default=None)
    parser.add_argument(
        "--atlite-base-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--atlite-cases", nargs="+", default=None)
    parser.add_argument("--generation-long-csv", type=Path, default=None)
    parser.add_argument("--target-capacity-csv", type=Path, default=None)
    parser.add_argument("--excluded-countries-csv", type=Path, default=None)
    parser.add_argument("--pv-power-csv", type=Path, default=None)
    parser.add_argument("--onwind-power-csv", type=Path, default=None)
    parser.add_argument("--offwind-power-csv", type=Path, default=None)
    parser.add_argument(
        "--offshore-eez-geojson",
        type=Path,
        default=None,
    )
    parser.add_argument("--onshore-mask-geojson", action="append", default=None)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--min-p-max-pu", default=None)
    parser.add_argument("--min-p-nom-max", default=None)
    parser.add_argument("--min-distance-offshore-km", type=float, default=None)
    parser.add_argument("--max-distance-offshore-km", type=float, default=None)
    parser.add_argument("--plot-weather-year", type=int, default=None)
    parser.add_argument(
        "--plot-bounds",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=None,
    )
    parser.add_argument("--skip-plot-comparison", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def default_settings() -> dict[str, object]:
    return {
        "project_root": str(DEFAULT_PROJECT_ROOT),
        "network_dir": None,
        "target_year": None,
        "atlite_base_dir": str(DEFAULT_PROJECT_ROOT / "renewables" / "atlite_copy"),
        "atlite_cases": list(DEFAULT_CASES),
        "generation_long_csv": None,
        "target_capacity_csv": None,
        "excluded_countries_csv": None,
        "pv_power_csv": None,
        "onwind_power_csv": None,
        "offwind_power_csv": None,
        "offshore_eez_geojson": str(
            DEFAULT_PROJECT_ROOT / "datashapes" / "eez_offshore_eu27_uk_no_europe_only.geojson"
        ),
        "onshore_mask_geojsons": [str(DEFAULT_PROJECT_ROOT / "datashapes" / "europe_shape.geojson")],
        "start_year": DEFAULT_START_YEAR,
        "end_year": DEFAULT_END_YEAR,
        "min_p_max_pu": 0.0,
        "min_p_nom_max": 0.0,
        "min_distance_offshore_km": None,
        "max_distance_offshore_km": None,
        "plot_weather_year": 2013,
        "plot_bounds": None,
        "skip_plot_comparison": False,
        "overwrite": False,
    }


def resolve_settings(args: argparse.Namespace) -> dict[str, object]:
    settings = default_settings()
    config_dir = None
    if args.config is not None:
        cfg_path = resolve_cli_path(args.config)
        assert cfg_path is not None
        settings.update(load_yaml_config(cfg_path))
        config_dir = cfg_path.parent

    overrides = {
        "project_root": resolve_cli_path(args.project_root),
        "network_dir": resolve_cli_path(args.network_dir),
        "target_year": args.target_year,
        "atlite_base_dir": resolve_cli_path(args.atlite_base_dir),
        "atlite_cases": args.atlite_cases,
        "generation_long_csv": resolve_cli_path(args.generation_long_csv),
        "target_capacity_csv": resolve_cli_path(args.target_capacity_csv),
        "excluded_countries_csv": resolve_cli_path(args.excluded_countries_csv),
        "pv_power_csv": resolve_cli_path(args.pv_power_csv),
        "onwind_power_csv": resolve_cli_path(args.onwind_power_csv),
        "offwind_power_csv": resolve_cli_path(args.offwind_power_csv),
        "offshore_eez_geojson": resolve_cli_path(args.offshore_eez_geojson),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "min_p_max_pu": args.min_p_max_pu,
        "min_p_nom_max": args.min_p_nom_max,
        "min_distance_offshore_km": args.min_distance_offshore_km,
        "max_distance_offshore_km": args.max_distance_offshore_km,
        "plot_weather_year": args.plot_weather_year,
        "plot_bounds": args.plot_bounds,
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    if args.onshore_mask_geojson:
        settings["onshore_mask_geojsons"] = [resolve_cli_path(path) for path in args.onshore_mask_geojson]
    settings["skip_plot_comparison"] = args.skip_plot_comparison or bool(settings.get("skip_plot_comparison", False))
    settings["overwrite"] = args.overwrite or bool(settings.get("overwrite", False))

    project_root = resolve_path(settings.get("project_root"), base_dir=config_dir) or DEFAULT_PROJECT_ROOT
    settings["project_root"] = project_root
    network_dir = resolve_path(settings.get("network_dir"), base_dir=project_root)
    if network_dir is None:
        raise ValueError("network_dir is required.")
    settings["network_dir"] = network_dir
    settings["target_year"] = infer_target_year(network_dir, settings.get("target_year"))
    if settings["target_year"] is None:
        raise ValueError("Could not infer target year from network_dir. Pass target_year explicitly.")

    atlite_base_dir = resolve_path(settings.get("atlite_base_dir"), base_dir=project_root)
    if atlite_base_dir is None:
        atlite_base_dir = project_root / "renewables" / "atlite_copy"
    settings["atlite_base_dir"] = atlite_base_dir

    atlite_cases = settings.get("atlite_cases")
    if atlite_cases in (None, ""):
        settings["atlite_cases"] = list(DEFAULT_CASES)
    elif isinstance(atlite_cases, str):
        settings["atlite_cases"] = [part.strip() for part in atlite_cases.split(",") if part.strip()]
    else:
        settings["atlite_cases"] = [str(case).strip() for case in atlite_cases if str(case).strip()]
    if not settings["atlite_cases"]:
        raise ValueError("atlite_cases must contain at least one case name.")

    target_year = int(settings["target_year"])
    generation_long_csv = resolve_path(settings.get("generation_long_csv"), base_dir=project_root)
    if generation_long_csv is None:
        generation_long_csv = project_root / "renewables" / f"res_load_country_long_{target_year}_tyndp2024.csv"
    settings["generation_long_csv"] = generation_long_csv

    target_capacity_csv = resolve_path(settings.get("target_capacity_csv"), base_dir=project_root)
    if target_capacity_csv is None:
        target_capacity_csv = project_root / "renewables" / f"res_generation_mapping_diag_{target_year}_tyndp2024.csv"
    settings["target_capacity_csv"] = target_capacity_csv
    excluded_countries_csv = resolve_path(settings.get("excluded_countries_csv"), base_dir=project_root)
    if excluded_countries_csv is None:
        candidate = network_dir / "excluded_countries.csv"
        excluded_countries_csv = candidate if candidate.exists() else None
    settings["excluded_countries_csv"] = excluded_countries_csv
    settings["pv_power_csv"] = resolve_path(settings.get("pv_power_csv"), base_dir=project_root)
    settings["onwind_power_csv"] = resolve_path(settings.get("onwind_power_csv"), base_dir=project_root)
    settings["offwind_power_csv"] = resolve_path(settings.get("offwind_power_csv"), base_dir=project_root)

    offshore_eez_geojson = resolve_path(settings.get("offshore_eez_geojson"), base_dir=project_root)
    if offshore_eez_geojson is None:
        offshore_eez_geojson = project_root / "datashapes" / "eez_offshore_eu27_uk_no_europe_only.geojson"
    settings["offshore_eez_geojson"] = offshore_eez_geojson

    onshore_masks_raw = settings.get("onshore_mask_geojsons")
    if onshore_masks_raw in (None, ""):
        onshore_masks_raw = [project_root / "datashapes" / "europe_shape.geojson"]
    onshore_masks: list[Path] = []
    for path in onshore_masks_raw:
        resolved = resolve_path(path, base_dir=project_root)
        if resolved is not None:
            onshore_masks.append(resolved)
    settings["onshore_mask_geojsons"] = onshore_masks
    if not settings["onshore_mask_geojsons"]:
        raise ValueError("onshore_mask_geojsons must contain at least one path.")

    settings["min_p_max_pu"] = parse_threshold_value(settings.get("min_p_max_pu")) or 0.0
    settings["min_p_nom_max"] = parse_threshold_value(settings.get("min_p_nom_max")) or 0.0
    settings["min_distance_offshore_km"] = parse_optional_float(
        settings.get("min_distance_offshore_km"),
        "min_distance_offshore_km",
    )
    settings["max_distance_offshore_km"] = parse_optional_float(
        settings.get("max_distance_offshore_km"),
        "max_distance_offshore_km",
    )
    if (
        settings["min_distance_offshore_km"] is not None
        and settings["max_distance_offshore_km"] is not None
        and settings["min_distance_offshore_km"] > settings["max_distance_offshore_km"]
    ):
        raise ValueError("min_distance_offshore_km must be <= max_distance_offshore_km.")
    if settings.get("plot_bounds") in (None, ""):
        settings["plot_bounds"] = None
    else:
        plot_bounds = settings["plot_bounds"]
        if isinstance(plot_bounds, str):
            plot_bounds = [part.strip() for part in plot_bounds.split(",") if part.strip()]
        if len(plot_bounds) != 4:
            raise ValueError("plot_bounds must contain exactly four values: min_lon, min_lat, max_lon, max_lat.")
        settings["plot_bounds"] = [float(value) for value in plot_bounds]
    return settings


def infer_target_year(network_dir: Path, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    for part in network_dir.parts:
        if part.startswith("target_year_"):
            return int(part.rsplit("_", 1)[-1])
    raise ValueError("Could not infer target year from network_dir. Pass --target-year.")


def run_subprocess(command: list[str]) -> None:
    LOG.info("running: %s", " ".join(f'"{part}"' if " " in part else part for part in command))
    subprocess.run(command, check=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = resolve_settings(parse_args())

    project_root = Path(settings["project_root"]).resolve()
    network_dir = Path(settings["network_dir"]).resolve()
    target_year = int(settings["target_year"])
    min_p_max_pu = settings["min_p_max_pu"]
    min_p_nom_max = settings["min_p_nom_max"]
    min_distance_offshore_km = settings["min_distance_offshore_km"]
    max_distance_offshore_km = settings["max_distance_offshore_km"]
    atlite_base_dir = Path(settings["atlite_base_dir"]).resolve()
    generation_long_csv = Path(settings["generation_long_csv"]).resolve()
    target_capacity_csv = Path(settings["target_capacity_csv"]).resolve()
    pv_power_csv = Path(settings["pv_power_csv"]).resolve() if settings.get("pv_power_csv") is not None else None
    onwind_power_csv = (
        Path(settings["onwind_power_csv"]).resolve() if settings.get("onwind_power_csv") is not None else None
    )
    offwind_power_csv = (
        Path(settings["offwind_power_csv"]).resolve() if settings.get("offwind_power_csv") is not None else None
    )
    offshore_eez_geojson = Path(settings["offshore_eez_geojson"]).resolve()
    onshore_masks = [Path(path).resolve() for path in settings["onshore_mask_geojsons"]]
    start_year = int(settings["start_year"])
    end_year = int(settings["end_year"])
    plot_weather_year = int(settings["plot_weather_year"])
    plot_bounds = settings["plot_bounds"]
    skip_plot_comparison = bool(settings["skip_plot_comparison"])
    overwrite = bool(settings["overwrite"])

    script_dir = Path(__file__).resolve().parent
    capacity_script = script_dir / "res_capacity_preprocessing.py"
    generation_script = script_dir / "res_generation_disaggregation.py"
    compare_plot_script = script_dir / "plot_pypsa_vs_ours_compare.py"
    shared_voronoi_geojson = (
        project_root
        / "datashapes"
        / "generated_regions"
        / f"{network_dir.parent.name}_{network_dir.name}_onshore_voronoi.geojson"
    )

    summary: dict[str, dict[str, str]] = {}

    for index, case_name in enumerate(settings["atlite_cases"]):
        case_dir = (atlite_base_dir / case_name).resolve()
        if not case_dir.exists():
            raise FileNotFoundError(f"Atlite case directory does not exist: {case_dir}")

        simulation_dir = project_root / "renewables"
        case_output_dir = build_simulation_case_dir(simulation_dir, network_dir, case_dir)
        preprocessed_dir = case_output_dir / "res_bus_cap_preprocessed"
        disaggregated_dir = case_output_dir / "disaggregated"
        comparison_dir = case_output_dir / "plot_comparison"
        case_overwrite = bool(overwrite and (index == 0 or not shared_voronoi_geojson.exists()))

        capacity_cmd = [
            sys.executable,
            str(capacity_script),
            "--project-root",
            str(project_root),
            "--network-dir",
            str(network_dir),
            "--target-year",
            str(target_year),
            "--simulation-dir",
            str(simulation_dir),
            "--atlite-case-dir",
            str(case_dir),
            "--target-capacity-csv",
            str(target_capacity_csv),
            "--offshore-eez-geojson",
            str(offshore_eez_geojson),
            "--onshore-voronoi-geojson",
            str(shared_voronoi_geojson),
            "--generated-onshore-voronoi-geojson",
            str(shared_voronoi_geojson),
            "--output-dir",
            str(preprocessed_dir),
            "--start-year",
            str(start_year),
            "--end-year",
            str(end_year),
        ]
        if pv_power_csv is not None:
            capacity_cmd.extend(["--pv-power-csv", str(pv_power_csv)])
        if onwind_power_csv is not None:
            capacity_cmd.extend(["--onwind-power-csv", str(onwind_power_csv)])
        if offwind_power_csv is not None:
            capacity_cmd.extend(["--offwind-power-csv", str(offwind_power_csv)])
        if settings.get("excluded_countries_csv") is not None:
            capacity_cmd.extend(["--excluded-countries-csv", str(settings["excluded_countries_csv"])])
        for mask_path in onshore_masks:
            capacity_cmd.extend(["--onshore-mask-geojson", str(mask_path)])
        if min_p_max_pu != 0.0:
            capacity_cmd.extend(["--min-p-max-pu", format_threshold_value(min_p_max_pu)])
        if min_p_nom_max != 0.0:
            capacity_cmd.extend(["--min-p-nom-max", format_threshold_value(min_p_nom_max)])
        if min_distance_offshore_km is not None:
            capacity_cmd.extend(["--min-distance-offshore-km", str(min_distance_offshore_km)])
        if max_distance_offshore_km is not None:
            capacity_cmd.extend(["--max-distance-offshore-km", str(max_distance_offshore_km)])
        if case_overwrite:
            capacity_cmd.append("--overwrite")
        run_subprocess(capacity_cmd)

        generation_cmd = [
            sys.executable,
            str(generation_script),
            "--project-root",
            str(project_root),
            "--network-dir",
            str(network_dir),
            "--target-year",
            str(target_year),
            "--simulation-dir",
            str(simulation_dir),
            "--atlite-case-dir",
            str(case_dir),
            "--capacity-output-dir",
            str(preprocessed_dir),
            "--generation-long-csv",
            str(generation_long_csv),
            "--output-dir",
            str(disaggregated_dir),
            "--start-year",
            str(start_year),
            "--end-year",
            str(end_year),
        ]
        if settings.get("excluded_countries_csv") is not None:
            generation_cmd.extend(["--excluded-countries-csv", str(settings["excluded_countries_csv"])])
        run_subprocess(generation_cmd)

        if not skip_plot_comparison:
            compare_cmd = [
                sys.executable,
                str(compare_plot_script),
                "--project-root",
                str(project_root),
                "--atlite-case-dir",
                str(case_dir),
                "--simulation-dir",
                str(simulation_dir),
                "--network-dir",
                str(network_dir),
                "--weather-year",
                str(plot_weather_year),
                "--start-year",
                str(start_year),
                "--end-year",
                str(end_year),
                "--output-dir",
                str(comparison_dir),
            ]
            if min_p_max_pu != 0.0:
                compare_cmd.extend(["--min-p-max-pu", format_threshold_value(min_p_max_pu)])
            if min_p_nom_max != 0.0:
                compare_cmd.extend(["--min-p-nom-max", format_threshold_value(min_p_nom_max)])
            if min_distance_offshore_km is not None:
                compare_cmd.extend(["--min-distance-offshore-km", str(min_distance_offshore_km)])
            if max_distance_offshore_km is not None:
                compare_cmd.extend(["--max-distance-offshore-km", str(max_distance_offshore_km)])
            if min_distance_offshore_km is not None or max_distance_offshore_km is not None:
                for mask_path in onshore_masks:
                    compare_cmd.extend(["--offshore-distance-geojson", str(mask_path)])
            if plot_bounds is not None:
                compare_cmd.extend(["--plot-bounds", *(str(value) for value in plot_bounds)])
            run_subprocess(compare_cmd)

        summary[case_name] = {
            "atlite_case_dir": str(case_dir),
            "simulation_dir": str(simulation_dir),
            "preprocessed_dir": str(preprocessed_dir),
            "disaggregated_dir": str(disaggregated_dir),
            "plot_comparison_dir": str(comparison_dir),
            "onshore_voronoi_geojson": str(shared_voronoi_geojson),
            "excluded_countries_csv": (
                str(settings["excluded_countries_csv"]) if settings.get("excluded_countries_csv") is not None else None
            ),
        }

    manifest_path = project_root / "renewables" / network_dir.parent.name / network_dir.name / (
        f"res_workflow_cases_manifest_{target_year}.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        manifest_path,
        {
            "project_root": str(project_root),
            "network_dir": str(network_dir),
            "target_year": target_year,
            "generation_long_csv": str(generation_long_csv),
            "target_capacity_csv": str(target_capacity_csv),
            "pv_power_csv": str(pv_power_csv) if pv_power_csv is not None else None,
            "onwind_power_csv": str(onwind_power_csv) if onwind_power_csv is not None else None,
            "offwind_power_csv": str(offwind_power_csv) if offwind_power_csv is not None else None,
            "start_year": start_year,
            "end_year": end_year,
            "min_p_max_pu": min_p_max_pu,
            "min_p_nom_max": min_p_nom_max,
            "min_distance_offshore_km": min_distance_offshore_km,
            "max_distance_offshore_km": max_distance_offshore_km,
            "plot_weather_year": plot_weather_year,
            "plot_bounds": plot_bounds,
            "skip_plot_comparison": skip_plot_comparison,
            "cases": summary,
        },
    )
    LOG.info("wrote %s", manifest_path)


if __name__ == "__main__":
    main()
