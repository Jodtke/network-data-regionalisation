from __future__ import annotations

"""Allocate national hydro capacities, constraints, and inflows to buses.

Hydropower is treated separately from other plant technologies because capacity,
storage volume, operating constraints, and natural inflows must remain mutually
consistent. The allocation uses current hydro sites where available, separates
turbine-capacity shares from storage-energy shares, and falls back to strongest
hydro sites or load shares only when the site basis is insufficient.

Open-loop and closed-loop pumped storage are handled explicitly. Natural inflows
are assigned only to technologies that can receive them, and missing TYNDP weekly
inflow series can be imputed from the self-generated Atlite-like hydro profiles
while keeping the selected source visible in the output diagnostics.
"""

import argparse
import csv
import importlib
import importlib.util
import json
import math
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable


COUNTRY_ALIASES = {"UK": "GB"}
WEEKS = tuple(range(1, 53))
WEATHER_YEARS = tuple(range(1982, 2017))
EPS = 1e-9
NC_HYDRO_TYPE_BY_PLANT_TYPE = {"ror": "ror", "wr": "reservoir", "phs": "phs"}
PATH_ARG_NAMES = {
    "project_root", "hydro_dir", "network_dir", "load_csv", "output_dir",
    "country_clusters_csv", "country_reductions_csv", "country_profiles_dir",
    "excluded_countries_csv",
}
BOOL_ARG_NAMES = {"include_inflows", "audit_only", "resolve_phs"}

CAPACITY_FIELDS = [
    "country", "country_model", "country_label", "bus", "ref_year", "plant_type", "technology",
    "current_turb_mw", "current_storage_mwh", "target_turb_mw", "target_storage_mwh",
    "turbine_share", "storage_share", "allocation_rule",
]
CONSTRAINT_FIELDS = [
    "country", "country_model", "country_label", "bus", "ref_year", "plant_type", "technology",
    "temporal_resolution", "week", "period_start_date", "period_end_date", "days_in_period",
    "installed_turb_mw", "installed_pump_mw", "installed_storage_mwh",
    "min_turb_mw", "max_turb_mw", "min_turb_pu", "max_turb_pu",
    "min_pump_mw", "max_pump_mw", "min_pump_pu", "max_pump_pu",
    "min_turb_en_mwh_day", "max_turb_en_mwh_day", "min_pump_en_mwh_day", "max_pump_en_mwh_day",
    "min_turb_en_mwh_period", "max_turb_en_mwh_period", "min_pump_en_mwh_period", "max_pump_en_mwh_period",
    "min_res_hist_pu", "max_res_hist_pu", "min_res_tech_pu", "max_res_tech_pu",
    "turbine_share", "storage_share", "allocation_rule",
]
INFLOW_FIELDS = [
    "country", "country_model", "country_label", "bus", "ref_year", "weather_year", "week",
    "plant_type", "technology", "national_inflow_source", "national_inflow_mwh_week",
    "bus_inflow_total_mwh_week", "allocated_inflow_mwh_week",
    "bus_total_storage_share", "bus_tech_turbine_share", "allocation_rule",
]
SHARE_FIELDS = [
    "country", "country_model", "country_label", "bus", "plant_type", "technology",
    "current_turb_mw", "current_storage_mwh", "turbine_share", "storage_share",
    "load_share_within_country", "allocation_rule",
]


@dataclass(frozen=True)
class BusMeta:
    bus_id: str
    country_model: str
    country_label: str


@dataclass(frozen=True)
class CountryClusterMap:
    source_to_target: dict[str, str]
    target_to_sources: dict[str, tuple[str, ...]]
    target_to_label: dict[str, str]


@dataclass(frozen=True)
class NcCountryProfilesMeta:
    countries: tuple[str, ...]
    hydro_types: tuple[str, ...]
    member_countries: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Disaggregate hydro country data to reduced network buses.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--target-year", type=int, default=None)
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--hydro-dir", type=Path, default=None)
    p.add_argument("--network-dir", type=Path, default=None)
    p.add_argument("--load-csv", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--country-clusters-csv", type=Path, default=None)
    p.add_argument("--country-reductions-csv", type=Path, default=None)
    p.add_argument("--country-profiles-dir", type=Path, default=None)
    p.add_argument("--excluded-countries-csv", type=Path, default=None)
    resolve_group = p.add_mutually_exclusive_group()
    resolve_group.add_argument("--resolve-phs", dest="resolve_phs", action="store_true", default=None)
    resolve_group.add_argument("--no-resolve-phs", dest="resolve_phs", action="store_false")
    include_group = p.add_mutually_exclusive_group()
    include_group.add_argument("--include-inflows", dest="include_inflows", action="store_true", default=None)
    include_group.add_argument("--no-include-inflows", dest="include_inflows", action="store_false")
    audit_group = p.add_mutually_exclusive_group()
    audit_group.add_argument("--audit-only", dest="audit_only", action="store_true", default=None)
    audit_group.add_argument("--no-audit-only", dest="audit_only", action="store_false")
    return p.parse_args()


def normalize_country(value: object) -> str:
    code = str(value or "").strip().upper()
    return COUNTRY_ALIASES.get(code, code)


def strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    out = []
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).rstrip()


def parse_yaml_scalar(value: str) -> Any:
    text = value.strip()
    if text == "":
        return ""
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", text):
        return float(text)
    return text


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(0, root)]
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = strip_yaml_comment(raw_line)
        if not stripped.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if "\t" in raw_line[:indent]:
            raise ValueError(f"Tabs are not supported in YAML config: {path}:{line_no}")
        key_value = stripped.lstrip(" ")
        if ":" not in key_value:
            raise ValueError(f"Expected 'key: value' in YAML config: {path}:{line_no}")
        key, value = key_value.split(":", 1)
        key = key.strip().replace("-", "_")
        value = value.strip()
        while len(stack) > 1 and indent < stack[-1][0]:
            stack.pop()
        if indent != stack[-1][0]:
            raise ValueError(f"Unsupported indentation in YAML config: {path}:{line_no}")
        target = stack[-1][1]
        if value == "":
            nested: dict[str, Any] = {}
            target[key] = nested
            stack.append((indent + 2, nested))
        else:
            target[key] = parse_yaml_scalar(value)
    return root


def load_yaml_config(path: Path) -> dict[str, Any]:
    yaml_spec = importlib.util.find_spec("yaml")
    if yaml_spec is not None:
        yaml = importlib.import_module("yaml")
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Top-level YAML content must be a mapping: {path}")
        return {str(key).replace("-", "_"): value for key, value in payload.items()}
    return parse_simple_yaml(path)


def flatten_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            merged[key] = value
    for section_name in ("common", "disaggregation", "hydro_bus_disaggregation", "run"):
        section = payload.get(section_name)
        if isinstance(section, dict):
            for key, value in section.items():
                merged[str(key).replace("-", "_")] = value
    return merged


def coerce_config_value(name: str, value: Any, config_path: Path) -> Any:
    if name in PATH_ARG_NAMES and value is not None:
        path = Path(str(value))
        return path if path.is_absolute() else (config_path.parent / path)
    if name in BOOL_ARG_NAMES and value is not None:
        return bool(value)
    if name == "target_year" and value is not None:
        return int(value)
    return value


def resolve_args() -> argparse.Namespace:
    args = parse_args()
    config_path = args.config.resolve() if args.config else None
    if config_path is not None:
        payload = load_yaml_config(config_path)
        merged = flatten_config_payload(payload)
        for key, value in merged.items():
            if not hasattr(args, key):
                continue
            if getattr(args, key) is None:
                setattr(args, key, coerce_config_value(key, value, config_path))
    if args.target_year is None:
        raise ValueError("target_year must be provided via --target-year or --config.")
    if args.project_root is None:
        args.project_root = Path.cwd()
    for name in PATH_ARG_NAMES:
        value = getattr(args, name)
        if value is not None and not isinstance(value, Path):
            setattr(args, name, Path(str(value)))
    if args.resolve_phs is None:
        args.resolve_phs = True
    if args.include_inflows is None:
        args.include_inflows = False
    if args.audit_only is None:
        args.audit_only = False
    return args


def split_countries(value: object) -> list[str]:
    text = str(value or "").strip()
    out: list[str] = []
    seen: set[str] = set()
    for part in text.split(","):
        code = normalize_country(part)
        if not code or code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def parse_float(value: object, default: float = 0.0) -> float:
    text = str(value or "").strip()
    return default if text in ("", "NA", "None", "nan", "NaN") else float(text)


def parse_optional_float(value: object) -> float | None:
    text = str(value or "").strip()
    return None if text in ("", "NA", "None", "nan", "NaN") else float(text)


def fmt(value: float, digits: int = 6) -> str:
    if abs(value) < 5e-13:
        value = 0.0
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def fmt_amount(value: float) -> str:
    return fmt(value, 0)


def fmt_share(value: float) -> str:
    if abs(value) < 5e-13:
        value = 0.0
    return f"{value:.3f}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as h:
        header = h.readline()
    counts = {sep: header.count(sep) for sep in (";", ",", "\t")}
    sep, count = max(counts.items(), key=lambda x: x[1])
    return sep if count > 0 else ","


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as h:
        return list(csv.DictReader(h, delimiter=detect_delimiter(path)))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_tick_value(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1000 or value == int(value):
        return f"{value:,.0f}".replace(",", " ")
    if abs_value >= 100:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def nice_step(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = int(f"{value:e}".split("e")[1])
    fraction = value / (10 ** exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10 ** exponent)


def build_y_ticks(vmax: float, approx_count: int = 5) -> list[float]:
    if vmax <= 0:
        return [0.0, 1.0]
    step = nice_step(vmax / max(approx_count, 1))
    tick_max = step * int((vmax + step - EPS) // step)
    if tick_max < vmax:
        tick_max += step
    ticks = [0.0]
    current = step
    while current <= tick_max + EPS:
        ticks.append(current)
        current += step
    if ticks[-1] < vmax:
        ticks.append(tick_max + step)
    return ticks


def build_x_ticks(minx: int, maxx: int) -> list[int]:
    if minx == maxx:
        return [minx]
    preferred = [1, 13, 26, 39, 52]
    ticks = [tick for tick in preferred if minx <= tick <= maxx]
    if minx not in ticks:
        ticks.insert(0, minx)
    if maxx not in ticks:
        ticks.append(maxx)
    return sorted(set(ticks))


def write_bar_svg(path: Path, title: str, labels: list[str], values: list[float], x_label: str = "", y_label: str = "") -> None:
    ensure_dir(path.parent)
    if not values:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='400' height='100'><text x='20' y='40'>No data</text></svg>", encoding="utf-8")
        return
    width, height, left, right, top, bottom = 1200, 700, 110, 40, 60, 170
    pw, ph = width - left - right, height - top - bottom
    vmax = max(values) or 1.0
    y_ticks = build_y_ticks(vmax)
    y_tick_max = y_ticks[-1] or 1.0
    bw = pw / max(len(values), 1)
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<style>text{font-family:Arial,sans-serif;font-size:12px}.title{font-size:18px;font-weight:bold}</style>",
        f"<text class='title' x='{left}' y='30'>{title}</text>",
        f"<line x1='{left}' y1='{top+ph}' x2='{width-right}' y2='{top+ph}' stroke='black'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+ph}' stroke='black'/>",
    ]
    for tick in y_ticks:
        y = top + ph - ph * (tick / y_tick_max)
        parts.append(f"<line x1='{left-6}' y1='{y:.1f}' x2='{left}' y2='{y:.1f}' stroke='black'/>")
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' stroke='#d9d9d9' stroke-dasharray='3,3'/>")
        parts.append(f"<text x='{left-10}' y='{y+4:.1f}' text-anchor='end'>{format_tick_value(tick)}</text>")
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + i * bw + 4
        bh = 0.0 if y_tick_max <= 0 else ph * (value / y_tick_max)
        y = top + ph - bh
        parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{max(bw-8,1):.1f}' height='{bh:.1f}' fill='#4C78A8'/>")
        tx, ty = x + max(bw-8,1) / 2.0, top + ph + 14
        parts.append(f"<text transform='rotate(60 {tx:.1f},{ty:.1f})' x='{tx:.1f}' y='{ty:.1f}'>{label}</text>")
    if x_label:
        parts.append(f"<text x='{left + pw / 2:.1f}' y='{height - 20}' text-anchor='middle'>{x_label}</text>")
    if y_label:
        parts.append(f"<text transform='rotate(-90 22,{top + ph / 2:.1f})' x='22' y='{top + ph / 2:.1f}' text-anchor='middle'>{y_label}</text>")
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf-8")


def write_line_svg(path: Path, title: str, xvals: list[int], series: list[tuple[str, list[float]]], x_label: str = "", y_label: str = "") -> None:
    ensure_dir(path.parent)
    if not xvals or not series:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='400' height='100'><text x='20' y='40'>No data</text></svg>", encoding="utf-8")
        return
    width, height, left, right, top, bottom = 1200, 700, 110, 160, 60, 90
    pw, ph = width - left - right, height - top - bottom
    minx, maxx = min(xvals), max(xvals)
    xrange = max(maxx - minx, 1)
    vmax = max((max(vals) for _, vals in series if vals), default=1.0) or 1.0
    y_ticks = build_y_ticks(vmax)
    y_tick_max = y_ticks[-1] or 1.0
    x_ticks = build_x_ticks(minx, maxx)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<style>text{font-family:Arial,sans-serif;font-size:12px}.title{font-size:18px;font-weight:bold}</style>",
        f"<text class='title' x='{left}' y='30'>{title}</text>",
        f"<line x1='{left}' y1='{top+ph}' x2='{width-right}' y2='{top+ph}' stroke='black'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top+ph}' stroke='black'/>",
    ]
    for tick in y_ticks:
        y = top + ph - ph * (tick / y_tick_max)
        parts.append(f"<line x1='{left-6}' y1='{y:.1f}' x2='{left}' y2='{y:.1f}' stroke='black'/>")
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' stroke='#d9d9d9' stroke-dasharray='3,3'/>")
        parts.append(f"<text x='{left-10}' y='{y+4:.1f}' text-anchor='end'>{format_tick_value(tick)}</text>")
    for tick in x_ticks:
        x = left + pw * ((tick - minx) / xrange)
        parts.append(f"<line x1='{x:.1f}' y1='{top+ph}' x2='{x:.1f}' y2='{top+ph+6}' stroke='black'/>")
        parts.append(f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{top+ph}' stroke='#efefef' stroke-dasharray='3,3'/>")
        parts.append(f"<text x='{x:.1f}' y='{top+ph+20}' text-anchor='middle'>{tick}</text>")
    for idx, (label, vals) in enumerate(series):
        color = colors[idx % len(colors)]
        pts = []
        for x, y in zip(xvals, vals):
            px = left + pw * ((x - minx) / xrange)
            py = top + ph - (0.0 if y_tick_max <= 0 else ph * (y / y_tick_max))
            pts.append(f"{px:.1f},{py:.1f}")
        parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(pts)}'/>")
        ly = top + 20 + idx * 20
        parts.append(f"<line x1='{width-right+10}' y1='{ly}' x2='{width-right+34}' y2='{ly}' stroke='{color}' stroke-width='3'/>")
        parts.append(f"<text x='{width-right+40}' y='{ly+4}'>{label}</text>")
    if x_label:
        parts.append(f"<text x='{left + pw / 2:.1f}' y='{height - 20}' text-anchor='middle'>{x_label}</text>")
    if y_label:
        parts.append(f"<text transform='rotate(-90 22,{top + ph / 2:.1f})' x='22' y='{top + ph / 2:.1f}' text-anchor='middle'>{y_label}</text>")
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf-8")


def empty_cluster_map() -> CountryClusterMap:
    return CountryClusterMap(source_to_target={}, target_to_sources={}, target_to_label={})


def read_country_clusters(path: Path | None) -> CountryClusterMap:
    if path is None or not path.exists():
        return empty_cluster_map()
    source_to_target: dict[str, str] = {}
    target_to_sources: dict[str, set[str]] = defaultdict(set)
    target_to_label: dict[str, str] = {}
    for row in read_csv(path):
        source = normalize_country(row.get("source_country"))
        target = normalize_country(row.get("target_country"))
        label = str(row.get("target_label") or target).strip() or target
        if not source or not target:
            continue
        source_to_target[source] = target
        target_to_sources[target].add(source)
        target_to_label[target] = label
    return CountryClusterMap(
        source_to_target=source_to_target,
        target_to_sources={key: tuple(sorted(values)) for key, values in target_to_sources.items()},
        target_to_label=target_to_label,
    )


def read_excluded_countries(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    rows = read_csv(path)
    if not rows:
        return set()
    country_col = "source_country" if "source_country" in rows[0] else ("country" if "country" in rows[0] else None)
    if country_col is None:
        return set()
    return {normalize_country(row.get(country_col)) for row in rows if normalize_country(row.get(country_col))}


def map_to_model_country(source_country: str, cluster_map: CountryClusterMap) -> str:
    return cluster_map.source_to_target.get(source_country, source_country)


def label_for_model_country(model_country: str, cluster_map: CountryClusterMap) -> str:
    return cluster_map.target_to_label.get(model_country, model_country)


def infer_network_dir(root: Path, year: int) -> Path:
    return root / "grid" / f"target_year_{year}" / "electrical_spectral_line_equivalent_dc_effective_reactance"


def infer_country_profiles_dir(root: Path) -> Path:
    return root / "hydro" / "atlite_copy" / "country_profiles"


def auto_load_csv(root: Path, year: int, network_dir: Path) -> Path:
    matches = sorted((root / "load" / f"target_year_{year}" / network_dir.name).glob("disaggregated_load_country_bus_load_*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one load CSV, found {len(matches)}.")
    return matches[0]


def find_hydro_csv(hydro_dir: Path, prefix: str, year: int) -> Path:
    matches = sorted(hydro_dir.glob(f"{prefix}_{year}*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {prefix} file for {year}, found {len(matches)}.")
    return matches[0]


def week_bounds(year: int, week: int) -> tuple[str, str, str]:
    start = date(year, 1, 1) + timedelta(days=(week - 1) * 7)
    end = start + timedelta(days=6)
    return start.isoformat(), end.isoformat(), "7"


def find_country_profile_file(country_profiles_dir: Path, weather_year: int) -> Path:
    matches = []
    for pattern in (f"hydro_country_profiles_{weather_year}.nc", f"hydro_country_profiles_{weather_year}.nc4"):
        path = country_profiles_dir / pattern
        if path.exists():
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one country profile file for {weather_year}, found {len(matches)}.")
    return matches[0]


def detect_nc_loader() -> str:
    return "xarray" if importlib.util.find_spec("xarray") is not None else "ncdump"


def load_country_profiles_meta_xarray(country_profiles_dir: Path) -> NcCountryProfilesMeta:
    xr = importlib.import_module("xarray")
    sample = find_country_profile_file(country_profiles_dir, WEATHER_YEARS[0])
    with xr.open_dataset(sample) as ds:
        countries = tuple(normalize_country(str(value)) for value in ds["country"].values.tolist())
        hydro_types = tuple(str(value) for value in ds["hydro_type"].values.tolist())
        member_countries = tuple(str(value) for value in ds["member_countries"].values.tolist())
    return NcCountryProfilesMeta(countries=countries, hydro_types=hydro_types, member_countries=member_countries)


def load_ncdump_dimensions(path: Path) -> dict[str, int]:
    proc = subprocess.run(
        ["ncdump", "-h", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    dims: dict[str, int] = {}
    dim_re = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(\d+)\s*;")
    for line in proc.stdout.splitlines():
        match = dim_re.match(line)
        if match:
            dims[match.group(1)] = int(match.group(2))
    return dims


def stream_ncdump_values(path: Path, var_name: str) -> Iterable[str]:
    proc = subprocess.Popen(
        ["ncdump", "-v", var_name, str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    started = False
    in_data_section = False
    collected = []
    for line in proc.stdout:
        if not in_data_section:
            if line.strip() == "data:":
                in_data_section = True
            continue
        if not started:
            marker = f"{var_name} ="
            if marker not in line:
                continue
            started = True
            line = line.split("=", 1)[1]
        if ";" in line:
            collected.append(line.split(";", 1)[0])
            break
        collected.append(line)
    stdout_tail, stderr_text = proc.communicate()
    if proc.returncode:
        raise RuntimeError(f"ncdump failed for {path.name} variable {var_name}: {stderr_text.strip() or stdout_tail.strip()}")
    for chunk in collected:
        yield chunk


def ncdump_string_values(path: Path, var_name: str) -> tuple[str, ...]:
    values: list[str] = []
    for chunk in stream_ncdump_values(path, var_name):
        values.extend(re.findall(r'"([^"]*)"', chunk))
    return tuple(values)


def ncdump_numeric_values(path: Path, var_name: str) -> Iterable[float]:
    token_re = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|NaNf?|nanf?")
    for chunk in stream_ncdump_values(path, var_name):
        for token in token_re.findall(chunk):
            if token.lower().startswith("nan"):
                yield float("nan")
            else:
                yield float(token)


def load_country_profiles_meta(country_profiles_dir: Path) -> NcCountryProfilesMeta:
    if detect_nc_loader() == "xarray":
        return load_country_profiles_meta_xarray(country_profiles_dir)
    sample = find_country_profile_file(country_profiles_dir, WEATHER_YEARS[0])
    return NcCountryProfilesMeta(
        countries=tuple(normalize_country(value) for value in ncdump_string_values(sample, "country")),
        hydro_types=ncdump_string_values(sample, "hydro_type"),
        member_countries=ncdump_string_values(sample, "member_countries"),
    )


def weekly_sums_from_nc_hourly_total(path: Path, country_indices: dict[str, int], n_country: int) -> dict[tuple[str, int], float]:
    countries = list(country_indices.items())
    if not countries:
        return {}
    weekly: dict[tuple[str, int], float] = defaultdict(float)
    for linear_idx, value in enumerate(ncdump_numeric_values(path, "e_avail_total")):
        country_idx = linear_idx % n_country
        hour_idx = linear_idx // n_country
        week_idx = hour_idx // 168
        if week_idx >= 52:
            continue
        for country, idx in countries:
            if country_idx == idx:
                weekly[(country, week_idx + 1)] += value * 1000.0
                break
    return weekly


def weekly_sums_from_nc_hourly_by_type(path: Path, country_indices: dict[str, int], hydro_type_indices: dict[str, int], n_country: int, n_hydro_type: int, n_time: int) -> dict[tuple[str, str, int], float]:
    if not country_indices or not hydro_type_indices:
        return {}
    weekly: dict[tuple[str, str, int], float] = defaultdict(float)
    reverse_country = {idx: country for country, idx in country_indices.items()}
    reverse_hydro_type = {idx: hydro_type for hydro_type, idx in hydro_type_indices.items()}
    for linear_idx, value in enumerate(ncdump_numeric_values(path, "e_avail")):
        time_idx = linear_idx % n_time
        hydro_type_idx = (linear_idx // n_time) % n_hydro_type
        country_idx = linear_idx // (n_time * n_hydro_type)
        if country_idx not in reverse_country or hydro_type_idx not in reverse_hydro_type:
            continue
        week_idx = time_idx // 168
        if week_idx >= 52:
            continue
        weekly[(reverse_country[country_idx], reverse_hydro_type[hydro_type_idx], week_idx + 1)] += value * 1000.0
    return weekly


def load_nc_weekly_profiles_ncdump(country_profiles_dir: Path, countries: set[str]) -> tuple[dict[tuple[str, int, int], float], dict[tuple[str, str, int, int], float], NcCountryProfilesMeta]:
    meta = load_country_profiles_meta(country_profiles_dir)
    available_countries = {normalize_country(code): idx for idx, code in enumerate(meta.countries)}
    available_hydro_types = {str(code): idx for idx, code in enumerate(meta.hydro_types)}
    requested_countries = {country: available_countries[country] for country in sorted(countries) if country in available_countries}
    n_country = len(meta.countries)
    n_hydro_type = len(meta.hydro_types)
    totals: dict[tuple[str, int, int], float] = {}
    by_type: dict[tuple[str, str, int, int], float] = {}
    for weather_year in WEATHER_YEARS:
        path = find_country_profile_file(country_profiles_dir, weather_year)
        dims = load_ncdump_dimensions(path)
        yearly_totals = weekly_sums_from_nc_hourly_total(path, requested_countries, n_country)
        for (country, week), value in yearly_totals.items():
            totals[(country, weather_year, week)] = value
        yearly_by_type = weekly_sums_from_nc_hourly_by_type(path, requested_countries, available_hydro_types, n_country, n_hydro_type, dims.get("time", 8760))
        for (country, hydro_type, week), value in yearly_by_type.items():
            by_type[(country, hydro_type, weather_year, week)] = value
    return totals, by_type, meta


def aggregate_hourly_to_weekly_mwh(values: Any) -> list[float]:
    np = importlib.import_module("numpy")
    arr = np.asarray(values, dtype=float)
    trimmed = arr[: 52 * 168]
    return list(trimmed.reshape(52, 168).sum(axis=1) * 1000.0)


def load_nc_weekly_profiles_xarray(country_profiles_dir: Path, countries: set[str]) -> tuple[dict[tuple[str, int, int], float], dict[tuple[str, str, int, int], float], NcCountryProfilesMeta]:
    xr = importlib.import_module("xarray")
    meta = load_country_profiles_meta_xarray(country_profiles_dir)
    requested_countries = [country for country in sorted(countries) if country in set(meta.countries)]
    country_indices = {country: idx for idx, country in enumerate(meta.countries)}
    hydro_type_indices = {hydro_type: idx for idx, hydro_type in enumerate(meta.hydro_types)}
    totals: dict[tuple[str, int, int], float] = {}
    by_type: dict[tuple[str, str, int, int], float] = {}
    for weather_year in WEATHER_YEARS:
        path = find_country_profile_file(country_profiles_dir, weather_year)
        with xr.open_dataset(path) as ds:
            total_values = ds["e_avail_total"].transpose("country", "time").values
            type_values = ds["e_avail"].transpose("country", "hydro_type", "time").values
            for country in requested_countries:
                weekly_totals = aggregate_hourly_to_weekly_mwh(total_values[country_indices[country], :])
                for week, value in enumerate(weekly_totals, start=1):
                    totals[(country, weather_year, week)] = value
                for hydro_type, hydro_idx in hydro_type_indices.items():
                    weekly_type = aggregate_hourly_to_weekly_mwh(type_values[country_indices[country], hydro_idx, :])
                    for week, value in enumerate(weekly_type, start=1):
                        by_type[(country, hydro_type, weather_year, week)] = value
    return totals, by_type, meta


def load_nc_weekly_profiles(country_profiles_dir: Path, countries: set[str]) -> tuple[dict[tuple[str, int, int], float], dict[tuple[str, str, int, int], float], NcCountryProfilesMeta, str]:
    loader = detect_nc_loader()
    if loader == "xarray":
        totals, by_type, meta = load_nc_weekly_profiles_xarray(country_profiles_dir, countries)
    else:
        totals, by_type, meta = load_nc_weekly_profiles_ncdump(country_profiles_dir, countries)
    return totals, by_type, meta, loader


def load_bus_meta(
    plants_csv: Path,
    buses_csv: Path,
    cluster_map: CountryClusterMap,
) -> tuple[dict[str, BusMeta], dict[str, set[str]], dict[str, set[str]]]:
    bus_meta: dict[str, BusMeta] = {}
    source_country_to_buses: dict[str, set[str]] = defaultdict(set)
    model_country_to_buses: dict[str, set[str]] = defaultdict(set)
    for path in (plants_csv, buses_csv):
        for row in read_csv(path):
            bus = str(row.get("bus_id") or "").strip()
            if not bus:
                continue
            model = normalize_country(row.get("country"))
            label = str(row.get("country_label") or label_for_model_country(model, cluster_map)).strip() or model
            bus_meta.setdefault(bus, BusMeta(bus, model, label))
            if model:
                model_country_to_buses[model].add(bus)
            countries = split_countries(row.get("original_country")) or ([model] if model else [])
            for country in countries:
                source_country_to_buses[country].add(bus)
                model_country_to_buses[map_to_model_country(country, cluster_map)].add(bus)
    return bus_meta, source_country_to_buses, model_country_to_buses


def map_existing_hydro(country: str, tech: str, presence: dict[str, set[str]], resolve_phs: bool) -> str | None:
    if tech == "Run-Of-River":
        return "ror"
    if tech == "Reservoir":
        return "wr"
    if tech == "Pumped Storage":
        if not resolve_phs:
            return "phs"
        if "Reservoir" in presence.get(country, set()):
            return "wr"
        if "Run-Of-River" in presence.get(country, set()):
            return "ror"
        return "wr"
    return None


def load_current_weights(plants_csv: Path, resolve_phs: bool) -> dict[tuple[str, str, str], dict[str, float]]:
    rows = [r for r in read_csv(plants_csv) if str(r.get("Fueltype") or "").strip() == "Hydro"]
    presence: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        countries = split_countries(row.get("original_country"))
        if len(countries) == 1:
            presence[countries[0]].add(str(row.get("Technology") or "").strip())
    weights: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: {"turb": 0.0, "storage": 0.0})
    for row in rows:
        countries = split_countries(row.get("original_country"))
        if len(countries) != 1:
            continue
        country, bus = countries[0], str(row.get("bus_id") or "").strip()
        plant_type = map_existing_hydro(country, str(row.get("Technology") or "").strip(), presence, resolve_phs)
        if plant_type is None:
            continue
        key = (country, plant_type, bus)
        weights[key]["turb"] += parse_float(row.get("Capacity"))
        weights[key]["storage"] += parse_float(row.get("StorageCapacity_MWh"))
    return weights


def load_bus_scores(load_csv: Path) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for row in read_csv(load_csv):
        scores[str(row.get("bus") or "")] += parse_float(row.get("allocated_load_mw"))
    return scores


def score_shares(buses: Iterable[str], scores: dict[str, float]) -> dict[str, float]:
    items = sorted(set(b for b in buses if b))
    if not items:
        return {}
    total = sum(scores.get(b, 0.0) for b in items)
    if total > EPS:
        return {b: scores.get(b, 0.0) / total for b in items}
    equal = 1.0 / len(items)
    return {b: equal for b in items}


def is_hvdc_helper_bus(bus: str) -> bool:
    return bus.startswith("cl_dc")


def build_load_shares_by_source_country(
    source_country_to_buses: dict[str, set[str]],
    model_country_to_buses: dict[str, set[str]],
    scores: dict[str, float],
    cluster_map: CountryClusterMap,
) -> dict[str, dict[str, float]]:
    countries = sorted(set(source_country_to_buses) | set(cluster_map.source_to_target))
    out: dict[str, dict[str, float]] = {}
    for country in countries:
        candidates = candidate_buses_for_country(country, source_country_to_buses, model_country_to_buses, cluster_map)
        load_candidates = {bus for bus in candidates & set(scores) if not is_hvdc_helper_bus(bus)}
        if load_candidates:
            out[country] = score_shares(load_candidates, scores)
        elif candidates:
            out[country] = score_shares(bus for bus in candidates if not is_hvdc_helper_bus(bus))
    return out


CONSTRAINT_SUM_FIELDS = [
    "min_turb_mw", "max_turb_mw", "min_pump_mw", "max_pump_mw",
    "min_turb_en_mwh_day", "max_turb_en_mwh_day", "min_pump_en_mwh_day", "max_pump_en_mwh_day",
    "min_turb_en_mwh_period", "max_turb_en_mwh_period", "min_pump_en_mwh_period", "max_pump_en_mwh_period",
]
RESERVOIR_FIELDS = ["min_res_hist_pu", "max_res_hist_pu", "min_res_tech_pu", "max_res_tech_pu"]
PUMP_FIELDS = ["installed_pump_mw", "min_pump_mw", "max_pump_mw", "min_pump_en_mwh_day", "max_pump_en_mwh_day", "min_pump_en_mwh_period", "max_pump_en_mwh_period"]


def row_has_capacity(row: dict[str, Any]) -> bool:
    return (
        parse_float(row.get("installed_turb_mw")) > EPS
        or parse_float(row.get("installed_pump_mw")) > EPS
        or parse_float(row.get("installed_storage_mwh")) > EPS
    )


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None:
        return None
    if denominator is None or abs(denominator) <= EPS:
        return 0.0 if abs(numerator) <= EPS else None
    return numerator / denominator


def weighted_average(pairs: Iterable[tuple[float | None, float]]) -> float | None:
    total = 0.0
    total_weight = 0.0
    for value, weight in pairs:
        if value is None or abs(weight) <= EPS:
            continue
        total += value * weight
        total_weight += weight
    return None if total_weight <= EPS else total / total_weight


def default_constraint_value(field: str, capacity_row: dict[str, Any], days_in_period: int) -> float:
    installed_turb_mw = parse_float(capacity_row.get("installed_turb_mw"))
    installed_pump_mw = abs(parse_float(capacity_row.get("installed_pump_mw")))
    if field in {"min_turb_mw", "min_pump_mw", "min_turb_en_mwh_day", "min_pump_en_mwh_day", "min_turb_en_mwh_period", "min_pump_en_mwh_period", "min_res_hist_pu", "min_res_tech_pu"}:
        return 0.0
    if field == "max_turb_mw":
        return installed_turb_mw
    if field == "max_pump_mw":
        return installed_pump_mw
    if field == "max_turb_en_mwh_day":
        return installed_turb_mw * 24.0
    if field == "max_pump_en_mwh_day":
        return installed_pump_mw * 24.0
    if field == "max_turb_en_mwh_period":
        return installed_turb_mw * 24.0 * days_in_period
    if field == "max_pump_en_mwh_period":
        return installed_pump_mw * 24.0 * days_in_period
    if field in {"max_res_hist_pu", "max_res_tech_pu"}:
        return 1.0
    raise KeyError(f"Unsupported constraint field '{field}'.")


def constraint_value_with_default(field: str, constraint_row: dict[str, Any], capacity_row: dict[str, Any], days_in_period: int) -> float:
    value = parse_optional_float(constraint_row.get(field))
    if value is None:
        return default_constraint_value(field, capacity_row, days_in_period)
    return value


def fill_missing_constraint_defaults(row: dict[str, Any]) -> dict[str, Any]:
    filled = dict(row)
    installed_turb_mw = parse_float(filled.get("installed_turb_mw"))
    installed_pump_mw = parse_float(filled.get("installed_pump_mw"))
    days_in_period = int(float(filled.get("days_in_period") or 0))
    default_values: dict[str, float] = {
        "min_turb_mw": 0.0,
        "max_turb_mw": installed_turb_mw,
        "min_turb_pu": 0.0,
        "max_turb_pu": 1.0 if installed_turb_mw > 0.0 else 0.0,
        "min_pump_mw": 0.0,
        "max_pump_mw": installed_pump_mw,
        "min_pump_pu": 0.0,
        "max_pump_pu": 1.0 if installed_pump_mw > 0.0 else 0.0,
        "min_turb_en_mwh_day": 0.0,
        "max_turb_en_mwh_day": installed_turb_mw * 24.0,
        "min_pump_en_mwh_day": 0.0,
        "max_pump_en_mwh_day": installed_pump_mw * 24.0,
        "min_turb_en_mwh_period": 0.0,
        "max_turb_en_mwh_period": installed_turb_mw * 24.0 * days_in_period,
        "min_pump_en_mwh_period": 0.0,
        "max_pump_en_mwh_period": installed_pump_mw * 24.0 * days_in_period,
        "min_res_hist_pu": 0.0,
        "max_res_hist_pu": 1.0,
        "min_res_tech_pu": 0.0,
        "max_res_tech_pu": 1.0,
    }
    for field, default_value in default_values.items():
        if filled.get(field) in (None, ""):
            filled[field] = default_value
    return filled


def resolve_phs_target(country: str, technology: str, has_wr: bool, has_ror: bool) -> tuple[str, str, str]:
    if technology == "closed_loop":
        if not has_wr and has_ror:
            return country, "ror", "open_loop"
        if not has_wr and not has_ror:
            return country, "wr", "closed_loop"
    return country, "wr", "open_loop"


def resolve_phs_component_map(capacity_rows: list[dict[str, Any]], presence_rows: list[dict[str, Any]] | None = None) -> dict[tuple[str, str, str], list[tuple[str, str, str]]]:
    if presence_rows is None:
        presence_rows = capacity_rows
    by_country: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in capacity_rows:
        if row_has_capacity(row):
            by_country[str(row["country"])].append(row)
    presence_by_country: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in presence_rows:
        if row_has_capacity(row):
            presence_by_country[str(row["country"])].append(row)
    component_map: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}
    for country, rows in by_country.items():
        country_presence = presence_by_country.get(country, [])
        has_wr = any(str(row["plant_type"]) == "wr" and str(row["technology"]) == "open_loop" for row in country_presence)
        has_ror = any(str(row["plant_type"]) == "ror" and str(row["technology"]) == "open_loop" for row in country_presence)
        for row in rows:
            source_key = (str(row["country"]), str(row["plant_type"]), str(row["technology"]))
            target_key = resolve_phs_target(country, str(row["technology"]), has_wr, has_ror) if str(row["plant_type"]) == "phs" else source_key
            component_map.setdefault(target_key, []).append(source_key)
    return {key: sorted(values) for key, values in component_map.items()}


def keep_phs_component_map(capacity_rows: list[dict[str, Any]], presence_rows: list[dict[str, Any]] | None = None) -> dict[tuple[str, str, str], list[tuple[str, str, str]]]:
    if presence_rows is None:
        presence_rows = capacity_rows
    by_country: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in capacity_rows:
        if row_has_capacity(row):
            by_country[str(row["country"])].append(row)
    presence_by_country: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in presence_rows:
        if row_has_capacity(row):
            presence_by_country[str(row["country"])].append(row)
    component_map: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}
    for country, rows in by_country.items():
        country_presence = presence_by_country.get(country, [])
        has_phs_open = any(str(row["plant_type"]) == "phs" and str(row["technology"]) == "open_loop" for row in country_presence)
        has_phs_closed = any(str(row["plant_type"]) == "phs" and str(row["technology"]) == "closed_loop" for row in country_presence)
        mixed_phs = has_phs_open and has_phs_closed
        for row in rows:
            source_key = (str(row["country"]), str(row["plant_type"]), str(row["technology"]))
            if str(row["plant_type"]) == "phs" and mixed_phs and str(row["technology"]) == "open_loop":
                target_key = (country, "phs", "closed_loop")
            else:
                target_key = source_key
            component_map.setdefault(target_key, []).append(source_key)
    return {key: sorted(values) for key, values in component_map.items()}


def aggregate_capacity_rows(
    capacity_rows: list[dict[str, Any]],
    component_map: dict[tuple[str, str, str], list[tuple[str, str, str]]],
) -> list[dict[str, Any]]:
    capacity_index = {(str(row["country"]), str(row["plant_type"]), str(row["technology"])): row for row in capacity_rows}
    merged_rows: list[dict[str, Any]] = []
    for target_key, source_keys in sorted(component_map.items()):
        sample_row = capacity_index[source_keys[0]]
        combined: dict[str, Any] = {
            "country": target_key[0],
            "ref_year": sample_row["ref_year"],
            "plant_type": target_key[1],
            "technology": target_key[2],
            "installed_turb_mw": 0.0,
            "installed_pump_mw": 0.0,
            "installed_storage_mwh": 0.0,
        }
        receives_phs = False
        for source_key in source_keys:
            row = capacity_index[source_key]
            combined["installed_turb_mw"] += parse_float(row.get("installed_turb_mw"))
            combined["installed_pump_mw"] += abs(parse_float(row.get("installed_pump_mw")))
            combined["installed_storage_mwh"] += parse_float(row.get("installed_storage_mwh"))
            receives_phs = receives_phs or source_key[1] == "phs"
        if receives_phs:
            combined["installed_pump_mw"] = 0.0
        merged_rows.append(combined)
    return merged_rows


def load_constraints(path: Path, year: int, resolve_phs: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str, str], list[tuple[str, str, str]]]]:
    raw_rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        row["country"] = normalize_country(row.get("country"))
        if int(row["ref_year"]) == year and int(row["week"]) != 53:
            raw_rows.append(row)
    by_combo: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        by_combo[(str(row["country"]), str(row["plant_type"]), str(row["technology"]))].append(row)

    expanded_raw_rows: list[dict[str, Any]] = []
    diag: list[dict[str, Any]] = []
    for combo, group in sorted(by_combo.items()):
        weeks = sorted({int(r["week"]) for r in group})
        if weeks == list(WEEKS):
            expanded_raw_rows.extend(sorted(group, key=lambda r: int(r["week"])))
            diag.append({"country": combo[0], "plant_type": combo[1], "technology": combo[2], "input_week_rows": len(group), "expanded_to_weeks": 52, "expansion_rule": "as_is"})
        elif len(group) == 1 and weeks == [1]:
            template = group[0]
            for week in WEEKS:
                row = dict(template)
                row["week"] = week
                row["period_start_date"], row["period_end_date"], row["days_in_period"] = week_bounds(year, week)
                expanded_raw_rows.append(row)
            diag.append({"country": combo[0], "plant_type": combo[1], "technology": combo[2], "input_week_rows": 1, "expanded_to_weeks": 52, "expansion_rule": "replicate_week1"})
        else:
            raise ValueError(f"Unsupported constraint week pattern for {combo}: {weeks}")
    expanded_raw_rows.sort(key=lambda r: (str(r["country"]), str(r["plant_type"]), str(r["technology"]), int(r["week"])))

    source_capacity_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in expanded_raw_rows:
        if int(row["week"]) != 1 or not row_has_capacity(row):
            continue
        key = (str(row["country"]), str(row["plant_type"]), str(row["technology"]))
        source_capacity_index[key] = {
            "country": key[0],
            "ref_year": row["ref_year"],
            "plant_type": key[1],
            "technology": key[2],
            "installed_turb_mw": parse_float(row.get("installed_turb_mw")),
            "installed_pump_mw": abs(parse_float(row.get("installed_pump_mw"))),
            "installed_storage_mwh": parse_float(row.get("installed_storage_mwh")),
        }
    source_capacity_rows = list(source_capacity_index.values())
    if resolve_phs:
        resolved_component_map = resolve_phs_component_map(source_capacity_rows, source_capacity_rows)
    else:
        resolved_component_map = keep_phs_component_map(source_capacity_rows, source_capacity_rows)
    resolved_capacity_rows = aggregate_capacity_rows(source_capacity_rows, resolved_component_map)
    resolved_capacity_index = {(str(row["country"]), str(row["plant_type"]), str(row["technology"])): row for row in resolved_capacity_rows}
    raw_constraint_map = {
        (str(row["country"]), int(row["week"]), str(row["plant_type"]), str(row["technology"])): row
        for row in expanded_raw_rows
    }

    resolved_rows: list[dict[str, Any]] = []
    for target_key, capacity_row in sorted(resolved_capacity_index.items()):
        country, plant_type, technology = target_key
        installed_turb_mw = parse_float(capacity_row.get("installed_turb_mw"))
        installed_pump_mw = abs(parse_float(capacity_row.get("installed_pump_mw")))
        installed_storage_mwh = parse_float(capacity_row.get("installed_storage_mwh"))
        source_keys = resolved_component_map.get(target_key, [target_key])
        receives_phs = any(source_plant_type == "phs" for _, source_plant_type, _ in source_keys)
        for week in WEEKS:
            component_rows = []
            for source_country, source_plant_type, source_technology in source_keys:
                key = (source_country, week, source_plant_type, source_technology)
                if key in raw_constraint_map:
                    component_rows.append((source_country, source_plant_type, source_technology, raw_constraint_map[key]))
            if not component_rows:
                continue
            sample_row = component_rows[0][3]
            days_in_period = int(float(sample_row.get("days_in_period") or 0)) or 7
            aggregated: dict[str, Any] = {
                "country": country,
                "ref_year": sample_row["ref_year"],
                "plant_type": plant_type,
                "technology": technology,
                "temporal_resolution": sample_row.get("temporal_resolution") or "weekly",
                "week": week,
                "period_start_date": sample_row.get("period_start_date") or week_bounds(year, week)[0],
                "period_end_date": sample_row.get("period_end_date") or week_bounds(year, week)[1],
                "days_in_period": days_in_period,
                "installed_turb_mw": installed_turb_mw,
                "installed_pump_mw": installed_pump_mw,
                "installed_storage_mwh": installed_storage_mwh,
            }
            for field in CONSTRAINT_SUM_FIELDS:
                aggregated[field] = sum(
                    constraint_value_with_default(field, row, source_capacity_index[(source_country, source_plant_type, source_technology)], days_in_period)
                    for source_country, source_plant_type, source_technology, row in component_rows
                )
            if plant_type != "phs" and receives_phs:
                for field in ("min_pump_mw", "max_pump_mw", "min_pump_en_mwh_day", "max_pump_en_mwh_day", "min_pump_en_mwh_period", "max_pump_en_mwh_period"):
                    aggregated[field] = 0.0
            for field in RESERVOIR_FIELDS:
                aggregated[field] = weighted_average(
                    (
                        constraint_value_with_default(field, row, source_capacity_index[(source_country, source_plant_type, source_technology)], days_in_period),
                        parse_float(source_capacity_index[(source_country, source_plant_type, source_technology)].get("installed_storage_mwh")),
                    )
                    for source_country, source_plant_type, source_technology, row in component_rows
                )
            for field in PUMP_FIELDS:
                if field in aggregated and aggregated[field] is not None:
                    aggregated[field] = abs(float(aggregated[field]))
            aggregated["min_turb_pu"] = safe_ratio(aggregated.get("min_turb_mw"), installed_turb_mw)
            aggregated["max_turb_pu"] = safe_ratio(aggregated.get("max_turb_mw"), installed_turb_mw)
            aggregated["min_pump_pu"] = safe_ratio(aggregated.get("min_pump_mw"), installed_pump_mw)
            aggregated["max_pump_pu"] = safe_ratio(aggregated.get("max_pump_mw"), installed_pump_mw)
            resolved_rows.append(fill_missing_constraint_defaults(aggregated))

    resolved_rows.sort(key=lambda r: (str(r["country"]), str(r["plant_type"]), str(r["technology"]), int(r["week"])))
    return resolved_rows, diag, resolved_component_map


def interpolate(values: list[float | None]) -> tuple[list[float | None], int]:
    known = [(i, v) for i, v in enumerate(values) if v is not None]
    if not known:
        return values, 0
    if len(known) == 1:
        only = known[0][1]
        assert only is not None
        return [only if v is None else v for v in values], sum(1 for v in values if v is None)
    filled, count = list(values), 0
    li, lv = known[0]
    assert lv is not None
    for i in range(li):
        filled[i] = lv
        count += 1
    ri, rv = known[-1]
    assert rv is not None
    for i in range(ri + 1, len(values)):
        filled[i] = rv
        count += 1
    for (li, lv), (ri, rv) in zip(known, known[1:]):
        assert lv is not None and rv is not None
        gap = ri - li
        if gap <= 1:
            continue
        slope = (rv - lv) / gap
        for off in range(1, gap):
            i = li + off
            if filled[i] is None:
                filled[i] = lv + slope * off
                count += 1
    return filled, count


def load_inflows(
    path: Path,
    year: int,
    resolved_component_map: dict[tuple[str, str, str], list[tuple[str, str, str]]],
) -> tuple[
    dict[tuple[str, str, str, int, int], float],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[tuple[str, int, int], float],
]:
    base: list[dict[str, Any]] = []
    for row in read_csv(path):
        row["country"] = normalize_country(row.get("country"))
        if int(row["ref_year"]) == year and int(row["week"]) != 53:
            base.append(row)
    inflow_map: dict[tuple[str, int, int, str, str], float | None] = {}
    input_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in base:
        combo = (str(row["country"]), str(row["plant_type"]), str(row["technology"]))
        input_counts[combo] += 1
        inflow_map[(combo[0], int(row["weather_year"]), int(row["week"]), combo[1], combo[2])] = parse_optional_float(row.get("inflow_mwh_week"))

    source_combos = sorted({source_key for source_keys in resolved_component_map.values() for source_key in source_keys})
    source_data: dict[tuple[str, str, str], dict[int, list[float | None]]] = {
        combo: {wy: [None] * 52 for wy in WEATHER_YEARS}
        for combo in source_combos
    }
    for combo in source_combos:
        for wy in WEATHER_YEARS:
            for week in WEEKS:
                source_data[combo][wy][week - 1] = inflow_map.get((combo[0], wy, week, combo[1], combo[2]))

    source_profiles: dict[tuple[str, str, str, int, int], float] = {}
    source_meta: dict[tuple[str, str, str], dict[str, Any]] = {}
    for combo, yearly in sorted(source_data.items()):
        complete: list[list[float]] = []
        missing_years: list[int] = []
        interp_weeks = 0
        all_zero_years = 0
        normalized: dict[int, list[float] | None] = {}
        combo_zero_expected = combo[1] == "phs" and combo[2] == "closed_loop"
        for wy in WEATHER_YEARS:
            vals = list(yearly.get(wy, [None] * 52))
            non_null = [v for v in vals if v is not None]
            all_none = not non_null
            all_zero = bool(non_null) and all(abs(float(v or 0.0)) <= EPS for v in non_null)
            if combo_zero_expected and (all_none or all_zero):
                normalized[wy] = [0.0] * 52
                complete.append(normalized[wy] or [])
                continue
            if all_none or all_zero:
                normalized[wy] = None
                missing_years.append(wy)
                if all_zero:
                    all_zero_years += 1
                continue
            filled, c = interpolate(vals)
            interp_weeks += c
            if any(v is None for v in filled):
                normalized[wy] = None
                missing_years.append(wy)
                continue
            full = [float(v) for v in filled if v is not None]
            if all(abs(v) <= EPS for v in full) and not combo_zero_expected:
                normalized[wy] = None
                missing_years.append(wy)
                all_zero_years += 1
                continue
            normalized[wy] = full
            complete.append(full)
        imputed_years = 0
        if complete and missing_years:
            means = [sum(year_vals[i] for year_vals in complete) / len(complete) for i in range(52)]
            for wy in missing_years:
                normalized[wy] = list(means)
                imputed_years += 1
        all_missing = not complete and not combo_zero_expected
        source_meta[combo] = {
            "zero_inflow_expected": combo_zero_expected,
            "all_years_missing_or_zero": all_missing,
            "input_rows": input_counts.get(combo, 0),
            "interpolated_weeks": interp_weeks,
            "imputed_weather_years": imputed_years,
            "all_zero_weather_years": all_zero_years,
        }
        for wy, vals in normalized.items():
            if vals is None:
                continue
            for week, value in enumerate(vals, start=1):
                source_profiles[(combo[0], combo[1], combo[2], wy, week)] = value

    combo_profiles: dict[tuple[str, str, str, int, int], float] = {}
    redirected_profiles: dict[tuple[str, int, int], float] = defaultdict(float)
    diag: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []
    by_country_status: dict[str, list[tuple[tuple[str, str, str], bool]]] = defaultdict(list)
    combo_meta: dict[tuple[str, str, str], dict[str, Any]] = {}
    for combo, source_keys in sorted(resolved_component_map.items()):
        mixed_phs_to_closed = (
            combo[1] == "phs"
            and combo[2] == "closed_loop"
            and any(source_plant_type == "phs" and source_technology == "open_loop" for _, source_plant_type, source_technology in source_keys)
            and any(source_plant_type == "phs" and source_technology == "closed_loop" for _, source_plant_type, source_technology in source_keys)
        )
        redirect_source_keys = [
            source_key
            for source_key in source_keys
            if mixed_phs_to_closed and source_key[1] == "phs" and source_key[2] == "open_loop"
        ]
        effective_source_keys = [source_key for source_key in source_keys if source_key not in redirect_source_keys]
        combo_zero_expected = mixed_phs_to_closed or all(source_meta.get(source_key, {}).get("zero_inflow_expected", False) for source_key in effective_source_keys)
        combo_has_data = any(not source_meta.get(source_key, {}).get("all_years_missing_or_zero", True) for source_key in effective_source_keys)
        redirect_has_data = any(not source_meta.get(source_key, {}).get("all_years_missing_or_zero", True) for source_key in redirect_source_keys)
        if redirect_source_keys and redirect_has_data:
            flags.append({
                "country": combo[0],
                "plant_type": "phs",
                "technology": "closed_loop",
                "review_reason": "mixed_phs_open_loop_collapsed_to_closed_loop",
                "redirected_inflow_target": "wr_ror_open_loop",
            })
            for weather_year in WEATHER_YEARS:
                for week in WEEKS:
                    redirected_profiles[(combo[0], weather_year, week)] += sum(
                        source_profiles.get((source_country, source_plant_type, source_technology, weather_year, week), 0.0)
                        for source_country, source_plant_type, source_technology in redirect_source_keys
                    )
        elif redirect_source_keys:
            flags.append({
                "country": combo[0],
                "plant_type": "phs",
                "technology": "open_loop",
                "review_reason": "mixed_phs_open_loop_missing_after_collapse",
            })

        combo_interpolated_weeks = sum(int(source_meta.get(source_key, {}).get("interpolated_weeks", 0)) for source_key in source_keys)
        combo_imputed_years = sum(int(source_meta.get(source_key, {}).get("imputed_weather_years", 0)) for source_key in source_keys)
        combo_all_zero_years = sum(int(source_meta.get(source_key, {}).get("all_zero_weather_years", 0)) for source_key in source_keys)
        combo_input_rows = sum(int(source_meta.get(source_key, {}).get("input_rows", 0)) for source_key in source_keys)
        all_missing = not combo_has_data and not combo_zero_expected
        by_country_status[combo[0]].append((combo, all_missing))
        combo_meta[combo] = {
            "zero_inflow_expected": combo_zero_expected,
            "all_years_missing_or_zero": all_missing,
            "input_rows": combo_input_rows,
            "redirected_mixed_phs_open_loop": bool(redirect_source_keys),
        }
        diag.append({
            "country": combo[0],
            "plant_type": combo[1],
            "technology": combo[2],
            "input_rows": combo_input_rows,
            "interpolated_weeks": combo_interpolated_weeks,
            "imputed_weather_years": combo_imputed_years,
            "all_zero_weather_years": combo_all_zero_years,
            "all_years_missing_or_zero": str(all_missing),
            "zero_inflow_expected": str(combo_zero_expected),
        })
        if all_missing:
            flags.append({"country": combo[0], "plant_type": combo[1], "technology": combo[2], "review_reason": "all_weather_years_missing_or_zero_nc4_fallback_needed"})
        if combo_zero_expected:
            for weather_year in WEATHER_YEARS:
                for week in WEEKS:
                    combo_profiles[(combo[0], combo[1], combo[2], weather_year, week)] = 0.0
            continue
        if not combo_has_data:
            continue
        for weather_year in WEATHER_YEARS:
            for week in WEEKS:
                value = sum(
                    source_profiles.get((source_country, source_plant_type, source_technology, weather_year, week), 0.0)
                    for source_country, source_plant_type, source_technology in effective_source_keys
                )
                combo_profiles[(combo[0], combo[1], combo[2], weather_year, week)] = value
    for country, items in sorted(by_country_status.items()):
        miss = [c for c, bad in items if bad]
        pos = [c for c, bad in items if not bad]
        if miss and pos:
            flags.append({"country": country, "plant_type": "", "technology": "", "review_reason": "partial_missing_country_combo_requires_closed_loop_review", "missing_combos": ";".join(f"{c[1]}:{c[2]}" for c in miss), "positive_combos": ";".join(f"{c[1]}:{c[2]}" for c in pos)})
    return combo_profiles, diag, flags, combo_meta, redirected_profiles


def build_inflow_groups(
    combo_profiles: dict[tuple[str, str, str, int, int], float],
    combo_meta: dict[tuple[str, str, str], dict[str, Any]],
    redirected_profiles: dict[tuple[str, int, int], float],
    target: dict[tuple[str, str, str, str], dict[str, float]],
    nc_totals: dict[tuple[str, int, int], float],
    nc_by_type: dict[tuple[str, str, int, int], float],
    resolve_phs: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []
    target_open_loop: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for country, plant_type, technology, _bus in target:
        if technology != "closed_loop":
            target_open_loop[country].add((plant_type, technology))

    redirected_countries = {country for country, _weather_year, _week in redirected_profiles}
    countries = sorted(set(target_open_loop) | redirected_countries)
    for country in countries:
        open_targets = sorted(target_open_loop.get(country, set()))
        natural_open_targets = [combo for combo in open_targets if combo[0] in {"wr", "ror"}]
        valid_open = [combo for combo in open_targets if not combo_meta.get((country, combo[0], combo[1]), {}).get("all_years_missing_or_zero", True)]
        missing_open = [combo for combo in open_targets if combo_meta.get((country, combo[0], combo[1]), {}).get("all_years_missing_or_zero", False)]
        has_zero_expected = any(meta.get("zero_inflow_expected") for combo, meta in combo_meta.items() if combo[0] == country)
        if resolve_phs and has_zero_expected and valid_open:
            flags.append({"country": country, "plant_type": "", "technology": "", "review_reason": "nc_total_replaces_tyndp_due_to_closed_loop_mixed_country"})
            for weather_year in WEATHER_YEARS:
                for week in WEEKS:
                    value = nc_totals.get((country, weather_year, week))
                    if value is None:
                        continue
                    groups.append({
                        "country": country,
                        "weather_year": weather_year,
                        "week": week,
                        "inflow_mwh_week": value,
                        "eligible_techs": tuple(open_targets),
                        "source": "nc_total_closed_loop_override",
                    })
            continue

        if valid_open:
            for weather_year in WEATHER_YEARS:
                for week in WEEKS:
                    value = sum(combo_profiles.get((country, plant_type, technology, weather_year, week), 0.0) for plant_type, technology in valid_open)
                    groups.append({
                        "country": country,
                        "weather_year": weather_year,
                        "week": week,
                        "inflow_mwh_week": value,
                        "eligible_techs": tuple(valid_open),
                        "source": "tyndp_resolved_open_loop",
                    })

        redirect_present = country in redirected_countries
        if redirect_present and natural_open_targets:
            flags.append({
                "country": country,
                "plant_type": "phs",
                "technology": "open_loop",
                "review_reason": "mixed_phs_open_inflow_redirected_to_natural_hydro",
                "redirect_targets": ";".join(f"{plant_type}:{technology}" for plant_type, technology in natural_open_targets),
            })
            for weather_year in WEATHER_YEARS:
                for week in WEEKS:
                    value = redirected_profiles.get((country, weather_year, week), 0.0)
                    if abs(value) <= EPS:
                        continue
                    groups.append({
                        "country": country,
                        "weather_year": weather_year,
                        "week": week,
                        "inflow_mwh_week": value,
                        "eligible_techs": tuple(natural_open_targets),
                        "source": "tyndp_phs_open_redirect_to_natural_hydro",
                    })
        elif redirect_present:
            flags.append({
                "country": country,
                "plant_type": "phs",
                "technology": "open_loop",
                "review_reason": "mixed_phs_open_inflow_dropped_no_natural_hydro",
            })

        if not missing_open:
            continue

        fallback_missing_open: list[tuple[str, str]] = []
        tech_specific_missing = []
        for plant_type, technology in missing_open:
            hydro_type = NC_HYDRO_TYPE_BY_PLANT_TYPE.get(plant_type)
            if hydro_type is None:
                continue
            fallback_missing_open.append((plant_type, technology))
            tech_specific_missing.append((plant_type, technology, hydro_type))

        if not fallback_missing_open:
            continue

        if len(fallback_missing_open) == 1 and tech_specific_missing:
            plant_type, technology, hydro_type = tech_specific_missing[0]
            flags.append({"country": country, "plant_type": plant_type, "technology": technology, "review_reason": "nc_tech_profile_replaces_missing_combo"})
            for weather_year in WEATHER_YEARS:
                for week in WEEKS:
                    value = nc_by_type.get((country, hydro_type, weather_year, week))
                    if value is None:
                        value = nc_totals.get((country, weather_year, week))
                    if value is None:
                        continue
                    groups.append({
                        "country": country,
                        "weather_year": weather_year,
                        "week": week,
                        "inflow_mwh_week": value,
                        "eligible_techs": ((plant_type, technology),),
                        "source": "nc_tech_missing_combo" if (country, hydro_type, weather_year, week) in nc_by_type else "nc_total_missing_combo",
                    })
            continue

        if len(tech_specific_missing) == len(fallback_missing_open) and tech_specific_missing:
            flags.append({"country": country, "plant_type": "", "technology": "", "review_reason": "nc_tech_profiles_replace_multiple_missing_combos", "missing_combos": ";".join(f"{pt}:{tech}" for pt, tech in fallback_missing_open)})
            for plant_type, technology, hydro_type in tech_specific_missing:
                for weather_year in WEATHER_YEARS:
                    for week in WEEKS:
                        value = nc_by_type.get((country, hydro_type, weather_year, week))
                        if value is None:
                            continue
                        groups.append({
                            "country": country,
                            "weather_year": weather_year,
                            "week": week,
                            "inflow_mwh_week": value,
                            "eligible_techs": ((plant_type, technology),),
                            "source": "nc_tech_missing_combos",
                        })
            continue

        flags.append({"country": country, "plant_type": "", "technology": "", "review_reason": "nc_total_replaces_missing_open_loop_combos", "missing_combos": ";".join(f"{pt}:{tech}" for pt, tech in fallback_missing_open)})
        for weather_year in WEATHER_YEARS:
            for week in WEEKS:
                value = nc_totals.get((country, weather_year, week))
                if value is None:
                    continue
                groups.append({
                    "country": country,
                    "weather_year": weather_year,
                    "week": week,
                    "inflow_mwh_week": value,
                    "eligible_techs": tuple(fallback_missing_open),
                    "source": "nc_total_missing_combos",
                })
    return groups, flags


def candidate_buses_for_country(
    country: str,
    source_country_to_buses: dict[str, set[str]],
    model_country_to_buses: dict[str, set[str]],
    cluster_map: CountryClusterMap,
) -> set[str]:
    # Prefer buses that originate from the source country. Country aggregates are
    # only used when the reduced network no longer contains a direct source match.
    direct = set(source_country_to_buses.get(country, set()))
    if direct:
        return direct
    model_country = map_to_model_country(country, cluster_map)
    return set(model_country_to_buses.get(model_country, set()))


def filter_hydro_countries_for_network(
    expanded_constraints: list[dict[str, Any]],
    resolved_component_map: dict[tuple[str, str, str], list[tuple[str, str, str]]],
    source_country_to_buses: dict[str, set[str]],
    model_country_to_buses: dict[str, set[str]],
    cluster_map: CountryClusterMap,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], list[tuple[str, str, str]]], list[dict[str, Any]], list[str]]:
    countries = sorted({str(row["country"]) for row in expanded_constraints})
    supported = {
        country
        for country in countries
        if candidate_buses_for_country(country, source_country_to_buses, model_country_to_buses, cluster_map)
    }
    skipped = [country for country in countries if country not in supported]
    if not skipped:
        return expanded_constraints, resolved_component_map, [], []
    filtered_constraints = [row for row in expanded_constraints if str(row["country"]) in supported]
    filtered_component_map = {
        target_key: [source_key for source_key in source_keys if source_key[0] in supported]
        for target_key, source_keys in resolved_component_map.items()
        if target_key[0] in supported
    }
    filtered_component_map = {key: value for key, value in filtered_component_map.items() if value}
    flags = [
        {
            "country": country,
            "plant_type": "",
            "technology": "",
            "review_reason": "country_not_in_reduced_network_skipped",
            "model_country": map_to_model_country(country, cluster_map),
        }
        for country in skipped
    ]
    return filtered_constraints, filtered_component_map, flags, skipped


def filter_excluded_hydro_countries(
    expanded_constraints: list[dict[str, Any]],
    resolved_component_map: dict[tuple[str, str, str], list[tuple[str, str, str]]],
    excluded_countries: set[str],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], list[tuple[str, str, str]]], list[str]]:
    if not excluded_countries:
        return expanded_constraints, resolved_component_map, []
    excluded = {normalize_country(country) for country in excluded_countries if normalize_country(country)}
    filtered_constraints = [
        row for row in expanded_constraints
        if normalize_country(row.get("country")) not in excluded
    ]
    filtered_component_map = {
        target_key: [source_key for source_key in source_keys if normalize_country(source_key[0]) not in excluded]
        for target_key, source_keys in resolved_component_map.items()
        if normalize_country(target_key[0]) not in excluded
    }
    filtered_component_map = {key: values for key, values in filtered_component_map.items() if values}
    skipped = sorted({country for country in excluded if country})
    return filtered_constraints, filtered_component_map, skipped


def choose_top_bus(
    country: str,
    source_country_to_buses: dict[str, set[str]],
    model_country_to_buses: dict[str, set[str]],
    load_shares: dict[str, dict[str, float]],
    cluster_map: CountryClusterMap,
) -> str:
    shares = load_shares.get(country, {})
    if shares:
        return max(shares.items(), key=lambda x: (x[1], x[0]))[0]
    candidates = sorted(candidate_buses_for_country(country, source_country_to_buses, model_country_to_buses, cluster_map))
    if not candidates:
        raise ValueError(f"No candidate buses for {country}.")
    return candidates[0]


def choose_top_hydro_bus(
    country: str,
    quantity: str,
    current: dict[tuple[str, str, str], dict[str, float]],
    source_country_to_buses: dict[str, set[str]],
    model_country_to_buses: dict[str, set[str]],
    load_shares: dict[str, dict[str, float]],
    cluster_map: CountryClusterMap,
) -> str:
    scores: dict[str, float] = defaultdict(float)
    fallback_turb_scores: dict[str, float] = defaultdict(float)
    for (src_country, _plant_type, bus), vals in current.items():
        if src_country != country:
            continue
        scores[bus] += float(vals.get(quantity, 0.0))
        fallback_turb_scores[bus] += float(vals.get("turb", 0.0))
    positive_scores = {bus: value for bus, value in scores.items() if value > EPS}
    if positive_scores:
        return max(positive_scores.items(), key=lambda item: (item[1], item[0]))[0]
    # Storage data are often sparser than turbine data. In that case the largest
    # known hydro bus is still a better anchor than an arbitrary load-dominated bus.
    positive_turb_scores = {bus: value for bus, value in fallback_turb_scores.items() if value > EPS}
    if positive_turb_scores:
        return max(positive_turb_scores.items(), key=lambda item: (item[1], item[0]))[0]
    return choose_top_bus(country, source_country_to_buses, model_country_to_buses, load_shares, cluster_map)


def build_country_plant_type_profiles(
    expanded_constraints: list[dict[str, Any]],
    current: dict[tuple[str, str, str], dict[str, float]],
    source_country_to_buses: dict[str, set[str]],
    model_country_to_buses: dict[str, set[str]],
    load_shares: dict[str, dict[str, float]],
    cluster_map: CountryClusterMap,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    target_totals: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"turb": 0.0, "storage": 0.0})
    for row in expanded_constraints:
        if int(row["week"]) != 1:
            continue
        key = (str(row["country"]), str(row["plant_type"]))
        target_totals[key]["turb"] += parse_float(row.get("installed_turb_mw"))
        target_totals[key]["storage"] += parse_float(row.get("installed_storage_mwh"))

    current_totals: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"turb": 0.0, "storage": 0.0})
    current_bus_values: dict[tuple[str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    countries: set[str] = set()
    for (country, plant_type, bus), vals in current.items():
        countries.add(country)
        current_totals[(country, plant_type)]["turb"] += vals["turb"]
        current_totals[(country, plant_type)]["storage"] += vals["storage"]
        current_bus_values[(country, plant_type, "turb")][bus] += vals["turb"]
        current_bus_values[(country, plant_type, "storage")][bus] += vals["storage"]
    countries.update(country for country, _plant_type in target_totals)

    profiles: dict[tuple[str, str, str], dict[str, Any]] = {}
    for country in sorted(countries):
        plant_types = sorted({plant_type for c, plant_type in current_totals if c == country} | {plant_type for c, plant_type in target_totals if c == country})
        for quantity in ("turb", "storage"):
            current_by_pt = {plant_type: current_totals.get((country, plant_type), {}).get(quantity, 0.0) for plant_type in plant_types}
            target_by_pt = {plant_type: target_totals.get((country, plant_type), {}).get(quantity, 0.0) for plant_type in plant_types}
            positive = {plant_type: max(target_by_pt[plant_type] - current_by_pt[plant_type], 0.0) for plant_type in plant_types}
            negative = {plant_type: max(current_by_pt[plant_type] - target_by_pt[plant_type], 0.0) for plant_type in plant_types}
            total_positive = sum(positive.values())
            total_negative = sum(negative.values())
            shift_pool = min(total_positive, total_negative)
            for plant_type in plant_types:
                current_total = current_by_pt[plant_type]
                target_total = target_by_pt[plant_type]
                delta = target_total - current_total
                synthetic: dict[str, float] = defaultdict(float)
                rule_parts: list[str] = []
                current_map = current_bus_values.get((country, plant_type, quantity), {})
                if current_total > EPS:
                    for bus, value in current_map.items():
                        if value > EPS:
                            synthetic[bus] += value
                    if target_total < current_total - EPS:
                        rule_parts.append("existing_country_plant_type_shares_downscale")
                    else:
                        rule_parts.append("existing_country_plant_type_shares")

                shift_in = 0.0
                if delta > EPS and shift_pool > EPS and total_positive > EPS and total_negative > EPS:
                    shift_in = delta * shift_pool / total_positive
                    if shift_in > EPS:
                        for source_pt, loss in negative.items():
                            if loss <= EPS:
                                continue
                            source_amount = shift_in * loss / total_negative
                            source_map = current_bus_values.get((country, source_pt, quantity), {})
                            source_total = sum(value for value in source_map.values() if value > EPS)
                            if source_total <= EPS:
                                continue
                            for bus, value in source_map.items():
                                if value > EPS:
                                    synthetic[bus] += source_amount * value / source_total
                        rule_parts.append("internal_type_shift")

                true_new = max(delta - shift_in, 0.0)
                if target_total > EPS and not synthetic:
                    priority_bus = choose_top_hydro_bus(country, quantity, current, source_country_to_buses, model_country_to_buses, load_shares, cluster_map)
                    synthetic[priority_bus] += target_total
                    rule_parts.append("true_new_highest_hydro_bus")
                elif true_new > EPS:
                    priority_bus = choose_top_hydro_bus(country, quantity, current, source_country_to_buses, model_country_to_buses, load_shares, cluster_map)
                    synthetic[priority_bus] += true_new
                    rule_parts.append("true_new_highest_hydro_bus")

                if target_total > EPS:
                    total_synthetic = sum(value for value in synthetic.values() if value > EPS)
                    if total_synthetic <= EPS:
                        priority_bus = choose_top_hydro_bus(country, quantity, current, source_country_to_buses, model_country_to_buses, load_shares, cluster_map)
                        shares = {priority_bus: 1.0}
                        rule_parts.append("fallback_highest_hydro_bus")
                    else:
                        shares = {bus: value / total_synthetic for bus, value in synthetic.items() if value > EPS}
                else:
                    if current_total > EPS:
                        shares = {bus: value / current_total for bus, value in current_map.items() if value > EPS}
                    else:
                        shares = {}
                profiles[(country, plant_type, quantity)] = {
                    "shares": shares,
                    "rule": ";".join(dict.fromkeys(rule_parts)) if rule_parts else "no_capacity",
                    "current_total": current_total,
                    "target_total": target_total,
                    "delta": delta,
                    "shift_in": shift_in,
                    "true_new": true_new,
                }
    return profiles


def build_target_rows(
    expanded_constraints: list[dict[str, Any]],
    current: dict[tuple[str, str, str], dict[str, float]],
    source_country_to_buses: dict[str, set[str]],
    model_country_to_buses: dict[str, set[str]],
    load_shares: dict[str, dict[str, float]],
    bus_meta: dict[str, BusMeta],
    cluster_map: CountryClusterMap,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, str], dict[str, float]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    target: dict[tuple[str, str, str, str], dict[str, float]] = {}
    diag: list[dict[str, Any]] = []
    profiles = build_country_plant_type_profiles(
        expanded_constraints,
        current,
        source_country_to_buses,
        model_country_to_buses,
        load_shares,
        cluster_map,
    )
    for row in [r for r in expanded_constraints if int(r["week"]) == 1]:
        country, plant_type, tech = row["country"], row["plant_type"], row["technology"]
        target_turb, target_storage = parse_float(row["installed_turb_mw"]), parse_float(row["installed_storage_mwh"])
        keys = sorted(k for k in current if k[0] == country and k[1] == plant_type)
        curr_turb, curr_storage = sum(current[k]["turb"] for k in keys), sum(current[k]["storage"] for k in keys)
        turb_profile = profiles.get((country, plant_type, "turb"), {"shares": {}, "rule": "no_capacity"})
        stor_profile = profiles.get((country, plant_type, "storage"), {"shares": {}, "rule": "no_capacity"})
        turb_shares = dict(turb_profile["shares"])
        stor_shares = dict(stor_profile["shares"])
        # The bus split is derived once from installed capacity and then reused
        # for all weekly constraints. This keeps limits, storage, and inflows on
        # the same nodal hydro representation.
        if target_turb > EPS and not turb_shares:
            turb_shares = {choose_top_hydro_bus(country, "turb", current, source_country_to_buses, model_country_to_buses, load_shares, cluster_map): 1.0}
        if target_storage > EPS and not stor_shares:
            stor_shares = {choose_top_hydro_bus(country, "storage", current, source_country_to_buses, model_country_to_buses, load_shares, cluster_map): 1.0}
        alloc_rule = turb_profile["rule"] if turb_profile["rule"] == stor_profile["rule"] else f"turb={turb_profile['rule']}|storage={stor_profile['rule']}"
        for bus in sorted(set(turb_shares) | set(stor_shares)):
            meta = bus_meta[bus]
            bus_curr_turb = current.get((country, plant_type, bus), {}).get("turb", 0.0)
            bus_curr_storage = current.get((country, plant_type, bus), {}).get("storage", 0.0)
            bus_target_turb = target_turb * turb_shares.get(bus, 0.0)
            bus_target_storage = target_storage * stor_shares.get(bus, 0.0)
            target[(country, plant_type, tech, bus)] = {"turb": bus_target_turb, "storage": bus_target_storage}
            rows.append({"country": country, "country_model": meta.country_model, "country_label": meta.country_label, "bus": bus, "ref_year": row["ref_year"], "plant_type": plant_type, "technology": tech, "current_turb_mw": fmt_amount(bus_curr_turb), "current_storage_mwh": fmt_amount(bus_curr_storage), "target_turb_mw": fmt_amount(bus_target_turb), "target_storage_mwh": fmt_amount(bus_target_storage), "turbine_share": fmt_share(turb_shares.get(bus, 0.0)), "storage_share": fmt_share(stor_shares.get(bus, 0.0)), "allocation_rule": alloc_rule})
        diag.append({"country": country, "plant_type": plant_type, "technology": tech, "target_turb_mw": fmt_amount(target_turb), "target_storage_mwh": fmt_amount(target_storage), "current_turb_mw": fmt_amount(curr_turb), "current_storage_mwh": fmt_amount(curr_storage), "allocation_rule": alloc_rule, "n_buses": len(set(turb_shares) | set(stor_shares))})
    return rows, target, diag


def build_share_rows(
    target_rows: list[dict[str, Any]],
    current: dict[tuple[str, str, str], dict[str, float]],
    source_country_to_buses: dict[str, set[str]],
    model_country_to_buses: dict[str, set[str]],
    load_shares: dict[str, dict[str, float]],
    bus_meta: dict[str, BusMeta],
    cluster_map: CountryClusterMap,
) -> list[dict[str, Any]]:
    target_by_key = {
        (str(row["country"]), str(row["plant_type"]), str(row["technology"]), str(row["bus"])): row
        for row in target_rows
    }
    combo_rules: dict[tuple[str, str, str], str] = {}
    for row in target_rows:
        combo_rules.setdefault(
            (str(row["country"]), str(row["plant_type"]), str(row["technology"])),
            str(row.get("allocation_rule") or "no_allocation_rule"),
        )
    rows: list[dict[str, Any]] = []
    for country, plant_type, technology in sorted(combo_rules):
        candidate_buses = {bus for bus in load_shares.get(country, {}) if not is_hvdc_helper_bus(bus)}
        if not candidate_buses:
            candidate_buses = {
                bus
                for bus in candidate_buses_for_country(country, source_country_to_buses, model_country_to_buses, cluster_map)
                if not is_hvdc_helper_bus(bus)
            }
        buses = sorted(candidate_buses | {bus for c, pt, tech, bus in target_by_key if (c, pt, tech) == (country, plant_type, technology)})
        for bus in buses:
            existing = target_by_key.get((country, plant_type, technology, bus))
            if existing is not None:
                rows.append({
                    "country": existing["country"],
                    "country_model": existing["country_model"],
                    "country_label": existing["country_label"],
                    "bus": existing["bus"],
                    "plant_type": existing["plant_type"],
                    "technology": existing["technology"],
                    "current_turb_mw": existing["current_turb_mw"],
                    "current_storage_mwh": existing["current_storage_mwh"],
                    "turbine_share": existing["turbine_share"],
                    "storage_share": existing["storage_share"],
                    "load_share_within_country": fmt_share(load_shares.get(country, {}).get(bus, 0.0)),
                    "allocation_rule": existing["allocation_rule"],
                })
                continue
            meta = bus_meta[bus]
            current_values = current.get((country, plant_type, bus), {})
            rows.append({
                "country": country,
                "country_model": meta.country_model,
                "country_label": meta.country_label,
                "bus": bus,
                "plant_type": plant_type,
                "technology": technology,
                "current_turb_mw": fmt_amount(float(current_values.get("turb", 0.0))),
                "current_storage_mwh": fmt_amount(float(current_values.get("storage", 0.0))),
                "turbine_share": fmt_share(0.0),
                "storage_share": fmt_share(0.0),
                "load_share_within_country": fmt_share(load_shares.get(country, {}).get(bus, 0.0)),
                "allocation_rule": "zero_share_candidate_bus",
            })
    return rows


def build_constraint_rows(expanded_constraints: list[dict[str, Any]], target: dict[tuple[str, str, str, str], dict[str, float]], bus_meta: dict[str, BusMeta]) -> list[dict[str, Any]]:
    out, turb_fields = [], ["installed_turb_mw", "min_turb_mw", "max_turb_mw", "min_turb_en_mwh_day", "max_turb_en_mwh_day", "min_turb_en_mwh_period", "max_turb_en_mwh_period"]
    pump_fields = ["installed_pump_mw", "min_pump_mw", "max_pump_mw", "min_pump_en_mwh_day", "max_pump_en_mwh_day", "min_pump_en_mwh_period", "max_pump_en_mwh_period"]
    for row in expanded_constraints:
        country, plant_type, tech = row["country"], row["plant_type"], row["technology"]
        keys = sorted(k for k in target if k[0] == country and k[1] == plant_type and k[2] == tech)
        total_turb, total_storage = sum(target[k]["turb"] for k in keys), sum(target[k]["storage"] for k in keys)
        for key in keys:
            bus = key[3]
            meta = bus_meta[bus]
            tsh = 0.0 if total_turb <= EPS else target[key]["turb"] / total_turb
            ssh = 0.0 if total_storage <= EPS else target[key]["storage"] / total_storage
            new = {"country": country, "country_model": meta.country_model, "country_label": meta.country_label, "bus": bus, "ref_year": row["ref_year"], "plant_type": plant_type, "technology": tech, "temporal_resolution": row["temporal_resolution"], "week": row["week"], "period_start_date": row["period_start_date"], "period_end_date": row["period_end_date"], "days_in_period": row["days_in_period"], "installed_storage_mwh": fmt_amount(target[key]["storage"]), "min_turb_pu": row["min_turb_pu"], "max_turb_pu": row["max_turb_pu"], "min_pump_pu": row["min_pump_pu"], "max_pump_pu": row["max_pump_pu"], "min_res_hist_pu": row["min_res_hist_pu"], "max_res_hist_pu": row["max_res_hist_pu"], "min_res_tech_pu": row["min_res_tech_pu"], "max_res_tech_pu": row["max_res_tech_pu"], "turbine_share": fmt_share(tsh), "storage_share": fmt_share(ssh), "allocation_rule": "capacity_weighted_country_tech_bus_scaling"}
            for field in turb_fields + pump_fields:
                new[field] = fmt_amount(parse_float(row.get(field)) * tsh)
            out.append(new)
    return out


def build_inflow_rows(inflow_groups: list[dict[str, Any]], target: dict[tuple[str, str, str, str], dict[str, float]], bus_meta: dict[str, BusMeta], year: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group in inflow_groups:
        country = str(group["country"])
        weather_year = int(group["weather_year"])
        week = int(group["week"])
        national_total = float(group["inflow_mwh_week"])
        eligible_techs = set(group["eligible_techs"])
        if not eligible_techs:
            continue
        bus_storage: dict[str, float] = defaultdict(float)
        bus_turb: dict[str, float] = defaultdict(float)
        for (c, plant_type, technology, bus), vals in target.items():
            if c == country and (plant_type, technology) in eligible_techs and technology != "closed_loop":
                bus_storage[bus] += vals["storage"]
                bus_turb[bus] += vals["turb"]
        total_storage, total_turb = sum(bus_storage.values()), sum(bus_turb.values())
        if total_storage <= EPS and total_turb <= EPS:
            continue
        # Reservoir inflow is storage-driven where possible. Run-of-river or
        # incomplete storage data fall back to turbine shares, which better
        # represent the local conversion capacity.
        bus_shares = (
            {bus: val / total_storage for bus, val in bus_storage.items() if val > EPS}
            if total_storage > EPS
            else {bus: val / total_turb for bus, val in bus_turb.items() if val > EPS}
        )
        bus_rule = "bus_total_storage_shares" if total_storage > EPS else "fallback_bus_total_turbine_shares"
        for bus, bshare in sorted(bus_shares.items()):
            bus_total = national_total * bshare
            bus_tech = {
                (plant_type, technology): vals["turb"]
                for (c, plant_type, technology, b), vals in target.items()
                if c == country and b == bus and (plant_type, technology) in eligible_techs and technology != "closed_loop" and vals["turb"] > EPS
            }
            total_bus_turb = sum(bus_tech.values())
            if total_bus_turb <= EPS:
                continue
            for (plant_type, technology), turb in sorted(bus_tech.items()):
                tech_share = turb / total_bus_turb
                meta = bus_meta[bus]
                out.append({
                    "country": country,
                    "country_model": meta.country_model,
                    "country_label": meta.country_label,
                    "bus": bus,
                    "ref_year": str(year),
                    "weather_year": weather_year,
                    "week": week,
                    "plant_type": plant_type,
                    "technology": technology,
                    "national_inflow_source": str(group["source"]),
                    "national_inflow_mwh_week": fmt_amount(national_total),
                    "bus_inflow_total_mwh_week": fmt_amount(bus_total),
                    "allocated_inflow_mwh_week": fmt_amount(bus_total * tech_share),
                    "bus_total_storage_share": fmt_share(bshare),
                    "bus_tech_turbine_share": fmt_share(tech_share),
                    "allocation_rule": f"{bus_rule};bus_tech_turbine_shares",
                })
    return out


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or not xs:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x <= EPS or den_y <= EPS:
        return None
    return num / (den_x * den_y)


def build_tyndp_nc_comparison_rows(
    combo_profiles: dict[tuple[str, str, str, int, int], float],
    combo_meta: dict[tuple[str, str, str], dict[str, Any]],
    target: dict[tuple[str, str, str, str], dict[str, float]],
    nc_totals: dict[tuple[str, int, int], float],
    nc_by_type: dict[tuple[str, str, int, int], float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[tuple[str, float]], list[tuple[str, list[float]]]]:
    open_target_combos = sorted({(country, plant_type, technology) for country, plant_type, technology, _bus in target if technology != "closed_loop"})
    countries = sorted({country for country, _plant_type, _technology in open_target_combos})

    country_weather_rows: list[dict[str, Any]] = []
    country_summary_rows: list[dict[str, Any]] = []
    tech_summary_rows: list[dict[str, Any]] = []
    tech_weekly_rows: list[dict[str, Any]] = []
    country_bar_values: list[tuple[str, float]] = []
    weekly_total_series: list[tuple[str, list[float]]] = []

    overall_tyndp_weekly = [0.0] * 52
    overall_nc_weekly = [0.0] * 52

    for country, plant_type, technology in open_target_combos:
        hydro_type = NC_HYDRO_TYPE_BY_PLANT_TYPE.get(plant_type)
        tyndp_weekly_means = []
        nc_weekly_means = []
        tyndp_year_totals = []
        nc_year_totals = []
        for weather_year in WEATHER_YEARS:
            tyndp_values = [combo_profiles.get((country, plant_type, technology, weather_year, week), 0.0) for week in WEEKS]
            nc_values = [nc_by_type.get((country, hydro_type, weather_year, week), 0.0) if hydro_type else 0.0 for week in WEEKS]
            tyndp_year_totals.append(sum(tyndp_values))
            nc_year_totals.append(sum(nc_values))
            tyndp_weekly_means.append(tyndp_values)
            nc_weekly_means.append(nc_values)
        mean_tyndp_weekly = [sum(year_vals[i] for year_vals in tyndp_weekly_means) / len(WEATHER_YEARS) for i in range(52)]
        mean_nc_weekly = [sum(year_vals[i] for year_vals in nc_weekly_means) / len(WEATHER_YEARS) for i in range(52)]
        corr = pearson_correlation(mean_tyndp_weekly, mean_nc_weekly)
        for week in WEEKS:
            tech_weekly_rows.append({
                "country": country,
                "plant_type": plant_type,
                "technology": technology,
                "week": week,
                "tyndp_mean_mwh_week": fmt_amount(mean_tyndp_weekly[week - 1]),
                "nc_mean_mwh_week": fmt_amount(mean_nc_weekly[week - 1]),
                "delta_mean_mwh_week": fmt_amount(mean_tyndp_weekly[week - 1] - mean_nc_weekly[week - 1]),
            })
        tech_summary_rows.append({
            "country": country,
            "plant_type": plant_type,
            "technology": technology,
            "nc_hydro_type": hydro_type or "",
            "tyndp_mean_annual_mwh": fmt_amount(sum(tyndp_year_totals) / len(tyndp_year_totals)),
            "nc_mean_annual_mwh": fmt_amount(sum(nc_year_totals) / len(nc_year_totals)),
            "delta_mean_annual_mwh": fmt_amount((sum(tyndp_year_totals) - sum(nc_year_totals)) / len(nc_year_totals)),
            "relative_delta_to_nc": fmt(0.0 if sum(nc_year_totals) <= EPS else ((sum(tyndp_year_totals) / len(tyndp_year_totals)) - (sum(nc_year_totals) / len(nc_year_totals))) / (sum(nc_year_totals) / len(nc_year_totals)), 8),
            "weekly_mean_corr": "" if corr is None else fmt(corr, 8),
            "tyndp_missing_or_zero": str(combo_meta.get((country, plant_type, technology), {}).get("all_years_missing_or_zero", False)),
            "zero_inflow_expected": str(combo_meta.get((country, plant_type, technology), {}).get("zero_inflow_expected", False)),
        })

    for country in countries:
        country_open_combos = [(c, pt, tech) for c, pt, tech in open_target_combos if c == country]
        tyndp_weekly_by_year: dict[int, list[float]] = {}
        nc_weekly_by_year: dict[int, list[float]] = {}
        for weather_year in WEATHER_YEARS:
            tyndp_values = [0.0] * 52
            nc_values = [nc_totals.get((country, weather_year, week), 0.0) for week in WEEKS]
            for _country, plant_type, technology in country_open_combos:
                for week in WEEKS:
                    tyndp_values[week - 1] += combo_profiles.get((country, plant_type, technology, weather_year, week), 0.0)
            tyndp_weekly_by_year[weather_year] = tyndp_values
            nc_weekly_by_year[weather_year] = nc_values
            country_weather_rows.append({
                "country": country,
                "weather_year": weather_year,
                "tyndp_total_mwh_year": fmt_amount(sum(tyndp_values)),
                "nc_total_mwh_year": fmt_amount(sum(nc_values)),
                "delta_mwh_year": fmt_amount(sum(tyndp_values) - sum(nc_values)),
                "relative_delta_to_nc": fmt(0.0 if sum(nc_values) <= EPS else (sum(tyndp_values) - sum(nc_values)) / sum(nc_values), 8),
            })
            for idx in range(52):
                overall_tyndp_weekly[idx] += tyndp_values[idx]
                overall_nc_weekly[idx] += nc_values[idx]
        mean_tyndp_weekly = [sum(tyndp_weekly_by_year[wy][i] for wy in WEATHER_YEARS) / len(WEATHER_YEARS) for i in range(52)]
        mean_nc_weekly = [sum(nc_weekly_by_year[wy][i] for wy in WEATHER_YEARS) / len(WEATHER_YEARS) for i in range(52)]
        mean_tyndp_annual = sum(sum(vals) for vals in tyndp_weekly_by_year.values()) / len(WEATHER_YEARS)
        mean_nc_annual = sum(sum(vals) for vals in nc_weekly_by_year.values()) / len(WEATHER_YEARS)
        corr = pearson_correlation(mean_tyndp_weekly, mean_nc_weekly)
        country_summary_rows.append({
            "country": country,
            "n_open_loop_target_combos": len(country_open_combos),
            "n_missing_or_zero_tyndp_combos": sum(1 for combo in country_open_combos if combo_meta.get(combo, {}).get("all_years_missing_or_zero", False)),
            "tyndp_mean_annual_mwh": fmt_amount(mean_tyndp_annual),
            "nc_mean_annual_mwh": fmt_amount(mean_nc_annual),
            "delta_mean_annual_mwh": fmt_amount(mean_tyndp_annual - mean_nc_annual),
            "relative_delta_to_nc": fmt(0.0 if mean_nc_annual <= EPS else (mean_tyndp_annual - mean_nc_annual) / mean_nc_annual, 8),
            "weekly_mean_corr": "" if corr is None else fmt(corr, 8),
        })
        country_bar_values.append((country, abs(mean_tyndp_annual - mean_nc_annual)))

    weekly_total_series.append(("TYNDP", [value / len(WEATHER_YEARS) for value in overall_tyndp_weekly]))
    weekly_total_series.append(("NC", [value / len(WEATHER_YEARS) for value in overall_nc_weekly]))
    return country_weather_rows, country_summary_rows, tech_summary_rows, tech_weekly_rows, country_bar_values, weekly_total_series


def write_tyndp_nc_comparison_report(
    outdir: Path,
    combo_profiles: dict[tuple[str, str, str, int, int], float],
    combo_meta: dict[tuple[str, str, str], dict[str, Any]],
    target: dict[tuple[str, str, str, str], dict[str, float]],
    nc_totals: dict[tuple[str, int, int], float],
    nc_by_type: dict[tuple[str, str, int, int], float],
) -> None:
    audit = ensure_dir(outdir / "audit")
    country_weather_rows, country_summary_rows, tech_summary_rows, tech_weekly_rows, country_bar_values, weekly_total_series = build_tyndp_nc_comparison_rows(
        combo_profiles,
        combo_meta,
        target,
        nc_totals,
        nc_by_type,
    )
    write_csv(
        audit / "hydro_tyndp_vs_nc_country_weather_year.csv",
        country_weather_rows,
        ["country", "weather_year", "tyndp_total_mwh_year", "nc_total_mwh_year", "delta_mwh_year", "relative_delta_to_nc"],
    )
    write_csv(
        audit / "hydro_tyndp_vs_nc_country_summary.csv",
        country_summary_rows,
        ["country", "n_open_loop_target_combos", "n_missing_or_zero_tyndp_combos", "tyndp_mean_annual_mwh", "nc_mean_annual_mwh", "delta_mean_annual_mwh", "relative_delta_to_nc", "weekly_mean_corr"],
    )
    write_csv(
        audit / "hydro_tyndp_vs_nc_country_tech_summary.csv",
        tech_summary_rows,
        ["country", "plant_type", "technology", "nc_hydro_type", "tyndp_mean_annual_mwh", "nc_mean_annual_mwh", "delta_mean_annual_mwh", "relative_delta_to_nc", "weekly_mean_corr", "tyndp_missing_or_zero", "zero_inflow_expected"],
    )
    write_csv(
        audit / "hydro_tyndp_vs_nc_country_tech_weekly_means.csv",
        tech_weekly_rows,
        ["country", "plant_type", "technology", "week", "tyndp_mean_mwh_week", "nc_mean_mwh_week", "delta_mean_mwh_week"],
    )
    top_countries = sorted(country_bar_values, key=lambda item: item[1], reverse=True)[:20]
    write_bar_svg(
        audit / "hydro_tyndp_vs_nc_mean_annual_diff_top20.svg",
        "TYNDP vs NC Mean Annual Inflow Difference Top 20",
        [country for country, _ in top_countries],
        [value for _, value in top_countries],
        x_label="Country",
        y_label="Absolute Mean Annual Difference [MWh]",
    )
    write_line_svg(
        audit / "hydro_tyndp_vs_nc_mean_weekly_total.svg",
        "Mean Weekly National Hydro Inflow: TYNDP vs NC",
        list(WEEKS),
        weekly_total_series,
        x_label="Week",
        y_label="Mean Weekly Total Inflow [MWh/week]",
    )


def build_capacity_summary_tables(
    capacity_diag: list[dict[str, Any]],
    current: dict[tuple[str, str, str], dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    current_totals: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"turb": 0.0, "storage": 0.0})
    for (country, plant_type, _bus), vals in current.items():
        current_totals[(country, plant_type)]["turb"] += vals["turb"]
        current_totals[(country, plant_type)]["storage"] += vals["storage"]

    target_totals: dict[tuple[str, str], dict[str, Any]] = {}
    for row in capacity_diag:
        key = (str(row["country"]), str(row["plant_type"]))
        entry = target_totals.setdefault(key, {
            "country": key[0],
            "plant_type": key[1],
            "target_turb_mw": 0.0,
            "target_storage_mwh": 0.0,
            "allocation_rule": set(),
            "n_buses": 0,
        })
        entry["target_turb_mw"] += parse_float(row["target_turb_mw"])
        entry["target_storage_mwh"] += parse_float(row["target_storage_mwh"])
        entry["allocation_rule"].add(str(row["allocation_rule"]))
        entry["n_buses"] = max(int(entry["n_buses"]), int(row["n_buses"]))

    all_keys = sorted(set(current_totals) | set(target_totals))
    plant_type_rows_raw: list[dict[str, Any]] = []
    for country, plant_type in all_keys:
        cur = current_totals.get((country, plant_type), {"turb": 0.0, "storage": 0.0})
        tgt = target_totals.get((country, plant_type), {
            "target_turb_mw": 0.0,
            "target_storage_mwh": 0.0,
            "allocation_rule": {"current_only_not_in_tyndp_target"},
            "n_buses": 0,
        })
        current_turb = float(cur["turb"])
        target_turb = float(tgt["target_turb_mw"])
        current_storage = float(cur["storage"])
        target_storage = float(tgt["target_storage_mwh"])
        delta_turb = target_turb - current_turb
        delta_storage = target_storage - current_storage
        if current_turb <= EPS and target_turb > EPS:
            status = "new_in_target"
        elif current_turb > EPS and target_turb <= EPS:
            status = "removed_in_target"
        elif abs(delta_turb) <= EPS and abs(delta_storage) <= EPS:
            status = "unchanged"
        else:
            status = "changed"
        plant_type_rows_raw.append({
            "country": country,
            "plant_type": plant_type,
            "technology": "all_technologies",
            "current_turb_mw": current_turb,
            "target_turb_mw": target_turb,
            "delta_turb_mw": delta_turb,
            "current_storage_mwh": current_storage,
            "target_storage_mwh": target_storage,
            "delta_storage_mwh": delta_storage,
            "allocation_rule": ";".join(sorted(tgt["allocation_rule"])),
            "n_buses": int(tgt["n_buses"]),
            "status": status,
            "current_positive": str(current_turb > EPS or current_storage > EPS),
            "target_positive": str(target_turb > EPS or target_storage > EPS),
        })

    country_total_acc: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "current_total_turb_mw": 0.0,
        "target_total_turb_mw": 0.0,
        "delta_total_turb_mw": 0.0,
        "current_total_storage_mwh": 0.0,
        "target_total_storage_mwh": 0.0,
        "delta_total_storage_mwh": 0.0,
        "sum_abs_plant_type_delta_turb_mw": 0.0,
        "sum_abs_plant_type_delta_storage_mwh": 0.0,
        "n_current_plant_types": 0,
        "n_target_plant_types": 0,
        "n_changed_plant_types": 0,
    })
    signature_acc: dict[str, dict[str, float]] = defaultdict(dict)
    tracked_types = ("ror", "wr", "phs")
    for row in plant_type_rows_raw:
        country = str(row["country"])
        plant_type = str(row["plant_type"])
        acc = country_total_acc[country]
        acc["current_total_turb_mw"] += float(row["current_turb_mw"])
        acc["target_total_turb_mw"] += float(row["target_turb_mw"])
        acc["delta_total_turb_mw"] += float(row["delta_turb_mw"])
        acc["current_total_storage_mwh"] += float(row["current_storage_mwh"])
        acc["target_total_storage_mwh"] += float(row["target_storage_mwh"])
        acc["delta_total_storage_mwh"] += float(row["delta_storage_mwh"])
        acc["sum_abs_plant_type_delta_turb_mw"] += abs(float(row["delta_turb_mw"]))
        acc["sum_abs_plant_type_delta_storage_mwh"] += abs(float(row["delta_storage_mwh"]))
        if float(row["current_turb_mw"]) > EPS or float(row["current_storage_mwh"]) > EPS:
            acc["n_current_plant_types"] += 1
        if float(row["target_turb_mw"]) > EPS or float(row["target_storage_mwh"]) > EPS:
            acc["n_target_plant_types"] += 1
        if abs(float(row["delta_turb_mw"])) > EPS or abs(float(row["delta_storage_mwh"])) > EPS:
            acc["n_changed_plant_types"] += 1
        if plant_type in tracked_types:
            for prefix in ("current", "target", "delta"):
                signature_acc[country][f"{prefix}_{plant_type}_turb_mw"] = float(row[f"{prefix}_turb_mw"])
                signature_acc[country][f"{prefix}_{plant_type}_storage_mwh"] = float(row[f"{prefix}_storage_mwh"])

    gap_rows = [{
        "country": row["country"],
        "plant_type": row["plant_type"],
        "technology": row["technology"],
        "target_turb_mw": fmt_amount(float(row["target_turb_mw"])),
        "current_turb_mw": fmt_amount(float(row["current_turb_mw"])),
        "delta_turb_mw": fmt_amount(float(row["delta_turb_mw"])),
        "target_storage_mwh": fmt_amount(float(row["target_storage_mwh"])),
        "current_storage_mwh": fmt_amount(float(row["current_storage_mwh"])),
        "delta_storage_mwh": fmt_amount(float(row["delta_storage_mwh"])),
        "allocation_rule": row["allocation_rule"],
        "n_buses": row["n_buses"],
        "status": row["status"],
        "current_positive": row["current_positive"],
        "target_positive": row["target_positive"],
    } for row in plant_type_rows_raw]

    country_total_rows_raw: list[dict[str, Any]] = []
    for country, acc in sorted(country_total_acc.items()):
        country_total_rows_raw.append({
            "country": country,
            "current_total_turb_mw": acc["current_total_turb_mw"],
            "target_total_turb_mw": acc["target_total_turb_mw"],
            "delta_total_turb_mw": acc["delta_total_turb_mw"],
            "current_total_storage_mwh": acc["current_total_storage_mwh"],
            "target_total_storage_mwh": acc["target_total_storage_mwh"],
            "delta_total_storage_mwh": acc["delta_total_storage_mwh"],
            "sum_abs_plant_type_delta_turb_mw": acc["sum_abs_plant_type_delta_turb_mw"],
            "possible_internal_type_shift_turb_mw": max(0.0, 0.5 * (acc["sum_abs_plant_type_delta_turb_mw"] - abs(acc["delta_total_turb_mw"]))),
            "sum_abs_plant_type_delta_storage_mwh": acc["sum_abs_plant_type_delta_storage_mwh"],
            "possible_internal_type_shift_storage_mwh": max(0.0, 0.5 * (acc["sum_abs_plant_type_delta_storage_mwh"] - abs(acc["delta_total_storage_mwh"]))),
            "n_current_plant_types": acc["n_current_plant_types"],
            "n_target_plant_types": acc["n_target_plant_types"],
            "n_changed_plant_types": acc["n_changed_plant_types"],
        })

    country_total_rows = [{
        "country": row["country"],
        "current_total_turb_mw": fmt_amount(float(row["current_total_turb_mw"])),
        "target_total_turb_mw": fmt_amount(float(row["target_total_turb_mw"])),
        "delta_total_turb_mw": fmt_amount(float(row["delta_total_turb_mw"])),
        "current_total_storage_mwh": fmt_amount(float(row["current_total_storage_mwh"])),
        "target_total_storage_mwh": fmt_amount(float(row["target_total_storage_mwh"])),
        "delta_total_storage_mwh": fmt_amount(float(row["delta_total_storage_mwh"])),
        "sum_abs_plant_type_delta_turb_mw": fmt_amount(float(row["sum_abs_plant_type_delta_turb_mw"])),
        "possible_internal_type_shift_turb_mw": fmt_amount(float(row["possible_internal_type_shift_turb_mw"])),
        "sum_abs_plant_type_delta_storage_mwh": fmt_amount(float(row["sum_abs_plant_type_delta_storage_mwh"])),
        "possible_internal_type_shift_storage_mwh": fmt_amount(float(row["possible_internal_type_shift_storage_mwh"])),
        "n_current_plant_types": row["n_current_plant_types"],
        "n_target_plant_types": row["n_target_plant_types"],
        "n_changed_plant_types": row["n_changed_plant_types"],
    } for row in country_total_rows_raw]

    signature_rows: list[dict[str, Any]] = []
    for country in sorted(country_total_acc):
        total_row = next(row for row in country_total_rows_raw if row["country"] == country)
        sig = signature_acc.get(country, {})
        out: dict[str, Any] = {"country": country}
        for plant_type in tracked_types:
            out[f"current_{plant_type}_turb_mw"] = fmt_amount(float(sig.get(f"current_{plant_type}_turb_mw", 0.0)))
            out[f"target_{plant_type}_turb_mw"] = fmt_amount(float(sig.get(f"target_{plant_type}_turb_mw", 0.0)))
            out[f"delta_{plant_type}_turb_mw"] = fmt_amount(float(sig.get(f"delta_{plant_type}_turb_mw", 0.0)))
            out[f"current_{plant_type}_storage_mwh"] = fmt_amount(float(sig.get(f"current_{plant_type}_storage_mwh", 0.0)))
            out[f"target_{plant_type}_storage_mwh"] = fmt_amount(float(sig.get(f"target_{plant_type}_storage_mwh", 0.0)))
            out[f"delta_{plant_type}_storage_mwh"] = fmt_amount(float(sig.get(f"delta_{plant_type}_storage_mwh", 0.0)))
        out["current_total_turb_mw"] = fmt_amount(float(total_row["current_total_turb_mw"]))
        out["target_total_turb_mw"] = fmt_amount(float(total_row["target_total_turb_mw"]))
        out["delta_total_turb_mw"] = fmt_amount(float(total_row["delta_total_turb_mw"]))
        out["possible_internal_type_shift_turb_mw"] = fmt_amount(float(total_row["possible_internal_type_shift_turb_mw"]))
        out["current_total_storage_mwh"] = fmt_amount(float(total_row["current_total_storage_mwh"]))
        out["target_total_storage_mwh"] = fmt_amount(float(total_row["target_total_storage_mwh"]))
        out["delta_total_storage_mwh"] = fmt_amount(float(total_row["delta_total_storage_mwh"]))
        out["possible_internal_type_shift_storage_mwh"] = fmt_amount(float(total_row["possible_internal_type_shift_storage_mwh"]))
        signature_rows.append(out)

    return gap_rows, country_total_rows, signature_rows


def write_audit(outdir: Path, capacity_diag: list[dict[str, Any]], inflow_diag: list[dict[str, Any]], flags: list[dict[str, Any]], target_rows: list[dict[str, Any]], current: dict[tuple[str, str, str], dict[str, float]], inflow_rows: list[dict[str, Any]]) -> None:
    audit = ensure_dir(outdir / "audit")
    gap_rows, country_total_rows, signature_rows = build_capacity_summary_tables(capacity_diag, current)
    gap_rows_basic = [{key: row[key] for key in ["country", "plant_type", "technology", "target_turb_mw", "current_turb_mw", "delta_turb_mw", "target_storage_mwh", "current_storage_mwh", "delta_storage_mwh", "allocation_rule", "n_buses"]} for row in gap_rows]
    plant_type_rows = [{key: row[key] for key in ["country", "plant_type", "current_turb_mw", "target_turb_mw", "delta_turb_mw", "current_storage_mwh", "target_storage_mwh", "delta_storage_mwh", "allocation_rule", "n_buses", "status", "current_positive", "target_positive"]} for row in gap_rows]
    write_csv(audit / "hydro_capacity_gap_by_country_tech.csv", gap_rows_basic, ["country", "plant_type", "technology", "target_turb_mw", "current_turb_mw", "delta_turb_mw", "target_storage_mwh", "current_storage_mwh", "delta_storage_mwh", "allocation_rule", "n_buses"])
    write_csv(audit / "hydro_current_vs_target_country_plant_type.csv", plant_type_rows, ["country", "plant_type", "current_turb_mw", "target_turb_mw", "delta_turb_mw", "current_storage_mwh", "target_storage_mwh", "delta_storage_mwh", "allocation_rule", "n_buses", "status", "current_positive", "target_positive"])
    write_csv(audit / "hydro_current_vs_target_country_total.csv", country_total_rows, ["country", "current_total_turb_mw", "target_total_turb_mw", "delta_total_turb_mw", "current_total_storage_mwh", "target_total_storage_mwh", "delta_total_storage_mwh", "sum_abs_plant_type_delta_turb_mw", "possible_internal_type_shift_turb_mw", "sum_abs_plant_type_delta_storage_mwh", "possible_internal_type_shift_storage_mwh", "n_current_plant_types", "n_target_plant_types", "n_changed_plant_types"])
    write_csv(audit / "hydro_current_vs_target_country_type_shift_signature.csv", signature_rows, ["country", "current_ror_turb_mw", "target_ror_turb_mw", "delta_ror_turb_mw", "current_wr_turb_mw", "target_wr_turb_mw", "delta_wr_turb_mw", "current_phs_turb_mw", "target_phs_turb_mw", "delta_phs_turb_mw", "current_ror_storage_mwh", "target_ror_storage_mwh", "delta_ror_storage_mwh", "current_wr_storage_mwh", "target_wr_storage_mwh", "delta_wr_storage_mwh", "current_phs_storage_mwh", "target_phs_storage_mwh", "delta_phs_storage_mwh", "current_total_turb_mw", "target_total_turb_mw", "delta_total_turb_mw", "possible_internal_type_shift_turb_mw", "current_total_storage_mwh", "target_total_storage_mwh", "delta_total_storage_mwh", "possible_internal_type_shift_storage_mwh"])
    write_csv(audit / "hydro_inflow_quality_by_country_tech.csv", inflow_diag, ["country", "plant_type", "technology", "input_rows", "interpolated_weeks", "imputed_weather_years", "all_zero_weather_years", "all_years_missing_or_zero", "zero_inflow_expected"])
    flag_fields = sorted({k for row in flags for k in row.keys()}) or ["country", "plant_type", "technology", "review_reason"]
    write_csv(audit / "hydro_review_flags.csv", flags, flag_fields)
    top = sorted(gap_rows, key=lambda r: abs(parse_float(r["delta_turb_mw"])), reverse=True)[:20]
    write_bar_svg(
        audit / "hydro_turbine_gap_top20.svg",
        "Hydro Target Minus Current Turbine Capacity Top 20",
        [f"{r['country']} {r['plant_type']}" for r in top],
        [abs(parse_float(r["delta_turb_mw"])) for r in top],
        x_label="Country / Plant Type",
        y_label="Absolute Turbine Capacity Gap [MW]",
    )
    total_top = sorted(country_total_rows, key=lambda r: abs(parse_float(r["delta_total_turb_mw"])), reverse=True)[:20]
    write_bar_svg(
        audit / "hydro_total_hydro_delta_turb_top20.svg",
        "Total Hydro Target Minus Current Turbine Capacity Top 20",
        [str(r["country"]) for r in total_top],
        [abs(parse_float(r["delta_total_turb_mw"])) for r in total_top],
        x_label="Country",
        y_label="Absolute Total Hydro Difference [MW]",
    )
    shift_top = sorted(country_total_rows, key=lambda r: parse_float(r["possible_internal_type_shift_turb_mw"]), reverse=True)[:20]
    write_bar_svg(
        audit / "hydro_internal_type_shift_turb_top20.svg",
        "Possible Internal Hydro Type Shift Top 20",
        [str(r["country"]) for r in shift_top],
        [parse_float(r["possible_internal_type_shift_turb_mw"]) for r in shift_top],
        x_label="Country",
        y_label="Possible Internal Type Shift [MW]",
    )
    if inflow_rows:
        totals, counts = defaultdict(lambda: [0.0] * 52), defaultdict(lambda: [0] * 52)
        for row in inflow_rows:
            pt, w = row["plant_type"], int(row["week"]) - 1
            totals[pt][w] += parse_float(row["allocated_inflow_mwh_week"])
            counts[pt][w] += 1
        series = [(pt, [(0.0 if counts[pt][i] == 0 else totals[pt][i] / counts[pt][i]) for i in range(52)]) for pt in sorted(totals)]
        write_line_svg(
            audit / "hydro_mean_weekly_bus_inflow_by_plant_type.svg",
            "Mean Weekly Hydro Bus Inflow by Plant Type",
            list(WEEKS),
            series,
            x_label="Week",
            y_label="Mean Allocated Inflow [MWh/week]",
        )
    write_json(audit / "hydro_audit_summary.json", {"n_target_capacity_rows": len(target_rows), "n_gap_rows": len(gap_rows), "n_inflow_quality_rows": len(inflow_diag), "n_review_flags": len(flags)})


def main() -> None:
    args = resolve_args()
    root = args.project_root.resolve()
    hydro_dir = args.hydro_dir.resolve() if args.hydro_dir else (root / "hydro" / "tyndp2024")
    network_dir = args.network_dir.resolve() if args.network_dir else infer_network_dir(root, args.target_year)
    country_profiles_dir = args.country_profiles_dir.resolve() if args.country_profiles_dir else infer_country_profiles_dir(root)
    load_csv = args.load_csv.resolve() if args.load_csv else auto_load_csv(root, args.target_year, network_dir)
    outdir = args.output_dir.resolve() if args.output_dir else (root / "hydro" / f"target_year_{args.target_year}" / network_dir.name)
    country_clusters_csv = args.country_clusters_csv.resolve() if args.country_clusters_csv else (network_dir / "cesa_country_clusters.csv")
    country_reductions_csv = args.country_reductions_csv.resolve() if args.country_reductions_csv else (root / "country_reductions.csv")
    excluded_countries_csv = args.excluded_countries_csv.resolve() if args.excluded_countries_csv else (network_dir / "excluded_countries.csv")
    ensure_dir(outdir)
    constraints_csv = find_hydro_csv(hydro_dir, "hydro_constraints_country_weekly", args.target_year)
    inflows_csv = find_hydro_csv(hydro_dir, "hydro_inflows_country_weekly", args.target_year)
    plants_csv, buses_csv = network_dir / "plants.csv", network_dir / "buses.csv"

    cluster_map = read_country_clusters(country_clusters_csv)
    excluded_countries = read_excluded_countries(excluded_countries_csv)
    bus_meta, source_country_to_buses, model_country_to_buses = load_bus_meta(plants_csv, buses_csv, cluster_map)
    current = load_current_weights(plants_csv, args.resolve_phs)
    load_scores = load_bus_scores(load_csv)
    load_shares = build_load_shares_by_source_country(source_country_to_buses, model_country_to_buses, load_scores, cluster_map)
    expanded_constraints, constraint_diag, resolved_component_map = load_constraints(constraints_csv, args.target_year, args.resolve_phs)
    expanded_constraints, resolved_component_map, skipped_excluded_hydro_countries = filter_excluded_hydro_countries(
        expanded_constraints,
        resolved_component_map,
        excluded_countries,
    )
    expanded_constraints, resolved_component_map, country_filter_flags, skipped_hydro_countries = filter_hydro_countries_for_network(
        expanded_constraints,
        resolved_component_map,
        source_country_to_buses,
        model_country_to_buses,
        cluster_map,
    )
    target_rows, target, capacity_diag = build_target_rows(
        expanded_constraints,
        current,
        source_country_to_buses,
        model_country_to_buses,
        load_shares,
        bus_meta,
        cluster_map,
    )

    write_csv(outdir / "disaggregated_hydro_bus_capacities.csv", target_rows, CAPACITY_FIELDS)
    share_rows = build_share_rows(
        target_rows,
        current,
        source_country_to_buses,
        model_country_to_buses,
        load_shares,
        bus_meta,
        cluster_map,
    )
    write_csv(outdir / "hydro_bus_allocation_shares.csv", share_rows, SHARE_FIELDS)

    inflow_diag: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = list(country_filter_flags)
    inflow_rows: list[dict[str, Any]] = []
    nc_profile_meta: dict[str, Any] = {}
    if not args.audit_only:
        constraint_rows = build_constraint_rows(expanded_constraints, target, bus_meta)
        write_csv(outdir / "disaggregated_hydro_bus_constraints_weekly.csv", constraint_rows, CONSTRAINT_FIELDS)
        if args.include_inflows:
            combo_profiles, inflow_diag, inflow_flags, combo_meta, redirected_profiles = load_inflows(inflows_csv, args.target_year, resolved_component_map)
            flags.extend(inflow_flags)
            nc_countries = {
                country for (country, _plant_type, technology, _bus) in target if technology != "closed_loop"
            }
            nc_totals: dict[tuple[str, int, int], float] = {}
            nc_by_type: dict[tuple[str, str, int, int], float] = {}
            if nc_countries:
                nc_totals, nc_by_type, nc_meta, nc_loader = load_nc_weekly_profiles(country_profiles_dir, nc_countries)
                nc_profile_meta = {
                    "loader": nc_loader,
                    "requested_countries": sorted(nc_countries),
                    "available_countries": list(nc_meta.countries),
                    "hydro_types": list(nc_meta.hydro_types),
                }
                write_tyndp_nc_comparison_report(outdir, combo_profiles, combo_meta, target, nc_totals, nc_by_type)
            inflow_groups, nc_flags = build_inflow_groups(combo_profiles, combo_meta, redirected_profiles, target, nc_totals, nc_by_type, args.resolve_phs)
            flags.extend(nc_flags)
            inflow_rows = build_inflow_rows(inflow_groups, target, bus_meta, args.target_year)
            write_csv(outdir / "disaggregated_hydro_bus_inflows_weekly.csv", inflow_rows, INFLOW_FIELDS)

    write_audit(outdir, capacity_diag, inflow_diag, flags, target_rows, current, inflow_rows)
    write_json(outdir / "hydro_disaggregation_manifest.json", {
        "target_year": args.target_year,
        "config": None if args.config is None else str(args.config.resolve()),
        "project_root": str(root),
        "hydro_dir": str(hydro_dir),
        "network_dir": str(network_dir),
        "country_profiles_dir": str(country_profiles_dir),
        "load_csv": str(load_csv),
        "output_dir": str(outdir),
        "country_clusters_csv": str(country_clusters_csv),
        "country_reductions_csv": str(country_reductions_csv),
        "excluded_countries_csv": str(excluded_countries_csv) if excluded_countries_csv.exists() else None,
        "excluded_countries": sorted(excluded_countries),
        "constraints_csv": str(constraints_csv),
        "inflows_csv": str(inflows_csv),
        "resolve_phs": bool(args.resolve_phs),
        "include_inflows": bool(args.include_inflows),
        "audit_only": bool(args.audit_only),
        "skipped_excluded_hydro_countries": skipped_excluded_hydro_countries,
        "skipped_hydro_countries_without_buses": skipped_hydro_countries,
        "constraint_expansion_diagnostics": constraint_diag,
        "cluster_map": {
            "source_to_target": cluster_map.source_to_target,
            "target_to_sources": cluster_map.target_to_sources,
            "target_to_label": cluster_map.target_to_label,
        },
        "nc_profile_meta": nc_profile_meta,
        "notes": [
            "Week 53 is filtered out.",
            "Constraint combos with only week 1 are replicated to weeks 1-52.",
            "PHS resolution is controlled by resolve_phs. If enabled, input PHS is internally resolved with the same country-level logic as the original hydro preprocessing: open_loop PHS -> WR/open_loop, closed_loop PHS -> ROR/open_loop if no WR but ROR exists, otherwise -> WR/closed_loop if neither WR nor ROR exists, else -> WR/open_loop.",
            "If resolve_phs is disabled and a country contains both PHS/open_loop and PHS/closed_loop, the open-loop PHS capacity is collapsed into PHS/closed_loop. Its former natural inflow is redirected to WR/ROR open-loop targets when available and otherwise dropped.",
            "If resolve_phs is disabled and a country keeps PHS/open_loop as an open target, missing or zero TYNDP inflows for that combo can fall back to the NC hydro_type=phs profile and otherwise to the national NC total profile.",
            "If resolve_phs is enabled, existing Pumped Storage in plants.csv is mapped heuristically to WR if a reservoir exists in the country, otherwise to ROR if no reservoir exists but ROR exists.",
            "Country inflows can mix TYNDP-derived and nc fallback groups before bus allocation.",
            "Source-country to model-country aggregation is driven by the active network cluster file.",
            "Closed-loop targets are excluded from bus inflow allocation because they do not receive natural inflow.",
            "Country profile fallbacks accept hydro_country_profiles_YYYY.nc and .nc4 files and use hourly e_avail/e_avail_total aggregated to weeks 1-52.",
            "The nc loader prefers xarray/netCDF4 when available and falls back to ncdump otherwise.",
            "hydro_bus_allocation_shares.csv includes all load buses per country/technology target combo; non-allocated load buses are written with zero shares, while pure DC/HVDC helper buses such as cl_dc... are not added as zero-share rows.",
        ],
    })
    print(f"Wrote hydro disaggregation outputs to {outdir}")


if __name__ == "__main__":
    main()
