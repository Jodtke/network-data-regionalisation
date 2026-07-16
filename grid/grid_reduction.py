# -*- coding: utf-8 -*-
"""Build reduced European AC network representations for scenario regionalisation.

The reduction starts from a PyPSA-style grid, optionally adds selected
TYNDP expansion projects, and maps the full bus set to a smaller set of
AC clusters. The reduced grid is not meant to be a purely geographic
aggregation: cluster budgets are distributed by country or country aggregate,
weighted by existing generation and load, and the actual clustering can use
electrical similarities based on inverse reactance or DC-effective distance.

DC buses and converter terminals are treated as boundary objects. They are not
clustered as independent AC nodes; instead, their AC terminal assignment follows
the reduced AC cluster map, after which HVDC links are either rerouted between
clusters or removed if both terminals collapse into the same reduced AC node.
The outputs are the reduced buses, equivalent AC corridors, HVDC links or
AC-link substitutes, and diagnostics used by the downstream disaggregation
modules.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import warnings
from typing import Any, Callable, Sequence
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx

from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.neighbors import BallTree
import sklearn.cluster._kmeans as sklearn_kmeans
import sklearn.utils.parallel as sklearn_parallel

import pycountry
from shapely.geometry import Point, box
from shapely.ops import nearest_points
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grid import diagnose_tyndp2020_integration as tyndp2020
try:
    from configs.pipeline_config import deep_merge, ensure_list, load_yaml_like, resolve_path
except ModuleNotFoundError:
    from pipeline_config import deep_merge, ensure_list, load_yaml_like, resolve_path


#SYNC_COLLAPSE = False
#INCLUDE_TYNDP2020 = True
#TARGET_YEAR = 2030
#TYNDP_BASE_SNAPSHOT_YEAR = 2025

#PROJECT_ROOT = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf")
#BASE_DATA = PROJECT_ROOT / "grid"
#MODIFIED_FOLDER = "modified"
#if INCLUDE_TYNDP2020:
#    MODIFIED_PATH = BASE_DATA / MODIFIED_FOLDER / f"target_year_{TARGET_YEAR}"


class _NoThreadpoolController:
    def info(self) -> list[object]:
        return []

    def limit(self, *args: object, **kwargs: object) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


def _disable_sklearn_threadpool_probe() -> None:
    """Avoid threadpoolctl crashes during scikit-learn threadpool diagnostics."""
    def _noop_check(self: object, X: object, n_samples: object) -> None:
        return None

    for cls in (KMeans, MiniBatchKMeans):
        if hasattr(cls, "_check_mkl_vcomp"):
            setattr(cls, "_check_mkl_vcomp", _noop_check)

    def _controller_factory() -> _NoThreadpoolController:
        return _NoThreadpoolController()

    sklearn_kmeans._get_threadpool_controller = _controller_factory
    sklearn_parallel._get_threadpool_controller = _controller_factory


_disable_sklearn_threadpool_probe()
#else:
#    MODIFIED_PATH = BASE_DATA / MODIFIED_FOLDER
#os.makedirs(MODIFIED_PATH, exist_ok=True)

#CESA_COUNTRY_CLUSTER_CODES = [
    #"A1", # Baltikum (Estland-Lettland-Litauen)
    #"A2", # Deutschland-Luxemburg
    #"A3", # Moldau-Ukraine
    #"A4", # Serbien-Kosovo
    #"A5", # Albanien-Mazedonien
    #"A6", # Slowenien-Kroatien
    #"A7", # Montenegro-Bosnien
    #"A8", # Iberia
    #"A9", # Deutschland-Luxemburg-Dänemark
    #"A10", # Balkan (Bosnien-Herzegowina-Serbien-Montenegro-Kosovo)
    #"A11" # Griechenland-Albanien-Mazedonien
#]

FUEL_COLORS = {
    "Hydro": "#4169E1",               # RoyalBlue
    "Hard Coal": "#2F4F4F",           # DarkSlateGray
    "Natural Gas": "#FFA500",         # Orange
    "Lignite": "#8B4513",             # SaddleBrown
    "Oil": "#000000",                 # black
    "Wind": "#87CEFA",                # LightSkyBlue
    "Solid Biomass": "#006400",       # DarkGreen
    "Waste": "#800080",               # Purple
    "Solar": "#FFD700",               # Gold
    "Geothermal": "#D2B48C",          # Tan
    "Battery": "#FF69B4",             # HotPink
    "Heat Storage": "#8B0000",        # DarkRed
    "Nuclear": "#FF6347",             # Tomato
    "Other": "#D3D3D3",               # LightGray
    "Biogas": "#90EE90",              # LightGreen
    "Mechanical Storage": "#808080",  # Gray
    "Hydrogen Storage": "#40E0D0",    # Turquoise
}

VOLTAGE_COLORS = {
    110: "#66c2a5",
    132: "#66c2a5",
    150: "#66c2a5",
    220: "#00cc44", # green
    275: "#00cc44",
    300: "#ff7f00", # orange
    330: "#ff7f00",
    380: "#e41a1c", # red
    400: "#e41a1c",
    500: "#984ea3",
    750: "#f781bf",
}

HVDC_COLOR = "#00BFFF" # cyan

EARTH_R_KM = 6371.0088

DEFAULT_PROJECT_ROOT = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\pypsa\opf")
DEFAULT_COUNTRIES_URL = Path(r"Y:\Data\Natural Earth\110m_cultural\ne_110m_admin_0_countries.shp")
DEFAULT_ENTSOE_LOAD_DIR = Path(r"Y:\Data\ENTSOE\ftp_server\load\ActualTotalLoad_6.1.A")
COUNTRIES_URL = str(DEFAULT_COUNTRIES_URL)
GRID_SOURCE_BASE = "base"
GRID_SOURCE_TYNDP_NEW = "tyndp2020_new"
GRID_SOURCE_TYNDP_UPGRADE = "tyndp2020_upgrade"
DEFAULT_BBOX = (-12.0, 35.0, 35.0, 72.0)
DEFAULT_CESA_COUNTRY_CLUSTER_CODES = [
    "A3",
    "A4"
]
ENTSOE_LOAD_EXCLUDED_MAP_CODES = (
    "DE_50HZT",
    "DE_AMPRION",
    "DE_LU",
    "DE_TENNET_GER",
    "DE_TRANSNETBW",
    "DK1",
    "DK2",
    "SE1",
    "SE2",
    "SE3",
    "SE4",
    "NO1",
    "NO2",
    "NO3",
    "NO4",
    "NO5",
    "IT-CALABRIA",
    "IT-CNORTH",
    "IT-CSOUTH",
    "IT-NORTH",
    "IT-SARDINIA",
    "IT-SICILY",
    "IT-SOUTH",
)

    
# ============================================================
# I/O
# ============================================================
def read_plants_csv(path: Path) -> pd.DataFrame:
    out = pd.read_csv(path, sep=_detect_delimiter(path), engine="python")
    unnamed_cols = [col for col in out.columns if str(col).startswith("Unnamed:")]
    if "id" not in out.columns and unnamed_cols:
        out = out.rename(columns={unnamed_cols[0]: "id"})
    return out


def read_net_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=",", quotechar="'", engine="python")


def _detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        header = fh.readline()

    counts = {sep: header.count(sep) for sep in ("\t", ";", ",")}
    sep, count = max(counts.items(), key=lambda kv: kv[1])
    return sep if count > 0 else ","


def _find_existing_column(columns: Sequence[str], candidates: Sequence[str], *, context: str) -> str:
    by_lower = {str(c).strip().lower(): str(c) for c in columns}
    for cand in candidates:
        match = by_lower.get(str(cand).strip().lower())
        if match is not None:
            return match
    raise KeyError(f"Could not find {context} column. Available columns: {list(columns)}")


def _to_numeric_loose(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce")

    out = values.astype(str).str.strip()
    out = out.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
    out = out.str.replace(" ", "", regex=False)

    comma_decimal = out.str.contains(",", regex=False, na=False) & ~out.str.contains(".", regex=False, na=False)
    out = out.where(~comma_decimal, out.str.replace(",", ".", regex=False))
    return pd.to_numeric(out, errors="coerce")


def _sum_min_count_one(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.sum(min_count=1)


def _resolution_code_to_hours(code: Any) -> float:
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?", str(code).strip().upper())
    if match is None:
        return np.nan

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    total_hours = hours + minutes / 60.0
    return total_hours if total_hours > 0 else np.nan


def read_entsoe_country_mean_load(
    load_dir: str | Path,
    *,
    year: int = 2025,
    excluded_map_codes: Sequence[str] | None = None,
    target_countries: Sequence[str] | None = None,
) -> pd.Series:
    """
    Read ENTSOE monthly ActualTotalLoad files and return the mean load per country
    for the given year in MW. If multiple rows for the same country/timestamp exist
    (e.g. BZN/CTY duplicates), they are averaged first so they do not bias the mean.
    """
    load_dir = Path(load_dir)
    if not load_dir.exists():
        raise FileNotFoundError(f"ENTSOE load directory does not exist: {load_dir}")

    excluded = {
        str(code).strip().upper()
        for code in (excluded_map_codes if excluded_map_codes is not None else ENTSOE_LOAD_EXCLUDED_MAP_CODES)
    }

    target_idx: pd.Index | None = None
    if target_countries is not None:
        target_idx = pd.Index(pd.Series(list(target_countries), dtype="object").dropna().astype(str).str.upper().unique())

    files = sorted(load_dir.glob(f"{year}_*_ActualTotalLoad_6.1.A.csv"))
    if not files:
        files = sorted(load_dir.glob(f"{year}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No ENTSOE load files found for {year} in {load_dir}")

    load_numerator = defaultdict(float)
    load_hours = defaultdict(float)

    for path in files:
        sep = _detect_delimiter(path)
        header = pd.read_csv(path, sep=sep, nrows=0, encoding="utf-8-sig", engine="python")

        dt_col = _find_existing_column(header.columns, ["DateTime(UTC)", "DateTime"], context="datetime")
        map_col = _find_existing_column(header.columns, ["AreaMapCode", "MapCode"], context="area map code")
        load_col = _find_existing_column(header.columns, ["TotalLoad[MW]", "TotalLoadValue"], context="load")
        res_col = _find_existing_column(header.columns, ["ResolutionCode"], context="resolution")

        df = pd.read_csv(
            path,
            sep=sep,
            usecols=[dt_col, map_col, load_col, res_col],
            encoding="utf-8-sig",
            engine="python",
        ).rename(
            columns={
                dt_col: "timestamp",
                map_col: "country_code",
                load_col: "load_mw",
                res_col: "resolution_code",
            }
        )

        df["country_code"] = df["country_code"].astype(str).str.strip().str.upper()
        df = df.loc[~df["country_code"].isin(excluded)].copy()

        if target_idx is not None:
            df = df.loc[df["country_code"].isin(target_idx)].copy()

        if df.empty:
            continue

        ts_raw = df["timestamp"].astype(str).str.strip()
        iso_mask = ts_raw.str.match(r"^\d{4}-\d{2}-\d{2}")
        ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

        if iso_mask.any():
            ts.loc[iso_mask] = pd.to_datetime(
                ts_raw.loc[iso_mask],
                errors="coerce",
                utc=True,
            )
        if (~iso_mask).any():
            ts.loc[~iso_mask] = pd.to_datetime(
                ts_raw.loc[~iso_mask],
                errors="coerce",
                utc=True,
                dayfirst=True,
            )

        df = df.loc[ts.dt.year.eq(year)].copy()
        if df.empty:
            continue

        df["timestamp"] = ts.loc[df.index].dt.strftime("%Y-%m-%d %H:%M:%S")
        df["load_mw"] = _to_numeric_loose(df["load_mw"])
        df["resolution_hours"] = df["resolution_code"].map(_resolution_code_to_hours).fillna(1.0)
        df = df.dropna(subset=["timestamp", "country_code", "load_mw"])
        df = df.loc[df["resolution_hours"] > 0].copy()

        if df.empty:
            continue

        dedup = (
            df.groupby(["country_code", "timestamp"], as_index=False)
            .agg(
                load_mw=("load_mw", "mean"),
                resolution_hours=("resolution_hours", "max"),
            )
        )
        dedup["weighted_load_mwh"] = dedup["load_mw"] * dedup["resolution_hours"]

        by_country = (
            dedup.groupby("country_code", as_index=False)
            .agg(
                weighted_load_mwh=("weighted_load_mwh", "sum"),
                total_hours=("resolution_hours", "sum"),
            )
        )

        for row in by_country.itertuples(index=False):
            load_numerator[str(row.country_code)] += float(row.weighted_load_mwh)
            load_hours[str(row.country_code)] += float(row.total_hours)

    if target_idx is None:
        index = pd.Index(sorted(load_numerator.keys()), dtype="object")
    else:
        index = target_idx

    if len(index) == 0:
        return pd.Series(dtype=float, name=f"mean_load_mw_{year}")

    values = []
    for country in index:
        hours = float(load_hours.get(str(country), 0.0))
        if hours > 0:
            values.append(float(load_numerator.get(str(country), 0.0)) / hours)
        else:
            values.append(0.0)

    out = pd.Series(values, index=index, dtype=float, name=f"mean_load_mw_{year}")
    out.index = out.index.astype(str)
    return out


def read_country_reductions_csv(
    path: str | Path,
    *,
    selected_codes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Read country reduction definitions from CSV and return them in the
    target_country / target_label / member_countries format expected by the
    CESA country-cluster normalization.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Country reductions CSV does not exist: {path}")

    sep = _detect_delimiter(path)
    header = pd.read_csv(path, sep=sep, nrows=0, encoding="utf-8-sig", engine="python")

    code_col = _find_existing_column(
        header.columns,
        ["country_code", "target_country", "code"],
        context="country reduction code",
    )
    label_col = _find_existing_column(
        header.columns,
        ["country_label", "target_label", "label"],
        context="country reduction label",
    )
    members_col = _find_existing_column(
        header.columns,
        ["member_countries", "source_countries", "countries"],
        context="member countries",
    )

    out = pd.read_csv(
        path,
        sep=sep,
        usecols=[code_col, label_col, members_col],
        encoding="utf-8-sig",
        engine="python",
    ).rename(
        columns={
            code_col: "target_country",
            label_col: "target_label",
            members_col: "member_countries",
        }
    )

    out["target_country"] = out["target_country"].astype(str).str.strip().str.upper()
    out["target_label"] = out["target_label"].astype(str).str.strip()
    out["member_countries"] = out["member_countries"].apply(
        lambda x: ", ".join(_parse_country_members(x))
    )
    out = out.loc[
        out["target_country"].ne("")
        & out["target_label"].ne("")
        & out["member_countries"].ne("")
    ].copy()

    if selected_codes is None:
        return out.reset_index(drop=True)

    if isinstance(selected_codes, str):
        selected_codes = [selected_codes]

    selected_idx = pd.Index(
        pd.Series(list(selected_codes), dtype="object")
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    ).drop_duplicates()

    if len(selected_idx) == 0:
        return pd.DataFrame(columns=["target_country", "target_label", "member_countries"])

    out_idx = out.set_index("target_country", drop=False)
    missing = selected_idx.difference(out_idx.index)
    if not missing.empty:
        raise ValueError(
            "Requested country reduction codes are missing in CSV: "
            f"{list(missing)}"
        )

    return out_idx.loc[selected_idx].reset_index(drop=True)


def filter_plants_in_operation(
    plants: pd.DataFrame,
    *,
    target_year: int,
    date_out_col: str = "DateOut",
) -> pd.DataFrame:
    """
    Keep plants that are still in operation in the target year.

    Rule:
      - keep if DateOut is missing
      - keep if DateOut > target_year
      - drop if DateOut <= target_year
    """
    out = plants.copy()
    if date_out_col not in out.columns:
        return out

    date_out = _to_numeric_loose(out[date_out_col])
    keep = date_out.isna() | date_out.gt(float(target_year))
    return out.loc[keep].copy()


def _parse_country_members(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return []
        txt = txt.strip("[](){}")
        parts = re.split(r"[;,]", txt)
    elif isinstance(value, (list, tuple, set, pd.Series, np.ndarray)):
        parts = list(value)
    else:
        parts = [value]

    out: list[str] = []
    for item in parts:
        val = str(item).strip().strip("'\"").upper()
        if val:
            out.append(val)
    return out


def _normalize_country_token(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    upper = text.upper()
    aliases = {
        "UK": "GB",
        "GBR": "GB",
        "UNITED KINGDOM": "GB",
        "GREAT BRITAIN": "GB",
        "NORTHERN IRELAND": "NI",
        "NIR": "NI",
        "UKRAINE": "UA",
        "UKR": "UA",
        "MOLDOVA": "MD",
        "REPUBLIC OF MOLDOVA": "MD",
        "MDA": "MD",
        "EL": "GR",
        "KV": "XK",
        "KO": "XK",
        "KOSOVO": "XK",
    }
    if upper in aliases:
        return aliases[upper]
    try:
        match = pycountry.countries.lookup(text)
    except LookupError:
        compact = re.sub(r"[^A-Z0-9]+", "", upper)
        return aliases.get(compact, upper)
    return str(match.alpha_2).upper()


def _normalize_cesa_country_clusters(country_clusters: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["source_country", "target_country", "target_label"]
    empty = pd.DataFrame(columns=cols)

    if country_clusters is None:
        return empty

    D = country_clusters.copy()
    if D.empty:
        return empty

    if {"source_country", "target_country"}.issubset(D.columns):
        out = D.copy()
        if "target_label" not in out.columns:
            out["target_label"] = out["target_country"]
        out = out[cols].copy()

    elif {"member_countries", "target_country"}.issubset(D.columns):
        rows: list[dict[str, str]] = []
        for row in D.to_dict(orient="records"):
            target_country = str(row["target_country"]).strip().upper()
            target_label = str(row.get("target_label", target_country)).strip()
            members = _parse_country_members(row["member_countries"])

            for source_country in members:
                rows.append(
                    {
                        "source_country": source_country,
                        "target_country": target_country,
                        "target_label": target_label if target_label else target_country,
                    }
                )

        out = pd.DataFrame(rows, columns=cols)
    else:
        raise ValueError(
            "cesa_country_clusters must contain either "
            "('source_country', 'target_country') or ('member_countries', 'target_country')."
        )

    if out.empty:
        return empty

    out["source_country"] = out["source_country"].astype(str).str.strip().str.upper()
    out["target_country"] = out["target_country"].astype(str).str.strip().str.upper()
    out["target_label"] = out["target_label"].astype(str).str.strip()
    out = out[(out["source_country"] != "") & (out["target_country"] != "")].copy()

    if out.empty:
        return empty

    conflict = out.groupby("source_country")["target_country"].nunique()
    conflict = conflict[conflict > 1]
    if not conflict.empty:
        raise ValueError(
            "Each source country may only map to one target country. "
            f"Conflicts found for: {list(conflict.index)}"
        )

    out = (
        out.drop_duplicates(subset=["source_country", "target_country"])
        .sort_values(["target_country", "source_country"])
        .reset_index(drop=True)
    )
    return out


def build_excluded_country_rows(
    country_reductions: pd.DataFrame,
    *,
    excluded_country_cluster_codes: Sequence[Any] | None,
    excluded_countries: Sequence[Any] | None,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    excluded_codes = {
        str(value).strip().upper()
        for value in ensure_list(excluded_country_cluster_codes)
        if str(value).strip()
    }

    if excluded_codes and country_reductions is not None and not country_reductions.empty:
        reductions_raw = country_reductions.copy()
        if "target_country" not in reductions_raw.columns:
            raise ValueError("Country reductions for excluded cluster codes require a target_country column.")
        reductions_raw = reductions_raw[
            reductions_raw["target_country"].astype(str).str.strip().str.upper().isin(excluded_codes)
        ].copy()
        reductions = _normalize_cesa_country_clusters(reductions_raw)
        for row in reductions.itertuples(index=False):
            target = str(row.target_country).strip().upper()
            if target not in excluded_codes:
                continue
            rows.append(
                {
                    "source_country": _normalize_country_token(row.source_country),
                    "exclusion_code": target,
                    "exclusion_label": str(row.target_label or target).strip() or target,
                    "exclusion_source": "country_cluster_code",
                }
            )

    for value in ensure_list(excluded_countries):
        country = _normalize_country_token(value)
        if not country:
            continue
        rows.append(
            {
                "source_country": country,
                "exclusion_code": country,
                "exclusion_label": country,
                "exclusion_source": "country",
            }
        )

    if not rows:
        return pd.DataFrame(columns=["source_country", "exclusion_code", "exclusion_label", "exclusion_source"])

    out = pd.DataFrame(rows)
    out = out[out["source_country"].astype(str).str.strip().ne("")]
    return (
        out.drop_duplicates(subset=["source_country", "exclusion_code", "exclusion_source"])
        .sort_values(["exclusion_code", "source_country", "exclusion_source"])
        .reset_index(drop=True)
    )


def filter_country_clusters_for_exclusions(
    country_clusters: pd.DataFrame,
    excluded_country_rows: pd.DataFrame,
    excluded_country_cluster_codes: Sequence[Any] | None,
) -> pd.DataFrame:
    out = _normalize_cesa_country_clusters(country_clusters)
    if out.empty:
        return out

    excluded_sources = set(
        excluded_country_rows.get("source_country", pd.Series(dtype="object")).astype(str)
    )
    excluded_codes = {
        str(value).strip().upper()
        for value in ensure_list(excluded_country_cluster_codes)
        if str(value).strip()
    }
    keep = pd.Series(True, index=out.index)
    if excluded_sources:
        keep &= ~out["source_country"].astype(str).isin(excluded_sources)
    if excluded_codes:
        keep &= ~out["target_country"].astype(str).isin(excluded_codes)
    return out.loc[keep].reset_index(drop=True)


def _collapse_country_codes(values: pd.Series, source_to_target: pd.Series) -> pd.Series:
    out = values.astype(str).str.strip().str.upper()
    return out.map(source_to_target).fillna(out)


def _collapse_country_series(country_values: pd.Series | None, source_to_target: pd.Series) -> pd.Series | None:
    if country_values is None:
        return None

    out = _to_numeric_loose(country_values).fillna(0.0).astype(float).copy()
    if out.empty:
        return out

    idx = pd.Series(out.index.astype(str), index=out.index, dtype="object")
    out.index = _collapse_country_codes(idx, source_to_target).values
    out = out.groupby(level=0).sum().sort_index()
    return out


def _apply_cesa_country_clusters_to_buses(
    buses: pd.DataFrame,
    country_clusters: pd.DataFrame,
) -> pd.DataFrame:
    out = buses.copy()
    out["country_original_raw"] = out["country"].astype(str).str.strip().str.upper()
    out["country"] = out["country_original_raw"]

    if country_clusters.empty:
        out["country_group_label"] = out["country"]
        return out

    source_to_target = (
        country_clusters.drop_duplicates(subset=["source_country"])
        .set_index("source_country")["target_country"]
        .astype(str)
    )
    target_to_label = (
        country_clusters.drop_duplicates(subset=["target_country"])
        .set_index("target_country")["target_label"]
        .astype(str)
    )

    mask = out["sync_area"].astype(str).eq("CESA")
    out.loc[mask, "country"] = _collapse_country_codes(out.loc[mask, "country"], source_to_target).values
    out["country_group_label"] = out["country"].map(target_to_label).fillna(out["country"])
    return out


# ============================================================
# Plant -> bus assignment
# ============================================================
def _country_to_alpha2(name: str) -> str | None:
    try:
        return pycountry.countries.lookup(name).alpha_2
    except Exception:
        return None


def assign_plants_to_buses(
    plants: pd.DataFrame,
    buses: pd.DataFrame,
    *,
    bus_lat_col: str = "y",
    bus_lon_col: str = "x",
    plant_lat_col: str = "lat",
    plant_lon_col: str = "lon",
    plant_country_col: str = "Country",
    bus_country_col: str = "country",
    bus_dc_col: str = "dc",
) -> pd.DataFrame:
    buses = buses.copy()
    buses["lat"] = pd.to_numeric(buses[bus_lat_col], errors="coerce")
    buses["lon"] = pd.to_numeric(buses[bus_lon_col], errors="coerce")
    buses["dc_bool"] = buses[bus_dc_col].astype(str).str.lower().isin(["t", "true", "1", "yes", "y"])
    buses = buses.dropna(subset=["lat", "lon"])

    # AC buses only as candidates
    buses = buses.loc[~buses["dc_bool"]].copy()

    uniq = plants[plant_country_col].astype(str).unique()
    c_map = {c: _country_to_alpha2(c) for c in uniq}
    c_map["Kosovo"] = "XK"

    plants = plants.copy().reset_index(drop=True)
    plants["country_code"] = plants[plant_country_col].map(c_map)
    plants["lat"] = pd.to_numeric(plants[plant_lat_col], errors="coerce")
    plants["lon"] = pd.to_numeric(plants[plant_lon_col], errors="coerce")

    groups: dict[str, tuple[BallTree, np.ndarray]] = {}
    for code, g in buses.groupby(buses[bus_country_col].astype(str)):
        coords = np.deg2rad(np.c_[g["lat"].values, g["lon"].values])
        groups[str(code)] = (
            BallTree(coords, metric="haversine"),
            g["bus_id"].astype(str).values,
        )

    coords_all = np.deg2rad(np.c_[buses["lat"].values, buses["lon"].values])
    tree_all = BallTree(coords_all, metric="haversine")
    ids_all = buses["bus_id"].astype(str).values

    valid = plants["lat"].notna() & plants["lon"].notna()
    assigned = np.empty(len(plants), dtype=object)
    dist_km = np.full(len(plants), np.nan)

    idx_by_cc: dict[str | None, list[int]] = defaultdict(list)
    for i in np.where(valid.values)[0]:
        idx_by_cc[plants.at[i, "country_code"]].append(int(i))

    for cc, idxs in idx_by_cc.items():
        coords = np.deg2rad(np.c_[plants.loc[idxs, "lat"].values, plants.loc[idxs, "lon"].values])
        if cc in groups:
            tree, bus_ids = groups[str(cc)]
            d, ind = tree.query(coords, k=1)
            assigned[idxs] = bus_ids[ind[:, 0]]
            dist_km[idxs] = d[:, 0] * EARTH_R_KM
        else:
            d, ind = tree_all.query(coords, k=1)
            assigned[idxs] = ids_all[ind[:, 0]]
            dist_km[idxs] = d[:, 0] * EARTH_R_KM

    plants["assigned_bus"] = assigned
    plants["dist_km"] = dist_km
    return plants


def _drop_existing_plant_bus_metadata(plants: pd.DataFrame) -> pd.DataFrame:
    return plants.drop(columns=["sync_area", "sync_node", "sync_area_bus", "sync_node_bus"], errors="ignore")


def _drop_edges_touching_buses(edges: pd.DataFrame, removed_bus_ids: set[str]) -> pd.DataFrame:
    if edges.empty or not removed_bus_ids or not {"bus0", "bus1"}.issubset(edges.columns):
        return edges.copy()
    out = edges.copy()
    keep = (
        ~out["bus0"].astype(str).isin(removed_bus_ids)
        & ~out["bus1"].astype(str).isin(removed_bus_ids)
    )
    return out.loc[keep].reset_index(drop=True)


def _filter_plants_by_excluded_countries(plants: pd.DataFrame, excluded_countries: set[str]) -> pd.DataFrame:
    if plants.empty or not excluded_countries:
        return plants.copy()
    country_col = next((col for col in ("country_code", "Country", "country", "country_iso") if col in plants.columns), None)
    if country_col is None:
        return plants.copy()
    out = plants.copy()
    normalized = out[country_col].map(_normalize_country_token)
    return out.loc[~normalized.isin(excluded_countries)].reset_index(drop=True)


def apply_country_exclusions_to_raw_network(
    *,
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    links: pd.DataFrame,
    converters: pd.DataFrame,
    transformers: pd.DataFrame,
    plants: pd.DataFrame,
    excluded_country_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    excluded_countries = set(
        excluded_country_rows.get("source_country", pd.Series(dtype="object")).astype(str)
    )
    if not excluded_countries:
        return buses, lines, links, converters, transformers, plants, {
            "excluded_countries": [],
            "removed_buses": 0,
            "removed_lines": 0,
            "removed_links": 0,
            "removed_converters": 0,
            "removed_transformers": 0,
            "removed_plants": 0,
        }

    if "country" not in buses.columns or "bus_id" not in buses.columns:
        raise ValueError("Country exclusion requires buses.csv columns 'bus_id' and 'country'.")

    bus_country = buses["country"].map(_normalize_country_token)
    removed_bus_ids = set(buses.loc[bus_country.isin(excluded_countries), "bus_id"].astype(str))
    buses_out = buses.loc[~buses["bus_id"].astype(str).isin(removed_bus_ids)].reset_index(drop=True)

    lines_out = _drop_edges_touching_buses(lines, removed_bus_ids)
    links_out = _drop_edges_touching_buses(links, removed_bus_ids)
    converters_out = _drop_edges_touching_buses(converters, removed_bus_ids)
    transformers_out = _drop_edges_touching_buses(transformers, removed_bus_ids)
    plants_out = _filter_plants_by_excluded_countries(plants, excluded_countries)

    summary = {
        "excluded_countries": sorted(excluded_countries),
        "removed_buses": int(len(buses) - len(buses_out)),
        "removed_lines": int(len(lines) - len(lines_out)),
        "removed_links": int(len(links) - len(links_out)),
        "removed_converters": int(len(converters) - len(converters_out)),
        "removed_transformers": int(len(transformers) - len(transformers_out)),
        "removed_plants": int(len(plants) - len(plants_out)),
    }
    return buses_out, lines_out, links_out, converters_out, transformers_out, plants_out, summary


# ============================================================
# Preprocessing
# ============================================================
def _prep_buses(buses: pd.DataFrame) -> pd.DataFrame:
    b = buses.copy()
    b["bus_id"] = b["bus_id"].astype(str)

    if "y" in b.columns:
        b["lat"] = pd.to_numeric(b["y"], errors="coerce")
    else:
        b["lat"] = pd.to_numeric(b["lat"], errors="coerce")

    if "x" in b.columns:
        b["lon"] = pd.to_numeric(b["x"], errors="coerce")
    else:
        b["lon"] = pd.to_numeric(b["lon"], errors="coerce")

    b["voltage"] = pd.to_numeric(b["voltage"], errors="coerce").astype("Int64")
    b["dc_bool"] = b["dc"].astype(str).str.lower().isin(["t", "true", "1", "yes", "y"])
    b = b.dropna(subset=["lat", "lon", "voltage"])
    b["voltage"] = b["voltage"].astype(int)
    b["country"] = b["country"].astype(str).str.upper()
    return b


def _prep_lines(lines: pd.DataFrame) -> pd.DataFrame:
    L = lines.copy()
    if "line_id" not in L.columns:
        L["line_id"] = [f"line_{i}" for i in range(len(L))]
    L["line_id"] = L["line_id"].astype(str)
    L["bus0"] = L["bus0"].astype(str)
    L["bus1"] = L["bus1"].astype(str)

    for c in ["r", "x", "b", "s_nom", "circuits", "length", "voltage"]:
        if c in L.columns:
            L[c] = pd.to_numeric(L[c], errors="coerce")

    if "circuits" not in L.columns:
        L["circuits"] = 1.0
    L["circuits"] = L["circuits"].fillna(1.0)
    return L


def _prep_transformers(transformers: pd.DataFrame) -> pd.DataFrame:
    T = transformers.copy()
    if "transformer_id" not in T.columns:
        T["transformer_id"] = [f"trafo_{i}" for i in range(len(T))]
    T["transformer_id"] = T["transformer_id"].astype(str)
    T["bus0"] = T["bus0"].astype(str)
    T["bus1"] = T["bus1"].astype(str)

    for c in ["voltage_bus0", "voltage_bus1", "s_nom"]:
        if c in T.columns:
            T[c] = pd.to_numeric(T[c], errors="coerce")
    return T


def _prep_edge_table(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    D = df.copy()
    if id_col not in D.columns:
        D[id_col] = [f"{id_col}_{i}" for i in range(len(D))]
    D[id_col] = D[id_col].astype(str)
    D["bus0"] = D["bus0"].astype(str)
    D["bus1"] = D["bus1"].astype(str)
    return D


# ============================================================
# Synchronous areas
# ============================================================
def add_sync_area_to_buses(
    buses0: pd.DataFrame,
    *,
    country_col: str = "country",
    lon_col: str = "lon",
    lat_col: str = "lat",
    override_bus_to_area: dict[str, str] | None = None,
) -> pd.DataFrame:
    b = buses0.copy()
    b["bus_id"] = b["bus_id"].astype(str)

    c = b[country_col].astype(str).str.upper()

    dk_mask = c.eq("DK")
    if dk_mask.any():
        lon = pd.to_numeric(b[lon_col], errors="coerce")
        c = c.where(~dk_mask, np.where(lon >= 10.95, "DK2", "DK1"))

    is_gb = c.eq("GB") | c.eq("UK")
    lon = pd.to_numeric(b[lon_col], errors="coerce")
    lat = pd.to_numeric(b[lat_col], errors="coerce")
    ni_box = is_gb & lon.between(-8.5, -5.0) & lat.between(54.0, 55.6)
    c = c.where(~ni_box, "NIR")

    area = pd.Series("CESA", index=b.index, dtype="object")
    area.loc[c.isin(["GB", "UK"])] = "GB"
    area.loc[c.isin(["IE", "NIR"])] = "IE_NOIE"
    area.loc[c.isin(["NO", "SE", "FI", "DK2"])] = "NORDICS"

    if override_bus_to_area:
        for bus_id, a in override_bus_to_area.items():
            m = b["bus_id"].eq(str(bus_id))
            area.loc[m] = a

    b["sync_area"] = area

    b["sync_node"] = pd.Series(pd.NA, index=b.index, dtype="object")
    b.loc[b["sync_area"].eq("GB"), "sync_node"] = "SYNC_GB"
    b.loc[b["sync_area"].eq("IE_NOIE"), "sync_node"] = "SYNC_IE"
    b.loc[b["sync_area"].eq("NORDICS"), "sync_node"] = "SYNC_NORD"

    return b


def _sub_network_prefix(sync_area: str) -> str:
    area = str(sync_area).strip().upper()
    if area == "CESA":
        return "sn"

    safe = re.sub(r"[^0-9A-Za-z]+", "_", area).strip("_").lower()
    return f"{safe or 'sync'}_sn"


def _fill_missing_ac_sub_networks(
    buses: pd.DataFrame,
    lines: pd.DataFrame,
) -> pd.DataFrame:
    out = buses.copy()
    if "sub_network" not in out.columns:
        out["sub_network"] = pd.Series(pd.NA, index=out.index, dtype="object")

    missing_mask = (~out["dc_bool"]) & out["sub_network"].isna()
    if not missing_mask.any():
        return out

    L = lines.copy()
    if not L.empty:
        L["bus0"] = L["bus0"].astype(str)
        L["bus1"] = L["bus1"].astype(str)

    sync_areas = (
        out.loc[missing_mask, "sync_area"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    for sync_area in sync_areas:
        area_mask = (
            out["sync_area"].astype(str).eq(sync_area)
            & (~out["dc_bool"])
            & out["sub_network"].isna()
        )
        bus_ids = out.loc[area_mask, "bus_id"].astype(str).to_numpy()
        if len(bus_ids) == 0:
            continue

        idx = {bus_id: i for i, bus_id in enumerate(bus_ids)}
        rows: list[int] = []
        cols: list[int] = []

        if not L.empty:
            area_lines = L.loc[L["bus0"].isin(bus_ids) & L["bus1"].isin(bus_ids), ["bus0", "bus1"]]
            for u, v in area_lines.itertuples(index=False, name=None):
                if u in idx and v in idx and u != v:
                    i, j = idx[str(u)], idx[str(v)]
                    rows.extend([i, j])
                    cols.extend([j, i])

        adjacency = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(len(bus_ids), len(bus_ids)),
        )
        _, labels = connected_components(adjacency, directed=False, return_labels=True)
        prefix = _sub_network_prefix(sync_area)
        sn_map = {bus_ids[i]: f"{prefix}_{labels[i]}" for i in range(len(bus_ids))}

        out.loc[area_mask, "sub_network"] = (
            out.loc[area_mask, "bus_id"].astype(str).map(sn_map)
        )

    still_missing = (~out["dc_bool"]) & out["sub_network"].isna()
    if still_missing.any():
        fallback_prefix = out.loc[still_missing, "sync_area"].astype(str).map(_sub_network_prefix)
        out.loc[still_missing, "sub_network"] = (
            fallback_prefix + "_single_" + out.loc[still_missing, "bus_id"].astype(str)
        )

    return out


# ============================================================
# Helpers
# ============================================================
def _as_group_tuple(key: Any) -> tuple[Any, ...]:
    return key if isinstance(key, tuple) else (key,)


def _group_key_to_str_tuple(key: Any) -> tuple[str, ...]:
    return tuple(str(x) for x in _as_group_tuple(key))


def _group_key_to_label(key: Any) -> str:
    vals = _group_key_to_str_tuple(key)
    out = []
    for v in vals:
        v = v.replace(" ", "")
        v = v.replace("/", "-")
        v = v.replace("\\", "-")
        out.append(v)
    return "_".join(out)


def _group_index_from_frame(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(df.loc[:, list(group_cols)].astype(str))


def _grouping_mode_name(group_cols: Sequence[str]) -> str:
    return "x".join(group_cols)


def _resolve_mapping_once(mapping: pd.Series) -> pd.Series:
    out = mapping.copy()
    while True:
        new = out.map(out).fillna(out)
        if new.equals(out):
            return out
        out = new


def _deduplicate_mapping_index(mapping: pd.Series, *, name: str) -> pd.Series:
    out = mapping.copy()
    out.index = pd.Index(out.index.astype(str), dtype="object")

    if not out.index.has_duplicates:
        return out

    def _collapse(group: pd.Series) -> Any:
        vals = pd.Index(group.dropna().astype(str))
        uniq = pd.unique(vals)
        if len(uniq) == 0:
            return pd.NA
        if len(uniq) > 1:
            warnings.warn(
                f"{name} has duplicate index label '{group.name}' with conflicting mappings {list(uniq)}; keeping the first value."
            )
        return str(uniq[0])

    return out.groupby(level=0, sort=False).agg(_collapse)


def _haversine_array(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0088
    lat1r = np.deg2rad(np.asarray(lat1, dtype=float))
    lon1r = np.deg2rad(np.asarray(lon1, dtype=float))
    lat2r = np.deg2rad(np.asarray(lat2, dtype=float))
    lon2r = np.deg2rad(np.asarray(lon2, dtype=float))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def _allocate_integer_budget(
    sizes: np.ndarray,
    k_total: int,
    *,
    weights: np.ndarray | None = None,
    min_per_nonempty: int = 1,
) -> np.ndarray:
    sizes = np.asarray(sizes, dtype=int)
    if np.any(sizes < 0):
        raise ValueError("sizes must be non-negative")
    if k_total <= 0:
        raise ValueError("k_total must be positive")

    nonempty = sizes > 0
    out = np.zeros_like(sizes)
    out[nonempty] = min_per_nonempty
    out = np.minimum(out, sizes)

    min_needed = int(out.sum())
    if k_total < min_needed:
        warnings.warn(
            f"k_total={k_total} is smaller than the minimum feasible allocation ({min_needed}); returning minimum."
        )
        return out

    remaining = int(k_total - out.sum())
    if remaining == 0:
        return out

    if weights is None:
        base = sizes.astype(float)
    else:
        base = np.asarray(weights, dtype=float)
        if base.shape != sizes.shape:
            raise ValueError("weights must have same shape as sizes")
        base = np.where(np.isfinite(base) & (base > 0), base, 0.0)

    if base.sum() <= 0:
        base = sizes.astype(float)
    if base.sum() <= 0:
        return out

    cap_left = sizes - out
    ideal = remaining * base / base.sum()

    add = np.floor(ideal).astype(int)
    add = np.minimum(add, cap_left)
    out += add

    remaining = int(k_total - out.sum())
    if remaining == 0:
        return out

    frac = ideal - np.floor(ideal)
    score = frac + 1e-12 * sizes.astype(float)

    order = np.argsort(-score)
    for idx in order:
        if remaining == 0:
            break
        if out[idx] < sizes[idx]:
            out[idx] += 1
            remaining -= 1

    if remaining > 0:
        order = np.argsort(-(sizes - out))
        for idx in order:
            if remaining == 0:
                break
            while out[idx] < sizes[idx] and remaining > 0:
                out[idx] += 1
                remaining -= 1

    return out


def _allocate_k_across_components(comp_sizes: np.ndarray, k_total: int) -> np.ndarray:
    return _allocate_integer_budget(
        np.asarray(comp_sizes, dtype=int),
        k_total=k_total,
        weights=np.asarray(comp_sizes, dtype=float),
        min_per_nonempty=1,
    )


def _unique_sorted_join(s: pd.Series, sep: str = ",") -> str:
    vals = sorted(pd.unique(s.dropna().astype(str)))
    return sep.join(vals) if len(vals) else ""


def _infer_line_voltage_base_kv(
    lines: pd.DataFrame,
    buses: pd.DataFrame,
) -> pd.Series:
    """
    Determine the voltage base [kV] for each AC line.

    Priority:
      1) explicit line["voltage"], if available and > 0
      2) max(bus0.voltage, bus1.voltage)

    Returns:
      pd.Series indexed like lines, with v_base_kv
    """
    L = lines.copy()
    L["bus0"] = L["bus0"].astype(str)
    L["bus1"] = L["bus1"].astype(str)

    B = buses[["bus_id", "voltage"]].copy()
    B["bus_id"] = B["bus_id"].astype(str)
    B["voltage"] = pd.to_numeric(B["voltage"], errors="coerce")
    bus_v = B.set_index("bus_id")["voltage"]

    if "voltage" in L.columns:
        v_line = pd.to_numeric(L["voltage"], errors="coerce")
    else:
        v_line = pd.Series(np.nan, index=L.index, dtype=float)

    v0 = pd.to_numeric(L["bus0"].map(bus_v), errors="coerce")
    v1 = pd.to_numeric(L["bus1"].map(bus_v), errors="coerce")
    v_bus = pd.Series(np.fmax(v0.values, v1.values), index=L.index, dtype=float)

    v_base = np.where(
        np.isfinite(v_line.values) & (v_line.values > 0),
        v_line.values,
        v_bus.values,
    )

    return pd.Series(v_base, index=L.index, name="v_base_kv")


def _convert_lines_physical_to_internal_pu(
    lines: pd.DataFrame,
    buses: pd.DataFrame,
    *,
    s_base_mva: float = 100.0,
) -> pd.DataFrame:
    """
    Build an internal per-unit copy from physical AC line parameters.

    Assumptions:
      - r, x in Ohm
      - b in Siemens
      - s_nom in MVA
      - voltage in kV

    Formulas:
      Z_base [Ohm] = V_base[kV]^2 / S_base[MVA]
      Y_base [S]   = 1 / Z_base

      r_pu = r / Z_base
      x_pu = x / Z_base
      b_pu = b / Y_base = b * Z_base
      s_nom_pu = s_nom / S_base
    """
    if s_base_mva <= 0:
        raise ValueError("s_base_mva must be positive")

    L = _prep_lines(lines).copy()
    v_base = _infer_line_voltage_base_kv(L, buses)

    z_base = (v_base.astype(float) ** 2) / float(s_base_mva)   # Ohm
    y_base = 1.0 / z_base                                      # Siemens

    out = L.copy()

    # Preserve original values
    for c in ["r", "x", "b", "s_nom", "voltage"]:
        if c in out.columns and f"{c}_phys" not in out.columns:
            out[f"{c}_phys"] = out[c]

    out["v_base_kv_internal"] = v_base.astype(float)
    out["s_base_mva_internal"] = float(s_base_mva)
    out["internal_units"] = f"pu_on_{float(s_base_mva):g}MVA"

    if "r" in out.columns:
        out["r"] = pd.to_numeric(out["r"], errors="coerce") / z_base
    if "x" in out.columns:
        out["x"] = pd.to_numeric(out["x"], errors="coerce") / z_base
    if "b" in out.columns:
        out["b"] = pd.to_numeric(out["b"], errors="coerce") / y_base
    if "s_nom" in out.columns:
        out["s_nom"] = pd.to_numeric(out["s_nom"], errors="coerce") / float(s_base_mva)

    return out


def _prepare_internal_line_table(
    lines_physical: pd.DataFrame,
    buses_work: pd.DataFrame,
    *,
    voltage_mode: str,
    s_base_mva: float,
) -> pd.DataFrame:
    """
    Return the AC line table used for internal electrical calculations.

    - standard       -> physical values
    - simplify_380   -> physical values of the 380-kV equivalent network
    - per_unit       -> per-unit copy on a common base
    """
    if voltage_mode == "per_unit":
        return _convert_lines_physical_to_internal_pu(
            lines_physical,
            buses_work,
            s_base_mva=s_base_mva,
        )
    return lines_physical.copy()


# ============================================================
# TYNDP 2020 transmission expansion integration
# ============================================================
def _ensure_grid_source(df: pd.DataFrame, *, default_source: str) -> pd.DataFrame:
    out = df.copy()
    if "grid_source" not in out.columns:
        out["grid_source"] = default_source
    else:
        mask = out["grid_source"].isna() | out["grid_source"].astype(str).str.strip().eq("")
        out.loc[mask, "grid_source"] = default_source
    return out


def _resolve_tyndp_dataset_path(base_data: Path, dataset_path: Path) -> Path:
    primary = base_data / dataset_path
    if primary.exists():
        return primary
    flat = base_data / dataset_path.name
    if flat.exists():
        return flat
    return primary


def _sanitize_tyndp_id(value: Any, *, fallback: str) -> str:
    txt = str(value).strip()
    if not txt or txt.lower() in {"nan", "<na>"}:
        txt = fallback
    txt = re.sub(r"[^A-Za-z0-9._-]+", "_", txt)
    txt = txt.strip("_")
    return txt if txt else fallback


def _bool_to_tf(value: Any) -> str:
    if pd.isna(value):
        return "f"
    return "t" if bool(value) else "f"


def _source_label_from_variant(project_variant: str) -> str:
    return GRID_SOURCE_TYNDP_NEW if str(project_variant).strip().lower() == "new" else GRID_SOURCE_TYNDP_UPGRADE


def _build_tyndp_linestring(x0: Any, y0: Any, x1: Any, y1: Any) -> Any:
    vals = pd.to_numeric(pd.Series([x0, y0, x1, y1]), errors="coerce")
    if vals.isna().any():
        return pd.NA
    return f"LINESTRING ({vals.iloc[0]} {vals.iloc[1]}, {vals.iloc[2]} {vals.iloc[3]})"


def _maybe_build_bus_snapper(buses: pd.DataFrame) -> tyndp2020.BusSnapper | None:
    if buses.empty:
        return None
    return tyndp2020.build_bus_snapper(buses)


def _resnap_tyndp_line_projects_to_matching_voltage(
    line_diag: pd.DataFrame,
    buses_prepped: pd.DataFrame,
) -> pd.DataFrame:
    out = line_diag.copy()

    for src, dst in [
        ("snap_ac_bus0", "build_bus0"),
        ("snap_ac_bus1", "build_bus1"),
        ("snap_ac_voltage0_kv", "build_voltage0_kv"),
        ("snap_ac_voltage1_kv", "build_voltage1_kv"),
        ("snap_ac_distance0_km", "build_distance0_km"),
        ("snap_ac_distance1_km", "build_distance1_km"),
        ("snap_ac_lat0", "build_lat0"),
        ("snap_ac_lat1", "build_lat1"),
        ("snap_ac_lon0", "build_lon0"),
        ("snap_ac_lon1", "build_lon1"),
    ]:
        out[dst] = out[src]

    out["build_resnapped_to_voltage"] = False

    ac_buses = buses_prepped.loc[~buses_prepped["dc_bool"]].copy()
    if ac_buses.empty:
        return out

    ac_buses["voltage_class"] = ac_buses["voltage"].map(tyndp2020._normalize_voltage_class)
    snap_v0 = out["snap_ac_voltage0_kv"].map(tyndp2020._normalize_voltage_class)
    snap_v1 = out["snap_ac_voltage1_kv"].map(tyndp2020._normalize_voltage_class)

    include_mask = out["recommended_action"].isin(["include_new_line", "include_as_upgrade"])
    mismatch_mask = include_mask & (
        snap_v0.ne(out["project_voltage_class"]) | snap_v1.ne(out["project_voltage_class"])
    )

    for voltage_class in pd.unique(out.loc[mismatch_mask, "project_voltage_class"].dropna()):
        buses_v = ac_buses.loc[ac_buses["voltage_class"].eq(voltage_class)].copy()
        snapper_v = _maybe_build_bus_snapper(buses_v)
        if snapper_v is None:
            continue

        mask_v = mismatch_mask & out["project_voltage_class"].eq(voltage_class)
        if not mask_v.any():
            continue

        resnapped = tyndp2020.snap_project_endpoints(
            out.loc[mask_v, ["x0", "y0", "x1", "y1"]].copy(),
            snapper_v,
            prefix="snap_vclass",
        )

        out.loc[mask_v, "build_bus0"] = resnapped["snap_vclass_bus0"].values
        out.loc[mask_v, "build_bus1"] = resnapped["snap_vclass_bus1"].values
        out.loc[mask_v, "build_voltage0_kv"] = resnapped["snap_vclass_voltage0_kv"].values
        out.loc[mask_v, "build_voltage1_kv"] = resnapped["snap_vclass_voltage1_kv"].values
        out.loc[mask_v, "build_distance0_km"] = resnapped["snap_vclass_distance0_km"].values
        out.loc[mask_v, "build_distance1_km"] = resnapped["snap_vclass_distance1_km"].values
        out.loc[mask_v, "build_lat0"] = resnapped["snap_vclass_lat0"].values
        out.loc[mask_v, "build_lat1"] = resnapped["snap_vclass_lat1"].values
        out.loc[mask_v, "build_lon0"] = resnapped["snap_vclass_lon0"].values
        out.loc[mask_v, "build_lon1"] = resnapped["snap_vclass_lon1"].values
        out.loc[mask_v, "build_resnapped_to_voltage"] = True

    return out


def _build_tyndp_line_assets(
    line_diag: pd.DataFrame,
    buses_prepped: pd.DataFrame,
) -> pd.DataFrame:
    if line_diag.empty:
        return pd.DataFrame()

    work = _resnap_tyndp_line_projects_to_matching_voltage(line_diag, buses_prepped)
    include_mask = work["recommended_action"].isin(["include_new_line", "include_as_upgrade"])
    include_mask &= work["template_available"].fillna(False)

    rows: list[dict[str, Any]] = []
    for i, row in work.loc[include_mask].iterrows():
        bus0 = str(row["build_bus0"]) if pd.notna(row["build_bus0"]) else ""
        bus1 = str(row["build_bus1"]) if pd.notna(row["build_bus1"]) else ""
        if not bus0 or not bus1 or bus0 == bus1:
            continue

        circuits = pd.to_numeric(pd.Series([row["project_num_parallel"]]), errors="coerce").iloc[0]
        if not np.isfinite(circuits) or circuits <= 0:
            circuits = pd.to_numeric(pd.Series([row["template_circuits_mode"]]), errors="coerce").iloc[0]
        circuits = float(max(1.0, round(float(circuits)))) if np.isfinite(circuits) else 1.0

        length_m = pd.to_numeric(pd.Series([row["length_m"]]), errors="coerce").iloc[0]
        if not np.isfinite(length_m) or length_m <= 0:
            lat0 = pd.to_numeric(pd.Series([row["build_lat0"]]), errors="coerce").iloc[0]
            lon0 = pd.to_numeric(pd.Series([row["build_lon0"]]), errors="coerce").iloc[0]
            lat1 = pd.to_numeric(pd.Series([row["build_lat1"]]), errors="coerce").iloc[0]
            lon1 = pd.to_numeric(pd.Series([row["build_lon1"]]), errors="coerce").iloc[0]
            if np.isfinite(lat0) and np.isfinite(lon0) and np.isfinite(lat1) and np.isfinite(lon1):
                length_m = float(_haversine_array([lat0], [lon0], [lat1], [lon1])[0] * 1000.0)

        voltage = pd.to_numeric(pd.Series([row["project_voltage_kv"]]), errors="coerce").iloc[0]
        i_nom = pd.to_numeric(pd.Series([row["template_i_nom_mode_ka"]]), errors="coerce").iloc[0]
        s_nom_per_circuit = pd.to_numeric(
            pd.Series([row["template_s_nom_per_circuit_mva_median"]]),
            errors="coerce",
        ).iloc[0]

        if (not np.isfinite(s_nom_per_circuit)) and np.isfinite(i_nom) and np.isfinite(voltage):
            s_nom_per_circuit = np.sqrt(3.0) * float(voltage) * float(i_nom)

        r_spec = pd.to_numeric(pd.Series([row["template_r_per_km_median"]]), errors="coerce").iloc[0]
        x_spec = pd.to_numeric(pd.Series([row["template_x_per_km_median"]]), errors="coerce").iloc[0]
        b_spec = pd.to_numeric(pd.Series([row["template_b_per_km_median"]]), errors="coerce").iloc[0]

        length_for_impedance = length_m if np.isfinite(length_m) and length_m > 0 else np.nan
        r_total = float(r_spec * length_for_impedance) if np.isfinite(r_spec) and np.isfinite(length_for_impedance) else np.nan
        x_total = float(x_spec * length_for_impedance) if np.isfinite(x_spec) and np.isfinite(length_for_impedance) else np.nan
        b_total = float(b_spec * length_for_impedance) if np.isfinite(b_spec) and np.isfinite(length_for_impedance) else np.nan

        source_label = _source_label_from_variant(str(row["project_variant"]))
        project_id_safe = _sanitize_tyndp_id(row["project_id"], fallback=f"line_{i}")
        build_year = pd.to_numeric(pd.Series([row["build_year"]]), errors="coerce").astype("Int64").iloc[0]
        tags = (
            f"tyndp_project_id={row['project_id']};"
            f"build_year={build_year if pd.notna(build_year) else ''};"
            f"status={row['project_status']};"
            f"recommended_action={row['recommended_action']};"
            f"detailed_recommendation={row['detailed_recommendation']}"
        )

        rows.append(
            {
                "line_id": f"{source_label}_line_{project_id_safe}",
                "bus0": bus0,
                "bus1": bus1,
                "voltage": float(voltage) if np.isfinite(voltage) else np.nan,
                "i_nom": float(i_nom) if np.isfinite(i_nom) else np.nan,
                "circuits": circuits,
                "s_nom": float(s_nom_per_circuit * circuits) if np.isfinite(s_nom_per_circuit) else np.nan,
                "r": r_total,
                "x": x_total,
                "b": b_total,
                "length": float(length_m) if np.isfinite(length_m) else np.nan,
                "underground": _bool_to_tf(row["underground_bool"]),
                "under_construction": "f",
                "type": str(row["project_line_type"]).strip(),
                "tags": tags,
                "geometry": _build_tyndp_linestring(row["x0"], row["y0"], row["x1"], row["y1"]),
                "grid_source": source_label,
            }
        )

    return pd.DataFrame(rows)


def _build_tyndp_link_assets(link_diag: pd.DataFrame) -> pd.DataFrame:
    if link_diag.empty:
        return pd.DataFrame()

    include_mask = link_diag["recommended_action"].isin(
        ["include_new_link_reuse_dc", "include_new_link_to_nearest_ac", "include_as_upgrade"]
    )

    rows: list[dict[str, Any]] = []
    for i, row in link_diag.loc[include_mask].iterrows():
        use_dc = False
        if row["recommended_action"] == "include_new_link_reuse_dc":
            use_dc = True
        elif row["recommended_action"] == "include_as_upgrade":
            bus0_dc = str(row["snap_dc_bus0"]) if pd.notna(row["snap_dc_bus0"]) else ""
            bus1_dc = str(row["snap_dc_bus1"]) if pd.notna(row["snap_dc_bus1"]) else ""
            use_dc = bool(bus0_dc and bus1_dc and bus0_dc != bus1_dc)

        if use_dc:
            bus0 = str(row["snap_dc_bus0"]) if pd.notna(row["snap_dc_bus0"]) else ""
            bus1 = str(row["snap_dc_bus1"]) if pd.notna(row["snap_dc_bus1"]) else ""
            lat0, lon0 = row["snap_dc_lat0"], row["snap_dc_lon0"]
            lat1, lon1 = row["snap_dc_lat1"], row["snap_dc_lon1"]
            link_model = "tyndp_reuse_dc_terminals"
        else:
            bus0 = str(row["snap_ac_bus0"]) if pd.notna(row["snap_ac_bus0"]) else ""
            bus1 = str(row["snap_ac_bus1"]) if pd.notna(row["snap_ac_bus1"]) else ""
            lat0, lon0 = row["snap_ac_lat0"], row["snap_ac_lon0"]
            lat1, lon1 = row["snap_ac_lat1"], row["snap_ac_lon1"]
            link_model = "tyndp_direct_ac"

        if not bus0 or not bus1 or bus0 == bus1:
            continue

        length_m = pd.to_numeric(pd.Series([row["length_m"]]), errors="coerce").iloc[0]
        if not np.isfinite(length_m) or length_m <= 0:
            lat0 = pd.to_numeric(pd.Series([lat0]), errors="coerce").iloc[0]
            lon0 = pd.to_numeric(pd.Series([lon0]), errors="coerce").iloc[0]
            lat1 = pd.to_numeric(pd.Series([lat1]), errors="coerce").iloc[0]
            lon1 = pd.to_numeric(pd.Series([lon1]), errors="coerce").iloc[0]
            if np.isfinite(lat0) and np.isfinite(lon0) and np.isfinite(lat1) and np.isfinite(lon1):
                length_m = float(_haversine_array([lat0], [lon0], [lat1], [lon1])[0] * 1000.0)

        voltage = pd.to_numeric(pd.Series([row["inferred_voltage_selected_kv"]]), errors="coerce").iloc[0]
        p_nom = pd.to_numeric(pd.Series([row["project_p_nom_mw"]]), errors="coerce").iloc[0]

        source_label = _source_label_from_variant(str(row["project_variant"]))
        project_id_safe = _sanitize_tyndp_id(row["project_id"], fallback=f"link_{i}")
        build_year = pd.to_numeric(pd.Series([row["build_year"]]), errors="coerce").astype("Int64").iloc[0]
        tags = (
            f"tyndp_project_id={row['project_id']};"
            f"build_year={build_year if pd.notna(build_year) else ''};"
            f"status={row['project_status']};"
            f"recommended_action={row['recommended_action']};"
            f"detailed_recommendation={row['detailed_recommendation']};"
            f"voltage_source={row['inferred_voltage_source']}"
        )

        rows.append(
            {
                "link_id": f"{source_label}_link_{project_id_safe}",
                "bus0": bus0,
                "bus1": bus1,
                "voltage": float(voltage) if np.isfinite(voltage) else np.nan,
                "p_nom": float(p_nom) if np.isfinite(p_nom) else np.nan,
                "length": float(length_m) if np.isfinite(length_m) else np.nan,
                "underground": _bool_to_tf(row["underground_bool"]),
                "under_construction": "f",
                "tags": tags,
                "geometry": _build_tyndp_linestring(row["x0"], row["y0"], row["x1"], row["y1"]),
                "grid_source": source_label,
                "carrier": "HVDC",
                "link_model": link_model,
            }
        )

    return pd.DataFrame(rows)


def integrate_tyndp2020_projects(
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    links: pd.DataFrame,
    *,
    base_data: str | Path,
    target_year: int,
    base_snapshot_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_data = Path(base_data)

    buses_prepped = tyndp2020.prep_buses(buses)
    lines_prepped = tyndp2020.prep_lines(lines)
    links_prepped = tyndp2020.prep_links(links)

    ac_buses = buses_prepped.loc[~buses_prepped["dc_bool"]].copy()
    dc_buses = buses_prepped.loc[buses_prepped["dc_bool"]].copy()

    ac_snapper = tyndp2020.build_bus_snapper(ac_buses)
    dc_snapper = _maybe_build_bus_snapper(dc_buses)

    line_lookup = tyndp2020.build_line_lookup(lines_prepped)
    line_templates = tyndp2020.build_line_parameter_templates(lines_prepped)
    link_pair_lookup, link_country_pair_lookup, global_link_voltage_mode = tyndp2020.build_link_lookup(links_prepped, buses_prepped)

    new_lines = tyndp2020.read_tyndp_line_projects(
        _resolve_tyndp_dataset_path(base_data, tyndp2020.TYNDP_DATASETS["new_lines"]),
        dataset="new_lines",
    )
    upgraded_lines = tyndp2020.read_tyndp_line_projects(
        _resolve_tyndp_dataset_path(base_data, tyndp2020.TYNDP_DATASETS["upgraded_lines"]),
        dataset="upgraded_lines",
    )
    new_links = tyndp2020.read_tyndp_link_projects(
        _resolve_tyndp_dataset_path(base_data, tyndp2020.TYNDP_DATASETS["new_links"]),
        dataset="new_links",
    )
    upgraded_links = tyndp2020.read_tyndp_link_projects(
        _resolve_tyndp_dataset_path(base_data, tyndp2020.TYNDP_DATASETS["upgraded_links"]),
        dataset="upgraded_links",
    )

    diag_lines = pd.concat(
        [
            tyndp2020.diagnose_line_projects(
                new_lines,
                ac_snapper=ac_snapper,
                line_lookup=line_lookup,
                line_templates=line_templates,
                target_year=target_year,
                base_snapshot_year=base_snapshot_year,
            ),
            tyndp2020.diagnose_line_projects(
                upgraded_lines,
                ac_snapper=ac_snapper,
                line_lookup=line_lookup,
                line_templates=line_templates,
                target_year=target_year,
                base_snapshot_year=base_snapshot_year,
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    diag_links = pd.concat(
        [
            tyndp2020.diagnose_link_projects(
                new_links,
                dc_snapper=dc_snapper if dc_snapper is not None else ac_snapper,
                ac_snapper=ac_snapper,
                link_pair_lookup=link_pair_lookup,
                link_country_pair_lookup=link_country_pair_lookup,
                global_link_voltage_mode=global_link_voltage_mode,
                target_year=target_year,
                base_snapshot_year=base_snapshot_year,
            ),
            tyndp2020.diagnose_link_projects(
                upgraded_links,
                dc_snapper=dc_snapper if dc_snapper is not None else ac_snapper,
                ac_snapper=ac_snapper,
                link_pair_lookup=link_pair_lookup,
                link_country_pair_lookup=link_country_pair_lookup,
                global_link_voltage_mode=global_link_voltage_mode,
                target_year=target_year,
                base_snapshot_year=base_snapshot_year,
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    lines_added = _build_tyndp_line_assets(diag_lines, buses_prepped)
    links_added = _build_tyndp_link_assets(diag_links)

    lines_base = _ensure_grid_source(lines, default_source=GRID_SOURCE_BASE)
    links_base = _ensure_grid_source(links, default_source=GRID_SOURCE_BASE)

    if not lines_added.empty:
        lines_all = pd.concat([lines_base, lines_added], ignore_index=True, sort=False)
    else:
        lines_all = lines_base

    if not links_added.empty:
        links_all = pd.concat([links_base, links_added], ignore_index=True, sort=False)
    else:
        links_all = links_base

    return lines_all.reset_index(drop=True), links_all.reset_index(drop=True)


# ============================================================
# 380-kV equivalent simplification
# ============================================================
def simplify_network_to_380_equivalent(
    buses0: pd.DataFrame,
    lines: pd.DataFrame,
    links: pd.DataFrame,
    converters: pd.DataFrame,
    transformers: pd.DataFrame,
    plants_assigned: pd.DataFrame,
) -> dict[str, Any]:
    """
    CSV-native approximation of PyPSA-Eur's 380-kV simplification.

    For AC buses within the same synchronous area:
      - lower-voltage buses are mapped via domestic transformers to representative higher-voltage buses
      - internal AC lines of a synchronous area are mapped onto a single 380-kV layer
      - transformers that collapse to self-maps disappear
    """
    B = buses0.copy()
    L = _prep_lines(lines)
    T = _prep_transformers(transformers)
    LK = _prep_edge_table(links, "link_id")
    CV = _prep_edge_table(converters, "converter_id")
    P = plants_assigned.copy()
    P["assigned_bus"] = P["assigned_bus"].astype(str)

    bus_country = B.set_index("bus_id")["country"]
    bus_dc = B.set_index("bus_id")["dc_bool"]
    bus_sync = B.set_index("bus_id")["sync_area"]
    bus_voltage = B.set_index("bus_id")["voltage"]

    T["country0"] = T["bus0"].map(bus_country)
    T["country1"] = T["bus1"].map(bus_country)
    T["dc0"] = T["bus0"].map(bus_dc)
    T["dc1"] = T["bus1"].map(bus_dc)
    T["sync0"] = T["bus0"].map(bus_sync)
    T["sync1"] = T["bus1"].map(bus_sync)
    T["v0"] = T["bus0"].map(bus_voltage)
    T["v1"] = T["bus1"].map(bus_voltage)

    # Only domestic AC transformers inside the same synchronous area participate
    # in the single-layer simplification.
    tsel = T[
        T["country0"].notna()
        & T["country0"].eq(T["country1"])
        & T["sync0"].notna()
        & T["sync0"].eq(T["sync1"])
        & T["dc0"].eq(False)
        & T["dc1"].eq(False)
    ].copy()

    if not tsel.empty:
        drop_bus = np.where(tsel["v0"].fillna(-1) <= tsel["v1"].fillna(-1), tsel["bus0"], tsel["bus1"])
        keep_bus = np.where(tsel["v0"].fillna(-1) <= tsel["v1"].fillna(-1), tsel["bus1"], tsel["bus0"])
        trafo_map = pd.Series(keep_bus, index=pd.Index(drop_bus, dtype="object")).astype(str)
        trafo_map = trafo_map[~trafo_map.index.duplicated(keep="first")]
        trafo_map = _resolve_mapping_once(trafo_map)
    else:
        trafo_map = pd.Series(dtype=object)

    bus_ids_380 = B["bus_id"].astype(str).to_numpy(copy=True)
    busmap_380 = pd.Series(
        bus_ids_380.copy(),
        index=pd.Index(bus_ids_380.copy(), dtype="object"),
        dtype="object",
    )
    intersect = busmap_380.index.intersection(trafo_map.index)
    busmap_380.loc[intersect] = trafo_map.loc[intersect]
    busmap_380 = _resolve_mapping_once(busmap_380)

    B["bus_id_380"] = B["bus_id"].map(busmap_380).fillna(B["bus_id"])

    def mode(s: pd.Series):
        s = s.dropna()
        return s.value_counts().idxmax() if len(s) else np.nan

    buses_380 = (
        B.groupby("bus_id_380", as_index=False)
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            country=("country", mode),
            sync_area=("sync_area", mode),
            dc_bool=("dc_bool", "first"),
            voltage_max_orig=("voltage", "max"),
            voltage_min_orig=("voltage", "min"),
            n_buses_merged=("bus_id", "count"),
        )
        .rename(columns={"bus_id_380": "bus_id"})
    )

    buses_380["voltage"] = np.where(
        ~buses_380["dc_bool"],
        380,
        buses_380["voltage_max_orig"],
    ).astype(int)
    buses_380["dc"] = np.where(buses_380["dc_bool"], "t", "f")

    buses_380 = add_sync_area_to_buses(buses_380)
    buses_380_idx = buses_380.set_index("bus_id")

    # -----------------------------
    # Lines
    # -----------------------------
    L["bus0"] = L["bus0"].map(busmap_380).fillna(L["bus0"])
    L["bus1"] = L["bus1"].map(busmap_380).fillna(L["bus1"])
    L = L.dropna(subset=["bus0", "bus1"]).copy()
    L = L[L["bus0"] != L["bus1"]].copy()

    L["sync0"] = L["bus0"].map(buses_380_idx["sync_area"])
    L["sync1"] = L["bus1"].map(buses_380_idx["sync_area"])
    L["dc0"] = L["bus0"].map(buses_380_idx["dc_bool"])
    L["dc1"] = L["bus1"].map(buses_380_idx["dc_bool"])

    # keep only AC branches in the line table
    L = L[L["dc0"].eq(False) & L["dc1"].eq(False)].copy()

    # Lift internal AC branches of the same synchronous area to the common
    # 380-kV representation.
    L["voltage"] = np.where(
        L["sync0"].notna() & L["sync0"].eq(L["sync1"]),
        380,
        pd.to_numeric(L["voltage"], errors="coerce").fillna(380),
    )

    L = L.drop(columns=[c for c in ["sync0", "sync1", "dc0", "dc1"] if c in L.columns])

    # -----------------------------
    # Transformers
    # -----------------------------
    T["bus0"] = T["bus0"].map(busmap_380).fillna(T["bus0"])
    T["bus1"] = T["bus1"].map(busmap_380).fillna(T["bus1"])
    T = T.dropna(subset=["bus0", "bus1"]).copy()
    T = T[T["bus0"] != T["bus1"]].copy()

    # -----------------------------
    # Links / Converter
    # -----------------------------
    for D in [LK, CV]:
        D["bus0"] = D["bus0"].map(busmap_380).fillna(D["bus0"])
        D["bus1"] = D["bus1"].map(busmap_380).fillna(D["bus1"])
        D.dropna(subset=["bus0", "bus1"], inplace=True)
        D.drop(D.index[D["bus0"].eq(D["bus1"])], inplace=True)

    # -----------------------------
    # Plants
    # -----------------------------
    P["assigned_bus_380"] = P["assigned_bus"].map(busmap_380).fillna(P["assigned_bus"])

    # Derive AC sub-networks for all synchronous areas. CESA keeps the historical
    # `sn_*` naming, external areas get area-specific prefixes when they are not collapsed.
    buses_380 = _fill_missing_ac_sub_networks(buses_380, L)

    return {
        "busmap_380": busmap_380.rename("bus_id_380"),
        "buses": buses_380,
        "lines": L.reset_index(drop=True),
        "links": LK.reset_index(drop=True),
        "converters": CV.reset_index(drop=True),
        "transformers": T.reset_index(drop=True),
        "plants": P.reset_index(drop=True),
    }


# ============================================================
# nominal capacity (s nom) imputation
# ============================================================
def impute_zero_s_nom_by_country_voltage(
    lines: pd.DataFrame,
    buses0: pd.DataFrame,
    *,
    s_nom_col: str = "s_nom",
    voltage_col: str = "voltage",
) -> pd.DataFrame:
    """
    Impute lines with s_nom <= 0 (or NaN) by median of (country, voltage).

    Country for a line:
      - if both endpoints are in same country -> that country
      - else cross-border:
          * average of both country-voltage medians, if available
          * else one side's country-voltage median
          * else voltage-level median
          * else global positive median
    """
    L = lines.copy()

    # -----------------------------
    # Make function idempotent
    # -----------------------------
    helper_cols = [
        "country0",
        "country1",
        "line_country",
        "s_nom_original",
        "s_nom_imputed",
        "s_nom_impute_source",
    ]
    L = L.drop(columns=[c for c in helper_cols if c in L.columns], errors="ignore")

    for c in [s_nom_col, voltage_col]:
        if c in L.columns:
            L[c] = pd.to_numeric(L[c], errors="coerce")
        else:
            raise KeyError(f"Missing required line column: {c}")

    B = buses0[["bus_id", "country"]].copy()
    B["bus_id"] = B["bus_id"].astype(str)
    B["country"] = B["country"].astype(str)

    L["bus0"] = L["bus0"].astype(str)
    L["bus1"] = L["bus1"].astype(str)

    b0 = B.rename(columns={"bus_id": "bus0", "country": "country0"})
    b1 = B.rename(columns={"bus_id": "bus1", "country": "country1"})
    L = L.merge(b0, on="bus0", how="left").merge(b1, on="bus1", how="left")

    L["line_country"] = np.where(
        L["country0"].notna() & L["country1"].notna() & L["country0"].eq(L["country1"]),
        L["country0"],
        pd.NA,
    )

    positive = L[s_nom_col].notna() & (L[s_nom_col] > 0)

    med_country_voltage = (
        L.loc[positive & L["line_country"].notna()]
        .groupby(["line_country", voltage_col])[s_nom_col]
        .median()
    )

    med_voltage = L.loc[positive].groupby(voltage_col)[s_nom_col].median()
    med_global = float(L.loc[positive, s_nom_col].median()) if positive.any() else np.nan

    L["s_nom_original"] = L[s_nom_col]
    L["s_nom_imputed"] = False
    L["s_nom_impute_source"] = pd.Series(pd.NA, index=L.index, dtype="object")

    need = L[s_nom_col].isna() | (L[s_nom_col] <= 0)

    def _cv(country: str | None, voltage: float | int | None) -> float | None:
        if country is None or pd.isna(country) or pd.isna(voltage):
            return None
        key = (str(country), voltage)
        return float(med_country_voltage.loc[key]) if key in med_country_voltage.index else None

    for i in L.index[need]:
        c0 = L.at[i, "country0"]
        c1 = L.at[i, "country1"]
        v = L.at[i, voltage_col]
        lc = L.at[i, "line_country"]

        fill = None
        source = None

        if pd.notna(lc):
            fill = _cv(str(lc), v)
            source = "country_voltage"
        else:
            m0 = _cv(str(c0), v) if pd.notna(c0) else None
            m1 = _cv(str(c1), v) if pd.notna(c1) else None
            vals = [x for x in [m0, m1] if x is not None and np.isfinite(x)]

            if len(vals) == 2:
                fill = float(np.mean(vals))
                source = "crossborder_mean_country_voltage"
            elif len(vals) == 1:
                fill = float(vals[0])
                source = "crossborder_single_country_voltage"

        if (fill is None or not np.isfinite(fill)) and (v in med_voltage.index):
            fill = float(med_voltage.loc[v])
            source = "voltage"

        if fill is None or not np.isfinite(fill):
            fill = med_global
            source = "global"

        if fill is None or not np.isfinite(fill) or fill <= 0:
            fill = 1.0
            source = "fallback_1mw"

        L.at[i, s_nom_col] = float(fill)
        L.at[i, "s_nom_imputed"] = True
        L.at[i, "s_nom_impute_source"] = source

    return L


# ============================================================
# k allocation by group with country-level generation+load weighting
# ============================================================
def _normalize_positive_series(values: pd.Series) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    out = out.where(out > 0.0, 0.0)
    total = float(out.sum())
    if total <= 0:
        return pd.Series(0.0, index=out.index, dtype=float)
    return out / total


def _group_generation_weights(
    plants_assigned_s: pd.DataFrame,
    buses_s: pd.DataFrame,
    groups_df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    plant_bus_col: str = "assigned_bus_380",
) -> pd.Series:
    idx = _group_index_from_frame(groups_df, group_cols)

    if plants_assigned_s.empty:
        return pd.Series(0.0, index=idx, name="generation_mw")

    if plant_bus_col not in plants_assigned_s.columns:
        return pd.Series(0.0, index=idx, name="generation_mw")

    P = plants_assigned_s.copy()
    P[plant_bus_col] = P[plant_bus_col].astype(str)
    P["Capacity"] = pd.to_numeric(P.get("Capacity", 0.0), errors="coerce").fillna(0.0)

    meta_cols = ["bus_id", *group_cols]
    meta = buses_s[meta_cols].copy()
    meta["bus_id"] = meta["bus_id"].astype(str)
    for c in group_cols:
        meta[c] = meta[c].astype(str)

    P = P.merge(meta, left_on=plant_bus_col, right_on="bus_id", how="left")

    if P.empty:
        return pd.Series(0.0, index=idx, name="generation_mw")

    for c in group_cols:
        P[c] = P[c].astype(str)

    g = P.groupby(list(group_cols))["Capacity"].sum()

    out = g.reindex(idx).fillna(0.0).astype(float)
    out.name = "generation_mw"
    return out


def _allocate_k_to_groups_from_generation(
    buses_cluster: pd.DataFrame,
    plants_assigned_s: pd.DataFrame,
    *,
    k_total: int,
    group_cols: Sequence[str],
    plant_bus_col: str = "assigned_bus_380",
    country_load_mean: pd.Series | None = None,
    generation_country_weight: float = 1.0,
    load_country_weight: float = 0.0,
) -> pd.DataFrame:
    if generation_country_weight < 0 or load_country_weight < 0:
        raise ValueError("generation_country_weight and load_country_weight must be non-negative")

    groups = (
        buses_cluster.groupby(list(group_cols))["bus_id"]
        .count()
        .reset_index(name="n_buses")
        .sort_values(list(group_cols))
        .reset_index(drop=True)
    )

    gen_weights = _group_generation_weights(
        plants_assigned_s,
        buses_cluster,
        groups,
        group_cols=group_cols,
        plant_bus_col=plant_bus_col,
    )

    groups["generation_weight_mw"] = gen_weights.values

    if "country" in group_cols:
        country_load = pd.Series(dtype=float) if country_load_mean is None else country_load_mean.copy()
        country_load.index = country_load.index.astype(str)
        country_load = _to_numeric_loose(country_load).fillna(0.0)

        # The country budget fixes the spatial scale before clustering starts.
        # Generation anchors the reduction at existing supply locations; load can
        # be mixed in for studies where demand-side representation is dominant.
        country_generation = groups.groupby("country")["generation_weight_mw"].sum().astype(float)
        country_bus_counts = groups.groupby("country")["n_buses"].sum().astype(float)
        country_load = country_load.reindex(country_generation.index).fillna(0.0).astype(float)

        gen_country_share = _normalize_positive_series(country_generation)
        load_country_share = _normalize_positive_series(country_load)
        bus_country_share = _normalize_positive_series(country_bus_counts)

        combined_country_share = (
            float(generation_country_weight) * gen_country_share
            + float(load_country_weight) * load_country_share
        )
        combined_country_share = _normalize_positive_series(combined_country_share)
        if float(combined_country_share.sum()) <= 0:
            combined_country_share = bus_country_share

        groups["country_generation_mw"] = groups["country"].map(country_generation).fillna(0.0)
        groups["country_load_mean_mw"] = groups["country"].map(country_load).fillna(0.0)
        groups["country_generation_share"] = groups["country"].map(gen_country_share).fillna(0.0)
        groups["country_load_share"] = groups["country"].map(load_country_share).fillna(0.0)
        groups["country_combined_share"] = groups["country"].map(combined_country_share).fillna(0.0)

        groups["group_share_within_country"] = 0.0
        # Within each country aggregate, plant capacity distributes the assigned
        # clusters over synchronous areas. Bus counts are used only when there is
        # no meaningful generation basis in the input data.
        for country, idxs in groups.groupby("country").groups.items():
            idxs = list(idxs)
            base = groups.loc[idxs, "generation_weight_mw"].astype(float)
            if float(np.nansum(base.values)) <= 0:
                base = groups.loc[idxs, "n_buses"].astype(float)

            base_share = _normalize_positive_series(pd.Series(base.values, index=groups.index[idxs]))
            groups.loc[idxs, "group_share_within_country"] = base_share.values

        groups["allocation_weight"] = (
            groups["country_combined_share"].astype(float) * groups["group_share_within_country"].astype(float)
        )
    else:
        groups["country_generation_mw"] = 0.0
        groups["country_load_mean_mw"] = 0.0
        groups["country_generation_share"] = 0.0
        groups["country_load_share"] = 0.0
        groups["country_combined_share"] = 0.0
        groups["group_share_within_country"] = 0.0
        groups["allocation_weight"] = gen_weights.values.astype(float)

    weights = groups["allocation_weight"].values.astype(float)
    if np.nansum(weights) <= 0:
        weights = groups["n_buses"].values.astype(float)

    k_alloc = _allocate_integer_budget(
        groups["n_buses"].values,
        k_total=k_total,
        weights=weights,
        min_per_nonempty=1,
    )

    groups["k_group"] = k_alloc
    groups["grouping_mode"] = _grouping_mode_name(group_cols)
    return groups


# ============================================================
# Graph helpers
# ============================================================
def _subset_lines_for_bus_ids(lines: pd.DataFrame, bus_ids: np.ndarray) -> pd.DataFrame:
    ids = set(pd.Index(bus_ids).astype(str))
    L = lines.copy()
    L["bus0"] = L["bus0"].astype(str)
    L["bus1"] = L["bus1"].astype(str)
    return L[L["bus0"].isin(ids) & L["bus1"].isin(ids)].copy()


def _build_unweighted_topology(bus_ids: np.ndarray, lines_g: pd.DataFrame) -> sparse.csr_matrix:
    idx = {bid: i for i, bid in enumerate(pd.Index(bus_ids).astype(str))}
    rows = []
    cols = []
    for u, v in zip(lines_g["bus0"].astype(str).values, lines_g["bus1"].astype(str).values):
        if u in idx and v in idx and u != v:
            i, j = idx[u], idx[v]
            rows.extend([i, j])
            cols.extend([j, i])

    n = len(bus_ids)
    if not rows:
        return sparse.csr_matrix((n, n))

    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    A.data[:] = 1.0
    return A


def _coords_feature_matrix(bg: pd.DataFrame, bus_ids_subset: np.ndarray) -> np.ndarray:
    sub = (
        bg.loc[bg["bus_id"].isin(bus_ids_subset), ["bus_id", "lat", "lon"]]
        .drop_duplicates("bus_id")
        .set_index("bus_id")
        .loc[pd.Index(bus_ids_subset).astype(str)]
    )
    lat = sub["lat"].values.astype(float)
    lon = sub["lon"].values.astype(float)
    lat0 = np.deg2rad(np.nanmean(lat))
    return np.c_[lon * np.cos(lat0), lat]


def _edge_weights(
    lines_g: pd.DataFrame,
    buses_g: pd.DataFrame,
    *,
    similarity: str = "electrical",   # electrical | geographical
    electrical_weight_mode: str = "b",   # b | b*s_nom
) -> pd.DataFrame:
    L = lines_g.copy()
    if L.empty:
        return pd.DataFrame(columns=["bus0", "bus1", "weight"])

    if similarity == "electrical":
        x = pd.to_numeric(L["x"], errors="coerce")
        ok = x.notna() & (x.abs() > 1e-9)
        L = L.loc[ok].copy()
        if L.empty:
            return pd.DataFrame(columns=["bus0", "bus1", "weight"])

        w = np.abs(
            pd.to_numeric(L["circuits"], errors="coerce").fillna(1.0).values.astype(float)
            / pd.to_numeric(L["x"], errors="coerce").values.astype(float)
        )
        if electrical_weight_mode == "b*s_nom":
            s = pd.to_numeric(L["s_nom"], errors="coerce").fillna(0.0).values.astype(float)
            w = w * np.maximum(s, 0.0)
        elif electrical_weight_mode != "b":
            raise ValueError("electrical_weight_mode must be 'b' or 'b*s_nom'")

    elif similarity == "geographical":
        bxy = buses_g[["bus_id", "lat", "lon"]].drop_duplicates("bus_id").set_index("bus_id")
        coords0 = bxy.loc[L["bus0"].astype(str), ["lat", "lon"]].values
        coords1 = bxy.loc[L["bus1"].astype(str), ["lat", "lon"]].values
        dist_km = _haversine_array(coords0[:, 0], coords0[:, 1], coords1[:, 0], coords1[:, 1])
        w = 1.0 / np.maximum(dist_km, 1e-3)

    else:
        raise ValueError("similarity must be 'electrical' or 'geographical'")

    out = pd.DataFrame(
        {
            "bus0": L["bus0"].astype(str).values,
            "bus1": L["bus1"].astype(str).values,
            "weight": w.astype(float),
        }
    )
    out["u"] = np.minimum(out["bus0"].values, out["bus1"].values)
    out["v"] = np.maximum(out["bus0"].values, out["bus1"].values)
    out = out.groupby(["u", "v"], as_index=False)["weight"].sum()
    return out.rename(columns={"u": "bus0", "v": "bus1"})


def _build_sparse_weighted_adjacency(bus_ids: np.ndarray, edges: pd.DataFrame) -> sparse.csr_matrix:
    idx = {bid: i for i, bid in enumerate(pd.Index(bus_ids).astype(str))}
    rows = []
    cols = []
    data = []

    for u, v, w in zip(edges["bus0"].astype(str).values, edges["bus1"].astype(str).values, edges["weight"].values):
        if u in idx and v in idx and u != v and np.isfinite(w) and w > 0:
            i, j = idx[u], idx[v]
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([float(w), float(w)])

    n = len(bus_ids)
    if not rows:
        return sparse.csr_matrix((n, n))

    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    W.sum_duplicates()
    return W


def _dense_symmetric_eigenvectors(
    matrix: sparse.spmatrix,
    n_vecs: int,
    *,
    largest: bool,
) -> np.ndarray:
    dense = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    vals, vecs = np.linalg.eigh(dense)
    order = np.argsort(vals)
    if largest:
        order = order[::-1]
    return vecs[:, order[:n_vecs]]


def _safe_symmetric_eigenvectors(
    matrix: sparse.spmatrix,
    n_vecs: int,
    *,
    largest: bool,
    random_state: int = 0,
    dense_max_n: int = 2500,
) -> np.ndarray:
    """Return selected eigenvectors without SciPy sparse eigensolvers.

    ARPACK and LOBPCG have shown native crashes on some Windows/SciPy builds
    for the larger 256-node reduction. The dense path is safe for the country
    and subnetwork blocks used here; oversized blocks fall back to coordinate
    clustering through the caller's normal exception handling.
    """
    n = int(matrix.shape[0])
    n_vecs = max(1, min(int(n_vecs), max(1, n - 1)))
    if n <= 1:
        return np.ones((n, 1), dtype=float)

    if n > dense_max_n:
        raise ValueError(
            f"Spectral block with {n} nodes exceeds dense fallback limit "
            f"({dense_max_n})."
        )

    vecs = _dense_symmetric_eigenvectors(matrix, n_vecs, largest=largest)
    if not np.isfinite(vecs).all():
        raise ValueError("Dense eigensolver returned non-finite eigenvectors.")
    return vecs


def _kmeans_labels(
    X: np.ndarray,
    n_clusters: int,
    *,
    random_state: int = 0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n = int(X.shape[0])
    k = max(1, min(int(n_clusters), n))
    if n == 0:
        return np.zeros(0, dtype=int)
    if k <= 1:
        return np.zeros(n, dtype=int)
    if n <= k:
        return np.arange(n, dtype=int)

    finite = np.isfinite(X)
    if not finite.all():
        col_means = np.divide(
            np.where(finite, X, 0.0).sum(axis=0),
            finite.sum(axis=0),
            out=np.zeros(X.shape[1], dtype=float),
            where=finite.sum(axis=0) > 0,
        )
        X = np.where(finite, X, col_means)

    rng = np.random.default_rng(random_state)
    first = int(rng.integers(n))
    centers = np.empty((k, X.shape[1]), dtype=float)
    centers[0] = X[first]
    chosen = {first}
    closest = np.sum((X - centers[0]) ** 2, axis=1)
    for j in range(1, k):
        idx = int(np.argmax(closest))
        if closest[idx] <= 0.0 or idx in chosen:
            remaining = [i for i in range(n) if i not in chosen]
            idx = remaining[0]
        centers[j] = X[idx]
        chosen.add(idx)
        closest = np.minimum(closest, np.sum((X - centers[j]) ** 2, axis=1))

    labels = np.full(n, -1, dtype=int)
    for _ in range(max_iter):
        dist2 = np.empty((n, k), dtype=float)
        for j in range(k):
            delta = X - centers[j]
            dist2[:, j] = np.sum(delta * delta, axis=1)

        new_labels = np.argmin(dist2, axis=1).astype(int)
        counts = np.bincount(new_labels, minlength=k)
        empty = np.where(counts == 0)[0]
        if empty.size:
            assigned_dist = dist2[np.arange(n), new_labels]
            for j in empty:
                for idx in np.argsort(assigned_dist)[::-1]:
                    src = new_labels[idx]
                    if counts[src] > 1:
                        counts[src] -= 1
                        new_labels[idx] = int(j)
                        counts[j] += 1
                        assigned_dist[idx] = 0.0
                        break

        new_centers = centers.copy()
        for j in range(k):
            members = X[new_labels == j]
            if len(members):
                new_centers[j] = members.mean(axis=0)

        movement = float(np.max(np.sum((new_centers - centers) ** 2, axis=1)))
        if np.array_equal(labels, new_labels) and movement <= tol:
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers

    return labels


def _spectral_labels(W: sparse.csr_matrix, k: int, *, random_state: int = 0) -> np.ndarray:
    n = W.shape[0]
    if k <= 1 or n <= 1:
        return np.zeros(n, dtype=int)
    if n <= k:
        return np.arange(n, dtype=int)

    d = np.asarray(W.sum(axis=1)).ravel()
    d = np.maximum(d, 1e-12)
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(d))
    Lsym = sparse.eye(n, format="csr") - D_inv_sqrt @ W @ D_inv_sqrt

    nev = min(k, n - 1)
    vecs = _safe_symmetric_eigenvectors(
        Lsym,
        n_vecs=nev,
        largest=False,
        random_state=random_state,
    )
    rn = np.linalg.norm(vecs, axis=1)
    rn[rn == 0] = 1.0
    X = vecs / rn[:, None]
    return _kmeans_labels(X, n_clusters=k, random_state=random_state)


def _build_dc_susceptance_laplacian(
    bus_ids: np.ndarray,
    lines_g: pd.DataFrame,
) -> sparse.csr_matrix:
    """
    Build DC susceptance Laplacian Bbus for one connected AC component.

    Uses:
        b_ij = circuits / |x|
    and then
        Bbus = diag(sum b_ij) - W

    Notes
    -----
    - This is intentionally based on susceptance only, not s_nom.
    - For the intended 'effective reactance distance' backend, this is the
      cleanest DC analogue to an admittance-based electrical proximity.
    """
    ids = pd.Index(bus_ids).astype(str)
    idx = {bid: i for i, bid in enumerate(ids)}

    L = lines_g.copy()
    L["bus0"] = L["bus0"].astype(str)
    L["bus1"] = L["bus1"].astype(str)

    x = pd.to_numeric(L["x"], errors="coerce")
    ok = x.notna() & (x.abs() > 1e-9)
    L = L.loc[ok].copy()

    if L.empty:
        return sparse.csr_matrix((len(ids), len(ids)))

    circuits = pd.to_numeric(L["circuits"], errors="coerce").fillna(1.0).values.astype(float)
    xabs = np.abs(pd.to_numeric(L["x"], errors="coerce").values.astype(float))
    bij = circuits / xabs

    rows = []
    cols = []
    data = []

    for u, v, b in zip(L["bus0"].values, L["bus1"].values, bij):
        if u in idx and v in idx and u != v and np.isfinite(b) and b > 0:
            i, j = idx[u], idx[v]
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([float(b), float(b)])

    n = len(ids)
    if not rows:
        return sparse.csr_matrix((n, n))

    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    W.sum_duplicates()

    d = np.asarray(W.sum(axis=1)).ravel()
    Bbus = sparse.diags(d) - W
    return Bbus.tocsr()


def _effective_reactance_distance_dense(
    bus_ids: np.ndarray,
    lines_g: pd.DataFrame,
) -> np.ndarray:
    """
    Pairwise DC effective-reactance distance matrix for one connected component.

    For a connected weighted graph with Laplacian L = Bbus,
    the distance is

        d_ij = L^+_ii + L^+_jj - 2 L^+_ij

    where L^+ is the Moore-Penrose pseudoinverse.

    Returns
    -------
    D : (n, n) ndarray
        Symmetric nonnegative distance matrix with zeros on the diagonal.
    """
    n = len(bus_ids)
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    Bbus = _build_dc_susceptance_laplacian(bus_ids, lines_g)

    if Bbus.nnz == 0:
        return np.zeros((n, n), dtype=float)

    # exact dense pseudoinverse; acceptable because we apply it component-wise
    Ldense = Bbus.toarray().astype(float)
    Ldag = np.linalg.pinv(Ldense, hermitian=True)

    diag = np.diag(Ldag)
    D = diag[:, None] + diag[None, :] - 2.0 * Ldag
    D = np.maximum(D, 0.0)

    # symmetrize numerically + clean diagonal
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def _distance_matrix_to_similarity_edges(
    bus_ids: np.ndarray,
    D: np.ndarray,
    *,
    k_nearest: int | None = 12,
    kernel: str = "exp",          # "exp" | "reciprocal"
    tau_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Convert a dense pairwise distance matrix into a sparse similarity edge list.

    Similarity:
        exp(-d/tau)     if kernel == "exp"
        1/(eps + d)     if kernel == "reciprocal"

    Sparsification:
        keep k nearest neighbours per node (if k_nearest is not None)

    Returns
    -------
    DataFrame with columns: bus0, bus1, weight
    """
    ids = pd.Index(bus_ids).astype(str)
    n = len(ids)

    if n <= 1:
        return pd.DataFrame(columns=["bus0", "bus1", "weight"])

    D = np.asarray(D, dtype=float).copy()
    D[~np.isfinite(D)] = np.inf
    np.fill_diagonal(D, np.inf)

    pos = D[np.isfinite(D)]
    pos = pos[pos > 0]

    if pos.size == 0:
        return pd.DataFrame(columns=["bus0", "bus1", "weight"])

    tau = float(np.median(pos)) * float(tau_scale)
    if not np.isfinite(tau) or tau <= 0:
        tau = 1.0

    if kernel == "exp":
        S = np.exp(-D / tau)
    elif kernel == "reciprocal":
        S = 1.0 / np.maximum(D, 1e-9)
    else:
        raise ValueError("kernel must be 'exp' or 'reciprocal'")

    S[~np.isfinite(S)] = 0.0
    np.fill_diagonal(S, 0.0)

    if k_nearest is None or k_nearest >= n - 1:
        keep = np.ones_like(S, dtype=bool)
        np.fill_diagonal(keep, False)
    else:
        # Effective-reactance distances are dense by construction. Keeping only a
        # symmetric nearest-neighbour graph retains electrical proximity without
        # letting very weak long-range similarities dominate the spectral step.
        keep = np.zeros_like(S, dtype=bool)
        kk = max(1, int(k_nearest))
        for i in range(n):
            order = np.argsort(D[i, :])
            nbrs = [j for j in order[:kk] if np.isfinite(D[i, j])]
            keep[i, nbrs] = True

        # symmetrize neighbourhood
        keep = keep | keep.T

    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            if keep[i, j] and S[i, j] > 0:
                rows.append((ids[i], ids[j], float(S[i, j])))

    if not rows:
        return pd.DataFrame(columns=["bus0", "bus1", "weight"])

    return pd.DataFrame(rows, columns=["bus0", "bus1", "weight"])


def _similarity_edges_for_component(
    bus_ids_c: np.ndarray,
    lines_c: pd.DataFrame,
    buses_c: pd.DataFrame,
    *,
    similarity: str,
    electrical_weight_mode: str = "b",
    effrx_k_nearest: int | None = 12,
    effrx_kernel: str = "exp",
    effrx_tau_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Unified similarity-edge builder for one connected component.

    Supported similarity modes:
      - "electrical"
      - "geographical"
      - "dc_effective_reactance"
    """
    if similarity in {"electrical", "geographical"}:
        return _edge_weights(
            lines_c,
            buses_c[["bus_id", "lat", "lon"]],
            similarity=similarity,
            electrical_weight_mode=electrical_weight_mode,
        )

    if similarity == "dc_effective_reactance":
        # This backend uses the DC Laplacian as a network-distance model first
        # and converts distances to graph similarities only afterwards.
        D = _effective_reactance_distance_dense(bus_ids_c, lines_c)
        return _distance_matrix_to_similarity_edges(
            bus_ids_c,
            D,
            k_nearest=effrx_k_nearest,
            kernel=effrx_kernel,
            tau_scale=effrx_tau_scale,
        )

    raise ValueError(
        "similarity must be one of "
        "'electrical', 'geographical', 'dc_effective_reactance'"
    )


# ============================================================
# Clustering methods
# ============================================================
def cluster_kmeans_geo(
    buses_s: pd.DataFrame,
    *,
    k_by_group: dict[tuple[str, ...], int],
    group_cols: Sequence[str],
    random_state: int = 0,
) -> pd.Series:
    out = pd.Series(index=buses_s.index, dtype="object")

    for gkey_raw, bg in buses_s.groupby(list(group_cols)):
        gkey = _group_key_to_str_tuple(gkey_raw)
        glabel = _group_key_to_label(gkey)

        bus_ids = bg["bus_id"].astype(str).values
        k_group = int(k_by_group.get(gkey, 1))
        k_group = max(1, min(k_group, len(bus_ids)))

        X = _coords_feature_matrix(bg, bus_ids)
        if k_group <= 1:
            labels = np.zeros(len(bg), dtype=int)
        else:
            labels = _kmeans_labels(X, n_clusters=k_group, random_state=random_state)

        for local in np.unique(labels):
            members = bus_ids[labels == local]
            rows = bg.index[bg["bus_id"].isin(members)]
            out.loc[rows] = f"cl_ac_{glabel}_geo_{int(local)}"

    return out


def cluster_electrical_spectral(
    buses_s: pd.DataFrame,
    lines_s: pd.DataFrame,
    *,
    k_by_group: dict[tuple[str, ...], int],
    group_cols: Sequence[str],
    similarity: str = "electrical",              # electrical | geographical | dc_effective_reactance
    electrical_weight_mode: str = "b",
    effrx_k_nearest: int | None = 12,
    effrx_kernel: str = "exp",
    effrx_tau_scale: float = 1.0,
    random_state: int = 0,
) -> pd.Series:
    out = pd.Series(index=buses_s.index, dtype="object")

    for gkey_raw, bg in buses_s.groupby(list(group_cols)):
        gkey = _group_key_to_str_tuple(gkey_raw)
        glabel = _group_key_to_label(gkey)

        bus_ids_all = bg["bus_id"].astype(str).values
        k_group = int(k_by_group.get(gkey, 1))
        k_group = max(1, min(k_group, len(bus_ids_all)))

        lg = _subset_lines_for_bus_ids(lines_s, bus_ids_all)
        A = _build_unweighted_topology(bus_ids_all, lg)
        n_comp, labels_cc = connected_components(A, directed=False, return_labels=True)
        comp_sizes = np.bincount(labels_cc, minlength=n_comp)
        k_by_comp = _allocate_k_across_components(comp_sizes, k_group)

        for c in range(n_comp):
            idx_c = np.where(labels_cc == c)[0]
            bus_ids_c = bus_ids_all[idx_c]
            if len(idx_c) == 0:
                continue

            k_c = int(k_by_comp[c])
            if len(idx_c) <= 1 or k_c <= 1:
                lab = np.zeros(len(idx_c), dtype=int)
            else:
                lgc = _subset_lines_for_bus_ids(lg, bus_ids_c)

                edges_c = _similarity_edges_for_component(
                    bus_ids_c,
                    lgc,
                    bg.loc[bg["bus_id"].isin(bus_ids_c)],
                    similarity=similarity,
                    electrical_weight_mode=electrical_weight_mode,
                    effrx_k_nearest=effrx_k_nearest,
                    effrx_kernel=effrx_kernel,
                    effrx_tau_scale=effrx_tau_scale,
                )

                Wc = _build_sparse_weighted_adjacency(bus_ids_c, edges_c)

                if Wc.nnz == 0:
                    Xc = _coords_feature_matrix(bg, bus_ids_c)
                    lab = _kmeans_labels(
                        Xc,
                        n_clusters=min(k_c, len(bus_ids_c)),
                        random_state=random_state,
                    )
                else:
                    try:
                        lab = _spectral_labels(Wc, k=k_c, random_state=random_state)
                    except Exception:
                        Xc = _coords_feature_matrix(bg, bus_ids_c)
                        lab = _kmeans_labels(
                            Xc,
                            n_clusters=min(k_c, len(bus_ids_c)),
                            random_state=random_state,
                        )

            for local in np.unique(lab):
                members = bus_ids_c[lab == local]
                rows = bg.index[bg["bus_id"].isin(members)]
                out.loc[rows] = f"cl_ac_{glabel}_spec_{c}_{int(local)}"

    return out


def cluster_hac(
    buses_s: pd.DataFrame,
    lines_s: pd.DataFrame,
    *,
    k_by_group: dict[tuple[str, ...], int],
    group_cols: Sequence[str],
    feature: str = "coords",                      # coords | spectral
    similarity: str = "electrical",              # electrical | geographical | dc_effective_reactance
    electrical_weight_mode: str = "b*s_nom",
    effrx_k_nearest: int | None = 12,
    effrx_kernel: str = "exp",
    effrx_tau_scale: float = 1.0,
    n_embed: int = 4,
) -> pd.Series:
    out = pd.Series(index=buses_s.index, dtype="object")

    for gkey_raw, bg in buses_s.groupby(list(group_cols)):
        gkey = _group_key_to_str_tuple(gkey_raw)
        glabel = _group_key_to_label(gkey)

        bus_ids_all = bg["bus_id"].astype(str).values
        k_group = int(k_by_group.get(gkey, 1))
        k_group = max(1, min(k_group, len(bus_ids_all)))

        lg = _subset_lines_for_bus_ids(lines_s, bus_ids_all)
        A = _build_unweighted_topology(bus_ids_all, lg)
        n_comp, labels_cc = connected_components(A, directed=False, return_labels=True)
        comp_sizes = np.bincount(labels_cc, minlength=n_comp)
        k_by_comp = _allocate_k_across_components(comp_sizes, k_group)

        for c in range(n_comp):
            nodes = np.where(labels_cc == c)[0]
            if nodes.size == 0:
                continue

            bus_ids_c = bus_ids_all[nodes]
            k_c = int(k_by_comp[c])
            k_c = max(1, min(k_c, len(bus_ids_c)))

            if nodes.size == 1:
                rows = bg.index[bg["bus_id"].isin(bus_ids_c)]
                out.loc[rows] = f"cl_ac_{glabel}_hac_{c}_0"
                continue

            A_c = A[nodes[:, None], nodes]

            if feature == "coords":
                X_c = _coords_feature_matrix(bg, bus_ids_c)

            elif feature == "spectral":
                lgc = _subset_lines_for_bus_ids(lg, bus_ids_c)

                edges_c = _similarity_edges_for_component(
                    bus_ids_c,
                    lgc,
                    bg.loc[bg["bus_id"].isin(bus_ids_c)],
                    similarity=similarity,
                    electrical_weight_mode=electrical_weight_mode,
                    effrx_k_nearest=effrx_k_nearest,
                    effrx_kernel=effrx_kernel,
                    effrx_tau_scale=effrx_tau_scale,
                )

                W_c = _build_sparse_weighted_adjacency(bus_ids_c, edges_c)

                if W_c.nnz == 0:
                    X_c = _coords_feature_matrix(bg, bus_ids_c)
                else:
                    d = np.asarray(W_c.sum(axis=1)).ravel()
                    d = np.maximum(d, 1e-12)
                    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(d))
                    S = D_inv_sqrt @ W_c @ D_inv_sqrt

                    nev = min(n_embed, nodes.size - 1)
                    if nev <= 0:
                        X_c = _coords_feature_matrix(bg, bus_ids_c)
                    else:
                        try:
                            vecs = _safe_symmetric_eigenvectors(
                                S,
                                n_vecs=nev,
                                largest=True,
                            )
                            rn = np.linalg.norm(vecs, axis=1)
                            rn[rn == 0] = 1.0
                            X_c = vecs / rn[:, None]
                        except Exception:
                            X_c = _coords_feature_matrix(bg, bus_ids_c)
            else:
                raise ValueError("feature must be 'coords' or 'spectral'")

            labels = AgglomerativeClustering(
                n_clusters=k_c,
                linkage="ward",
                metric="euclidean",
                connectivity=A_c,
            ).fit_predict(X_c)

            for local in np.unique(labels):
                members = bus_ids_c[labels == local]
                rows = bg.index[bg["bus_id"].isin(members)]
                out.loc[rows] = f"cl_ac_{glabel}_hac_{c}_{int(local)}"

    return out


def cluster_mst(
    buses_s: pd.DataFrame,
    lines_s: pd.DataFrame,
    *,
    k_by_group: dict[tuple[str, ...], int],
    group_cols: Sequence[str],
    similarity: str = "electrical",
    electrical_weight_mode: str = "b",
) -> pd.Series:
    """
    Maximum spanning tree clustering:
    build a maximum spanning tree per connected component and cut the
    (k_c - 1) weakest tree edges.
    """
    out = pd.Series(index=buses_s.index, dtype="object")

    for gkey_raw, bg in buses_s.groupby(list(group_cols)):
        gkey = _group_key_to_str_tuple(gkey_raw)
        glabel = _group_key_to_label(gkey)

        bus_ids_all = bg["bus_id"].astype(str).values
        k_group = int(k_by_group.get(gkey, 1))
        k_group = max(1, min(k_group, len(bus_ids_all)))

        lg = _subset_lines_for_bus_ids(lines_s, bus_ids_all)
        A = _build_unweighted_topology(bus_ids_all, lg)
        n_comp, labels_cc = connected_components(A, directed=False, return_labels=True)
        comp_sizes = np.bincount(labels_cc, minlength=n_comp)
        k_by_comp = _allocate_k_across_components(comp_sizes, k_group)

        edges = _edge_weights(
            lg,
            bg[["bus_id", "lat", "lon"]],
            similarity=similarity,
            electrical_weight_mode=electrical_weight_mode,
        )

        for c in range(n_comp):
            idx_c = np.where(labels_cc == c)[0]
            if idx_c.size == 0:
                continue

            bus_ids_c = bus_ids_all[idx_c]
            k_c = int(k_by_comp[c])

            if idx_c.size <= 1 or k_c <= 1:
                lab = np.zeros(idx_c.size, dtype=int)
            else:
                egc = edges[edges["bus0"].isin(bus_ids_c) & edges["bus1"].isin(bus_ids_c)].copy()

                G = nx.Graph()
                G.add_nodes_from(list(bus_ids_c))
                for u, v, ww in zip(egc["bus0"].values, egc["bus1"].values, egc["weight"].values):
                    if ww > 0:
                        G.add_edge(str(u), str(v), weight=float(ww))

                if G.number_of_edges() == 0:
                    Xc = _coords_feature_matrix(bg, bus_ids_c)
                    lab = _kmeans_labels(
                        Xc,
                        n_clusters=min(k_c, len(bus_ids_c)),
                        random_state=0,
                    )
                else:
                    Tm = nx.maximum_spanning_tree(G, weight="weight")
                    tree_edges = sorted(
                        Tm.edges(data=True),
                        key=lambda e: e[2].get("weight", 0.0)
                    )

                    for u, v, _ in tree_edges[:max(0, k_c - 1)]:
                        if Tm.has_edge(u, v):
                            Tm.remove_edge(u, v)

                    comps = list(nx.connected_components(Tm))
                    if len(comps) < k_c:
                        Xc = _coords_feature_matrix(bg, bus_ids_c)
                        lab = _kmeans_labels(
                            Xc,
                            n_clusters=min(k_c, len(bus_ids_c)),
                            random_state=0,
                        )
                    else:
                        lab = np.zeros(len(bus_ids_c), dtype=int)
                        pos = {b: i for i, b in enumerate(bus_ids_c)}
                        for ci, nodes_cc in enumerate(comps):
                            for b in nodes_cc:
                                lab[pos[str(b)]] = ci

            for local in np.unique(lab):
                members = bus_ids_c[lab == local]
                rows = bg.index[bg["bus_id"].isin(members)]
                out.loc[rows] = f"cl_ac_{glabel}_mst_{c}_{int(local)}"

    return out


# ============================================================
# Line aggregation
# ============================================================
def _aggregate_lines_line_equivalent(lines_work: pd.DataFrame) -> pd.DataFrame:
    L = lines_work.copy()

    for col in ["r", "x", "b", "s_nom", "circuits", "length", "voltage"]:
        if col in L.columns:
            L[col] = pd.to_numeric(L[col], errors="coerce")

    u = np.minimum(L["bus0_red"].values, L["bus1_red"].values)
    v = np.maximum(L["bus0_red"].values, L["bus1_red"].values)
    L["u"], L["v"] = u, v

    mask = L["r"].notna() & L["x"].notna() & ~((L["r"] == 0) & (L["x"] == 0))
    y = np.full(len(L), np.nan + 1j * np.nan, dtype=complex)
    y[mask.values] = 1.0 / (L.loc[mask, "r"].values + 1j * L.loc[mask, "x"].values)
    L["y_real"], L["y_imag"] = np.real(y), np.imag(y)

    def wmean(a: pd.Series, w: np.ndarray):
        a = a.values.astype(float)
        m = np.isfinite(a) & np.isfinite(w) & (w > 0)
        return float(np.sum(a[m] * w[m]) / np.sum(w[m])) if np.any(m) else np.nan

    agg_kwargs = dict(
        s_nom=("s_nom", "sum"),
        circuits=("circuits", "sum"),
        b=("b", "sum"),
        y_real=("y_real", "sum"),
        y_imag=("y_imag", "sum"),
        n_lines=("line_id", "count"),
        n_lines_imputed=("s_nom_imputed", "sum"),
        length=("length", lambda s: wmean(s, L.loc[s.index, "s_nom"].fillna(1.0).values)),
    )
    if "grid_source" in L.columns:
        agg_kwargs["grid_source"] = ("grid_source", _unique_sorted_join)

    lines_red = L.groupby(["u", "v", "voltage"], as_index=False).agg(**agg_kwargs)

    ysum = lines_red["y_real"].values + 1j * lines_red["y_imag"].values
    zeq = np.where(np.abs(ysum) > 0, 1.0 / ysum, np.nan + 1j * np.nan)
    lines_red["r_eq"], lines_red["x_eq"] = np.real(zeq), np.imag(zeq)
    lines_red["line_id"] = [f"eq_{i}" for i in range(len(lines_red))]

    cols = ["line_id", "u", "v", "voltage", "s_nom", "circuits", "r_eq", "x_eq", "b", "length", "n_lines", "n_lines_imputed"]
    if "grid_source" in lines_red.columns:
        cols.append("grid_source")
    return lines_red[cols]


def _aggregate_lines_location_based(
    lines_work: pd.DataFrame,
    buses_red: pd.DataFrame,
    *,
    length_factor: float = 1.0,
) -> pd.DataFrame:
    L = lines_work.copy()
    B = buses_red[["bus_id", "lat", "lon"]].copy().set_index("bus_id")

    for col in ["r", "x", "b", "s_nom", "circuits", "length", "voltage"]:
        if col in L.columns:
            L[col] = pd.to_numeric(L[col], errors="coerce")

    u = np.minimum(L["bus0_red"].values, L["bus1_red"].values)
    v = np.maximum(L["bus0_red"].values, L["bus1_red"].values)
    L["u"], L["v"] = u, v

    len0 = pd.to_numeric(L["length"], errors="coerce").astype(float).values
    len0 = np.where(np.isfinite(len0) & (len0 > 1e-6), len0, np.nan)

    L["r_spec"] = pd.to_numeric(L["r"], errors="coerce").astype(float).values / len0
    L["x_spec"] = pd.to_numeric(L["x"], errors="coerce").astype(float).values / len0
    L["b_spec"] = pd.to_numeric(L["b"], errors="coerce").astype(float).values / len0

    def wmean(series: pd.Series, weights: np.ndarray) -> float:
        a = pd.to_numeric(series, errors="coerce").values.astype(float)
        m = np.isfinite(a) & np.isfinite(weights) & (weights > 0)
        return float(np.sum(a[m] * weights[m]) / np.sum(weights[m])) if np.any(m) else np.nan

    agg_kwargs = dict(
        s_nom=("s_nom", "sum"),
        circuits=("circuits", "sum"),
        n_lines=("line_id", "count"),
        n_lines_imputed=("s_nom_imputed", "sum"),
        r_spec=("r_spec", lambda s: wmean(s, L.loc[s.index, "s_nom"].fillna(1.0).values)),
        x_spec=("x_spec", lambda s: wmean(s, L.loc[s.index, "s_nom"].fillna(1.0).values)),
        b_spec=("b_spec", lambda s: wmean(s, L.loc[s.index, "s_nom"].fillna(1.0).values)),
    )
    if "grid_source" in L.columns:
        agg_kwargs["grid_source"] = ("grid_source", _unique_sorted_join)

    grouped = L.groupby(["u", "v", "voltage"], as_index=False).agg(**agg_kwargs)

    coords_u = B.loc[grouped["u"].values, ["lat", "lon"]].values
    coords_v = B.loc[grouped["v"].values, ["lat", "lon"]].values
    new_len = _haversine_array(coords_u[:, 0], coords_u[:, 1], coords_v[:, 0], coords_v[:, 1]) * float(length_factor)

    grouped["length"] = new_len
    grouped["r_eq"] = grouped["r_spec"] * grouped["length"]
    grouped["x_eq"] = grouped["x_spec"] * grouped["length"]
    grouped["b"] = grouped["b_spec"] * grouped["length"]
    grouped["line_id"] = [f"loc_{i}" for i in range(len(grouped))]

    cols = ["line_id", "u", "v", "voltage", "s_nom", "circuits", "r_eq", "x_eq", "b", "length", "n_lines", "n_lines_imputed"]
    if "grid_source" in grouped.columns:
        cols.append("grid_source")
    return grouped[cols]


def _weighted_mean(values: pd.Series, weights: pd.Series | np.ndarray | None = None) -> float:
    x = pd.to_numeric(values, errors="coerce").astype(float)
    if weights is None:
        x = x[np.isfinite(x)]
        return float(x.mean()) if len(x) else np.nan

    w = pd.to_numeric(pd.Series(weights, index=values.index), errors="coerce").astype(float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        x = x[np.isfinite(x)]
        return float(x.mean()) if len(x) else np.nan
    return float(np.sum(x[m] * w[m]) / np.sum(w[m]))


def _is_simple_dc_path_component(dc_links: pd.DataFrame) -> bool:
    if dc_links.empty:
        return False

    edges = dc_links[["bus0", "bus1"]].astype(str).copy()
    edges["u"] = np.minimum(edges["bus0"].values, edges["bus1"].values)
    edges["v"] = np.maximum(edges["bus0"].values, edges["bus1"].values)
    edges = edges[["u", "v"]].drop_duplicates()

    if len(edges) != len(dc_links):
        return False

    G = nx.Graph()
    G.add_edges_from(edges.itertuples(index=False, name=None))

    if G.number_of_nodes() < 3:
        return False
    if G.number_of_edges() != G.number_of_nodes() - 1:
        return False

    degrees = dict(G.degree())
    if not degrees:
        return False
    if max(degrees.values()) > 2:
        return False

    n_endpoints = sum(1 for degree in degrees.values() if degree == 1)
    return n_endpoints == 2


def _collapse_point_to_point_hvdc_to_ac_links(
    links_red: pd.DataFrame,
    conv_red: pd.DataFrame,
    buses_red: pd.DataFrame,
    *,
    fallback_length_factor: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collapse standard HVDC topologies into direct AC-AC links.

    Supported cases:
      A: AC -- converter -- DC -- link -- DC -- converter -- AC
      A2: AC -- converter -- DC -- link -- ... -- link -- DC -- converter -- AC
      B: AC -- converter -- DC -- link -- AC
      C: AC -- link -- DC -- converter -- AC

    Typical case B/C appears after a true 1-node collapse of external synchronous areas.
    """

    if links_red.empty:
        return links_red.reset_index(drop=True), conv_red.reset_index(drop=True)

    B = buses_red.copy()
    B["bus_id"] = B["bus_id"].astype(str)

    if "dc" in B.columns:
        B["is_dc"] = B["dc"].astype(str).str.lower().isin(["t", "true", "1", "yes", "y"])
    elif "dc_bool" in B.columns:
        B["is_dc"] = B["dc_bool"].astype(bool)
    else:
        B["is_dc"] = False

    bus_is_dc = B.set_index("bus_id")["is_dc"]
    bus_coords = B.set_index("bus_id")[["lat", "lon"]]

    # AC clustering may leave auxiliary DC buses in simple converter-link chains.
    # For PyPSA-style OPF inputs these chains are represented more robustly as
    # direct controllable links between the adjacent reduced AC buses.
    L_all = links_red.copy()
    L_all["bus0"] = L_all["bus0"].astype(str)
    L_all["bus1"] = L_all["bus1"].astype(str)
    L_all["bus0_is_dc"] = L_all["bus0"].map(bus_is_dc).fillna(False)
    L_all["bus1_is_dc"] = L_all["bus1"].map(bus_is_dc).fillna(False)

    C_all = conv_red.copy()
    if not C_all.empty:
        C_all["bus0"] = C_all["bus0"].astype(str)
        C_all["bus1"] = C_all["bus1"].astype(str)
        c0_dc = C_all["bus0"].map(bus_is_dc)
        c1_dc = C_all["bus1"].map(bus_is_dc)
        mask_valid = c0_dc.notna() & c1_dc.notna() & (c0_dc != c1_dc)
        C_all = C_all.loc[mask_valid].copy()
        c0_dc = c0_dc.loc[mask_valid]
        c1_dc = c1_dc.loc[mask_valid]
        C_all["dc_bus"] = np.where(c0_dc.values, C_all["bus0"].values, C_all["bus1"].values)
        C_all["ac_bus"] = np.where(c0_dc.values, C_all["bus1"].values, C_all["bus0"].values)

    used_link_idx = set()
    used_conv_idx = set()
    new_links = []
    new_id = 0

    # ---------------------------------------------------------
    # Case A: DC-DC link + 2 converter
    # ---------------------------------------------------------
    L_dd = L_all.loc[L_all["bus0_is_dc"] & L_all["bus1_is_dc"]].copy()

    if not L_dd.empty and not C_all.empty:
        dc_nodes = pd.unique(pd.concat([L_dd["bus0"], L_dd["bus1"], C_all["dc_bus"]], ignore_index=True))
        Gdc = nx.Graph()
        Gdc.add_nodes_from(dc_nodes.astype(str))
        Gdc.add_edges_from(L_dd[["bus0", "bus1"]].astype(str).itertuples(index=False, name=None))

        for comp in nx.connected_components(Gdc):
            comp = set(str(x) for x in comp)

            Lc = L_dd.loc[L_dd["bus0"].isin(comp) & L_dd["bus1"].isin(comp)].copy()
            Cc = C_all.loc[C_all["dc_bus"].isin(comp)].copy()

            dc_buses = sorted(comp)
            ac_buses = sorted(pd.unique(Cc["ac_bus"].astype(str)))

            if len(ac_buses) != 2:
                continue

            ac0, ac1 = sorted(ac_buses)
            if ac0 == ac1:
                continue

            is_parallel_two_terminal = len(dc_buses) == 2
            is_simple_chain = _is_simple_dc_path_component(Lc)

            if not (is_parallel_two_terminal or is_simple_chain):
                continue

            caps = []
            if "p_nom" in Cc.columns:
                p_side = pd.to_numeric(Cc["p_nom"], errors="coerce").fillna(0.0).groupby(Cc["ac_bus"]).sum()
                for val in p_side.values:
                    if val > 0:
                        caps.append(float(val))
            if "p_nom" in Lc.columns:
                link_caps = pd.to_numeric(Lc["p_nom"], errors="coerce")
                link_caps = link_caps[link_caps > 0]
                if len(link_caps):
                    if is_parallel_two_terminal:
                        caps.append(float(link_caps.sum()))
                    else:
                        caps.append(float(link_caps.min()))
            p_nom = min(caps) if caps else np.nan

            length = np.nan
            if "length" in Lc.columns:
                ll = pd.to_numeric(Lc["length"], errors="coerce")
                ll = ll[ll > 0]
                if len(ll):
                    if is_parallel_two_terminal and "p_nom" in Lc.columns:
                        length = _weighted_mean(Lc.loc[ll.index, "length"], pd.to_numeric(Lc.loc[ll.index, "p_nom"], errors="coerce"))
                    elif is_parallel_two_terminal:
                        length = float(ll.mean())
                    else:
                        length = float(ll.sum())

            if not np.isfinite(length) and ac0 in bus_coords.index and ac1 in bus_coords.index:
                length = float(
                    _haversine_array(
                        [bus_coords.at[ac0, "lat"]],
                        [bus_coords.at[ac0, "lon"]],
                        [bus_coords.at[ac1, "lat"]],
                        [bus_coords.at[ac1, "lon"]],
                    )[0]
                ) * float(fallback_length_factor)

            voltage = np.nan
            if "voltage" in Lc.columns:
                vv = pd.to_numeric(Lc["voltage"], errors="coerce")
                vv = vv[np.isfinite(vv)]
                if len(vv):
                    voltage = float(vv.max())

            new_links.append(
                {
                    "link_id": f"hvdc_ac_{new_id}",
                    "bus0": ac0,
                    "bus1": ac1,
                    "p_nom": p_nom,
                    "voltage": voltage,
                    "length": length,
                    "carrier": "HVDC",
                    "link_model": "collapsed_hvdc_ac",
                    "collapse_case": "dc_dc_plus_2conv" if is_parallel_two_terminal else "dc_path_plus_converters",
                    "n_dc_links": int(len(Lc)),
                    "n_converters": int(len(Cc)),
                    "grid_source": _unique_sorted_join(Lc["grid_source"]) if "grid_source" in Lc.columns else pd.NA,
                }
            )
            new_id += 1
            used_link_idx.update(Lc.index.tolist())
            used_conv_idx.update(Cc.index.tolist())

    # ---------------------------------------------------------
    # Case B/C: mixed AC-DC link + 1 converter
    # ---------------------------------------------------------
    L_mixed = L_all.loc[L_all["bus0_is_dc"] ^ L_all["bus1_is_dc"]].copy()

    if not L_mixed.empty and not C_all.empty:
        for idx, row in L_mixed.iterrows():
            if idx in used_link_idx:
                continue

            if row["bus0_is_dc"] and (not row["bus1_is_dc"]):
                dc_bus = str(row["bus0"])
                ac_bus_direct = str(row["bus1"])
            elif row["bus1_is_dc"] and (not row["bus0_is_dc"]):
                dc_bus = str(row["bus1"])
                ac_bus_direct = str(row["bus0"])
            else:
                continue

            Cc = C_all.loc[C_all["dc_bus"].astype(str).eq(dc_bus)].copy()

            # Only exactly one matching converter -> classic edge case
            if len(Cc) != 1:
                continue

            c_row = Cc.iloc[0]
            ac_bus_other = str(c_row["ac_bus"])

            if ac_bus_direct == ac_bus_other:
                continue

            bus0, bus1 = sorted([ac_bus_direct, ac_bus_other])

            caps = []
            if "p_nom" in row.index:
                x = pd.to_numeric(pd.Series([row["p_nom"]]), errors="coerce").fillna(0.0).iloc[0]
                if x > 0:
                    caps.append(float(x))
            if "p_nom" in c_row.index:
                x = pd.to_numeric(pd.Series([c_row["p_nom"]]), errors="coerce").fillna(0.0).iloc[0]
                if x > 0:
                    caps.append(float(x))
            p_nom = min(caps) if caps else np.nan

            length = np.nan
            if "length" in row.index:
                x = pd.to_numeric(pd.Series([row["length"]]), errors="coerce").iloc[0]
                if np.isfinite(x) and x > 0:
                    length = float(x)

            if not np.isfinite(length) and bus0 in bus_coords.index and bus1 in bus_coords.index:
                length = float(
                    _haversine_array(
                        [bus_coords.at[bus0, "lat"]],
                        [bus_coords.at[bus0, "lon"]],
                        [bus_coords.at[bus1, "lat"]],
                        [bus_coords.at[bus1, "lon"]],
                    )[0]
                ) * float(fallback_length_factor)

            voltage = np.nan
            vals = []
            if "voltage" in row.index:
                vals.append(pd.to_numeric(pd.Series([row["voltage"]]), errors="coerce").iloc[0])
            if "voltage" in c_row.index:
                vals.append(pd.to_numeric(pd.Series([c_row["voltage"]]), errors="coerce").iloc[0])
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                voltage = float(max(vals))

            new_links.append(
                {
                    "link_id": f"hvdc_ac_{new_id}",
                    "bus0": bus0,
                    "bus1": bus1,
                    "p_nom": p_nom,
                    "voltage": voltage,
                    "length": length,
                    "carrier": "HVDC",
                    "link_model": "collapsed_hvdc_ac",
                    "collapse_case": "mixed_link_plus_1conv",
                    "n_dc_links": 1,
                    "n_converters": 1,
                    "grid_source": str(row["grid_source"]) if "grid_source" in row.index and pd.notna(row["grid_source"]) else pd.NA,
                }
            )
            new_id += 1
            used_link_idx.add(idx)
            used_conv_idx.add(Cc.index[0])

    L_keep = links_red.drop(index=list(used_link_idx), errors="ignore").copy()
    C_keep = conv_red.drop(index=list(used_conv_idx), errors="ignore").copy()

    if not L_keep.empty:
        if "link_model" not in L_keep.columns:
            L_keep["link_model"] = "explicit_dc"
        else:
            L_keep["link_model"] = L_keep["link_model"].fillna("explicit_dc")

    new_links_df = pd.DataFrame(new_links)
    out_links = pd.concat([L_keep, new_links_df], ignore_index=True, sort=False)

    return out_links.reset_index(drop=True), C_keep.reset_index(drop=True)


def _prune_one_sided_dc_components(
    links_red: pd.DataFrame,
    conv_red: pd.DataFrame,
    buses_red: pd.DataFrame,
    plants_red: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove residual explicit DC components that do not connect two distinct AC buses.

    These are one-sided stubs or internal DC remnants with at most one unique AC
    endpoint in the reduced model. Multi-terminal structures with two or more AC
    endpoints are kept explicit.
    """

    if buses_red.empty or (links_red.empty and conv_red.empty):
        return links_red.reset_index(drop=True), conv_red.reset_index(drop=True)

    B = buses_red.copy()
    B["bus_id"] = B["bus_id"].astype(str)

    if "dc" in B.columns:
        B["is_dc"] = B["dc"].astype(str).str.lower().isin(["t", "true", "1", "yes", "y"])
    elif "dc_bool" in B.columns:
        B["is_dc"] = B["dc_bool"].astype(bool)
    else:
        B["is_dc"] = False

    dc_nodes = set(B.loc[B["is_dc"], "bus_id"].astype(str))
    if not dc_nodes:
        return links_red.reset_index(drop=True), conv_red.reset_index(drop=True)

    L_all = links_red.copy()
    if not L_all.empty:
        L_all["bus0"] = L_all["bus0"].astype(str)
        L_all["bus1"] = L_all["bus1"].astype(str)
        L_all["bus0_is_dc"] = L_all["bus0"].isin(dc_nodes)
        L_all["bus1_is_dc"] = L_all["bus1"].isin(dc_nodes)
    else:
        L_all = pd.DataFrame(columns=["bus0", "bus1", "bus0_is_dc", "bus1_is_dc"])

    C_all = conv_red.copy()
    if not C_all.empty:
        C_all["bus0"] = C_all["bus0"].astype(str)
        C_all["bus1"] = C_all["bus1"].astype(str)
        c0_dc = C_all["bus0"].isin(dc_nodes)
        c1_dc = C_all["bus1"].isin(dc_nodes)
        mask_valid = c0_dc != c1_dc
        C_all = C_all.loc[mask_valid].copy()
        c0_dc = c0_dc.loc[mask_valid]
        C_all["dc_bus"] = np.where(c0_dc.values, C_all["bus0"].values, C_all["bus1"].values)
        C_all["ac_bus"] = np.where(c0_dc.values, C_all["bus1"].values, C_all["bus0"].values)
    else:
        C_all = pd.DataFrame(columns=["dc_bus", "ac_bus"])

    Gdc = nx.Graph()
    Gdc.add_nodes_from(dc_nodes)

    if not L_all.empty:
        L_dd = L_all.loc[L_all["bus0_is_dc"] & L_all["bus1_is_dc"], ["bus0", "bus1"]].copy()
        if not L_dd.empty:
            Gdc.add_edges_from(L_dd.itertuples(index=False, name=None))

        L_mixed = L_all.loc[L_all["bus0_is_dc"] ^ L_all["bus1_is_dc"]].copy()
        if not L_mixed.empty:
            dc_mixed_nodes = np.where(L_mixed["bus0_is_dc"].values, L_mixed["bus0"].values, L_mixed["bus1"].values)
            Gdc.add_nodes_from(dc_mixed_nodes.tolist())

    if not C_all.empty:
        Gdc.add_nodes_from(C_all["dc_bus"].astype(str).tolist())

    plants_on_dc = set()
    if not plants_red.empty and "bus_id" in plants_red.columns:
        plants_on_dc = set(plants_red["bus_id"].astype(str)).intersection(dc_nodes)

    prune_link_idx = set()
    prune_conv_idx = set()

    for comp in nx.connected_components(Gdc):
        comp = set(str(x) for x in comp)
        if not comp:
            continue

        if plants_on_dc.intersection(comp):
            continue

        Cc = C_all.loc[C_all["dc_bus"].isin(comp)].copy()
        Lc = L_all.loc[
            (L_all["bus0"].isin(comp) | L_all["bus1"].isin(comp))
            & (L_all["bus0_is_dc"] | L_all["bus1_is_dc"])
        ].copy()

        ac_endpoints = set(Cc["ac_bus"].astype(str).tolist())
        if not Lc.empty:
            mixed = Lc.loc[Lc["bus0_is_dc"] ^ Lc["bus1_is_dc"]].copy()
            if not mixed.empty:
                ac_mixed = np.where(mixed["bus0_is_dc"].values, mixed["bus1"].values, mixed["bus0"].values)
                ac_endpoints.update(str(x) for x in ac_mixed)

        if len(ac_endpoints) > 1:
            continue

        prune_link_idx.update(Lc.index.tolist())
        prune_conv_idx.update(Cc.index.tolist())

    L_keep = links_red.drop(index=list(prune_link_idx), errors="ignore").copy()
    C_keep = conv_red.drop(index=list(prune_conv_idx), errors="ignore").copy()
    return L_keep.reset_index(drop=True), C_keep.reset_index(drop=True)


def _representative_cluster_positions(
    buses_clustered: pd.DataFrame,
    *,
    mode: str = "mean",
) -> pd.DataFrame:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"mean", "medoid"}:
        raise ValueError("plot_bus_position_mode must be 'mean' or 'medoid'")

    B = buses_clustered[["cluster_id", "lat", "lon"]].copy()
    B["cluster_id"] = B["cluster_id"].astype(str)
    B["lat"] = pd.to_numeric(B["lat"], errors="coerce")
    B["lon"] = pd.to_numeric(B["lon"], errors="coerce")

    rows = []
    for cluster_id, bg in B.groupby("cluster_id", sort=False):
        coords = bg[["lat", "lon"]].dropna().to_numpy(dtype=float)
        if len(coords) == 0:
            rows.append({"cluster_id": cluster_id, "lat": np.nan, "lon": np.nan})
            continue

        if mode_norm == "mean" or len(coords) == 1:
            lat, lon = float(coords[:, 0].mean()), float(coords[:, 1].mean())
        else:
            lat_rad = np.radians(coords[:, 0])[:, None]
            lon_rad = np.radians(coords[:, 1])[:, None]
            dlat = lat_rad - lat_rad.T
            dlon = lon_rad - lon_rad.T
            a = (
                np.sin(dlat / 2.0) ** 2
                + np.cos(lat_rad) * np.cos(lat_rad.T) * np.sin(dlon / 2.0) ** 2
            )
            a = np.clip(a, 0.0, 1.0)
            dist = 6371.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
            idx = int(np.argmin(dist.sum(axis=1)))
            lat, lon = float(coords[idx, 0]), float(coords[idx, 1])

        rows.append({"cluster_id": cluster_id, "lat": lat, "lon": lon})

    return pd.DataFrame(rows)


def _contract_tiny_ac_leaf_nodes(
    buses_red: pd.DataFrame,
    lines_red: pd.DataFrame,
    links_red: pd.DataFrame,
    conv_red: pd.DataFrame,
    tr_red: pd.DataFrame,
    plants_red: pd.DataFrame,
    *,
    merge_mode: str = "same_country",
    max_member_buses: int = 2,
    max_total_capacity_mw: float = 300.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Contract very small reduced AC leaf nodes into their only AC neighbor.

    This is intentionally conservative:
      - only AC nodes
      - only one AC neighbor via lines/transformers
      - no incident links/converters
      - same sync area and sub-network as the neighbor
      - either same-country or cross-border, depending on merge_mode
      - small original member count and small total generation capacity
    """

    merge_mode_norm = str(merge_mode).strip().lower()
    if merge_mode_norm not in {"same_country", "cross_border"}:
        raise ValueError("merge_mode must be 'same_country' or 'cross_border'")

    if buses_red.empty:
        return buses_red, lines_red, links_red, conv_red, tr_red, plants_red

    B = buses_red.copy()
    B["bus_id"] = B["bus_id"].astype(str)
    if "dc" in B.columns:
        B["dc_bool"] = B["dc"].astype(str).str.lower().isin(["t", "true", "1", "yes", "y"])
    elif "dc_bool" in B.columns:
        B["dc_bool"] = B["dc_bool"].astype(bool)
    else:
        B["dc_bool"] = False

    B["n_buses_simplified"] = pd.to_numeric(B.get("n_buses_simplified", 0), errors="coerce").fillna(0).astype(int)
    bus_meta = B.set_index("bus_id")
    ac_bus_ids = set(B.loc[~B["dc_bool"], "bus_id"].astype(str))
    country_counts = (
        B.loc[~B["dc_bool"]]
        .assign(country=B["country"].astype(str))
        .groupby("country")["bus_id"]
        .nunique()
        .to_dict()
    )

    plant_cap = pd.Series(dtype=float)
    if not plants_red.empty and "bus_id" in plants_red.columns and "Capacity" in plants_red.columns:
        plant_cap = (
            plants_red.assign(
                bus_id=plants_red["bus_id"].astype(str),
                Capacity=pd.to_numeric(plants_red["Capacity"], errors="coerce").fillna(0.0),
            )
            .groupby("bus_id")["Capacity"]
            .sum()
        )

    neighbors: dict[str, set[str]] = {bus_id: set() for bus_id in ac_bus_ids}
    incident_branch_count: dict[str, int] = {bus_id: 0 for bus_id in ac_bus_ids}

    def _add_ac_edge(u: str, v: str) -> None:
        if u == v or u not in ac_bus_ids or v not in ac_bus_ids:
            return
        neighbors[u].add(v)
        neighbors[v].add(u)
        incident_branch_count[u] += 1
        incident_branch_count[v] += 1

    if not lines_red.empty:
        b0 = "bus0" if "bus0" in lines_red.columns else "u"
        b1 = "bus1" if "bus1" in lines_red.columns else "v"
        for u, v in lines_red[[b0, b1]].astype(str).itertuples(index=False, name=None):
            _add_ac_edge(str(u), str(v))

    if not tr_red.empty:
        b0 = "bus0_red" if "bus0_red" in tr_red.columns else "bus0"
        b1 = "bus1_red" if "bus1_red" in tr_red.columns else "bus1"
        for u, v in tr_red[[b0, b1]].astype(str).itertuples(index=False, name=None):
            _add_ac_edge(str(u), str(v))

    blocked_buses = set()
    if not links_red.empty:
        blocked_buses.update(links_red["bus0"].astype(str).tolist())
        blocked_buses.update(links_red["bus1"].astype(str).tolist())
    if not conv_red.empty:
        blocked_buses.update(conv_red["bus0"].astype(str).tolist())
        blocked_buses.update(conv_red["bus1"].astype(str).tolist())

    prelim_map: dict[str, str] = {}
    planned_country_merges: defaultdict[str, int] = defaultdict(int)
    for bus_id in ac_bus_ids:
        if bus_id not in bus_meta.index:
            continue
        if bus_id in blocked_buses:
            continue
        if int(bus_meta.at[bus_id, "n_buses_simplified"]) > int(max_member_buses):
            continue
        if float(plant_cap.get(bus_id, 0.0)) > float(max_total_capacity_mw):
            continue
        if incident_branch_count.get(bus_id, 0) != 1:
            continue
        neigh = sorted(neighbors.get(bus_id, set()))
        if len(neigh) != 1:
            continue
        dst = neigh[0]
        if dst not in bus_meta.index:
            continue
        if bool(bus_meta.at[dst, "dc_bool"]):
            continue
        same_country = str(bus_meta.at[bus_id, "country"]) == str(bus_meta.at[dst, "country"])
        same_sync = str(bus_meta.at[bus_id, "sync_area"]) == str(bus_meta.at[dst, "sync_area"])
        same_subnet = str(bus_meta.at[bus_id, "sub_network"]) == str(bus_meta.at[dst, "sub_network"])
        if not (same_sync and same_subnet):
            continue
        if merge_mode_norm == "same_country":
            if not same_country:
                continue
        else:
            if same_country:
                continue
            src_country = str(bus_meta.at[bus_id, "country"])
            if int(country_counts.get(src_country, 0)) - int(planned_country_merges[src_country]) <= 1:
                continue
        prelim_map[bus_id] = dst
        if merge_mode_norm == "cross_border":
            planned_country_merges[str(bus_meta.at[bus_id, "country"])] += 1

    merge_map = {
        src: dst
        for src, dst in prelim_map.items()
        if dst not in prelim_map
    }
    if not merge_map:
        return buses_red, lines_red, links_red, conv_red, tr_red, plants_red

    buses_out = B.copy()
    for src, dst in merge_map.items():
        if src not in buses_out["bus_id"].values or dst not in buses_out["bus_id"].values:
            continue
        src_idx = buses_out.index[buses_out["bus_id"].astype(str).eq(src)]
        dst_idx = buses_out.index[buses_out["bus_id"].astype(str).eq(dst)]
        if len(src_idx) != 1 or len(dst_idx) != 1:
            continue
        src_idx = src_idx[0]
        dst_idx = dst_idx[0]
        buses_out.at[dst_idx, "voltage"] = max(
            pd.to_numeric(pd.Series([buses_out.at[dst_idx, "voltage"]]), errors="coerce").fillna(0.0).iloc[0],
            pd.to_numeric(pd.Series([buses_out.at[src_idx, "voltage"]]), errors="coerce").fillna(0.0).iloc[0],
        )
        buses_out.at[dst_idx, "voltage_min"] = min(
            pd.to_numeric(pd.Series([buses_out.at[dst_idx, "voltage_min"]]), errors="coerce").fillna(np.inf).iloc[0],
            pd.to_numeric(pd.Series([buses_out.at[src_idx, "voltage_min"]]), errors="coerce").fillna(np.inf).iloc[0],
        )
        buses_out.at[dst_idx, "voltage_max"] = max(
            pd.to_numeric(pd.Series([buses_out.at[dst_idx, "voltage_max"]]), errors="coerce").fillna(0.0).iloc[0],
            pd.to_numeric(pd.Series([buses_out.at[src_idx, "voltage_max"]]), errors="coerce").fillna(0.0).iloc[0],
        )
        buses_out.at[dst_idx, "n_voltage_levels"] = max(
            int(pd.to_numeric(pd.Series([buses_out.at[dst_idx, "n_voltage_levels"]]), errors="coerce").fillna(1).iloc[0]),
            int(pd.to_numeric(pd.Series([buses_out.at[src_idx, "n_voltage_levels"]]), errors="coerce").fillna(1).iloc[0]),
        )
        buses_out.at[dst_idx, "n_buses_simplified"] = int(buses_out.at[dst_idx, "n_buses_simplified"]) + int(
            buses_out.at[src_idx, "n_buses_simplified"]
        )
        if "original_country" in buses_out.columns:
            src_orig = pd.Series([buses_out.at[src_idx, "original_country"]], dtype="object")
            dst_orig = pd.Series([buses_out.at[dst_idx, "original_country"]], dtype="object")
            buses_out.at[dst_idx, "original_country"] = _unique_sorted_join(pd.concat([dst_orig, src_orig], ignore_index=True))
        if "n_original_countries" in buses_out.columns and "original_country" in buses_out.columns:
            orig_vals = [
                x
                for x in str(buses_out.at[dst_idx, "original_country"]).split(",")
                if x and x.lower() != "nan"
            ]
            buses_out.at[dst_idx, "n_original_countries"] = len(sorted(set(orig_vals)))

    buses_out = buses_out.loc[~buses_out["bus_id"].astype(str).isin(merge_map.keys())].copy().reset_index(drop=True)

    def _map_bus_id(value: Any) -> str:
        key = str(value)
        return merge_map.get(key, key)

    lines_out = lines_red.copy()
    if not lines_out.empty:
        b0 = "bus0" if "bus0" in lines_out.columns else "u"
        b1 = "bus1" if "bus1" in lines_out.columns else "v"
        lines_out[b0] = lines_out[b0].astype(str).map(_map_bus_id)
        lines_out[b1] = lines_out[b1].astype(str).map(_map_bus_id)
        lines_out = lines_out.loc[lines_out[b0] != lines_out[b1]].copy().reset_index(drop=True)

    links_out = links_red.copy()
    if not links_out.empty:
        links_out["bus0"] = links_out["bus0"].astype(str).map(_map_bus_id)
        links_out["bus1"] = links_out["bus1"].astype(str).map(_map_bus_id)
        links_out = links_out.loc[links_out["bus0"] != links_out["bus1"]].copy().reset_index(drop=True)

    conv_out = conv_red.copy()
    if not conv_out.empty:
        conv_out["bus0"] = conv_out["bus0"].astype(str).map(_map_bus_id)
        conv_out["bus1"] = conv_out["bus1"].astype(str).map(_map_bus_id)
        conv_out = conv_out.loc[conv_out["bus0"] != conv_out["bus1"]].copy().reset_index(drop=True)

    tr_out = tr_red.copy()
    if not tr_out.empty:
        b0 = "bus0_red" if "bus0_red" in tr_out.columns else "bus0"
        b1 = "bus1_red" if "bus1_red" in tr_out.columns else "bus1"
        tr_out[b0] = tr_out[b0].astype(str).map(_map_bus_id)
        tr_out[b1] = tr_out[b1].astype(str).map(_map_bus_id)
        tr_out = tr_out.loc[tr_out[b0] != tr_out[b1]].copy().reset_index(drop=True)

    plants_out = plants_red.copy()
    if not plants_out.empty and "bus_id" in plants_out.columns:
        plants_out["bus_id"] = plants_out["bus_id"].astype(str).map(_map_bus_id)
        bus_country_out = buses_out.set_index("bus_id")["country"].to_dict()
        bus_country_label_out = buses_out.set_index("bus_id")["country_label"].to_dict()
        if "country" in plants_out.columns:
            plants_out["country"] = plants_out["bus_id"].astype(str).map(bus_country_out)
        if "country_label" in plants_out.columns:
            plants_out["country_label"] = plants_out["bus_id"].astype(str).map(bus_country_label_out)

        group_cols = [
            col
            for col in ["bus_id", "country", "country_label", "original_country", "Fueltype", "Technology", "Set"]
            if col in plants_out.columns
        ]
        agg_cols = {}
        if "Capacity" in plants_out.columns:
            agg_cols["Capacity"] = ("Capacity", "sum")
        if "n_plants" in plants_out.columns:
            agg_cols["n_plants"] = ("n_plants", "sum")
        plants_out = plants_out.groupby(group_cols, as_index=False).agg(**agg_cols)

    return buses_out, lines_out, links_out, conv_out, tr_out, plants_out


def _pre_prune_tiny_ac_leaf_nodes(
    buses_w: pd.DataFrame,
    lines_w: pd.DataFrame,
    links_w: pd.DataFrame,
    conv_w: pd.DataFrame,
    trafo_w: pd.DataFrame,
    plants_w: pd.DataFrame,
    busmap_precluster: pd.Series,
    *,
    merge_mode: str = "same_country",
    plant_bus_col: str = "assigned_bus_380",
    member_count_col: str = "n_buses_merged",
    max_member_buses: int = 2,
    max_total_capacity_mw: float = 300.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Contract tiny AC leaf buses before k-allocation and clustering.

    This changes the candidate clustering topology itself, so removed leaf buses do
    not consume any k-budget. The logic is intentionally conservative and mirrors
    the reduced-network leaf pruning as closely as possible.
    """

    merge_mode_norm = str(merge_mode).strip().lower()
    if merge_mode_norm not in {"same_country", "cross_border"}:
        raise ValueError("merge_mode must be 'same_country' or 'cross_border'")

    if buses_w.empty:
        return buses_w, lines_w, links_w, conv_w, trafo_w, plants_w, busmap_precluster

    B = buses_w.copy()
    B["bus_id"] = B["bus_id"].astype(str)
    if "dc" in B.columns:
        B["dc_bool"] = B["dc"].astype(str).str.lower().isin(["t", "true", "1", "yes", "y"])
    elif "dc_bool" in B.columns:
        B["dc_bool"] = B["dc_bool"].astype(bool)
    else:
        B["dc_bool"] = False

    if member_count_col in B.columns:
        B[member_count_col] = pd.to_numeric(B[member_count_col], errors="coerce").fillna(1).astype(int)
    else:
        B[member_count_col] = 1

    bus_meta = B.set_index("bus_id")
    ac_bus_ids = set(B.loc[~B["dc_bool"], "bus_id"].astype(str))
    country_counts = (
        B.loc[~B["dc_bool"]]
        .assign(country=B.loc[~B["dc_bool"], "country"].astype(str))
        .groupby("country")["bus_id"]
        .nunique()
        .to_dict()
    )

    plant_cap = pd.Series(dtype=float)
    if (not plants_w.empty) and (plant_bus_col in plants_w.columns):
        plant_cap = (
            plants_w.assign(
                _plant_bus=plants_w[plant_bus_col].astype(str),
                Capacity=pd.to_numeric(plants_w.get("Capacity", 0.0), errors="coerce").fillna(0.0),
            )
            .groupby("_plant_bus")["Capacity"]
            .sum()
        )

    neighbors: dict[str, set[str]] = {bus_id: set() for bus_id in ac_bus_ids}
    incident_branch_count: dict[str, int] = {bus_id: 0 for bus_id in ac_bus_ids}

    def _add_ac_edge(u: str, v: str) -> None:
        if u == v or u not in ac_bus_ids or v not in ac_bus_ids:
            return
        neighbors[u].add(v)
        neighbors[v].add(u)
        incident_branch_count[u] += 1
        incident_branch_count[v] += 1

    if not lines_w.empty:
        for u, v in lines_w[["bus0", "bus1"]].astype(str).itertuples(index=False, name=None):
            _add_ac_edge(str(u), str(v))

    if not trafo_w.empty:
        for u, v in trafo_w[["bus0", "bus1"]].astype(str).itertuples(index=False, name=None):
            _add_ac_edge(str(u), str(v))

    blocked_buses = set()
    if not links_w.empty:
        blocked_buses.update(links_w["bus0"].astype(str).tolist())
        blocked_buses.update(links_w["bus1"].astype(str).tolist())
    if not conv_w.empty:
        blocked_buses.update(conv_w["bus0"].astype(str).tolist())
        blocked_buses.update(conv_w["bus1"].astype(str).tolist())

    prelim_map: dict[str, str] = {}
    planned_country_merges: defaultdict[str, int] = defaultdict(int)
    for bus_id in ac_bus_ids:
        if bus_id not in bus_meta.index:
            continue
        if bus_id in blocked_buses:
            continue
        if int(bus_meta.at[bus_id, member_count_col]) > int(max_member_buses):
            continue
        if float(plant_cap.get(bus_id, 0.0)) > float(max_total_capacity_mw):
            continue
        if incident_branch_count.get(bus_id, 0) != 1:
            continue
        neigh = sorted(neighbors.get(bus_id, set()))
        if len(neigh) != 1:
            continue
        dst = neigh[0]
        if dst not in bus_meta.index:
            continue
        if bool(bus_meta.at[dst, "dc_bool"]):
            continue

        same_country = str(bus_meta.at[bus_id, "country"]) == str(bus_meta.at[dst, "country"])
        same_sync = str(bus_meta.at[bus_id, "sync_area"]) == str(bus_meta.at[dst, "sync_area"])
        same_subnet = str(bus_meta.at[bus_id, "sub_network"]) == str(bus_meta.at[dst, "sub_network"])
        if not (same_sync and same_subnet):
            continue

        if merge_mode_norm == "same_country":
            if not same_country:
                continue
        else:
            if same_country:
                continue
            src_country = str(bus_meta.at[bus_id, "country"])
            if int(country_counts.get(src_country, 0)) - int(planned_country_merges[src_country]) <= 1:
                continue

        prelim_map[bus_id] = dst
        if merge_mode_norm == "cross_border":
            planned_country_merges[str(bus_meta.at[bus_id, "country"])] += 1

    merge_map = {
        src: dst
        for src, dst in prelim_map.items()
        if dst not in prelim_map
    }
    if not merge_map:
        return buses_w, lines_w, links_w, conv_w, trafo_w, plants_w, busmap_precluster

    B_out = B.copy()
    for src, dst in merge_map.items():
        if src not in B_out["bus_id"].values or dst not in B_out["bus_id"].values:
            continue
        src_idx = B_out.index[B_out["bus_id"].astype(str).eq(src)]
        dst_idx = B_out.index[B_out["bus_id"].astype(str).eq(dst)]
        if len(src_idx) != 1 or len(dst_idx) != 1:
            continue
        src_idx = src_idx[0]
        dst_idx = dst_idx[0]

        if member_count_col in B_out.columns:
            B_out.at[dst_idx, member_count_col] = int(B_out.at[dst_idx, member_count_col]) + int(B_out.at[src_idx, member_count_col])
        for col in ["voltage", "voltage_max_orig"]:
            if col in B_out.columns:
                B_out.at[dst_idx, col] = max(
                    pd.to_numeric(pd.Series([B_out.at[dst_idx, col]]), errors="coerce").fillna(0.0).iloc[0],
                    pd.to_numeric(pd.Series([B_out.at[src_idx, col]]), errors="coerce").fillna(0.0).iloc[0],
                )
        for col in ["voltage_min_orig"]:
            if col in B_out.columns:
                B_out.at[dst_idx, col] = min(
                    pd.to_numeric(pd.Series([B_out.at[dst_idx, col]]), errors="coerce").fillna(np.inf).iloc[0],
                    pd.to_numeric(pd.Series([B_out.at[src_idx, col]]), errors="coerce").fillna(np.inf).iloc[0],
                )
        if "country_original_raw" in B_out.columns:
            src_orig = pd.Series([B_out.at[src_idx, "country_original_raw"]], dtype="object")
            dst_orig = pd.Series([B_out.at[dst_idx, "country_original_raw"]], dtype="object")
            B_out.at[dst_idx, "country_original_raw"] = _unique_sorted_join(pd.concat([dst_orig, src_orig], ignore_index=True))

    B_out = B_out.loc[~B_out["bus_id"].astype(str).isin(merge_map.keys())].copy().reset_index(drop=True)

    def _map_bus(value: Any) -> str:
        key = str(value)
        return merge_map.get(key, key)

    L_out = lines_w.copy()
    if not L_out.empty:
        L_out["bus0"] = L_out["bus0"].astype(str).map(_map_bus)
        L_out["bus1"] = L_out["bus1"].astype(str).map(_map_bus)
        L_out = L_out.loc[L_out["bus0"] != L_out["bus1"]].copy().reset_index(drop=True)

    LK_out = links_w.copy()
    if not LK_out.empty:
        LK_out["bus0"] = LK_out["bus0"].astype(str).map(_map_bus)
        LK_out["bus1"] = LK_out["bus1"].astype(str).map(_map_bus)
        LK_out = LK_out.loc[LK_out["bus0"] != LK_out["bus1"]].copy().reset_index(drop=True)

    CV_out = conv_w.copy()
    if not CV_out.empty:
        CV_out["bus0"] = CV_out["bus0"].astype(str).map(_map_bus)
        CV_out["bus1"] = CV_out["bus1"].astype(str).map(_map_bus)
        CV_out = CV_out.loc[CV_out["bus0"] != CV_out["bus1"]].copy().reset_index(drop=True)

    T_out = trafo_w.copy()
    if not T_out.empty:
        T_out["bus0"] = T_out["bus0"].astype(str).map(_map_bus)
        T_out["bus1"] = T_out["bus1"].astype(str).map(_map_bus)
        T_out = T_out.loc[T_out["bus0"] != T_out["bus1"]].copy().reset_index(drop=True)

    P_out = plants_w.copy()
    if (not P_out.empty) and (plant_bus_col in P_out.columns):
        P_out[plant_bus_col] = P_out[plant_bus_col].astype(str).map(_map_bus)

    busmap_df = pd.DataFrame(
        {
            "orig_bus_id": pd.Index(pd.Series(busmap_precluster.index, dtype="object").astype(str).values, dtype="object"),
            "mapped_bus_id": pd.Series(busmap_precluster, copy=False, dtype="object").astype(str).map(_map_bus).values,
        }
    )
    busmap_out = pd.Series(
        busmap_df["mapped_bus_id"].to_numpy(),
        index=pd.Index(busmap_df["orig_bus_id"].to_numpy(), dtype="object"),
        name=busmap_precluster.name,
        dtype="object",
    )

    return B_out, L_out, LK_out, CV_out, T_out, P_out, busmap_out


def _prune_unused_buses(
    buses_red: pd.DataFrame,
    *,
    lines_red: pd.DataFrame,
    links_red: pd.DataFrame,
    conv_red: pd.DataFrame,
    tr_red: pd.DataFrame,
    plants_red: pd.DataFrame,
) -> pd.DataFrame:
    used = set()

    if not lines_red.empty:
        b0 = "bus0" if "bus0" in lines_red.columns else "u"
        b1 = "bus1" if "bus1" in lines_red.columns else "v"
        used.update(lines_red[b0].astype(str).tolist())
        used.update(lines_red[b1].astype(str).tolist())

    if not links_red.empty:
        used.update(links_red["bus0"].astype(str).tolist())
        used.update(links_red["bus1"].astype(str).tolist())

    if not conv_red.empty:
        used.update(conv_red["bus0"].astype(str).tolist())
        used.update(conv_red["bus1"].astype(str).tolist())

    if not tr_red.empty:
        b0 = "bus0_red" if "bus0_red" in tr_red.columns else "bus0"
        b1 = "bus1_red" if "bus1_red" in tr_red.columns else "bus1"
        used.update(tr_red[b0].astype(str).tolist())
        used.update(tr_red[b1].astype(str).tolist())

    if not plants_red.empty:
        used.update(plants_red["bus_id"].astype(str).tolist())

    out = buses_red.copy()
    out["bus_id"] = out["bus_id"].astype(str)
    out = out.loc[out["bus_id"].isin(used)].copy().reset_index(drop=True)
    return out


# ============================================================
# Full workflow
# ============================================================
def reduce_network(
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    links: pd.DataFrame,
    converters: pd.DataFrame,
    transformers: pd.DataFrame,
    plants_assigned: pd.DataFrame,
    *,
    k_total: int = 256,
    method: str = "electrical_spectral",   # kmeans_geo | electrical_spectral | hac | maximum_spanning_tree
    sync_collapse: bool = True,
    voltage_mode: str = "simplify_380",    # standard | simplify_380 | per_unit
    s_base_mva: float = 100.0,             # voltage_mode="per_unit"
    similarity: str = "electrical",        # electrical | geographical | dc_effective_reactance
    electrical_weight_mode: str = "b",     # b | b*s_nom
    hac_feature: str = "coords",           # coords | spectral
    hac_n_embed: int = 4,
    aggregation_mode: str = "line_equivalent",   # line_equivalent | location_based
    line_length_factor: float = 1.0,
    effrx_k_nearest: int | None = 12,
    effrx_kernel: str = "exp",
    effrx_tau_scale: float = 1.0,
    hvdc_model: str = "pypsa_ac_links",    # explicit_dc | pypsa_ac_links
    country_load_mean: pd.Series | None = None,
    generation_country_weight: float = 1.0,
    load_country_weight: float = 0.0,
    cesa_country_clusters: pd.DataFrame | None = None,
    plot_bus_position_mode: str = "medoid",
    pre_prune_tiny_ac_leaf_nodes: bool = True,
    pre_prune_tiny_cross_border_leaf_nodes: bool = True,
    prune_tiny_ac_leaf_nodes: bool = True,
    prune_tiny_cross_border_leaf_nodes: bool = True,
    tiny_ac_leaf_max_member_buses: int = 2,
    tiny_ac_leaf_max_capacity_mw: float = 300.0,
) -> dict[str, pd.DataFrame]:

    if voltage_mode not in {"standard", "simplify_380", "per_unit"}:
        raise ValueError("voltage_mode must be 'standard', 'simplify_380' or 'per_unit'")

    do_simplify_380 = (voltage_mode == "simplify_380")
    cesa_country_clusters_n = _normalize_cesa_country_clusters(cesa_country_clusters)

    if cesa_country_clusters_n.empty:
        source_to_target = pd.Series(dtype="object")
        target_to_label = pd.Series(dtype="object")
    else:
        source_to_target = (
            cesa_country_clusters_n.drop_duplicates(subset=["source_country"])
            .set_index("source_country")["target_country"]
            .astype(str)
        )
        target_to_label = (
            cesa_country_clusters_n.drop_duplicates(subset=["target_country"])
            .set_index("target_country")["target_label"]
            .astype(str)
        )

    # Normalise the electrical input first, then apply country aggregates. The
    # reduced buses therefore inherit a clean AC topology and a traceable mapping
    # back to the source countries used for data regionalisation.
    buses0 = add_sync_area_to_buses(_prep_buses(buses))
    lines0 = impute_zero_s_nom_by_country_voltage(_prep_lines(lines), buses0)
    links0 = _prep_edge_table(links, "link_id")
    conv0 = _prep_edge_table(converters, "converter_id")
    trafo0 = _prep_transformers(transformers)

    P0 = _drop_existing_plant_bus_metadata(plants_assigned.copy())
    P0["assigned_bus"] = P0["assigned_bus"].astype(str)
    P0 = P0.merge(
        buses0[["bus_id", "sync_area", "sync_node"]],
        left_on="assigned_bus",
        right_on="bus_id",
        how="left",
        suffixes=("", "_bus"),
    ).drop(columns=["bus_id"])

    if do_simplify_380:
        simp = simplify_network_to_380_equivalent(
            buses0, lines0, links0, conv0, trafo0, P0.rename(columns={"assigned_bus": "assigned_bus"})
        )
        buses_w = simp["buses"].copy()
        #lines_w = impute_zero_s_nom_by_country_voltage(_prep_lines(simp["lines"]), buses_w)
        lines_tmp = _prep_lines(simp["lines"]).drop(
            columns=["country0", "country1", "line_country", "s_nom_original", "s_nom_imputed", "s_nom_impute_source"],
            errors="ignore")
        lines_w = impute_zero_s_nom_by_country_voltage(lines_tmp, buses_w)
        links_w = _prep_edge_table(simp["links"], "link_id")
        conv_w = _prep_edge_table(simp["converters"], "converter_id")
        trafo_w = _prep_transformers(simp["transformers"])
        plants_w = simp["plants"].copy()
        busmap_precluster = simp["busmap_380"].copy()
    else:
        buses_w = buses0.copy()
        buses_w["sub_network"] = np.where(
            buses_w["sync_area"].eq("CESA") & (~buses_w["dc_bool"]),
            "sn_0",
            np.nan,
        )
        lines_w = lines0.copy()
        links_w = links0.copy()
        conv_w = conv0.copy()
        trafo_w = trafo0.copy()
        plants_w = P0.copy()
        plants_w["assigned_bus_380"] = plants_w["assigned_bus"]
        bus_ids_precluster = buses0["bus_id"].astype(str).to_numpy(copy=True)
        busmap_precluster = pd.Series(
            bus_ids_precluster.copy(),
            index=pd.Index(bus_ids_precluster.copy(), dtype="object"),
            name="bus_id_380",
            dtype="object",
        )

    buses_w = _fill_missing_ac_sub_networks(buses_w, lines_w)
    buses_w = _apply_cesa_country_clusters_to_buses(buses_w, cesa_country_clusters_n)
    country_load_mean_work = _collapse_country_series(country_load_mean, source_to_target)

    if pre_prune_tiny_ac_leaf_nodes:
        buses_w, lines_w, links_w, conv_w, trafo_w, plants_w, busmap_precluster = _pre_prune_tiny_ac_leaf_nodes(
            buses_w,
            lines_w,
            links_w,
            conv_w,
            trafo_w,
            plants_w,
            busmap_precluster,
            merge_mode="same_country",
            max_member_buses=tiny_ac_leaf_max_member_buses,
            max_total_capacity_mw=tiny_ac_leaf_max_capacity_mw,
        )

    if pre_prune_tiny_cross_border_leaf_nodes:
        buses_w, lines_w, links_w, conv_w, trafo_w, plants_w, busmap_precluster = _pre_prune_tiny_ac_leaf_nodes(
            buses_w,
            lines_w,
            links_w,
            conv_w,
            trafo_w,
            plants_w,
            busmap_precluster,
            merge_mode="cross_border",
            max_member_buses=tiny_ac_leaf_max_member_buses,
            max_total_capacity_mw=tiny_ac_leaf_max_capacity_mw,
        )
    
    # --------------------------------------------------------
    # AC-line table for internal electrical calculations
    # --------------------------------------------------------
    lines_cluster = _prepare_internal_line_table(
        lines_w,
        buses_w,
        voltage_mode=voltage_mode,
        s_base_mva=s_base_mva,
    )
    lines_output = lines_w.copy()

    # --------------------------------------------------------
    # Grouping logic:
    #   with 380-kV simplification  -> country x sub_network
    #   without simplification      -> country x voltage
    # --------------------------------------------------------
    if do_simplify_380:
        group_cols = ("country", "sub_network")
        grouping_mode = "country_x_subnetwork"
        plant_bus_col_for_alloc = "assigned_bus_380"
    else:
        group_cols = ("country", "voltage")
        grouping_mode = "country_x_voltage"
        plant_bus_col_for_alloc = "assigned_bus_380"   # equals assigned_bus in the non-380 case
    
    mask_cluster_ac = (~buses_w["dc_bool"]) & (
        buses_w["sync_area"].eq("CESA") | (not sync_collapse)
    )
    buses_cluster = buses_w.loc[mask_cluster_ac].copy()
    if buses_cluster.empty:
        raise ValueError("No AC buses found for clustering after preprocessing/simplification.")

    alloc_df = _allocate_k_to_groups_from_generation(
        buses_cluster[["bus_id", *group_cols]],
        plants_w,
        k_total=k_total,
        group_cols=group_cols,
        plant_bus_col=plant_bus_col_for_alloc,
        country_load_mean=country_load_mean_work,
        generation_country_weight=generation_country_weight,
        load_country_weight=load_country_weight,
    )
    if "country" in alloc_df.columns:
        alloc_df["country_label"] = (
            alloc_df["country"].astype(str).map(target_to_label).fillna(alloc_df["country"].astype(str))
        )
    
    k_by_group = {}
    for _, row in alloc_df.iterrows():
        key = tuple(str(row[c]) for c in group_cols)
        k_by_group[key] = int(row["k_group"])
    
    if method == "kmeans_geo":
        cluster_s = cluster_kmeans_geo(buses_cluster, k_by_group=k_by_group, group_cols=group_cols, random_state=0)

    elif method == "electrical_spectral":
        cluster_s = cluster_electrical_spectral(
            buses_cluster,
            lines_cluster,
            k_by_group=k_by_group,
            group_cols=group_cols,
            similarity=similarity,
            electrical_weight_mode=electrical_weight_mode,
            random_state=0,
            effrx_k_nearest=effrx_k_nearest,
            effrx_kernel=effrx_kernel,
            effrx_tau_scale=effrx_tau_scale,
        )

    elif method == "hac":
        cluster_s = cluster_hac(
            buses_cluster,
            lines_cluster,
            k_by_group=k_by_group,
            group_cols=group_cols,
            feature=hac_feature,
            similarity=similarity,
            electrical_weight_mode=electrical_weight_mode,
            n_embed=hac_n_embed,
            effrx_k_nearest=effrx_k_nearest,
            effrx_kernel=effrx_kernel,
            effrx_tau_scale=effrx_tau_scale,
        )

    elif method == "maximum_spanning_tree":
        cluster_s = cluster_mst(
            buses_cluster,
            lines_cluster,
            k_by_group=k_by_group,
            group_cols=group_cols,
            similarity=similarity,
            electrical_weight_mode=electrical_weight_mode,
        )

    else:
        raise ValueError(
            "method must be one of: "
            "'kmeans_geo', 'electrical_spectral', 'hac', 'maximum_spanning_tree'"
        )
    
    cluster_w = pd.Series(index=buses_w.index, dtype="object")

    if sync_collapse:
        mask_sync_all = buses_w["sync_node"].notna() & (~buses_w["sync_area"].eq("CESA"))
        cluster_w.loc[mask_sync_all] = buses_w.loc[mask_sync_all, "sync_node"].values
    
    dc_keep_mask = buses_w["dc_bool"] & cluster_w.isna()
    cluster_w.loc[dc_keep_mask] = buses_w.loc[dc_keep_mask, "bus_id"].map(lambda x: f"cl_dc_keep_{x}")
    
    cluster_w.update(cluster_s)
    
    if cluster_w.isna().any():
        raise ValueError("Some simplified buses have no cluster assignment.")

    buses_w = buses_w.copy()
    buses_w["cluster_id"] = cluster_w.values
    bus_map_work = buses_w.set_index("bus_id")["cluster_id"].to_dict()

    # ---------------------------------------------------------
    # Country output label:
    # - For collapsed external sync areas use the sync-area label
    # - keep original country separately
    # ---------------------------------------------------------
    collapsed_sync_area_labels = {"GB", "IE_NOIE", "NORDICS"}
    collapsed_sync_mask = bool(sync_collapse) & buses_w["sync_area"].astype(str).isin(collapsed_sync_area_labels)
    ni_noncollapsed_mask = (
        (not bool(sync_collapse))
        & buses_w["sync_area"].astype(str).eq("IE_NOIE")
        & buses_w["country"].astype(str).isin(["GB", "UK"])
    )

    buses_w["original_country"] = buses_w["country_original_raw"].astype(str)
    buses_w["country_out"] = np.where(
        collapsed_sync_mask,
        buses_w["sync_area"].astype(str),
        np.where(
            ni_noncollapsed_mask,
            "NI",
            buses_w["country"].astype(str),
        ),
    )
    buses_w["country_label_out"] = np.where(
        collapsed_sync_mask,
        buses_w["sync_area"].astype(str),
        np.where(
            ni_noncollapsed_mask,
            "NI",
            buses_w["country_group_label"].astype(str),
        ),
    )

    busmap_precluster = _deduplicate_mapping_index(busmap_precluster, name="busmap_precluster")
    bus_map_final = busmap_precluster.map(pd.Series(bus_map_work)).rename("bus_id_red")
    bus_map_final = bus_map_final.reindex(pd.Index(buses0["bus_id"].astype(str).values, dtype="object"))

    buses0_clusters = buses0.copy()
    buses0_clusters["cluster_id"] = buses0_clusters["bus_id"].map(bus_map_final)
    if buses0_clusters["cluster_id"].isna().any():
        raise ValueError("Some original buses have no final cluster assignment.")

    def mode(s: pd.Series):
        s = s.dropna()
        return s.value_counts().idxmax() if len(s) else np.nan

    cluster_pos = _representative_cluster_positions(
        buses_w,
        mode=plot_bus_position_mode,
    )

    buses_red = (
        buses_w.groupby("cluster_id", as_index=False)
        .agg(
            voltage=("voltage", "max"),
            voltage_min=("voltage", "min"),
            voltage_max=("voltage", "max"),
            n_voltage_levels=("voltage", "nunique"),
            dc_bool=("dc_bool", "all"),
            country=("country_out", mode),
            country_label=("country_label_out", mode),
            original_country=("original_country", _unique_sorted_join),
            n_original_countries=("original_country", "nunique"),
            sync_area=("sync_area", mode),
            sub_network=("sub_network", mode),
            n_buses_simplified=("bus_id", "count"),
        )
    )
    buses_red = buses_red.merge(cluster_pos, on="cluster_id", how="left")
    buses_red["dc"] = buses_red["dc_bool"].map({True: "t", False: "f"})
    buses_red["bus_id"] = buses_red["cluster_id"]
    buses_red = buses_red[
        [
            "bus_id",
            "voltage",
            "voltage_min",
            "voltage_max",
            "n_voltage_levels",
            "dc",
            "lat",
            "lon",
            "country",
            "country_label",
            "original_country",
            "n_original_countries",
            "sync_area",
            "sub_network",
            "n_buses_simplified",
        ]
    ]

    lines_work = lines_output.copy()

    lines_work["bus0_red"] = lines_work["bus0"].astype(str).map(bus_map_work)
    lines_work["bus1_red"] = lines_work["bus1"].astype(str).map(bus_map_work)
    lines_work = lines_work.dropna(subset=["bus0_red", "bus1_red"])
    lines_work = lines_work[lines_work["bus0_red"] != lines_work["bus1_red"]].copy()

    if aggregation_mode == "line_equivalent":
        lines_red = _aggregate_lines_line_equivalent(lines_work)
    elif aggregation_mode == "location_based":
        lines_red = _aggregate_lines_location_based(
            lines_work,
            buses_red,
            length_factor=line_length_factor,
        )
    else:
        raise ValueError("aggregation_mode must be 'line_equivalent' or 'location_based'")

    def keep_edge_individual(df: pd.DataFrame, cap_col: str | None = None, volt_col: str | None = None):
        D = df.copy()
        D["bus0"] = D["bus0"].astype(str).map(bus_map_work)
        D["bus1"] = D["bus1"].astype(str).map(bus_map_work)
        D = D.dropna(subset=["bus0", "bus1"])
        D = D[D["bus0"] != D["bus1"]].copy()
        if cap_col is not None and cap_col in D.columns:
            D[cap_col] = pd.to_numeric(D[cap_col], errors="coerce")
        if volt_col is not None and volt_col in D.columns:
            D[volt_col] = pd.to_numeric(D[volt_col], errors="coerce")
        return D.reset_index(drop=True)

    links_red = keep_edge_individual(links_w, cap_col="p_nom", volt_col="voltage")
    conv_red = keep_edge_individual(conv_w, cap_col="p_nom", volt_col="voltage")

    if hvdc_model not in {"explicit_dc", "pypsa_ac_links"}:
        raise ValueError("hvdc_model must be 'explicit_dc' or 'pypsa_ac_links'")

    if hvdc_model == "pypsa_ac_links":
        links_red, conv_red = _collapse_point_to_point_hvdc_to_ac_links(
            links_red,
            conv_red,
            buses_red,
            fallback_length_factor=line_length_factor
        )

    T = trafo_w.copy()
    T["bus0_red"] = T["bus0"].astype(str).map(bus_map_work)
    T["bus1_red"] = T["bus1"].astype(str).map(bus_map_work)
    T = T.dropna(subset=["bus0_red", "bus1_red"])
    T = T[T["bus0_red"] != T["bus1_red"]].copy()

    for c in ["voltage_bus0", "voltage_bus1", "s_nom"]:
        if c in T.columns:
            T[c] = pd.to_numeric(T[c], errors="coerce")

    tr_red = (
        T.groupby(["bus0_red", "bus1_red", "voltage_bus0", "voltage_bus1"], as_index=False)
        .agg(s_nom=("s_nom", "sum"), n_tr=("transformer_id", "count"))
    )
    tr_red["transformer_id"] = [f"tr_{i}" for i in range(len(tr_red))]
    tr_red = tr_red[
        ["transformer_id", "bus0_red", "bus1_red", "voltage_bus0", "voltage_bus1", "s_nom", "n_tr"]
    ]

    P = plants_w.copy()
    P["bus_red"] = P["assigned_bus_380"].astype(str).map(bus_map_work)
    P = P.dropna(subset=["bus_red"]).copy()
    P["Capacity"] = pd.to_numeric(P.get("Capacity", 0.0), errors="coerce").fillna(0.0)
    storage_capacity_col = None
    for candidate in ("StorageCapacity_MWh", "storage_capacity_mwh", "storagecapacity_mwh"):
        if candidate in P.columns:
            storage_capacity_col = candidate
            break
    if storage_capacity_col is not None:
        P[storage_capacity_col] = pd.to_numeric(P[storage_capacity_col], errors="coerce")

    if "country_code" in P.columns:
        P["original_country"] = P["country_code"].astype(str)
    elif "Country" in P.columns:
        P["original_country"] = P["Country"].astype(str)
    elif "country" in P.columns:
        P["original_country"] = P["country"].astype(str)
    else:
        P["original_country"] = pd.NA

    bus_country_out = buses_red.set_index("bus_id")["country"].to_dict()
    bus_country_label_out = buses_red.set_index("bus_id")["country_label"].to_dict()
    P["country"] = P["bus_red"].astype(str).map(bus_country_out)
    P["country_label"] = P["bus_red"].astype(str).map(bus_country_label_out)

    plant_aggs: dict[str, tuple[str, str | Callable[[pd.Series], float]]] = {
        "Capacity": ("Capacity", "sum"),
        "n_plants": ("id", "count"),
    }
    if storage_capacity_col is not None:
        plant_aggs["StorageCapacity_MWh"] = (storage_capacity_col, _sum_min_count_one)

    plants_red = (
        P.groupby(
            ["bus_red", "country", "country_label", "original_country", "Fueltype", "Technology", "Set"],
            as_index=False
        )
        .agg(**plant_aggs)
        .rename(columns={"bus_red": "bus_id"})
    )

    if hvdc_model == "pypsa_ac_links":
        links_red, conv_red = _prune_one_sided_dc_components(
            links_red,
            conv_red,
            buses_red,
            plants_red,
        )

    if prune_tiny_ac_leaf_nodes:
        buses_red, lines_red, links_red, conv_red, tr_red, plants_red = _contract_tiny_ac_leaf_nodes(
            buses_red,
            lines_red,
            links_red,
            conv_red,
            tr_red,
            plants_red,
            merge_mode="same_country",
            max_member_buses=tiny_ac_leaf_max_member_buses,
            max_total_capacity_mw=tiny_ac_leaf_max_capacity_mw,
        )

    if prune_tiny_cross_border_leaf_nodes:
        buses_red, lines_red, links_red, conv_red, tr_red, plants_red = _contract_tiny_ac_leaf_nodes(
            buses_red,
            lines_red,
            links_red,
            conv_red,
            tr_red,
            plants_red,
            merge_mode="cross_border",
            max_member_buses=tiny_ac_leaf_max_member_buses,
            max_total_capacity_mw=tiny_ac_leaf_max_capacity_mw,
        )

    # Remove unused (dc) buses
    buses_red = _prune_unused_buses(
        buses_red,
        lines_red=lines_red,
        links_red=links_red,
        conv_red=conv_red,
        tr_red=tr_red,
        plants_red=plants_red,
    )

    return {
        "country_group_allocation": alloc_df,
        "busmap_380": busmap_precluster.reset_index().rename(columns={"index": "bus_id", "bus_id_380": "bus_id_380"}),
        "bus_map": pd.DataFrame({"bus_id": bus_map_final.index.astype(str), "bus_id_red": bus_map_final.values}),
        "buses_with_clusters": buses0_clusters[
            ["bus_id", "voltage", "dc_bool", "cluster_id", "lat", "lon", "country", "sync_area"]],
        "buses_simplified_380": buses_w[
            [
                "bus_id",
                "voltage",
                "dc_bool",
                "country_out",
                "country_label_out",
                "original_country",
                "sync_area",
                "sub_network",
                "cluster_id",
                "lat",
                "lon",
            ]].rename(columns={"country_out": "country", "country_label_out": "country_label"}),
        "buses": buses_red,
        "lines": lines_red,
        "links": links_red,
        "converters": conv_red,
        "transformers": tr_red,
        "plants": plants_red,
        "cesa_country_clusters": cesa_country_clusters_n,
        "grouping_mode": pd.DataFrame({"grouping_mode": [grouping_mode]}),
        "group_columns": pd.DataFrame({"group_col": list(group_cols)}),
        "hvdc_model": pd.DataFrame({"hvdc_model": [hvdc_model]}),
        "voltage_mode": pd.DataFrame({
        "voltage_mode": [voltage_mode],
        "s_base_mva": [s_base_mva if voltage_mode == "per_unit" else np.nan]
        })
    }


# ============================================================
# Plot helpers
# ============================================================
#def _inline_svg_to_html(svg_path, html_path, title="Network"):
#    svg_txt = Path(svg_path).read_text(encoding="utf-8")
#    html = f"""<!doctype html>
#<html lang="en"><head><meta charset="utf-8"><title>{title}</title></head>
#<body style="margin:0;padding:0;font-family:sans-serif;">
#<div style="padding:10px;"><h3 style="margin:0 0 10px 0;">{title}</h3></div>
#<div style="width:100%;overflow:auto;">{svg_txt}</div>
#</body></html>"""
#    Path(html_path).write_text(html, encoding="utf-8")


def _country_plot_path(countries_path: str | Path | None) -> Path | None:
    raw_path = countries_path if countries_path is not None else COUNTRIES_URL
    if raw_path in (None, "", "None"):
        return None

    path = Path(raw_path)
    if not path.exists():
        return None
    if path.suffix.lower() == ".shp" and not path.with_suffix(".shx").exists():
        return None
    return path


def _try_plot_countries(
    ax,
    bbox=None,
    countries_path: str | Path | None = None,
    *,
    land_color: str = "#d8ddc7",
    water_color: str = "#dfeaf4",
):
    ax.set_facecolor(water_color)
    ax.figure.set_facecolor(water_color)

    try:
        import geopandas as gpd

        plot_path = _country_plot_path(countries_path)
        if plot_path is None:
            return
        world = gpd.read_file(plot_path)

        if world.crs is not None and world.crs.to_string() != "EPSG:4326":
            world = world.to_crs("EPSG:4326")

        if bbox is not None:
            xmin, xmax, ymin, ymax = bbox
            world = world.cx[xmin:xmax, ymin:ymax]
            clip_geom = box(xmin, ymin, xmax, ymax)
            world = world.clip(clip_geom)

        world.plot(ax=ax, color=land_color, edgecolor="none", alpha=0.95, zorder=-5)
        world.boundary.plot(ax=ax, linewidth=0.6, color="black", alpha=0.8, zorder=0)

    except Exception as e:
        warnings.warn(f"Could not plot country boundaries: {e}")


def _snap_plot_points_to_land(
    plot_xy: pd.DataFrame,
    *,
    bbox=None,
    countries_path: str | Path | None = None,
    max_snap_distance_km: float = 20.0,
) -> pd.DataFrame:
    if plot_xy.empty:
        return plot_xy

    out = plot_xy.copy()
    try:
        import geopandas as gpd

        plot_path = _country_plot_path(countries_path)
        if plot_path is None:
            return out
        world = gpd.read_file(plot_path)

        if world.crs is not None and world.crs.to_string() != "EPSG:4326":
            world = world.to_crs("EPSG:4326")

        if bbox is not None:
            xmin, xmax, ymin, ymax = bbox
            world = world.cx[xmin:xmax, ymin:ymax]
            clip_geom = box(xmin, ymin, xmax, ymax)
            world = world.clip(clip_geom)

        if world.empty:
            return out

        land_geom = world.geometry.union_all() if hasattr(world.geometry, "union_all") else world.geometry.unary_union
        if land_geom.is_empty:
            return out

        for idx, row in out[["x", "y"]].dropna().iterrows():
            point = Point(float(row["x"]), float(row["y"]))
            if land_geom.covers(point):
                continue
            snapped, _ = nearest_points(land_geom, point)
            dist_km = float(
                _haversine_array(
                    [float(row["y"])],
                    [float(row["x"])],
                    [float(snapped.y)],
                    [float(snapped.x)],
                )[0]
            )
            if dist_km > float(max_snap_distance_km):
                continue
            out.at[idx, "x"] = float(snapped.x)
            out.at[idx, "y"] = float(snapped.y)

    except Exception as e:
        warnings.warn(f"Could not snap plot points to land: {e}")

    return out


def _bus_xy(buses: pd.DataFrame, *, buses_are_latlon: bool) -> pd.DataFrame:
    b = buses.copy()
    b["bus_id"] = b["bus_id"].astype(str)

    if buses_are_latlon:
        b["x"] = pd.to_numeric(b["lon"], errors="coerce")
        b["y"] = pd.to_numeric(b["lat"], errors="coerce")
    else:
        b["x"] = pd.to_numeric(b.get("x", b.get("lon")), errors="coerce")
        b["y"] = pd.to_numeric(b.get("y", b.get("lat")), errors="coerce")

    b = b.dropna(subset=["x", "y"])
    return b


def _attach_plot_xy(df: pd.DataFrame, bxy: pd.DataFrame, *, bus_col: str = "bus_id") -> pd.DataFrame:
    out = df.copy()
    for col in ["x", "y"]:
        if col in out.columns:
            out = out.drop(columns=col)

    out[bus_col] = out[bus_col].astype(str)
    return out.merge(
        bxy[["bus_id", "x", "y"]].rename(columns={"bus_id": bus_col}),
        on=bus_col,
        how="left",
    )


def _segments_from_edges(edges: pd.DataFrame, bxy: pd.DataFrame, bus0: str, bus1: str) -> np.ndarray:
    e = edges[[bus0, bus1]].copy()
    e[bus0] = e[bus0].astype(str)
    e[bus1] = e[bus1].astype(str)

    tmp0 = bxy.rename(columns={"bus_id": bus0, "x": "x0", "y": "y0"})[[bus0, "x0", "y0"]]
    tmp1 = bxy.rename(columns={"bus_id": bus1, "x": "x1", "y": "y1"})[[bus1, "x1", "y1"]]

    e = e.merge(tmp0, on=bus0, how="left").merge(tmp1, on=bus1, how="left")
    e = e.dropna(subset=["x0", "y0", "x1", "y1"])

    if e.empty:
        return np.zeros((0, 2, 2))
    return np.stack([e[["x0", "y0"]].values, e[["x1", "y1"]].values], axis=1)


def _capacity_to_marker_size(cap_mw: float, *, pie: bool = False) -> float:
    """
    Scatter size (points^2) derived from installed capacity [MW].
    For pie=True, use a slightly larger minimum size so pie charts stay readable.
    """
    cap_mw = max(float(cap_mw), 0.0)
    if pie:
        s = np.sqrt(cap_mw) * 0.9
        return float(np.clip(s, 20, 600))
    else:
        s = np.sqrt(cap_mw) * 0.8
        return float(np.clip(s, 6, 220))


def _format_capacity_label(cap_mw: float) -> str:
    cap_mw = float(cap_mw)
    if cap_mw >= 1000:
        return f"{cap_mw/1000:.0f} GW"
    return f"{cap_mw:.0f} MW"


def _capacity_legend_values(capacities: np.ndarray | list[float]) -> list[float]:
    caps = np.asarray(capacities, dtype=float)
    caps = caps[np.isfinite(caps) & (caps > 0)]

    if caps.size == 0:
        return [1000, 5000, 10000]

    q = np.quantile(caps, [0.25, 0.7, 0.95])
    vals = []
    for x in q:
        if x < 1000:
            vals.append(max(100, round(x / 100) * 100))
        else:
            vals.append(max(1000, round(x / 1000) * 1000))

    vals = sorted(set(vals))
    while len(vals) < 3:
        vals.append(vals[-1] * 2)

    return vals[:3]


def _scatter_pie(
    ax,
    x: float,
    y: float,
    ratios: np.ndarray,
    colors: list[str],
    size: float,
    *,
    edgecolor: str = "white",
    linewidth: float = 0.25,
    outline_color: str = "#333333",
    outline_width: float = 0.35,
    alpha: float = 0.95,
    zorder: int = 5,
):
    """
    Draw a pie chart marker at position (x, y).
    size follows the same points^2 convention as ax.scatter.
    """
    ratios = np.asarray(ratios, dtype=float)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0 or ratios.sum() <= 0:
        return

    ratios = ratios / ratios.sum()

    start = 0.0
    for frac, col in zip(ratios, colors):
        if frac <= 0:
            continue

        theta = np.linspace(2 * np.pi * start, 2 * np.pi * (start + frac), 40)
        verts = np.column_stack([
            np.r_[0.0, np.cos(theta), 0.0],
            np.r_[0.0, np.sin(theta), 0.0],
        ])

        ax.scatter(
            [x],
            [y],
            marker=verts,
            s=size,
            facecolor=col,
            edgecolors=edgecolor,
            linewidths=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
        start += frac

    # Outer ring for better readability
    ax.scatter(
        [x],
        [y],
        marker="o",
        s=size,
        facecolors="none",
        edgecolors=outline_color,
        linewidths=outline_width,
        zorder=zorder + 0.1,
    )


def _add_plant_map_legends(
    ax,
    *,
    capacities_mw: np.ndarray | list[float],
    fuels_present: list[str],
):
    # --- Branch / bus legend ---
    branch_handles = [
        Line2D([0], [0], color="#808080", lw=1.5, linestyle="-", label="AC line"),
        Line2D([0], [0], color=HVDC_COLOR, lw=1.8, linestyle="--", label="HVDC link"),
        Line2D([0], [0], color="#ff8c00", lw=1.5, linestyle="-.", label="Converter"),
        Line2D([0], [0], color="#555555", lw=1.2, linestyle=":", label="Transformer"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4",
               markeredgecolor="#1f77b4", markersize=5, label="AC bus"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#ff8c00",
               markeredgecolor="#ff8c00", markersize=6, label="DC terminal"),
    ]
    leg1 = ax.legend(
        handles=branch_handles,
        title="Grid / branches",
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(leg1)

    # --- Node capacity legend ---
    cap_vals = _capacity_legend_values(capacities_mw)
    cap_handles = []
    for c in cap_vals:
        s = _capacity_to_marker_size(c, pie=True)
        cap_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor="none",
                markeredgecolor="#333333",
                markersize=np.sqrt(s),
                label=_format_capacity_label(c),
            )
        )

    leg2 = ax.legend(
        handles=cap_handles,
        title="Node capacity",
        loc="upper left",
        bbox_to_anchor=(0.33, 0.99),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(leg2)

    # --- Fuel / technology legend ---
    fuel_handles = [
        Patch(facecolor=FUEL_COLORS.get(f, "#999999"), edgecolor="none", label=f)
        for f in fuels_present
    ]
    leg3 = ax.legend(
        handles=fuel_handles,
        title="Generation / storage",
        loc="upper left",
        bbox_to_anchor=(0.16, 0.99),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        ncol=1,
    )
    ax.add_artist(leg3)


def _add_voltage_map_legends(
    ax,
    *,
    voltages_present: list[int],
):
    # AC voltage levels
    voltage_handles = [
        Line2D(
            [0], [0],
            color=VOLTAGE_COLORS.get(int(v), "#ff0000"),
            lw=2.0,
            linestyle="-",
            label=f"AC {int(v)} kV",
        )
        for v in voltages_present
    ]

    other_handles = [
        Line2D([0], [0], color=HVDC_COLOR, lw=2.0, linestyle="--", label="HVDC link"),
        Line2D([0], [0], color="#ff8c00", lw=1.6, linestyle="-.", label="Converter"),
        Line2D([0], [0], color="#555555", lw=1.4, linestyle=":", label="Transformer"),
    ]

    leg1 = ax.legend(
        handles=voltage_handles + other_handles,
        title="Lines / branches",
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(leg1)

    bus_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4",
               markeredgecolor="#1f77b4", markersize=5, label="AC bus"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#ff8c00",
               markeredgecolor="#ff8c00", markersize=6, label="DC terminal"),
    ]

    leg2 = ax.legend(
        handles=bus_handles,
        title="Buses",
        loc="upper left",
        bbox_to_anchor=(0.20, 0.99),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(leg2)


def plot_plants_map(
    *,
    out_svg: Path,
    title: str,
    buses_are_latlon: bool,
    plant_bus_col: str,
    plant_fuel_col: str = "Fueltype",
    plant_cap_col: str = "Capacity",
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    plants: pd.DataFrame,
    links: pd.DataFrame | None = None,
    transformers: pd.DataFrame | None = None,
    converters: pd.DataFrame | None = None,
    bbox=None,
    countries_path: str | Path | None = None,
    snap_plot_positions_to_land: bool = False,
    snap_plot_max_distance_km: float = 20.0,
    downsample_lines: int | None = None,
    downsample_links: int | None = None,
    downsample_transformers: int | None = None,
    downsample_converters: int | None = None
):
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    bxy = _bus_xy(buses, buses_are_latlon=buses_are_latlon)
    if snap_plot_positions_to_land:
        bxy = _snap_plot_points_to_land(
            bxy,
            bbox=bbox,
            countries_path=countries_path,
            max_snap_distance_km=snap_plot_max_distance_km,
        )

    fig, ax = plt.subplots(figsize=(14, 10), dpi=140)
    ax.set_title(title)
    ax.set_axis_off()
    _try_plot_countries(ax, bbox=bbox, countries_path=countries_path)
    
    # --- AC lines ---
    seg = _segments_from_edges(
        lines,
        bxy,
        bus0=("bus0" if "bus0" in lines.columns else "u"),
        bus1=("bus1" if "bus1" in lines.columns else "v"),
    )

    if downsample_lines is not None and len(seg) > downsample_lines:
        rng = np.random.default_rng(0)
        seg = seg[rng.choice(len(seg), size=downsample_lines, replace=False)]

    if len(seg) > 0:
        ax.add_collection(LineCollection(seg, linewidths=0.25, alpha=0.35, color="#808080"))
        
    # --- HVDC links ---
    if links is not None and len(links) > 0:
        D = links.copy()
        b0 = "bus0" if "bus0" in D.columns else ("u" if "u" in D.columns else None)
        b1 = "bus1" if "bus1" in D.columns else ("v" if "v" in D.columns else None)

        if b0 is not None and b1 is not None:
            seg_dc = _segments_from_edges(D, bxy, bus0=b0, bus1=b1)
            if downsample_links is not None and len(seg_dc) > downsample_links:
                rng = np.random.default_rng(10)
                seg_dc = seg_dc[rng.choice(len(seg_dc), size=downsample_links, replace=False)]
            if len(seg_dc) > 0:
                ax.add_collection(
                    LineCollection(seg_dc, linewidths=1.0, alpha=0.7, color=HVDC_COLOR, linestyles="dashed")
                )
    
    # --- Converters ---
    if converters is not None and len(converters) > 0:
        C = converters.copy()
    
        b0 = "bus0" if "bus0" in C.columns else ("u" if "u" in C.columns else None)
        b1 = "bus1" if "bus1" in C.columns else ("v" if "v" in C.columns else None)
    
        if b0 is not None and b1 is not None:
            seg_cv = _segments_from_edges(C, bxy, bus0=b0, bus1=b1)
    
            if downsample_converters is not None and len(seg_cv) > downsample_converters:
                rng = np.random.default_rng(30)
                seg_cv = seg_cv[rng.choice(len(seg_cv), size=downsample_converters, replace=False)]
    
            if len(seg_cv) > 0:
                ax.add_collection(
                    LineCollection(
                        seg_cv,
                        linewidths=0.7,
                        alpha=0.75,
                        color="#ff8c00",
                        linestyles="dashdot",
                    )
                )
    
    # --- Transformers ---
    if transformers is not None and len(transformers) > 0:
        T = transformers.copy()
    
        b0 = "bus0_red" if "bus0_red" in T.columns else ("bus0" if "bus0" in T.columns else None)
        b1 = "bus1_red" if "bus1_red" in T.columns else ("bus1" if "bus1" in T.columns else None)
    
        if b0 is not None and b1 is not None:
            seg_tr = _segments_from_edges(T, bxy, bus0=b0, bus1=b1)
    
            if downsample_transformers is not None and len(seg_tr) > downsample_transformers:
                rng = np.random.default_rng(20)
                seg_tr = seg_tr[rng.choice(len(seg_tr), size=downsample_transformers, replace=False)]
    
            if len(seg_tr) > 0:
                ax.add_collection(
                    LineCollection(
                        seg_tr,
                        linewidths=0.5,
                        alpha=0.5,
                        color="#555555",
                        linestyles="dotted",
                    )
                )

    #ax.scatter(bxy["x"].values, bxy["y"].values, s=1.0, alpha=0.25, color="#1f77b4")
    bplot = _attach_plot_xy(buses, bxy).dropna(subset=["x", "y"])

    if "dc" in bplot.columns:
        dc_mask = bplot["dc"].astype(str).str.lower().isin(["t", "true", "1", "yes", "y"])
    elif "dc_bool" in bplot.columns:
        dc_mask = bplot["dc_bool"].astype(bool)
    else:
        dc_mask = pd.Series(False, index=bplot.index)

    # AC buses
    ax.scatter(
        bplot.loc[~dc_mask, "x"].values,
        bplot.loc[~dc_mask, "y"].values,
        s=1.0,
        alpha=0.25,
        color="#1f77b4",
        zorder=2,
    )

    # DC buses
    ax.scatter(
        bplot.loc[dc_mask, "x"].values,
        bplot.loc[dc_mask, "y"].values,
        s=8.0,
        alpha=0.9,
        color="#ff8c00",
        marker="s",
        zorder=3,
    )
    
    # --- Plants ---
    P = plants.copy()
    P[plant_bus_col] = P[plant_bus_col].astype(str)
    P[plant_fuel_col] = P[plant_fuel_col].astype(str)
    P[plant_cap_col] = pd.to_numeric(P[plant_cap_col], errors="coerce").fillna(0.0)

    Pagg = (
        P.groupby([plant_bus_col, plant_fuel_col], as_index=False)
        .agg(**{plant_cap_col: (plant_cap_col, "sum")})
        .rename(columns={plant_bus_col: "bus_id"})
    )

    Pagg = _attach_plot_xy(Pagg, bxy).dropna(subset=["x", "y"])

    fuels_present = []

    if not Pagg.empty:
        fuel_order = list(FUEL_COLORS.keys())
        Pagg["_fuel_order"] = pd.Categorical(
            Pagg[plant_fuel_col],
            categories=fuel_order,
            ordered=True,
        )
        Pagg = Pagg.sort_values(["bus_id", "_fuel_order", plant_fuel_col])

        total_by_bus = (
            Pagg.groupby("bus_id", as_index=False)[plant_cap_col]
            .sum()
            .rename(columns={plant_cap_col: "total_cap_mw"})
        )

        Pagg = Pagg.merge(total_by_bus, on="bus_id", how="left")

        for bus_id, g in Pagg.groupby("bus_id", sort=False):
            x = g["x"].iloc[0]
            y = g["y"].iloc[0]
            total_cap = float(g["total_cap_mw"].iloc[0])

            vals = g[plant_cap_col].values.astype(float)
            ratios = vals / vals.sum() if vals.sum() > 0 else np.ones_like(vals) / len(vals)
            colors = [FUEL_COLORS.get(f, "#999999") for f in g[plant_fuel_col].values]
            size = _capacity_to_marker_size(total_cap, pie=True)

            _scatter_pie(
                ax,
                x=x,
                y=y,
                ratios=ratios,
                colors=colors,
                size=size,
                zorder=5,
            )

        fuels_present = sorted(
            Pagg[plant_fuel_col].dropna().astype(str).unique().tolist(),
            key=lambda f: (list(FUEL_COLORS.keys()).index(f) if f in FUEL_COLORS else 999, f)
        )

        _add_plant_map_legends(
            ax,
            capacities_mw=total_by_bus["total_cap_mw"].values,
            fuels_present=fuels_present,
        )
    else:
        _add_plant_map_legends(
            ax,
            capacities_mw=np.array([]),
            fuels_present=[],
        )

    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    plt.close(fig)
    #_inline_svg_to_html(out_svg, out_svg.with_suffix(".html"), title)


def plot_voltage_map(
    *,
    out_svg: Path,
    title: str,
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    links: pd.DataFrame | None,
    buses_are_latlon: bool,
    converters: pd.DataFrame | None = None,
    transformers: pd.DataFrame | None = None,
    bbox=None,
    countries_path: str | Path | None = None,
    snap_plot_positions_to_land: bool = False,
    snap_plot_max_distance_km: float = 20.0,
    downsample_lines: int | None = None,
    downsample_links: int | None = None,
    downsample_converters: int | None = None,
    downsample_transformers: int | None = None,
):
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    bxy = _bus_xy(buses, buses_are_latlon=buses_are_latlon)
    if snap_plot_positions_to_land:
        bxy = _snap_plot_points_to_land(
            bxy,
            bbox=bbox,
            countries_path=countries_path,
            max_snap_distance_km=snap_plot_max_distance_km,
        )

    fig, ax = plt.subplots(figsize=(14, 10), dpi=140)
    ax.set_title(title)
    ax.set_axis_off()
    _try_plot_countries(ax, bbox=bbox, countries_path=countries_path)

    # --- AC lines by voltage ---
    L = lines.copy()
    if "voltage" in L.columns:
        L["voltage"] = pd.to_numeric(L["voltage"], errors="coerce").astype("Int64")
    else:
        L["voltage"] = pd.NA

    bus0 = "bus0" if "bus0" in L.columns else "u"
    bus1 = "bus1" if "bus1" in L.columns else "v"

    if downsample_lines is not None and len(L) > downsample_lines:
        rng = np.random.default_rng(1)
        L = L.iloc[rng.choice(len(L), size=downsample_lines, replace=False)].copy()

    for v in sorted([vv for vv in pd.unique(L["voltage"]) if pd.notna(vv)]):
        Lv = L[L["voltage"] == v]
        if Lv.empty:
            continue
        seg = _segments_from_edges(Lv, bxy, bus0=bus0, bus1=bus1)
        if len(seg) == 0:
            continue
        col = VOLTAGE_COLORS.get(int(v), "#ff0000")
        lw = 0.6 if int(v) >= 380 else 0.45
        ax.add_collection(LineCollection(seg, linewidths=lw, alpha=0.8, color=col))

    # --- HVDC links ---
    if links is not None and len(links) > 0:
        D = links.copy()
        b0 = "bus0" if "bus0" in D.columns else ("u" if "u" in D.columns else None)
        b1 = "bus1" if "bus1" in D.columns else ("v" if "v" in D.columns else None)
        if b0 is not None and b1 is not None:
            seg = _segments_from_edges(D, bxy, bus0=b0, bus1=b1)
            if downsample_links is not None and len(seg) > downsample_links:
                rng = np.random.default_rng(11)
                seg = seg[rng.choice(len(seg), size=downsample_links, replace=False)]
            if len(seg) > 0:
                ax.add_collection(
                    LineCollection(
                        seg,
                        linewidths=1.2,
                        alpha=0.85,
                        color=HVDC_COLOR,
                        linestyles="dashed",
                    )
                )

    # --- Converters ---
    if converters is not None and len(converters) > 0:
        C = converters.copy()
        b0 = "bus0" if "bus0" in C.columns else ("u" if "u" in C.columns else None)
        b1 = "bus1" if "bus1" in C.columns else ("v" if "v" in C.columns else None)

        if b0 is not None and b1 is not None:
            seg_cv = _segments_from_edges(C, bxy, bus0=b0, bus1=b1)

            if downsample_converters is not None and len(seg_cv) > downsample_converters:
                rng = np.random.default_rng(31)
                seg_cv = seg_cv[rng.choice(len(seg_cv), size=downsample_converters, replace=False)]

            if len(seg_cv) > 0:
                ax.add_collection(
                    LineCollection(
                        seg_cv,
                        linewidths=0.7,
                        alpha=0.8,
                        color="#ff8c00",
                        linestyles="dashdot",
                    )
                )

    # --- Transformers ---
    if transformers is not None and len(transformers) > 0:
        T = transformers.copy()
        b0 = "bus0_red" if "bus0_red" in T.columns else ("bus0" if "bus0" in T.columns else None)
        b1 = "bus1_red" if "bus1_red" in T.columns else ("bus1" if "bus1" in T.columns else None)

        if b0 is not None and b1 is not None:
            seg_tr = _segments_from_edges(T, bxy, bus0=b0, bus1=b1)

            if downsample_transformers is not None and len(seg_tr) > downsample_transformers:
                rng = np.random.default_rng(21)
                seg_tr = seg_tr[rng.choice(len(seg_tr), size=downsample_transformers, replace=False)]

            if len(seg_tr) > 0:
                ax.add_collection(
                    LineCollection(
                        seg_tr,
                        linewidths=0.5,
                        alpha=0.5,
                        color="#555555",
                        linestyles="dotted",
                    )
                )

    # --- AC/DC buses ---
    bplot = _attach_plot_xy(buses, bxy).dropna(subset=["x", "y"])

    if "dc" in bplot.columns:
        dc_mask = bplot["dc"].astype(str).str.lower().isin(["t", "true", "1", "yes", "y"])
    elif "dc_bool" in bplot.columns:
        dc_mask = bplot["dc_bool"].astype(bool)
    else:
        dc_mask = pd.Series(False, index=bplot.index)

    ax.scatter(
        bplot.loc[~dc_mask, "x"].values,
        bplot.loc[~dc_mask, "y"].values,
        s=1.0,
        alpha=0.25,
        color="#1f77b4",
        zorder=2,
    )

    ax.scatter(
        bplot.loc[dc_mask, "x"].values,
        bplot.loc[dc_mask, "y"].values,
        s=8.0,
        alpha=0.9,
        color="#ff8c00",
        marker="s",
        zorder=3,
    )

    voltages_present = sorted(
        [int(v) for v in pd.unique(lines["voltage"]) if pd.notna(v)]
    )

    _add_voltage_map_legends(
        ax,
        voltages_present=voltages_present,
    )

    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    plt.close(fig)
    #_inline_svg_to_html(out_svg, out_svg.with_suffix(".html"), title)


# ============================================================
# Diagnostics
# ============================================================
def diagnose_reduced_network(red: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Diagnostics for the reduced network.

    Checks in particular:
    - isolated buses in the full graph
    - plants on DC buses
    - plants on isolated buses
    - remaining explicit DC buses / converters
    - mixed AC/DC links
    - DC components
    """
    buses = red["buses"].copy()
    lines = red.get("lines", pd.DataFrame()).copy()
    links = red.get("links", pd.DataFrame()).copy()
    converters = red.get("converters", pd.DataFrame()).copy()
    transformers = red.get("transformers", pd.DataFrame()).copy()
    plants = red.get("plants", pd.DataFrame()).copy()

    buses["bus_id"] = buses["bus_id"].astype(str)

    if "dc" in buses.columns:
        buses["is_dc"] = buses["dc"].astype(str).str.lower().isin(["t", "true", "1", "yes", "y"])
    elif "dc_bool" in buses.columns:
        buses["is_dc"] = buses["dc_bool"].astype(bool)
    else:
        buses["is_dc"] = False

    bus_is_dc = buses.set_index("bus_id")["is_dc"].to_dict()

    # ---------------------------------------------------------
    # Overall graph assembled from all branch types
    # ---------------------------------------------------------
    G = nx.Graph()
    G.add_nodes_from(buses["bus_id"].tolist())

    def _add_edges(df: pd.DataFrame, b0_candidates: tuple[str, ...], b1_candidates: tuple[str, ...], edge_type: str):
        if df is None or df.empty:
            return

        D = df.copy()
        b0 = next((c for c in b0_candidates if c in D.columns), None)
        b1 = next((c for c in b1_candidates if c in D.columns), None)
        if b0 is None or b1 is None:
            return

        D[b0] = D[b0].astype(str)
        D[b1] = D[b1].astype(str)
        D = D.dropna(subset=[b0, b1])
        D = D.loc[D[b0] != D[b1], [b0, b1]].copy()

        for u, v in D.itertuples(index=False, name=None):
            G.add_edge(str(u), str(v), edge_type=edge_type)

    _add_edges(lines, ("bus0", "u"), ("bus1", "v"), "line")
    _add_edges(links, ("bus0", "u"), ("bus1", "v"), "link")
    _add_edges(converters, ("bus0", "u"), ("bus1", "v"), "converter")
    _add_edges(transformers, ("bus0_red", "bus0"), ("bus1_red", "bus1"), "transformer")

    degree = pd.Series(dict(G.degree()), name="degree")
    buses_diag = buses.copy()
    buses_diag["degree"] = buses_diag["bus_id"].map(degree).fillna(0).astype(int)

    isolated_buses = buses_diag.loc[buses_diag["degree"] == 0].copy()

    # ---------------------------------------------------------
    # Plant
    # ---------------------------------------------------------
    plants_diag = plants.copy()
    if not plants_diag.empty:
        plants_diag["bus_id"] = plants_diag["bus_id"].astype(str)
        plants_diag = plants_diag.merge(
            buses_diag[["bus_id", "is_dc", "degree", "country", "sync_area"]],
            on="bus_id",
            how="left"
        )
    else:
        plants_diag = pd.DataFrame()

    plants_on_dc = plants_diag.loc[plants_diag["is_dc"].fillna(False)].copy() if not plants_diag.empty else pd.DataFrame()
    plants_on_isolated = plants_diag.loc[plants_diag["degree"].fillna(0).astype(int) == 0].copy() if not plants_diag.empty else pd.DataFrame()

    # ---------------------------------------------------------
    # Link
    # ---------------------------------------------------------
    links_diag = links.copy()
    if not links_diag.empty:
        links_diag["bus0"] = links_diag["bus0"].astype(str)
        links_diag["bus1"] = links_diag["bus1"].astype(str)
        links_diag["bus0_is_dc"] = links_diag["bus0"].map(bus_is_dc)
        links_diag["bus1_is_dc"] = links_diag["bus1"].map(bus_is_dc)

        links_diag["link_topology"] = np.select(
            [
                links_diag["bus0_is_dc"].fillna(False) & links_diag["bus1_is_dc"].fillna(False),
                (~links_diag["bus0_is_dc"].fillna(False)) & (~links_diag["bus1_is_dc"].fillna(False)),
            ],
            [
                "dc_dc",
                "ac_ac",
            ],
            default="ac_dc_mixed"
        )
    else:
        links_diag = pd.DataFrame()

    mixed_links = links_diag.loc[links_diag["link_topology"] == "ac_dc_mixed"].copy() if not links_diag.empty else pd.DataFrame()
    dc_dc_links = links_diag.loc[links_diag["link_topology"] == "dc_dc"].copy() if not links_diag.empty else pd.DataFrame()
    ac_ac_links = links_diag.loc[links_diag["link_topology"] == "ac_ac"].copy() if not links_diag.empty else pd.DataFrame()

    # ---------------------------------------------------------
    # Converter
    # ---------------------------------------------------------
    conv_diag = converters.copy()
    if not conv_diag.empty:
        conv_diag["bus0"] = conv_diag["bus0"].astype(str)
        conv_diag["bus1"] = conv_diag["bus1"].astype(str)
        conv_diag["bus0_is_dc"] = conv_diag["bus0"].map(bus_is_dc)
        conv_diag["bus1_is_dc"] = conv_diag["bus1"].map(bus_is_dc)

        conv_diag["converter_topology"] = np.select(
            [
                conv_diag["bus0_is_dc"].fillna(False) & conv_diag["bus1_is_dc"].fillna(False),
                (~conv_diag["bus0_is_dc"].fillna(False)) & (~conv_diag["bus1_is_dc"].fillna(False)),
            ],
            [
                "dc_dc",
                "ac_ac",
            ],
            default="ac_dc_mixed"
        )
    else:
        conv_diag = pd.DataFrame()

    strange_converters = conv_diag.loc[conv_diag["converter_topology"] != "ac_dc_mixed"].copy() if not conv_diag.empty else pd.DataFrame()

    # ---------------------------------------------------------
    # DC
    # ---------------------------------------------------------
    dc_nodes = set(buses_diag.loc[buses_diag["is_dc"], "bus_id"].astype(str))
    Gdc = nx.Graph()
    Gdc.add_nodes_from(dc_nodes)

    # only links/converters/transformers/lines whose both endpoints are DC
    for D, b0c, b1c in [
        (links, ("bus0", "u"), ("bus1", "v")),
        (converters, ("bus0", "u"), ("bus1", "v")),
        (transformers, ("bus0_red", "bus0"), ("bus1_red", "bus1")),
        (lines, ("bus0", "u"), ("bus1", "v")),
    ]:
        if D is None or D.empty:
            continue
        X = D.copy()
        b0 = next((c for c in b0c if c in X.columns), None)
        b1 = next((c for c in b1c if c in X.columns), None)
        if b0 is None or b1 is None:
            continue
        X[b0] = X[b0].astype(str)
        X[b1] = X[b1].astype(str)
        for u, v in X[[b0, b1]].dropna().itertuples(index=False, name=None):
            if (u in dc_nodes) and (v in dc_nodes) and (u != v):
                Gdc.add_edge(u, v)

    dc_components = []
    for cid, comp in enumerate(nx.connected_components(Gdc)):
        comp = sorted(comp)
        dc_components.append(
            {
                "dc_component_id": cid,
                "n_buses": len(comp),
                "buses": ",".join(comp),
            }
        )
    dc_components_df = pd.DataFrame(dc_components)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    summary = pd.DataFrame(
        {
            "metric": [
                "n_buses",
                "n_isolated_buses",
                "n_dc_buses",
                "n_plants",
                "n_plants_on_dc",
                "n_plants_on_isolated",
                "n_links",
                "n_links_ac_ac",
                "n_links_dc_dc",
                "n_links_ac_dc_mixed",
                "n_converters",
                "n_strange_converters",
                "n_dc_components",
            ],
            "value": [
                len(buses_diag),
                len(isolated_buses),
                int(buses_diag["is_dc"].sum()),
                len(plants_diag),
                len(plants_on_dc),
                len(plants_on_isolated),
                len(links_diag),
                len(ac_ac_links),
                len(dc_dc_links),
                len(mixed_links),
                len(conv_diag),
                len(strange_converters),
                len(dc_components_df),
            ],
        }
    )

    return {
        "summary": summary,
        "buses_with_degree": buses_diag,
        "isolated_buses": isolated_buses,
        "plants_diagnostic": plants_diag,
        "plants_on_dc_buses": plants_on_dc,
        "plants_on_isolated_buses": plants_on_isolated,
        "links_diagnostic": links_diag,
        "mixed_ac_dc_links": mixed_links,
        "dc_dc_links": dc_dc_links,
        "ac_ac_links": ac_ac_links,
        "converters_diagnostic": conv_diag,
        "strange_converters": strange_converters,
        "dc_components": dc_components_df
    }


# ============================================================
# MAIN
# ============================================================
def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reduced-grid scenarios from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML scenario config. Without it, the historical defaults are used.",
    )
    return parser.parse_args()


def _coerce_config_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    txt = str(value).strip().lower()
    if txt in {"true", "t", "yes", "y", "on", "1"}:
        return True
    if txt in {"false", "f", "no", "n", "off", "0", ""}:
        return False
    raise ValueError(f"{name} must be a boolean-like value, got: {value!r}")


def default_grid_settings() -> dict[str, Any]:
    project_root = DEFAULT_PROJECT_ROOT
    return {
        "scenario_name": "default_grid_reduction",
        "project_root": str(project_root),
        "countries_shapefile": str(DEFAULT_COUNTRIES_URL),
        "entsoe_load_dir": str(DEFAULT_ENTSOE_LOAD_DIR),
        "country_reductions_csv": str(project_root / "country_reductions.csv"),
        "base_data_dir": str(project_root / "grid"),
        "output_root": None,
        "sync_collapse": True,
        "include_tyndp2020": True,
        "target_year": 2030,
        "tyndp_base_snapshot_year": 2025,
        "k_total": 128,
        "load_year": 2025,
        "generation_country_weight": 0.5,
        "load_country_weight": 0.5,
        "voltage_mode": "simplify_380",
        "s_base_mva": 100.0,
        "hvdc_model": "pypsa_ac_links",
        "write_raw_outputs": False,
        "raw_only": False,
        "raw_output_case_name": "raw_tyndp2020_network",
        "network_case_suffix": "",
        "snap_plot_positions_to_land": True,
        "snap_plot_max_distance_km": 20.0,
        "plot_bus_position_mode": "medoid",
        "pre_prune_tiny_ac_leaf_nodes": True,
        "pre_prune_tiny_cross_border_leaf_nodes": True,
        "prune_tiny_ac_leaf_nodes": True,
        "prune_tiny_cross_border_leaf_nodes": True,
        "tiny_ac_leaf_max_member_buses": 2,
        "tiny_ac_leaf_max_capacity_mw": 300.0,
        "aggregation_modes": ["line_equivalent"],
        "similarities": ["electrical", "dc_effective_reactance"],
        "selected_country_cluster_codes": list(DEFAULT_CESA_COUNTRY_CLUSTER_CODES),
        "excluded_country_cluster_codes": [],
        "excluded_countries": [],
        "bbox": list(DEFAULT_BBOX),
        "paths": {
            "plants": str(project_root / "powerplants" / "powerplants.csv"),
            "lines": str(project_root / "grid" / "xiong2025 v07" / "lines.csv"),
            "links": str(project_root / "grid" / "xiong2025 v07" / "links.csv"),
            "buses": str(project_root / "grid" / "xiong2025 v07" / "buses.csv"),
            "converters": str(project_root / "grid" / "xiong2025 v07" / "converters.csv"),
            "transformers": str(project_root / "grid" / "xiong2025 v07" / "transformers.csv"),
        },
    }

def resolve_grid_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = default_grid_settings()
    config_path = Path(args.config).resolve() if args.config is not None else None

    if config_path is not None:
        settings = deep_merge(settings, load_yaml_like(config_path))

    base_dir = config_path.parent if config_path is not None else None
    project_root = resolve_path(settings["project_root"], base_dir=base_dir)
    if project_root is None:
        raise ValueError("project_root could not be resolved.")

    settings["project_root"] = project_root
    settings["config_path"] = config_path
    settings["scenario_name"] = str(
        settings.get("scenario_name")
        or (config_path.stem if config_path is not None else "default_grid_reduction")
    )
    settings["countries_shapefile"] = resolve_path(settings["countries_shapefile"], base_dir=base_dir)
    settings["entsoe_load_dir"] = resolve_path(settings["entsoe_load_dir"], base_dir=base_dir)
    settings["country_reductions_csv"] = resolve_path(settings["country_reductions_csv"], base_dir=base_dir)
    settings["base_data_dir"] = resolve_path(settings["base_data_dir"], base_dir=base_dir)

    settings["output_root"] = resolve_path(settings.get("output_root"), base_dir=base_dir)
    if settings["output_root"] is None:
        if settings["include_tyndp2020"]:
            settings["output_root"] = project_root / "modified" / f"target_year_{settings['target_year']}"
        else:
            settings["output_root"] = project_root / "modified"
    settings["write_raw_outputs"] = _coerce_config_bool(
        settings.get("write_raw_outputs", False),
        name="write_raw_outputs",
    )
    settings["raw_only"] = _coerce_config_bool(
        settings.get("raw_only", False),
        name="raw_only",
    )
    if settings["raw_only"] and not settings["write_raw_outputs"]:
        raise ValueError("raw_only requires write_raw_outputs: true")
    settings["raw_output_case_name"] = str(
        settings.get("raw_output_case_name") or "raw_tyndp2020_network"
    ).strip()
    if not settings["raw_output_case_name"]:
        raise ValueError("raw_output_case_name must not be empty")
    settings["network_case_suffix"] = str(settings.get("network_case_suffix") or "").strip()
    if settings["network_case_suffix"] and not re.fullmatch(r"[A-Za-z0-9_.-]+", settings["network_case_suffix"]):
        raise ValueError("network_case_suffix may only contain letters, numbers, '_', '-' and '.'.")
    settings["snap_plot_positions_to_land"] = _coerce_config_bool(
        settings.get("snap_plot_positions_to_land", True),
        name="snap_plot_positions_to_land",
    )
    settings["snap_plot_max_distance_km"] = float(settings.get("snap_plot_max_distance_km", 20.0))
    if settings["snap_plot_max_distance_km"] < 0:
        raise ValueError("snap_plot_max_distance_km must be non-negative")
    settings["prune_tiny_ac_leaf_nodes"] = _coerce_config_bool(
        settings.get("prune_tiny_ac_leaf_nodes", True),
        name="prune_tiny_ac_leaf_nodes",
    )
    settings["prune_tiny_cross_border_leaf_nodes"] = _coerce_config_bool(
        settings.get("prune_tiny_cross_border_leaf_nodes", True),
        name="prune_tiny_cross_border_leaf_nodes",
    )
    settings["plot_bus_position_mode"] = str(settings.get("plot_bus_position_mode", "medoid")).strip().lower()
    if settings["plot_bus_position_mode"] not in {"mean", "medoid"}:
        raise ValueError("plot_bus_position_mode must be 'mean' or 'medoid'")
    settings["pre_prune_tiny_ac_leaf_nodes"] = _coerce_config_bool(
        settings.get("pre_prune_tiny_ac_leaf_nodes", True),
        name="pre_prune_tiny_ac_leaf_nodes",
    )
    settings["pre_prune_tiny_cross_border_leaf_nodes"] = _coerce_config_bool(
        settings.get("pre_prune_tiny_cross_border_leaf_nodes", True),
        name="pre_prune_tiny_cross_border_leaf_nodes",
    )
    settings["tiny_ac_leaf_max_member_buses"] = int(settings.get("tiny_ac_leaf_max_member_buses", 2))
    settings["tiny_ac_leaf_max_capacity_mw"] = float(settings.get("tiny_ac_leaf_max_capacity_mw", 300.0))

    resolved_paths: dict[str, Path] = {}
    for key, value in settings["paths"].items():
        path = resolve_path(value, base_dir=base_dir)
        if path is None:
            raise ValueError(f"Could not resolve input path for '{key}'.")
        resolved_paths[key] = path
    settings["paths"] = resolved_paths

    settings["aggregation_modes"] = [str(value).strip() for value in ensure_list(settings["aggregation_modes"])]
    settings["similarities"] = [str(value).strip() for value in ensure_list(settings["similarities"])]
    settings["selected_country_cluster_codes"] = [
        str(value).strip().upper()
        for value in ensure_list(settings["selected_country_cluster_codes"])
        if str(value).strip()
    ]
    settings["excluded_country_cluster_codes"] = [
        str(value).strip().upper()
        for value in ensure_list(settings.get("excluded_country_cluster_codes"))
        if str(value).strip()
    ]
    settings["excluded_countries"] = [
        str(value).strip()
        for value in ensure_list(settings.get("excluded_countries"))
        if str(value).strip()
    ]
    bbox = tuple(float(value) for value in ensure_list(settings["bbox"]))
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four numbers.")
    settings["bbox"] = bbox
    return settings


def build_methods_for_similarity(similarity: str) -> list[tuple[str, dict[str, Any]]]:
    feature = "coords" if similarity == "geographical" else "spectral"
    electrical_weight_mode = None if similarity == "dc_effective_reactance" else "b"
    return [
        (
            "electrical_spectral",
            dict(
                similarity=similarity,
                electrical_weight_mode=electrical_weight_mode,
                effrx_k_nearest=12,
                effrx_kernel="exp",
                effrx_tau_scale=1.0,
            ),
        ),
        (
            "hac",
            dict(
                hac_feature=feature,
                similarity=similarity,
                electrical_weight_mode=electrical_weight_mode,
                effrx_k_nearest=12,
                effrx_kernel="exp",
                effrx_tau_scale=1.0,
            ),
        ),
    ]


def write_raw_grid_outputs(
    *,
    out_dir: Path,
    settings: dict[str, Any],
    excluded_country_rows: pd.DataFrame,
    exclusion_summary: dict[str, Any],
    bbox: tuple[float, float, float, float],
    countries_path: str | Path | None,
    snap_plot_positions_to_land: bool,
    snap_plot_max_distance_km: float,
    buses: pd.DataFrame,
    lines: pd.DataFrame,
    links: pd.DataFrame,
    converters: pd.DataFrame,
    transformers: pd.DataFrame,
    plants_assigned: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    buses_out = _raw_network_buses_for_disaggregation(buses, lines)
    buses_with_clusters = _raw_network_buses_with_clusters(buses_out)
    bus_map = pd.DataFrame(
        {
            "bus_id": buses_out["bus_id"].astype(str),
            "bus_id_red": buses_out["bus_id"].astype(str),
        }
    )

    buses_out.to_csv(out_dir / "buses.csv", sep=";", index=False)
    buses_with_clusters.to_csv(out_dir / "buses_with_clusters.csv", sep=";", index=False)
    bus_map.to_csv(out_dir / "bus_map.csv", sep=";", index=False)
    _ensure_grid_source(lines, default_source=GRID_SOURCE_BASE).to_csv(
        out_dir / "lines.csv", sep=";", index=False
    )
    _ensure_grid_source(links, default_source=GRID_SOURCE_BASE).to_csv(
        out_dir / "links.csv", sep=";", index=False
    )
    converters.to_csv(out_dir / "converters.csv", sep=";", index=False)
    transformers.to_csv(out_dir / "transformers.csv", sep=";", index=False)
    plants_assigned.to_csv(out_dir / "plants.csv", sep=";", index=False)
    plants_assigned.to_csv(out_dir / "plants_with_bus.csv", sep=";", index=False)
    excluded_country_rows.to_csv(out_dir / "excluded_countries.csv", sep=";", index=False)

    raw_manifest = {
        "scenario_name": settings["scenario_name"],
        "config_path": str(settings["config_path"]) if settings["config_path"] is not None else None,
        "network_case": settings["raw_output_case_name"],
        "network_kind": "raw_target_year_network",
        "target_year": int(settings["target_year"]),
        "include_tyndp2020": bool(settings["include_tyndp2020"]),
        "raw_only": bool(settings["raw_only"]),
        "tyndp_base_snapshot_year": int(settings["tyndp_base_snapshot_year"]),
        "excluded_country_cluster_codes": settings["excluded_country_cluster_codes"],
        "excluded_countries": settings["excluded_countries"],
        "country_exclusion_summary": exclusion_summary,
        "source_grid_paths": {key: str(value) for key, value in settings["paths"].items()},
        "notes": [
            "Each raw bus is its own disaggregation target.",
            "buses_with_clusters.csv uses cluster_id = bus_id for compatibility with existing disaggregation workflows.",
            "No country-cluster mapping is written; model countries remain physical bus countries.",
        ],
        "outputs": {
            "buses": str(out_dir / "buses.csv"),
            "buses_with_clusters": str(out_dir / "buses_with_clusters.csv"),
            "bus_map": str(out_dir / "bus_map.csv"),
            "lines": str(out_dir / "lines.csv"),
            "links": str(out_dir / "links.csv"),
            "converters": str(out_dir / "converters.csv"),
            "transformers": str(out_dir / "transformers.csv"),
            "plants": str(out_dir / "plants.csv"),
            "excluded_countries": str(out_dir / "excluded_countries.csv"),
        },
    }
    (out_dir / "scenario_manifest.json").write_text(
        json.dumps(raw_manifest, indent=2),
        encoding="utf-8",
    )

    plot_plants_map(
        out_svg=out_dir / "raw_plants_map.svg",
        title="Raw Plants Map (network + plants)",
        buses=buses,
        lines=lines,
        links=links,
        plants=plants_assigned,
        converters=converters,
        transformers=transformers,
        buses_are_latlon=False,
        plant_bus_col="assigned_bus",
        bbox=bbox,
        countries_path=countries_path,
        snap_plot_positions_to_land=snap_plot_positions_to_land,
        snap_plot_max_distance_km=snap_plot_max_distance_km,
        downsample_lines=250000,
        downsample_links=50000,
    )

    plot_voltage_map(
        out_svg=out_dir / "raw_voltage_map.svg",
        title="Raw Voltage Map (AC by kV + HVDC dashed)",
        buses=buses,
        lines=lines,
        links=links,
        converters=converters,
        transformers=transformers,
        buses_are_latlon=False,
        bbox=bbox,
        countries_path=countries_path,
        snap_plot_positions_to_land=snap_plot_positions_to_land,
        snap_plot_max_distance_km=snap_plot_max_distance_km,
        downsample_lines=250000,
        downsample_links=50000,
    )


def _raw_network_buses_for_disaggregation(buses: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    prepped = add_sync_area_to_buses(_prep_buses(buses))
    prepped = _fill_missing_ac_sub_networks(prepped, lines)
    out = pd.DataFrame(
        {
            "bus_id": prepped["bus_id"].astype(str),
            "voltage": prepped["voltage"].astype(int),
            "voltage_min": prepped["voltage"].astype(int),
            "voltage_max": prepped["voltage"].astype(int),
            "n_voltage_levels": 1,
            "dc": np.where(prepped["dc_bool"], "t", "f"),
            "lat": prepped["lat"].astype(float),
            "lon": prepped["lon"].astype(float),
            "country": prepped["country"].astype(str),
            "country_label": prepped["country"].astype(str),
            "original_country": prepped["country"].astype(str),
            "n_original_countries": 1,
            "sync_area": prepped["sync_area"].astype(str),
            "sub_network": prepped["sub_network"].astype("object"),
            "n_buses_simplified": 1,
        }
    )
    return out.sort_values("bus_id").reset_index(drop=True)


def _raw_network_buses_with_clusters(buses_out: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bus_id": buses_out["bus_id"].astype(str),
            "voltage": buses_out["voltage"].astype(int),
            "dc_bool": buses_out["dc"].astype(str).str.lower().isin(["t", "true", "1", "yes"]),
            "cluster_id": buses_out["bus_id"].astype(str),
            "lat": buses_out["lat"].astype(float),
            "lon": buses_out["lon"].astype(float),
            "country": buses_out["country"].astype(str),
            "sync_area": buses_out["sync_area"].astype(str),
            "sub_network": buses_out["sub_network"].astype("object"),
        }
    )


def write_scenario_manifest(
    out_dir: Path,
    settings: dict[str, Any],
    run_name: str,
    method: str,
    aggregation_mode: str,
    similarity: str,
    method_kwargs: dict[str, Any],
    exclusion_summary: dict[str, Any],
) -> None:
    manifest = {
        "scenario_name": settings["scenario_name"],
        "config_path": str(settings["config_path"]) if settings["config_path"] is not None else None,
        "run_name": run_name,
        "method": method,
        "aggregation_mode": aggregation_mode,
        "similarity": similarity,
        "target_year": settings["target_year"],
        "k_total": settings["k_total"],
        "network_case_suffix": settings["network_case_suffix"],
        "voltage_mode": settings["voltage_mode"],
        "hvdc_model": settings["hvdc_model"],
        "write_raw_outputs": settings["write_raw_outputs"],
        "snap_plot_positions_to_land": settings["snap_plot_positions_to_land"],
        "snap_plot_max_distance_km": settings["snap_plot_max_distance_km"],
        "plot_bus_position_mode": settings["plot_bus_position_mode"],
        "pre_prune_tiny_ac_leaf_nodes": settings["pre_prune_tiny_ac_leaf_nodes"],
        "pre_prune_tiny_cross_border_leaf_nodes": settings["pre_prune_tiny_cross_border_leaf_nodes"],
        "prune_tiny_ac_leaf_nodes": settings["prune_tiny_ac_leaf_nodes"],
        "prune_tiny_cross_border_leaf_nodes": settings["prune_tiny_cross_border_leaf_nodes"],
        "tiny_ac_leaf_max_member_buses": settings["tiny_ac_leaf_max_member_buses"],
        "tiny_ac_leaf_max_capacity_mw": settings["tiny_ac_leaf_max_capacity_mw"],
        "include_tyndp2020": settings["include_tyndp2020"],
        "selected_country_cluster_codes": settings["selected_country_cluster_codes"],
        "excluded_country_cluster_codes": settings["excluded_country_cluster_codes"],
        "excluded_countries": settings["excluded_countries"],
        "country_exclusion_summary": exclusion_summary,
        "country_reductions_csv": str(settings["country_reductions_csv"]),
        "output_dir": str(out_dir),
        "network_dir": str(out_dir),
        "method_kwargs": method_kwargs,
        "paths": {key: str(value) for key, value in settings["paths"].items()},
    }
    (out_dir / "scenario_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def run_grid_reduction(settings: dict[str, Any]) -> None:
    global COUNTRIES_URL
    COUNTRIES_URL = str(settings["countries_shapefile"])

    output_root = settings["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    country_reductions_all = read_country_reductions_csv(
        settings["country_reductions_csv"],
        selected_codes=None,
    )
    cesa_country_clusters = read_country_reductions_csv(
        settings["country_reductions_csv"],
        selected_codes=settings["selected_country_cluster_codes"],
    )
    excluded_country_rows = build_excluded_country_rows(
        country_reductions_all,
        excluded_country_cluster_codes=settings["excluded_country_cluster_codes"],
        excluded_countries=settings["excluded_countries"],
    )
    cesa_country_clusters = filter_country_clusters_for_exclusions(
        cesa_country_clusters,
        excluded_country_rows,
        settings["excluded_country_cluster_codes"],
    )

    plants = filter_plants_in_operation(
        read_plants_csv(settings["paths"]["plants"]),
        target_year=settings["target_year"],
    )
    buses = read_net_csv(settings["paths"]["buses"])
    lines = read_net_csv(settings["paths"]["lines"])
    links = read_net_csv(settings["paths"]["links"])
    conv = read_net_csv(settings["paths"]["converters"])
    trafo = read_net_csv(settings["paths"]["transformers"])

    if settings["include_tyndp2020"]:
        lines, links = integrate_tyndp2020_projects(
            buses,
            lines,
            links,
            base_data=settings["base_data_dir"],
            target_year=settings["target_year"],
            base_snapshot_year=settings["tyndp_base_snapshot_year"],
        )
    else:
        lines = _ensure_grid_source(lines, default_source=GRID_SOURCE_BASE)
        links = _ensure_grid_source(links, default_source=GRID_SOURCE_BASE)

    buses, lines, links, conv, trafo, plants, exclusion_summary = apply_country_exclusions_to_raw_network(
        buses=buses,
        lines=lines,
        links=links,
        converters=conv,
        transformers=trafo,
        plants=plants,
        excluded_country_rows=excluded_country_rows,
    )

    plants_assigned = assign_plants_to_buses(plants, buses)
    buses0 = add_sync_area_to_buses(_prep_buses(buses))
    plants_assigned = _drop_existing_plant_bus_metadata(plants_assigned)
    plants_assigned = plants_assigned.merge(
        buses0[["bus_id", "sync_area", "sync_node"]],
        left_on="assigned_bus",
        right_on="bus_id",
        how="left",
        suffixes=("", "_bus"),
    ).drop(columns=["bus_id"])

    if settings["write_raw_outputs"]:
        write_raw_grid_outputs(
            out_dir=output_root / settings["raw_output_case_name"],
            settings=settings,
            excluded_country_rows=excluded_country_rows,
            exclusion_summary=exclusion_summary,
            bbox=settings["bbox"],
            countries_path=settings["countries_shapefile"],
            snap_plot_positions_to_land=settings["snap_plot_positions_to_land"],
            snap_plot_max_distance_km=settings["snap_plot_max_distance_km"],
            buses=buses,
            lines=lines,
            links=links,
            converters=conv,
            transformers=trafo,
            plants_assigned=plants_assigned,
        )

    if settings["raw_only"]:
        return

    country_load_mean = read_entsoe_country_mean_load(
        settings["entsoe_load_dir"],
        year=settings["load_year"],
        excluded_map_codes=ENTSOE_LOAD_EXCLUDED_MAP_CODES,
        target_countries=sorted(buses0["country"].astype(str).str.upper().unique()),
    )

    for similarity in settings["similarities"]:
        for method, method_kwargs in build_methods_for_similarity(similarity):
            for aggregation_mode in settings["aggregation_modes"]:
                run_name = f"{method}_{aggregation_mode}_{similarity}{settings['network_case_suffix']}"
                out_dir = output_root / run_name
                out_dir.mkdir(parents=True, exist_ok=True)

                red = reduce_network(
                    buses=buses,
                    lines=lines,
                    links=links,
                    converters=conv,
                    transformers=trafo,
                    plants_assigned=plants_assigned,
                    k_total=settings["k_total"],
                    method=method,
                    voltage_mode=settings["voltage_mode"],
                    s_base_mva=settings["s_base_mva"],
                    sync_collapse=settings["sync_collapse"],
                    aggregation_mode=aggregation_mode,
                    line_length_factor=1.25,
                    hvdc_model=settings["hvdc_model"],
                    country_load_mean=country_load_mean,
                    generation_country_weight=settings["generation_country_weight"],
                    load_country_weight=settings["load_country_weight"],
                    cesa_country_clusters=cesa_country_clusters,
                    plot_bus_position_mode=settings["plot_bus_position_mode"],
                    pre_prune_tiny_ac_leaf_nodes=settings["pre_prune_tiny_ac_leaf_nodes"],
                    pre_prune_tiny_cross_border_leaf_nodes=settings["pre_prune_tiny_cross_border_leaf_nodes"],
                    prune_tiny_ac_leaf_nodes=settings["prune_tiny_ac_leaf_nodes"],
                    prune_tiny_cross_border_leaf_nodes=settings["prune_tiny_cross_border_leaf_nodes"],
                    tiny_ac_leaf_max_member_buses=settings["tiny_ac_leaf_max_member_buses"],
                    tiny_ac_leaf_max_capacity_mw=settings["tiny_ac_leaf_max_capacity_mw"],
                    **method_kwargs,
                )

                for name, df in red.items():
                    if isinstance(df, pd.DataFrame):
                        df.to_csv(out_dir / f"{name}.csv", sep=";", index=False)
                excluded_country_rows.to_csv(out_dir / "excluded_countries.csv", sep=";", index=False)

                diag = diagnose_reduced_network(red)
                for name, df in diag.items():
                    if isinstance(df, pd.DataFrame):
                        os.makedirs(out_dir / "diagnostic", exist_ok=True)
                        df.to_csv(out_dir / "diagnostic" / f"diagnostic_{name}.csv", sep=";", index=False)

                plot_plants_map(
                    out_svg=out_dir / f"reduced_plants_map_{run_name}.svg",
                    title=f"Reduced Power Plants Map from: {run_name}",
                    buses=red["buses"],
                    lines=red["lines"],
                    links=red["links"],
                    plants=red["plants"],
                    converters=red["converters"],
                    transformers=red["transformers"],
                    buses_are_latlon=True,
                    plant_bus_col="bus_id",
                    bbox=settings["bbox"],
                    countries_path=settings["countries_shapefile"],
                    snap_plot_positions_to_land=settings["snap_plot_positions_to_land"],
                    snap_plot_max_distance_km=settings["snap_plot_max_distance_km"],
                )

                plot_voltage_map(
                    out_svg=out_dir / f"reduced_voltage_map_{run_name}.svg",
                    title=f"Reduced Grid Map from: {run_name}",
                    buses=red["buses"],
                    lines=red["lines"],
                    links=red["links"],
                    converters=red["converters"],
                    transformers=red["transformers"],
                    buses_are_latlon=True,
                    bbox=settings["bbox"],
                    countries_path=settings["countries_shapefile"],
                    snap_plot_positions_to_land=settings["snap_plot_positions_to_land"],
                    snap_plot_max_distance_km=settings["snap_plot_max_distance_km"],
                )

                write_scenario_manifest(
                    out_dir=out_dir,
                    settings=settings,
                    run_name=run_name,
                    method=method,
                    aggregation_mode=aggregation_mode,
                    similarity=similarity,
                    method_kwargs=method_kwargs,
                    exclusion_summary=exclusion_summary,
                )
                print(f"\nStored reduction results in: {out_dir}")


def main() -> None:
    args = parse_cli_args()
    settings = resolve_grid_settings(args)
    run_grid_reduction(settings)


if __name__ == "__main__":
    main()
