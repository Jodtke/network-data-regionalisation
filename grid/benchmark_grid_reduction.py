# -*- coding: utf-8 -*-
"""
Benchmark and evaluation of precomputed network reductions for a future
frequency-constrained unit commitment with DC-OPF and inertia-neighbourhood logic.

Main idea
---------
For every reduction run (folder with bus_map.csv, buses.csv, lines.csv, plants.csv),
the script evaluates:

1) Reduction benefit
   - number of buses / lines
   - reduction ratios

2) Corridor preservation
   - exact inter-cluster corridor susceptance and capacity derived from the raw network
   - comparison against the reduced line model

3) DC transfer fidelity
   - random cluster-to-cluster transfer tests
   - raw-network DC flows are solved on the original AC graph
   - reduced-network DC flows are solved on the reduced graph
   - both are compared on aggregated inter-cluster corridor flows

4) Electrical-distance / inertia suitability
   - effective-reactance distance matrix on the exact clustered raw graph
   - effective-reactance distance matrix on the reduced graph
   - inertia-density proxy based on synchronous capacity and electrical proximity

5) Cluster compactness
   - geographic radius / diameter of original buses inside each reduced cluster

The ranking is therefore tailored to a reduced network that will later be used in:
    DC-OPF + frequency constraints + inertia/proximity index
and not just for generic graph compression.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import factorized


# ============================================================
# Configuration
# ============================================================

@dataclass
class EvalConfig:
    raw_buses_csv: Path
    raw_lines_csv: Path
    plants_with_bus_csv: Path
    reductions_root: Path
    out_dir: Path

    # Scope of the benchmark
    evaluate_only_cesa_ac: bool = True

    # Transfer benchmark
    n_transfer_samples: int = 250
    transfer_injection_mw: float = 1000.0
    random_seed: int = 42

    # Inertia / proximity benchmark
    proximity_kernel: str = "exp"     # exp | reciprocal
    proximity_tau_scale: float = 1.0
    
    # New config for multi-injection/PTDF-like benchmark
    multiinj_k_active_min: int = 3
    multiinj_k_active_max: int = 8
    multiinj_injection_mw: float = 1000.0

    # Composite score weights
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        # existing electrical
         "transfer_nrmse_mean": 0.20,
         "transfer_cosine_mean": 0.07,
         "multiinj_nrmse_mean": 0.15,
         "multiinj_cosine_mean": 0.08,
         "inertia_proxy_nrmse": 0.18,
         "effrx_distance_nrmse": 0.10,
         "corridor_b_wape": 0.05,
        
         # mesh / loop realism
         "cluster_cycle_rank_raw": 0.05,
         "cluster_lambda2_weighted_raw": 0.05,
         "cluster_kirchhoff_weighted_abs_pct_error": 0.04,
        
         "geo_radius95_km_mean": 0.02,
         "bus_reduction_ratio": 0.01
    })

    # Heuristic list of synchronous technologies/fuels
    sync_fuel_whitelist: Tuple[str, ...] = (
        "Hydro", "Hard Coal", "Natural Gas", "Lignite", "Oil",
        "Solid Biomass", "Waste", "Geothermal", "Nuclear", "Biogas",
        "Mechanical Storage",   # e.g. pumped hydro
    )
    sync_technology_blacklist: Tuple[str, ...] = (
        "Battery", "Solar", "Wind", "PV", "Photovoltaic",
        "Hydrogen Storage", "Heat Storage",
    )


# ============================================================
# I/O helpers
# ============================================================
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def read_net_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",", quotechar="'", engine="python", low_memory=True)
    return _clean_columns(df)


def read_semicolon_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", low_memory=True)
    return _clean_columns(df)


def read_run_csv(path: Path, expected_cols: set[str] | None = None) -> pd.DataFrame:
    """
    Robust reader for the already exported reduction result files in each run folder.
    Tries multiple parsings and returns the first one that matches expected columns.
    """
    path = Path(path)
    tried: list[pd.DataFrame] = []

    candidates = [
        dict(sep=";", low_memory=True),
        dict(sep=",", low_memory=True),
        dict(sep=",", quotechar="'", engine="python", low_memory=True),
    ]

    for kwargs in candidates:
        try:
            df = pd.read_csv(path, **kwargs)
            df = _clean_columns(df)
            tried.append(df)

            if expected_cols is None:
                return df

            cols = set(df.columns)
            if len(expected_cols & cols) > 0:
                return df

        except Exception:
            continue

    if tried:
        # fallback: take parse with most columns
        best = max(tried, key=lambda x: x.shape[1])
        return best

    raise ValueError(f"Could not parse file: {path}")


def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def find_run_dirs(root: Path) -> List[Path]:
    root = Path(root)
    run_dirs: List[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        req = ["bus_map.csv", "buses.csv", "lines.csv", "plants.csv"]
        if all((p / f).exists() for f in req):
            run_dirs.append(p)
    return run_dirs


# ============================================================
# Generic preprocessing
# ============================================================

def prep_raw_buses(df: pd.DataFrame) -> pd.DataFrame:
    b = df.copy()
    b["bus_id"] = b["bus_id"].astype(str)

    if "x" in b.columns:
        b["lon"] = pd.to_numeric(b["x"], errors="coerce")
    elif "lon" in b.columns:
        b["lon"] = pd.to_numeric(b["lon"], errors="coerce")
    else:
        b["lon"] = np.nan

    if "y" in b.columns:
        b["lat"] = pd.to_numeric(b["y"], errors="coerce")
    elif "lat" in b.columns:
        b["lat"] = pd.to_numeric(b["lat"], errors="coerce")
    else:
        b["lat"] = np.nan

    if "dc" in b.columns:
        b["dc_bool"] = b["dc"].astype(str).str.lower().isin(["t", "true", "1", "yes", "y"])
    elif "dc_bool" not in b.columns:
        b["dc_bool"] = False

    if "country" in b.columns:
        b["country"] = b["country"].astype(str).str.upper()

    return b


def prep_raw_lines(df: pd.DataFrame) -> pd.DataFrame:
    L = df.copy()
    if "line_id" not in L.columns:
        L["line_id"] = [f"line_{i}" for i in range(len(L))]
    L["line_id"] = L["line_id"].astype(str)
    L["bus0"] = L["bus0"].astype(str)
    L["bus1"] = L["bus1"].astype(str)

    for col in ["x", "r", "b", "s_nom", "circuits", "length", "voltage"]:
        if col in L.columns:
            L[col] = pd.to_numeric(L[col], errors="coerce")

    if "circuits" not in L.columns:
        L["circuits"] = 1.0
    L["circuits"] = pd.to_numeric(L["circuits"], errors="coerce").fillna(1.0)

    return L


def prep_run_buses(df: pd.DataFrame) -> pd.DataFrame:
    b = df.copy()

    if "bus_id" not in b.columns:
        raise KeyError(f"Run buses file missing column 'bus_id'. Parsed columns were: {list(b.columns)}")

    b["bus_id"] = b["bus_id"].astype(str)

    if "dc_bool" in b.columns:
        pass
    elif "dc" in b.columns:
        b["dc_bool"] = b["dc"].astype(str).str.lower().isin(["t", "true", "1", "yes", "y"])
    else:
        b["dc_bool"] = False

    if "lat" in b.columns:
        b["lat"] = pd.to_numeric(b["lat"], errors="coerce")
    if "lon" in b.columns:
        b["lon"] = pd.to_numeric(b["lon"], errors="coerce")

    if "sync_area" in b.columns:
        b["sync_area"] = b["sync_area"].astype(str)

    return b


def prep_run_lines(df: pd.DataFrame) -> pd.DataFrame:
    L = df.copy()
    if "u" in L.columns and "bus0" not in L.columns:
        L = L.rename(columns={"u": "bus0", "v": "bus1"})
    if "line_id" not in L.columns:
        L["line_id"] = [f"redline_{i}" for i in range(len(L))]
    L["line_id"] = L["line_id"].astype(str)
    L["bus0"] = L["bus0"].astype(str)
    L["bus1"] = L["bus1"].astype(str)

    for col in ["x_eq", "r_eq", "b", "s_nom", "circuits", "length", "voltage"]:
        if col in L.columns:
            L[col] = pd.to_numeric(L[col], errors="coerce")
    return L


def prep_plants_with_bus(df: pd.DataFrame) -> pd.DataFrame:
    P = df.copy()
    if "assigned_bus" in P.columns:
        P["assigned_bus"] = P["assigned_bus"].astype(str)
    if "bus_id" in P.columns:
        P["bus_id"] = P["bus_id"].astype(str)
    if "Fueltype" in P.columns:
        P["Fueltype"] = P["Fueltype"].astype(str)
    if "Technology" in P.columns:
        P["Technology"] = P["Technology"].astype(str)
    if "Capacity" in P.columns:
        P["Capacity"] = pd.to_numeric(P["Capacity"], errors="coerce").fillna(0.0)
    return P


# ============================================================
# Small math helpers
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0088
    lat1r = np.deg2rad(np.asarray(lat1, dtype=float))
    lon1r = np.deg2rad(np.asarray(lon1, dtype=float))
    lat2r = np.deg2rad(np.asarray(lat2, dtype=float))
    lon2r = np.deg2rad(np.asarray(lon2, dtype=float))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def safe_div(a: float, b: float, default: float = np.nan) -> float:
    return float(a / b) if (b is not None and np.isfinite(b) and abs(b) > 1e-12) else float(default)


def weighted_abs_pct_error(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if weights is None:
        weights = np.abs(y_true)
    weights = np.asarray(weights, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(weights) & (weights >= 0)
    if not np.any(m):
        return np.nan
    num = np.sum(weights[m] * np.abs(y_pred[m] - y_true[m]))
    den = np.sum(weights[m] * np.abs(y_true[m]))
    return float(num / den) if den > 0 else np.nan


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return np.nan
    num = np.linalg.norm(y_pred[m] - y_true[m])
    den = np.linalg.norm(y_true[m])
    return float(num / den) if den > 1e-12 else np.nan


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan
    na = np.linalg.norm(a[m])
    nb = np.linalg.norm(b[m])
    if na <= 1e-12 or nb <= 1e-12:
        return np.nan
    return float(np.dot(a[m], b[m]) / (na * nb))


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 2:
        return np.nan
    if np.std(a[m]) <= 1e-12 or np.std(b[m]) <= 1e-12:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def upper_tri_values(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("Matrix must be square.")
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


# ============================================================
# Synchronous capacity logic
# ============================================================

def is_sync_plant(row: pd.Series, cfg: EvalConfig) -> bool:
    fuel = str(row.get("Fueltype", ""))
    tech = str(row.get("Technology", ""))
    if any(x.lower() in tech.lower() for x in cfg.sync_technology_blacklist):
        return False
    if fuel in cfg.sync_fuel_whitelist:
        return True
    return False


# ============================================================
# Graph / DC helpers
# ============================================================

def line_susceptance_from_raw(lines: pd.DataFrame) -> np.ndarray:
    x = pd.to_numeric(lines["x"], errors="coerce").values.astype(float)
    circuits = pd.to_numeric(lines.get("circuits", 1.0), errors="coerce").fillna(1.0).values.astype(float)
    out = np.zeros(len(lines), dtype=float)
    ok = np.isfinite(x) & (np.abs(x) > 1e-9)
    out[ok] = circuits[ok] / np.abs(x[ok])
    return out


def line_susceptance_from_reduced(lines: pd.DataFrame) -> np.ndarray:
    """
    Convert reduced line records to corridor susceptance.

    Important:
    - For `line_equivalent` outputs (`line_id` usually starts with "eq_"),
      x_eq is already the equivalent reactance of the whole parallel corridor,
      so susceptance should be 1 / |x_eq|.
    - For `location_based` outputs (`line_id` usually starts with "loc_"),
      x_eq behaves more like a representative single-line reactance and the
      summed `circuits` should still scale the corridor susceptance.

    If no clear prefix is available, the function falls back to the conservative
    assumption that x_eq is already equivalent.
    """
    xeq = pd.to_numeric(lines["x_eq"], errors="coerce").values.astype(float)
    circuits = pd.to_numeric(lines.get("circuits", 1.0), errors="coerce").fillna(1.0).values.astype(float)

    line_ids = lines.get("line_id", pd.Series([""] * len(lines))).astype(str).values
    use_circuits = np.array([lid.startswith("loc_") for lid in line_ids], dtype=bool)

    out = np.zeros(len(lines), dtype=float)
    ok = np.isfinite(xeq) & (np.abs(xeq) > 1e-9)

    # default: x_eq is already an equivalent reactance
    out[ok] = 1.0 / np.abs(xeq[ok])

    # for location-based aggregation, circuits still scale the corridor
    ok_loc = ok & use_circuits
    out[ok_loc] = circuits[ok_loc] / np.abs(xeq[ok_loc])
    return out


def build_sparse_bbus_from_edges(nodes: List[str], edges: pd.DataFrame, b_col: str) -> Tuple[sparse.csr_matrix, Dict[str, int]]:
    idx = {n: i for i, n in enumerate(nodes)}
    rows = []
    cols = []
    data = []

    bus0 = edges["bus0"].astype(str).values
    bus1 = edges["bus1"].astype(str).values

    bdat = edges[b_col]
    if isinstance(bdat, pd.DataFrame):
        # if duplicate column names exist, take the first one explicitly
        bvals = pd.to_numeric(bdat.iloc[:, 0], errors="coerce").values.astype(float)
    else:
        bvals = pd.to_numeric(bdat, errors="coerce").values.astype(float)

    for u, v, b in zip(bus0, bus1, bvals):
        if u == v:
            continue
        if (u not in idx) or (v not in idx):
            continue
        if not np.isfinite(b) or b <= 0:
            continue
        i = idx[u]
        j = idx[v]
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([float(b), float(b)])

    n = len(nodes)
    if not rows:
        return sparse.csr_matrix((n, n)), idx

    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    W.sum_duplicates()
    d = np.asarray(W.sum(axis=1)).ravel()
    B = sparse.diags(d) - W
    return B.tocsr(), idx


def factorize_component_bbus(B: sparse.csr_matrix) -> Tuple[Any, int]:
    n = B.shape[0]
    if n <= 1:
        raise ValueError("Cannot factorize trivial component.")
    slack = 0
    keep = np.arange(n) != slack
    Breduced = B[keep][:, keep].tocsc()
    solve = factorized(Breduced)
    return solve, slack


def solve_dc_angles_factorized(solve, slack: int, p: np.ndarray) -> np.ndarray:
    n = len(p)
    theta = np.zeros(n, dtype=float)
    keep = np.arange(n) != slack
    rhs = p[keep]
    theta[keep] = solve(rhs)
    theta[slack] = 0.0
    return theta


def dense_effective_reactance_distance(nodes: List[str], edges: pd.DataFrame, b_col: str) -> np.ndarray:
    B, _ = build_sparse_bbus_from_edges(nodes, edges, b_col=b_col)
    n = B.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)
    if B.nnz == 0:
        return np.zeros((n, n), dtype=float)

    L = B.toarray().astype(float)
    Ldag = np.linalg.pinv(L, hermitian=True)
    diag = np.diag(Ldag)
    D = diag[:, None] + diag[None, :] - 2.0 * Ldag
    D = np.maximum(D, 0.0)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def proximity_matrix_from_distance(D: np.ndarray, *, kernel: str, tau_scale: float) -> np.ndarray:
    D = np.asarray(D, dtype=float).copy()
    #n = D.shape[0]
    pos = D[np.isfinite(D) & (D > 0)]
    tau = float(np.median(pos)) * float(tau_scale) if pos.size else 1.0
    if not np.isfinite(tau) or tau <= 0:
        tau = 1.0

    if kernel == "exp":
        K = np.exp(-D / tau)
    elif kernel == "reciprocal":
        K = 1.0 / np.maximum(D, 1e-9)
    else:
        raise ValueError("kernel must be 'exp' or 'reciprocal'")

    np.fill_diagonal(K, 1.0)
    K[~np.isfinite(K)] = 0.0

    rs = K.sum(axis=1, keepdims=True)
    rs[rs <= 1e-12] = 1.0
    K = K / rs
    return K


# ============================================================
# Exact clustered-raw corridor graph
# ============================================================

def build_exact_cluster_corridors(
    raw_lines: pd.DataFrame,
    bus_map: pd.DataFrame,
    *,
    evaluated_bus_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    bm = bus_map.copy()
    bm["bus_id"] = bm["bus_id"].astype(str)
    bm["bus_id_red"] = bm["bus_id_red"].astype(str)
    mp = bm.set_index("bus_id")["bus_id_red"]

    L = raw_lines.copy()
    L["bus0"] = L["bus0"].astype(str)
    L["bus1"] = L["bus1"].astype(str)

    if evaluated_bus_ids is not None:
        evaluated_bus_ids = set(map(str, evaluated_bus_ids))
        L = L[L["bus0"].isin(evaluated_bus_ids) & L["bus1"].isin(evaluated_bus_ids)].copy()

    L["cl0"] = L["bus0"].map(mp)
    L["cl1"] = L["bus1"].map(mp)
    L = L.dropna(subset=["cl0", "cl1"]).copy()
    L = L[L["cl0"] != L["cl1"]].copy()

    if L.empty:
        return pd.DataFrame(columns=["bus0", "bus1", "b_raw", "s_nom_raw", "n_lines_raw"])

    b = line_susceptance_from_raw(L)
    sn = pd.to_numeric(L.get("s_nom", 0.0), errors="coerce").fillna(0.0).values.astype(float)

    u = np.minimum(L["cl0"].astype(str).values, L["cl1"].astype(str).values)
    v = np.maximum(L["cl0"].astype(str).values, L["cl1"].astype(str).values)

    T = pd.DataFrame({
        "bus0": u,
        "bus1": v,
        "b_raw": b,
        "s_nom_raw": sn,
        "n_lines_raw": 1,
    })

    out = (
        T.groupby(["bus0", "bus1"], as_index=False)
         .agg(
             b_raw=("b_raw", "sum"),
             s_nom_raw=("s_nom_raw", "sum"),
             n_lines_raw=("n_lines_raw", "sum"),
         )
    )
    return out


def build_reduced_corridors(red_lines: pd.DataFrame) -> pd.DataFrame:
    L = red_lines.copy()
    if L.empty:
        return pd.DataFrame(columns=["bus0", "bus1", "b_red", "s_nom_red", "n_lines_red"])

    L["bus0"] = L["bus0"].astype(str)
    L["bus1"] = L["bus1"].astype(str)
    L = L[L["bus0"] != L["bus1"]].copy()

    b = line_susceptance_from_reduced(L)
    sn = pd.to_numeric(L.get("s_nom", 0.0), errors="coerce").fillna(0.0).values.astype(float)

    u = np.minimum(L["bus0"].values, L["bus1"].values)
    v = np.maximum(L["bus0"].values, L["bus1"].values)

    T = pd.DataFrame({
        "bus0": u,
        "bus1": v,
        "b_red": b,
        "s_nom_red": sn,
        "n_lines_red": 1,
    })

    out = (
        T.groupby(["bus0", "bus1"], as_index=False)
         .agg(
             b_red=("b_red", "sum"),
             s_nom_red=("s_nom_red", "sum"),
             n_lines_red=("n_lines_red", "sum"),
         )
    )
    return out


# ============================================================
# Raw-network DC evaluation vs reduced-network DC evaluation
# ============================================================

def build_membership_weights(
    raw_buses_eval: pd.DataFrame,
    plants_eval: pd.DataFrame,
    bus_map_eval: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """
    For each cluster:
      - source weights: plant-capacity weighted if possible, otherwise uniform over member buses
      - sink weights: uniform over member buses (load data is not available here)
    """
    bm = bus_map_eval.copy()
    bm["bus_id"] = bm["bus_id"].astype(str)
    bm["bus_id_red"] = bm["bus_id_red"].astype(str)

    buses = raw_buses_eval.copy()
    buses["bus_id"] = buses["bus_id"].astype(str)
    buses = buses.merge(bm, on="bus_id", how="inner")

    bus_ids = buses["bus_id"].astype(str).tolist()
    bus_pos = {b: i for i, b in enumerate(bus_ids)}

    members = buses.groupby("bus_id_red")["bus_id"].apply(list).to_dict()

    cap_by_bus = None
    if (not plants_eval.empty) and ("assigned_bus" in plants_eval.columns):
        P = plants_eval.copy()
        P["assigned_bus"] = P["assigned_bus"].astype(str)
        P["Capacity"] = pd.to_numeric(P.get("Capacity", 0.0), errors="coerce").fillna(0.0)
        cap_by_bus = P.groupby("assigned_bus")["Capacity"].sum()

    source_w: Dict[str, np.ndarray] = {}
    sink_w: Dict[str, np.ndarray] = {}

    for cl, bus_list in members.items():
        w_pos = np.zeros(len(bus_ids), dtype=float)
        w_neg = np.zeros(len(bus_ids), dtype=float)

        if len(bus_list) == 0:
            continue

        if cap_by_bus is not None:
            caps = np.array([float(cap_by_bus.get(b, 0.0)) for b in bus_list], dtype=float)
        else:
            caps = np.zeros(len(bus_list), dtype=float)

        if np.sum(caps) > 1e-9:
            caps = caps / np.sum(caps)
        else:
            caps = np.full(len(bus_list), 1.0 / len(bus_list))

        unif = np.full(len(bus_list), 1.0 / len(bus_list))

        for bi, b in enumerate(bus_list):
            w_pos[bus_pos[b]] = caps[bi]
            w_neg[bus_pos[b]] = unif[bi]

        source_w[str(cl)] = w_pos
        sink_w[str(cl)] = w_neg

    return source_w, sink_w, bus_ids


def build_raw_solver_objects(
    raw_buses_eval: pd.DataFrame,
    raw_lines_eval: pd.DataFrame,
    bus_map_eval: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Build:
      - bus index
      - Bbus factorisations per connected component
      - line arrays for fast flow evaluation
      - inter-cluster aggregation map for corridor flows
      - cluster -> component mapping
    """
    buses = raw_buses_eval.copy()
    buses["bus_id"] = buses["bus_id"].astype(str)
    bus_ids = buses["bus_id"].tolist()
    bus_idx = {b: i for i, b in enumerate(bus_ids)}

    bm = bus_map_eval.copy()
    bm["bus_id"] = bm["bus_id"].astype(str)
    bm["bus_id_red"] = bm["bus_id_red"].astype(str)
    cluster_of_bus = bm.set_index("bus_id")["bus_id_red"].to_dict()

    L = raw_lines_eval.copy()
    L["bus0"] = L["bus0"].astype(str)
    L["bus1"] = L["bus1"].astype(str)
    L = L[L["bus0"].isin(bus_idx) & L["bus1"].isin(bus_idx)].copy()

    bvals = line_susceptance_from_raw(L)
    L["b_line"] = bvals
    L = L[np.isfinite(L["b_line"]) & (L["b_line"] > 0)].copy()

    nodes = bus_ids
    B, idx = build_sparse_bbus_from_edges(nodes, L, b_col="b_line")
    n_comp, labels = connected_components(B, directed=False, return_labels=True)

    # component data
    component_solvers: Dict[int, Any] = {}
    component_nodes: Dict[int, np.ndarray] = {}
    global_to_local: Dict[int, Dict[int, int]] = {}

    for c in range(n_comp):
        gl = np.where(labels == c)[0]
        component_nodes[c] = gl
        if len(gl) >= 2:
            Bc = B[gl][:, gl].tocsr()
            solve, slack = factorize_component_bbus(Bc)
            component_solvers[c] = (solve, slack)

            m = {int(g): li for li, g in enumerate(gl)}
            global_to_local[c] = m

    # map cluster -> component
    bus_comp_df = pd.DataFrame({
        "bus_id": nodes,
        "comp": labels,
    })
    bus_comp_df["cluster"] = bus_comp_df["bus_id"].map(cluster_of_bus)
    tmp = bus_comp_df.groupby("cluster")["comp"].agg(lambda s: s.value_counts().idxmax())
    cl_comp = {str(k): int(v) for k, v in tmp.to_dict().items()}

    # line arrays for fast flow computation
    gi = np.array([idx[b] for b in L["bus0"].astype(str).values], dtype=int)
    gj = np.array([idx[b] for b in L["bus1"].astype(str).values], dtype=int)
    gb = L["b_line"].values.astype(float)

    cl0 = L["bus0"].map(cluster_of_bus).astype(str).values
    cl1 = L["bus1"].map(cluster_of_bus).astype(str).values
    inter = cl0 != cl1

    # oriented corridor mapping
    c_low = np.minimum(cl0, cl1)
    c_high = np.maximum(cl0, cl1)
    keys = np.array([f"{a}||{b}" for a, b in zip(c_low, c_high)], dtype=object)
    uniq_keys = pd.Index(pd.unique(keys[inter]))
    corridor_index = {k: i for i, k in enumerate(uniq_keys.tolist())}
    corr_idx = np.full(len(L), -1, dtype=int)
    corr_idx[inter] = np.array([corridor_index[k] for k in keys[inter]], dtype=int)
    corr_sign = np.zeros(len(L), dtype=float)
    corr_sign[inter] = np.where(cl0[inter] == c_low[inter], 1.0, -1.0)

    return {
        "bus_ids": nodes,
        "bus_idx": idx,
        "B": B,
        "component_labels": labels,
        "component_solvers": component_solvers,
        "component_nodes": component_nodes,
        "global_to_local": global_to_local,
        "cluster_to_component": cl_comp,
        "line_bus0_idx": gi,
        "line_bus1_idx": gj,
        "line_b": gb,
        "line_corr_idx": corr_idx,
        "line_corr_sign": corr_sign,
        "corridor_keys": uniq_keys.tolist(),
    }


def build_reduced_solver_objects(red_buses_eval: pd.DataFrame, red_lines_eval: pd.DataFrame) -> Dict[str, Any]:
    nodes = red_buses_eval["bus_id"].astype(str).tolist()
    edges = build_reduced_corridors(red_lines_eval)
    if edges.empty:
        B, idx = sparse.csr_matrix((len(nodes), len(nodes))), {n: i for i, n in enumerate(nodes)}
    else:
        B, idx = build_sparse_bbus_from_edges(nodes, edges.rename(columns={"b_red": "b"}), b_col="b")

    n_comp, labels = connected_components(B, directed=False, return_labels=True)

    component_solvers: Dict[int, Any] = {}
    component_nodes: Dict[int, np.ndarray] = {}
    for c in range(n_comp):
        gl = np.where(labels == c)[0]
        component_nodes[c] = gl
        if len(gl) >= 2:
            Bc = B[gl][:, gl].tocsr()
            solve, slack = factorize_component_bbus(Bc)
            component_solvers[c] = (solve, slack)

    E = edges.copy()
    if E.empty:
        return {
            "nodes": nodes,
            "B": B,
            "idx": idx,
            "component_labels": labels,
            "component_nodes": component_nodes,
            "component_solvers": component_solvers,
            "edges": E,
            "edge_bus0_idx": np.array([], dtype=int),
            "edge_bus1_idx": np.array([], dtype=int),
            "edge_b": np.array([], dtype=float),
            "corridor_keys": [],
        }

    u = E["bus0"].astype(str).values
    v = E["bus1"].astype(str).values
    keys = [f"{a}||{b}" for a, b in zip(np.minimum(u, v), np.maximum(u, v))]

    return {
        "nodes": nodes,
        "B": B,
        "idx": idx,
        "component_labels": labels,
        "component_nodes": component_nodes,
        "component_solvers": component_solvers,
        "edges": E,
        "edge_bus0_idx": np.array([idx[x] for x in u], dtype=int),
        "edge_bus1_idx": np.array([idx[x] for x in v], dtype=int),
        "edge_b": E["b_red"].values.astype(float),
        "corridor_keys": keys,
    }


def solve_raw_transfer_corridor_flows(
    src_cluster: str,
    sink_cluster: str,
    inj_mw: float,
    source_weights: Dict[str, np.ndarray],
    sink_weights: Dict[str, np.ndarray],
    raw_solver: Dict[str, Any],
) -> np.ndarray | None:
    if src_cluster not in source_weights or sink_cluster not in sink_weights:
        return None

    csrc = raw_solver["cluster_to_component"].get(src_cluster)
    csnk = raw_solver["cluster_to_component"].get(sink_cluster)
    if csrc is None or csnk is None or csrc != csnk:
        return None
    if csrc not in raw_solver["component_solvers"]:
        return None

    p_global = inj_mw * source_weights[src_cluster] - inj_mw * sink_weights[sink_cluster]
    gl = raw_solver["component_nodes"][csrc]
    p = p_global[gl]

    solve, slack = raw_solver["component_solvers"][csrc]
    theta_loc = solve_dc_angles_factorized(solve, slack=slack, p=p)

    theta_global = np.zeros(len(raw_solver["bus_ids"]), dtype=float)
    theta_global[gl] = theta_loc

    f_line = raw_solver["line_b"] * (theta_global[raw_solver["line_bus0_idx"]] - theta_global[raw_solver["line_bus1_idx"]])

    corr_idx = raw_solver["line_corr_idx"]
    inter = corr_idx >= 0
    if not np.any(inter):
        return np.zeros(len(raw_solver["corridor_keys"]), dtype=float)

    agg = np.bincount(
        corr_idx[inter],
        weights=raw_solver["line_corr_sign"][inter] * f_line[inter],
        minlength=len(raw_solver["corridor_keys"]),
    )
    return agg.astype(float)


def solve_reduced_transfer_corridor_flows(
    src_cluster: str,
    sink_cluster: str,
    inj_mw: float,
    red_solver: Dict[str, Any],
) -> np.ndarray | None:
    if src_cluster not in red_solver["idx"] or sink_cluster not in red_solver["idx"]:
        return None

    isrc = red_solver["idx"][src_cluster]
    isnk = red_solver["idx"][sink_cluster]
    csrc = int(red_solver["component_labels"][isrc])
    csnk = int(red_solver["component_labels"][isnk])
    if csrc != csnk or csrc not in red_solver["component_solvers"]:
        return None

    gl = red_solver["component_nodes"][csrc]
    p = np.zeros(len(gl), dtype=float)

    li_src = int(np.where(gl == isrc)[0][0])
    li_snk = int(np.where(gl == isnk)[0][0])
    p[li_src] = inj_mw
    p[li_snk] = -inj_mw

    solve, slack = red_solver["component_solvers"][csrc]
    theta_loc = solve_dc_angles_factorized(solve, slack=slack, p=p)

    theta = np.zeros(len(red_solver["nodes"]), dtype=float)
    theta[gl] = theta_loc

    f_edge = red_solver["edge_b"] * (theta[red_solver["edge_bus0_idx"]] - theta[red_solver["edge_bus1_idx"]])
    return f_edge.astype(float)


# ============================================================
# Metrics
# ============================================================

def evaluate_corridor_preservation(corr_raw: pd.DataFrame, corr_red: pd.DataFrame) -> Dict[str, float]:
    M = corr_raw.merge(corr_red, on=["bus0", "bus1"], how="outer").fillna(0.0)

    edge_set_raw = set(zip(corr_raw["bus0"], corr_raw["bus1"]))
    edge_set_red = set(zip(corr_red["bus0"], corr_red["bus1"]))
    overlap = edge_set_raw & edge_set_red

    precision = safe_div(len(overlap), len(edge_set_red), np.nan)
    recall = safe_div(len(overlap), len(edge_set_raw), np.nan)
    jaccard = safe_div(len(overlap), len(edge_set_raw | edge_set_red), np.nan)

    b_wape = weighted_abs_pct_error(
        M["b_raw"].values,
        M["b_red"].values,
        weights=np.maximum(M["b_raw"].values, 1e-9),
    )
    snom_wape = weighted_abs_pct_error(
        M["s_nom_raw"].values,
        M["s_nom_red"].values,
        weights=np.maximum(M["s_nom_raw"].values, 1e-9),
    )

    return {
        "corridor_edges_raw": float(len(edge_set_raw)),
        "corridor_edges_red": float(len(edge_set_red)),
        "corridor_edge_precision": precision,
        "corridor_edge_recall": recall,
        "corridor_edge_jaccard": jaccard,
        "corridor_b_wape": b_wape,
        "corridor_snom_wape": snom_wape,
    }


def evaluate_transfer_fidelity(
    cluster_ids: List[str],
    raw_solver: Dict[str, Any],
    red_solver: Dict[str, Any],
    source_weights: Dict[str, np.ndarray],
    sink_weights: Dict[str, np.ndarray],
    *,
    n_samples: int,
    inj_mw: float,
    random_seed: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    rng = np.random.default_rng(random_seed)

    common_clusters = [c for c in cluster_ids if c in red_solver["idx"] and c in raw_solver["cluster_to_component"]]
    if len(common_clusters) < 2:
        return {
            "transfer_samples_used": 0.0,
            "transfer_nrmse_mean": np.nan,
            "transfer_nrmse_p95": np.nan,
            "transfer_cosine_mean": np.nan,
            "transfer_corr_mean": np.nan,
        }, pd.DataFrame()

    cluster_comp = {c: raw_solver["cluster_to_component"].get(c) for c in common_clusters}
    valid_pairs = [(a, b) for i, a in enumerate(common_clusters) for b in common_clusters[i+1:]
                   if cluster_comp.get(a) is not None and cluster_comp.get(a) == cluster_comp.get(b)]

    if not valid_pairs:
        return {
            "transfer_samples_used": 0.0,
            "transfer_nrmse_mean": np.nan,
            "transfer_nrmse_p95": np.nan,
            "transfer_cosine_mean": np.nan,
            "transfer_corr_mean": np.nan,
        }, pd.DataFrame()

    n_take = min(n_samples, len(valid_pairs))
    chosen_idx = rng.choice(len(valid_pairs), size=n_take, replace=False)
    chosen_pairs = [valid_pairs[int(i)] for i in chosen_idx]

    raw_corr_keys = raw_solver["corridor_keys"]
    red_corr_keys = red_solver["corridor_keys"]
    union_keys = pd.Index(pd.unique(raw_corr_keys + red_corr_keys))
    union_pos = {k: i for i, k in enumerate(union_keys.tolist())}
    raw_to_union = np.array([union_pos[k] for k in raw_corr_keys], dtype=int) if raw_corr_keys else np.array([], dtype=int)
    red_to_union = np.array([union_pos[k] for k in red_corr_keys], dtype=int) if red_corr_keys else np.array([], dtype=int)

    rows = []

    for src, sink in chosen_pairs:
        f_raw = solve_raw_transfer_corridor_flows(
            src, sink, inj_mw,
            source_weights=source_weights,
            sink_weights=sink_weights,
            raw_solver=raw_solver,
        )
        f_red = solve_reduced_transfer_corridor_flows(
            src, sink, inj_mw,
            red_solver=red_solver,
        )

        if f_raw is None or f_red is None:
            continue

        v_raw = np.zeros(len(union_keys), dtype=float)
        v_red = np.zeros(len(union_keys), dtype=float)
        if len(raw_to_union):
            v_raw[raw_to_union] = f_raw
        if len(red_to_union):
            v_red[red_to_union] = f_red

        rows.append({
            "src_cluster": src,
            "sink_cluster": sink,
            "flow_nrmse": nrmse(v_raw, v_red),
            "flow_cosine": cosine_similarity(v_raw, v_red),
            "flow_corr": corrcoef_safe(v_raw, v_red),
        })

    samples = pd.DataFrame(rows)

    if samples.empty:
        return {
            "transfer_samples_used": 0.0,
            "transfer_nrmse_mean": np.nan,
            "transfer_nrmse_p95": np.nan,
            "transfer_cosine_mean": np.nan,
            "transfer_corr_mean": np.nan,
        }, samples

    metrics = {
        "transfer_samples_used": float(len(samples)),
        "transfer_nrmse_mean": float(samples["flow_nrmse"].mean()),
        "transfer_nrmse_p95": float(samples["flow_nrmse"].quantile(0.95)),
        "transfer_cosine_mean": float(samples["flow_cosine"].mean()),
        "transfer_corr_mean": float(samples["flow_corr"].mean()),
    }
    return metrics, samples


def evaluate_inertia_distance_fidelity(
    cluster_ids: List[str],
    corr_raw: pd.DataFrame,
    corr_red: pd.DataFrame,
    sync_cap_by_cluster: pd.Series,
    *,
    kernel: str,
    tau_scale: float,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    nodes = list(map(str, cluster_ids))

    corr_raw_use = corr_raw[corr_raw["bus0"].isin(nodes) & corr_raw["bus1"].isin(nodes)].copy()
    corr_red_use = corr_red[corr_red["bus0"].isin(nodes) & corr_red["bus1"].isin(nodes)].copy()

    D_raw = dense_effective_reactance_distance(nodes, corr_raw_use.rename(columns={"b_raw": "b"}), b_col="b")
    D_red = dense_effective_reactance_distance(nodes, corr_red_use.rename(columns={"b_red": "b"}), b_col="b")

    d_raw_vec = upper_tri_values(D_raw)
    d_red_vec = upper_tri_values(D_red)

    K_raw = proximity_matrix_from_distance(D_raw, kernel=kernel, tau_scale=tau_scale)
    K_red = proximity_matrix_from_distance(D_red, kernel=kernel, tau_scale=tau_scale)

    sync_cap = sync_cap_by_cluster.reindex(nodes).fillna(0.0).values.astype(float)
    inertia_raw = K_raw @ sync_cap
    inertia_red = K_red @ sync_cap

    dist_metrics = {
        "effrx_distance_nrmse": nrmse(d_raw_vec, d_red_vec),
        "effrx_distance_corr": corrcoef_safe(d_raw_vec, d_red_vec),
        "inertia_proxy_nrmse": nrmse(inertia_raw, inertia_red),
        "inertia_proxy_corr": corrcoef_safe(inertia_raw, inertia_red),
    }

    df_dist = pd.DataFrame({
        "bus_id": nodes,
        "sync_capacity_mw": sync_cap,
        "inertia_proxy_raw": inertia_raw,
        "inertia_proxy_red": inertia_red,
    })

    df_pair = pd.DataFrame({
        "distance_raw": d_raw_vec,
        "distance_red": d_red_vec,
    })

    return dist_metrics, df_dist, df_pair


def evaluate_cluster_compactness(raw_buses_eval: pd.DataFrame, bus_map_eval: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    b = raw_buses_eval.copy()
    b["bus_id"] = b["bus_id"].astype(str)
    b["lat"] = pd.to_numeric(b["lat"], errors="coerce")
    b["lon"] = pd.to_numeric(b["lon"], errors="coerce")

    bm = bus_map_eval.copy()
    bm["bus_id"] = bm["bus_id"].astype(str)
    bm["bus_id_red"] = bm["bus_id_red"].astype(str)

    x = b.merge(bm, on="bus_id", how="inner").dropna(subset=["lat", "lon"]).copy()
    if x.empty:
        return {
            "geo_radius95_km_mean": np.nan,
            "geo_radius95_km_p95": np.nan,
            "geo_diameter_km_mean": np.nan,
            "geo_diameter_km_p95": np.nan,
        }, pd.DataFrame()

    rows = []
    for cl, g in x.groupby("bus_id_red"):
        lat0 = g["lat"].mean()
        lon0 = g["lon"].mean()
        rr = haversine_km(g["lat"].values, g["lon"].values, lat0, lon0)

        if len(g) >= 2:
            i = int(np.argmax(rr))
            d2 = haversine_km(g["lat"].values, g["lon"].values, g["lat"].iloc[i], g["lon"].iloc[i])
            j = int(np.argmax(d2))
            d3 = haversine_km(
                np.array([g["lat"].iloc[i]]),
                np.array([g["lon"].iloc[i]]),
                np.array([g["lat"].iloc[j]]),
                np.array([g["lon"].iloc[j]]),
            )[0]
            diam = float(d3)
        else:
            diam = 0.0

        rows.append({
            "bus_id_red": cl,
            "n_buses_raw": int(len(g)),
            "radius95_km": float(np.quantile(rr, 0.95)) if len(rr) else 0.0,
            "radius_mean_km": float(np.mean(rr)) if len(rr) else 0.0,
            "diameter_km": diam,
        })

    df = pd.DataFrame(rows)
    metrics = {
        "geo_radius95_km_mean": float(df["radius95_km"].mean()),
        "geo_radius95_km_p95": float(df["radius95_km"].quantile(0.95)),
        "geo_diameter_km_mean": float(df["diameter_km"].mean()),
        "geo_diameter_km_p95": float(df["diameter_km"].quantile(0.95)),
    }
    return metrics, df


# ============================================================
# Additional graph / mesh / PTDF-like benchmarking helpers
# ============================================================

def build_sparse_adjacency_from_edges(
    nodes: List[str],
    edges: pd.DataFrame,
    w_col: str,
) -> Tuple[sparse.csr_matrix, Dict[str, int]]:
    idx = {n: i for i, n in enumerate(nodes)}
    rows = []
    cols = []
    data = []

    bus0 = edges["bus0"].astype(str).values
    bus1 = edges["bus1"].astype(str).values
    wvals = pd.to_numeric(edges[w_col], errors="coerce").values.astype(float)

    for u, v, w in zip(bus0, bus1, wvals):
        if u == v:
            continue
        if (u not in idx) or (v not in idx):
            continue
        if not np.isfinite(w) or w <= 0:
            continue
        i = idx[u]
        j = idx[v]
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([float(w), float(w)])

    n = len(nodes)
    if not rows:
        return sparse.csr_matrix((n, n)), idx

    W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    W.sum_duplicates()
    return W.tocsr(), idx


def algebraic_connectivity_from_adjacency(W: sparse.csr_matrix) -> float:
    n = W.shape[0]
    if n <= 1:
        return 0.0
    if W.nnz == 0:
        return 0.0

    d = np.asarray(W.sum(axis=1)).ravel()
    L = sparse.diags(d) - W

    # small clustered graphs -> dense eigenvalues are fine and robust
    vals = np.linalg.eigvalsh(L.toarray().astype(float))
    vals = np.sort(np.real(vals))
    if len(vals) < 2:
        return 0.0
    return float(max(vals[1], 0.0))


def kirchhoff_index_from_adjacency(W: sparse.csr_matrix) -> float:
    """
    Weighted Kirchhoff index = sum over connected components of n_c * trace(L_c^+)
    Lower means electrically tighter / better meshed.
    """
    n = W.shape[0]
    if n <= 1 or W.nnz == 0:
        return 0.0

    n_comp, labels = connected_components(W, directed=False, return_labels=True)
    total = 0.0

    for c in range(n_comp):
        gl = np.where(labels == c)[0]
        nc = len(gl)
        if nc <= 1:
            continue
        Wc = W[gl][:, gl].tocsr()
        d = np.asarray(Wc.sum(axis=1)).ravel()
        Lc = (sparse.diags(d) - Wc).toarray().astype(float)
        Ldag = np.linalg.pinv(Lc, hermitian=True)
        total += float(nc * np.trace(Ldag))

    return float(total)


def corridor_graph_stats(
    nodes: List[str],
    edges: pd.DataFrame,
    *,
    b_col: str,
) -> Dict[str, float]:
    """
    Structural stats of a cluster graph.
    """
    n = len(nodes)
    if n == 0 or edges.empty:
        return {
            "n_nodes": float(n),
            "n_edges": 0.0,
            "n_components": float(n),
            "cycle_rank": 0.0,
            "mesh_ratio": 0.0,
            "avg_degree": 0.0,
            "lambda2_weighted": 0.0,
            "lambda2_unweighted": 0.0,
            "kirchhoff_weighted": 0.0,
        }

    E = edges.copy()
    E["one"] = 1.0

    Ww, _ = build_sparse_adjacency_from_edges(nodes, E.rename(columns={b_col: "w"}), w_col="w")
    Wu, _ = build_sparse_adjacency_from_edges(nodes, E.rename(columns={"one": "w"}), w_col="w")

    n_comp, _ = connected_components(Wu, directed=False, return_labels=True)
    m = int(len(E))
    cycle_rank = max(m - n + n_comp, 0)
    mesh_ratio = safe_div(cycle_rank, m, 0.0)
    avg_degree = safe_div(2.0 * m, n, 0.0)

    return {
        "n_nodes": float(n),
        "n_edges": float(m),
        "n_components": float(n_comp),
        "cycle_rank": float(cycle_rank),
        "mesh_ratio": float(mesh_ratio),
        "avg_degree": float(avg_degree),
        "lambda2_weighted": algebraic_connectivity_from_adjacency(Ww),
        "lambda2_unweighted": algebraic_connectivity_from_adjacency(Wu),
        "kirchhoff_weighted": kirchhoff_index_from_adjacency(Ww),
    }


def evaluate_mesh_loop_characteristics(
    cluster_ids: List[str],
    corr_raw: pd.DataFrame,
    corr_red: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    nodes = list(map(str, cluster_ids))

    corr_raw_use = corr_raw[corr_raw["bus0"].isin(nodes) & corr_raw["bus1"].isin(nodes)].copy()
    corr_red_use = corr_red[corr_red["bus0"].isin(nodes) & corr_red["bus1"].isin(nodes)].copy()

    stats_raw = corridor_graph_stats(nodes, corr_raw_use.rename(columns={"b_raw": "b"}), b_col="b")
    stats_red = corridor_graph_stats(nodes, corr_red_use.rename(columns={"b_red": "b"}), b_col="b")

    metrics = {
        # absolute retained structure after clustering (important across methods)
        "cluster_cycle_rank_raw": stats_raw["cycle_rank"],
        "cluster_mesh_ratio_raw": stats_raw["mesh_ratio"],
        "cluster_lambda2_weighted_raw": stats_raw["lambda2_weighted"],
        "cluster_lambda2_unweighted_raw": stats_raw["lambda2_unweighted"],
        "cluster_kirchhoff_weighted_raw": stats_raw["kirchhoff_weighted"],

        # preservation of clustered-raw graph by reduced electrical model
        "cluster_cycle_rank_abs_pct_error": safe_div(
            abs(stats_red["cycle_rank"] - stats_raw["cycle_rank"]),
            max(stats_raw["cycle_rank"], 1.0),
            np.nan,
        ),
        "cluster_mesh_ratio_abs_pct_error": safe_div(
            abs(stats_red["mesh_ratio"] - stats_raw["mesh_ratio"]),
            max(abs(stats_raw["mesh_ratio"]), 1e-12),
            np.nan,
        ),
        "cluster_lambda2_weighted_abs_pct_error": safe_div(
            abs(stats_red["lambda2_weighted"] - stats_raw["lambda2_weighted"]),
            max(abs(stats_raw["lambda2_weighted"]), 1e-12),
            np.nan,
        ),
        "cluster_kirchhoff_weighted_abs_pct_error": safe_div(
            abs(stats_red["kirchhoff_weighted"] - stats_raw["kirchhoff_weighted"]),
            max(abs(stats_raw["kirchhoff_weighted"]), 1e-12),
            np.nan,
        ),
    }

    detail = pd.DataFrame([
        {"graph": "clustered_raw", **stats_raw},
        {"graph": "reduced", **stats_red},
    ])
    return metrics, detail


def build_cluster_graph_solver(
    nodes: List[str],
    edges: pd.DataFrame,
    *,
    b_col: str,
) -> Dict[str, Any]:
    E = edges.copy()
    if E.empty:
        idx = {n: i for i, n in enumerate(nodes)}
        return {
            "nodes": nodes,
            "idx": idx,
            "component_labels": np.arange(len(nodes)),
            "component_nodes": {i: np.array([i], dtype=int) for i in range(len(nodes))},
            "component_solvers": {},
            "edge_bus0_idx": np.array([], dtype=int),
            "edge_bus1_idx": np.array([], dtype=int),
            "edge_b": np.array([], dtype=float),
            "corridor_keys": [],
        }

    B, idx = build_sparse_bbus_from_edges(
        nodes,
        E.rename(columns={b_col: "b"}),
        b_col="b",
    )
    W, _ = build_sparse_adjacency_from_edges(
        nodes,
        E.rename(columns={b_col: "w"}),
        w_col="w",
    )

    n_comp, labels = connected_components(W, directed=False, return_labels=True)

    component_solvers: Dict[int, Any] = {}
    component_nodes: Dict[int, np.ndarray] = {}

    for c in range(n_comp):
        gl = np.where(labels == c)[0]
        component_nodes[c] = gl
        if len(gl) >= 2:
            Bc = B[gl][:, gl].tocsr()
            solve, slack = factorize_component_bbus(Bc)
            component_solvers[c] = (solve, slack)

    E["bus0"] = E["bus0"].astype(str)
    E["bus1"] = E["bus1"].astype(str)
    keys = [
        f"{min(u, v)}||{max(u, v)}"
        for u, v in zip(E["bus0"].values, E["bus1"].values)
    ]

    return {
        "nodes": nodes,
        "idx": idx,
        "component_labels": labels,
        "component_nodes": component_nodes,
        "component_solvers": component_solvers,
        "edge_bus0_idx": np.array([idx[x] for x in E["bus0"].values], dtype=int),
        "edge_bus1_idx": np.array([idx[x] for x in E["bus1"].values], dtype=int),
        "edge_b": pd.to_numeric(E[b_col], errors="coerce").fillna(0.0).values.astype(float),
        "corridor_keys": keys,
    }


def solve_cluster_graph_flows(
    solver: Dict[str, Any],
    p_global: np.ndarray,
) -> np.ndarray | None:
    p_global = np.asarray(p_global, dtype=float)
    if len(p_global) != len(solver["nodes"]):
        raise ValueError("Injection vector length does not match solver nodes.")

    theta = np.zeros(len(solver["nodes"]), dtype=float)

    for c, gl in solver["component_nodes"].items():
        pc = p_global[gl]
        if abs(np.sum(pc)) > 1e-8:
            return None

        if len(gl) == 1:
            if abs(pc[0]) > 1e-8:
                return None
            theta[gl[0]] = 0.0
            continue

        if c not in solver["component_solvers"]:
            return None

        solve, slack = solver["component_solvers"][c]
        theta_loc = solve_dc_angles_factorized(solve, slack=slack, p=pc)
        theta[gl] = theta_loc

    flows = solver["edge_b"] * (
        theta[solver["edge_bus0_idx"]] - theta[solver["edge_bus1_idx"]]
    )
    return flows.astype(float)


def draw_random_balanced_cluster_injection(
    member_nodes: List[str],
    node_pos: Dict[str, int],
    *,
    n_total: int,
    inj_mw: float,
    rng: np.random.Generator,
    k_active_min: int,
    k_active_max: int,
) -> np.ndarray:
    n_group = len(member_nodes)
    if n_group < 2:
        raise ValueError("Need at least two nodes for balanced injection.")

    k_hi = min(k_active_max, n_group)
    k_lo = min(k_active_min, k_hi)
    if k_lo < 2:
        k_lo = 2
    k = int(rng.integers(k_lo, k_hi + 1))

    chosen = list(rng.choice(member_nodes, size=k, replace=False))

    # random signed pattern with zero sum
    z = rng.normal(size=k)
    z = z - np.mean(z)

    if np.all(np.abs(z) < 1e-12):
        z[0] = 1.0
        z[-1] = -1.0

    pos = np.clip(z, 0.0, None)
    neg = np.clip(-z, 0.0, None)

    if pos.sum() <= 1e-12 or neg.sum() <= 1e-12:
        pos = np.zeros(k)
        neg = np.zeros(k)
        pos[0] = 1.0
        neg[-1] = 1.0

    p = np.zeros(n_total, dtype=float)
    for i, node in enumerate(chosen):
        val = 0.0
        if pos[i] > 0:
            val += inj_mw * pos[i] / pos.sum()
        if neg[i] > 0:
            val -= inj_mw * neg[i] / neg.sum()
        p[node_pos[node]] = val

    # exact rebalance for numerical safety
    p = p - np.mean(p[np.abs(p) > 0]) * (np.abs(p) > 0)
    # final correction on the last active node
    active_idx = np.where(np.abs(p) > 0)[0]
    if len(active_idx):
        p[active_idx[-1]] -= np.sum(p)

    return p


def evaluate_multi_injection_fidelity(
    cluster_ids: List[str],
    corr_raw: pd.DataFrame,
    corr_red: pd.DataFrame,
    *,
    n_samples: int,
    inj_mw: float,
    random_seed: int,
    k_active_min: int,
    k_active_max: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    nodes = list(map(str, cluster_ids))

    corr_raw_use = corr_raw[corr_raw["bus0"].isin(nodes) & corr_raw["bus1"].isin(nodes)].copy()
    corr_red_use = corr_red[corr_red["bus0"].isin(nodes) & corr_red["bus1"].isin(nodes)].copy()

    raw_solver = build_cluster_graph_solver(nodes, corr_raw_use, b_col="b_raw")
    red_solver = build_cluster_graph_solver(nodes, corr_red_use, b_col="b_red")

    raw_labels = raw_solver["component_labels"]
    red_labels = red_solver["component_labels"]

    eligible_groups: List[List[str]] = []
    for rc in np.unique(raw_labels):
        members = [nodes[i] for i in np.where(raw_labels == rc)[0]]
        if len(members) < 3:
            continue

        red_comps = {int(red_labels[raw_solver["idx"][m]]) for m in members}
        if len(red_comps) != 1:
            continue

        eligible_groups.append(members)

    if not eligible_groups:
        return {
            "multiinj_samples_used": 0.0,
            "multiinj_nrmse_mean": np.nan,
            "multiinj_nrmse_p95": np.nan,
            "multiinj_cosine_mean": np.nan,
            "multiinj_corr_mean": np.nan,
        }, pd.DataFrame()

    rng = np.random.default_rng(random_seed)

    union_keys = pd.Index(pd.unique(raw_solver["corridor_keys"] + red_solver["corridor_keys"]))
    union_pos = {k: i for i, k in enumerate(union_keys.tolist())}
    raw_to_union = np.array([union_pos[k] for k in raw_solver["corridor_keys"]], dtype=int)
    red_to_union = np.array([union_pos[k] for k in red_solver["corridor_keys"]], dtype=int)

    node_pos = {n: i for i, n in enumerate(nodes)}
    rows = []

    for _ in range(int(n_samples)):
        members = eligible_groups[int(rng.integers(0, len(eligible_groups)))]

        p = draw_random_balanced_cluster_injection(
            members,
            node_pos,
            n_total=len(nodes),
            inj_mw=inj_mw,
            rng=rng,
            k_active_min=k_active_min,
            k_active_max=k_active_max,
        )

        f_raw = solve_cluster_graph_flows(raw_solver, p)
        f_red = solve_cluster_graph_flows(red_solver, p)

        if f_raw is None or f_red is None:
            continue

        v_raw = np.zeros(len(union_keys), dtype=float)
        v_red = np.zeros(len(union_keys), dtype=float)
        if len(raw_to_union):
            v_raw[raw_to_union] = f_raw
        if len(red_to_union):
            v_red[red_to_union] = f_red

        rows.append({
            "n_active_clusters": int(np.sum(np.abs(p) > 1e-9)),
            "flow_nrmse": nrmse(v_raw, v_red),
            "flow_cosine": cosine_similarity(v_raw, v_red),
            "flow_corr": corrcoef_safe(v_raw, v_red),
        })

    samples = pd.DataFrame(rows)
    if samples.empty:
        return {
            "multiinj_samples_used": 0.0,
            "multiinj_nrmse_mean": np.nan,
            "multiinj_nrmse_p95": np.nan,
            "multiinj_cosine_mean": np.nan,
            "multiinj_corr_mean": np.nan,
        }, samples

    metrics = {
        "multiinj_samples_used": float(len(samples)),
        "multiinj_nrmse_mean": float(samples["flow_nrmse"].mean()),
        "multiinj_nrmse_p95": float(samples["flow_nrmse"].quantile(0.95)),
        "multiinj_cosine_mean": float(samples["flow_cosine"].mean()),
        "multiinj_corr_mean": float(samples["flow_corr"].mean()),
    }
    return metrics, samples


# ============================================================
# Composite ranking
# ============================================================

def build_ranking(df: pd.DataFrame, score_weights: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()

    higher_better = {
        "transfer_cosine_mean",
        "multiinj_cosine_mean",
        "bus_reduction_ratio",
        "cluster_cycle_rank_raw",
        "cluster_lambda2_weighted_raw",
    }
    #lower_better = set(score_weights) - higher_better

    score_parts = {}
    for m, w in score_weights.items():
        if m not in out.columns:
            continue
        col = pd.to_numeric(out[m], errors="coerce").astype(float)

        valid = np.isfinite(col.values)
        if not np.any(valid):
            score_parts[m] = np.full(len(out), np.nan)
            continue

        lo = np.nanmin(col.values)
        hi = np.nanmax(col.values)

        if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) <= 1e-12:
            norm = np.full(len(out), 1.0)
        else:
            if m in higher_better:
                norm = (col - lo) / (hi - lo)
            else:
                norm = (hi - col) / (hi - lo)

        score_parts[m] = norm * float(w)

    score_df = pd.DataFrame(score_parts)
    out["composite_score"] = score_df.sum(axis=1, min_count=1)
    out = out.sort_values("composite_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


# ============================================================
# Main per-run evaluation
# ============================================================

def evaluate_run(
    run_dir: Path,
    raw_buses: pd.DataFrame,
    raw_lines: pd.DataFrame,
    plants_with_bus: pd.DataFrame,
    cfg: EvalConfig,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    run_name = run_dir.name

    bus_map = read_run_csv(run_dir / "bus_map.csv", expected_cols={"bus_id", "bus_id_red"})
    red_buses = prep_run_buses(read_run_csv(run_dir / "buses.csv", expected_cols={"bus_id"}))
    red_lines = prep_run_lines(read_run_csv(run_dir / "lines.csv", expected_cols={"bus0", "bus1", "u", "v"}))
    #red_plants = prep_plants_with_bus(read_run_csv(run_dir / "plants.csv", expected_cols={"bus_id", "Fueltype", "Technology"}))

    bus_map["bus_id"] = bus_map["bus_id"].astype(str)
    bus_map["bus_id_red"] = bus_map["bus_id_red"].astype(str)

    # If available, use the run's original-bus classification file for a cleaner filter
    buses_with_clusters_path = run_dir / "buses_with_clusters.csv"
    if buses_with_clusters_path.exists():
        bwc = read_run_csv(buses_with_clusters_path)
        bwc["bus_id"] = bwc["bus_id"].astype(str)
        if "dc_bool" not in bwc.columns and "dc" in bwc.columns:
            bwc["dc_bool"] = bwc["dc"].astype(str).str.lower().isin(["t", "true", "1", "yes", "y"])
        raw_filter = bwc[["bus_id"]].copy()
        if cfg.evaluate_only_cesa_ac and {"sync_area", "dc_bool"}.issubset(bwc.columns):
            raw_filter = bwc[bwc["sync_area"].astype(str).eq("CESA") & (~bwc["dc_bool"].astype(bool))][["bus_id"]].copy()
        eval_raw_bus_ids = set(raw_filter["bus_id"].astype(str))
    else:
        eval_raw_bus_ids = set(bus_map["bus_id"].astype(str))

    raw_buses_eval = raw_buses[raw_buses["bus_id"].isin(eval_raw_bus_ids)].copy()
    raw_lines_eval = raw_lines[
        raw_lines["bus0"].isin(eval_raw_bus_ids) &
        raw_lines["bus1"].isin(eval_raw_bus_ids)
    ].copy()

    # AC-only electrical benchmark
    raw_lines_eval = raw_lines_eval.copy()
    btmp = line_susceptance_from_raw(raw_lines_eval)
    raw_lines_eval = raw_lines_eval[np.isfinite(btmp) & (btmp > 0)].copy()

    bus_map_eval = bus_map[bus_map["bus_id"].isin(eval_raw_bus_ids)].copy()

    if cfg.evaluate_only_cesa_ac and {"sync_area", "dc_bool"}.issubset(red_buses.columns):
        red_buses_eval = red_buses[red_buses["sync_area"].astype(str).eq("CESA") & (~red_buses["dc_bool"].astype(bool))].copy()
    else:
        red_buses_eval = red_buses.copy()

    red_cluster_ids = set(red_buses_eval["bus_id"].astype(str))
    bus_map_eval = bus_map_eval[bus_map_eval["bus_id_red"].isin(red_cluster_ids)].copy()

    red_lines_eval = red_lines[
        red_lines["bus0"].astype(str).isin(red_cluster_ids) &
        red_lines["bus1"].astype(str).isin(red_cluster_ids)
    ].copy()

    plants_eval = plants_with_bus.copy()
    if "assigned_bus" in plants_eval.columns:
        plants_eval = plants_eval[plants_eval["assigned_bus"].astype(str).isin(eval_raw_bus_ids)].copy()

    summary = {
        "run_name": run_name,
        "n_raw_buses_eval": float(raw_buses_eval["bus_id"].nunique()),
        "n_red_buses_eval": float(red_buses_eval["bus_id"].nunique()),
        "n_raw_lines_eval": float(len(raw_lines_eval)),
        "n_red_lines_eval": float(len(red_lines_eval)),
    }
    summary["bus_reduction_ratio"] = safe_div(summary["n_raw_buses_eval"], summary["n_red_buses_eval"], np.nan)
    summary["line_reduction_ratio"] = safe_div(summary["n_raw_lines_eval"], summary["n_red_lines_eval"], np.nan)
    
    # Corridors
    corr_raw = build_exact_cluster_corridors(raw_lines_eval, bus_map_eval, evaluated_bus_ids=eval_raw_bus_ids)
    corr_red = build_reduced_corridors(red_lines_eval)
    summary.update(evaluate_corridor_preservation(corr_raw, corr_red))
    
    # Mesh, Loop
    mesh_metrics, mesh_detail_df = evaluate_mesh_loop_characteristics(
        cluster_ids=sorted(set(bus_map_eval["bus_id_red"].astype(str))),
        corr_raw=corr_raw,
        corr_red=corr_red,
    )
    summary.update(mesh_metrics)
    
    # Compactnes
    comp_metrics, compactness_df = evaluate_cluster_compactness(raw_buses_eval, bus_map_eval)
    summary.update(comp_metrics)
    
    # Transfers
    source_w, sink_w, _ = build_membership_weights(raw_buses_eval, plants_eval, bus_map_eval)
    raw_solver = build_raw_solver_objects(raw_buses_eval, raw_lines_eval, bus_map_eval)
    red_solver = build_reduced_solver_objects(red_buses_eval, red_lines_eval)

    cluster_ids = sorted(set(bus_map_eval["bus_id_red"].astype(str)))
    transfer_metrics, transfer_samples = evaluate_transfer_fidelity(
        cluster_ids=cluster_ids,
        raw_solver=raw_solver,
        red_solver=red_solver,
        source_weights=source_w,
        sink_weights=sink_w,
        n_samples=cfg.n_transfer_samples,
        inj_mw=cfg.transfer_injection_mw,
        random_seed=cfg.random_seed,
    )
    summary.update(transfer_metrics)
    
    # Multi-injection
    multiinj_metrics, multiinj_samples = evaluate_multi_injection_fidelity(
        cluster_ids=cluster_ids,
        corr_raw=corr_raw,
        corr_red=corr_red,
        n_samples=cfg.n_transfer_samples,
        inj_mw=cfg.multiinj_injection_mw,
        random_seed=cfg.random_seed + 1000,
        k_active_min=cfg.multiinj_k_active_min,
        k_active_max=cfg.multiinj_k_active_max,
    )
    summary.update(multiinj_metrics)
    
    # Inertia
    sync_raw = plants_eval[plants_eval.apply(is_sync_plant, axis=1, cfg=cfg)].copy()
    if "assigned_bus" in sync_raw.columns:
        sync_raw = sync_raw.merge(bus_map_eval, left_on="assigned_bus", right_on="bus_id", how="inner")
        sync_cap_by_cluster = sync_raw.groupby("bus_id_red")["Capacity"].sum()
    else:
        sync_cap_by_cluster = pd.Series(dtype=float)

    inertia_metrics, inertia_by_bus_df, pairdist_df = evaluate_inertia_distance_fidelity(
        cluster_ids=cluster_ids,
        corr_raw=corr_raw,
        corr_red=corr_red,
        sync_cap_by_cluster=sync_cap_by_cluster,
        kernel=cfg.proximity_kernel,
        tau_scale=cfg.proximity_tau_scale,
    )
    summary.update(inertia_metrics)

    run_out = cfg.out_dir / run_name
    ensure_dir(run_out)
    corr_raw.to_csv(run_out / "corridors_exact_clustered_raw.csv", sep=";", index=False)
    corr_red.to_csv(run_out / "corridors_reduced.csv", sep=";", index=False)
    compactness_df.to_csv(run_out / "cluster_compactness.csv", sep=";", index=False)
    transfer_samples.to_csv(run_out / "transfer_samples.csv", sep=";", index=False)
    inertia_by_bus_df.to_csv(run_out / "inertia_proxy_by_cluster.csv", sep=";", index=False)
    pairdist_df.to_csv(run_out / "effective_reactance_pairs.csv", sep=";", index=False)
    mesh_detail_df.to_csv(run_out / "mesh_loop_stats.csv", sep=";", index=False)
    multiinj_samples.to_csv(run_out / "multiinj_samples.csv", sep=";", index=False)

    return {
        "summary": summary,
        "corr_raw": corr_raw,
        "corr_red": corr_red
    }


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark precomputed network reductions for DC-OPF + inertia use cases.")
    p.add_argument("--raw-buses", required=True, type=Path, help="Path to raw buses.csv")
    p.add_argument("--raw-lines", required=True, type=Path, help="Path to raw lines.csv")
    p.add_argument("--plants-with-bus", required=True, type=Path, help="Path to plants_with_bus.csv")
    p.add_argument("--reductions-root", required=True, type=Path, help="Folder that contains run subfolders")
    p.add_argument("--out-dir", required=True, type=Path, help="Output folder for benchmark results")
    p.add_argument("--all-ac", action="store_true", help="Evaluate all AC buses, not only CESA AC")
    p.add_argument("--n-transfer-samples", type=int, default=250)
    p.add_argument("--transfer-injection-mw", type=float, default=1000.0)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--proximity-kernel", type=str, default="exp", choices=["exp", "reciprocal"])
    p.add_argument("--proximity-tau-scale", type=float, default=1.0)
    p.add_argument("--multiinj-k-active-min", type=int, default=3)
    p.add_argument("--multiinj-k-active-max", type=int, default=8)
    p.add_argument("--multiinj-injection-mw", type=float, default=1000.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = EvalConfig(
        raw_buses_csv=args.raw_buses,
        raw_lines_csv=args.raw_lines,
        plants_with_bus_csv=args.plants_with_bus,
        reductions_root=args.reductions_root,
        out_dir=args.out_dir,
        evaluate_only_cesa_ac=not args.all_ac,
        n_transfer_samples=args.n_transfer_samples,
        transfer_injection_mw=args.transfer_injection_mw,
        random_seed=args.random_seed,
        proximity_kernel=args.proximity_kernel,
        proximity_tau_scale=args.proximity_tau_scale,
        multiinj_k_active_min=args.multiinj_k_active_min,
        multiinj_k_active_max=args.multiinj_k_active_max,
        multiinj_injection_mw=args.multiinj_injection_mw,
    )

    ensure_dir(cfg.out_dir)

    raw_buses = prep_raw_buses(read_net_csv(cfg.raw_buses_csv))
    raw_lines = prep_raw_lines(read_net_csv(cfg.raw_lines_csv))
    plants_with_bus = prep_plants_with_bus(read_semicolon_csv(cfg.plants_with_bus_csv))

    run_dirs = find_run_dirs(cfg.reductions_root)
    if not run_dirs:
        raise FileNotFoundError(
            f"No reduction run folders found in {cfg.reductions_root}. "
            "Expected subfolders with bus_map.csv, buses.csv, lines.csv, plants.csv."
        )

    summaries = []
    for run_dir in run_dirs:
        print(f"Evaluating run: {run_dir.name}")
        res = evaluate_run(run_dir, raw_buses, raw_lines, plants_with_bus, cfg)
        summaries.append(res["summary"])

    summary_df = pd.DataFrame(summaries)
    ranking_df = build_ranking(summary_df, cfg.score_weights)

    summary_df.to_csv(cfg.out_dir / "benchmark_summary.csv", sep=";", index=False)
    ranking_df.to_csv(cfg.out_dir / "benchmark_ranking.csv", sep=";", index=False)

    print("\nDone.")
    print(f"Stored summary in   : {cfg.out_dir / 'benchmark_summary.csv'}")
    print(f"Stored ranking in   : {cfg.out_dir / 'benchmark_ranking.csv'}")



if __name__ == "__main__":
    main()
