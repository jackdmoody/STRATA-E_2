"""
Sessionization and transition count building.
===============================================
Best of:
  - pipeline_updated: session assignment, delta bucketing, transition count builder
  - v12_modular:      build_state_map, normalize_rows, build_host_markov_matrix
  - Fix 3:            adaptive tau_gap fitted from inter-event time distribution per role
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

from .config import StrataConfig


# ---------------------------------------------------------------------------
# Fix 3: Adaptive session gap fitting
# ---------------------------------------------------------------------------

def fit_gap_threshold(
    inter_event_times: np.ndarray,
    method: str = "kde_valley",
    fallback_percentile: float = 95.0,
) -> float:
    """
    Fit τgap from the inter-event time distribution rather than using a fixed constant.

    Finds the natural valley between "within-session" and "between-session" gaps
    using KDE on log(Δt). Falls back to a percentile if no clear bimodal structure.
    """
    iet = inter_event_times[inter_event_times > 0]
    if len(iet) < 50:
        return float(np.percentile(iet, fallback_percentile))

    if method == "kde_valley":
        log_iet = np.log1p(iet).reshape(-1, 1)
        kde = KernelDensity(kernel="gaussian", bandwidth=0.3)
        kde.fit(log_iet)
        x_grid = np.linspace(log_iet.min(), log_iet.max(), 500).reshape(-1, 1)
        density = np.exp(kde.score_samples(x_grid))
        valleys = argrelextrema(density, np.less, order=10)[0]
        if len(valleys) > 0:
            return float(np.expm1(x_grid[valleys[0]][0]))

    return float(np.percentile(iet, fallback_percentile))


def build_role_gap_thresholds(
    events: pd.DataFrame,
    cfg: StrataConfig,
    role_col: str = "role_id",
) -> Dict[str, float]:
    """
    Fit one τgap per role by pooling inter-event times across all hosts in that role.
    Call once during baseline construction; persist the result with your artifacts.
    """
    thresholds: Dict[str, float] = {}

    for role, group in events.groupby(role_col):
        all_iets: List[float] = []
        for _, host_events in group.groupby("host"):
            times = host_events["ts"].sort_values().values.astype("datetime64[s]").astype(float)
            if len(times) > 1:
                all_iets.extend(np.diff(times).tolist())

        fallback = cfg.time.session_gap_seconds
        thresholds[str(role)] = fit_gap_threshold(np.array(all_iets)) if all_iets else fallback

    return thresholds


# ---------------------------------------------------------------------------
# Sessionization (from pipeline_updated, extended with adaptive gap)
# ---------------------------------------------------------------------------

def assign_sessions(
    df: pd.DataFrame,
    cfg: StrataConfig,
    role_gaps: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Assign session_id per host using τgap (fixed or role-adaptive).

    If role_gaps is provided and use_adaptive_tau_gap=True, each host uses
    the gap threshold for its assigned role.
    """
    out = df.sort_values(["host", "ts"]).copy()

    # Determine gap per host
    default_gap = float(cfg.time.session_gap_seconds)

    if cfg.ablation.use_adaptive_tau_gap and role_gaps and "role_id" in out.columns:
        host_role_map = (
            out[["host", "role_id"]]
            .drop_duplicates("host")
            .set_index("host")["role_id"]
            .to_dict()
        )
        host_gap_map = {
            h: role_gaps.get(str(r), default_gap)
            for h, r in host_role_map.items()
        }
    else:
        host_gap_map = {}

    out["_prev_ts"] = out.groupby("host")["ts"].shift(1)
    out["_gap_s"] = (out["ts"] - out["_prev_ts"]).dt.total_seconds()

    def _new_session_mask(group):
        # group.name is the host key when grouping by "host"
        host = group.name
        gap = host_gap_map.get(str(host), default_gap)
        return group["_gap_s"].isna() | (group["_gap_s"] > gap)

    new_sess = out.groupby("host", group_keys=False)["_gap_s"].apply(
        lambda g: g.isna() | (g > host_gap_map.get(str(g.name), default_gap))
    ) if host_gap_map else (out["_gap_s"].isna() | (out["_gap_s"] > default_gap))
    out["_session_num"] = new_sess.groupby(out["host"]).cumsum()
    out["session_id"] = out["host"].astype(str) + ":" + out["_session_num"].astype(int).astype(str)

    out = out.drop(columns=["_prev_ts", "_gap_s", "_session_num"])
    return out


def bucket_deltas(df: pd.DataFrame, cfg: StrataConfig) -> pd.DataFrame:
    """Compute dt_s and dt_bucket within each session (from pipeline_updated).

    Vectorized: uses np.searchsorted instead of a Python loop over every row.
    """
    out = df.sort_values(["session_id", "ts"]).copy()
    out["_prev_ts_sess"] = out.groupby("session_id")["ts"].shift(1)
    out["dt_s"] = (out["ts"] - out["_prev_ts_sess"]).dt.total_seconds().fillna(0.0)

    bounds = cfg.time.buckets
    # Build parallel arrays for searchsorted: upper bounds and labels
    upper_bounds = np.array([ub for ub, _ in bounds])
    bucket_labels = [lab for _, lab in bounds]

    dt_vals = out["dt_s"].to_numpy()
    # searchsorted finds the first bucket where dt <= upper_bound
    indices = np.searchsorted(upper_bounds, dt_vals, side="left")
    # Clip to valid range (shouldn't exceed, but safety)
    indices = np.clip(indices, 0, len(bucket_labels) - 1)
    out["dt_bucket"] = [bucket_labels[i] for i in indices]

    out = out.drop(columns=["_prev_ts_sess"])
    return out


# ---------------------------------------------------------------------------
# Transition counting (combines pipeline_updated + v12_modular)
# ---------------------------------------------------------------------------

def build_transition_counts(
    df: pd.DataFrame,
    cfg: StrataConfig,
    level: str = "token_medium",
) -> pd.DataFrame:
    """
    Build time-aware transition counts: (host, state, next_state, dt_bucket, count).

    This is the pipeline_updated approach (richer than v12_modular's raw matrix dict)
    because it keeps transitions as a tidy DataFrame, which is required for:
    - Dirichlet shrinkage baseline fitting
    - JSD calibration
    - Drift scoring
    - Explainability (top rare transitions)
    """
    out = df.sort_values(["session_id", "ts"]).copy()
    out["state"] = out[level].astype(str)
    out["next_state"] = out.groupby("session_id")[level].shift(-1)
    out["next_dt_bucket"] = out.groupby("session_id")["dt_bucket"].shift(-1)

    trans = out.dropna(subset=["next_state", "next_dt_bucket"]).copy()
    trans = (
        trans.groupby(["host", "state", "next_state", "next_dt_bucket"])
        .size()
        .reset_index(name="count")
        .rename(columns={"next_dt_bucket": "dt_bucket"})
    )
    return trans


def build_state_map(df: pd.DataFrame, event_id_col: str = "event_id") -> Dict[int, int]:
    """Map raw event IDs to consecutive integer indices (from v12_modular)."""
    uniq = pd.Series(df[event_id_col].dropna().astype("Int64").astype(int).unique()).sort_values()
    return {int(v): i for i, v in enumerate(uniq.tolist())}


def build_host_markov_matrix(transition_counts: np.ndarray) -> np.ndarray:
    """Row-normalize a transition count matrix into a probability matrix (from v12_modular)."""
    rs = transition_counts.sum(axis=1, keepdims=True)
    return transition_counts / (rs + 1e-12)


def compute_baseline_markov_matrix(
    host_markov: Dict[str, np.ndarray],
    baseline_hosts: List[str],
) -> np.ndarray:
    """Average Markov matrices over baseline hosts (from v12_modular)."""
    mats = [host_markov[h] for h in baseline_hosts if h in host_markov]
    if not mats:
        raise ValueError("No baseline hosts found in host_markov dict.")
    avg = np.mean(mats, axis=0)
    rs = avg.sum(axis=1, keepdims=True)
    return avg / (rs + 1e-12)
