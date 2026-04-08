"""
Schema normalization and validation.
=====================================
Takes the best from all three versions:
  - mvp_repo:        flexible _first_present() column detection, tolerant of missing optionals
  - pipeline_updated: richer canonical schema (parent_image, cmdline, integrity_level, signed)
  - v12_modular:      SchemaSpec dataclass, coerce_types(), split_by_host()

Canonical output columns after normalization:
  ts, host, event_id, image, parent_image, cmdline, user, integrity_level, signed,
  hash_sha256, dest_ip, dest_port, protocol
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
import pandas as pd

from .config import StrataConfig


# ---------------------------------------------------------------------------
# Schema spec (from v12_modular)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchemaSpec:
    required: Tuple[str, ...]
    optional: Tuple[str, ...] = ()


CANONICAL_REQUIRED = ("ts", "host", "event_id")
CANONICAL_OPTIONAL = (
    "image", "parent_image", "cmdline", "user",
    "integrity_level", "signed",
    "hash_sha256", "dest_ip", "dest_port", "protocol",
)
ALL_CANONICAL = CANONICAL_REQUIRED + CANONICAL_OPTIONAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_present(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return first column name from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Main normalization entry point
# ---------------------------------------------------------------------------

def normalize_schema(raw: pd.DataFrame, cfg: StrataConfig) -> pd.DataFrame:
    """
    Coerce raw Sysmon/WEC exports into the STRATA canonical schema.

    Combines:
    - Flexible multi-candidate column detection (mvp_repo)
    - Rich canonical column set (pipeline_updated)
    - Type coercion pattern (v12_modular)

    Required output: ts, host, event_id
    Optional output: image, parent_image, cmdline, user, integrity_level,
                     signed, hash_sha256, dest_ip, dest_port, protocol
    """
    out = raw.copy()
    icfg = cfg.io

    # --- Timestamp ---
    ts_col = _first_present(out, icfg.timestamp_cols)
    if ts_col is None:
        raise ValueError(
            f"No timestamp column found. Tried: {icfg.timestamp_cols}\n"
            f"Available: {list(out.columns)[:30]}"
        )
    out["ts"] = pd.to_datetime(out[ts_col], errors="coerce", utc=True)

    # --- Host ---
    host_col = _first_present(out, icfg.host_cols)
    if host_col is None:
        raise ValueError(
            f"No host column found. Tried: {icfg.host_cols}\n"
            f"Available: {list(out.columns)[:30]}"
        )
    out["host"] = out[host_col].astype(str)

    # --- Event ID ---
    eid_col = _first_present(out, icfg.event_id_cols)
    if eid_col is None:
        raise ValueError(
            f"No event_id column found. Tried: {icfg.event_id_cols}\n"
            f"Available: {list(out.columns)[:30]}"
        )
    out["event_id"] = pd.to_numeric(out[eid_col], errors="coerce").astype("Int64")

    # --- Optional columns (tolerant: create empty if not found) ---
    def _pull(candidates, fill="", dtype="string"):
        col = _first_present(out, candidates)
        if col:
            return out[col].fillna(fill).astype(dtype)
        return pd.Series([fill] * len(out), dtype=dtype)

    out["image"]           = _pull(icfg.image_cols)
    out["parent_image"]    = _pull(icfg.parent_image_cols)
    out["cmdline"]         = _pull(icfg.cmdline_cols)
    out["user"]            = _pull(icfg.user_cols)
    out["integrity_level"] = _pull(icfg.integrity_cols)

    signed_col = _first_present(out, icfg.signed_cols)
    out["signed"] = out[signed_col].fillna(False).astype(bool) if signed_col else False

    # Network / hash columns (create empty if missing)
    for col in ("hash_sha256", "dest_ip", "dest_port", "protocol"):
        if col not in out.columns:
            out[col] = None

    # --- Drop events with unusable required fields ---
    out = out.dropna(subset=["ts", "host"]).copy()
    out["event_id"] = out["event_id"].astype("Int64")

    # --- Drop configured event IDs ---
    if cfg.scoring.drop_event_ids:
        out = out[~out["event_id"].isin(list(cfg.scoring.drop_event_ids))]

    # --- Optional time window filter ---
    if icfg.time_min:
        out = out[out["ts"] >= pd.to_datetime(icfg.time_min, utc=True)]
    if icfg.time_max:
        out = out[out["ts"] < pd.to_datetime(icfg.time_max, utc=True)]

    # --- Sort for determinism ---
    out = out.sort_values(["host", "ts"]).reset_index(drop=True)

    # --- Return only canonical columns ---
    return out[list(ALL_CANONICAL)].copy()


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if required canonical columns are missing or malformed."""
    spec = SchemaSpec(required=CANONICAL_REQUIRED)
    missing = [c for c in spec.required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")
    if df["ts"].isna().any():
        raise ValueError("Null timestamps found after normalization.")
    if df["host"].isna().any():
        raise ValueError("Null hosts found after normalization.")


def split_by_host(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Partition events by host. Returns {host_str: DataFrame}."""
    return {str(h): g.copy() for h, g in df.groupby("host", dropna=False)}
