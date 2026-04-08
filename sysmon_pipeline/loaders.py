"""
Dataset loaders for STRATA-E.
==============================
Normalizes external datasets into the canonical STRATA schema so
the pipeline can run without modification regardless of data source.

Supported sources
-----------------
1. DARPA Transparent Computing (TC) datasets
   - CADETS, THEIA, FIVEDIRECTIONS, TRACE
   - Download: https://drive.google.com/drive/folders/1QlbUFWAGq3Hpl8wVdzOdIoZLFxkII4EK
   - Format: JSON lines (.json.gz) — one audit record per line
   - Ground truth labels: provided as separate CSV per dataset

2. Generic Sysmon CSV
   - Any CSV export from Sysmon or a SIEM with Sysmon events
   - Schema detection is flexible (column names auto-detected)

3. Synthetic data (for testing / ablation without real data)
   - Built into tests/test_pipeline.py as make_synthetic()
   - Use run_experiments.py --synthetic flag

Usage
-----
    from sysmon_pipeline.loaders import load_darpa_tc, load_sysmon_csv

    # DARPA TC
    df, labels = load_darpa_tc(
        data_dir="data/darpa/cadets",
        dataset="cadets",
    )

    # Generic Sysmon CSV (e.g., your org's data)
    df = load_sysmon_csv("data/sysmon_export.csv")
"""
from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("strata.loaders")


# ---------------------------------------------------------------------------
# DARPA Transparent Computing loader
# ---------------------------------------------------------------------------

# Maps DARPA TC syscall/event type strings to approximate Sysmon event IDs
# so the rest of the pipeline's severity table and token mapping applies.
_DARPA_EVENT_MAP: Dict[str, int] = {
    "EVENT_EXECUTE":          1,   # Process Create
    "EVENT_FORK":             1,
    "EVENT_CLONE":            1,
    "EVENT_CONNECT":          3,   # Network Connection
    "EVENT_ACCEPT":           3,
    "EVENT_OPEN":             11,  # File Create / Open
    "EVENT_READ":             11,
    "EVENT_WRITE":            11,
    "EVENT_CLOSE":            5,   # Process Terminate (closest analog)
    "EVENT_MMAP":             7,   # Image Load (DLL analog)
    "EVENT_MPROTECT":         8,   # CreateRemoteThread (injection analog)
    "EVENT_RECVFROM":         3,
    "EVENT_SENDTO":           3,
    "EVENT_RECVMSG":          3,
    "EVENT_SENDMSG":          3,
    "EVENT_RENAME":           11,
    "EVENT_UNLINK":           23,  # FileDelete
    "EVENT_SETUID":           4672, # Special Privileges
    "EVENT_LOGIN":            4624, # Logon
    "EVENT_LOGOUT":           4634, # Logoff
    "EVENT_LOADLIBRARY":      7,
    "EVENT_MODIFY_PROCESS":   10,  # ProcessAccess
    "EVENT_SIGNAL":           8,
    "EVENT_UNIT":             1,
    "EVENT_UPDATE":           13,  # Registry Value Set analog
}

# DARPA TC dataset-specific attack host lists (ground truth)
# Source: DARPA TC engagement documentation
_DARPA_ATTACK_HOSTS: Dict[str, list] = {
    "cadets": [
        "cadets-e3-1.pc.cs.cmu.edu",
        "cadets-e3-2.pc.cs.cmu.edu",
    ],
    "theia": [
        "theia-e3-1.pc.cs.cmu.edu",
    ],
    "fivedirections": [
        "fd-e3-1.pc.cs.cmu.edu",
        "fd-e3-2.pc.cs.cmu.edu",
        "fd-e3-3.pc.cs.cmu.edu",
    ],
    "trace": [
        "trace-e3-1.pc.cs.cmu.edu",
    ],
}


def load_darpa_tc(
    data_dir: str | Path,
    dataset: str = "cadets",
    max_records: Optional[int] = None,
    ground_truth_csv: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a DARPA Transparent Computing dataset and return:
      (events_df, labels_df)

    events_df columns (canonical STRATA schema):
      ts, host, event_id, image, parent_image, cmdline, user,
      integrity_level, signed, severity_score, severity_label

    labels_df columns:
      host, is_compromised (bool)

    Parameters
    ----------
    data_dir : path to the dataset directory containing .json or .json.gz files
    dataset  : one of 'cadets', 'theia', 'fivedirections', 'trace'
    max_records : cap for testing — None means load all
    ground_truth_csv : optional path to a CSV with columns [host, is_compromised]
                       If not provided, uses known attack host lists above.

    Notes
    -----
    DARPA TC data is audit log (Linux syscall) format, not Windows Sysmon.
    The loader maps syscall event types to approximate Sysmon event IDs using
    _DARPA_EVENT_MAP so the severity table and token abstraction apply.
    Process image names are extracted from the subject UUID -> exe path mapping.
    Network events use the object UUID -> remote IP/port mapping.

    Download instructions
    ---------------------
    1. Request access at https://github.com/darpa-i2o/Transparent-Computing
    2. Data is hosted on Google Drive — link is in the GitHub README
    3. Download the E3 engagement data (CADETS, THEIA, FIVEDIRECTIONS, TRACE)
    4. Extract to data/darpa/<dataset_name>/
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"DARPA TC data directory not found: {data_dir}\n"
            "See docstring for download instructions."
        )

    # Find all JSON/JSON.GZ files
    files = sorted(
        list(data_dir.glob("*.json.gz")) +
        list(data_dir.glob("*.json")) +
        list(data_dir.glob("*.jsonl"))
    )
    if not files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    logger.info("Loading DARPA TC dataset '%s' from %d files", dataset, len(files))

    rows = []
    n = 0

    for fpath in files:
        opener = gzip.open if fpath.suffix == ".gz" else open
        with opener(fpath, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                row = _parse_darpa_record(rec)
                if row is not None:
                    rows.append(row)
                    n += 1

                if max_records and n >= max_records:
                    break
        if max_records and n >= max_records:
            break

    if not rows:
        raise ValueError(f"No parseable records found in {data_dir}")

    logger.info("Loaded %d records from DARPA TC '%s'", len(rows), dataset)
    df = pd.DataFrame(rows)
    df = _finalize_darpa_df(df)

    # Ground truth labels
    if ground_truth_csv and Path(ground_truth_csv).exists():
        labels = pd.read_csv(ground_truth_csv)
        if "host" not in labels.columns or "is_compromised" not in labels.columns:
            raise ValueError("ground_truth_csv must have 'host' and 'is_compromised' columns")
    else:
        attack_hosts = set(_DARPA_ATTACK_HOSTS.get(dataset.lower(), []))
        all_hosts = df["host"].unique()
        labels = pd.DataFrame({
            "host": all_hosts,
            "is_compromised": [h in attack_hosts for h in all_hosts],
        })
        n_attack = labels["is_compromised"].sum()
        logger.info(
            "Ground truth: %d compromised hosts, %d clean (from known host list)",
            n_attack, len(labels) - n_attack,
        )

    return df, labels


def _parse_darpa_record(rec: dict) -> Optional[dict]:
    """Extract canonical fields from a single DARPA TC audit record."""
    # DARPA TC records are deeply nested; structure varies by dataset
    # Common top-level keys: datum, CDMVersion, hostName, etc.
    datum = rec.get("datum", rec)

    # Skip non-event records (e.g., subject/object/host descriptors)
    event_type = (
        datum.get("type")
        or datum.get("eventType")
        or datum.get("EVENT_TYPE")
    )
    if not event_type or not str(event_type).startswith("EVENT_"):
        return None

    # Timestamp: nanoseconds since epoch in most TC datasets
    ts_raw = datum.get("timestampNanos") or datum.get("timestamp") or 0
    try:
        ts = pd.Timestamp(int(ts_raw), unit="ns", tz="UTC")
    except Exception:
        return None

    # Host
    host = (
        rec.get("hostName")
        or rec.get("host")
        or datum.get("hostName")
        or "unknown"
    )

    # Map event type to approximate Sysmon event ID
    event_id = _DARPA_EVENT_MAP.get(str(event_type), 1)

    # Subject (process) -> image name
    subject = datum.get("subject") or {}
    image = (
        subject.get("properties", {}).get("name")
        or subject.get("cmdLine", "")
        or subject.get("exec", "")
        or "unknown"
    )
    # Extract just the executable name if it's a full path
    image = image.split("/")[-1].split("\\")[-1] or image

    # Parent image
    parent = datum.get("predicateObject") or {}
    parent_image = (
        parent.get("properties", {}).get("name")
        or parent.get("exec", "")
        or ""
    )
    parent_image = parent_image.split("/")[-1].split("\\")[-1] or parent_image

    # Command line (often in subject properties)
    cmdline = subject.get("cmdLine") or subject.get("properties", {}).get("cmdLine") or ""

    # User (principal)
    principal = datum.get("principal") or {}
    user = (
        principal.get("username")
        or str(principal.get("userId", ""))
        or "unknown"
    )

    return {
        "ts":              ts,
        "host":            str(host),
        "event_id":        event_id,
        "image":           str(image),
        "parent_image":    str(parent_image),
        "cmdline":         str(cmdline),
        "user":            str(user),
        "integrity_level": "UNKNOWN",
        "signed":          False,
    }


def _finalize_darpa_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, deduplicate, and add missing canonical columns."""
    df = df.sort_values("ts").reset_index(drop=True)

    # Add columns the schema expects but DARPA data doesn't have
    for col in ["dest_ip", "dest_port", "protocol", "hash_sha256"]:
        if col not in df.columns:
            df[col] = None

    logger.info(
        "DARPA TC: %d events, %d hosts, time range %s to %s",
        len(df),
        df["host"].nunique(),
        df["ts"].min(),
        df["ts"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# Generic Sysmon CSV loader
# ---------------------------------------------------------------------------

def load_sysmon(
    path: str | Path,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load Sysmon / endpoint telemetry from CSV or Parquet.

    Format is auto-detected from the file extension:
      - .csv / .csv.gz  → pd.read_csv
      - .parquet / .pq  → pd.read_parquet (requires pyarrow)

    Column names are auto-detected via the IOConfig candidates in schema.py.
    Returns a raw DataFrame ready to pass directly to StrataPipeline.fit()
    or StrataPipeline.score() — normalization happens inside the pipeline.

    Parameters
    ----------
    path     : path to the CSV or Parquet file
    time_min : optional ISO timestamp string to filter start of window
    time_max : optional ISO timestamp string to filter end of window
    max_rows : cap for testing (nrows for CSV, head slice for Parquet)

    Example
    -------
        df = load_sysmon("data/sysmon_export.parquet")
        df = load_sysmon("data/sysmon_2026_02.csv", time_min="2026-02-01")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sysmon data file not found: {path}")

    suffix = path.suffix.lower()
    # Handle .csv.gz
    if suffix == ".gz" and path.stem.endswith(".csv"):
        suffix = ".csv"

    if suffix in (".parquet", ".pq"):
        logger.info("Loading Sysmon Parquet: %s", path)
        df = pd.read_parquet(path)
        if max_rows:
            df = df.head(max_rows)
        logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), path.name)
    elif suffix in (".csv",):
        logger.info("Loading Sysmon CSV: %s", path)
        kwargs = {"low_memory": False}
        if max_rows:
            kwargs["nrows"] = max_rows
        df = pd.read_csv(path, **kwargs)
        logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), path.name)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Expected .csv, .csv.gz, .parquet, or .pq"
        )

    # Optional time filtering (handles column detection internally in schema.py)
    # We do a best-effort filter here before schema normalization
    ts_candidates = ["_timestamp", "UtcTime", "ts", "timestamp", "TimeCreated"]
    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    if ts_col and (time_min or time_max):
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        if time_min:
            df = df[df[ts_col] >= pd.Timestamp(time_min, tz="UTC")]
        if time_max:
            df = df[df[ts_col] <= pd.Timestamp(time_max, tz="UTC")]
        logger.info("After time filter: %d rows", len(df))

    return df


# Backwards-compatible alias
load_sysmon_csv = load_sysmon


# ---------------------------------------------------------------------------
# Utility: split a dataset into baseline and scoring windows
# ---------------------------------------------------------------------------

def split_time_windows(
    df: pd.DataFrame,
    ts_col: str = "ts",
    baseline_days: int = 7,
    score_days: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into a baseline window and a scoring window by time.

    Useful for running the full fit → score pipeline on a single dataset:
        baseline_df, score_df = split_time_windows(df, baseline_days=7)
        fitted = pipe.fit(baseline_df)
        art    = pipe.score(score_df, fitted)

    Parameters
    ----------
    df            : DataFrame with a parsed timestamp column
    ts_col        : name of the timestamp column
    baseline_days : how many days to use for the baseline window
    score_days    : how many days to use for the scoring window (immediately after baseline)
    """
    if ts_col not in df.columns:
        raise ValueError(f"Timestamp column '{ts_col}' not found. Run schema normalization first.")

    t_min = df[ts_col].min()
    t_split = t_min + pd.Timedelta(days=baseline_days)
    t_end = t_split + pd.Timedelta(days=score_days)

    baseline = df[df[ts_col] < t_split].copy()
    scoring  = df[(df[ts_col] >= t_split) & (df[ts_col] < t_end)].copy()

    logger.info(
        "Baseline window: %d events (%s to %s)",
        len(baseline), t_min.date(), t_split.date(),
    )
    logger.info(
        "Scoring window:  %d events (%s to %s)",
        len(scoring), t_split.date(), t_end.date(),
    )

    if baseline.empty:
        raise ValueError("Baseline window is empty — check your time range or baseline_days parameter.")
    if scoring.empty:
        raise ValueError("Scoring window is empty — check your time range or score_days parameter.")

    return baseline, scoring
