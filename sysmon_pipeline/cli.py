#!/usr/bin/env python3
"""
strata_cli.py
===============
Interactive CLI for the STRATA-E Endpoint Behavioral Anomaly Detection Pipeline.

Provides three modes:
  1. Interactive  — walk through config sections, change what you want, run
  2. Config file  — load a JSON config, optionally override specific fields
  3. Quick run    — all defaults with just an input path specified

Usage
------
    # Interactive mode — prompts for every section
    python strata_cli.py --interactive

    # Quick run on a Sysmon CSV with HTML report
    python strata_cli.py --input data/sysmon.csv --report

    # Quick run on Parquet with custom split
    python strata_cli.py --input data/sysmon.parquet --baseline-days 14 --score-days 2

    # Load a previously fitted model and score new data
    python strata_cli.py --input data/scoring_window.csv --load-model models/fitted.pkl

    # Fit on explicit baseline file, score on separate scoring file
    python strata_cli.py --input data/scoring.csv --baseline data/baseline.csv --save-model models/fitted.pkl

    # DARPA TC dataset
    python strata_cli.py --dataset darpa --data-dir data/darpa/cadets/ --darpa-name cadets --report

    # Synthetic mode — generate data and run
    python strata_cli.py --synthetic --report --browser

    # Load config JSON, override scoring weights
    python strata_cli.py --input data/sysmon.csv --config my_config.json --override scoring.w_seq=0.5

    # Ablation study (sequence-only condition)
    python strata_cli.py --input data/sysmon.csv --ablation sequence_only --report

    # Save current config to JSON (dry run)
    python strata_cli.py --interactive --save-config my_config.json --dry-run

    # Show all defaults
    python strata_cli.py --show-defaults
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("strata_cli")


# ---------------------------------------------------------------------------
# Helpers (terminal formatting)
# ---------------------------------------------------------------------------

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"

def _dim(s: str) -> str:
    return f"\033[2m{s}\033[0m"

def _cyan(s: str) -> str:
    return f"\033[36m{s}\033[0m"

def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"

def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"

def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"

def _banner(title: str) -> None:
    w = 60
    print(f"\n{'=' * w}")
    print(f"  {_bold(title)}")
    print(f"{'=' * w}")


def _prompt(label: str, default: Any, type_hint: type = str) -> Any:
    """Prompt for a single config value with type coercion."""
    default_str = str(default)
    if isinstance(default, bool):
        default_str = "yes" if default else "no"

    raw = input(f"  {label} [{_dim(default_str)}]: ").strip()

    if not raw:
        return default

    # Type coercion
    if isinstance(default, bool):
        return raw.lower() in ("yes", "y", "true", "1")
    elif isinstance(default, int):
        try:
            return int(raw)
        except ValueError:
            print(f"    {_yellow('Invalid integer, keeping default')}")
            return default
    elif isinstance(default, float):
        try:
            return float(raw)
        except ValueError:
            print(f"    {_yellow('Invalid float, keeping default')}")
            return default
    elif isinstance(default, tuple):
        items = [x.strip() for x in raw.split(",")]
        if all(isinstance(x, int) for x in default):
            try:
                return tuple(int(x) for x in items)
            except ValueError:
                print(f"    {_yellow('Invalid, keeping default')}")
                return default
        return tuple(items)
    elif isinstance(default, Path):
        return Path(raw)
    else:
        return raw


def _prompt_section(name: str, dc_instance: Any) -> Any:
    """Interactively prompt for all fields in a dataclass section."""
    _banner(f"Config: {name}")

    # Skip fields that are long tuples/lists users shouldn't hand-edit
    skip_fields = {
        "timestamp_cols", "host_cols", "event_id_cols", "image_cols",
        "parent_image_cols", "cmdline_cols", "user_cols", "integrity_cols",
        "signed_cols", "role_feature_cols", "buckets", "critical_labels",
        "drop_event_ids", "backoff_lambdas", "baseline_host_allowlist",
    }

    fields = dataclasses.fields(dc_instance)
    print(f"  {_dim(f'{len(fields)} parameters — press Enter to keep default')}\n")

    changes = {}
    for f in fields:
        if f.name in skip_fields:
            continue
        val = getattr(dc_instance, f.name)
        if dataclasses.is_dataclass(val):
            val = _prompt_section(f"  {name}.{f.name}", val)
            changes[f.name] = val
        else:
            new_val = _prompt(f.name, val)
            if new_val != val:
                changes[f.name] = new_val

    if changes:
        d = dataclasses.asdict(dc_instance)
        for k, v in changes.items():
            if dataclasses.is_dataclass(v):
                d[k] = dataclasses.asdict(v)
            else:
                d[k] = v
        return type(dc_instance)(**d)

    return dc_instance


def _prompt_yes_no(question: str, default: bool = True) -> bool:
    """Simple yes/no prompt."""
    hint = "Y/n" if default else "y/N"
    raw = input(f"  {question} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1")


# ---------------------------------------------------------------------------
# Config sections menu
# ---------------------------------------------------------------------------

SECTION_NAMES = [
    ("io",       "I/O paths and column detection"),
    ("time",     "Time bucketing and session gaps"),
    ("baseline", "Peer baseline and Dirichlet shrinkage"),
    ("role",     "Host role inference"),
    ("scoring",  "Multi-channel fusion and gating"),
    ("ablation", "Ablation flags (toggle components)"),
]


def _interactive_config() -> "StrataConfig":
    """Walk through config sections interactively."""
    from sysmon_pipeline.config import StrataConfig

    cfg = StrataConfig()

    _banner("STRATA-E Configuration")
    print(f"  {_dim('Select which sections to configure.')}")
    print(f"  {_dim('Press Enter to skip a section (keeps defaults).')}\n")

    print(f"  {'#':<4} {'Section':<18} Description")
    print(f"  {'—'*4} {'—'*18} {'—'*35}")
    for i, (key, desc) in enumerate(SECTION_NAMES, 1):
        print(f"  {i:<4} {_cyan(key):<27} {desc}")

    print()
    raw = input(f"  Sections to configure (e.g. 1,3,5 or 'all') [{_dim('Enter=skip all')}]: ").strip()

    if not raw:
        print(f"\n  {_green('Using all defaults.')}")
        return cfg

    if raw.lower() == "all":
        indices = list(range(len(SECTION_NAMES)))
    else:
        try:
            indices = [int(x.strip()) - 1 for x in raw.split(",")]
            indices = [i for i in indices if 0 <= i < len(SECTION_NAMES)]
        except ValueError:
            print(f"  {_yellow('Invalid input, using defaults.')}")
            return cfg

    for idx in indices:
        key, _ = SECTION_NAMES[idx]
        section = getattr(cfg, key)
        updated = _prompt_section(key, section)
        setattr(cfg, key, updated)

    # Also prompt for token_resolution (top-level field, not in a sub-config)
    if _prompt_yes_no("Configure token resolution?", default=False):
        cfg.token_resolution = _prompt(
            "token_resolution (coarse/medium)", cfg.token_resolution
        )

    return cfg


# ---------------------------------------------------------------------------
# Config display
# ---------------------------------------------------------------------------

def _show_config(cfg: "StrataConfig") -> None:
    """Pretty-print the current config."""
    _banner("Current Configuration")
    d = cfg.as_dict()
    for section_name, section_dict in d.items():
        if isinstance(section_dict, dict):
            print(f"\n  {_cyan(section_name)}:")
            for k, v in section_dict.items():
                if isinstance(v, dict):
                    print(f"    {k}:")
                    for k2, v2 in v.items():
                        val_str = str(v2)
                        if len(val_str) > 60:
                            val_str = val_str[:57] + "..."
                        print(f"      {k2:<35} = {val_str}")
                else:
                    val_str = str(v)
                    if len(val_str) > 60:
                        val_str = val_str[:57] + "..."
                    print(f"    {k:<35} = {val_str}")
        else:
            print(f"\n  {_cyan(section_name):<27} = {section_dict}")


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="strata_cli",
        description="STRATA-E — Endpoint Behavioral Anomaly Detection Pipeline (CLI Interface)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data/sysmon.csv --report
  %(prog)s --input data/sysmon.parquet --baseline-days 14 --report --browser
  %(prog)s --input data/scoring.csv --load-model models/fitted.pkl --report
  %(prog)s --input data/scoring.csv --baseline data/baseline.csv --save-model models/fitted.pkl
  %(prog)s --dataset darpa --data-dir data/darpa/cadets/ --report
  %(prog)s --synthetic --n-hosts 100 --n-events 500 --report --browser
  %(prog)s --interactive
  %(prog)s --config my_config.json --override scoring.w_seq=0.5
  %(prog)s --show-defaults
        """,
    )

    # ── Input modes ──────────────────────────────────────────────────────
    inp = p.add_argument_group("Input")
    inp.add_argument("--input",       type=str,
                     help="Path to Sysmon/Windows event log (CSV, Parquet, or JSONL)")
    inp.add_argument("--baseline",    type=str,
                     help="Explicit baseline file (skips auto-split; --input becomes scoring window)")
    inp.add_argument("--labels",      type=str,
                     help="Ground truth labels CSV (columns: host, is_compromised)")
    inp.add_argument("--dataset",     choices=["sysmon", "darpa"],
                     help="Dataset loader to use (default: auto-detect from file extension)")
    inp.add_argument("--data-dir",    type=str,
                     help="DARPA TC dataset directory (used with --dataset darpa)")
    inp.add_argument("--darpa-name",  type=str, default="cadets",
                     choices=["cadets", "theia", "fivedirections", "trace"],
                     help="DARPA TC dataset name (default: cadets)")
    inp.add_argument("--max-records", type=int, default=None,
                     help="Cap records loaded (for testing on large files)")

    # ── Synthetic mode ───────────────────────────────────────────────────
    syn = p.add_argument_group("Synthetic data")
    syn.add_argument("--synthetic",   action="store_true",
                     help="Generate synthetic data and run (no dataset needed)")
    syn.add_argument("--n-hosts",     type=int, default=50,
                     help="Number of hosts (default: 50)")
    syn.add_argument("--n-events",    type=int, default=300,
                     help="Events per host (default: 300)")
    syn.add_argument("--n-attack",    type=int, default=10,
                     help="Number of attack hosts (default: 10)")
    syn.add_argument("--seed",        type=int, default=42,
                     help="RNG seed (default: 42)")

    # ── Time split ───────────────────────────────────────────────────────
    split = p.add_argument_group("Baseline / Scoring split")
    split.add_argument("--baseline-days", type=int, default=7,
                       help="Days for baseline window (default: 7)")
    split.add_argument("--score-days",    type=int, default=1,
                       help="Days for scoring window (default: 1)")
    split.add_argument("--split-ratio",   type=float, default=None,
                       help="Fractional split instead of calendar split (e.g. 0.8)")
    split.add_argument("--no-split",      action="store_true",
                       help="Use fit_score on the full dataset (no baseline/scoring split). "
                            "Use this when data spans < 7 days.")

    # ── Model persistence ────────────────────────────────────────────────
    mdl = p.add_argument_group("Model persistence")
    mdl.add_argument("--save-model",  type=str,
                     help="Save fitted model (FittedArtifacts) to pickle file after fit()")
    mdl.add_argument("--load-model",  type=str,
                     help="Load a previously saved model — skip fit(), go straight to score()")

    # ── Configuration ────────────────────────────────────────────────────
    conf = p.add_argument_group("Configuration")
    conf.add_argument("--config",        type=str,
                      help="Load config from JSON file")
    conf.add_argument("--save-config",   type=str,
                      help="Save final config to JSON before running")
    conf.add_argument("--interactive",   action="store_true",
                      help="Interactive config walkthrough")
    conf.add_argument("--override",      type=str, nargs="*", metavar="KEY=VALUE",
                      help="Override config fields (e.g. scoring.w_seq=0.5 baseline.dirichlet_kappa=15)")
    conf.add_argument("--show-defaults", action="store_true",
                      help="Print default config and exit")
    conf.add_argument("--ablation",      type=str, default="full",
                      choices=["full", "sequence_only", "no_shrinkage",
                               "no_role_baselining", "no_calibration", "no_drift"],
                      help="Ablation condition preset (default: full)")

    # ── Output ───────────────────────────────────────────────────────────
    out = p.add_argument_group("Output")
    out.add_argument("--output",     type=str, default="results",
                     help="Output directory (default: results)")
    out.add_argument("--report",     action="store_true",
                     help="Generate self-contained HTML report")
    out.add_argument("--eval",       action="store_true",
                     help="Run extended evaluation (H2-H5 plots, channel contribution, "
                          "SHAP, workload reduction, explainability, reproducibility)")
    out.add_argument("--eval-full",  action="store_true",
                     help="Run full evaluation including ablation, sensitivity sweeps, "
                          "and scalability benchmarks (adds ~25 min)")
    out.add_argument("--browser",    action="store_true",
                     help="Open HTML report in browser when done")
    out.add_argument("--no-plots",   action="store_true",
                     help="Suppress matplotlib diagnostic plots")
    out.add_argument("--fast",       action="store_true",
                     help="Dev mode: reduce bootstrap samples (200), IF trees (50), "
                          "TF-IDF features (300) for ~3-10× speedup. All channels "
                          "and SHAP remain active. Use defaults for paper-quality runs.")
    out.add_argument("--top-k",      type=int, default=20,
                     help="Number of hosts to show in triage results (default: 20)")
    out.add_argument("--dry-run",    action="store_true",
                     help="Build config and exit without running the pipeline")
    out.add_argument("--quiet",      action="store_true",
                     help="Suppress info-level log output")

    return p


# ---------------------------------------------------------------------------
# Config overrides from CLI flags
# ---------------------------------------------------------------------------

def _apply_overrides(cfg: "StrataConfig", overrides: list[str]) -> "StrataConfig":
    """
    Apply key=value overrides like 'scoring.w_seq=0.5'.
    Supports dotted paths into nested config sections.
    """
    for item in overrides:
        if "=" not in item:
            print(f"  {_yellow(f'Skipping invalid override (no =): {item}')}")
            continue

        key, val = item.split("=", 1)
        parts = key.strip().split(".")

        if len(parts) == 1:
            # Top-level field (e.g. token_resolution=coarse)
            if hasattr(cfg, parts[0]):
                current = getattr(cfg, parts[0])
                try:
                    if isinstance(current, bool):
                        coerced = val.lower() in ("true", "yes", "1")
                    elif isinstance(current, int):
                        coerced = int(val)
                    elif isinstance(current, float):
                        coerced = float(val)
                    else:
                        coerced = val
                    setattr(cfg, parts[0], coerced)
                    print(f"  {_green(f'Override: {key} = {coerced}')}")
                except (ValueError, TypeError):
                    print(f"  {_yellow(f'Type error for {key}={val}, keeping default')}")
            else:
                print(f"  {_yellow(f'Unknown top-level field: {parts[0]}')}")
            continue

        section_name = parts[0]
        field_name = parts[1]

        section = getattr(cfg, section_name, None)
        if section is None:
            print(f"  {_yellow(f'Unknown config section: {section_name}')}")
            continue

        if not hasattr(section, field_name):
            print(f"  {_yellow(f'Unknown field: {key}')}")
            continue

        current = getattr(section, field_name)

        try:
            if isinstance(current, bool):
                coerced = val.lower() in ("true", "yes", "1")
            elif isinstance(current, int):
                coerced = int(val)
            elif isinstance(current, float):
                coerced = float(val)
            elif isinstance(current, Path):
                coerced = Path(val)
            else:
                coerced = val
        except (ValueError, TypeError):
            print(f"  {_yellow(f'Type error for {key}={val}, keeping default')}")
            continue

        setattr(section, field_name, coerced)
        print(f"  {_green(f'Override: {key} = {coerced}')}")

    return cfg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data(
    args: argparse.Namespace,
) -> tuple["pd.DataFrame", "Optional[pd.DataFrame]"]:
    """
    Load the input dataset and optional labels.
    Returns (events_df, labels_df_or_None).
    """
    import pandas as pd
    from sysmon_pipeline.loaders import load_sysmon_csv, load_darpa_tc

    labels = None

    # ── Synthetic mode ────────────────────────────────────────────────
    if args.synthetic:
        _banner("Generating Synthetic Data")
        t0 = time.perf_counter()

        # Import make_synthetic from run_experiments if available,
        # otherwise fall back to a minimal inline generator
        try:
            from run_experiments import make_synthetic
        except ImportError:
            from _strata_synthetic import make_synthetic  # type: ignore[import]

        df, labels = make_synthetic(
            n_hosts=args.n_hosts,
            n_events_per_host=args.n_events,
            n_attack_hosts=args.n_attack,
            seed=args.seed,
        )
        elapsed = time.perf_counter() - t0
        # Detect host column (synthetic uses 'host', raw Sysmon uses 'Computer')
        host_col = next((c for c in ("host", "Computer", "Hostname", "host.fqdn") if c in df.columns), None)
        n_hosts = df[host_col].nunique() if host_col else "?"
        print(f"  Generated {len(df):,} events across {n_hosts} hosts in {elapsed:.1f}s")
        return df, labels

    # ── DARPA TC mode ─────────────────────────────────────────────────
    if args.dataset == "darpa":
        if not args.data_dir:
            print(f"  {_red('--data-dir is required when --dataset darpa')}")
            sys.exit(1)
        _banner("Loading DARPA TC Dataset")
        df, labels = load_darpa_tc(
            data_dir=args.data_dir,
            dataset=args.darpa_name,
            max_records=args.max_records,
        )
        print(f"  Loaded {len(df):,} events, {labels['is_compromised'].sum()} compromised hosts")
        return df, labels

    # ── File-based input ──────────────────────────────────────────────
    input_path = args.input
    if not input_path:
        print(f"\n  {_red('No input specified.')}")
        print(f"  Use --input, --synthetic, or --dataset darpa to provide data.")
        print(f"  Run with --help for usage examples.")
        sys.exit(1)

    path = Path(input_path)
    if not path.exists():
        print(f"  {_red(f'File not found: {path}')}")
        sys.exit(1)

    _banner("Loading Data")
    t0 = time.perf_counter()

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        print(f"  Loaded Parquet: {len(df):,} rows, {len(df.columns)} columns")
    elif path.suffix in (".jsonl", ".json", ".gz"):
        if str(path).endswith(".json.gz") or str(path).endswith(".jsonl.gz"):
            df = pd.read_json(path, lines=True, compression="gzip")
        else:
            df = pd.read_json(path, lines=True)
        print(f"  Loaded JSONL: {len(df):,} rows, {len(df.columns)} columns")
    else:
        # CSV (or anything else)
        kwargs = {"low_memory": False}
        if args.max_records:
            kwargs["nrows"] = args.max_records
        df = pd.read_csv(path, **kwargs)
        print(f"  Loaded CSV: {len(df):,} rows, {len(df.columns)} columns")

    elapsed = time.perf_counter() - t0
    print(f"  Load time: {elapsed:.1f}s")

    # ── Load labels if provided ───────────────────────────────────────
    if args.labels:
        labels_path = Path(args.labels)
        if labels_path.exists():
            labels = pd.read_csv(labels_path)
            n_pos = int(labels["is_compromised"].sum()) if "is_compromised" in labels.columns else 0
            print(f"  Labels: {len(labels)} hosts, {n_pos} compromised")
        else:
            print(f"  {_yellow(f'Labels file not found: {labels_path} — continuing without ground truth')}")

    return df, labels


def _load_baseline(args: argparse.Namespace) -> "Optional[pd.DataFrame]":
    """Load an explicit baseline file if provided."""
    import pandas as pd

    if not args.baseline:
        return None

    path = Path(args.baseline)
    if not path.exists():
        print(f"  {_red(f'Baseline file not found: {path}')}")
        sys.exit(1)

    print(f"  Loading baseline: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path, low_memory=False)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def _save_model(fitted: "FittedArtifacts", path: str) -> None:
    """Serialize FittedArtifacts to a pickle file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(fitted, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  {_green(f'Model saved: {out} ({size_mb:.1f} MB)')}")


def _load_model(path: str) -> "FittedArtifacts":
    """Deserialize FittedArtifacts from a pickle file."""
    p = Path(path)
    if not p.exists():
        print(f"  {_red(f'Model file not found: {p}')}")
        sys.exit(1)
    print(f"  Loading fitted model: {p}")
    with open(p, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def _print_triage_funnel(art: "StrataArtifacts") -> None:
    """Print the triage funnel — how many hosts passed each stage."""
    import pandas as pd

    n_events = len(art.events)
    n_hosts  = art.events["host"].nunique() if "host" in art.events.columns else 0

    # Channel-level counts (hosts above 75th percentile per channel)
    def _count_above_p75(scores_df, col):
        if scores_df is None or scores_df.empty or col not in scores_df.columns:
            return 0
        s = scores_df[col].dropna()
        if s.empty or s.max() == 0:
            return 0
        return int((s > s.quantile(0.75)).sum())

    n_seq   = 0
    n_freq  = 0
    n_ctx   = 0
    n_drift = 0

    # Sequence channel: prefer p-value if calibrated, else percentile
    if art.seq_scores is not None and not art.seq_scores.empty:
        if "S_seq_pvalue" in art.seq_scores.columns:
            n_seq = int((art.seq_scores["S_seq_pvalue"].dropna() < 0.05).sum())
        else:
            n_seq = _count_above_p75(art.seq_scores, "S_seq")

    n_freq  = _count_above_p75(art.freq_scores, "S_freq")
    n_ctx   = _count_above_p75(art.ctx_scores, "S_ctx")
    n_drift = _count_above_p75(art.drift_scores, "S_drift")

    # Gate pass count
    n_gate_pass = 0
    if art.triage is not None and not art.triage.empty:
        if "gate_pass" in art.triage.columns:
            n_gate_pass = int(art.triage["gate_pass"].sum())
        elif "corroborated" in art.triage.columns:
            n_gate_pass = int(art.triage["corroborated"].sum())

    print(f"\n  {'Stage':<45} {'Count':>10}")
    print(f"  {'—'*45} {'—'*10}")
    print(f"  {'Events in scoring window':<45} {n_events:>10,}")
    print(f"  {'Unique hosts scored':<45} {n_hosts:>10,}")
    print(f"  {'Sequence channel (>p75 or p<0.05)':<45} {n_seq:>10,}")
    print(f"  {'Frequency channel (>p75)':<45} {n_freq:>10,}")
    print(f"  {'Context channel (>p75)':<45} {n_ctx:>10,}")
    print(f"  {'Drift channel (>p75)':<45} {n_drift:>10,}")
    print(f"  {_bold('Gate pass (corroborated)'):<54} {_bold(str(n_gate_pass)):>10}")


def _print_triage_table(art: "StrataArtifacts", top_k: int = 20) -> None:
    """Print the top-K triage results."""
    import pandas as pd

    if art.triage is None or art.triage.empty:
        print(f"\n  {_yellow('No triage results available.')}")
        return

    triage = art.triage.head(top_k).copy()

    _banner(f"Top-{top_k} Triage Results")

    # Pick display columns based on what's available
    display_cols = ["host"]
    score_cols = ["score", "fused_score", "S_seq", "S_freq", "S_ctx", "S_drift"]
    for c in score_cols:
        if c in triage.columns:
            display_cols.append(c)

    gate_cols = ["gate_pass", "gate_reason", "corroborated", "n_channels_above"]
    for c in gate_cols:
        if c in triage.columns:
            display_cols.append(c)

    extra_cols = ["evasion_signal", "triage_rank", "S_seq_pvalue"]
    for c in extra_cols:
        if c in triage.columns:
            display_cols.append(c)

    display = triage[[c for c in display_cols if c in triage.columns]].copy()

    # Format numeric columns
    for c in display.columns:
        if display[c].dtype in ("float64", "float32"):
            display[c] = display[c].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "—")

    print(f"\n{display.to_string(index=False)}")


def _print_ground_truth(
    art: "StrataArtifacts",
    labels: "pd.DataFrame",
) -> None:
    """Evaluate triage against ground truth labels and print results."""
    import pandas as pd

    if art.triage is None or art.triage.empty:
        return
    if labels is None or labels.empty:
        return

    _banner("Ground-Truth Evaluation")

    triage = art.triage.copy()

    # Determine which hosts are flagged (gate_pass or corroborated)
    if "gate_pass" in triage.columns:
        flagged_hosts = set(triage[triage["gate_pass"] == True]["host"])
    elif "corroborated" in triage.columns:
        flagged_hosts = set(triage[triage["corroborated"] == True]["host"])
    elif "n_channels_above" in triage.columns:
        flagged_hosts = set(triage[triage["n_channels_above"] >= 2]["host"])
    else:
        # Use top 25% by score as flagged
        score_col = "score" if "score" in triage.columns else "fused_score"
        if score_col in triage.columns:
            cutoff = triage[score_col].quantile(0.75)
            flagged_hosts = set(triage[triage[score_col] >= cutoff]["host"])
        else:
            print(f"  {_yellow('Cannot determine flagged hosts — no score column found')}")
            return

    # Merge with labels
    if "host" not in labels.columns or "is_compromised" not in labels.columns:
        print(f"  {_yellow('Labels must have columns: host, is_compromised')}")
        return

    compromised = set(labels[labels["is_compromised"] == True]["host"])
    benign      = set(labels[labels["is_compromised"] == False]["host"])
    all_hosts   = set(triage["host"])

    tp = len(flagged_hosts & compromised)
    fp = len(flagged_hosts & benign)
    fn = len(compromised - flagged_hosts)
    tn = len(benign - flagged_hosts)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Per-host breakdown
    print(f"\n  {'Host':<30} {'Truth':>12} {'Flagged':>10}")
    print(f"  {'—'*30} {'—'*12} {'—'*10}")
    for _, row in labels.iterrows():
        host = row["host"]
        if host not in all_hosts:
            continue
        truth = "COMPROMISED" if row["is_compromised"] else "benign"
        flagged = host in flagged_hosts
        flag_str = _green("YES ✓") if flagged and row["is_compromised"] else \
                   _red("FP ✗") if flagged and not row["is_compromised"] else \
                   _red("MISS ✗") if not flagged and row["is_compromised"] else \
                   _dim("—")
        print(f"  {str(host):<30} {truth:>12} {flag_str:>10}")

    # Summary metrics
    p_color = _green if precision >= 0.75 else (_yellow if precision >= 0.5 else _red)
    r_color = _green if recall >= 0.75 else (_yellow if recall >= 0.5 else _red)
    f_color = _green if f1 >= 0.60 else (_yellow if f1 >= 0.4 else _red)

    print(f"\n  TP: {tp}   FP: {fp}   FN: {fn}   TN: {tn}")
    print(f"\n  Precision : {p_color(f'{precision:.3f}')}")
    print(f"  Recall    : {r_color(f'{recall:.3f}')}")
    print(f"  F1        : {f_color(f'{f1:.3f}')}")
    print(f"  FPR       : {fpr:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import pandas as pd

    parser = build_parser()
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    if args.no_plots:
        import matplotlib
        matplotlib.use("Agg")

    from sysmon_pipeline.config import StrataConfig, AblationConfig
    from sysmon_pipeline.pipeline import StrataPipeline
    from sysmon_pipeline.schema import normalize_schema

    # ── Show defaults and exit ────────────────────────────────────────
    if args.show_defaults:
        cfg = StrataConfig()
        _show_config(cfg)
        return

    # ── Build config ──────────────────────────────────────────────────
    if args.config:
        print(f"  Loading config from {_cyan(args.config)}")
        cfg = StrataConfig.from_json(args.config)
    elif args.interactive:
        cfg = _interactive_config()
    else:
        cfg = StrataConfig()

    # Apply ablation preset
    if args.ablation != "full":
        ablation_map = {
            "full":              AblationConfig.full_pipeline,
            "sequence_only":     AblationConfig.sequence_only,
            "no_shrinkage":      AblationConfig.no_shrinkage,
            "no_role_baselining": AblationConfig.no_role_baselining,
            "no_calibration":    AblationConfig.no_calibration,
            "no_drift":          AblationConfig.no_drift,
        }
        cfg.ablation = ablation_map[args.ablation]()
        print(f"  Ablation preset: {_cyan(args.ablation)}")

    # Apply --fast preset (reduces bootstrap, IF trees, TF-IDF for speed)
    if args.fast:
        cfg.apply_fast_preset()
        print(f"  {_cyan('--fast mode')}: bootstrap={cfg.baseline.bootstrap_samples}, "
              f"IF trees={cfg.scoring.iforest_n_estimators}, "
              f"TF-IDF features={cfg.scoring.tfidf_max_features}")

    # Apply CLI overrides
    if args.override:
        cfg = _apply_overrides(cfg, args.override)

    # Set output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.io.output_dir = output_dir

    # Show final config
    if args.interactive or args.config:
        if _prompt_yes_no("Show final config?", default=False):
            _show_config(cfg)

    # Save config if requested
    if args.save_config:
        cfg.to_json(args.save_config)
        print(f"\n  {_green(f'Config saved to {args.save_config}')}")

    if args.dry_run:
        print(f"\n  {_yellow('Dry run — exiting without running pipeline.')}")
        return

    # ── Load data ─────────────────────────────────────────────────────

    t_start = time.perf_counter()
    df, labels = _load_data(args)
    baseline_explicit = _load_baseline(args)

    # Set input path on config
    if args.input:
        cfg.io.input_path = Path(args.input)

    # ── Resolve baseline vs scoring split ─────────────────────────────

    fitted = None
    baseline_df = None
    score_df = None
    use_fit_score = False

    if args.no_split:
        # No split — use fit_score on full dataset
        _banner("Using Full Dataset (--no-split)")
        score_df = df
        use_fit_score = True
        print(f"  Full dataset: {len(df):,} events (fit_score mode)")

    elif args.load_model:
        # Skip fitting entirely — load a pre-fitted model
        _banner("Loading Pre-Fitted Model")
        fitted = _load_model(args.load_model)
        score_df = df  # Entire input is the scoring window
        print(f"  Scoring window: {len(score_df):,} events")

    elif baseline_explicit is not None:
        # Explicit baseline file — input becomes scoring window
        baseline_df = baseline_explicit
        score_df = df
        print(f"\n  Baseline: {len(baseline_df):,} events (from --baseline)")
        print(f"  Scoring:  {len(score_df):,} events (from --input)")

    else:
        # Auto-split the input into baseline and scoring windows
        # First normalize schema so we have a 'ts' column for time-based splitting
        df_norm = normalize_schema(df, cfg)

        if args.split_ratio:
            # Fractional split (e.g., 80/20)
            _banner(f"Splitting Data ({args.split_ratio:.0%} / {1-args.split_ratio:.0%})")
            split_idx = int(len(df_norm) * args.split_ratio)
            baseline_df = df_norm.iloc[:split_idx].copy()
            score_df    = df_norm.iloc[split_idx:].copy()
            print(f"  Baseline: {len(baseline_df):,} events (first {args.split_ratio:.0%})")
            print(f"  Scoring:  {len(score_df):,} events (remaining {1-args.split_ratio:.0%})")
        else:
            # Calendar-based split (default: 7-day baseline, 1-day scoring)
            from sysmon_pipeline.loaders import split_time_windows
            _banner(f"Splitting Data ({args.baseline_days}d baseline / {args.score_days}d scoring)")
            try:
                baseline_df, score_df = split_time_windows(
                    df_norm,
                    ts_col="ts",
                    baseline_days=args.baseline_days,
                    score_days=args.score_days,
                )
                print(f"  Baseline: {len(baseline_df):,} events")
                print(f"  Scoring:  {len(score_df):,} events")
            except ValueError as e:
                log.warning("Calendar split failed (%s) — falling back to fit_score (no split)", e)
                score_df = df
                use_fit_score = True
                print(f"  {_yellow(f'Calendar split failed: {e}')}")
                print(f"  Falling back to fit_score on full dataset ({len(df):,} events)")

    # ── Fit (if needed) ───────────────────────────────────────────────

    fit_elapsed = 0.0
    if use_fit_score:
        # fit_score on full dataset
        _banner("Fitting & Scoring (fit_score)")
        print(f"  Full dataset: {len(score_df):,} events")
        print(f"  Output:         {output_dir.resolve()}")
        print()

        t_fit = time.perf_counter()
        pipe = StrataPipeline(cfg)

        if args.report:
            from sysmon_pipeline.report import ReportContext
            report_dir = output_dir / "report"
            report_dir.mkdir(exist_ok=True)

            with ReportContext(
                output_dir=report_dir,
                open_browser=args.browser,
            ) as report:
                art = pipe.fit_score(score_df)
                fitted = pipe._last_fitted if hasattr(pipe, '_last_fitted') else None
                report_path = report.finalise(art)
            print(f"\n  HTML report: {report_path.resolve()}")
        else:
            art = pipe.fit_score(score_df)
            fitted = pipe._last_fitted if hasattr(pipe, '_last_fitted') else None

        fit_elapsed = time.perf_counter() - t_fit
        score_elapsed = fit_elapsed  # combined
        print(f"\n  fit_score completed in {fit_elapsed:.1f}s")

    elif fitted is None:
        _banner("Fitting Baseline Model")
        t_fit = time.perf_counter()
        pipe = StrataPipeline(cfg)
        fitted = pipe.fit(baseline_df)
        fit_elapsed = time.perf_counter() - t_fit
        print(f"  Fit completed in {fit_elapsed:.1f}s")

        n_roles = fitted.host_roles["role_id"].nunique() if fitted.host_roles is not None else 0
        n_baselines = len(fitted.peer_baselines) if fitted.peer_baselines else 0
        print(f"  Roles inferred: {n_roles}")
        print(f"  Peer baselines: {n_baselines}")

        # Save model if requested
        if args.save_model:
            _save_model(fitted, args.save_model)
    else:
        pipe = StrataPipeline(fitted.cfg)

    # ── Score ─────────────────────────────────────────────────────────

    if not use_fit_score:
        _banner("Scoring")
        print(f"  Scoring window: {len(score_df):,} events")
        print(f"  Output:         {output_dir.resolve()}")
        print()

        t_score = time.perf_counter()

        if args.report:
            from sysmon_pipeline.report import ReportContext
            report_dir = output_dir / "report"
            report_dir.mkdir(exist_ok=True)

            with ReportContext(
                output_dir=report_dir,
                open_browser=args.browser,
            ) as report:
                art = pipe.score(score_df, fitted)
                report_path = report.finalise(art)
            print(f"\n  HTML report: {report_path.resolve()}")
        else:
            art = pipe.score(score_df, fitted)

        score_elapsed = time.perf_counter() - t_score
        print(f"\n  Score completed in {score_elapsed:.1f}s")

    # Print per-stage timing breakdown if available
    if art.stage_timings:
        total_staged = sum(art.stage_timings.values())
        print(f"\n  {'Stage':<28} {'Time':>8}  {'Share':>6}")
        print(f"  {'—'*28} {'—'*8}  {'—'*6}")
        for stage, elapsed in sorted(art.stage_timings.items(), key=lambda x: -x[1]):
            pct = 100.0 * elapsed / (total_staged + 1e-9)
            bar = "█" * max(1, int(pct / 5))
            print(f"  {stage:<28} {elapsed:>7.2f}s  {pct:>5.1f}% {bar}")

    # ── Save triage CSV ───────────────────────────────────────────────

    if art.triage is not None and not art.triage.empty:
        triage_path = output_dir / "triage.csv"
        art.triage.to_csv(triage_path, index=False)
        print(f"  Triage CSV: {triage_path}")

    # ── Print results ─────────────────────────────────────────────────

    _banner("Results")
    _print_triage_funnel(art)
    _print_triage_table(art, top_k=args.top_k)

    # ── Ground truth evaluation ───────────────────────────────────────

    if labels is not None:
        _print_ground_truth(art, labels)

    # ── Post-hoc analysis (Nomad-style rigor) ─────────────────────────

    from sysmon_pipeline.analysis import (
        compute_shap_importance, plot_shap_importance,
        compute_deployment_scenario, format_deployment_scenario,
        format_timing, PipelineTiming,
        analyze_errors, format_error_cases,
        get_channel_taxonomy, format_channel_taxonomy,
    )

    # --- SHAP Feature Importance ---
    if art.rate_features is not None and not art.rate_features.empty:
        _banner("Feature Importance (SHAP — Frequency Channel)")
        importance_df = compute_shap_importance(fitted, art.rate_features)
        if importance_df is not None:
            print(f"\n  Top features driving the frequency channel anomaly scores:\n")
            top = importance_df.head(10)
            for _, row in top.iterrows():
                bar_len = int(row["mean_abs_shap"] / top["mean_abs_shap"].max() * 30)
                bar = "█" * bar_len
                print(f"  {row['rank']:>3}. {row['feature']:<30} {bar} {row['mean_abs_shap']:.4f}")

            # Save CSV
            shap_path = output_dir / "shap_importance.csv"
            importance_df.to_csv(shap_path, index=False)
            print(f"\n  SHAP importance CSV: {shap_path}")

            # Plot if not suppressed
            if not args.no_plots:
                plot_path = output_dir / "shap_importance.png"
                try:
                    plot_shap_importance(importance_df, output_path=plot_path)
                except Exception:
                    pass  # Non-critical — skip if matplotlib fails
        else:
            print(f"  {_yellow('SHAP analysis skipped (install shap: pip install shap)')}")

    # --- Deployment-Prevalence Math ---
    _banner("Deployment Scenario (SOC Triage Projection)")
    scenario = compute_deployment_scenario(
        art, labels=labels,
        fleet_size=10_000,
        attack_prevalence=0.001,
        analyst_capacity=100,
    )
    print(f"\n  {format_deployment_scenario(scenario)}")

    # Save scenario to JSON
    import json
    scenario_path = output_dir / "deployment_scenario.json"
    scenario_dict = {
        "fleet_size": scenario.fleet_size,
        "attack_prevalence": scenario.attack_prevalence,
        "expected_compromised": scenario.expected_compromised,
        "fpr": scenario.fpr,
        "recall": scenario.recall,
        "expected_true_positives": scenario.expected_true_positives,
        "expected_false_positives": scenario.expected_false_positives,
        "expected_alerts_per_run": scenario.expected_alerts_per_run,
        "deployment_ppv": scenario.deployment_ppv,
        "detection_lift_vs_random": scenario.detection_lift_vs_random,
    }
    with open(scenario_path, "w") as f:
        json.dump(scenario_dict, f, indent=2)
    print(f"\n  Deployment scenario JSON: {scenario_path}")

    # --- Latency Benchmarking ---
    _banner("Latency & Throughput")
    timing = PipelineTiming(
        fit_seconds=fit_elapsed,
        score_seconds=score_elapsed,
        total_seconds=time.perf_counter() - t_start,
        n_hosts_scored=art.events["host"].nunique() if "host" in art.events.columns else 0,
        n_events_scored=len(art.events),
    )
    print(f"\n  {format_timing(timing)}")

    # Save timing to JSON
    timing_path = output_dir / "timing.json"
    timing_dict = {
        "fit_seconds": timing.fit_seconds,
        "score_seconds": timing.score_seconds,
        "total_seconds": timing.total_seconds,
        "n_hosts_scored": timing.n_hosts_scored,
        "n_events_scored": timing.n_events_scored,
        "hosts_per_second": timing.hosts_per_second,
        "events_per_second": timing.events_per_second,
        "ms_per_host": timing.ms_per_host,
    }
    with open(timing_path, "w") as f:
        json.dump(timing_dict, f, indent=2)
    print(f"\n  Timing JSON: {timing_path}")

    # --- Qualitative Error Analysis ---
    if labels is not None:
        _banner("Error Analysis (FP/FN Case Studies)")
        error_cases = analyze_errors(art, labels, top_k=3)
        if error_cases:
            print(f"\n{format_error_cases(error_cases)}")

            # Save to CSV
            error_rows = []
            for c in error_cases:
                row = {
                    "host": c.host, "case_type": c.case_type,
                    "score": c.score, "gate_pass": c.gate_pass,
                    "explanation": c.explanation, "n_events": c.n_events,
                }
                row.update(c.channel_scores)
                error_rows.append(row)
            error_df = pd.DataFrame(error_rows)
            error_path = output_dir / "error_analysis.csv"
            error_df.to_csv(error_path, index=False)
            print(f"  Error analysis CSV: {error_path}")
        else:
            print(f"  {_green('No false positives or false negatives to analyze.')}")

    # --- Channel Taxonomy Table ---
    taxonomy = get_channel_taxonomy()
    taxonomy_path = output_dir / "channel_taxonomy.csv"
    taxonomy.to_csv(taxonomy_path, index=False)

    # ── Extended Evaluation (--eval / --eval-full) ────────────────────

    if getattr(args, 'eval', False) or getattr(args, 'eval_full', False):
        _banner("Extended Evaluation")
        print("  Running eval_extended analysis...")

        import subprocess
        import sys

        eval_args = [
            sys.executable, "eval_extended.py",
            "--input", str(args.input),
            "--output", str(output_dir),
        ]

        if not getattr(args, 'eval_full', False):
            eval_args.extend(["--skip-ablation", "--skip-sensitivity", "--skip-scalability"])

        try:
            result = subprocess.run(eval_args, check=True)
            print(f"  Extended evaluation complete — results in {output_dir}/")
        except FileNotFoundError:
            print(f"  {_yellow('eval_extended.py not found — run it separately:')}")
            print(f"    python eval_extended.py --input {args.input} --output {output_dir}")
        except subprocess.CalledProcessError as e:
            print(f"  {_yellow('eval_extended.py failed (exit code ' + str(e.returncode) + ')')}")
            print(f"    Run it separately to see errors:")
            print(f"    python eval_extended.py --input {args.input} --output {output_dir}")

    # ── Done ──────────────────────────────────────────────────────────

    total = time.perf_counter() - t_start
    _banner(f"Done — total wall time {total:.1f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
