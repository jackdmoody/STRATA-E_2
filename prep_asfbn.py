#!/usr/bin/env python3
"""
prep_asfbn.py
==============
Preprocessing script for ASFBN Sysmon/Security parquet data.

Cleans known data quality issues before feeding into STRATA-E:
  1. Converts "UNKNOWN" and "-" placeholder strings to proper NaN
  2. Drops Puppet configuration management noise events
  3. Drops the "unknown" host (unparsed events)
  4. Optionally filters to a specific host list
  5. Saves cleaned data as parquet or runs STRATA-E directly

Usage
------
    # Clean and save to parquet (inspect before running pipeline)
    python prep_asfbn.py --input first_sample_3.parquet --output cleaned.parquet

    # Clean and run STRATA-E immediately in fast mode
    python prep_asfbn.py --input first_sample_3.parquet --run --fast

    # Clean, filter to specific hosts, and run preprocess_check only
    python prep_asfbn.py --input first_sample_3.parquet --check

    # Clean and run full pipeline with HTML report
    python prep_asfbn.py --input first_sample_3.parquet --run --report
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prep_asfbn")


# ── Placeholder values to convert to NaN ─────────────────────────────────
# These appear in the parquet as literal strings but represent missing data.
# If STRATA-E sees "UNKNOWN" as a process name, it tokenizes it as
# PROC:UNKNOWN and every host looks identical (sequence channel collapses).
PLACEHOLDER_VALUES = {"UNKNOWN", "-", "unknown", "", "N/A", "n/a", "(null)"}

# Columns to clean — only string columns that feed tokenization/scoring
COLUMNS_TO_CLEAN = [
    "Image",
    "ParentImage",
    "CommandLine",
    "ParentCommandLine",
    "IntegrityLevel",
]

# Event providers that are pure noise (config management, not user/attacker)
NOISE_PROVIDERS = {"Puppet"}

# Host values to drop (unparsed/broken events)
DROP_HOSTS = {"unknown", "UNKNOWN", ""}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps. Returns a new DataFrame.

    Steps:
      1. Replace placeholder strings with NaN in key columns
      2. Drop noise event providers (Puppet)
      3. Drop events from the "unknown" host
      4. Log a summary of what was removed
    """
    n_start = len(df)
    log.info("Starting with %d events", n_start)

    # ── Step 1: Placeholders → NaN ────────────────────────────────────
    for col in COLUMNS_TO_CLEAN:
        if col not in df.columns:
            continue
        mask = df[col].isin(PLACEHOLDER_VALUES)
        n_replaced = mask.sum()
        if n_replaced > 0:
            df.loc[mask, col] = pd.NA
            pct = 100 * n_replaced / len(df)
            log.info("  %s: %d placeholder values → NaN (%.1f%%)",
                     col, n_replaced, pct)

    # ── Step 2: Drop noise providers ──────────────────────────────────
    if "event_provider" in df.columns:
        noise_mask = df["event_provider"].isin(NOISE_PROVIDERS)
        n_noise = noise_mask.sum()
        if n_noise > 0:
            df = df[~noise_mask].copy()
            log.info("  Dropped %d events from noise providers %s",
                     n_noise, NOISE_PROVIDERS)

    # ── Step 3: Drop unknown hosts ────────────────────────────────────
    if "host" in df.columns:
        bad_host_mask = df["host"].isin(DROP_HOSTS) | df["host"].isna()
        n_bad = bad_host_mask.sum()
        if n_bad > 0:
            df = df[~bad_host_mask].copy()
            log.info("  Dropped %d events from unknown/empty hosts", n_bad)

    n_end = len(df)
    log.info("Cleaning done: %d → %d events (%d removed, %.1f%%)",
             n_start, n_end, n_start - n_end,
             100 * (n_start - n_end) / n_start)

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print a quick data summary after cleaning."""
    print(f"\n  Cleaned Data Summary")
    print(f"  {'='*50}")
    print(f"  Events:       {len(df):,}")
    print(f"  Hosts:        {df['host'].nunique()}")
    print(f"  Time range:   {df['ts'].min()} → {df['ts'].max()}")
    span_h = (df["ts"].max() - df["ts"].min()).total_seconds() / 3600
    print(f"  Time span:    {span_h:.1f} hours ({span_h/24:.1f} days)")

    print(f"\n  Events per host:")
    hc = df.groupby("host").size()
    print(f"    min={hc.min():,}  median={int(hc.median()):,}  "
          f"max={hc.max():,}  mean={hc.mean():,.0f}")

    print(f"\n  Event ID distribution:")
    for eid, cnt in df["EventID"].value_counts().sort_index().items():
        print(f"    {eid:>5}: {cnt:>8,}  ({100*cnt/len(df):.1f}%)")

    print(f"\n  Null rates after cleaning:")
    for col in ["Image", "ParentImage", "CommandLine", "IntegrityLevel"]:
        if col in df.columns:
            null_rate = df[col].isna().sum() / len(df)
            print(f"    {col:<20} {100*null_rate:.1f}% null")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Clean ASFBN parquet data for STRATA-E"
    )
    parser.add_argument("--input", required=True,
                        help="Path to raw parquet file")
    parser.add_argument("--output", default=None,
                        help="Save cleaned data to this path (parquet or csv)")
    parser.add_argument("--check", action="store_true",
                        help="Run STRATA-E preprocess_check on cleaned data")
    parser.add_argument("--run", action="store_true",
                        help="Run full STRATA-E pipeline on cleaned data")
    parser.add_argument("--fast", action="store_true",
                        help="Use --fast preset (3-10× speedup)")
    parser.add_argument("--report", action="store_true",
                        help="Generate HTML report (requires --run)")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for pipeline results")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap number of rows for testing")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    log.info("Loading %s", args.input)
    df = pd.read_parquet(args.input)
    if args.max_rows:
        df = df.head(args.max_rows)
        log.info("Capped to %d rows", args.max_rows)
    log.info("Loaded %d rows in %.1fs", len(df), time.perf_counter() - t0)

    # ── Clean ─────────────────────────────────────────────────────────
    df = clean(df)
    print_summary(df)

    # ── Save ──────────────────────────────────────────────────────────
    if args.output:
        out = Path(args.output)
        if out.suffix in (".parquet", ".pq"):
            df.to_parquet(out, index=False)
        else:
            df.to_csv(out, index=False)
        log.info("Saved cleaned data to %s", out)

    # ── Preprocess check ──────────────────────────────────────────────
    if args.check:
        from sysmon_pipeline.debug import StrataDebugPipeline
        from sysmon_pipeline import StrataConfig

        cfg = StrataConfig.fast() if args.fast else StrataConfig()
        dbg = StrataDebugPipeline(cfg)
        dbg.preprocess_check(df)

    # ── Full run ──────────────────────────────────────────────────────
    if args.run:
        from sysmon_pipeline import StrataPipeline, StrataConfig

        cfg = StrataConfig.fast() if args.fast else StrataConfig()
        pipe = StrataPipeline(cfg)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("Running STRATA-E pipeline...")
        t_run = time.perf_counter()

        if args.report:
            from sysmon_pipeline.report import ReportContext
            report_dir = out_dir / "report"
            report_dir.mkdir(exist_ok=True)
            with ReportContext(output_dir=report_dir) as report:
                art = pipe.fit_score(df)
                report.finalise(art)
            log.info("HTML report: %s", report_dir)
        else:
            art = pipe.fit_score(df)

        elapsed = time.perf_counter() - t_run
        log.info("Pipeline completed in %.1fs", elapsed)

        # Save triage
        if art.triage is not None:
            triage_path = out_dir / "triage.csv"
            art.triage.to_csv(triage_path, index=False)
            log.info("Triage CSV: %s", triage_path)

            # Print top results
            print(f"\n  Top 20 Triage Results")
            print(f"  {'='*50}")
            cols = ["triage_rank", "host", "score", "gate_pass",
                    "S_seq", "S_freq", "S_ctx"]
            cols = [c for c in cols if c in art.triage.columns]
            print(art.triage[cols].head(20).to_string(index=False))

        # Print stage timings
        if art.stage_timings:
            print(f"\n  Stage Timings")
            print(f"  {'='*50}")
            total = sum(art.stage_timings.values())
            for stage, elapsed in sorted(art.stage_timings.items(),
                                         key=lambda x: -x[1]):
                pct = 100.0 * elapsed / (total + 1e-9)
                bar = "█" * max(1, int(pct / 3))
                print(f"  {stage:<28} {elapsed:>7.2f}s  {pct:>5.1f}% {bar}")

    total_time = time.perf_counter() - t0
    log.info("Total wall time: %.1fs", total_time)


if __name__ == "__main__":
    main()
