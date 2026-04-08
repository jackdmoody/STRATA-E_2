#!/usr/bin/env python3
"""
eval_extended.py
=================
Extended evaluation for STRATA-E poster and paper.

Runs on top of eval_asfbn.py results — produces additional analyses:
  1. Channel contribution breakdown (per-host Borda rank decomposition)
  2. Structural ablation comparison (all presets side by side)
  3. Deployment-prevalence projections (fleet scaling)
  4. Summary plots for all hypotheses

Usage:
    python eval_extended.py --input data/first_sample_3.parquet --output results_extended
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_extended")

# Plot style — default is dark, --light-plots switches to white
# Plot style — white background for publication-ready figures
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cccccc",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "grid.color": "#e0e0e0",
    "legend.facecolor": "#ffffff",
    "legend.edgecolor": "#cccccc",
    "font.size": 11,
})

CHANNEL_COLORS = {
    "S_seq": "#58a6ff",
    "S_freq": "#f0883e",
    "S_ctx": "#3fb950",
    "S_drift": "#bc8cff",
}


def clean(df):
    for col in ["Image", "ParentImage", "CommandLine",
                "ParentCommandLine", "IntegrityLevel"]:
        if col in df.columns:
            df[col] = df[col].replace({"UNKNOWN": pd.NA, "-": pd.NA})
    if "event_provider" in df.columns:
        df = df[df["event_provider"] != "Puppet"].copy()
    if "host" in df.columns:
        df = df[~df["host"].isin({"unknown", "UNKNOWN"})].copy()
    return df


def run_pipeline(df, **overrides):
    from sysmon_pipeline import StrataPipeline, StrataConfig
    cfg = StrataConfig.fast()
    for key, val in overrides.items():
        parts = key.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)
    pipe = StrataPipeline(cfg)
    return pipe.fit_score(df)


# ══════════════════════════════════════════════════════════
# 1. Channel Contribution Breakdown
# ══════════════════════════════════════════════════════════

def channel_contribution(art, out):
    """Per-host Borda rank decomposition across channels."""
    log.info("Computing channel contribution breakdown")
    triage = art.triage.copy()
    channels = ["S_seq", "S_freq", "S_ctx"]
    if "S_drift" in triage.columns and triage["S_drift"].sum() > 0:
        channels.append("S_drift")

    # Compute per-channel ranks
    from scipy.stats import rankdata
    for ch in channels:
        triage[f"rank_{ch}"] = rankdata(triage[ch].fillna(0), method="average")

    rank_cols = [f"rank_{ch}" for ch in channels]
    triage["rank_total"] = triage[rank_cols].sum(axis=1)

    # Save
    out_cols = ["host", "triage_rank", "score", "gate_pass"] + channels + rank_cols + ["rank_total"]
    out_cols = [c for c in out_cols if c in triage.columns]
    triage[out_cols].to_csv(out / "channel_contribution.csv", index=False)

    # Plot: stacked horizontal bar chart of rank contributions
    top_hosts = triage.sort_values("rank_total", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(top_hosts))
    left = np.zeros(len(top_hosts))

    for ch in channels:
        rcol = f"rank_{ch}"
        vals = top_hosts[rcol].values
        color = CHANNEL_COLORS.get(ch, "#888")
        ax.barh(y_pos, vals, left=left, height=0.7, label=ch, color=color, alpha=0.85)
        left += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_hosts["host"].values, fontsize=9, fontfamily="monospace")
    ax.set_xlabel("Borda Rank Contribution")
    ax.set_title("Channel Contribution to Triage Ranking (Top 15 Hosts)")
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out / "channel_contribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved channel_contribution.csv and .png")


# ══════════════════════════════════════════════════════════
# 2. Structural Ablation Comparison
# ══════════════════════════════════════════════════════════

def ablation_comparison(df, out):
    """Run all ablation presets and compare gate-pass counts, rankings, and timings."""
    log.info("Running structural ablation comparison")

    conditions = {
        "full": {},
        "no_shrinkage": {"ablation.use_dirichlet_shrinkage": False},
        "no_role_baselining": {"ablation.use_role_baselining": False},
        "no_calibration": {"ablation.use_jsd_calibration": False},
        "no_drift": {"ablation.use_drift_channel": False},
        "no_gate": {"ablation.use_corroboration_gate": False},
        "sequence_only": {
            "ablation.use_context_channel": False,
            "ablation.use_drift_channel": False,
            "ablation.use_seq_drift_covariance": False,
        },
    }

    results = []
    all_triages = {}
    all_timings = {}

    for name, overrides in conditions.items():
        log.info("  Running condition: %s", name)
        t0 = time.perf_counter()
        art = run_pipeline(df, **overrides)
        elapsed = time.perf_counter() - t0

        triage = art.triage
        gate_pass = int(triage["gate_pass"].sum()) if "gate_pass" in triage.columns else len(triage)

        # Top-5 hosts
        if "gate_pass" in triage.columns:
            top5 = triage[triage["gate_pass"]].head(5)["host"].tolist()
        else:
            top5 = triage.head(5)["host"].tolist()

        row_data = {
            "condition": name,
            "gate_pass": gate_pass,
            "total_hosts": len(triage),
            "top_1": top5[0] if len(top5) > 0 else "",
            "top_2": top5[1] if len(top5) > 1 else "",
            "top_3": top5[2] if len(top5) > 2 else "",
            "top_4": top5[3] if len(top5) > 3 else "",
            "top_5": top5[4] if len(top5) > 4 else "",
            "elapsed_s": round(elapsed, 1),
        }

        # Capture stage timings
        if art.stage_timings:
            all_timings[name] = art.stage_timings
            for stage, stime in art.stage_timings.items():
                row_data[f"t_{stage}"] = round(stime, 2)

        results.append(row_data)
        all_triages[name] = triage

    results_df = pd.DataFrame(results)
    results_df.to_csv(out / "ablation_comparison.csv", index=False)

    # Plot 1: gate-pass counts by condition
    fig, ax = plt.subplots(figsize=(10, 5))
    conditions_sorted = results_df.sort_values("gate_pass", ascending=True)
    colors = ["#3fb950" if c == "full" else "#58a6ff" for c in conditions_sorted["condition"]]
    ax.barh(conditions_sorted["condition"], conditions_sorted["gate_pass"],
            color=colors, alpha=0.85, height=0.6)
    ax.set_xlabel("Hosts Passing Corroboration Gate")
    ax.set_title("Ablation Study: Gate Throughput by Condition")
    for i, (_, row) in enumerate(conditions_sorted.iterrows()):
        ax.text(row["gate_pass"] + 0.3, i, str(row["gate_pass"]),
                va="center", fontsize=10, color="#e6edf3")
    plt.tight_layout()
    fig.savefig(out / "ablation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: stage timing comparison across conditions
    if all_timings:
        timing_df = pd.DataFrame(all_timings).T
        timing_df.index.name = "condition"

        # Save timing comparison
        timing_df.to_csv(out / "stage_timing_comparison.csv")

        # Stacked bar chart of stage timings
        stages_to_plot = ["preprocessing", "sequence_divergence", "context_channel",
                          "pair_correlation", "jsd_calibration", "frequency_channel"]
        stages_to_plot = [s for s in stages_to_plot if s in timing_df.columns]

        stage_colors = {
            "preprocessing": "#8b949e",
            "sequence_divergence": "#58a6ff",
            "context_channel": "#3fb950",
            "pair_correlation": "#f0883e",
            "jsd_calibration": "#bc8cff",
            "frequency_channel": "#f85149",
        }

        fig, ax = plt.subplots(figsize=(12, 6))
        bottom = np.zeros(len(timing_df))
        for stage in stages_to_plot:
            vals = timing_df[stage].fillna(0).values
            color = stage_colors.get(stage, "#888")
            ax.barh(timing_df.index, vals, left=bottom, height=0.6,
                    label=stage.replace("_", " ").title(), color=color, alpha=0.85)
            bottom += vals

        ax.set_xlabel("Time (seconds)")
        ax.set_title("Stage-Level Timing Breakdown by Ablation Condition")
        ax.legend(loc="lower right", fontsize=8)
        ax.invert_yaxis()

        # Add total time labels
        for i, (cond, total) in enumerate(zip(timing_df.index, bottom)):
            ax.text(total + 1, i, f"{total:.0f}s", va="center", fontsize=9, color="#e6edf3")

        plt.tight_layout()
        fig.savefig(out / "stage_timing_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Log the key finding: shrinkage speeds things up
        if "full" in all_timings and "no_shrinkage" in all_timings:
            full_total = sum(all_timings["full"].values())
            noshrink_total = sum(all_timings["no_shrinkage"].values())
            full_seq = all_timings["full"].get("sequence_divergence", 0)
            noshrink_seq = all_timings["no_shrinkage"].get("sequence_divergence", 0)
            log.info("  Timing insight: full pipeline %.0fs vs no_shrinkage %.0fs", full_total, noshrink_total)
            log.info("    sequence_divergence: %.1fs (shrinkage) vs %.1fs (MLE) — shrinkage is %.1fx faster",
                     full_seq, noshrink_seq, noshrink_seq / max(full_seq, 0.01))

    log.info("  Saved ablation_comparison.csv, .png, stage_timing_comparison.csv, .png")

    return all_triages


# ══════════════════════════════════════════════════════════
# 3. Deployment-Prevalence Projections
# ══════════════════════════════════════════════════════════

def deployment_projections(art, out):
    """Project pipeline performance to fleet-scale SOC scenarios."""
    log.info("Computing deployment-prevalence projections")

    triage = art.triage
    n_hosts = len(triage)
    n_gate_pass = int(triage["gate_pass"].sum()) if "gate_pass" in triage.columns else n_hosts
    observed_fpr = n_gate_pass / n_hosts  # fraction of hosts flagged (upper bound on FPR)

    fleet_sizes = [100, 500, 1000, 5000, 10000, 50000]
    prevalences = [0.01, 0.005, 0.001]  # 1%, 0.5%, 0.1% compromised

    rows = []
    for fleet in fleet_sizes:
        for prev in prevalences:
            n_compromised = max(1, int(fleet * prev))
            n_benign = fleet - n_compromised

            # Assume observed recall = 1.0 (conservative: we don't have ground truth
            # to measure actual recall, so we project with perfect recall)
            recall = 1.0
            true_positives = int(n_compromised * recall)

            # FPR projection using observed flagging rate
            false_positives = int(n_benign * observed_fpr)
            total_alerts = true_positives + false_positives

            if total_alerts > 0:
                ppv = true_positives / total_alerts
            else:
                ppv = 0.0

            rows.append({
                "fleet_size": fleet,
                "prevalence": prev,
                "expected_compromised": n_compromised,
                "expected_benign": n_benign,
                "projected_true_positives": true_positives,
                "projected_false_positives": false_positives,
                "projected_alerts": total_alerts,
                "projected_ppv": round(ppv, 4),
                "analyst_hours_at_30min_each": round(total_alerts * 0.5, 1),
            })

    proj_df = pd.DataFrame(rows)
    proj_df.to_csv(out / "deployment_projections.csv", index=False)

    # Plot: alerts per day by fleet size at 0.1% prevalence
    low_prev = proj_df[proj_df["prevalence"] == 0.001]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: total alerts
    ax1.plot(low_prev["fleet_size"], low_prev["projected_alerts"],
             "o-", color="#f0883e", linewidth=2, markersize=8)
    ax1.set_xlabel("Fleet Size (hosts)")
    ax1.set_ylabel("Projected Daily Alerts")
    ax1.set_title("Alert Volume at 0.1% Prevalence")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)
    for _, row in low_prev.iterrows():
        ax1.annotate(str(int(row["projected_alerts"])),
                     (row["fleet_size"], row["projected_alerts"]),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=9, color="#e6edf3")

    # Right: PPV by prevalence
    for prev in prevalences:
        subset = proj_df[proj_df["prevalence"] == prev]
        label = f"{prev*100:.1f}% prevalence"
        ax2.plot(subset["fleet_size"], subset["projected_ppv"],
                 "o-", linewidth=2, markersize=6, label=label)
    ax2.set_xlabel("Fleet Size (hosts)")
    ax2.set_ylabel("Projected Positive Predictive Value")
    ax2.set_title("Triage Precision by Fleet Size & Prevalence")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out / "deployment_projections.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved deployment_projections.csv and .png")


# ══════════════════════════════════════════════════════════
# 4. Hypothesis Summary Plots
# ══════════════════════════════════════════════════════════

def h2_plot(art_role, art_global, out):
    """H2: side-by-side S_seq comparison."""
    log.info("Generating H2 plot")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Global
    g = art_global.triage.sort_values("S_seq", ascending=False).head(15)
    ax1.barh(range(len(g)), g["S_seq"].values, color="#f85149", alpha=0.8, height=0.7)
    ax1.set_yticks(range(len(g)))
    ax1.set_yticklabels(g["host"].values, fontsize=8, fontfamily="monospace")
    ax1.set_xlabel("S_seq (Sequence Divergence)")
    ax1.set_title("Global Baseline", color="#f85149")
    ax1.invert_yaxis()

    # Role
    r = art_role.triage.sort_values("S_seq", ascending=False).head(15)
    ax2.barh(range(len(r)), r["S_seq"].values, color="#3fb950", alpha=0.8, height=0.7)
    ax2.set_yticks(range(len(r)))
    ax2.set_yticklabels(r["host"].values, fontsize=8, fontfamily="monospace")
    ax2.set_xlabel("S_seq (Sequence Divergence)")
    ax2.set_title("Role Baseline", color="#3fb950")
    ax2.invert_yaxis()

    fig.suptitle("H2: Role Baselining Activates the Sequence Channel",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "h2_seq_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved h2_seq_comparison.png")


def h5_plot(art_gated, art_ungated, out):
    """H5: gated vs ungated top-K overlap."""
    log.info("Generating H5 plot")

    ks = list(range(1, 21))
    overlaps = []
    for k in ks:
        gated_set = set(art_gated.triage[art_gated.triage["gate_pass"]].head(k)["host"])
        ungated_set = set(art_ungated.triage.head(k)["host"])
        if len(gated_set) == 0:
            overlaps.append(0)
        else:
            overlaps.append(len(gated_set & ungated_set) / len(gated_set))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, overlaps, "o-", color="#39d2c0", linewidth=2, markersize=6)
    ax.axhline(y=1.0, color="#30363d", linestyle="--", alpha=0.5)
    ax.set_xlabel("K (Top-K Triage Leads)")
    ax.set_ylabel("Overlap Fraction (Gated ∩ Ungated) / Gated")
    ax.set_title("H5: Corroboration Gate Effect on Triage Ranking")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    n_pass = int(art_gated.triage["gate_pass"].sum())
    ax.axvline(x=n_pass, color="#3fb950", linestyle=":", alpha=0.7)
    ax.text(n_pass + 0.3, 0.1, f"Gate passes {n_pass} hosts",
            fontsize=9, color="#3fb950")

    plt.tight_layout()
    fig.savefig(out / "h5_gate_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved h5_gate_overlap.png")


def h4_plot(art, out):
    """H4: channel independence radar/overlap."""
    log.info("Generating H4 plot")

    triage = art.triage
    channels = ["S_seq", "S_freq", "S_ctx"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ks = [3, 5, 10, 15, 20]
    fused_top = {k: set(triage.head(k)["host"]) for k in ks}

    for ch in channels:
        overlaps = []
        for k in ks:
            ch_top = set(triage.sort_values(ch, ascending=False).head(k)["host"])
            overlap = len(fused_top[k] & ch_top) / k if k > 0 else 0
            overlaps.append(overlap)
        ax.plot(ks, overlaps, "o-", color=CHANNEL_COLORS[ch], linewidth=2,
                markersize=8, label=ch)

    ax.set_xlabel("K (Top-K)")
    ax.set_ylabel("Overlap with Fused Ranking")
    ax.set_title("H4: Channel Independence — Each Channel Surfaces Different Hosts")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / "h4_channel_independence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved h4_channel_independence.png")


# ══════════════════════════════════════════════════════════
# 5. Scalability & Latency Benchmarking
# ══════════════════════════════════════════════════════════

def scalability_timing(df, out):
    """Time the pipeline on progressively larger data subsets."""
    log.info("Running scalability benchmarks")
    from sysmon_pipeline import StrataPipeline, StrataConfig

    fractions = [0.10, 0.25, 0.50, 0.75, 1.0]
    rows = []

    for frac in fractions:
        if frac < 1.0:
            sub = df.sample(frac=frac, random_state=42)
        else:
            sub = df

        cfg = StrataConfig.fast()
        pipe = StrataPipeline(cfg)

        t_fit = time.perf_counter()
        fitted = pipe.fit(sub)
        fit_time = time.perf_counter() - t_fit

        t_score = time.perf_counter()
        art = pipe.score(sub, fitted)
        score_time = time.perf_counter() - t_score

        total = fit_time + score_time
        n_events = len(sub)
        n_hosts = sub["host"].nunique() if "host" in sub.columns else 0

        rows.append({
            "fraction": frac,
            "n_events": n_events,
            "n_hosts": n_hosts,
            "fit_time_s": round(fit_time, 2),
            "score_time_s": round(score_time, 2),
            "total_time_s": round(total, 2),
            "events_per_sec": round(n_events / max(total, 0.01)),
        })
        log.info("  frac=%.2f  events=%d  hosts=%d  fit=%.1fs  score=%.1fs  total=%.1fs",
                 frac, n_events, n_hosts, fit_time, score_time, total)

    timing_df = pd.DataFrame(rows)
    timing_df.to_csv(out / "scalability_timing.csv", index=False)

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(timing_df["n_events"], timing_df["fit_time_s"],
             "^-", color="#58a6ff", linewidth=2, markersize=8, label="Fit time")
    ax1.plot(timing_df["n_events"], timing_df["score_time_s"],
             "v-", color="#f0883e", linewidth=2, markersize=8, label="Score time")
    ax1.plot(timing_df["n_events"], timing_df["total_time_s"],
             "o-", color="#3fb950", linewidth=2, markersize=8, label="Total")
    ax1.set_xlabel("Events")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("STRATA-E Runtime Scalability (Fast Mode)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(timing_df["n_events"], timing_df["events_per_sec"],
             "s--", color="#bc8cff", linewidth=1.5, markersize=6, label="Throughput")
    ax2.set_ylabel("Events/sec", color="#bc8cff")
    ax2.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(out / "scalability_timing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved scalability_timing.csv and .png")


# ══════════════════════════════════════════════════════════
# 6. Reproducibility Check
# ══════════════════════════════════════════════════════════

def reproducibility_check(df, out, n_replicates=3):
    """Run the pipeline N times and verify output hashes are identical."""
    log.info("Running reproducibility check (%d replicates)", n_replicates)
    import hashlib

    rows = []
    for i in range(n_replicates):
        art = run_pipeline(df)
        # Hash the triage table
        triage_sorted = art.triage.sort_values("host").reset_index(drop=True)
        cols = sorted([c for c in triage_sorted.columns if c != "triage_rank"])
        buf = triage_sorted[cols].to_csv(index=False)
        h = hashlib.sha256(buf.encode()).hexdigest()[:16]

        gate_pass = int(art.triage["gate_pass"].sum()) if "gate_pass" in art.triage.columns else 0
        rows.append({
            "replicate": i + 1,
            "output_hash": h,
            "gate_pass": gate_pass,
        })
        log.info("  Replicate %d: hash=%s  gate_pass=%d", i + 1, h, gate_pass)

    repro_df = pd.DataFrame(rows)
    repro_df.to_csv(out / "reproducibility_check.csv", index=False)

    all_identical = repro_df["output_hash"].nunique() == 1
    log.info("  Result: %s", "IDENTICAL" if all_identical else "DIVERGENT")
    if not all_identical:
        log.warning("  Unique hashes: %s", repro_df["output_hash"].unique().tolist())


# ══════════════════════════════════════════════════════════
# 7. Sensitivity Sweeps
# ══════════════════════════════════════════════════════════

SENSITIVITY_PARAMS = {
    "baseline.dirichlet_kappa": {
        "values": [1.0, 5.0, 10.0, 25.0, 50.0],
        "default": 10.0,
        "label": "Dirichlet Shrinkage kappa",
    },
    "scoring.iforest_contamination": {
        "values": [0.01, 0.02, 0.05, 0.10],
        "default": 0.02,
        "label": "IForest Contamination",
    },
    "scoring.gate_percentile_threshold": {
        "values": [50.0, 65.0, 75.0, 85.0, 95.0],
        "default": 75.0,
        "label": "Gate Percentile Threshold",
    },
    "scoring.min_corroborating_channels": {
        "values": [1, 2, 3],
        "default": 2,
        "label": "Min Corroborating Channels",
    },
    "scoring.extreme_threshold": {
        "values": [0.90, 0.95, 0.99],
        "default": 0.95,
        "label": "Extreme Bypass Threshold",
    },
    "baseline.min_events_per_host": {
        "values": [10, 25, 50, 100],
        "default": 25,
        "label": "Min Events per Host",
    },
}


def sensitivity_sweeps(df, out):
    """Sweep key hyperparameters and record gate-pass counts."""
    log.info("Running sensitivity sweeps (%d parameters)", len(SENSITIVITY_PARAMS))

    all_rows = []
    for param_key, spec in SENSITIVITY_PARAMS.items():
        log.info("  Sweeping: %s (%s)", spec["label"], spec["values"])

        for value in spec["values"]:
            try:
                art = run_pipeline(df, **{param_key: value})
                gate_pass = int(art.triage["gate_pass"].sum()) if "gate_pass" in art.triage.columns else 0
                flagged = list(art.triage[art.triage["gate_pass"]]["host"]) if "gate_pass" in art.triage.columns else []

                all_rows.append({
                    "parameter": spec["label"],
                    "param_key": param_key,
                    "value": value,
                    "is_default": value == spec["default"],
                    "gate_pass": gate_pass,
                    "flagged_hosts": ", ".join(str(h) for h in flagged),
                })
                log.info("    %s=%s -> gate_pass=%d", param_key, value, gate_pass)
            except Exception as e:
                log.warning("    %s=%s FAILED: %s", param_key, value, e)
                all_rows.append({
                    "parameter": spec["label"],
                    "param_key": param_key,
                    "value": value,
                    "is_default": value == spec["default"],
                    "gate_pass": -1,
                    "flagged_hosts": "",
                })

    sweep_df = pd.DataFrame(all_rows)
    sweep_df.to_csv(out / "sensitivity_sweeps.csv", index=False)

    # Plot: grid of bar charts
    params = sweep_df["parameter"].unique()
    n = len(params)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, param_name in enumerate(params):
        ax = axes[idx]
        sub = sweep_df[sweep_df["parameter"] == param_name]
        sub = sub[sub["gate_pass"] >= 0]  # skip failures

        colors = ["#3fb950" if d else "#58a6ff" for d in sub["is_default"]]
        ax.bar(range(len(sub)), sub["gate_pass"], color=colors, alpha=0.85)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([str(v) for v in sub["value"]], fontsize=8)
        ax.set_ylabel("Gate Pass")
        ax.set_title(param_name, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Annotate counts
        for j, gp in enumerate(sub["gate_pass"]):
            ax.text(j, gp + 0.3, str(gp), ha="center", fontsize=9, color="#e6edf3")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Sensitivity Analysis: Gate Throughput by Parameter",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out / "sensitivity_sweeps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Host stability: which hosts are flagged across ALL values of each param
    stability_rows = []
    for param_name in params:
        sub = sweep_df[(sweep_df["parameter"] == param_name) & (sweep_df["gate_pass"] >= 0)]
        all_host_sets = []
        for _, row in sub.iterrows():
            hosts = set(h.strip() for h in row["flagged_hosts"].split(",") if h.strip())
            all_host_sets.append(hosts)

        if all_host_sets:
            always_flagged = all_host_sets[0]
            for hs in all_host_sets[1:]:
                always_flagged = always_flagged & hs
            ever_flagged = set()
            for hs in all_host_sets:
                ever_flagged = ever_flagged | hs

            stability_rows.append({
                "parameter": param_name,
                "n_values_swept": len(sub),
                "always_flagged": len(always_flagged),
                "ever_flagged": len(ever_flagged),
                "stable_hosts": ", ".join(sorted(always_flagged)),
            })

    if stability_rows:
        pd.DataFrame(stability_rows).to_csv(out / "host_stability.csv", index=False)

    log.info("  Saved sensitivity_sweeps.csv, .png, and host_stability.csv")


# ══════════════════════════════════════════════════════════
# 8. Workload Reduction (Nomad-style operational framing)
# ══════════════════════════════════════════════════════════

def workload_reduction(art_role, art_ungated, df, out):
    """
    Frame results as analyst workload reduction rather than precision/recall.
    No ground truth needed — compares manual review vs. STRATA-E triage.
    """
    log.info("Computing workload reduction metrics")

    n_hosts = len(art_role.triage)
    n_gate_pass = int(art_role.triage["gate_pass"].sum()) if "gate_pass" in art_role.triage.columns else n_hosts
    n_events = len(df)
    runtime_s = sum(art_role.stage_timings.values()) if art_role.stage_timings else 0

    # Fleet scaling scenarios
    fleet_sizes = [54, 100, 500, 1000, 5000, 10000]
    flagging_rate = n_gate_pass / n_hosts  # observed: 13/54 = 24%

    rows = []
    for fleet in fleet_sizes:
        leads = max(1, int(fleet * flagging_rate))
        reduction_pct = (1 - leads / fleet) * 100

        # Analyst time: assume 30 min per host investigation
        manual_hours = fleet * 0.5
        strata_hours = leads * 0.5

        # Pipeline runtime scales linearly with events
        events_scaled = int(n_events * (fleet / n_hosts))
        runtime_scaled = runtime_s * (fleet / n_hosts)

        rows.append({
            "fleet_size": fleet,
            "hosts_flagged": leads,
            "hosts_filtered": fleet - leads,
            "reduction_pct": round(reduction_pct, 1),
            "manual_review_hours": round(manual_hours, 1),
            "strata_review_hours": round(strata_hours, 1),
            "time_saved_hours": round(manual_hours - strata_hours, 1),
            "pipeline_runtime_min": round(runtime_scaled / 60, 1),
            "events_estimated": events_scaled,
        })

    workload_df = pd.DataFrame(rows)
    workload_df.to_csv(out / "workload_reduction.csv", index=False)

    # Summary text
    summary = []
    summary.append("WORKLOAD REDUCTION ANALYSIS")
    summary.append("=" * 50)
    summary.append(f"Observed: {n_hosts} hosts -> {n_gate_pass} triage leads ({100*flagging_rate:.0f}% flagging rate)")
    summary.append(f"Pipeline runtime: {runtime_s:.0f}s ({runtime_s/60:.1f} min)")
    summary.append(f"Each lead has multi-channel corroboration (>= 2 channels above threshold)")
    summary.append("")
    summary.append(f"{'Fleet':>8} {'Leads':>6} {'Filtered':>9} {'Reduction':>10} {'Manual hrs':>11} {'STRATA hrs':>11} {'Saved':>8}")
    for _, row in workload_df.iterrows():
        summary.append(
            f"{row['fleet_size']:>8} {row['hosts_flagged']:>6} {row['hosts_filtered']:>9} "
            f"{row['reduction_pct']:>9.0f}% {row['manual_review_hours']:>11.1f} "
            f"{row['strata_review_hours']:>11.1f} {row['time_saved_hours']:>7.1f}h"
        )
    summary.append("")
    summary.append("Framing: Without STRATA-E, an analyst reviews all hosts or relies on")
    summary.append("single-channel alerts dominated by infrastructure noise. With STRATA-E,")
    summary.append("the analyst reviews only corroborated leads with channel breakdowns,")
    summary.append("MITRE ATT&CK annotations, and the specific behavioral sequences that")
    summary.append("drove each score.")

    summary_text = "\n".join(summary)
    (out / "workload_reduction.txt").write_text(summary_text)
    print("\n" + summary_text)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(
        [str(f) for f in workload_df["fleet_size"]],
        workload_df["manual_review_hours"],
        alpha=0.4, color="#f85149", label="Without STRATA-E"
    )
    ax1.bar(
        [str(f) for f in workload_df["fleet_size"]],
        workload_df["strata_review_hours"],
        alpha=0.85, color="#3fb950", label="With STRATA-E"
    )
    ax1.set_xlabel("Fleet Size (hosts)")
    ax1.set_ylabel("Analyst Review Hours")
    ax1.set_title("Analyst Workload: Manual vs. STRATA-E Triage")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    ax2.plot(workload_df["fleet_size"], workload_df["reduction_pct"],
             "o-", color="#58a6ff", linewidth=2, markersize=8)
    ax2.set_xlabel("Fleet Size (hosts)")
    ax2.set_ylabel("Review Volume Reduction (%)")
    ax2.set_title("Triage Queue Reduction")
    ax2.set_ylim(60, 100)
    ax2.grid(True, alpha=0.3)
    for _, row in workload_df.iterrows():
        ax2.annotate(f"{row['reduction_pct']:.0f}%",
                     (row["fleet_size"], row["reduction_pct"]),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=9, color="#e6edf3")

    plt.tight_layout()
    fig.savefig(out / "workload_reduction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved workload_reduction.csv, .txt, and .png")


# ══════════════════════════════════════════════════════════
# 9. Per-Host Explainability Summary
# ══════════════════════════════════════════════════════════

def host_explainability(art, df, out):
    """
    For each host that passes the gate, generate a human-readable
    explanation of why it was flagged — which channels, what evidence.
    """
    log.info("Generating per-host explainability summaries")

    triage = art.triage
    if "gate_pass" not in triage.columns:
        log.warning("No gate_pass column — skipping explainability")
        return

    flagged = triage[triage["gate_pass"]].copy()
    if flagged.empty:
        return

    # Get pair stats for context
    pair_stats = art.pair_stats if art.pair_stats is not None else pd.DataFrame()

    explanations = []
    for _, row in flagged.iterrows():
        host = row["host"]
        role = "unknown"
        if art.host_roles is not None:
            role_match = art.host_roles[art.host_roles["host"] == host]
            if not role_match.empty:
                role = str(role_match.iloc[0]["role_id"])

        # Channel breakdown
        channels_firing = []
        channels_detail = []

        s_seq = row.get("S_seq", 0)
        s_freq = row.get("S_freq", 0)
        s_ctx = row.get("S_ctx", 0)
        s_drift = row.get("S_drift", 0)

        if s_seq > 0.005:
            channels_firing.append("sequence")
            channels_detail.append(f"S_seq={s_seq:.4f}: process transition patterns diverge from {role} peer group")

        if s_freq > 0.3:
            channels_firing.append("frequency")
            channels_detail.append(f"S_freq={s_freq:.3f}: event volume/mix anomalous relative to fleet")

        if s_ctx > 0.8:
            channels_firing.append("context")
            channels_detail.append(f"S_ctx={s_ctx:.3f}: elevated command novelty or suspicious event pairs")

        if s_drift > 0.1:
            channels_firing.append("drift")
            channels_detail.append(f"S_drift={s_drift:.3f}: behavioral shift from prior window")

        # Pair correlation details
        pair_info = ""
        if not pair_stats.empty and host in pair_stats["host"].values:
            host_pairs = pair_stats[pair_stats["host"] == host].iloc[0]
            pair_info = (f"Pair correlation: {int(host_pairs['n_pairs'])} event pairs, "
                         f"top tactic: {host_pairs['top_tactic']}, "
                         f"weighted score: {host_pairs['weighted_score_sum']:,.0f}")

        explanations.append({
            "triage_rank": int(row.get("triage_rank", 0)),
            "host": host,
            "role": role,
            "score": round(float(row.get("score", 0)), 4),
            "channels_firing": ", ".join(channels_firing),
            "n_channels": len(channels_firing),
            "S_seq": round(s_seq, 6),
            "S_freq": round(s_freq, 4),
            "S_ctx": round(s_ctx, 4),
            "S_drift": round(s_drift, 4),
            "channel_details": " | ".join(channels_detail),
            "pair_info": pair_info,
        })

    expl_df = pd.DataFrame(explanations)
    expl_df.to_csv(out / "host_explainability.csv", index=False)

    # Human-readable summary
    lines = ["PER-HOST EXPLAINABILITY SUMMARY", "=" * 60, ""]
    for _, row in expl_df.iterrows():
        lines.append(f"#{row['triage_rank']}  {row['host']}  (role: {row['role']})  score={row['score']}")
        lines.append(f"  Channels firing: {row['channels_firing']} ({row['n_channels']} channels)")
        for detail in row["channel_details"].split(" | "):
            if detail.strip():
                lines.append(f"    - {detail.strip()}")
        if row["pair_info"]:
            lines.append(f"    - {row['pair_info']}")
        lines.append("")

    summary_text = "\n".join(lines)
    (out / "host_explainability.txt").write_text(summary_text)
    print("\n" + summary_text)
    log.info("  Saved host_explainability.csv and .txt")


# ══════════════════════════════════════════════════════════
# 10. SHAP Feature Importance (Frequency Channel)
# ══════════════════════════════════════════════════════════

def shap_analysis(art, df, out):
    """
    Run SHAP on the Isolation Forest frequency channel to identify
    which rate features drive the frequency anomaly scores.
    """
    log.info("Running SHAP feature importance analysis")

    try:
        import shap
    except ImportError:
        log.warning("shap not installed — skipping. Install with: pip install shap")
        return

    # Need the fitted model and rate features
    # Re-run fit to get access to FittedArtifacts (art only has triage output)
    from sysmon_pipeline import StrataPipeline, StrataConfig
    from sysmon_pipeline.pairs import compute_rate_features

    cfg = StrataConfig.fast()
    pipe = StrataPipeline(cfg)
    fitted = pipe.fit(df)

    # Compute rate features for SHAP
    from sysmon_pipeline.schema import normalize_schema
    from sysmon_pipeline.mapping import build_tokens
    events = normalize_schema(df, cfg)
    events = build_tokens(events)
    rates = compute_rate_features(events, cfg)

    model = fitted.freq_model
    X = rates.drop(columns=["host"], errors="ignore").fillna(0.0)
    feature_names = list(X.columns)

    # Subsample for speed
    if len(X) > 200:
        X_sample = X.sample(n=200, random_state=42)
    else:
        X_sample = X

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        log.warning("SHAP computation failed: %s", e)
        return

    # Mean absolute SHAP value per feature
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = range(1, len(importance_df) + 1)
    importance_df.to_csv(out / "shap_importance.csv", index=False)

    # Plot
    top_k = min(15, len(importance_df))
    top = importance_df.head(top_k).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.4)))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#58a6ff", edgecolor="none", alpha=0.85)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_k} Frequency Channel Features (SHAP Importance)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # SHAP summary plot (beeswarm)
    try:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                          show=False, max_display=top_k)
        plt.tight_layout()
        plt.savefig(out / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        log.warning("SHAP beeswarm plot failed: %s", e)

    log.info("  Saved shap_importance.csv, .png, and beeswarm")
    log.info("  Top 5 features: %s", ", ".join(importance_df.head(5)["feature"].tolist()))


# ══════════════════════════════════════════════════════════
# 11. Baseline Comparison (STRATA-E vs. simpler approaches)
# ══════════════════════════════════════════════════════════

def baseline_comparison(art_role, df, out):
    """
    Compare STRATA-E's triage against simpler baseline approaches.
    No ground truth needed — compare on triage composition and quality.
    """
    log.info("Running baseline comparison")

    from sysmon_pipeline.schema import normalize_schema
    from sysmon_pipeline.mapping import build_tokens
    from sysmon_pipeline import StrataConfig

    cfg = StrataConfig.fast()

    # STRATA-E triage (already computed)
    strata_triage = art_role.triage.copy()
    strata_top = strata_triage.head(12)[["host", "role_id", "score", "gate_pass",
                                          "S_seq", "S_freq", "S_ctx"]].copy()
    strata_top["method"] = "STRATA-E"
    strata_top["rank"] = range(1, len(strata_top) + 1)

    # ── Baseline 1: Event volume ranking ──
    log.info("  Baseline 1: Event volume ranking")
    vol_counts = df.groupby("host").size().reset_index(name="n_events")
    vol_counts = vol_counts.sort_values("n_events", ascending=False).reset_index(drop=True)
    vol_counts["rank"] = range(1, len(vol_counts) + 1)

    # ── Baseline 2: Sigma-style suspicious event count ──
    log.info("  Baseline 2: Sigma-style alert count")
    try:
        events = normalize_schema(df, cfg)
        events = build_tokens(events)

        # Count "suspicious" events per host
        suspicious_mask = pd.Series(False, index=events.index)
        for col in ["is_lolbin", "is_script", "has_encoded", "has_bypass",
                     "has_download_cradle", "has_reflection"]:
            if col in events.columns:
                suspicious_mask = suspicious_mask | (events[col] == True)

        sigma_counts = events[suspicious_mask].groupby("host").size().reset_index(name="sigma_alerts")
        # Add hosts with zero alerts
        all_hosts = pd.DataFrame({"host": df["host"].unique()})
        sigma_counts = all_hosts.merge(sigma_counts, on="host", how="left").fillna(0)
        sigma_counts["sigma_alerts"] = sigma_counts["sigma_alerts"].astype(int)
        sigma_counts = sigma_counts.sort_values("sigma_alerts", ascending=False).reset_index(drop=True)
        sigma_counts["rank"] = range(1, len(sigma_counts) + 1)
    except Exception as e:
        log.warning("  Sigma baseline failed: %s", e)
        sigma_counts = pd.DataFrame(columns=["host", "sigma_alerts", "rank"])

    # ── Baseline 3: Single-channel (frequency only) ──
    log.info("  Baseline 3: Frequency-only (Isolation Forest)")
    freq_only = strata_triage[["host", "role_id", "S_freq"]].copy()
    freq_only = freq_only.sort_values("S_freq", ascending=False).reset_index(drop=True)
    freq_only["rank"] = range(1, len(freq_only) + 1)

    # ── Baseline 4: Global baseline (no role conditioning) ──
    log.info("  Baseline 4: Global baseline (from ablation)")
    try:
        art_global = run_pipeline(df, **{"ablation.use_role_baselining": False})
        global_triage = art_global.triage.copy()
        global_top = global_triage.head(12)[["host", "role_id", "score"]].copy()
        global_top["method"] = "Global baseline"
        global_top["rank"] = range(1, len(global_top) + 1)
    except Exception as e:
        log.warning("  Global baseline failed: %s", e)
        global_top = pd.DataFrame()

    # ── Build comparison table ──
    K = 12  # compare top-K

    def classify_host(host, role):
        """Is this an infrastructure host or a workstation?"""
        infra_roles = {"dc", "mail", "dns", "wec", "sql", "sharepoint", "security", "proxy"}
        if role in infra_roles:
            return "infrastructure"
        return "workstation"

    # Merge role info
    role_map = dict(zip(strata_triage["host"], strata_triage["role_id"]))

    comparison_rows = []

    # STRATA-E
    for _, row in strata_triage.head(K).iterrows():
        comparison_rows.append({
            "method": "STRATA-E (full)",
            "rank": int(row.get("triage_rank", 0)),
            "host": row["host"],
            "role": row.get("role_id", "unknown"),
            "host_type": classify_host(row["host"], row.get("role_id", "")),
            "gate_pass": bool(row.get("gate_pass", False)),
            "n_channels_firing": sum(1 for ch in ["S_seq", "S_freq", "S_ctx"]
                                     if row.get(ch, 0) > 0.3),
        })

    # Volume baseline
    for _, row in vol_counts.head(K).iterrows():
        role = role_map.get(row["host"], "unknown")
        comparison_rows.append({
            "method": "Volume ranking",
            "rank": int(row["rank"]),
            "host": row["host"],
            "role": role,
            "host_type": classify_host(row["host"], role),
            "gate_pass": False,
            "n_channels_firing": 0,
        })

    # Sigma baseline
    if not sigma_counts.empty:
        for _, row in sigma_counts.head(K).iterrows():
            role = role_map.get(row["host"], "unknown")
            comparison_rows.append({
                "method": "Sigma alert count",
                "rank": int(row["rank"]),
                "host": row["host"],
                "role": role,
                "host_type": classify_host(row["host"], role),
                "gate_pass": False,
                "n_channels_firing": 0,
            })

    # Frequency-only baseline
    for _, row in freq_only.head(K).iterrows():
        role = role_map.get(row["host"], "unknown")
        comparison_rows.append({
            "method": "Frequency only (IF)",
            "rank": int(row["rank"]),
            "host": row["host"],
            "role": role,
            "host_type": classify_host(row["host"], role),
            "gate_pass": False,
            "n_channels_firing": 0,
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(out / "baseline_comparison.csv", index=False)

    # ── Summary statistics ──
    summary_rows = []
    for method in comp_df["method"].unique():
        mdf = comp_df[comp_df["method"] == method]
        n_infra = (mdf["host_type"] == "infrastructure").sum()
        n_wks = (mdf["host_type"] == "workstation").sum()
        # How many of STRATA-E's top-K are in this method's top-K?
        strata_hosts = set(comp_df[comp_df["method"] == "STRATA-E (full)"]["host"])
        method_hosts = set(mdf["host"])
        overlap = len(strata_hosts & method_hosts)

        summary_rows.append({
            "method": method,
            "top_k": K,
            "infrastructure_hosts": n_infra,
            "workstation_hosts": n_wks,
            "infra_fraction": round(n_infra / K, 2),
            "overlap_with_strata": overlap,
            "overlap_fraction": round(overlap / K, 2),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / "baseline_summary.csv", index=False)

    # ── Print summary ──
    print("\nBASELINE COMPARISON (Top-{} hosts)".format(K))
    print("=" * 70)
    for _, row in summary_df.iterrows():
        print(f"  {row['method']:<25} infra={row['infrastructure_hosts']:>2} "
              f" wks={row['workstation_hosts']:>2} "
              f" overlap_with_STRATA={row['overlap_with_strata']:>2}/{K}")
    print()

    # ── Plot: infrastructure vs workstation composition ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    methods = summary_df["method"].values
    x = range(len(methods))
    ax1.bar(x, summary_df["infrastructure_hosts"], color="#f85149", alpha=0.8,
            label="Infrastructure")
    ax1.bar(x, summary_df["workstation_hosts"],
            bottom=summary_df["infrastructure_hosts"], color="#3fb950", alpha=0.8,
            label="Workstation")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel(f"Hosts in Top-{K}")
    ax1.set_title(f"Top-{K} Composition: Infrastructure vs. Workstation")
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, K + 1)

    ax2.bar(x, summary_df["overlap_fraction"], color="#58a6ff", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("Overlap with STRATA-E")
    ax2.set_title(f"Top-{K} Overlap with STRATA-E Triage")
    ax2.set_ylim(0, 1.1)
    for i, v in enumerate(summary_df["overlap_fraction"]):
        ax2.text(i, v + 0.03, f"{v:.0%}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(out / "baseline_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save detailed per-method top-K lists
    vol_counts.head(K).to_csv(out / "baseline_volume_top.csv", index=False)
    if not sigma_counts.empty:
        sigma_counts.head(K).to_csv(out / "baseline_sigma_top.csv", index=False)
    freq_only.head(K).to_csv(out / "baseline_freq_only_top.csv", index=False)

    log.info("  Saved baseline_comparison.csv, baseline_summary.csv, .png")


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Extended STRATA-E evaluation")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results_extended")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip the ablation sweep (saves ~10 min)")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="Skip sensitivity sweeps (saves ~15 min)")
    parser.add_argument("--skip-scalability", action="store_true",
                        help="Skip scalability benchmarks")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # Load and clean
    log.info("Loading %s", args.input)
    df = pd.read_parquet(args.input)
    df = clean(df)
    log.info("Cleaned: %d events, %d hosts", len(df), df["host"].nunique())

    # ── Run core conditions ───────────────────────────────
    log.info("Running role baselining (default)")
    art_role = run_pipeline(df)

    log.info("Running global baselining")
    art_global = run_pipeline(df, **{"ablation.use_role_baselining": False})

    log.info("Running ungated")
    art_ungated = run_pipeline(df, **{"ablation.use_corroboration_gate": False})

    # ── Channel contribution ──────────────────────────────
    channel_contribution(art_role, out)

    # ── H2 plot ───────────────────────────────────────────
    h2_plot(art_role, art_global, out)

    # ── H4 plot ───────────────────────────────────────────
    h4_plot(art_role, out)

    # ── H5 plot ───────────────────────────────────────────
    h5_plot(art_role, art_ungated, out)

    # ── Deployment projections ────────────────────────────
    deployment_projections(art_role, out)

    # ── Workload reduction framing ────────────────────────
    workload_reduction(art_role, art_ungated, df, out)

    # ── Per-host explainability ───────────────────────────
    host_explainability(art_role, df, out)

    # ── SHAP feature importance ───────────────────────────
    shap_analysis(art_role, df, out)

    # ── Baseline comparison ───────────────────────────────
    if not args.skip_ablation:
        baseline_comparison(art_role, df, out)
    else:
        log.info("Skipping baseline comparison (--skip-ablation)")

    # ── Ablation comparison ───────────────────────────────
    if not args.skip_ablation:
        ablation_comparison(df, out)
    else:
        log.info("Skipping ablation sweep (--skip-ablation)")

    # ── Scalability benchmarks ────────────────────────────
    if not args.skip_scalability:
        scalability_timing(df, out)
    else:
        log.info("Skipping scalability benchmarks (--skip-scalability)")

    # ── Reproducibility check ─────────────────────────────
    reproducibility_check(df, out)

    # ── Sensitivity sweeps ────────────────────────────────
    if not args.skip_sensitivity:
        sensitivity_sweeps(df, out)
    else:
        log.info("Skipping sensitivity sweeps (--skip-sensitivity)")

    total = time.perf_counter() - t_start
    log.info("Done in %.1fs (%.1f min). Results in %s/", total, total/60, out)
    log.info("Files: %s", ", ".join(f.name for f in sorted(out.glob("*"))))


if __name__ == "__main__":
    main()
