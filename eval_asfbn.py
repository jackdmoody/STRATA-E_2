#!/usr/bin/env python3
"""
eval_asfbn.py
==============
Run all hypothesis evaluations on the ASFBN dataset and save results.

Produces:
  results_asfbn/
    triage_role.csv          — triage table with role baselining (default)
    triage_global.csv        — triage table with global baselining
    triage_ungated.csv       — triage table without corroboration gate
    triage_no_shrinkage.csv  — triage table without Dirichlet shrinkage
    role_assignments.csv     — host → role_id mapping
    pair_stats.csv           — per-host pair correlation statistics
    h2_comparison.csv        — role vs global ranking comparison
    h5_comparison.csv        — gated vs ungated ranking comparison
    h1_shrinkage.csv         — shrinkage vs MLE JSD variance comparison
    stage_timings.csv        — per-stage timing breakdown
    summary.txt              — human-readable summary of all results

Usage:
    python eval_asfbn.py --input data/first_sample_3.parquet
    python eval_asfbn.py --input data/first_sample_3.parquet --output results_asfbn
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_asfbn")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standard ASFBN cleaning."""
    for col in ["Image", "ParentImage", "CommandLine",
                "ParentCommandLine", "IntegrityLevel"]:
        if col in df.columns:
            df[col] = df[col].replace({"UNKNOWN": pd.NA, "-": pd.NA})
    if "event_provider" in df.columns:
        df = df[df["event_provider"] != "Puppet"].copy()
    if "host" in df.columns:
        df = df[~df["host"].isin({"unknown", "UNKNOWN"})].copy()
    return df


def run_condition(df, name, **overrides):
    """Run pipeline with specific config overrides, return artifacts."""
    from sysmon_pipeline import StrataPipeline, StrataConfig

    cfg = StrataConfig.fast()
    for key, val in overrides.items():
        parts = key.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)

    log.info("Running condition: %s", name)
    t0 = time.perf_counter()
    pipe = StrataPipeline(cfg)
    art = pipe.fit_score(df)
    elapsed = time.perf_counter() - t0
    log.info("  %s completed in %.1fs", name, elapsed)
    return art


def main():
    parser = argparse.ArgumentParser(description="ASFBN hypothesis evaluation")
    parser.add_argument("--input", required=True, help="Path to parquet file")
    parser.add_argument("--output", default="results_asfbn", help="Output directory")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # ── Load and clean ────────────────────────────────────────────────
    log.info("Loading %s", args.input)
    df = pd.read_parquet(args.input)
    df = clean(df)
    log.info("Cleaned: %d events, %d hosts", len(df), df["host"].nunique())

    summary_lines = []
    summary_lines.append("STRATA-E ASFBN Evaluation Results")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Data: {args.input}")
    summary_lines.append(f"Events: {len(df):,}")
    summary_lines.append(f"Hosts: {df['host'].nunique()}")
    summary_lines.append(f"Time range: {df['ts'].min()} to {df['ts'].max()}")
    summary_lines.append("")

    # ── Condition 1: Full pipeline (role baselining ON) ────────────────
    art_role = run_condition(df, "role_baselining")

    art_role.triage.to_csv(out / "triage_role.csv", index=False)
    art_role.host_roles.to_csv(out / "role_assignments.csv", index=False)
    if art_role.pair_stats is not None:
        art_role.pair_stats.to_csv(out / "pair_stats.csv", index=False)
    if art_role.stage_timings:
        pd.DataFrame([art_role.stage_timings]).to_csv(
            out / "stage_timings.csv", index=False
        )

    summary_lines.append("ROLE ASSIGNMENTS")
    summary_lines.append("-" * 40)
    for role, hosts in art_role.host_roles.groupby("role_id")["host"].apply(list).items():
        summary_lines.append(f"  {role} ({len(hosts)}): {', '.join(hosts[:5])}"
                             + ("..." if len(hosts) > 5 else ""))
    summary_lines.append("")

    # ── Condition 2: Global baselining (role baselining OFF) ──────────
    art_global = run_condition(df, "global_baselining",
                               **{"ablation.use_role_baselining": False})
    art_global.triage.to_csv(out / "triage_global.csv", index=False)

    # ── H2: Role vs Global comparison ─────────────────────────────────
    summary_lines.append("H2: ROLE BASELINING vs GLOBAL BASELINING")
    summary_lines.append("-" * 40)

    g_role = art_role.triage.set_index("host")
    g_glob = art_global.triage.set_index("host")

    h2_rows = []
    for host in g_role.index:
        h2_rows.append({
            "host": host,
            "rank_role": int(g_role.loc[host, "triage_rank"]),
            "rank_global": int(g_glob.loc[host, "triage_rank"]),
            "rank_change": int(g_glob.loc[host, "triage_rank"] - g_role.loc[host, "triage_rank"]),
            "gate_role": bool(g_role.loc[host, "gate_pass"]),
            "gate_global": bool(g_glob.loc[host, "gate_pass"]),
            "S_seq_role": float(g_role.loc[host, "S_seq"]),
            "S_seq_global": float(g_glob.loc[host, "S_seq"]),
            "S_freq": float(g_role.loc[host, "S_freq"]),
            "role_id": str(art_role.host_roles.set_index("host").loc[host, "role_id"]),
        })
    h2_df = pd.DataFrame(h2_rows).sort_values("rank_role")
    h2_df.to_csv(out / "h2_comparison.csv", index=False)

    gate_role = int(art_role.triage["gate_pass"].sum())
    gate_global = int(art_global.triage["gate_pass"].sum())
    seq_dead_global = (art_global.triage["S_seq"] == 0).all()
    rank_changes = h2_df[h2_df["rank_change"] != 0]
    biggest_mover = h2_df.loc[h2_df["rank_change"].abs().idxmax()]

    summary_lines.append(f"  Gate pass (role):   {gate_role}")
    summary_lines.append(f"  Gate pass (global): {gate_global}")
    summary_lines.append(f"  Sequence channel dead under global: {seq_dead_global}")
    summary_lines.append(f"  Hosts with rank change: {len(rank_changes)}")
    summary_lines.append(f"  Biggest mover: {biggest_mover['host']} "
                         f"(moved {int(biggest_mover['rank_change'])} positions)")
    summary_lines.append(f"  Result: {'SUPPORTED' if gate_role > gate_global else 'NOT SUPPORTED'}")
    summary_lines.append("")

    # ── Condition 3: No corroboration gate ────────────────────────────
    art_ungated = run_condition(df, "no_gate",
                                **{"ablation.use_corroboration_gate": False})
    art_ungated.triage.to_csv(out / "triage_ungated.csv", index=False)

    # ── H5: Gated vs Ungated comparison ───────────────────────────────
    summary_lines.append("H5: CORROBORATION GATE EFFECT")
    summary_lines.append("-" * 40)

    g_gated = art_role.triage
    g_ungated = art_ungated.triage

    h5_rows = []
    for k in [5, 10, 15, 20]:
        top_gated = set(g_gated[g_gated["gate_pass"]].head(k)["host"])
        top_ungated = set(g_ungated.head(k)["host"])
        overlap = top_gated & top_ungated
        gated_only = top_gated - top_ungated
        ungated_only = top_ungated - top_gated
        h5_rows.append({
            "K": k,
            "overlap": len(overlap),
            "gated_only": len(gated_only),
            "ungated_only": len(ungated_only),
            "gated_only_hosts": ", ".join(sorted(gated_only)) or "none",
            "ungated_only_hosts": ", ".join(sorted(ungated_only)) or "none",
        })

    h5_df = pd.DataFrame(h5_rows)
    h5_df.to_csv(out / "h5_comparison.csv", index=False)

    summary_lines.append(f"  With gate:    {gate_role} hosts pass out of {len(g_gated)}")
    summary_lines.append(f"  Without gate: all {len(g_ungated)} hosts ranked")
    for _, row in h5_df.iterrows():
        summary_lines.append(f"  Top-{row['K']}: overlap={row['overlap']}  "
                             f"gated_only={row['gated_only_hosts']}  "
                             f"ungated_only={row['ungated_only_hosts']}")
    summary_lines.append("")

    # ── Condition 4: No shrinkage ─────────────────────────────────────
    art_noshrink = run_condition(df, "no_shrinkage",
                                 **{"ablation.use_dirichlet_shrinkage": False})
    art_noshrink.triage.to_csv(out / "triage_no_shrinkage.csv", index=False)

    # ── H1: Shrinkage effect on JSD variance ──────────────────────────
    summary_lines.append("H1: DIRICHLET SHRINKAGE EFFECT")
    summary_lines.append("-" * 40)

    seq_shrink = art_role.seq_scores
    seq_noshrink = art_noshrink.seq_scores

    h1_rows = []
    if seq_shrink is not None and seq_noshrink is not None:
        merged = seq_shrink[["host", "S_seq", "n_events"]].merge(
            seq_noshrink[["host", "S_seq"]],
            on="host", suffixes=("_shrinkage", "_mle"),
        )

        # Bin by event count and compute variance per bin
        bins = [0, 50, 100, 250, 500, 1000, float("inf")]
        labels = ["<50", "50-100", "100-250", "250-500", "500-1000", "1000+"]
        merged["event_bin"] = pd.cut(merged["n_events"], bins=bins, labels=labels)

        for bin_label, group in merged.groupby("event_bin", observed=True):
            if len(group) < 2:
                continue
            var_shrink = group["S_seq_shrinkage"].var()
            var_mle = group["S_seq_mle"].var()
            reduction = 1.0 - (var_shrink / (var_mle + 1e-12))
            h1_rows.append({
                "event_bin": str(bin_label),
                "n_hosts": len(group),
                "jsd_var_shrinkage": round(var_shrink, 8),
                "jsd_var_mle": round(var_mle, 8),
                "variance_reduction": round(reduction, 4),
            })
            summary_lines.append(
                f"  {bin_label:>10}: n={len(group):>3}  "
                f"var_shrink={var_shrink:.6f}  var_mle={var_mle:.6f}  "
                f"reduction={reduction:.1%}"
            )

        # Overall
        overall_var_shrink = merged["S_seq_shrinkage"].var()
        overall_var_mle = merged["S_seq_mle"].var()
        overall_reduction = 1.0 - (overall_var_shrink / (overall_var_mle + 1e-12))
        summary_lines.append(
            f"  {'OVERALL':>10}: var_shrink={overall_var_shrink:.6f}  "
            f"var_mle={overall_var_mle:.6f}  reduction={overall_reduction:.1%}"
        )
    else:
        summary_lines.append("  Could not compute (missing seq_scores)")

    if h1_rows:
        pd.DataFrame(h1_rows).to_csv(out / "h1_shrinkage.csv", index=False)
    summary_lines.append("")

    # ── H3: Bootstrap calibration p-value uniformity ──────────────────
    summary_lines.append("H3: BOOTSTRAP CALIBRATION (P-VALUE UNIFORMITY)")
    summary_lines.append("-" * 40)

    if art_role.seq_scores is not None and "S_seq_pvalue" in art_role.seq_scores.columns:
        from scipy.stats import kstest

        pvals = art_role.seq_scores["S_seq_pvalue"].dropna().to_numpy()
        n_pvals = len(pvals)

        # Assume all hosts are benign (cyber range baseline, no active attack).
        # Under correct calibration, empirical p-values on benign hosts should
        # be uniform on [0, 1].
        ks_stat, ks_p = kstest(pvals, "uniform")
        mean_pval = float(pvals.mean())
        median_pval = float(np.median(pvals))
        calibration_ok = ks_p > 0.05

        h3_data = {
            "n_hosts": n_pvals,
            "mean_pvalue": round(mean_pval, 4),
            "median_pvalue": round(median_pval, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_p), 4),
            "calibration_valid": calibration_ok,
            "method": "empirical (distribution-free)",
        }
        pd.DataFrame([h3_data]).to_csv(out / "h3_calibration.csv", index=False)

        # Also save the raw p-values for histogram plotting
        pd.DataFrame({
            "host": art_role.seq_scores["host"],
            "S_seq": art_role.seq_scores["S_seq"],
            "S_seq_pvalue": art_role.seq_scores["S_seq_pvalue"],
            "S_seq_z": art_role.seq_scores.get("S_seq_z", pd.NA),
            "S_seq_percentile": art_role.seq_scores.get("S_seq_percentile", pd.NA),
        }).to_csv(out / "h3_pvalues.csv", index=False)

        summary_lines.append(f"  Method:          empirical (distribution-free)")
        summary_lines.append(f"  Hosts tested:    {n_pvals}")
        summary_lines.append(f"  Mean p-value:    {mean_pval:.4f}  (ideal: 0.50)")
        summary_lines.append(f"  Median p-value:  {median_pval:.4f}  (ideal: 0.50)")
        summary_lines.append(f"  KS statistic:    {ks_stat:.4f}")
        summary_lines.append(f"  KS p-value:      {ks_p:.4f}")
        summary_lines.append(f"  Calibration:     {'VALID (p > 0.05)' if calibration_ok else 'INVALID (p <= 0.05)'}")
        summary_lines.append(f"  Result:          {'SUPPORTED' if calibration_ok else 'NOT SUPPORTED'}")

        # Distribution summary (quartiles for the paper)
        q25, q50, q75 = np.percentile(pvals, [25, 50, 75])
        summary_lines.append(f"  P-value quartiles: Q1={q25:.3f}  Q2={q50:.3f}  Q3={q75:.3f}")
        summary_lines.append(f"    (uniform ideal:  Q1=0.250  Q2=0.500  Q3=0.750)")
    else:
        summary_lines.append("  Could not compute (S_seq_pvalue not available)")
    summary_lines.append("")

    # ── H4: Channel comparison ────────────────────────────────────────
    summary_lines.append("H4: CHANNEL CONTRIBUTION")
    summary_lines.append("-" * 40)

    triage = art_role.triage
    for channel in ["S_seq", "S_freq", "S_ctx"]:
        if channel in triage.columns:
            ranked = triage.sort_values(channel, ascending=False)
            top5 = list(ranked.head(5)["host"])
            summary_lines.append(f"  {channel} top 5: {', '.join(str(h) for h in top5)}")

    # Check if fused ranking differs from any single channel
    fused_top10 = set(triage.head(10)["host"])
    for channel in ["S_seq", "S_freq", "S_ctx"]:
        if channel in triage.columns:
            chan_top10 = set(triage.sort_values(channel, ascending=False).head(10)["host"])
            overlap = len(fused_top10 & chan_top10)
            summary_lines.append(f"  Fused vs {channel} top-10 overlap: {overlap}/10")
    summary_lines.append("")

    # ── Triage summary ────────────────────────────────────────────────
    summary_lines.append("TRIAGE RESULTS (ROLE BASELINING)")
    summary_lines.append("-" * 40)
    cols = ["triage_rank", "host", "score", "gate_pass", "S_seq", "S_freq", "S_ctx"]
    cols = [c for c in cols if c in triage.columns]
    for _, row in triage[cols].head(20).iterrows():
        parts = [f"{c}={row[c]}" for c in cols]
        summary_lines.append("  " + "  ".join(parts))
    summary_lines.append("")

    # ── Pair correlation highlights ───────────────────────────────────
    if art_role.pair_stats is not None and not art_role.pair_stats.empty:
        summary_lines.append("PAIR CORRELATION HIGHLIGHTS")
        summary_lines.append("-" * 40)
        top_pairs = art_role.pair_stats.sort_values(
            "weighted_score_sum", ascending=False
        ).head(10)
        for _, row in top_pairs.iterrows():
            summary_lines.append(
                f"  {row['host']:<20} score={row['weighted_score_sum']:>12,.1f}  "
                f"pairs={row['n_pairs']}  tactic={row['top_tactic']}"
            )
        summary_lines.append("")

    # ── Stage timings ─────────────────────────────────────────────────
    if art_role.stage_timings:
        summary_lines.append("STAGE TIMINGS")
        summary_lines.append("-" * 40)
        total = sum(art_role.stage_timings.values())
        for stage, elapsed in sorted(art_role.stage_timings.items(), key=lambda x: -x[1]):
            pct = 100 * elapsed / (total + 1e-9)
            summary_lines.append(f"  {stage:<28} {elapsed:>7.2f}s  ({pct:>5.1f}%)")
        summary_lines.append("")

    # ── Write summary ─────────────────────────────────────────────────
    total_time = time.perf_counter() - t_start
    summary_lines.append(f"Total evaluation time: {total_time:.1f}s ({total_time/60:.1f} min)")

    summary_text = "\n".join(summary_lines)
    (out / "summary.txt").write_text(summary_text)
    print()
    print(summary_text)

    log.info("All results saved to %s/", out)
    log.info("Files: %s", ", ".join(f.name for f in sorted(out.glob("*"))))


if __name__ == "__main__":
    main()
