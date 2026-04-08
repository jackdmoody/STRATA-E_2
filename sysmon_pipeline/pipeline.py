"""
STRATA-E Unified Pipeline Orchestrator
========================================
Combines the best structural patterns:
  - v12_modular:      stage-based run(), SysmonArtifacts dataclass, fit/score separation
  - pipeline_updated: SysmonBehavioralPipeline class, FittedArtifacts, richer fit() logic

All 6 fixes are wired in and controlled by cfg.ablation flags.
The same run() code path handles all ablation study conditions.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from .config import StrataConfig, AblationConfig
from .schema import normalize_schema, validate_schema
from .mapping import build_tokens
from .sequence import (
    assign_sessions, bucket_deltas, build_transition_counts,
    build_role_gap_thresholds,
)
from .pairs import (
    compute_rate_features, build_role_features, infer_roles,
    correlate_critical_events_by_host, compute_pair_stats,
    compute_role_pair_baselines,
)
from .divergence import (
    fit_peer_baselines, score_sequence_divergence, score_drift,
    compute_shrinkage_weights, detect_shrinkage_anomalies,
    calibrate_jsd_null_distribution, clear_bootstrap_cache,
)
from .scoring import (
    fit_frequency_model, score_frequency, score_context,
    fuse_scores, build_ranked_triage,
    build_cmdline_vectorizer, build_baseline_matrix,
)

logger = logging.getLogger("strata")


# ---------------------------------------------------------------------------
# Artifacts dataclass
# ---------------------------------------------------------------------------

@dataclass
class StrataArtifacts:
    """All outputs from a pipeline run. None = stage was skipped."""
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    host_roles: Optional[pd.DataFrame] = None
    peer_baselines: Optional[Dict[str, pd.DataFrame]] = None
    role_gap_thresholds: Optional[Dict[str, float]] = None
    transition_counts: Optional[pd.DataFrame] = None
    rate_features: Optional[pd.DataFrame] = None
    pair_stats: Optional[pd.DataFrame] = None
    seq_scores: Optional[pd.DataFrame] = None
    freq_scores: Optional[pd.DataFrame] = None
    ctx_scores: Optional[pd.DataFrame] = None
    drift_scores: Optional[pd.DataFrame] = None
    shrinkage_df: Optional[pd.DataFrame] = None
    triage: Optional[pd.DataFrame] = None
    calibration_report: Optional[dict] = None
    stage_timings: Optional[Dict[str, float]] = None


@dataclass
class FittedArtifacts:
    """Persisted state from fit(). Pass to score()."""
    cfg: StrataConfig
    host_roles: pd.DataFrame
    peer_baselines: Dict[str, pd.DataFrame]
    freq_model: object
    role_gap_thresholds: Dict[str, float]
    cmdline_vectorizer: Optional[object] = None
    baseline_commands: Optional[pd.Series] = None
    baseline_cmd_matrix: Optional[object] = None  # pre-built sparse matrix for TF-IDF
    historical_shrinkage: Optional[pd.DataFrame] = None
    role_pair_baselines: Optional[Dict[str, Dict]] = None


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class StrataPipeline:
    """
    Orchestrates fit/score for the STRATA-E unified behavioral analytic.

    Typical usage:
        cfg = StrataConfig()
        pipe = StrataPipeline(cfg)

        # Train on historical benign data
        fitted = pipe.fit(df_baseline)

        # Score a new window
        art = pipe.score(df_new, fitted)
        print(art.triage.head(20))

    Ablation study usage:
        for ablation in [AblationConfig.full_pipeline(),
                         AblationConfig.sequence_only(),
                         AblationConfig.no_shrinkage()]:
            cfg = StrataConfig(ablation=ablation)
            pipe = StrataPipeline(cfg)
            fitted = pipe.fit(df_baseline)
            art = pipe.score(df_new, fitted)
            evaluate(art.triage, labels, condition=ablation)
    """

    def __init__(self, cfg: Optional[StrataConfig] = None):
        self.cfg = cfg or StrataConfig()

    # ------------------------------------------------------------------
    # Stage 1: Fit (learn all baselines from historical data)
    # ------------------------------------------------------------------

    def fit(self, raw_df: pd.DataFrame) -> FittedArtifacts:
        """
        Fit all baseline components from historical/benign data.
        Returns FittedArtifacts to be passed to score().
        """
        cfg = self.cfg
        logger.info("fit(): normalizing schema")
        events = normalize_schema(raw_df, cfg)
        validate_schema(events)

        logger.info("fit(): building tokens")
        events = build_tokens(events)

        # Role inference
        logger.info("fit(): computing rate features + role inference")
        rates = compute_rate_features(events, cfg)
        role_feats = build_role_features(rates, cfg)
        host_roles = infer_roles(role_feats, cfg)
        events = events.merge(host_roles, on="host", how="left")

        # Fix 3: Adaptive session gap per role
        if cfg.ablation.use_adaptive_tau_gap:
            logger.info("fit(): fitting adaptive tau_gap per role")
            role_gap_thresholds = build_role_gap_thresholds(events, cfg)
        else:
            role_gap_thresholds = {"default": float(cfg.time.session_gap_seconds)}

        # Sessionization + transition counting
        logger.info("fit(): sessionizing and building transitions")
        events = assign_sessions(events, cfg, role_gap_thresholds)
        events = bucket_deltas(events, cfg)
        token_col = f"token_{cfg.token_resolution}"
        trans = build_transition_counts(events, cfg, level=token_col)

        # Peer baselines
        if cfg.ablation.use_role_baselining:
            logger.info("fit(): fitting peer baselines")
            peer_baselines = fit_peer_baselines(trans, host_roles, cfg)
        else:
            logger.info("fit(): role baselining disabled — using global baseline")
            host_roles_flat = host_roles.copy()
            host_roles_flat["role_id"] = "global"
            peer_baselines = fit_peer_baselines(trans, host_roles_flat, cfg)

        # Frequency model
        logger.info("fit(): fitting frequency model")
        freq_model = fit_frequency_model(rates, cfg)

        # Fix 6: Cmdline TF-IDF vectorizer
        cmdline_vectorizer = None
        baseline_commands = None
        baseline_cmd_matrix = None
        if cfg.ablation.use_cmdline_embeddings and "cmdline" in events.columns:
            logger.info("fit(): fitting cmdline TF-IDF vectorizer")
            baseline_commands = events["cmdline"].dropna()
            cmdline_vectorizer = build_cmdline_vectorizer(
                baseline_commands,
                max_features=cfg.scoring.tfidf_max_features,
            )
            # Pre-build compact baseline matrix (dedup + sample + transform)
            logger.info("fit(): building baseline command matrix")
            baseline_cmd_matrix = build_baseline_matrix(
                baseline_commands,
                cmdline_vectorizer,
                max_samples=cfg.scoring.tfidf_baseline_samples,
            )

        # Fix 5: Initial shrinkage baseline
        host_event_counts = events.groupby("host").size().to_dict()
        historical_shrinkage = compute_shrinkage_weights(host_event_counts, cfg.baseline.dirichlet_kappa)

        # Learn role-conditioned pair baselines (which pairs are routine per role)
        logger.info("fit(): learning role pair baselines")
        role_pair_baselines = compute_role_pair_baselines(events, cfg)
        if role_pair_baselines:
            for rid, discounts in role_pair_baselines.items():
                logger.info("  role %s: %d expected pairs learned", rid, len(discounts))

        return FittedArtifacts(
            cfg=cfg,
            host_roles=host_roles,
            peer_baselines=peer_baselines,
            freq_model=freq_model,
            role_gap_thresholds=role_gap_thresholds,
            cmdline_vectorizer=cmdline_vectorizer,
            baseline_commands=baseline_commands,
            baseline_cmd_matrix=baseline_cmd_matrix,
            historical_shrinkage=historical_shrinkage,
            role_pair_baselines=role_pair_baselines,
        )

    # ------------------------------------------------------------------
    # Stage 2: Score (apply fitted baselines to a new window)
    # ------------------------------------------------------------------

    def score(
        self,
        raw_df: pd.DataFrame,
        fitted: FittedArtifacts,
        prior_window_df: Optional[pd.DataFrame] = None,
        learned_weights: Optional[np.ndarray] = None,
    ) -> StrataArtifacts:
        """
        Score a new window of events against fitted baselines.

        prior_window_df: optional previous window for drift channel computation.
        learned_weights: optional weights from learn_fusion_weights() — if provided,
                         overrides Borda fusion with supervised weighted fusion.
        """
        cfg = self.cfg
        timings: Dict[str, float] = {}

        # --- Preprocessing ---
        t0 = time.perf_counter()
        logger.info("score(): normalizing schema")
        events = normalize_schema(raw_df, cfg)
        validate_schema(events)
        events = build_tokens(events)

        # Attach roles
        events = events.merge(fitted.host_roles, on="host", how="left")
        events["role_id"] = events["role_id"].fillna("default")

        # Sessionize + bucket
        events = assign_sessions(events, cfg, fitted.role_gap_thresholds)
        events = bucket_deltas(events, cfg)

        # Transition counts for scoring window
        token_col = f"token_{cfg.token_resolution}"
        trans = build_transition_counts(events, cfg, level=token_col)

        # Rate features
        rates = compute_rate_features(events, cfg)
        timings["preprocessing"] = time.perf_counter() - t0

        # --- Sequence channel ---
        t0 = time.perf_counter()
        logger.info("score(): computing sequence divergence")
        seq_scores = score_sequence_divergence(
            trans, fitted.host_roles, fitted.peer_baselines, cfg
        )
        timings["sequence_divergence"] = time.perf_counter() - t0

        # JSD calibration: add z-score, empirical p-value, and percentile per host
        # (paper Eq. 10-11, updated to distribution-free p-values)
        t0 = time.perf_counter()
        if cfg.ablation.use_jsd_calibration and fitted.peer_baselines:
            # Clear cache at start of each scoring run (baselines may differ)
            clear_bootstrap_cache()
            host_n    = trans.groupby("host")["count"].sum().to_dict()
            host_role = fitted.host_roles.set_index("host")["role_id"].to_dict()
            calibrated_rows = []
            for _, row in seq_scores.iterrows():
                role_key     = str(host_role.get(row["host"], "default"))
                baseline_ref = fitted.peer_baselines.get(
                    role_key, next(iter(fitted.peer_baselines.values()))
                )
                n_h       = host_n.get(row["host"], 50)
                null = calibrate_jsd_null_distribution(
                    baseline_ref, n_h, cfg, role_id=role_key,
                )
                observed  = row["S_seq"]
                z         = null.z_score(observed)
                p_value   = null.empirical_pvalue(observed)
                percentile = null.empirical_percentile(observed)
                calibrated_rows.append({
                    "host":             row["host"],
                    "S_seq_z":          round(float(z), 3),
                    "S_seq_pvalue":     round(p_value, 4),
                    "S_seq_percentile": round(percentile, 1),
                })
            if calibrated_rows:
                z_df = pd.DataFrame(calibrated_rows)
                seq_scores = seq_scores.merge(z_df, on="host", how="left")
        timings["jsd_calibration"] = time.perf_counter() - t0

        # --- Frequency channel ---
        t0 = time.perf_counter()
        logger.info("score(): computing frequency scores")
        freq_scores = score_frequency(rates, fitted.freq_model)
        timings["frequency_channel"] = time.perf_counter() - t0

        # --- Critical event pair correlation (feeds context channel) ---
        t0 = time.perf_counter()
        pairs = correlate_critical_events_by_host(
            events, cfg,
            role_pair_baselines=fitted.role_pair_baselines,
        )
        pair_stats = compute_pair_stats(pairs) if not pairs.empty else pd.DataFrame()
        timings["pair_correlation"] = time.perf_counter() - t0

        # --- Context channel ---
        t0 = time.perf_counter()
        logger.info("score(): computing context scores")
        ctx_scores = score_context(
            events, cfg,
            cmdline_vectorizer=fitted.cmdline_vectorizer,
            baseline_commands=fitted.baseline_commands,
            baseline_cmd_matrix=fitted.baseline_cmd_matrix,
            pair_stats=pair_stats if not pair_stats.empty else None,
        )
        timings["context_channel"] = time.perf_counter() - t0

        # --- Drift channel ---
        t0 = time.perf_counter()
        logger.info("score(): computing drift scores")
        prior_trans = None
        if prior_window_df is not None and cfg.ablation.use_drift_channel:
            prior_norm = normalize_schema(prior_window_df, cfg)
            prior_norm = build_tokens(prior_norm)
            prior_norm = assign_sessions(prior_norm, cfg, fitted.role_gap_thresholds)
            prior_norm = bucket_deltas(prior_norm, cfg)
            prior_trans = build_transition_counts(prior_norm, cfg, level=token_col)

        drift_scores = score_drift(trans, prior_trans, fitted.host_roles, cfg)
        timings["drift_channel"] = time.perf_counter() - t0

        # --- Fix 5: Shrinkage anomaly ---
        shrinkage_df = None
        if cfg.ablation.use_shrinkage_anomaly and fitted.historical_shrinkage is not None:
            host_event_counts = events.groupby("host").size().to_dict()
            current_shrinkage = compute_shrinkage_weights(
                host_event_counts, cfg.baseline.dirichlet_kappa
            )
            shrinkage_df = detect_shrinkage_anomalies(
                current_shrinkage, fitted.historical_shrinkage
            )

        # --- Fusion ---
        t0 = time.perf_counter()
        logger.info("score(): fusing channels")
        fused = fuse_scores(
            seq_scores, freq_scores, ctx_scores, drift_scores,
            cfg, learned_weights=learned_weights
        )

        # --- Triage table ---
        triage = build_ranked_triage(fused, pair_stats if not pair_stats.empty else None)

        # Append shrinkage evasion signal to triage
        if shrinkage_df is not None:
            triage = triage.merge(
                shrinkage_df[["host", "shrinkage_weight", "shrinkage_delta", "evasion_signal"]],
                on="host", how="left",
            )
        timings["fusion_and_triage"] = time.perf_counter() - t0

        # Log timing summary
        total = sum(timings.values())
        logger.info("score() stage timings (%.1fs total):", total)
        for stage, elapsed in sorted(timings.items(), key=lambda x: -x[1]):
            pct = 100.0 * elapsed / (total + 1e-9)
            logger.info("  %-25s %6.2fs  (%4.1f%%)", stage, elapsed, pct)

        return StrataArtifacts(
            events=events,
            host_roles=fitted.host_roles,
            peer_baselines=fitted.peer_baselines,
            role_gap_thresholds=fitted.role_gap_thresholds,
            transition_counts=trans,
            rate_features=rates,
            pair_stats=pair_stats if not pair_stats.empty else None,
            seq_scores=seq_scores,
            freq_scores=freq_scores,
            ctx_scores=ctx_scores,
            drift_scores=drift_scores,
            shrinkage_df=shrinkage_df,
            triage=triage,
            stage_timings=timings,
        )

    # ------------------------------------------------------------------
    # Convenience: fit + score in one call (for notebooks / quick runs)
    # ------------------------------------------------------------------

    def fit_score(
        self,
        raw_df: pd.DataFrame,
        prior_window_df: Optional[pd.DataFrame] = None,
    ) -> StrataArtifacts:
        """Fit on the same data window and immediately score it. For exploration only."""
        fitted = self.fit(raw_df)
        self._last_fitted = fitted  # store for CLI access
        return self.score(raw_df, fitted, prior_window_df=prior_window_df)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="STRATA-E unified pipeline")
    parser.add_argument("--input",   required=True, help="Path to Sysmon CSV/JSONL")
    parser.add_argument("--output",  default="output", help="Output directory")
    parser.add_argument("--config",  default=None, help="Path to config JSON (optional)")
    parser.add_argument("--ablation", default="full",
                        choices=["full", "sequence_only", "no_shrinkage", "no_role_baselining", "no_drift"],
                        help="Ablation condition")
    parser.add_argument("--fast", action="store_true",
                        help="Dev mode: reduce bootstrap samples, IF trees, TF-IDF features for ~3-10× speedup")
    parser.add_argument("--report",  action="store_true", help="Generate self-contained HTML report")
    parser.add_argument("--no-plots", action="store_true", help="Suppress matplotlib visualisations")
    parser.add_argument("--browser", action="store_true", help="Open HTML report in browser when done")
    args = parser.parse_args()

    if args.config:
        cfg = StrataConfig.from_json(args.config)
    else:
        ablation_map = {
            "full":              AblationConfig.full_pipeline,
            "sequence_only":     AblationConfig.sequence_only,
            "no_shrinkage":      AblationConfig.no_shrinkage,
            "no_role_baselining": AblationConfig.no_role_baselining,
            "no_drift":          AblationConfig.no_drift,
        }
        cfg = StrataConfig(ablation=ablation_map[args.ablation]())

    if args.fast:
        cfg.apply_fast_preset()
        logger.info("--fast mode: bootstrap=%d, IF trees=%d, TF-IDF features=%d",
                     cfg.baseline.bootstrap_samples,
                     cfg.scoring.iforest_n_estimators,
                     cfg.scoring.tfidf_max_features)

    cfg.io.input_path = Path(args.input)
    cfg.io.output_dir = Path(args.output)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.input.endswith(".parquet") or args.input.endswith(".pq"):
        raw = pd.read_parquet(args.input)
    elif args.input.endswith(".csv") or args.input.endswith(".csv.gz"):
        raw = pd.read_csv(args.input, low_memory=False)
    else:
        raw = pd.read_json(args.input, lines=True)

    if args.report:
        from .report import ReportContext
        with ReportContext(
            output_dir=args.output,
            open_browser=args.browser,
        ) as report:
            pipe = StrataPipeline(cfg)
            art  = pipe.fit_score(raw)
            report.finalise(art)
    else:
        pipe = StrataPipeline(cfg)
        art = pipe.fit_score(raw)

    if art.triage is not None:
        out_path = Path(args.output) / "triage.csv"
        art.triage.to_csv(out_path, index=False)
        logger.info("Triage written to %s", out_path)
        if not args.report:
            print(art.triage.head(20).to_string())


if __name__ == "__main__":
    main()
