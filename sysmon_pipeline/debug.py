"""
STRATA-E Debug Pipeline
========================
A separate pipeline for testing, validation, and stage-by-stage debugging.
Wraps the production StrataPipeline without modifying it.

Use this in notebooks and during development to:
  1. Validate data compatibility before committing to a full run
  2. Run each pipeline stage independently and inspect intermediate outputs
  3. Score individual hosts without processing the full fleet
  4. Diagnose where things break or slow down on new datasets

The production pipeline (pipeline.py) stays clean and untouched.

Typical notebook workflow::

    from sysmon_pipeline.debug import StrataDebugPipeline
    from sysmon_pipeline import StrataConfig

    dbg = StrataDebugPipeline(StrataConfig.fast())

    # Step 1: Check data compatibility (fast, catches schema issues)
    report = dbg.preprocess_check(raw_df)

    # Step 2: Run stages one at a time, inspect each
    events = dbg.preprocess(raw_df)
    events.head()

    trans, rates = dbg.build_features(events)
    trans.groupby("host").size()

    fitted = dbg.fit_baselines(events, trans, rates)
    fitted.host_roles

    # Step 3: Score individual channels
    seq = dbg.score_sequence(trans, fitted)
    seq.sort_values("S_seq", ascending=False).head(10)

    freq = dbg.score_frequency(rates, fitted)
    ctx = dbg.score_context(events, fitted)

    # Step 4: Fuse and triage
    triage = dbg.fuse_and_triage(seq, freq, ctx)

    # Or: score only specific hosts through the full pipeline
    art = dbg.score_hosts(raw_df, fitted, hosts=["SCRANTON", "NASHUA"])
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import StrataConfig
from .pipeline import StrataPipeline, StrataArtifacts, FittedArtifacts
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
    fit_peer_baselines, score_sequence_divergence,
    compute_shrinkage_weights,
    calibrate_jsd_null_distribution, clear_bootstrap_cache,
)
from .scoring import (
    fit_frequency_model, score_frequency,
    score_context as _score_context,
    fuse_scores, build_ranked_triage,
    build_cmdline_vectorizer, build_baseline_matrix,
)

logger = logging.getLogger("strata.debug")


class StrataDebugPipeline:
    """
    Debug and testing wrapper around StrataPipeline.

    Provides stage-by-stage execution, data validation, single-host
    scoring, and detailed diagnostics — all without modifying the
    production pipeline.
    """

    def __init__(self, cfg: Optional[StrataConfig] = None):
        self.cfg = cfg or StrataConfig()
        self._pipe = StrataPipeline(self.cfg)

    # ------------------------------------------------------------------
    # Pass-through to production pipeline
    # ------------------------------------------------------------------

    def fit(self, raw_df: pd.DataFrame) -> FittedArtifacts:
        """Delegate to production pipeline fit()."""
        return self._pipe.fit(raw_df)

    def score(self, raw_df: pd.DataFrame, fitted: FittedArtifacts,
              **kwargs) -> StrataArtifacts:
        """Delegate to production pipeline score()."""
        return self._pipe.score(raw_df, fitted, **kwargs)

    def fit_score(self, raw_df: pd.DataFrame, **kwargs) -> StrataArtifacts:
        """Delegate to production pipeline fit_score()."""
        return self._pipe.fit_score(raw_df, **kwargs)

    # ------------------------------------------------------------------
    # Stage 1: Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, raw_df: pd.DataFrame,
                   fitted: Optional[FittedArtifacts] = None) -> pd.DataFrame:
        """
        Run ingest → schema normalization → tokenization → role assignment
        → sessionization → delta bucketing.

        If ``fitted`` is provided, uses its roles and gap thresholds
        (scoring path). Otherwise infers roles from the data (fit path).

        Returns the fully preprocessed events DataFrame.
        """
        cfg = self.cfg
        t0 = time.perf_counter()

        events = normalize_schema(raw_df, cfg)
        validate_schema(events)
        events = build_tokens(events)

        if fitted is not None:
            events = events.merge(fitted.host_roles, on="host", how="left")
            events["role_id"] = events["role_id"].fillna("default")
            role_gaps = fitted.role_gap_thresholds
        else:
            rates = compute_rate_features(events, cfg)
            role_feats = build_role_features(rates, cfg)
            host_roles = infer_roles(role_feats, cfg)
            events = events.merge(host_roles, on="host", how="left")
            if cfg.ablation.use_adaptive_tau_gap:
                role_gaps = build_role_gap_thresholds(events, cfg)
            else:
                role_gaps = {"default": float(cfg.time.session_gap_seconds)}

        events = assign_sessions(events, cfg, role_gaps)
        events = bucket_deltas(events, cfg)

        elapsed = time.perf_counter() - t0
        logger.info("preprocess(): %d events, %d hosts, %.2fs",
                     len(events), events["host"].nunique(), elapsed)
        return events

    # ------------------------------------------------------------------
    # Stage 2: Feature construction
    # ------------------------------------------------------------------

    def build_features(self, events: pd.DataFrame
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build transition counts and rate features from preprocessed events.

        Returns ``(transition_counts, rate_features)``.
        """
        cfg = self.cfg
        t0 = time.perf_counter()

        token_col = f"token_{cfg.token_resolution}"
        trans = build_transition_counts(events, cfg, level=token_col)
        rates = compute_rate_features(events, cfg)

        elapsed = time.perf_counter() - t0
        logger.info("build_features(): %d transitions, %d hosts, "
                     "%d rate cols, %.2fs",
                     len(trans), trans["host"].nunique(),
                     rates.shape[1] - 1, elapsed)
        return trans, rates

    # ------------------------------------------------------------------
    # Stage 3: Baseline fitting
    # ------------------------------------------------------------------

    def fit_baselines(self, events: pd.DataFrame,
                      trans: pd.DataFrame,
                      rates: pd.DataFrame) -> FittedArtifacts:
        """
        Fit peer baselines, frequency model, TF-IDF vectorizer, and
        shrinkage baseline from pre-built features.

        Separated from preprocessing so you can build features once
        and re-fit baselines with different configs.
        """
        cfg = self.cfg
        t0 = time.perf_counter()

        host_roles = events[["host", "role_id"]].drop_duplicates("host")

        if cfg.ablation.use_role_baselining:
            peer_baselines = fit_peer_baselines(trans, host_roles, cfg)
        else:
            hr_flat = host_roles.copy()
            hr_flat["role_id"] = "global"
            peer_baselines = fit_peer_baselines(trans, hr_flat, cfg)

        if cfg.ablation.use_adaptive_tau_gap:
            role_gap_thresholds = build_role_gap_thresholds(events, cfg)
        else:
            role_gap_thresholds = {"default": float(cfg.time.session_gap_seconds)}

        freq_model = fit_frequency_model(rates, cfg)

        cmdline_vectorizer = None
        baseline_commands = None
        baseline_cmd_matrix = None
        if cfg.ablation.use_cmdline_embeddings and "cmdline" in events.columns:
            baseline_commands = events["cmdline"].dropna()
            cmdline_vectorizer = build_cmdline_vectorizer(
                baseline_commands,
                max_features=cfg.scoring.tfidf_max_features,
            )
            baseline_cmd_matrix = build_baseline_matrix(
                baseline_commands,
                cmdline_vectorizer,
                max_samples=cfg.scoring.tfidf_baseline_samples,
            )

        host_event_counts = events.groupby("host").size().to_dict()
        historical_shrinkage = compute_shrinkage_weights(
            host_event_counts, cfg.baseline.dirichlet_kappa
        )

        # Learn role-conditioned pair baselines
        role_pair_baselines = compute_role_pair_baselines(events, cfg)

        elapsed = time.perf_counter() - t0
        logger.info("fit_baselines(): %d roles, %d baselines, %.2fs",
                     host_roles["role_id"].nunique(),
                     len(peer_baselines), elapsed)

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
    # Stage 4: Individual channel scoring
    # ------------------------------------------------------------------

    def score_sequence(self, trans: pd.DataFrame,
                       fitted: FittedArtifacts) -> pd.DataFrame:
        """Score just the sequence channel with JSD calibration."""
        cfg = self.cfg
        t0 = time.perf_counter()

        seq_scores = score_sequence_divergence(
            trans, fitted.host_roles, fitted.peer_baselines, cfg
        )

        if cfg.ablation.use_jsd_calibration and fitted.peer_baselines:
            clear_bootstrap_cache()
            host_n = trans.groupby("host")["count"].sum().to_dict()
            host_role = fitted.host_roles.set_index("host")["role_id"].to_dict()
            rows = []
            for _, row in seq_scores.iterrows():
                role_key = str(host_role.get(row["host"], "default"))
                baseline_ref = fitted.peer_baselines.get(
                    role_key, next(iter(fitted.peer_baselines.values()))
                )
                null = calibrate_jsd_null_distribution(
                    baseline_ref, host_n.get(row["host"], 50),
                    cfg, role_id=role_key,
                )
                observed = row["S_seq"]
                rows.append({
                    "host":             row["host"],
                    "S_seq_z":          round(null.z_score(observed), 3),
                    "S_seq_pvalue":     round(null.empirical_pvalue(observed), 4),
                    "S_seq_percentile": round(null.empirical_percentile(observed), 1),
                })
            if rows:
                seq_scores = seq_scores.merge(
                    pd.DataFrame(rows), on="host", how="left"
                )

        elapsed = time.perf_counter() - t0
        logger.info("score_sequence(): %d hosts, %.2fs", len(seq_scores), elapsed)
        return seq_scores

    def score_frequency(self, rates: pd.DataFrame,
                        fitted: FittedArtifacts) -> pd.DataFrame:
        """Score just the frequency channel."""
        t0 = time.perf_counter()
        result = score_frequency(rates, fitted.freq_model)
        elapsed = time.perf_counter() - t0
        logger.info("score_frequency(): %d hosts, %.2fs", len(result), elapsed)
        return result

    def score_context(self, events: pd.DataFrame,
                      fitted: FittedArtifacts,
                      pair_stats: Optional[pd.DataFrame] = None
                      ) -> pd.DataFrame:
        """
        Score just the context channel.

        Computes pair correlation internally if ``pair_stats`` is None.
        """
        cfg = self.cfg
        t0 = time.perf_counter()

        if pair_stats is None:
            pairs = correlate_critical_events_by_host(
                events, cfg,
                role_pair_baselines=fitted.role_pair_baselines,
            )
            pair_stats = compute_pair_stats(pairs) if not pairs.empty else None

        result = _score_context(
            events, cfg,
            cmdline_vectorizer=fitted.cmdline_vectorizer,
            baseline_commands=fitted.baseline_commands,
            baseline_cmd_matrix=fitted.baseline_cmd_matrix,
            pair_stats=pair_stats,
        )

        elapsed = time.perf_counter() - t0
        logger.info("score_context(): %d hosts, %.2fs", len(result), elapsed)
        return result

    # ------------------------------------------------------------------
    # Stage 5: Fusion
    # ------------------------------------------------------------------

    def fuse_and_triage(
        self,
        seq_scores: pd.DataFrame,
        freq_scores: pd.DataFrame,
        ctx_scores: pd.DataFrame,
        drift_scores: Optional[pd.DataFrame] = None,
        pair_stats: Optional[pd.DataFrame] = None,
        learned_weights: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Fuse channel scores and build the ranked triage table.

        Creates zero-filled drift scores if ``drift_scores`` is None.
        """
        cfg = self.cfg
        t0 = time.perf_counter()

        if drift_scores is None:
            hosts = set(seq_scores["host"])
            drift_scores = pd.DataFrame({"host": list(hosts), "S_drift": 0.0})

        fused = fuse_scores(
            seq_scores, freq_scores, ctx_scores, drift_scores,
            cfg, learned_weights=learned_weights,
        )
        triage = build_ranked_triage(fused, pair_stats)

        elapsed = time.perf_counter() - t0
        n_pass = triage["gate_pass"].sum() if "gate_pass" in triage.columns else 0
        logger.info("fuse_and_triage(): %d hosts, %d gate_pass, %.2fs",
                     len(triage), n_pass, elapsed)
        return triage

    # ------------------------------------------------------------------
    # Single-host scoring
    # ------------------------------------------------------------------

    def score_hosts(
        self,
        raw_df: pd.DataFrame,
        fitted: FittedArtifacts,
        hosts: Sequence[str],
        prior_window_df: Optional[pd.DataFrame] = None,
    ) -> StrataArtifacts:
        """
        Score only specific hosts through the full pipeline.

        Preprocessing runs on all data (so baselines see the full fleet),
        but all scoring stages are filtered to just the listed hosts.
        Useful for debugging individual hosts on large datasets.

        Usage::

            art = dbg.score_hosts(raw_df, fitted, hosts=["SCRANTON"])
            print(art.triage)
            print(art.seq_scores)
        """
        cfg = self.cfg
        t0 = time.perf_counter()

        # Full preprocessing (baselines need all hosts)
        events = normalize_schema(raw_df, cfg)
        validate_schema(events)
        events = build_tokens(events)
        events = events.merge(fitted.host_roles, on="host", how="left")
        events["role_id"] = events["role_id"].fillna("default")
        events = assign_sessions(events, cfg, fitted.role_gap_thresholds)
        events = bucket_deltas(events, cfg)

        token_col = f"token_{cfg.token_resolution}"
        trans = build_transition_counts(events, cfg, level=token_col)
        rates = compute_rate_features(events, cfg)

        # Filter to requested hosts
        host_set = set(hosts)
        n_all = events["host"].nunique()
        events = events[events["host"].isin(host_set)].copy()
        trans = trans[trans["host"].isin(host_set)].copy()
        rates = rates[rates["host"].isin(host_set)].copy()
        n_filtered = events["host"].nunique()
        logger.info("score_hosts(): filtered %d → %d hosts", n_all, n_filtered)

        # Score each channel on filtered data
        seq_scores = self.score_sequence(trans, fitted)
        freq_scores = self.score_frequency(rates, fitted)
        ctx_scores = self.score_context(events, fitted)

        # Drift (if prior window provided)
        from .divergence import score_drift
        drift_scores = None
        if prior_window_df is not None and cfg.ablation.use_drift_channel:
            prior_norm = normalize_schema(prior_window_df, cfg)
            prior_norm = build_tokens(prior_norm)
            prior_norm = assign_sessions(prior_norm, cfg, fitted.role_gap_thresholds)
            prior_norm = bucket_deltas(prior_norm, cfg)
            prior_trans = build_transition_counts(prior_norm, cfg, level=token_col)
            drift_scores = score_drift(trans, prior_trans, fitted.host_roles, cfg)
        if drift_scores is None:
            drift_scores = pd.DataFrame({"host": list(host_set), "S_drift": 0.0})

        # Fuse
        pairs = correlate_critical_events_by_host(
            events, cfg,
            role_pair_baselines=fitted.role_pair_baselines,
        )
        pair_stats = compute_pair_stats(pairs) if not pairs.empty else None

        fused = fuse_scores(seq_scores, freq_scores, ctx_scores,
                            drift_scores, cfg)
        triage = build_ranked_triage(fused, pair_stats)

        elapsed = time.perf_counter() - t0
        logger.info("score_hosts(): %d hosts scored in %.2fs", n_filtered, elapsed)

        return StrataArtifacts(
            events=events,
            host_roles=fitted.host_roles,
            peer_baselines=fitted.peer_baselines,
            role_gap_thresholds=fitted.role_gap_thresholds,
            transition_counts=trans,
            rate_features=rates,
            pair_stats=pair_stats,
            seq_scores=seq_scores,
            freq_scores=freq_scores,
            ctx_scores=ctx_scores,
            drift_scores=drift_scores,
            triage=triage,
        )

    # ------------------------------------------------------------------
    # Data validation (dry-run diagnostic)
    # ------------------------------------------------------------------

    def preprocess_check(self, raw_df: pd.DataFrame) -> dict:
        """
        Run preprocessing only and print a diagnostic summary.

        Validates that data is compatible with the pipeline before
        committing to a full run. Catches schema detection failures,
        timestamp parsing issues, missing columns, and degenerate data.

        Returns a dict with status, event/host/role counts, column
        detection results, distribution stats, and actionable warnings.
        """
        cfg = self.cfg
        t0 = time.perf_counter()
        warnings_list: List[str] = []

        # --- Schema normalization ---
        try:
            events = normalize_schema(raw_df, cfg)
            validate_schema(events)
        except Exception as e:
            result = {
                "status": "error",
                "error": str(e),
                "elapsed_s": time.perf_counter() - t0,
                "n_events": len(raw_df),
                "columns_available": list(raw_df.columns),
            }
            print(f"\n  STRATA-E Preprocessing Check")
            print(f"  {'='*50}")
            print(f"  Status:  ERROR")
            print(f"  {e}")
            print(f"  Available columns: {list(raw_df.columns)}")
            return result

        events = build_tokens(events)

        # --- Role inference ---
        rates = compute_rate_features(events, cfg)
        role_feats = build_role_features(rates, cfg)
        host_roles = infer_roles(role_feats, cfg)
        events = events.merge(host_roles, on="host", how="left")

        # --- Sessionization ---
        if cfg.ablation.use_adaptive_tau_gap:
            role_gaps = build_role_gap_thresholds(events, cfg)
        else:
            role_gaps = {"default": float(cfg.time.session_gap_seconds)}
        events = assign_sessions(events, cfg, role_gaps)

        elapsed = time.perf_counter() - t0

        # --- Column detection ---
        detected = {}
        missing = []
        for canon, candidates in [
            ("ts", cfg.io.timestamp_cols),
            ("host", cfg.io.host_cols),
            ("event_id", cfg.io.event_id_cols),
            ("image", cfg.io.image_cols),
            ("parent_image", cfg.io.parent_image_cols),
            ("cmdline", cfg.io.cmdline_cols),
            ("user", cfg.io.user_cols),
        ]:
            found = None
            for c in candidates:
                if c in raw_df.columns:
                    found = c
                    break
            if found:
                detected[canon] = found
            else:
                missing.append(canon)

        # --- Events per host ---
        host_counts = events.groupby("host").size()
        epc = {
            "min": int(host_counts.min()),
            "median": int(host_counts.median()),
            "max": int(host_counts.max()),
            "mean": round(float(host_counts.mean()), 1),
        }

        # --- Warnings ---
        sparse_hosts = int((host_counts < cfg.baseline.min_events_per_host).sum())
        if sparse_hosts > 0:
            pct = 100 * sparse_hosts / len(host_counts)
            warnings_list.append(
                f"{sparse_hosts} hosts ({pct:.0f}%) have < "
                f"{cfg.baseline.min_events_per_host} events — heavily "
                f"shrinkage-dominated"
            )

        role_dist = host_roles["role_id"].value_counts().to_dict()
        if len(role_dist) == 1:
            warnings_list.append(
                "Only 1 role inferred — peer-role baselining equivalent "
                "to global baselining (H2 not testable)"
            )

        sess_counts = events.groupby("session_id").size()
        sess_stats = {
            "n_sessions": int(sess_counts.count()),
            "median_length": int(sess_counts.median()),
            "max_length": int(sess_counts.max()),
        }

        token_dist = events["token_coarse"].value_counts().to_dict()

        if "cmdline" not in events.columns or events["cmdline"].isna().all():
            warnings_list.append(
                "No command line data — TF-IDF novelty scoring disabled"
            )

        ts_min = events["ts"].min()
        ts_max = events["ts"].max()
        time_span = (ts_max - ts_min).total_seconds() / 3600
        if time_span < 1.0:
            warnings_list.append(
                f"Time span only {time_span:.1f}h — drift channel will "
                f"have minimal signal"
            )

        result = {
            "status": "ok",
            "elapsed_s": round(elapsed, 2),
            "n_events": len(events),
            "n_hosts": int(events["host"].nunique()),
            "n_roles": len(role_dist),
            "role_distribution": role_dist,
            "time_range": (str(ts_min), str(ts_max)),
            "time_span_hours": round(time_span, 1),
            "columns_detected": detected,
            "columns_missing": missing,
            "token_distribution": token_dist,
            "events_per_host": epc,
            "session_stats": sess_stats,
            "warnings": warnings_list,
        }

        # --- Print summary ---
        print(f"\n  STRATA-E Preprocessing Check")
        print(f"  {'='*50}")
        print(f"  Status:       {result['status']}")
        print(f"  Events:       {result['n_events']:,}")
        print(f"  Hosts:        {result['n_hosts']}")
        print(f"  Roles:        {result['n_roles']}  {role_dist}")
        print(f"  Time span:    {result['time_span_hours']:.1f} hours")
        print(f"  Time range:   {ts_min} → {ts_max}")
        print(f"  Sessions:     {sess_stats['n_sessions']:,}  "
              f"(median {sess_stats['median_length']} events, "
              f"max {sess_stats['max_length']})")
        print(f"  Events/host:  min={epc['min']}, median={epc['median']}, "
              f"max={epc['max']}, mean={epc['mean']}")
        print(f"  Columns:      {len(detected)} detected, "
              f"{len(missing)} missing {missing if missing else ''}")
        print(f"  Elapsed:      {elapsed:.2f}s")
        if warnings_list:
            print(f"\n  Warnings:")
            for w in warnings_list:
                print(f"    ⚠  {w}")
        else:
            print(f"\n  No warnings — data looks good for a full run.")
        print()

        return result
