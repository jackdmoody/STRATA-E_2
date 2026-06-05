"""
Multi-channel scoring and fusion.
====================================
Best of:
  - pipeline_updated scoring.py: all four channels, gating, fuse_scores()
  - v12_modular scoring.py:      compute_host_markov_scores, build_ranked_triage

Adds:
  - Fix 1: Borda rank fusion + corroboration gate (replaces static weighted sum)
  - Fix 1: learn_fusion_weights() for supervised weight learning from injection data
  - Fix 6: cmdline TF-IDF novelty scoring for context channel
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize, StandardScaler

from .config import StrataConfig
from .divergence import compute_seq_drift_covariance, check_channel_correlation


# ---------------------------------------------------------------------------
# Frequency channel
# ---------------------------------------------------------------------------

# Cmdline-content flag rates are owned by the context channel (component 2).
# They are excluded from the frequency channel's feature matrix so S_freq
# stays purely volumetric — otherwise S_freq and S_ctx would share input
# signal and "corroborate" each other by construction.
# Kept in sync with cfg.scoring.freq_exclude_cols; this module-level default
# guarantees fit and score always drop the SAME columns.
FREQ_EXCLUDE_COLS = ("has_encoded_rate", "has_download_cradle_rate", "has_bypass_rate")


def freq_feature_frame(df_rates: pd.DataFrame) -> pd.DataFrame:
    """Volumetric-only feature FRAME for the frequency channel — the single
    source of truth for both the model matrix and the feature names. Any
    consumer that names features (SHAP, debugging) MUST use this, not the
    raw rate_features frame: the model is fit on the reduced column set, so
    naming from the full frame misaligns every SHAP value."""
    return (
        df_rates
        .drop(columns=["host", *FREQ_EXCLUDE_COLS], errors="ignore")
        .fillna(0.0)
    )


def _freq_feature_matrix(df_rates: pd.DataFrame) -> np.ndarray:
    """Volumetric-only feature matrix for the frequency channel."""
    return freq_feature_frame(df_rates).to_numpy()


def fit_frequency_model(df_rates: pd.DataFrame, cfg: StrataConfig) -> IsolationForest:
    """Fit IsolationForest on per-host volumetric rate features."""
    X = _freq_feature_matrix(df_rates)
    model = IsolationForest(
        n_estimators=cfg.scoring.iforest_n_estimators,
        contamination=cfg.scoring.iforest_contamination,
        random_state=cfg.scoring.random_seed,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def score_frequency(df_rates: pd.DataFrame, model: IsolationForest) -> pd.DataFrame:
    """Return S_freq in [0,1] where higher = more anomalous."""
    X = _freq_feature_matrix(df_rates)
    normality = model.decision_function(X)
    ranks = pd.Series(normality).rank(pct=True)
    S_freq = 1.0 - ranks.to_numpy()
    return pd.DataFrame({"host": df_rates["host"].values, "S_freq": S_freq})


# ---------------------------------------------------------------------------
# Context channel
# ---------------------------------------------------------------------------

def build_cmdline_vectorizer(
    baseline_commands: pd.Series,
    ngram_range: tuple = (1, 3),
    max_features: int = 5000,
) -> TfidfVectorizer:
    """
    Fix 6: Fit TF-IDF vectorizer on baseline command lines.
    Character n-grams handle obfuscated commands better than word tokens.
    max_features is overridden by cfg.scoring.tfidf_max_features when called
    from the pipeline (see pipeline.py fit()).
    """
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        min_df=3,
    )
    vectorizer.fit(baseline_commands.fillna(""))
    return vectorizer


def score_cmdline_novelty(
    test_commands: pd.Series,
    baseline_matrix_sampled: "sparse matrix",
    vectorizer: TfidfVectorizer,
    k_nearest: int = 5,
) -> pd.Series:
    """
    Fix 6: Per-command semantic distance from baseline.
    High distance = novel command not seen in training = anomalous.
    Catches LOLBin variants not in the hardcoded keyword list.

    Takes a PRE-SAMPLED baseline matrix (built once during fit) rather
    than transforming the full baseline corpus on every call.  This is
    the key performance fix — the baseline is reduced from ~330k commands
    to ~1000 representative samples during fit().
    """
    test_matrix = normalize(vectorizer.transform(test_commands.fillna("")))

    chunk_size = 1000
    all_scores = []
    for i in range(0, test_matrix.shape[0], chunk_size):
        chunk = test_matrix[i:i + chunk_size]
        sim = (chunk @ baseline_matrix_sampled.T).toarray()
        top_k = np.sort(sim, axis=1)[:, -min(k_nearest, sim.shape[1]):]
        all_scores.extend((1.0 - top_k.mean(axis=1)).tolist())

    return pd.Series(all_scores, index=test_commands.index, name="cmdline_novelty")


def build_baseline_matrix(
    baseline_commands: pd.Series,
    vectorizer: TfidfVectorizer,
    max_samples: int = 1000,
    random_seed: int = 42,
) -> "sparse matrix":
    """
    Build a compact, pre-transformed baseline matrix for cmdline novelty scoring.

    Instead of transforming all 330k baseline commands every time
    score_cmdline_novelty is called, we:
      1. Deduplicate command lines (most are repeated across hosts)
      2. Sample down to max_samples unique commands
      3. Transform and normalize once

    The result is stored in FittedArtifacts and reused during scoring.
    Typical reduction: 330k raw commands → ~5k unique → 1000 sampled
    → a (1000, n_features) sparse matrix that fits in memory and makes
    the cosine similarity computation instant.
    """
    # Deduplicate first — huge reduction (330k → ~5k in typical data)
    unique_cmds = baseline_commands.dropna().drop_duplicates()

    if len(unique_cmds) > max_samples:
        unique_cmds = unique_cmds.sample(n=max_samples, random_state=random_seed)

    matrix = normalize(vectorizer.transform(unique_cmds.fillna("")))
    return matrix


def _peer_robust_score(
    values: np.ndarray,
    roles: np.ndarray,
    min_role: int = 5,
) -> np.ndarray:
    """
    Map a per-host quantity to [0,1] by how far ABOVE its role peers it sits,
    in robust (median/MAD) units — then squash. Replaces fixed-constant
    saturation (1 - exp(-x/5)) which pinned every host near 1.0 at enterprise
    volume and carried no discriminative signal.

    For each host:
        z = (x - median_peer) / (1.4826 * MAD_peer)      [clipped at 0]
        s = 1 - exp(-z / 2)                               [0 at the median]

    Roles with < min_role hosts fall back to the global (all-host) median/MAD,
    so singletons (a lone DC/DNS) are still scored against the fleet rather
    than against themselves (which would be degenerate).
    """
    values = np.asarray(values, dtype=float)
    out = np.zeros(len(values), dtype=float)
    if len(values) == 0:
        return out

    g_med = np.median(values)
    g_mad = np.median(np.abs(values - g_med))

    for role in pd.unique(roles):
        idx = np.where(roles == role)[0]
        if len(idx) >= min_role:
            med = np.median(values[idx])
            mad = np.median(np.abs(values[idx] - med))
        else:
            med, mad = g_med, g_mad
        scale = 1.4826 * mad
        if scale <= 1e-9:                  # no spread → nothing is an outlier
            continue
        z = np.clip((values[idx] - med) / scale, 0.0, None)
        out[idx] = 1.0 - np.exp(-z / 2.0)
    return out


def score_context(
    df: pd.DataFrame,
    cfg: StrataConfig,
    cmdline_vectorizer: Optional[TfidfVectorizer] = None,
    baseline_commands: Optional[pd.Series] = None,
    baseline_cmd_matrix: Optional[object] = None,
    pair_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Context anomaly channel. Combines four components:

      1. Severity-weighted event rate: mean severity_score per host anchors
         the base — a host dominated by Event 10 / Event 8 is inherently
         more suspicious than one dominated by Event 1.

      2. Hard flag aggregation: encoded commands, download cradles, LOLBin
         usage, and execution bypass flags (weighted by domain knowledge).

      3. Event pair scores: high-severity event co-occurrence within a
         short window (e.g., LSASS access followed by lateral movement).
         Passed in as pair_stats from correlate_critical_events_by_host().

      4. Cmdline novelty via TF-IDF (Fix 6): semantic distance from
         baseline command lines — catches obfuscated/novel LOLBin variants.

    Final S_ctx is a weighted blend of all active components, normalized
    to [0,1] via saturation functions so no single component dominates.
    """
    # --- Component 1: Severity-weighted event rate ---
    sev_g = df.groupby("host").agg(
        severity_mean=("severity_score", "mean"),
        severity_max=("severity_score", "max"),
        n_events=("severity_score", "count"),
    ).reset_index()
    sev_g["severity_mean"] = sev_g["severity_mean"].fillna(0.1)
    sev_g["severity_max"]  = sev_g["severity_max"].fillna(0.1)

    # --- Component 2: Flag aggregation ---
    agg_cols: dict = {
        "encoded_hits":  ("has_encoded",         "sum"),
        "download_hits": ("has_download_cradle",  "sum"),
        "lolbin_hits":   ("is_lolbin",            "sum"),
    }
    if "has_bypass" in df.columns:
        agg_cols["bypass_hits"] = ("has_bypass", "sum")

    flags_g = df.groupby("host").agg(**agg_cols).reset_index()
    for col in ["encoded_hits", "download_hits", "lolbin_hits", "bypass_hits"]:
        if col not in flags_g.columns:
            flags_g[col] = 0

    raw_flag = (
        0.60 * flags_g["encoded_hits"]
        + 0.40 * flags_g["download_hits"]
        + 0.30 * flags_g["lolbin_hits"]
        + 0.30 * flags_g["bypass_hits"]
    )
    flags_g["flag_weighted"] = raw_flag.to_numpy()

    # Merge components 1 + 2
    g = sev_g.merge(flags_g, on="host", how="left")
    # Attach role for peer-relative scoring (df carries role_id in score()).
    if "role_id" in df.columns:
        host_role = df.groupby("host")["role_id"].first().reset_index()
        g = g.merge(host_role, on="host", how="left")
        g["role_id"] = g["role_id"].fillna("default")
    # Flag intensity as a RATE per 1,000 events (volume-invariant), peer-scored
    # below. Raw counts saturated 1-exp(-x/5) to ~1.0 for every busy host.
    g["flag_rate"] = 1000.0 * g["flag_weighted"].fillna(0.0) / g["n_events"].clip(lower=1)

    # --- Component 3: Semantic event pair scores ---
    # pair_stats columns: host, n_pairs, weighted_score_sum, max_pair_weight,
    #                     n_tactics, top_tactic
    # weighted_score_sum = sum of (count × pair_weight) — primary signal
    # max_pair_weight    = confidence of the single highest-weight pair observed
    # n_tactics          = number of distinct MITRE tactics firing
    if pair_stats is not None and not pair_stats.empty:
        p_cols = ["host"]
        for c in ["n_pairs", "weighted_score_sum", "max_pair_weight", "n_tactics"]:
            if c in pair_stats.columns:
                p_cols.append(c)
        g = g.merge(pair_stats[p_cols], on="host", how="left")
        for c in ["n_pairs", "weighted_score_sum", "max_pair_weight", "n_tactics"]:
            if c not in g.columns:
                g[c] = 0.0
            g[c] = g[c].fillna(0.0)

        # Pair evidence as a RATE per 1,000 events (peer-scored below).
        g["pair_rate"] = 1000.0 * (
            g["weighted_score_sum"].fillna(0.0) * (1.0 + g["max_pair_weight"].fillna(0.0))
        ) / g["n_events"].clip(lower=1)
        # Tactic breadth: distinct MITRE tactics co-firing is a kill-chain
        # signal independent of volume — kept as a small additive bonus.
        tactic_bonus = np.clip((g["n_tactics"].to_numpy() - 1) * 0.10, 0.0, 0.30)
    else:
        for c in ["n_pairs", "weighted_score_sum", "max_pair_weight", "n_tactics"]:
            g[c] = 0.0
        g["pair_rate"] = 0.0
        tactic_bonus = np.zeros(len(g))

    # --- Component 4: Cmdline TF-IDF novelty ---
    if (
        cfg.ablation.use_cmdline_embeddings
        and cmdline_vectorizer is not None
        and "cmdline" in df.columns
    ):
        # Use pre-built baseline matrix if available, otherwise build on the fly
        if baseline_cmd_matrix is not None:
            bl_matrix = baseline_cmd_matrix
        elif baseline_commands is not None:
            bl_matrix = build_baseline_matrix(
                baseline_commands, cmdline_vectorizer,
                max_samples=cfg.scoring.tfidf_baseline_samples,
                random_seed=cfg.scoring.random_seed,
            )
        else:
            bl_matrix = None

        if bl_matrix is not None:
            # Score ALL commands at once, then aggregate per host
            cmd_mask = df["cmdline"].notna()
            if cmd_mask.sum() > 0:
                all_cmds = df.loc[cmd_mask, "cmdline"]
                all_novelty = score_cmdline_novelty(
                    all_cmds, bl_matrix, cmdline_vectorizer,
                )
                df_with_novelty = df.copy()
                df_with_novelty["_cmdline_novelty"] = 0.0
                df_with_novelty.loc[cmd_mask, "_cmdline_novelty"] = all_novelty.values

                host_novelty = (
                    df_with_novelty.groupby("host")["_cmdline_novelty"]
                    .mean()
                    .reset_index()
                    .rename(columns={"_cmdline_novelty": "cmdline_novelty"})
                )
                g = g.merge(host_novelty, on="host", how="left")
                g["cmdline_novelty"] = g["cmdline_novelty"].fillna(0.0)
                S_novelty = g["cmdline_novelty"].to_numpy()
            else:
                g["cmdline_novelty"] = 0.0
                S_novelty = np.zeros(len(g))
        else:
            g["cmdline_novelty"] = 0.0
            S_novelty = np.zeros(len(g))
    else:
        g["cmdline_novelty"] = 0.0
        S_novelty = np.zeros(len(g))

    # --- Blend all components (each peer-relative, volume-invariant) ---
    # Every component is scored as "how far above your role peers" in robust
    # units, so a uniformly busy fleet no longer pins S_ctx near a constant,
    # and the channel can't score the most anomalous host LOWER than the pack.
    roles = (
        g["role_id"].to_numpy() if "role_id" in g.columns
        else np.array(["default"] * len(g))
    )
    C_sev  = _peer_robust_score(g["severity_mean"].to_numpy(), roles)
    C_flag = _peer_robust_score(g["flag_rate"].to_numpy(), roles)
    C_pair = np.clip(_peer_robust_score(g["pair_rate"].to_numpy(), roles) + tactic_bonus, 0.0, 1.0)
    C_nov  = _peer_robust_score(g["cmdline_novelty"].to_numpy(), roles) \
        if "cmdline_novelty" in g.columns else np.zeros(len(g))

    # Weights: severity anchors (0.30), flags primary (0.35), pairs
    # corroborate (0.20), novelty catches obfuscation (0.15).
    S_ctx = 0.30 * C_sev + 0.35 * C_flag + 0.20 * C_pair + 0.15 * C_nov
    S_ctx = np.clip(S_ctx, 0.0, 1.0)

    return pd.DataFrame({
        "host":            g["host"].values,
        "S_ctx":           S_ctx,
        "severity_mean":   g["severity_mean"].values,
        "severity_max":    g["severity_max"].values,
        "n_pairs":         g["n_pairs"].values,
        "cmdline_novelty": g["cmdline_novelty"].values,
    })


# ---------------------------------------------------------------------------
# Fix 1: Borda rank fusion + corroboration gate
# ---------------------------------------------------------------------------

def borda_fusion(scores: pd.DataFrame, channel_cols: List[str]) -> pd.Series:
    """
    Rank-based Borda fusion. Robust to channel score distribution drift.
    Each channel contributes a rank rather than a raw score, eliminating the
    need to calibrate scales across channels.
    """
    rank_matrix = np.zeros((len(scores), len(channel_cols)))
    for i, col in enumerate(channel_cols):
        rank_matrix[:, i] = rankdata(scores[col].fillna(0), method="average")
    borda = rank_matrix.sum(axis=1)
    return pd.Series(borda, index=scores.index, name="fusion_score")


def corroboration_gate(
    scores: pd.DataFrame,
    channel_cols: List[str],
    cfg: StrataConfig,
) -> pd.DataFrame:
    """
    Gate: a host must be anomalous in >= min_corroborating_channels to surface.
    A single extreme channel still bypasses.

    Returns a DataFrame with two columns derived from ONE definition of
    "anomalous in a channel" (above the per-channel percentile threshold):
      gate_pass   : bool — multi-channel corroboration OR extreme bypass
      gate_reason : 'extreme_channel' | 'multi_channel' | 'low_support'

    (Previously gate_pass used percentile thresholds while gate_reason was
    recomputed in fuse_scores with a hardcoded 0.7 absolute threshold — the
    two could disagree, e.g. a host passing the gate labeled 'low_support'.)
    """
    threshold = np.percentile(
        scores[channel_cols].values,
        cfg.scoring.gate_percentile_threshold,
        axis=0,
    )
    above = (scores[channel_cols] > threshold).sum(axis=1)
    multi_channel = above >= cfg.scoring.min_corroborating_channels

    extreme = (scores[channel_cols] >= cfg.scoring.extreme_threshold).any(axis=1)

    gate_pass = (multi_channel | extreme)
    gate_reason = np.where(
        extreme, "extreme_channel",
        np.where(multi_channel, "multi_channel", "low_support"),
    )
    return pd.DataFrame(
        {"gate_pass": gate_pass.values, "gate_reason": gate_reason},
        index=scores.index,
    )


def learn_fusion_weights(
    channel_scores: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Fix 1: Learn fusion weights from synthetic injection ground truth.
    Call this after running your injection framework, then persist the weights.

    Returns normalized weight array (sums to 1.0).
    """
    from sklearn.linear_model import LogisticRegression
    X = StandardScaler().fit_transform(channel_scores)
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
    clf.fit(X, labels)
    raw = np.abs(clf.coef_[0])
    return raw / (raw.sum() + 1e-9)


def fuse_scores(
    seq_scores: pd.DataFrame,
    freq_scores: pd.DataFrame,
    ctx_scores: pd.DataFrame,
    drift_scores: pd.DataFrame,
    cfg: StrataConfig,
    learned_weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Fuse all channels using Borda (default) or weighted linear fusion.
    Applies corroboration gate.

    Output columns:
      host, score, gate_pass, gate_reason,
      S_seq, S_freq, S_ctx, S_drift, [S_seq_drift_cov],
      rare_transition_hits, n_events, seq_drift_correlation
    """
    df = (
        seq_scores
        .merge(freq_scores, on="host", how="outer")
        .merge(ctx_scores,  on="host", how="outer")
        .merge(drift_scores, on="host", how="outer")
    )

    for c in ["S_seq", "S_freq", "S_ctx", "S_drift"]:
        df[c] = df[c].fillna(0.0)

    active_channels = ["S_seq", "S_freq", "S_ctx"]
    if cfg.ablation.use_drift_channel:
        active_channels.append("S_drift")

    # Fix 2: add covariance meta-feature as a 5th channel
    if cfg.ablation.use_seq_drift_covariance and cfg.ablation.use_drift_channel:
        df["S_seq_drift_cov"] = compute_seq_drift_covariance(df).clip(lower=0)
        active_channels.append("S_seq_drift_cov")

        corr_info = check_channel_correlation(df)
        df["seq_drift_correlation"] = corr_info["seq_drift_correlation"]
    else:
        df["seq_drift_correlation"] = np.nan

    # Fusion
    if cfg.scoring.fusion_method == "borda":
        df["score"] = borda_fusion(df, active_channels)
    elif cfg.scoring.fusion_method == "weighted_linear":
        if learned_weights is not None:
            w = learned_weights
        else:
            w = np.array([
                cfg.scoring.w_seq, cfg.scoring.w_freq,
                cfg.scoring.w_ctx, cfg.scoring.w_drift
            ])
            if len(active_channels) > len(w):
                w = np.append(w, 0.05)  # small weight for covariance channel
            w = w[:len(active_channels)]
            w = w / w.sum()
        df["score"] = (df[active_channels].values * w).sum(axis=1)
    else:
        raise ValueError(f"Unknown fusion_method: {cfg.scoring.fusion_method}")

    # Normalize score to percentile rank for comparability
    df["score"] = df["score"].rank(pct=True)

    # Corroboration gate (Fix 1 - promoted from optional)
    if cfg.ablation.use_corroboration_gate:
        gate = corroboration_gate(df, active_channels, cfg)
        df["gate_pass"] = gate["gate_pass"]
        df["gate_reason"] = gate["gate_reason"]
        # Zero out non-corroborated hosts for ranking (they stay in output but ranked last)
        df.loc[~df["gate_pass"], "score"] = 0.0
    else:
        df["gate_pass"] = True
        df["gate_reason"] = "no_gating"

    return df.sort_values("score", ascending=False).reset_index(drop=True)


def build_ranked_triage(
    fused: pd.DataFrame,
    pair_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Final ranked triage table. Merges semantic pair stats for explainability.

    Key analyst-facing columns added from pair_stats:
      n_pairs             — distinct known attack-pattern pairs observed
      weighted_score_sum  — total weighted pair evidence
      max_pair_weight     — confidence of highest-weight pair (1.0 = near-certain)
      n_tactics           — distinct MITRE tactics represented
      top_tactic          — dominant MITRE tactic (e.g. 'credential_access')
    """
    triage = fused.copy()
    if pair_stats is not None and not pair_stats.empty:
        merge_cols = ["host"] + [
            c for c in [
                "n_pairs", "weighted_score_sum", "max_pair_weight",
                "n_tactics", "top_tactic",
            ]
            if c in pair_stats.columns
        ]
        triage = triage.merge(pair_stats[merge_cols], on="host", how="left")
        triage["top_tactic"] = triage.get("top_tactic", pd.Series("none", index=triage.index)).fillna("none")
    triage["triage_rank"] = range(1, len(triage) + 1)
    return triage
