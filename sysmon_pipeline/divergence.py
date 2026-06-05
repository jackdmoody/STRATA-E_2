"""
Divergence functions, Bayesian peer baselines, and JSD calibration.
=====================================================================
Best of:
  - v12_modular divergence.py:    clean kl/js matrix functions
  - pipeline_updated divergence.py: fit_peer_baselines, score_sequence_divergence

Adds:
  - Fix 2: seq/drift covariance meta-feature
  - Fix 5: shrinkage weight tracking as evasion signal
  - JSD null distribution calibration via bootstrap (from README)
  - Hierarchical Dirichlet shrinkage (from README spec)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import kstest

from .config import StrataConfig


# ---------------------------------------------------------------------------
# Core divergence functions (from v12_modular - cleanest implementation)
# ---------------------------------------------------------------------------

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0); p = p / p.sum()
    q = np.clip(q, eps, 1.0); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0); p = p / p.sum()
    q = np.clip(q, eps, 1.0); q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)


# ---------------------------------------------------------------------------
# Hierarchical Dirichlet peer baselines (README spec + pipeline_updated)
# ---------------------------------------------------------------------------

def fit_role_prior_concentration(
    role_baseline: pd.DataFrame,
    alpha: float = 0.5,
) -> float:
    """
    Estimate the Dirichlet concentration α₀ for the role-level prior (paper Eq. 5:
    θ_r ~ Dirichlet(α₀)) via moment matching on the role transition distribution.

    In practice we treat the Laplace-smoothed role baseline as a point estimate of
    θ̂_r (the posterior mean under a symmetric Dirichlet prior), and back out the
    implied concentration from the empirical variance of p_baseline across outgoing
    transitions.  This bridges the notation gap between Eq. 5 (full Bayesian model)
    and the implemented estimator (Eq. 7 posterior mean).

    Returns α₀ (scalar); used only for documentation / calibration reporting.
    If estimation is degenerate, returns the configured laplace_alpha as a fallback.
    """
    if role_baseline.empty or "p_baseline" not in role_baseline.columns:
        return alpha

    per_state_var = (
        role_baseline
        .groupby("state")["p_baseline"]
        .var(ddof=1)
        .dropna()
    )
    if per_state_var.empty or per_state_var.mean() == 0:
        return alpha

    # Moment-matching: for Dir(α₀ * p̄), Var(p_k) ≈ p̄_k(1-p̄_k)/(α₀+1)
    # => α₀ ≈ p̄_k(1-p̄_k)/Var(p_k) - 1
    per_state_mean = role_baseline.groupby("state")["p_baseline"].mean()
    p_bar = per_state_mean.mean()
    v_bar = per_state_var.mean()
    alpha0 = max(alpha, p_bar * (1.0 - p_bar) / (v_bar + 1e-12) - 1.0)
    return float(alpha0)


def fit_peer_baselines(
    trans: pd.DataFrame,
    host_roles: pd.DataFrame,
    cfg: StrataConfig,
) -> Dict[str, pd.DataFrame]:
    """
    Fit role-level transition distributions with Dirichlet smoothing.

    Returns {role_id -> DataFrame[state, next_state, dt_bucket, p_baseline, alpha0]}

    Relationship to paper Eqs. 5–7:
      - Eq. 5 (θ_r ~ Dir(α₀)): α₀ is estimated via moment matching on the role
        distribution using fit_role_prior_concentration(). The returned p_baseline
        is the posterior mean θ̂_r under this prior.
      - Eq. 6 (θ_h | θ_r ~ Dir(κ_r θ_r)): applied per-host in score_sequence_divergence()
        via _get_host_posterior(), using the stored p_baseline as θ̂_r.
      - Eq. 7 (θ̂_h = (N_h + κ θ̂_r)/(n_h + κ)): the closed-form posterior mean,
        implemented directly in _get_host_posterior().

    The Laplace-smoothed empirical counts serve as a point estimate for θ̂_r
    (equivalent to Eq. 7 with N_role replacing N_h and the role's total as n_role).
    """
    t = trans.merge(host_roles[["host", "role_id"]], on="host", how="left")
    baselines: Dict[str, pd.DataFrame] = {}
    alpha = cfg.baseline.laplace_alpha

    for role_id, g in t.groupby("role_id"):
        agg = (
            g.groupby(["state", "next_state", "dt_bucket"])["count"]
            .sum()
            .reset_index()
        )
        agg["total"] = agg.groupby("state")["count"].transform("sum")
        k = agg.groupby("state")["count"].transform("count")
        agg["p_baseline"] = (agg["count"] + alpha) / (agg["total"] + alpha * k)

        # Estimate α₀ (paper Eq. 5) for calibration reporting
        alpha0 = fit_role_prior_concentration(agg, alpha=alpha)
        agg["alpha0"] = alpha0

        baselines[str(role_id)] = agg[[
            "state", "next_state", "dt_bucket", "p_baseline", "alpha0"
        ]].copy()

    return baselines


def _get_host_posterior(
    host_trans: pd.DataFrame,
    role_baseline: pd.DataFrame,
    kappa: float,
    alpha: float,
) -> pd.DataFrame:
    """
    Compute host-level posterior distribution using Dirichlet shrinkage:
      θ̂_h = (N_h + κ θ̂_r) / (n_h + κ)

    Returns merged DataFrame with p_host (posterior) and p_baseline columns.
    """
    # Host empirical probs
    n_h = host_trans["count"].sum()
    host_emp = host_trans.copy()
    host_emp["p_host_empirical"] = host_emp["count"] / (n_h + 1e-9)

    # Merge with role baseline
    merged = host_emp.merge(
        role_baseline, on=["state", "next_state", "dt_bucket"], how="outer"
    ).fillna(0.0)

    # Posterior (Dirichlet shrinkage)
    merged["p_host"] = (
        (merged["count"] + kappa * merged["p_baseline"])
        / (n_h + kappa)
    )
    return merged


# ---------------------------------------------------------------------------
# Sequence channel scoring
# ---------------------------------------------------------------------------

def score_sequence_divergence(
    trans: pd.DataFrame,
    host_roles: pd.DataFrame,
    baselines: Dict[str, pd.DataFrame],
    cfg: StrataConfig,
) -> pd.DataFrame:
    """
    Per-host sequence anomaly score using JS divergence between:
      P_host(next, dt | state)  vs  P_peer(next, dt | state)

    Applies Dirichlet shrinkage if cfg.ablation.use_dirichlet_shrinkage.

    Returns: host, role_id, S_seq, rare_transition_hits, n_events
    """
    t = trans.merge(host_roles[["host", "role_id"]], on="host", how="left")
    kappa = cfg.baseline.dirichlet_kappa if cfg.ablation.use_dirichlet_shrinkage else 0.0
    rows = []

    for (host, role_id), g in t.groupby(["host", "role_id"]):
        base = baselines.get(str(role_id))
        n_h = int(g["count"].sum())

        if base is None or base.empty:
            rows.append((host, role_id, 0.0, 0, n_h))
            continue

        divs, sev_weights, rare_hits = [], [], 0
        for state, hs in g.groupby("state"):
            if cfg.ablation.use_dirichlet_shrinkage:
                merged = _get_host_posterior(hs, base[base["state"] == state], kappa, cfg.baseline.laplace_alpha)
                p = merged["p_host"].to_numpy()
                q = merged["p_baseline"].to_numpy()
            else:
                hb = hs.copy()
                hb["p_host"] = hb["count"] / hb["count"].sum()
                pb = base[base["state"] == state].copy()
                if pb.empty:
                    continue
                hb["key"] = hb["next_state"].astype(str) + "|" + hb["dt_bucket"].astype(str)
                pb["key"] = pb["next_state"].astype(str) + "|" + pb["dt_bucket"].astype(str)
                all_keys = sorted(set(hb["key"]).union(set(pb["key"])))
                p = np.array([hb.loc[hb["key"] == k, "p_host"].sum() for k in all_keys])
                q = np.array([pb.loc[pb["key"] == k, "p_baseline"].sum() for k in all_keys])

            if q.sum() == 0:
                continue

            jsd = js_divergence(p, q)
            rare_hits += int((q[q > 0].min() if (q > 0).any() else 1.0) < 1e-4)

            # Severity-weighted JSD: weight each state's contribution by the
            # count-weighted mean severity prior of events observed from that
            # state (carried as `state_severity` by build_transition_counts).
            # A structurally anomalous transition involving high-severity events
            # (LSASS access, CreateRemoteThread) contributes more than the same
            # deviation in low-severity process chains. Falls back to a constant
            # 0.5 (unweighted mean) when the column is absent.
            if "state_severity" in hs.columns and hs["count"].sum() > 0:
                state_sev = float(np.average(
                    hs["state_severity"].fillna(0.5), weights=hs["count"]
                ))
            else:
                state_sev = 0.5
            if np.isnan(state_sev):
                state_sev = 0.5

            divs.append(jsd * state_sev)
            sev_weights.append(state_sev)

        # Severity-weighted mean: sum(jsd_i * w_i) / sum(w_i)
        S_seq = float(np.sum(divs) / (np.sum(sev_weights) + 1e-9)) if divs else 0.0
        rows.append((host, role_id, S_seq, rare_hits, n_h))

    return pd.DataFrame(rows, columns=["host", "role_id", "S_seq", "rare_transition_hits", "n_events"])


# ---------------------------------------------------------------------------
# Drift channel scoring
# ---------------------------------------------------------------------------

def score_drift(
    current_trans: pd.DataFrame,
    prior_trans: Optional[pd.DataFrame],
    host_roles: pd.DataFrame,
    cfg: StrataConfig,
) -> pd.DataFrame:
    """
    Per-host drift score: JS divergence between current and prior window distributions.
    Uses Dirichlet smoothing for stability on sparse windows.

    Returns: host, S_drift
    """
    hosts = pd.Index(current_trans["host"].unique())
    if prior_trans is None or prior_trans.empty or not cfg.ablation.use_drift_channel:
        return pd.DataFrame({"host": hosts.values, "S_drift": 0.0})

    alpha = cfg.baseline.laplace_alpha
    rows = []

    for host in hosts:
        cur = current_trans[current_trans["host"] == host]
        pri = prior_trans[prior_trans["host"] == host]

        if cur.empty or pri.empty:
            rows.append({"host": host, "S_drift": 0.0})
            continue

        cur_agg = cur.groupby(["state", "next_state", "dt_bucket"])["count"].sum().reset_index()
        pri_agg = pri.groupby(["state", "next_state", "dt_bucket"])["count"].sum().reset_index()

        merged = cur_agg.merge(pri_agg, on=["state", "next_state", "dt_bucket"],
                                how="outer", suffixes=("_cur", "_pri")).fillna(0.0)
        n_cur = merged["count_cur"].sum()
        n_pri = merged["count_pri"].sum()

        p = (merged["count_cur"] + alpha) / (n_cur + alpha)
        q = (merged["count_pri"] + alpha) / (n_pri + alpha)

        rows.append({"host": host, "S_drift": js_divergence(p.to_numpy(), q.to_numpy())})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fix 2: Seq/Drift covariance meta-feature
# ---------------------------------------------------------------------------

def compute_seq_drift_covariance(scores: pd.DataFrame) -> pd.Series:
    """
    The product of z-scored seq and drift scores is a 5th detection signal.
    High positive value = both channels firing together = sustained behavioral change.
    Catches slow lateral movement that gradually shifts a host's profile.
    """
    seq_z = (scores["S_seq"] - scores["S_seq"].mean()) / (scores["S_seq"].std() + 1e-9)
    drft_z = (scores["S_drift"] - scores["S_drift"].mean()) / (scores["S_drift"].std() + 1e-9)
    return (seq_z * drft_z).rename("S_seq_drift_cov")


def check_channel_correlation(scores: pd.DataFrame) -> dict:
    """
    Report Pearson correlation between seq and drift channels.
    High correlation (>0.75) warns of single-source signal amplification.
    """
    r = scores["S_seq"].corr(scores["S_drift"])
    return {
        "seq_drift_correlation": round(float(r), 4),
        "high_correlation_warning": abs(r) > 0.75,
    }


# ---------------------------------------------------------------------------
# Fix 5: Shrinkage anomaly tracking (evasion detection)
# ---------------------------------------------------------------------------

def compute_shrinkage_weights(
    host_event_counts: Dict[str, int],
    kappa: float,
) -> pd.DataFrame:
    """
    For each host, compute how much its estimate is pulled toward the role baseline.
    Shrinkage weight = κ / (n_h + κ).

    High shrinkage in a previously active host = sudden event suppression.
    """
    rows = []
    for host, n_h in host_event_counts.items():
        sw = kappa / (n_h + kappa)
        rows.append({
            "host": host,
            "event_count": n_h,
            "shrinkage_weight": sw,
            "data_dominated": sw < 0.2,
            "prior_dominated": sw > 0.8,
        })
    return pd.DataFrame(rows)


def detect_shrinkage_anomalies(
    current_shrinkage: pd.DataFrame,
    historical_shrinkage: pd.DataFrame,
    delta_threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Flag hosts where shrinkage weight jumped significantly since last window.
    This signals a host that normally has lots of events has gone quiet —
    a potential log suppression or very quiet attacker evasion pattern.
    """
    merged = current_shrinkage.merge(
        historical_shrinkage[["host", "shrinkage_weight"]].rename(
            columns={"shrinkage_weight": "hist_shrinkage_weight"}
        ),
        on="host", how="left",
    )
    merged["shrinkage_delta"] = (
        merged["shrinkage_weight"] - merged["hist_shrinkage_weight"].fillna(0)
    )
    merged["evasion_signal"] = merged["shrinkage_delta"] > delta_threshold
    return merged


# ---------------------------------------------------------------------------
# JSD calibration (Fix 5 / README) — with bootstrap cache + vectorization
# ---------------------------------------------------------------------------

@dataclass
class BootstrapNull:
    """Bootstrap null distribution for JSD calibration.

    Stores enough information for both:
      - z-score (standardized effect size, no distributional assumption)
      - empirical p-value (distribution-free, computed from the sample array)
    """
    mu: float              # mean of bootstrap JSD samples
    sigma: float           # std of bootstrap JSD samples (+ eps)
    samples: np.ndarray    # full array of bootstrap JSD means (n_boot,)

    def z_score(self, observed_jsd: float) -> float:
        """Standardized effect size: how many null-SDs above the null mean.
        NOTE: computable for any distribution, but readers map sigma-counts
        onto Gaussian tail probabilities; on the right-skewed JSD null this
        overstates rarity. Prefer fold_over_null / empirical_percentile in
        reader-facing material."""
        return (observed_jsd - self.mu) / self.sigma

    def fold_over_null(self, observed_jsd: float) -> float:
        """Scale-free effect size: observed / null mean ("4.2x the null").
        Monotone, no distributional connotation, does not saturate when the
        observed value exceeds the bootstrap support (unlike the percentile)."""
        return float(observed_jsd / (self.mu + 1e-12))

    def empirical_pvalue(self, observed_jsd: float) -> float:
        """Distribution-free p-value with add-one correction:
            p = (1 + #{null >= observed}) / (1 + B)
        A finite-sample empirical p-value can never legitimately be 0 — the
        observed value is itself one draw from the pooled (null + observed)
        set, so the smallest defensible p is 1/(B+1). The uncorrected
        mean(null >= obs) form returned exactly 0.0 whenever the observed
        value exceeded every bootstrap sample, overstating significance."""
        b = len(self.samples)
        k = int(np.sum(self.samples >= observed_jsd))
        return float((1.0 + k) / (1.0 + b))

    def empirical_percentile(self, observed_jsd: float) -> float:
        """Percentile rank within the null, consistent with the add-one
        p-value (caps at 100*B/(B+1), never exactly 100)."""
        b = len(self.samples)
        return float(100.0 * np.sum(self.samples < observed_jsd) / (b + 1.0))


# Cache keyed by (role_id, binned_n_h) → BootstrapNull.
# Hosts with the same role and similar event counts share the same null
# distribution, so we only need to bootstrap once per bin.
_bootstrap_cache: Dict[Tuple[str, int], BootstrapNull] = {}


def _bin_event_count(n_h: int) -> int:
    """
    Bin host event counts for cache lookup.  Hosts in the same bin share a
    bootstrap null.  Bins widen for larger counts (where the null is more
    stable anyway): [0-25], [26-50], [51-100], [101-250], [251-500], 501+.
    """
    for edge in (25, 50, 100, 250, 500, 1000, 2500):
        if n_h <= edge:
            return edge
    return 5000


def clear_bootstrap_cache() -> None:
    """Call between runs if baselines change."""
    _bootstrap_cache.clear()


def _js_divergence_vectorized(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Inline JSD without the overhead of two kl_divergence calls."""
    p = np.clip(p, eps, None); p = p / p.sum()
    q = np.clip(q, eps, None); q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def calibrate_jsd_null_distribution(
    role_baseline: pd.DataFrame,
    n_h: int,
    cfg: StrataConfig,
    rng: Optional[np.random.Generator] = None,
    role_id: str = "default",
) -> BootstrapNull:
    """
    LEGACY global null (kept for API compatibility / ablation reference).

    Known mismatches vs. the observed statistic, which produced wildly
    inflated z-scores (hundreds of sigma) on real data:
      1. Allocates n_h uniformly across states; real hosts have highly
         uneven per-state sample sizes, and JSD sampling noise scales ~K/n.
      2. Does not apply the Dirichlet-shrinkage estimator the statistic uses.
      3. Weights states uniformly; the statistic is severity-weighted.

    Use calibrate_jsd_matched() for the matched per-host null, and the
    conformal peer p-values (pipeline.score) for decision-grade calibration.
    """
    if not cfg.ablation.use_jsd_calibration:
        return BootstrapNull(mu=0.0, sigma=1.0, samples=np.zeros(1))

    # --- Cache lookup ---
    n_bin = _bin_event_count(n_h)
    cache_key = (role_id, n_bin)
    if cache_key in _bootstrap_cache:
        return _bootstrap_cache[cache_key]

    rng = rng or np.random.default_rng(cfg.scoring.random_seed)
    alpha = cfg.baseline.laplace_alpha
    agg = role_baseline.groupby("state")

    state_groups = list(agg)
    n_states = len(state_groups)
    if n_states == 0:
        result = BootstrapNull(mu=0.0, sigma=1.0, samples=np.zeros(1))
        _bootstrap_cache[cache_key] = result
        return result

    state_weights = np.array([g["p_baseline"].sum() for _, g in state_groups])
    state_weights = state_weights / (state_weights.sum() + 1e-12)

    n_boot = cfg.baseline.bootstrap_samples
    eps = 1e-12

    per_state_jsds = []
    for (state, g), sw in zip(state_groups, state_weights):
        n_state = max(2, int(n_h * sw))
        p_base = (g["p_baseline"] + alpha).to_numpy().copy()
        p_base /= p_base.sum()
        k = len(p_base)

        sample_matrix = rng.multinomial(n_state, p_base, size=n_boot).astype(np.float64)
        sample_matrix += alpha
        row_sums = sample_matrix.sum(axis=1, keepdims=True)
        p_samples = sample_matrix / row_sums

        q = np.clip(p_base, eps, None); q = q / q.sum(); q = q[np.newaxis, :]
        p_s = np.clip(p_samples, eps, None)
        p_s = p_s / p_s.sum(axis=1, keepdims=True)

        m = 0.5 * (p_s + q)
        kl_pq = np.sum(p_s * np.log(p_s / m), axis=1)
        kl_qp = np.sum(q * np.log(q / m), axis=1)
        per_state_jsds.append(0.5 * kl_pq + 0.5 * kl_qp)

    if not per_state_jsds:
        result = BootstrapNull(mu=0.0, sigma=1.0, samples=np.zeros(1))
        _bootstrap_cache[cache_key] = result
        return result

    all_jsds = np.stack(per_state_jsds, axis=1)
    bootstrap_means = all_jsds.mean(axis=1)

    result = BootstrapNull(
        mu=float(bootstrap_means.mean()),
        sigma=float(bootstrap_means.std() + 1e-9),
        samples=bootstrap_means,
    )
    _bootstrap_cache[cache_key] = result
    return result


def calibrate_jsd_matched(
    host_trans: pd.DataFrame,
    role_baseline: pd.DataFrame,
    cfg: StrataConfig,
    rng: Optional[np.random.Generator] = None,
) -> BootstrapNull:
    """
    MATCHED parametric bootstrap null for one host's S_seq.

    The null distribution is computed by recomputing the EXACT observed
    statistic on data simulated under H0 ("this host's transitions are
    draws from its role baseline"):

      - per-state sample sizes = the host's ACTUAL per-state transition
        counts (sampling noise scales ~K/n per state, so this matters);
      - the same estimator the statistic uses (Dirichlet-shrinkage posterior
        when enabled, smoothed empirical otherwise);
      - the same severity weights (held fixed: they are conditioning
        variables of the statistic, not part of the randomness);
      - the same severity-weighted mean across states.

    The resulting z-score is a meaningful effect size ("how far above
    sampling noise"). Note the hypothesis itself is still 'iid draws from
    the pooled role baseline' — benign hosts with stable idiosyncratic
    behavior will legitimately exceed it. Decision-grade p-values therefore
    come from the conformal peer calibration in pipeline.score(), which
    tests the operationally correct hypothesis ('this host looks like a
    benign-window peer').
    """
    if not cfg.ablation.use_jsd_calibration:
        return BootstrapNull(mu=0.0, sigma=1.0, samples=np.zeros(1))

    rng = rng or np.random.default_rng(cfg.scoring.random_seed)
    alpha = cfg.baseline.laplace_alpha
    kappa = cfg.baseline.dirichlet_kappa if cfg.ablation.use_dirichlet_shrinkage else 0.0
    use_shrink = cfg.ablation.use_dirichlet_shrinkage
    n_boot = cfg.baseline.bootstrap_samples
    eps = 1e-12

    weighted_jsds = None   # (n_boot,) accumulator: sum of w_s * jsd_s
    weight_total = 0.0

    for state, hs in host_trans.groupby("state"):
        pb = role_baseline[role_baseline["state"] == state]
        if pb.empty:
            continue

        n_s = int(hs["count"].sum())
        if n_s <= 0:
            continue

        # Severity weight: same conditioning value the statistic uses
        if "state_severity" in hs.columns and hs["count"].sum() > 0:
            w_s = float(np.average(hs["state_severity"].fillna(0.5), weights=hs["count"]))
            if np.isnan(w_s):
                w_s = 0.5
        else:
            w_s = 0.5

        p_base = pb["p_baseline"].to_numpy().astype(np.float64)
        if p_base.sum() <= 0:
            continue
        p_draw = p_base / p_base.sum()
        k = len(p_draw)

        # Draw all bootstrap replicates for this state with the HOST'S n_s
        counts = rng.multinomial(n_s, p_draw, size=n_boot).astype(np.float64)  # (n_boot, k)

        # Apply the SAME estimator as the observed statistic
        if use_shrink:
            p_s = (counts + kappa * p_base[np.newaxis, :]) / (n_s + kappa)
        else:
            p_s = counts / max(n_s, 1)

        q = np.clip(p_base, eps, None); q = q / q.sum(); q = q[np.newaxis, :]
        p_s = np.clip(p_s, eps, None)
        p_s = p_s / p_s.sum(axis=1, keepdims=True)

        m = 0.5 * (p_s + q)
        kl_pq = np.sum(p_s * np.log(p_s / m), axis=1)
        kl_qp = np.sum(q * np.log(q / m), axis=1)
        jsd_s = 0.5 * kl_pq + 0.5 * kl_qp  # (n_boot,)

        weighted_jsds = jsd_s * w_s if weighted_jsds is None else weighted_jsds + jsd_s * w_s
        weight_total += w_s

    if weighted_jsds is None or weight_total <= 0:
        return BootstrapNull(mu=0.0, sigma=1.0, samples=np.zeros(1))

    samples = weighted_jsds / (weight_total + 1e-9)   # matches S_seq formula
    return BootstrapNull(
        mu=float(samples.mean()),
        sigma=float(samples.std() + 1e-9),
        samples=samples,
    )


def conformal_peer_pvalues(
    observed: pd.DataFrame,
    peer_sseq: Dict[str, np.ndarray],
    self_sseq: Optional[Dict[str, np.ndarray]] = None,
    min_self_folds: int = 3,
) -> pd.Series:
    """
    Conformal p-values for S_seq, distribution-free.

    Primary calibration is PER-HOST TEMPORAL (when self_sseq is supplied):
    the baseline window is split into folds and each host is scored against
    the role baseline per fold, giving that host its own benign S_seq
    distribution. The scoring-window p-value is then

        p = (1 + #{own benign fold scores >= s_h}) / (1 + n_folds)

    This tests "is this host anomalous vs. ITS OWN history", which is the
    exchangeable null — it does not penalize a host for having a stable,
    idiosyncratic-but-benign profile (the failure mode of cross-host pooling,
    where a persistently-different-but-benign host gets a low p-value every
    window). Resolution is 1/(n_folds+1), so more baseline folds → finer p.

    Hosts with fewer than min_self_folds benign folds (e.g. new/quiet hosts)
    fall back to the cross-host role-peer pool, then to the fleet pool.

    `observed` needs columns: host, role_id, S_seq.
    """
    global_pool = (
        np.concatenate([v for v in peer_sseq.values() if len(v)])
        if peer_sseq else np.array([])
    )
    self_sseq = self_sseq or {}
    pvals = []
    for _, row in observed.iterrows():
        s = float(row["S_seq"])
        own = self_sseq.get(str(row["host"]))
        if own is not None and len(own) >= min_self_folds:
            pvals.append((1.0 + float(np.sum(own >= s))) / (1.0 + len(own)))
            continue
        pool = peer_sseq.get(str(row["role_id"]), np.array([]))
        if len(pool) < 3:
            pool = global_pool
        if len(pool) == 0:
            pvals.append(np.nan)
            continue
        pvals.append((1.0 + float(np.sum(pool >= s))) / (1.0 + len(pool)))
    return pd.Series(pvals, index=observed.index, name="S_seq_pvalue")


def test_pvalue_uniformity(pvalues: np.ndarray, alpha: float = 0.05) -> dict:
    """
    KS test: are calibrated p-values uniform under benign data?
    Uniform = calibration is valid.

    With the empirical (distribution-free) p-values from BootstrapNull,
    this test should produce cleaner results than the previous Gaussian-CDF
    approach, because the p-values no longer carry the systematic bias
    introduced by fitting a symmetric distribution to a right-skewed null.
    """
    stat, p = kstest(pvalues, "uniform")
    return {
        "ks_statistic": round(float(stat), 4),
        "ks_pvalue": round(float(p), 4),
        "calibration_valid": bool(p > alpha),
        "interpretation": "Calibration OK" if p > alpha else "Non-uniform p-values — recalibrate null distribution",
    }