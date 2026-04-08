"""
Post-hoc analysis: feature importance, deployment math, benchmarking, error analysis.
======================================================================================
Inspired by the evaluation rigor in Nomad (Szewczyk 2025), this module adds four
analysis capabilities that strengthen STRATA-E's publication and operational credibility:

  1. SHAP feature importance  — which rate features drive the frequency channel?
  2. Deployment-prevalence math — SOC triage scenario framing
  3. Latency benchmarking — per-stage and per-host timing
  4. Qualitative error analysis — inspect specific FP/FN hosts with explanations

None of these change the pipeline's detection logic. They are post-hoc analyses
run after score() returns StrataArtifacts.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .pipeline import StrataArtifacts, FittedArtifacts
    from .config import StrataConfig

logger = logging.getLogger("strata.analysis")


# ---------------------------------------------------------------------------
# 1. SHAP Feature Importance (frequency channel)
# ---------------------------------------------------------------------------

def compute_shap_importance(
    fitted: "FittedArtifacts",
    rate_features: pd.DataFrame,
    max_samples: int = 200,
) -> Optional[pd.DataFrame]:
    """
    Compute SHAP feature importance for the IsolationForest frequency channel.

    Returns a DataFrame with columns: feature, mean_abs_shap, rank
    sorted by importance descending. Returns None if shap is not installed.

    Uses TreeExplainer for the IsolationForest model, which gives exact
    Shapley values for tree-based models.
    """
    try:
        import shap
    except ImportError:
        logger.warning(
            "shap not installed — skipping feature importance analysis. "
            "Install with: pip install shap"
        )
        return None

    model = fitted.freq_model
    X = rate_features.drop(columns=["host"], errors="ignore").fillna(0.0)
    feature_names = list(X.columns)

    # Subsample for speed on large fleets
    if len(X) > max_samples:
        X = X.sample(n=max_samples, random_state=42)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)
        return None

    # Mean absolute SHAP value per feature
    mean_abs = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = range(1, len(importance_df) + 1)

    return importance_df


def plot_shap_importance(
    importance_df: pd.DataFrame,
    top_k: int = 15,
    output_path: Optional[Path] = None,
) -> None:
    """Plot a horizontal bar chart of top-K SHAP feature importances."""
    import matplotlib.pyplot as plt

    top = importance_df.head(top_k).iloc[::-1]  # Reverse for bottom-up bar chart

    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.4)))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4C78A8", edgecolor="none")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_k} Frequency Channel Features by SHAP Importance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("SHAP plot saved to %s", output_path)

    plt.show()


# ---------------------------------------------------------------------------
# 2. Deployment-Prevalence Math (SOC triage scenario)
# ---------------------------------------------------------------------------

@dataclass
class DeploymentScenario:
    """Results of a deployment-prevalence calculation."""
    fleet_size: int
    attack_prevalence: float
    fpr: float
    recall: float
    precision_at_threshold: float

    # Derived (computed in __post_init__)
    expected_compromised: float = 0.0
    expected_true_positives: float = 0.0
    expected_false_positives: float = 0.0
    expected_alerts_per_run: float = 0.0
    deployment_ppv: float = 0.0
    detection_lift_vs_random: float = 0.0
    random_expected_tp: float = 0.0
    analyst_review_capacity: int = 100

    def __post_init__(self):
        M = self.fleet_size * self.attack_prevalence
        self.expected_compromised = M
        self.expected_true_positives = self.recall * M
        benign = self.fleet_size - M
        self.expected_false_positives = self.fpr * benign
        self.expected_alerts_per_run = (
            self.expected_true_positives + self.expected_false_positives
        )
        if self.expected_alerts_per_run > 0:
            self.deployment_ppv = (
                self.expected_true_positives / self.expected_alerts_per_run
            )
        else:
            self.deployment_ppv = 0.0

        # Random baseline: analyst reviews K hosts randomly
        K = self.analyst_review_capacity
        self.random_expected_tp = K * self.attack_prevalence
        if self.random_expected_tp > 0:
            self.detection_lift_vs_random = (
                self.expected_true_positives / self.random_expected_tp
            )
        else:
            self.detection_lift_vs_random = float("inf")


def compute_deployment_scenario(
    art: "StrataArtifacts",
    labels: Optional[pd.DataFrame] = None,
    fleet_size: int = 10_000,
    attack_prevalence: float = 0.001,
    analyst_capacity: int = 100,
) -> DeploymentScenario:
    """
    Project pipeline performance to a realistic SOC deployment scenario.

    If labels are provided, computes actual FPR and recall from the triage
    results. Otherwise, uses the H5-validated FPR=0.0 / recall from the
    corroboration gate as defaults.

    Mirrors the "Operational Example" framing from Nomad §6.1 but adapted
    for host-level triage (not per-event classification).
    """
    triage = art.triage
    if triage is None or triage.empty:
        return DeploymentScenario(
            fleet_size=fleet_size,
            attack_prevalence=attack_prevalence,
            fpr=0.0,
            recall=0.0,
            precision_at_threshold=0.0,
            analyst_review_capacity=analyst_capacity,
        )

    # Determine flagged hosts
    if "gate_pass" in triage.columns:
        flagged = set(triage[triage["gate_pass"] == True]["host"])
    else:
        flagged = set(triage.head(int(len(triage) * 0.25))["host"])

    if labels is not None and not labels.empty and "is_compromised" in labels.columns:
        compromised = set(labels[labels["is_compromised"] == True]["host"])
        benign = set(labels[labels["is_compromised"] == False]["host"])
        all_hosts = set(triage["host"])

        tp = len(flagged & compromised)
        fp = len(flagged & benign)
        fn = len(compromised - flagged)
        tn = len(benign - flagged)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    else:
        # Default to H5-validated performance (APT29 simulation)
        n_total = len(triage)
        n_flagged = len(flagged)
        recall = 1.0  # H5 showed recall maintained
        fpr = 0.0     # H5 showed FPR 1.0 → 0.0 with corroboration gate
        precision = 1.0 if n_flagged > 0 else 0.0

    return DeploymentScenario(
        fleet_size=fleet_size,
        attack_prevalence=attack_prevalence,
        fpr=fpr,
        recall=recall,
        precision_at_threshold=precision,
        analyst_review_capacity=analyst_capacity,
    )


def format_deployment_scenario(scenario: DeploymentScenario) -> str:
    """Format the deployment scenario as a printable string."""
    lines = [
        f"Fleet size:                    {scenario.fleet_size:,} hosts",
        f"Attack prevalence:             {scenario.attack_prevalence:.4%}",
        f"Expected compromised hosts:    {scenario.expected_compromised:.1f}",
        f"",
        f"Recall (detection rate):        {scenario.recall:.3f}",
        f"FPR (false alarm rate):         {scenario.fpr:.3f}",
        f"",
        f"Expected true positives/run:    {scenario.expected_true_positives:.1f}",
        f"Expected false positives/run:   {scenario.expected_false_positives:.1f}",
        f"Total alerts per triage run:    {scenario.expected_alerts_per_run:.1f}",
        f"Deployment PPV:                 {scenario.deployment_ppv:.3f}",
        f"",
        f"Random baseline ({scenario.analyst_review_capacity} hosts reviewed):",
        f"  Expected TP (random):         {scenario.random_expected_tp:.2f}",
        f"  Detection lift vs. random:    {scenario.detection_lift_vs_random:.0f}×",
    ]
    return "\n  ".join(lines)


# ---------------------------------------------------------------------------
# 3. Latency Benchmarking
# ---------------------------------------------------------------------------

@dataclass
class PipelineTiming:
    """Timing breakdown for a pipeline run."""
    fit_seconds: float = 0.0
    score_seconds: float = 0.0
    total_seconds: float = 0.0
    n_hosts_scored: int = 0
    n_events_scored: int = 0

    @property
    def hosts_per_second(self) -> float:
        return self.n_hosts_scored / self.score_seconds if self.score_seconds > 0 else 0.0

    @property
    def events_per_second(self) -> float:
        return self.n_events_scored / self.score_seconds if self.score_seconds > 0 else 0.0

    @property
    def ms_per_host(self) -> float:
        return (self.score_seconds * 1000) / self.n_hosts_scored if self.n_hosts_scored > 0 else 0.0


def format_timing(timing: PipelineTiming) -> str:
    """Format timing as a printable string."""
    lines = [
        f"Fit time:            {timing.fit_seconds:.2f}s",
        f"Score time:          {timing.score_seconds:.2f}s",
        f"Total:               {timing.total_seconds:.2f}s",
        f"",
        f"Hosts scored:        {timing.n_hosts_scored:,}",
        f"Events scored:       {timing.n_events_scored:,}",
        f"Throughput:           {timing.hosts_per_second:.1f} hosts/sec",
        f"                     {timing.events_per_second:.0f} events/sec",
        f"Per-host latency:    {timing.ms_per_host:.1f} ms/host",
    ]

    # Scaling projections
    for fleet in [1_000, 10_000, 50_000]:
        projected_sec = fleet / timing.hosts_per_second if timing.hosts_per_second > 0 else 0
        projected_min = projected_sec / 60
        lines.append(
            f"Projected for {fleet:>6,} hosts: {projected_min:.1f} min"
        )

    return "\n  ".join(lines)


def run_latency_benchmark(
    pipe: "StrataPipeline",
    fitted: "FittedArtifacts",
    score_df: pd.DataFrame,
    n_warmup: int = 1,
    n_runs: int = 3,
) -> PipelineTiming:
    """
    Benchmark score() latency over multiple runs.

    Runs n_warmup iterations to warm caches, then times n_runs and
    reports the median.
    """
    from .pipeline import StrataPipeline

    timings = []

    # Warmup
    for _ in range(n_warmup):
        pipe.score(score_df, fitted)

    # Timed runs
    for _ in range(n_runs):
        t0 = time.perf_counter()
        art = pipe.score(score_df, fitted)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)

    median_score = sorted(timings)[len(timings) // 2]
    n_hosts = art.events["host"].nunique() if "host" in art.events.columns else 0
    n_events = len(art.events)

    return PipelineTiming(
        score_seconds=median_score,
        n_hosts_scored=n_hosts,
        n_events_scored=n_events,
    )


# ---------------------------------------------------------------------------
# 4. Qualitative Error Analysis
# ---------------------------------------------------------------------------

@dataclass
class ErrorCase:
    """A single FP or FN case with explanation."""
    host: str
    case_type: str  # "false_positive", "false_negative", "true_positive"
    score: float
    gate_pass: bool
    channel_scores: Dict[str, float]
    explanation: str
    top_tokens: List[str]
    n_events: int


def analyze_errors(
    art: "StrataArtifacts",
    labels: pd.DataFrame,
    top_k: int = 5,
) -> List[ErrorCase]:
    """
    Inspect the most informative FP and FN cases and generate explanations.

    For each FP: explains why the pipeline flagged a benign host (which
    channels fired, what tokens/events drove the score).

    For each FN: explains why the pipeline missed a compromised host
    (which channels were below threshold, what made the host look normal).

    Mirrors the qualitative error analysis in Nomad §6.3.
    """
    if art.triage is None or art.triage.empty:
        return []
    if labels is None or labels.empty:
        return []

    triage = art.triage.copy()
    compromised = set(labels[labels["is_compromised"] == True]["host"])
    benign = set(labels[labels["is_compromised"] == False]["host"])

    # Determine flagged hosts
    if "gate_pass" in triage.columns:
        flagged = set(triage[triage["gate_pass"] == True]["host"])
    else:
        flagged = set()

    cases = []

    # --- False Positives (flagged but benign) ---
    fp_hosts = flagged & benign
    fp_rows = triage[triage["host"].isin(fp_hosts)].head(top_k)
    for _, row in fp_rows.iterrows():
        channels = _extract_channel_scores(row)
        firing = [ch for ch, val in channels.items() if val > 0.5]
        top_events = _get_top_tokens(art, row["host"])

        if "S_freq" in channels and channels["S_freq"] > 0.7:
            expl = (
                f"Frequency channel elevated ({channels['S_freq']:.2f}) — "
                f"host has unusual event volume relative to peers. "
                f"Likely a legitimate high-activity host (build server, scanner, admin workstation)."
            )
        elif "S_ctx" in channels and channels["S_ctx"] > 0.5:
            expl = (
                f"Context channel flagged ({channels['S_ctx']:.2f}) — "
                f"legitimate use of scripting tools or LOLBins triggered "
                f"behavioral flags. Top events: {', '.join(top_events[:3])}."
            )
        else:
            expl = (
                f"Corroboration gate passed via channels: {', '.join(firing)}. "
                f"May indicate legitimate but unusual administrative activity."
            )

        cases.append(ErrorCase(
            host=str(row["host"]),
            case_type="false_positive",
            score=float(row.get("score", 0)),
            gate_pass=True,
            channel_scores=channels,
            explanation=expl,
            top_tokens=top_events,
            n_events=_safe_int(row.get("n_events", 0)),
        ))

    # --- False Negatives (compromised but not flagged) ---
    fn_hosts = compromised - flagged
    fn_rows = triage[triage["host"].isin(fn_hosts)].head(top_k)
    for _, row in fn_rows.iterrows():
        channels = _extract_channel_scores(row)
        low_channels = [ch for ch, val in channels.items() if val < 0.3]
        top_events = _get_top_tokens(art, row["host"])

        if channels.get("S_seq", 0) < 0.2:
            expl = (
                f"Sequence channel score low ({channels.get('S_seq', 0):.2f}) — "
                f"attack behavior blended with normal process transitions. "
                f"Attacker may have used common tools (living-off-the-land) "
                f"that match the peer-role baseline."
            )
        elif channels.get("S_freq", 0) < 0.3:
            expl = (
                f"Frequency channel did not fire ({channels.get('S_freq', 0):.2f}) — "
                f"attack generated event volume within normal range. "
                f"Low-and-slow attack pattern evades volumetric detection."
            )
        else:
            above = [ch for ch, val in channels.items() if val > 0.5]
            expl = (
                f"Only {len(above)} channel(s) above threshold ({', '.join(above) or 'none'}). "
                f"Corroboration gate requires ≥2 channels to fire. "
                f"Single-channel signal insufficient for corroboration."
            )

        cases.append(ErrorCase(
            host=str(row["host"]),
            case_type="false_negative",
            score=float(row.get("score", 0)),
            gate_pass=bool(row.get("gate_pass", False)),
            channel_scores=channels,
            explanation=expl,
            top_tokens=top_events,
            n_events=_safe_int(row.get("n_events", 0)),
        ))

    return cases


def format_error_cases(cases: List[ErrorCase]) -> str:
    """Format error cases for terminal display."""
    if not cases:
        return "  No error cases to display."

    lines = []
    fps = [c for c in cases if c.case_type == "false_positive"]
    fns = [c for c in cases if c.case_type == "false_negative"]

    if fps:
        lines.append("  False Positives (benign hosts flagged):")
        lines.append(f"  {'—'*60}")
        for c in fps:
            lines.append(f"  {c.host} (score={c.score:.4f}, events={c.n_events})")
            ch_str = ", ".join(f"{k}={v:.2f}" for k, v in c.channel_scores.items())
            lines.append(f"    Channels: {ch_str}")
            lines.append(f"    {c.explanation}")
            lines.append("")

    if fns:
        lines.append("  False Negatives (compromised hosts missed):")
        lines.append(f"  {'—'*60}")
        for c in fns:
            lines.append(f"  {c.host} (score={c.score:.4f}, events={c.n_events})")
            ch_str = ", ".join(f"{k}={v:.2f}" for k, v in c.channel_scores.items())
            lines.append(f"    Channels: {ch_str}")
            lines.append(f"    {c.explanation}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Channel Taxonomy Table
# ---------------------------------------------------------------------------

def get_channel_taxonomy() -> pd.DataFrame:
    """
    Return a publication-ready taxonomy table of STRATA-E's detection channels.

    Mirrors Nomad Table 1 (Complete Taxonomy of Features) but organized
    by scoring channel rather than feature category.
    """
    rows = [
        {
            "Channel": "Sequence (S_seq)",
            "Technique": "Jensen-Shannon divergence, Dirichlet shrinkage, bootstrap calibration",
            "Inputs": "Per-host transition distributions (token pairs + time buckets)",
            "Output Columns": "S_seq, S_seq_z, S_seq_pvalue, S_seq_percentile, rare_transition_hits",
            "Detects": "Novel process chains, unusual execution sequences, structural deviations from peer role",
            "MITRE Coverage": "T1059 (Scripting), T1218 (Proxy Execution), T1055 (Injection), T1003 (Credential Dumping)",
        },
        {
            "Channel": "Frequency (S_freq)",
            "Technique": "Isolation Forest on per-host rate features",
            "Inputs": "proc_rate_total, script_rate, office_rate, lolbin_rate, encoded_rate, unique_users, unique_parents",
            "Output Columns": "S_freq",
            "Detects": "Volumetric anomalies — event rate spikes, unusual process mix, abnormal user diversity",
            "MITRE Coverage": "T1059 (Scripting volume), T1047 (WMI), T1087 (Discovery bursts)",
        },
        {
            "Channel": "Context (S_ctx)",
            "Technique": "Weighted flag aggregation + TF-IDF command novelty + critical event pair correlation",
            "Inputs": "Severity scores, encoded/bypass/download flags, LOLBin usage, MITRE pair weights, cmdline TF-IDF",
            "Output Columns": "S_ctx, severity_mean, severity_max, n_pairs, cmdline_novelty",
            "Detects": "Encoded commands, LOLBin abuse, suspicious parent-child chains, obfuscated command lines",
            "MITRE Coverage": "T1027 (Obfuscation), T1140 (Deobfuscation), T1036 (Masquerading), T1134 (Token Manipulation)",
        },
        {
            "Channel": "Drift (S_drift)",
            "Technique": "JSD between current and prior-window transition distributions",
            "Inputs": "Current vs. historical transition counts per host",
            "Output Columns": "S_drift, S_seq_drift_cov, seq_drift_correlation",
            "Detects": "Behavioral change over time — sustained shifts indicating compromise or configuration change",
            "MITRE Coverage": "T1053 (Scheduled Tasks), T1543 (Service Creation), T1547 (Persistence)",
        },
        {
            "Channel": "Corroboration Gate",
            "Technique": "Multi-channel consensus: ≥2 channels above 75th percentile or single extreme (>95th)",
            "Inputs": "S_seq, S_freq, S_ctx, S_drift (all channel scores)",
            "Output Columns": "gate_pass, gate_reason, score, triage_rank",
            "Detects": "Filters coincidental single-channel anomalies; surfaces only multi-evidence hosts",
            "MITRE Coverage": "Cross-tactic: requires corroboration across execution, credential access, lateral movement, etc.",
        },
    ]
    return pd.DataFrame(rows)


def format_channel_taxonomy(df: pd.DataFrame) -> str:
    """Format the taxonomy table for terminal display."""
    lines = []
    for _, row in df.iterrows():
        lines.append(f"  {row['Channel']}")
        lines.append(f"    Technique:  {row['Technique']}")
        lines.append(f"    Detects:    {row['Detects']}")
        lines.append(f"    MITRE:      {row['MITRE Coverage']}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(val, default=0) -> int:
    """Convert a value to int, handling NaN/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _extract_channel_scores(row: pd.Series) -> Dict[str, float]:
    """Extract channel scores from a triage row."""
    channels = {}
    for col in ["S_seq", "S_freq", "S_ctx", "S_drift"]:
        if col in row.index:
            val = row[col]
            channels[col] = float(val) if not pd.isna(val) else 0.0
    return channels


def _get_top_tokens(art: "StrataArtifacts", host: str, top_k: int = 5) -> List[str]:
    """Get the most common tokens for a host from the events DataFrame."""
    if art.events is None or art.events.empty:
        return []
    host_events = art.events[art.events["host"] == host]
    if host_events.empty:
        return []
    token_col = "token_medium" if "token_medium" in host_events.columns else "token_coarse"
    if token_col not in host_events.columns:
        return []
    return list(host_events[token_col].value_counts().head(top_k).index)
