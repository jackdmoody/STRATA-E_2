"""
STRATA-E Unified Configuration
================================
Combines the best config patterns from all three pipeline versions:
  - mvp_repo:           flexible column detection, I/O config
  - pipeline_updated:   time bucketing, role config, sub-config dataclasses
  - v12_modular:        frozen dataclass, as_dict(), cache/debug flags

Adds:
  - AblationConfig for toggling components (Fix 4)
  - Adaptive tau_gap flag (Fix 3)
  - Borda/corroboration fusion config (Fix 1)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict, Any
import json


# ---------------------------------------------------------------------------
# Sub-configs (from pipeline_updated, extended)
# ---------------------------------------------------------------------------

@dataclass
class IOConfig:
    """I/O paths and column detection (from mvp_repo)."""
    input_path: Path = Path("data/sample_sysmon.csv")
    output_dir: Path = Path("output")

    # Timestamp detection - ordered candidates (first found wins)
    timestamp_cols: Tuple[str, ...] = ("_timestamp", "UtcTime", "ts", "timestamp", "TimeCreated")
    host_cols: Tuple[str, ...] = ("host.fqdn", "Computer", "Host", "Hostname", "host")
    event_id_cols: Tuple[str, ...] = ("winlog.event_id", "EventID", "EventId", "event_id")
    image_cols: Tuple[str, ...] = ("Image", "ProcessImage", "process_image", "ProcessName")
    parent_image_cols: Tuple[str, ...] = ("ParentImage", "ParentProcessName", "parent_image")
    cmdline_cols: Tuple[str, ...] = ("CommandLine", "CmdLine", "cmdline", "command_line")
    user_cols: Tuple[str, ...] = ("User", "UserName", "SubjectUserName", "user")
    integrity_cols: Tuple[str, ...] = ("IntegrityLevel", "integrity_level")
    signed_cols: Tuple[str, ...] = ("Signed", "signed")

    # Optional time window filtering
    time_min: Optional[str] = None
    time_max: Optional[str] = None

    # Misc
    debug: bool = False
    cache_dir: Optional[str] = None


@dataclass
class TimeBucketingConfig:
    """Session gap + inter-event time bucketing (from pipeline_updated)."""
    buckets: List[Tuple[float, str]] = field(default_factory=lambda: [
        (10,           "<10s"),
        (60,           "10-60s"),
        (5  * 60,      "1-5m"),
        (30 * 60,      "5-30m"),
        (2  * 60 * 60, "30m-2h"),
        (float("inf"), ">2h"),
    ])
    # Fixed fallback gap; overridden per-role when use_adaptive_tau_gap=True
    session_gap_seconds: int = 30 * 60


@dataclass
class BaselineConfig:
    """Peer baseline and Dirichlet shrinkage (combines pipeline_updated + README spec)."""
    # Laplace smoothing alpha for transition probabilities
    laplace_alpha: float = 0.5

    # Dirichlet shrinkage concentration κ (Fix 4 / README spec)
    # Higher = more shrinkage toward role baseline for sparse hosts
    dirichlet_kappa: float = 10.0

    # Minimum events per host; below this, rely entirely on role baseline
    min_events_per_host: int = 25

    # Bootstrap samples for JSD null distribution calibration
    bootstrap_samples: int = 1000

    # Rolling window for drift channel history
    rolling_window_days: int = 7

    # Backoff weights for coarse/medium token mixing (from pipeline_updated)
    backoff_lambdas: Tuple[float, float] = (0.7, 0.3)

    # Baseline host selection (from v12_modular)
    baseline_host_allowlist: Optional[Sequence[str]] = None
    baseline_top_n_hosts: int = 20


@dataclass
class RoleConfig:
    """Host role inference configuration."""
    # Hostname regex patterns for role assignment.
    # Each entry is (regex_pattern, role_name). First match wins.
    # Set to None to use the built-in DEFAULT_ROLE_PATTERNS.
    # Override with environment-specific patterns for your fleet.
    role_patterns: Optional[List[Tuple[str, str]]] = None

    # Feature columns used for rate feature computation (still used by
    # the frequency channel even though role inference no longer clusters)
    role_feature_cols: List[str] = field(default_factory=lambda: [
        "proc_rate_total",
        "script_rate",
        "office_rate",
        "lolbin_rate",
        "has_encoded_rate",
        "has_download_cradle_rate",
        "unique_users",
        "unique_parents",
    ])
    min_hosts_per_role: int = 10


@dataclass
class ScoringConfig:
    """Multi-channel fusion and gating (from pipeline_updated, upgraded per Fix 1)."""
    # Fusion method: 'weighted_linear' matches paper Eq. 15 (default); 'borda' is more robust
    # but does not match the paper's stated formulation.
    fusion_method: str = "weighted_linear"

    # Static weights - ONLY used if fusion_method='weighted_linear'
    # Prefer learning these from injection ground truth (Fix 1)
    w_seq: float = 0.40
    w_freq: float = 0.30
    w_ctx: float = 0.20
    w_drift: float = 0.10

    # Corroboration gate thresholds (Fix 1 - promoted from optional to required)
    require_corroboration: bool = True
    min_corroborating_channels: int = 2
    gate_percentile_threshold: float = 75.0

    # Extreme-channel bypass: single-channel extreme still passes gate
    extreme_threshold: float = 0.95

    # Isolation Forest params for frequency channel
    iforest_contamination: float = 0.02
    iforest_n_estimators: int = 100
    random_seed: int = 42

    # TF-IDF max features for cmdline novelty scoring
    tfidf_max_features: int = 5000

    # Max unique baseline commands to sample for TF-IDF similarity.
    # Higher = more accurate but slower. 1000 is a good tradeoff.
    tfidf_baseline_samples: int = 1000

    # Event severity grading (from v12_modular mapping.py)
    severity_score_col: str = "severity_score"
    severity_label_col: str = "severity_label"
    critical_labels: Sequence[str] = field(default_factory=lambda: ["critical", "high"])
    window_seconds: int = 60  # pair correlation window

    # Events to drop before any scoring
    drop_event_ids: Sequence[int] = field(default_factory=tuple)


@dataclass
class AblationConfig:
    """
    Feature flags for ablation studies (Fix 4).
    Toggle components on/off; run the same pipeline.run() code for all conditions.
    """
    use_role_baselining: bool = True
    use_dirichlet_shrinkage: bool = True
    use_jsd_calibration: bool = True
    use_context_channel: bool = True
    use_drift_channel: bool = True
    use_seq_drift_covariance: bool = True   # Fix 2: explicit covariance meta-feature
    use_corroboration_gate: bool = True      # Fix 1: promoted from optional
    use_adaptive_tau_gap: bool = True        # Fix 3: role-fitted session gap
    use_cmdline_embeddings: bool = True      # Fix 6: TF-IDF novelty beyond keyword list
    use_shrinkage_anomaly: bool = True       # Fix 5: track κ drift as evasion signal

    # --- Preset factories for ablation study conditions ---
    @classmethod
    def full_pipeline(cls) -> "AblationConfig":
        return cls()

    @classmethod
    def no_role_baselining(cls) -> "AblationConfig":
        c = cls(); c.use_role_baselining = False; c.use_dirichlet_shrinkage = False; return c

    @classmethod
    def no_shrinkage(cls) -> "AblationConfig":
        c = cls(); c.use_dirichlet_shrinkage = False; return c

    @classmethod
    def no_calibration(cls) -> "AblationConfig":
        c = cls(); c.use_jsd_calibration = False; return c

    @classmethod
    def sequence_only(cls) -> "AblationConfig":
        c = cls()
        c.use_context_channel = False
        c.use_drift_channel = False
        c.use_seq_drift_covariance = False
        return c

    @classmethod
    def no_drift(cls) -> "AblationConfig":
        c = cls(); c.use_drift_channel = False; c.use_seq_drift_covariance = False; return c


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class StrataConfig:
    """
    Master config for STRATA-E unified pipeline.

    Usage:
        cfg = StrataConfig()                          # full defaults
        cfg = StrataConfig(ablation=AblationConfig.sequence_only())
        cfg = StrataConfig.from_json("config.json")
        cfg = StrataConfig.fast()                     # reduced-cost dev mode
    """
    io: IOConfig = field(default_factory=IOConfig)
    time: TimeBucketingConfig = field(default_factory=TimeBucketingConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    role: RoleConfig = field(default_factory=RoleConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    # Token resolution for transition modeling
    # 'medium' is the standard operating mode; 'coarse' for backoff
    token_resolution: str = "medium"

    def apply_fast_preset(self) -> "StrataConfig":
        """
        Apply --fast settings for iterative development runs.
        Reduces bootstrap samples, IsolationForest trees, and TF-IDF features
        to cut runtime by 3-10× while keeping all channels and SHAP active.

        Does NOT disable any channels or analytical components — just reduces
        the computational cost of each. Paper-quality runs should use defaults.
        """
        self.baseline.bootstrap_samples = 200       # down from 1000
        self.scoring.iforest_n_estimators = 50      # down from 100
        self.scoring.tfidf_max_features = 300       # down from 5000
        self.scoring.tfidf_baseline_samples = 500   # down from 1000
        return self

    @classmethod
    def fast(cls) -> "StrataConfig":
        """Factory for a fast-preset config."""
        return cls().apply_fast_preset()

    def as_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.as_dict(), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: str) -> "StrataConfig":
        with open(path) as f:
            d = json.load(f)
        return cls(
            io=IOConfig(**d.get("io", {})),
            time=TimeBucketingConfig(**d.get("time", {})),
            baseline=BaselineConfig(**d.get("baseline", {})),
            role=RoleConfig(**d.get("role", {})),
            scoring=ScoringConfig(**d.get("scoring", {})),
            ablation=AblationConfig(**d.get("ablation", {})),
            token_resolution=d.get("token_resolution", "medium"),
        )