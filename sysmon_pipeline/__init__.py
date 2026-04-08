"""
STRATA-E: Structural and Temporal Role-Aware Threat Analytics for Endpoint Telemetry.

A hierarchical, statistically calibrated behavioral anomaly detection
architecture for Windows Sysmon and Windows Event telemetry. Designed to
answer: "Which hosts are behaving abnormally, why, and based on what evidence?"

Architecture
------------
- Multi-resolution token abstraction (coarse / medium / fine)
- Hierarchical Dirichlet shrinkage peer-role baselines
- Four independent detection channels: sequence, frequency, context, drift
- Bootstrap-calibrated JSD significance testing
- Borda rank fusion with corroboration gate

Quick start
-----------
    from sysmon_pipeline import StrataPipeline, StrataConfig
    from sysmon_pipeline.loaders import load_sysmon_csv

    cfg = StrataConfig()
    df  = load_sysmon_csv("data/sysmon_export.csv")

    pipe   = StrataPipeline(cfg)
    fitted = pipe.fit(df)
    art    = pipe.score(df, fitted)
    print(art.triage.head(20))

Run with HTML report
--------------------
    from sysmon_pipeline import StrataPipeline, StrataConfig
    from sysmon_pipeline.report import ReportContext

    cfg = StrataConfig()
    df  = load_sysmon_csv("data/sysmon_export.csv")

    with ReportContext(output_dir="results", open_browser=True) as report:
        pipe   = StrataPipeline(cfg)
        fitted = pipe.fit(df)
        art    = pipe.score(df, fitted)
        report.finalise(art)

One-liner with report
---------------------
    from sysmon_pipeline.report import run_with_report
    run_with_report("data/sysmon_export.csv", output_dir="results")
"""
from .config import (
    StrataConfig,
    AblationConfig,
    IOConfig,
    TimeBucketingConfig,
    BaselineConfig,
    RoleConfig,
    ScoringConfig,
)
from .pipeline import StrataPipeline, StrataArtifacts, FittedArtifacts
from .loaders import load_darpa_tc, load_sysmon, load_sysmon_csv, split_time_windows
from .report import ReportContext

__version__ = "1.0.0"

__all__ = [
    # Pipeline
    "StrataPipeline",
    "StrataArtifacts",
    "FittedArtifacts",
    "ReportContext",
    # Config
    "StrataConfig",
    "AblationConfig",
    "IOConfig",
    "TimeBucketingConfig",
    "BaselineConfig",
    "RoleConfig",
    "ScoringConfig",
    # Loaders
    "load_darpa_tc",
    "load_sysmon",
    "load_sysmon_csv",
    "split_time_windows",
]
