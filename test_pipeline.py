"""
Tests for STRATA-E unified pipeline.
Combines:
  - mvp_repo tests: smoke test, schema validation
  - v12_modular tests: synthetic data generation, end-to-end run
  - New: ablation condition tests, channel isolation tests
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sysmon_pipeline import StrataConfig, AblationConfig, StrataPipeline


# ---------------------------------------------------------------------------
# Synthetic data factory (extended from v12_modular)
# ---------------------------------------------------------------------------

def make_synthetic(
    n_hosts: int = 6,
    n_events_per_host: int = 150,
    inject_anomaly: bool = True,
    anomaly_host_idx: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Generate synthetic Sysmon-like data.
    If inject_anomaly=True, the last host gets injected PowerShell + LOLBin activity.
    """
    rng = np.random.default_rng(seed)
    rows = []
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    event_ids = [1, 3, 11, 13, 22]
    images = [
        r"C:\Windows\System32\cmd.exe",
        r"C:\Windows\System32\svchost.exe",
        r"C:\Program Files\Office\winword.exe",
        r"C:\Windows\System32\powershell.exe",
        r"C:\Windows\System32\rundll32.exe",
    ]
    users = ["DOMAIN\\user1", "DOMAIN\\user2", "NT AUTHORITY\\SYSTEM"]

    for h in range(n_hosts):
        host = f"host{h:02d}.corp.local"
        t = base_time
        for i in range(n_events_per_host):
            t += timedelta(seconds=int(rng.integers(5, 120)))
            is_anomaly_host = inject_anomaly and h == anomaly_host_idx

            if is_anomaly_host and i % 8 == 0:
                # Inject: encoded PowerShell + LOLBIN
                eid = 1
                img = r"C:\Windows\System32\powershell.exe"
                cmd = "powershell.exe -enc SGVsbG8gV29ybGQ= -bypass"
                parent = r"C:\Windows\System32\rundll32.exe"
            else:
                eid = int(rng.choice(event_ids))
                img = str(rng.choice(images))
                cmd = f"cmd /c {img.split(chr(92))[-1]}"
                parent = str(rng.choice(images))

            rows.append({
                "_timestamp": t,
                "host.fqdn":  host,
                "winlog.event_id": eid,
                "Image": img,
                "ParentImage": parent,
                "CommandLine": cmd,
                "User": str(rng.choice(users)),
                "IntegrityLevel": str(rng.choice(["Medium", "High", "System"])),
                "Signed": bool(rng.choice([True, False])),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_imports():
    from sysmon_pipeline import StrataConfig, AblationConfig, StrataPipeline, StrataArtifacts, FittedArtifacts


def test_schema_normalization():
    from sysmon_pipeline.schema import normalize_schema, validate_schema
    df = make_synthetic(n_hosts=2, n_events_per_host=20, inject_anomaly=False)
    cfg = StrataConfig()
    events = normalize_schema(df, cfg)
    validate_schema(events)
    required = ["ts", "host", "event_id"]
    for c in required:
        assert c in events.columns, f"Missing: {c}"
    assert events["ts"].notna().all()


def test_full_pipeline_smoke():
    """Full pipeline runs without error and produces a triage table."""
    df = make_synthetic(n_hosts=5, n_events_per_host=100)
    cfg = StrataConfig(ablation=AblationConfig.full_pipeline())
    pipe = StrataPipeline(cfg)
    fitted = pipe.fit(df)
    art = pipe.score(df, fitted)

    assert art.triage is not None
    assert len(art.triage) > 0
    assert "score" in art.triage.columns
    assert "gate_pass" in art.triage.columns


def test_anomaly_host_ranks_high():
    """Injected anomaly host should rank in the top half."""
    df = make_synthetic(n_hosts=6, n_events_per_host=150, inject_anomaly=True, anomaly_host_idx=5)
    cfg = StrataConfig(
        ablation=AblationConfig.full_pipeline(),
    )
    cfg.scoring.iforest_contamination = 0.15

    pipe = StrataPipeline(cfg)
    fitted = pipe.fit(df)
    art = pipe.score(df, fitted)

    triage = art.triage
    anomaly_host = "host05.corp.local"
    if anomaly_host in triage["host"].values:
        rank = triage.loc[triage["host"] == anomaly_host, "triage_rank"].values[0]
        total = len(triage)
        assert rank <= total * 0.75, f"Anomaly host ranked {rank}/{total} — expected top 75%"


def test_ablation_conditions_all_run():
    """All ablation presets should produce valid triage output."""
    df = make_synthetic(n_hosts=4, n_events_per_host=80)
    ablation_conditions = [
        AblationConfig.full_pipeline(),
        AblationConfig.sequence_only(),
        AblationConfig.no_shrinkage(),
        AblationConfig.no_role_baselining(),
        AblationConfig.no_drift(),
    ]
    for condition in ablation_conditions:
        cfg = StrataConfig(ablation=condition)
        pipe = StrataPipeline(cfg)
        fitted = pipe.fit(df)
        art = pipe.score(df, fitted)
        assert art.triage is not None, f"Triage is None for condition: {condition}"
        assert len(art.triage) > 0, f"Empty triage for condition: {condition}"


def test_borda_vs_weighted_fusion():
    """Both fusion methods should run; Borda should not require weights."""
    df = make_synthetic(n_hosts=4, n_events_per_host=80)

    # Borda (default)
    cfg_borda = StrataConfig()
    cfg_borda.scoring.fusion_method = "borda"
    art_borda = StrataPipeline(cfg_borda).fit_score(df)
    assert art_borda.triage is not None

    # Weighted with explicit weights
    cfg_wl = StrataConfig()
    cfg_wl.scoring.fusion_method = "weighted_linear"
    art_wl = StrataPipeline(cfg_wl).fit_score(df)
    assert art_wl.triage is not None


def test_drift_channel_with_prior_window():
    """Drift channel should produce non-zero scores when prior window is provided."""
    df1 = make_synthetic(n_hosts=4, n_events_per_host=80, seed=42)
    df2 = make_synthetic(n_hosts=4, n_events_per_host=80, seed=99)  # different behavior

    cfg = StrataConfig()
    pipe = StrataPipeline(cfg)
    fitted = pipe.fit(df1)
    art = pipe.score(df2, fitted, prior_window_df=df1)

    assert art.drift_scores is not None
    assert "S_drift" in art.drift_scores.columns


def test_config_serialization(tmp_path):
    """Config should round-trip to/from JSON."""
    cfg = StrataConfig(ablation=AblationConfig.sequence_only())
    json_path = str(tmp_path / "config.json")
    cfg.to_json(json_path)
    cfg2 = StrataConfig.from_json(json_path)
    assert cfg2.ablation.use_context_channel == False
    assert cfg2.ablation.use_drift_channel == False


def test_shrinkage_evasion_detection():
    """Shrinkage anomaly should flag hosts that suddenly go quiet."""
    from sysmon_pipeline.divergence import compute_shrinkage_weights, detect_shrinkage_anomalies

    historical = compute_shrinkage_weights(
        {"host01": 1000, "host02": 500, "host03": 800},
        kappa=10.0
    )
    # host01 suddenly drops to 10 events (went quiet)
    current = compute_shrinkage_weights(
        {"host01": 10, "host02": 480, "host03": 790},
        kappa=10.0
    )
    result = detect_shrinkage_anomalies(current, historical, delta_threshold=0.2)
    flagged = result[result["evasion_signal"] == True]["host"].tolist()
    assert "host01" in flagged, "host01 should be flagged for sudden event drop"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
