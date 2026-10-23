"""
Tests for calibrated (Track C) scoring — sysmon_pipeline/calibrated.py.

Properties checked:
  1. Rank-mode outputs are unchanged when a calibrator is passed.
  2. Calibrated per-host results are batch-independent (a host's p-values do
     not change when other hosts are removed from the scored window), unlike
     rank mode.
  3. FP validity: on held-out benign windows, P(fused_pvalue <= b) ~<= b.
  4. An anomalous host gets a small fused p-value and passes the gate.
  5. JSON round-trip reproduces transform() exactly.
  6. D2: calib_folds=24 gives S_seq_pvalue resolution finer than 0.05.
"""
from __future__ import annotations

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from sysmon_pipeline import StrataConfig, StrataPipeline, CalibratedScorer
from test_pipeline import make_synthetic

warnings.filterwarnings("ignore")

N_HOSTS = 30
N_EVENTS = 400


def _day(day: int, anomaly: bool = False, n_hosts: int = N_HOSTS) -> pd.DataFrame:
    df = make_synthetic(n_hosts=n_hosts, n_events_per_host=N_EVENTS,
                        inject_anomaly=anomaly, anomaly_host_idx=7, seed=1000 + day)
    df["_timestamp"] = df["_timestamp"] + timedelta(days=day)
    return df


@pytest.fixture(scope="module")
def setup():
    cfg = StrataConfig.fast()
    pipe = StrataPipeline(cfg)
    baseline = pd.concat([_day(0), _day(1)], ignore_index=True)
    fitted = pipe.fit(baseline)
    days = {d: _day(d) for d in range(1, 9)}
    # Calibration days 2-5, each scored with the previous day as drift prior.
    calib = [pipe.score(days[d], fitted, prior_window_df=days[d - 1]) for d in range(2, 6)]
    cal = CalibratedScorer.fit(calib, cfg, min_reference=50)
    return cfg, pipe, fitted, days, cal


def test_rank_outputs_unchanged(setup):
    cfg, pipe, fitted, days, cal = setup
    a = pipe.score(days[6], fitted, prior_window_df=days[5])
    b = pipe.score(days[6], fitted, prior_window_df=days[5], calibrator=cal)
    cols = ["host", "S_seq", "S_freq", "S_ctx", "S_drift", "score", "gate_pass", "gate_reason"]
    pd.testing.assert_frame_equal(a.triage[cols], b.triage[cols])
    assert a.calibrated is None and b.calibrated is not None


def test_batch_independence(setup):
    cfg, pipe, fitted, days, cal = setup
    full = pipe.score(days[6], fitted, prior_window_df=days[5], calibrator=cal)
    keep = sorted(days[6]["host.fqdn"].unique())[:12]
    sub_df = days[6][days[6]["host.fqdn"].isin(keep)]
    sub_prior = days[5][days[5]["host.fqdn"].isin(keep)]
    sub = pipe.score(sub_df, fitted, prior_window_df=sub_prior, calibrator=cal)

    cal_cols = ["host"] + [f"p_{c}" for c in cal.channels] + ["fused_pvalue", "gate_pass_cal"]
    f = full.calibrated[cal_cols].set_index("host").loc[keep]
    s = sub.calibrated[cal_cols].set_index("host").loc[keep]
    pd.testing.assert_frame_equal(f, s)

    # Contrast: rank-mode scores DO depend on the batch.
    rf = full.triage.set_index("host").loc[keep, "S_freq"]
    rs = sub.triage.set_index("host").loc[keep, "S_freq"]
    assert not np.allclose(rf.to_numpy(), rs.to_numpy())


def test_fp_validity_on_heldout_benign(setup):
    cfg, pipe, fitted, days, cal = setup
    rows = [pipe.score(days[d], fitted, prior_window_df=days[d - 1], calibrator=cal).calibrated
            for d in (6, 7, 8)]
    p = pd.concat(rows)["fused_pvalue"].to_numpy()
    n = len(p)
    for b in (0.1, 0.2, 0.3):
        rate = float(np.mean(p <= b))
        slack = 3.0 * np.sqrt(b * (1 - b) / n)
        assert rate <= b + slack, f"budget {b}: realized {rate:.3f} (n={n})"


def test_anomalous_host_detected(setup):
    cfg, pipe, fitted, days, cal = setup
    bad = _day(6, anomaly=True)
    art = pipe.score(bad, fitted, prior_window_df=days[5], calibrator=cal)
    c = art.calibrated.set_index("host")
    row = c.loc["host07.corp.local"]
    assert row["fused_pvalue"] <= 0.05
    assert bool(row["gate_pass_cal"])
    assert row["fused_pvalue"] == c["fused_pvalue"].min()


def test_json_roundtrip(setup, tmp_path):
    cfg, pipe, fitted, days, cal = setup
    path = tmp_path / "cal.json"
    cal.to_json(str(path))
    cal2 = CalibratedScorer.from_json(str(path))
    art = pipe.score(days[7], fitted, prior_window_df=days[6])
    pd.testing.assert_frame_equal(cal.transform(art), cal2.transform(art))


def test_min_attainable_p_reported(setup):
    *_, cal = setup
    s = cal.calibration_summary
    assert s["min_attainable_fused_p"] == pytest.approx(1.0 / (s["n_host_windows"] + 1))


def test_calib_folds_24_resolution():
    """D2: with calib_folds=24 the self-conformal p-value can go below 0.05."""
    cfg = StrataConfig.fast()
    cfg.scoring.calib_folds = 24
    pipe = StrataPipeline(cfg)
    baseline = pd.concat([_day(0, n_hosts=8), _day(1, n_hosts=8)], ignore_index=True)
    fitted = pipe.fit(baseline)
    folds = [len(v) for v in (fitted.self_sseq or {}).values()]
    assert folds and max(folds) >= 19, f"self folds per host: {folds}"
    assert 1.0 / (max(folds) + 1) <= 0.05
