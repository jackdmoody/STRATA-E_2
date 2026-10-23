"""
Calibrated (batch-independent) scoring — "Track C".
====================================================

Why this exists
---------------
In rank mode (the default, unchanged), every alert decision is relative to the
batch of hosts scored together:

  * S_freq is a within-batch percentile rank of IsolationForest normality.
  * S_ctx components are scored against the batch's role median/MAD.
  * S_seq_drift_cov z-scores S_seq and S_drift within the batch.
  * The fused score is a within-batch percentile rank.
  * The corroboration gate thresholds each channel at the batch's 75th
    percentile, and the extreme bypass (>= 0.95) on a rank channel passes
    the top ~5% of hosts by construction.

So the alert rate on a scored window is largely fixed by fleet size, and no
threshold corresponds to a target false-positive rate per host-window.

What this module does
---------------------
``CalibratedScorer`` is fit on scored windows of *benign* data (the
calibration window). It freezes:

  1. the role medians/MADs used to peer-score the context components,
  2. the S_seq / S_drift moments used by the covariance channel,
  3. a reference distribution of each channel's batch-independent statistic
     (per role when the role has enough host-windows, else global), and
  4. a reference distribution of the fused statistic.

At score time each channel becomes a conformal p-value against its frozen
reference, the gate uses those p-values with the SAME thresholds as rank mode
(``gate_percentile_threshold``, ``extreme_threshold``,
``min_corroborating_channels``), and the fused statistic becomes a conformal
p-value. Under exchangeability of benign host-windows, an alert rule
``fused_pvalue <= b`` has a false-positive rate of at most b per host-window;
whether that holds on real data is what the evaluation measures.

Rank-mode outputs are not touched: this module only reads StrataArtifacts.

Usage
-----
    pipe   = StrataPipeline(cfg)
    fitted = pipe.fit(baseline_df)
    calib_arts = [pipe.score(day_df, fitted, prior_window_df=prev_df)
                  for day_df, prev_df in calibration_days]
    cal = CalibratedScorer.fit(calib_arts, cfg)
    art = pipe.score(test_day_df, fitted, prior_window_df=prev_test_df,
                     calibrator=cal)
    art.calibrated            # per-host p-values, fused p-value, gate
    cal.alerts(art.calibrated, fp_budget=0.05)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from .config import StrataConfig

logger = logging.getLogger("strata")

GLOBAL = "__global__"

# channel name -> batch-independent statistic column in the host-window frame
CHANNEL_STAT = {
    "seq":   "S_seq",
    "freq":  "S_freq_raw",
    "ctx":   "S_ctx_frozen",
    "drift": "S_drift",
    "cov":   "S_cov_frozen",
}

# Context components, mirroring scoring.score_context (weights and order).
_CTX_COMPONENTS = ("severity_mean", "ctx_flag_rate", "ctx_pair_rate", "cmdline_novelty")
_CTX_WEIGHTS = {"severity_mean": 0.30, "ctx_flag_rate": 0.35,
                "ctx_pair_rate": 0.20, "cmdline_novelty": 0.15}
_CTX_MIN_ROLE_HOSTS = 5   # same fallback rule as scoring._peer_robust_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def active_channels(cfg: StrataConfig) -> List[str]:
    """Same channel set as scoring.fuse_scores, by ablation flags."""
    ch = ["seq", "freq", "ctx"]
    if cfg.ablation.use_drift_channel:
        ch.append("drift")
        if cfg.ablation.use_seq_drift_covariance:
            ch.append("cov")
    return ch


def host_window_frame(art) -> pd.DataFrame:
    """One row per host for a scored window, with raw (pre-rank) statistics."""
    if art.seq_scores is None or art.freq_scores is None or art.ctx_scores is None:
        raise ValueError("StrataArtifacts is missing channel score tables.")
    if "S_freq_raw" not in art.freq_scores.columns:
        raise ValueError("freq_scores has no S_freq_raw; artifacts predate calibrated scoring.")
    seq = art.seq_scores[["host"] + [c for c in ("role_id", "S_seq") if c in art.seq_scores.columns]]
    freq = art.freq_scores[["host", "S_freq_raw"]]
    ctx_cols = ["host"] + [c for c in _CTX_COMPONENTS + ("ctx_n_tactics",)
                           if c in art.ctx_scores.columns]
    ctx = art.ctx_scores[ctx_cols]
    df = seq.merge(freq, on="host", how="outer").merge(ctx, on="host", how="outer")
    if art.drift_scores is not None and not art.drift_scores.empty:
        df = df.merge(art.drift_scores[["host", "S_drift"]], on="host", how="left")
    else:
        df["S_drift"] = 0.0
    if "role_id" not in df.columns or df["role_id"].isna().any():
        roles = art.host_roles.set_index("host")["role_id"] if art.host_roles is not None else None
        fill = df["host"].map(roles) if roles is not None else "default"
        df["role_id"] = df.get("role_id", pd.Series(index=df.index, dtype=object)).fillna(fill)
    df["role_id"] = df["role_id"].fillna("default").astype(str)
    for c in ("S_seq", "S_freq_raw", "S_drift", "ctx_n_tactics") + _CTX_COMPONENTS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df.reset_index(drop=True)


def _count_ge(sorted_ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Number of reference values >= each x."""
    return len(sorted_ref) - np.searchsorted(sorted_ref, x, side="left")


def conformal_pvalue(sorted_ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """p = (1 + #{ref >= x}) / (n + 1). Super-uniform under exchangeability."""
    x = np.asarray(x, dtype=float)
    n = len(sorted_ref)
    return (1.0 + _count_ge(sorted_ref, x)) / (n + 1.0)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

@dataclass
class CalibratedScorer:
    channels: List[str]
    # role -> component -> (median, mad); GLOBAL key always present
    ctx_params: Dict[str, Dict[str, List[float]]]
    cov_moments: Dict[str, float]             # seq_mean, seq_std, drift_mean, drift_std
    channel_refs: Dict[str, Dict[str, np.ndarray]]   # channel -> role -> sorted ref
    fused_ref: np.ndarray                     # sorted LOO fused statistics
    gate_percentile_threshold: float
    extreme_threshold: float
    min_corroborating_channels: int
    use_gate: bool
    min_reference: int
    n_calibration_host_windows: int
    calibration_summary: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(
        cls,
        artifacts: Sequence,
        cfg: StrataConfig,
        exclude_hosts: Optional[Sequence[Set[str]]] = None,
        min_reference: int = 50,
    ) -> "CalibratedScorer":
        """
        Fit on scored BENIGN windows (e.g. clean calibration days).

        artifacts:     StrataArtifacts from pipe.score(), one per window.
        exclude_hosts: optional per-window sets of hosts to drop (e.g. hosts
                       with injected or known-malicious activity).
        min_reference: minimum host-windows for a role-specific channel
                       reference; smaller roles use the global reference.
        """
        frames = []
        for i, art in enumerate(artifacts):
            f = host_window_frame(art)
            if exclude_hosts is not None and exclude_hosts[i]:
                f = f[~f["host"].isin(exclude_hosts[i])]
            f["_window"] = i
            frames.append(f)
        ref = pd.concat(frames, ignore_index=True)
        if ref.empty:
            raise ValueError("No calibration host-windows.")
        channels = active_channels(cfg)

        # 1. Frozen context peer parameters
        ctx_params: Dict[str, Dict[str, List[float]]] = {}
        ctx_params[GLOBAL] = {c: _med_mad(ref[c].to_numpy()) for c in _CTX_COMPONENTS}
        for role, g in ref.groupby("role_id"):
            if g["host"].nunique() >= _CTX_MIN_ROLE_HOSTS:
                ctx_params[str(role)] = {c: _med_mad(g[c].to_numpy()) for c in _CTX_COMPONENTS}

        # 2. Frozen covariance moments
        cov_moments = {
            "seq_mean": float(ref["S_seq"].mean()), "seq_std": float(ref["S_seq"].std(ddof=0)),
            "drift_mean": float(ref["S_drift"].mean()), "drift_std": float(ref["S_drift"].std(ddof=0)),
        }

        scorer = cls(
            channels=channels, ctx_params=ctx_params, cov_moments=cov_moments,
            channel_refs={}, fused_ref=np.array([]),
            gate_percentile_threshold=float(cfg.scoring.gate_percentile_threshold),
            extreme_threshold=float(cfg.scoring.extreme_threshold),
            min_corroborating_channels=int(cfg.scoring.min_corroborating_channels),
            use_gate=bool(cfg.ablation.use_corroboration_gate),
            min_reference=int(min_reference),
            n_calibration_host_windows=int(len(ref)),
        )
        ref = scorer._add_frozen_stats(ref)

        # 3. Channel references (per role if large enough, else global)
        for ch in channels:
            col = CHANNEL_STAT[ch]
            refs = {GLOBAL: np.sort(ref[col].to_numpy(dtype=float))}
            for role, g in ref.groupby("role_id"):
                if len(g) >= min_reference:
                    refs[str(role)] = np.sort(g[col].to_numpy(dtype=float))
            scorer.channel_refs[ch] = refs

        # 4. Fused reference from LEAVE-ONE-OUT channel p-values, so a
        #    calibration row is never scored against a reference containing itself.
        p_loo = scorer._channel_pvalues(ref, loo=True)
        scorer.fused_ref = np.sort(scorer._fused_stat(p_loo))

        # Descriptive: LOO gate pass rate on benign calibration windows.
        gate, _, _ = scorer._gate(p_loo)
        scorer.calibration_summary = {
            "n_host_windows": float(len(ref)),
            "n_windows": float(ref["_window"].nunique()),
            "loo_gate_pass_rate": float(np.mean(gate)),
            "min_attainable_fused_p": float(1.0 / (len(scorer.fused_ref) + 1)),
        }
        logger.info("CalibratedScorer.fit(): %d host-windows, LOO gate pass rate %.3f, "
                    "min attainable fused p %.4f", len(ref),
                    scorer.calibration_summary["loo_gate_pass_rate"],
                    scorer.calibration_summary["min_attainable_fused_p"])
        return scorer

    # ------------------------------------------------------------ transform
    def transform(self, art) -> pd.DataFrame:
        """Calibrated per-host table for one scored window."""
        df = self._add_frozen_stats(host_window_frame(art))
        p = self._channel_pvalues(df, loo=False)
        fused = self._fused_stat(p)
        gate, reason, n_above = self._gate(p)
        out = df[["host", "role_id"]].copy()
        for ch in self.channels:
            out[f"p_{ch}"] = p[ch]
        out["S_ctx_frozen"] = df["S_ctx_frozen"].to_numpy()
        if "cov" in self.channels:
            out["S_cov_frozen"] = df["S_cov_frozen"].to_numpy()
        out["fused_stat"] = fused
        out["fused_pvalue"] = conformal_pvalue(self.fused_ref, fused)
        out["n_channels_above_cal"] = n_above
        out["gate_pass_cal"] = gate
        out["gate_reason_cal"] = reason
        return out.sort_values("fused_pvalue", kind="mergesort").reset_index(drop=True)

    def alerts(self, calibrated: pd.DataFrame, fp_budget: float, gated: bool = True) -> pd.Series:
        """Alert rule at a per-host-window false-positive budget."""
        a = calibrated["fused_pvalue"] <= fp_budget
        if gated:
            a = a & calibrated["gate_pass_cal"]
        return a

    # ------------------------------------------------------------- internals
    def _add_frozen_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        comps = {}
        for c in _CTX_COMPONENTS:
            vals = df[c].to_numpy(dtype=float)
            out = np.zeros(len(df))
            for i, role in enumerate(df["role_id"].to_numpy()):
                med, mad = self.ctx_params.get(str(role), self.ctx_params[GLOBAL])[c]
                scale = 1.4826 * mad
                if scale <= 1e-9:
                    continue
                z = max((vals[i] - med) / scale, 0.0)
                out[i] = 1.0 - np.exp(-z / 2.0)
            comps[c] = out
        tactic_bonus = np.clip((df["ctx_n_tactics"].to_numpy(dtype=float) - 1) * 0.10, 0.0, 0.30)
        comps["ctx_pair_rate"] = np.clip(comps["ctx_pair_rate"] + tactic_bonus, 0.0, 1.0)
        s = sum(_CTX_WEIGHTS[c] * comps[c] for c in _CTX_COMPONENTS)
        df["S_ctx_frozen"] = np.clip(s, 0.0, 1.0)

        m = self.cov_moments
        zs = (df["S_seq"] - m["seq_mean"]) / (m["seq_std"] + 1e-9)
        zd = (df["S_drift"] - m["drift_mean"]) / (m["drift_std"] + 1e-9)
        df["S_cov_frozen"] = (zs * zd).clip(lower=0.0)
        return df

    def _ref_for(self, ch: str, role: str) -> np.ndarray:
        refs = self.channel_refs[ch]
        return refs.get(str(role), refs[GLOBAL])

    def _channel_pvalues(self, df: pd.DataFrame, loo: bool) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        roles = df["role_id"].astype(str).to_numpy()
        for ch in self.channels:
            x = df[CHANNEL_STAT[ch]].to_numpy(dtype=float)
            p = np.empty(len(df))
            for role in np.unique(roles):
                idx = np.where(roles == role)[0]
                ref = self._ref_for(ch, role)
                if loo:
                    # The row is a member of its reference: drop itself.
                    p[idx] = _count_ge(ref, x[idx]) / max(len(ref), 1)
                else:
                    p[idx] = conformal_pvalue(ref, x[idx])
            out[ch] = np.clip(p, 0.0, 1.0)
        return out

    def _fused_stat(self, p: Dict[str, np.ndarray]) -> np.ndarray:
        # Borda analogue: rank mode sums within-batch percentiles; here each
        # channel contributes its percentile against the frozen reference.
        return np.sum([1.0 - p[ch] for ch in self.channels], axis=0)

    def _gate(self, p: Dict[str, np.ndarray]):
        n = len(next(iter(p.values())))
        if not self.use_gate:
            return np.ones(n, bool), np.array(["no_gating"] * n), np.zeros(n, int)
        above_p = 1.0 - self.gate_percentile_threshold / 100.0
        extreme_p = 1.0 - self.extreme_threshold
        P = np.column_stack([p[ch] for ch in self.channels])
        n_above = (P <= above_p).sum(axis=1)
        multi = n_above >= self.min_corroborating_channels
        extreme = (P <= extreme_p).any(axis=1)
        gate = multi | extreme
        reason = np.where(extreme, "extreme_channel",
                          np.where(multi, "multi_channel", "low_support"))
        return gate, reason, n_above

    # --------------------------------------------------------- persistence
    def to_json(self, path: str) -> None:
        d = {
            "channels": self.channels, "ctx_params": self.ctx_params,
            "cov_moments": self.cov_moments,
            "channel_refs": {ch: {r: v.tolist() for r, v in refs.items()}
                             for ch, refs in self.channel_refs.items()},
            "fused_ref": self.fused_ref.tolist(),
            "gate_percentile_threshold": self.gate_percentile_threshold,
            "extreme_threshold": self.extreme_threshold,
            "min_corroborating_channels": self.min_corroborating_channels,
            "use_gate": self.use_gate, "min_reference": self.min_reference,
            "n_calibration_host_windows": self.n_calibration_host_windows,
            "calibration_summary": self.calibration_summary,
        }
        with open(path, "w") as fh:
            json.dump(d, fh)

    @classmethod
    def from_json(cls, path: str) -> "CalibratedScorer":
        with open(path) as fh:
            d = json.load(fh)
        d["channel_refs"] = {ch: {r: np.asarray(v, float) for r, v in refs.items()}
                             for ch, refs in d["channel_refs"].items()}
        d["fused_ref"] = np.asarray(d["fused_ref"], float)
        return cls(**d)


def _med_mad(x: np.ndarray) -> List[float]:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return [0.0, 0.0]
    med = float(np.median(x))
    return [med, float(np.median(np.abs(x - med)))]
