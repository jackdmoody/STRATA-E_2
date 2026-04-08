"""
Visualization utilities.
==========================
From v12_modular visuals.py (most complete implementation).
Plotly for interactive analysis; matplotlib for static exports.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Static matplotlib outputs (for batch reports)
# ---------------------------------------------------------------------------

def plot_score_histogram(triage: pd.DataFrame, out_dir: Path) -> Path:
    path = out_dir / "score_histogram.png"
    plt.figure(figsize=(10, 5))
    triage["score"].plot(kind="hist", bins=30, edgecolor="black")
    plt.title("Host Anomaly Score Distribution (STRATA-E)")
    plt.xlabel("Fused Anomaly Score (higher = more suspicious)")
    plt.ylabel("Number of Hosts")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_top_hosts(triage: pd.DataFrame, out_dir: Path, top_n: int = 20) -> Path:
    path = out_dir / "top_hosts.png"
    top = triage.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    colors = ["#d62728" if not row.get("gate_pass", True) else "#1f77b4"
              for _, row in top.iterrows()]
    ax.barh(top["host"].astype(str)[::-1], top["score"][::-1], color=colors[::-1])
    ax.set_xlabel("Anomaly Score")
    ax.set_title(f"Top {top_n} Hosts by Anomaly Score (red = failed corroboration gate)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_channel_scores(triage: pd.DataFrame, out_dir: Path, top_n: int = 20) -> Path:
    """Stacked bar showing per-channel contribution for top hosts."""
    path = out_dir / "channel_breakdown.png"
    top = triage.head(top_n).set_index("host")
    channels = [c for c in ["S_seq", "S_freq", "S_ctx", "S_drift"] if c in top.columns]
    if not channels:
        return path

    top[channels].plot(kind="barh", stacked=True, figsize=(12, max(6, top_n * 0.4)))
    plt.title(f"Channel Score Breakdown – Top {top_n} Hosts")
    plt.xlabel("Channel Contribution")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Interactive Plotly outputs (for notebook exploration)
# ---------------------------------------------------------------------------

def plot_sankey_for_host(
    pairs_df: pd.DataFrame,
    host: str,
    host_col: str = "host",
    source_col: str = "src_event",
    target_col: str = "dst_event",
    value_name: str = "count",
):
    """Sankey diagram of critical event co-occurrence pairs for one host."""
    if not PLOTLY_AVAILABLE:
        print("plotly not installed; run: pip install plotly")
        return

    hdf = pairs_df[pairs_df[host_col] == host]
    if hdf.empty:
        print(f"No pairs for host: {host}")
        return

    df_s = hdf.groupby([source_col, target_col]).size().reset_index(name=value_name)
    all_nodes = sorted(
        set(df_s[source_col].dropna().unique()) | set(df_s[target_col].dropna().unique())
    )
    idx = {n: i for i, n in enumerate(all_nodes)}

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=15, label=[str(n) for n in all_nodes]),
        link=dict(
            source=df_s[source_col].map(idx).tolist(),
            target=df_s[target_col].map(idx).tolist(),
            value=df_s[value_name].tolist(),
        ),
    )])
    fig.update_layout(title_text=f"Critical Event Pairs – {host}", font=dict(size=12))
    fig.show()


def plot_host_markov_heatmap(
    trans_df: pd.DataFrame,
    host: str,
):
    """Transition probability heatmap for one host's Markov chain."""
    if not PLOTLY_AVAILABLE:
        print("plotly not installed")
        return

    h = trans_df[trans_df["host"] == host].copy()
    if h.empty:
        print(f"No transitions for host: {host}")
        return

    states = sorted(set(h["state"].unique()) | set(h["next_state"].unique()))
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    mat = np.zeros((n, n))

    for _, row in h.iterrows():
        i, j = idx[row["state"]], idx[row["next_state"]]
        mat[i, j] += row["count"]

    # Row-normalize
    rs = mat.sum(axis=1, keepdims=True)
    mat = mat / (rs + 1e-12)

    fig = px.imshow(
        mat,
        x=states, y=states,
        labels=dict(x="Next State", y="Current State", color="P(Next|Current)"),
        title=f"Markov Transition Matrix – {host}",
        aspect="auto",
    )
    fig.update_xaxes(side="top")
    fig.show()


def plot_host_timeline(df: pd.DataFrame, host: str, host_col: str = "host"):
    """Event timeline for a single host, colored by tactic if available."""
    if not PLOTLY_AVAILABLE:
        print("plotly not installed")
        return

    hdf = df[df[host_col] == host].sort_values("ts")
    if hdf.empty:
        print(f"No events for host: {host}")
        return

    color_col = "mitre_tactic" if "mitre_tactic" in hdf.columns else "token_coarse"
    fig = px.scatter(
        hdf, x="ts", y="event_id",
        color=color_col,
        hover_data=["image", "cmdline"] if "image" in hdf.columns else None,
        title=f"Event Timeline – {host}",
    )
    fig.show()
