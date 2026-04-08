"""
Graph utilities for transition-based explainability.
======================================================
From v12_modular graph.py (cleanest implementation).
"""
from __future__ import annotations

import pandas as pd
import networkx as nx
from typing import Tuple


def build_transition_graph(trans_df: pd.DataFrame, host: str = None) -> nx.DiGraph:
    """
    Build a weighted directed graph from transition counts.
    Nodes = token states, edges = (state -> next_state), weight = count.
    """
    t = trans_df.copy()
    if host is not None:
        t = t[t["host"] == host]

    if t.empty:
        return nx.DiGraph()

    agg = (
        t.groupby(["state", "next_state"])["count"]
        .sum()
        .reset_index()
    )
    G = nx.DiGraph()
    for _, row in agg.iterrows():
        G.add_edge(row["state"], row["next_state"], weight=int(row["count"]))
    return G


def compute_graph_metrics(G: nx.DiGraph) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Node degree centrality + edge betweenness.
    High-betweenness edges are chokepoints in the process execution chain.
    """
    if G is None or len(G) == 0:
        return (
            pd.DataFrame(columns=["state", "degree_centrality"]),
            pd.DataFrame(columns=["state", "next_state", "edge_betweenness"]),
        )

    deg = nx.degree_centrality(G)
    nodes_df = pd.DataFrame([{"state": n, "degree_centrality": v} for n, v in deg.items()])

    k = min(500, len(G.edges())) if len(G.edges()) > 0 else None
    edge_bet = nx.edge_betweenness_centrality(G, k=k)
    edges_df = pd.DataFrame([
        {"state": u, "next_state": v, "edge_betweenness": eb}
        for (u, v), eb in edge_bet.items()
    ])
    return nodes_df, edges_df


def top_rare_transitions(
    trans_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    host: str,
    n: int = 15,
) -> pd.DataFrame:
    """
    Return the transitions for a host with the lowest peer baseline probability.
    This is the primary explainability output for the sequence channel.
    From pipeline_updated dashboard.py.
    """
    h = trans_df[trans_df["host"] == host].copy()
    m = h.merge(baseline_df, on=["state", "next_state", "dt_bucket"], how="left")
    m["p_baseline"] = m["p_baseline"].fillna(0.0)
    return m.sort_values(["p_baseline", "count"], ascending=[True, False]).head(n)
