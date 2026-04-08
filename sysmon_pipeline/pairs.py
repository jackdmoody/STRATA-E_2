"""
Per-host rate features, role inference, and critical event pair correlation.
=============================================================================
Merges:
  - pipeline_updated pairs.py: rate feature computation, role feature matrix
  - v12_modular pairs.py:      critical event pair correlation, IsolationForest on pair stats

The two files served different purposes and are genuinely complementary:
  pipeline_updated.pairs -> feeds frequency channel + role clustering
  v12_modular.pairs      -> feeds context channel (temporal co-occurrence of critical events)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import StrataConfig

logger = logging.getLogger("strata")


# ---------------------------------------------------------------------------
# Rate features (pipeline_updated - for frequency channel + role inference)
# ---------------------------------------------------------------------------

def compute_rate_features(df: pd.DataFrame, cfg: StrataConfig) -> pd.DataFrame:
    """
    Compute per-host volumetric rate features.

    These feed two places:
      1. Frequency channel (IsolationForest on rates)
      2. Role inference (clustering on behavioral rates to assign peer groups)

    Produces (per host):
      proc_rate_total, script_rate, office_rate, lolbin_rate,
      has_encoded_rate, has_download_cradle_rate,
      unique_users, unique_parents, event_count, hours
    """
    host_span = df.groupby("host")["ts"].agg(["min", "max"])
    host_span["hours"] = (host_span["max"] - host_span["min"]).dt.total_seconds() / 3600.0
    host_span["hours"] = host_span["hours"].clip(lower=1e-6)

    counts = df.groupby("host").size().rename("event_count").to_frame()
    feats = counts.join(host_span["hours"])
    feats["proc_rate_total"] = feats["event_count"] / feats["hours"]

    for feat_name, token in [
        ("script_rate", "SCRIPT"),
        ("office_rate", "OFFICE"),
        ("lolbin_rate", "LOLBIN"),
        ("browser_rate", "BROWSER"),
    ]:
        c = df[df["token_coarse"] == token].groupby("host").size().rename(feat_name)
        feats = feats.join(c, how="left")
        feats[feat_name] = feats[feat_name].fillna(0.0) / feats["hours"]

    for flag in ["has_encoded", "has_download_cradle", "has_bypass"]:
        feat_name = f"{flag}_rate"
        if flag in df.columns:
            c = df[df[flag] == True].groupby("host").size().rename(feat_name)
            feats = feats.join(c, how="left")
            feats[feat_name] = feats[feat_name].fillna(0.0) / feats["hours"]
        else:
            feats[feat_name] = 0.0

    feats["unique_users"] = df.groupby("host")["user"].nunique(dropna=True)
    feats["unique_parents"] = df.groupby("host")["parent_image"].nunique(dropna=True)

    return feats.reset_index()


def build_role_features(df_rates: pd.DataFrame, cfg: StrataConfig) -> pd.DataFrame:
    """Extract the feature columns used for role inference/clustering."""
    cols = ["host"] + [c for c in cfg.role.role_feature_cols if c in df_rates.columns]
    return df_rates[cols].copy()


# ---------------------------------------------------------------------------
# Role inference
# ---------------------------------------------------------------------------
# Priority order:
#   1. Hostname pattern matching — transparent, auditable, deterministic.
#      Operators define regex rules in RoleConfig.role_patterns.
#   2. Asset inventory / metadata column — join on a role field if available.
#   3. Single 'default' role — when no patterns match and no metadata exists.
#
# KMeans clustering was removed because:
#   - It requires choosing k with no principled basis
#   - It assumes spherical clusters in rate-feature space
#   - It forces every host into a cluster (no outlier concept)
#   - It produces opaque role_0/role_1 labels with no semantic meaning
#   - The hostnames already encode role information in most environments
# ---------------------------------------------------------------------------

# Default hostname patterns covering common naming conventions.
# Each tuple is (regex_pattern, role_name).  First match wins.
# Patterns are matched against the full hostname (case-insensitive).
# Operators should override these via cfg.role.role_patterns for their
# environment.
DEFAULT_ROLE_PATTERNS: List[Tuple[str, str]] = [
    # Domain controllers
    (r"[-.]dc[-.]|[-.]dc$|^dc[-.]", "dc"),
    # DNS servers
    (r"[-.]dns[-.]|[-.]dns$|^dns[-.]", "dns"),
    # Mail / Exchange servers
    (r"[-.]mail[-.]|[-.]mail$|[-.]smtp[-.]|[-.]smtp$|[-.]exchange[-.]", "mail"),
    # SQL / database servers
    (r"[-.]sql[-.]|[-.]sql$|[-.]db[-.]|[-.]db$", "sql"),
    # SharePoint servers
    (r"[-.]shrpt[-.]|[-.]shrpt$|[-.]sharepoint[-.]", "sharepoint"),
    # Web servers
    (r"[-.]web[-.]|[-.]web$|[-.]iis[-.]", "web"),
    # Windows Event Collector
    (r"[-.]wec[-.]|[-.]wec$", "wec"),
    # Security tools / SIEM
    (r"[-.]sec[-.]|[-.]siem[-.]|[-.]sec-win[-.]|[-.]sec-win$", "security"),
    # Proxy servers
    (r"[-.]proxy[-.]|[-.]proxy$", "proxy"),
    # VPN endpoints
    (r"[-.]vpn[-.]|[-.]vpn$", "vpn"),
    # Admin / command workstations
    (r"[-.]admin[-.]|[-.]admin$|[-.]cmd[-.]|[-.]cmd$", "admin"),
    # ICS/SCADA layers (Purdue model)
    (r"^l[0-5][-.]", "ics"),
]


def infer_roles(
    role_features: pd.DataFrame,
    cfg: StrataConfig,
    role_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Assign role_id per host using hostname pattern matching.

    Priority order:
      1. If role_column is specified and exists in role_features, use it directly
         (asset inventory / metadata join).
      2. Match hostnames against regex patterns from cfg.role.role_patterns
         (or DEFAULT_ROLE_PATTERNS if not configured).
      3. Hosts that match no pattern get role_id = 'workstation' (the most
         common default in enterprise environments).

    Returns DataFrame with columns: host, role_id

    The role names are semantically meaningful ('dc', 'mail', 'workstation')
    rather than opaque cluster labels ('role_0', 'role_1'), which makes the
    triage output interpretable and the pair correlation role-conditioning
    (via static discount tables) work correctly.
    """
    import re

    hosts = role_features[["host"]].copy()

    # Priority 1: metadata column
    if role_column and role_column in role_features.columns:
        hosts["role_id"] = role_features[role_column].fillna("workstation")
        logger.info("infer_roles(): assigned from metadata column '%s'", role_column)
        return hosts

    # Priority 2: hostname pattern matching
    patterns = getattr(cfg.role, "role_patterns", None) or DEFAULT_ROLE_PATTERNS
    compiled = [(re.compile(pat, re.IGNORECASE), role) for pat, role in patterns]

    def _match_role(hostname: str) -> str:
        h = str(hostname).lower()
        for regex, role in compiled:
            if regex.search(h):
                return role
        return "workstation"

    hosts["role_id"] = hosts["host"].apply(_match_role)

    # Log role distribution
    dist = hosts["role_id"].value_counts()
    logger.info("infer_roles(): %d roles from hostname patterns: %s",
                 len(dist), dist.to_dict())

    return hosts


# ---------------------------------------------------------------------------
# Semantically meaningful event pair definitions
# ---------------------------------------------------------------------------
# Each tuple (src, dst) represents a known adversarial transition pattern.
# These are grouped by MITRE ATT&CK tactic for readability and auditability.
# Weights reflect how unambiguous the pair is as an attack indicator:
#   1.00 = near-certain malicious (e.g. CreateRemoteThread -> LSASS access)
#   0.75 = strong indicator, possible benign explanation
#   0.50 = meaningful but common enough to require corroboration
#
# The pair list is consulted during scoring; only hits against known pairs
# contribute to S_ctx. Generic co-occurrence of high-severity events is NOT
# scored — that prevents noisy environments from inflating context scores.
# ---------------------------------------------------------------------------

DEFAULT_INTERESTING_PAIRS: List[Tuple[int, int]] = [
    # ---- Credential Access / LSASS / Sysmon chains ----
    (10, 1),   # ProcessAccess (LSASS) -> Process Create
    (1, 10),   # Process Create -> ProcessAccess (LSASS)
    (10, 3),   # LSASS access -> Network Connection (exfil/C2)
    (11, 10),  # File dropped -> LSASS access (tool written then executed)
    (7, 10),   # DLL Load -> LSASS access (reflective injection)
    (8, 10),   # CreateRemoteThread -> LSASS access (Mimikatz chain)

    # ---- Kerberos / Auth behavior ----
    (4768, 4769),  # TGT request -> Service Ticket (Kerberoasting)
    (4624, 4648),  # Successful logon -> explicit credential logon
    (4672, 4688),  # Special privileges assigned -> process creation
    (4769, 4688),  # Service ticket request -> process creation
    (4624, 10),    # Successful logon -> LSASS access

    # ---- Lateral Movement ----
    (4624, 7045),  # Logon -> service installed (remote service creation)
    (4688, 3),     # Process creation -> network connection
    (4648, 3),     # Explicit credential logon -> network connection
    (1, 3),        # Process Create -> Network Connection
    (4104, 3),     # PowerShell script block -> network

    # ---- Execution chains ----
    (1, 11),   # Process Create -> File Create (payload write)
    (11, 1),   # File dropped -> Process Create (payload execute)
    (22, 1),   # DNS Query -> Process Create (download-and-execute)
    (1, 6),    # Process Create -> Driver Loaded
    (1, 7),    # Process Create -> DLL Load
    (1, 8),    # Process Create -> CreateRemoteThread (injection)
    (3, 1),    # Network Connection -> Process Create
    (4104, 1), # PowerShell script block -> process

    # ---- Persistence ----
    (12, 1),   # Registry object -> Process Create
    (13, 1),   # Registry value set -> Process Create
    (7045, 1), # Service installed -> process
    (11, 12),  # File dropped -> Registry object (install artifact)
    (11, 13),  # File dropped -> Registry value set
    (1, 7045), # Process -> service install

    # ---- C2 / Beaconing ----
    (1, 22),   # Process -> DNS (unusual process resolving external name)
    (22, 3),   # DNS -> Network Connection (resolve then connect)
    (8, 3),    # Injection -> Network Connection (injected process calling out)
    (4104, 3), # PowerShell script block -> network

    # ---- Defense Evasion / Privilege Escalation ----
    (1, 4624),    # Process -> Logon event (token manipulation)
    (4688, 4672), # Process creation -> Special privileges assigned
    (7045, 10),   # Service installed -> LSASS access

    # ---- Reconnaissance ----
    (1, 4798),    # Process -> Local group enumeration
    (11, 4798),   # File dropped -> Local group enumeration
]

# Per-pair weights: maps (src, dst) -> float in [0,1]
# Pairs not in this dict get the default weight (0.50)
PAIR_WEIGHTS: Dict[Tuple[int, int], float] = {
    (8, 10):   1.00,  # CreateRemoteThread -> LSASS: Mimikatz, near certain
    (11, 10):  0.95,  # File dropped -> LSASS: credential dumper written to disk
    (7, 10):   0.90,  # DLL load -> LSASS: reflective injection then dump
    (10, 3):   0.90,  # LSASS access -> network: credential exfil
    (4768, 4769): 0.85,  # Kerberoasting chain
    (7045, 10):   0.85,  # Service install -> LSASS
    (4624, 7045): 0.80,  # Remote logon -> service install (lateral movement)
    (4624, 10):   0.80,  # Logon -> LSASS
    (4688, 4672): 0.75,  # Process -> special privileges (privesc)
    (4104, 3):    0.75,  # PS script block -> network (staged download)
    (22, 1):      0.70,  # DNS -> process (download-and-execute)
    (1, 8):       0.70,  # Process -> CreateRemoteThread
    (8, 3):       0.70,  # Injection -> network
}
_DEFAULT_PAIR_WEIGHT = 0.50

# Build a fast lookup set for O(1) pair membership testing
_PAIR_SET: set = set(DEFAULT_INTERESTING_PAIRS)


# ---------------------------------------------------------------------------
# Role-conditioned pair weight discounting
# ---------------------------------------------------------------------------
# Pairs that are EXPECTED for a given role should contribute less signal.
# A 4768->4769 (Kerberoasting chain) on a domain controller is normal
# Kerberos operation; the same pair on a workstation is genuinely suspicious.
#
# The discount factor (0.0-1.0) multiplies the base pair weight:
#   effective_weight = base_weight × (1.0 - discount)
#
# A discount of 0.95 means the pair contributes only 5% of its normal
# weight on that role.  A discount of 0.0 (or absence from the map)
# means no discounting — full weight applies.
#
# Role names are matched by substring so that "dc" matches "dc",
# "mail" matches "mail", etc.  Since role assignment is now hostname-
# pattern-based, role names are semantically meaningful (e.g. "dc",
# "workstation") and the static discount tables apply directly.
# The pipeline also learns expected pairs from baseline data at fit
# time via compute_role_pair_baselines().
# ---------------------------------------------------------------------------

# Static discounts for roles identifiable by name (asset-inventory or
# heuristic role assignment).  Used as a fallback when no baseline-learned
# discounts are available.
_ROLE_EXPECTED_PAIRS_STATIC: Dict[str, Dict[Tuple[int, int], float]] = {
    # Domain controllers: Kerberos traffic is routine
    "dc": {
        (4768, 4769): 0.95,   # TGT -> Service Ticket: normal DC operation
        (4769, 4688): 0.90,   # Service Ticket -> Process: service auth
        (4624, 4648): 0.85,   # Logon -> Explicit Credential: delegation
        (4672, 4688): 0.85,   # Special Privileges -> Process: admin tasks
        (4624, 10):   0.80,   # Logon -> LSASS: auth subsystem access
        (4688, 4672): 0.80,   # Process -> Special Privileges: service accounts
    },
    # Mail/Exchange servers
    "mail": {
        (1, 3):       0.80,   # Process -> Network: mail delivery
        (3, 1):       0.80,   # Network -> Process: inbound mail
        (22, 3):      0.70,   # DNS -> Network: MX lookups
        (1, 22):      0.70,   # Process -> DNS: recipient domain resolution
        (4624, 4648): 0.70,   # Logon -> Explicit Credential: service accounts
    },
    # SQL / database servers
    "sql": {
        (4624, 4648): 0.70,   # Logon -> Explicit Credential: app connections
        (4688, 4672): 0.70,   # Process -> Special Privileges: DB engine
        (1, 3):       0.60,   # Process -> Network: client connections
    },
    # SharePoint / web servers
    "shrpt": {
        (1, 3):       0.70,   # Process -> Network: HTTP responses
        (3, 1):       0.70,   # Network -> Process: HTTP requests
        (7, 10):      0.50,   # DLL Load -> LSASS: IIS auth module
    },
    # Windows Event Collector
    "wec": {
        (1, 3):       0.80,   # Process -> Network: event forwarding
        (3, 1):       0.80,   # Network -> Process: event collection
        (4624, 4648): 0.70,   # Logon -> Explicit Credential: collection service
    },
}


def _get_role_discount(role_id: str, pair: Tuple[int, int]) -> float:
    """
    Look up the discount factor for a pair on a given role.

    Checks static role name patterns first (substring match),
    then falls back to learned baselines if provided.
    Returns 0.0 (no discount) if the pair is not expected for this role.
    """
    role_lower = str(role_id).lower()
    for role_key, pair_discounts in _ROLE_EXPECTED_PAIRS_STATIC.items():
        if role_key in role_lower:
            discount = pair_discounts.get(pair, 0.0)
            if discount > 0.0:
                return discount
    return 0.0


def compute_role_pair_baselines(
    events: pd.DataFrame,
    cfg: StrataConfig,
    min_hosts_for_baseline: int = 2,
) -> Dict[str, Dict[Tuple[int, int], float]]:
    """
    Learn which pairs are routine for each role from baseline data.

    For each role, compute the fraction of hosts that exhibit each pair.
    Pairs seen on >50% of hosts in a role are considered expected for
    that role and receive a discount proportional to their prevalence.

    Returns {role_id -> {(src, dst) -> discount_factor}}.

    Call this during fit() and persist the result in FittedArtifacts.
    """
    if "role_id" not in events.columns:
        return {}

    baselines: Dict[str, Dict[Tuple[int, int], float]] = {}
    window = cfg.scoring.window_seconds

    for role_id, role_group in events.groupby("role_id"):
        hosts_in_role = role_group["host"].unique()
        if len(hosts_in_role) < min_hosts_for_baseline:
            continue

        # Track which pairs each host exhibits
        host_pairs: Dict[str, set] = {}
        for host, host_df in role_group.groupby("host"):
            hdf = host_df.sort_values("ts").dropna(subset=["ts", "event_id"])
            if hdf.empty:
                continue
            times = (hdf["ts"].astype("int64") // 10**9).to_numpy()
            evs = hdf["event_id"].fillna(0).astype(int).to_numpy()
            window_ends = np.searchsorted(times, times + window, side="right")
            seen: set = set()
            n = len(hdf)
            for i in range(n):
                ev_i = int(evs[i])
                for j in range(i + 1, min(window_ends[i], i + 50)):  # cap inner loop
                    key = (ev_i, int(evs[j]))
                    if key in _PAIR_SET:
                        seen.add(key)
            host_pairs[str(host)] = seen

        if not host_pairs:
            continue

        # Compute prevalence: fraction of hosts in this role that show each pair
        n_hosts = len(host_pairs)
        pair_counts: Dict[Tuple[int, int], int] = {}
        for seen in host_pairs.values():
            for pair in seen:
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # Pairs seen on >50% of hosts get a discount
        role_discounts: Dict[Tuple[int, int], float] = {}
        for pair, count in pair_counts.items():
            prevalence = count / n_hosts
            if prevalence > 0.5:
                # Scale discount with prevalence: 50% → 0.0, 100% → 0.90
                role_discounts[pair] = min(0.90, (prevalence - 0.5) * 1.8)

        if role_discounts:
            baselines[str(role_id)] = role_discounts

    return baselines


# ---------------------------------------------------------------------------
# Semantic pair correlation (replaces generic co-occurrence)
# ---------------------------------------------------------------------------

def correlate_critical_events_single_host(
    host_df: pd.DataFrame,
    *,
    cfg: StrataConfig,
    interesting_pairs: Optional[List[Tuple[int, int]]] = None,
    role_id: Optional[str] = None,
    role_pair_baselines: Optional[Dict[str, Dict[Tuple[int, int], float]]] = None,
) -> pd.DataFrame:
    """
    Detect occurrences of semantically meaningful event pairs within
    window_seconds on a single host.

    Role-conditioned weighting: if role_id is provided, pairs that are
    expected for that role (learned from baseline data or static config)
    have their weights discounted.  A 4768→4769 pair on a DC contributes
    only ~5% of its normal weight; the same pair on a workstation gets
    the full 0.85 weight.

    Returns DataFrame with columns:
        src_event, dst_event, pair, count, weight, role_discount,
        weighted_score, tactic
    """
    df = host_df.sort_values("ts").copy()
    df = df.dropna(subset=["ts", "event_id"])
    if df.empty:
        return pd.DataFrame(columns=[
            "src_event", "dst_event", "pair",
            "count", "weight", "role_discount", "weighted_score", "tactic",
        ])

    pair_set = set(interesting_pairs) if interesting_pairs else _PAIR_SET
    times = (df["ts"].astype("int64") // 10**9).to_numpy()
    evs   = df["event_id"].fillna(0).astype(int).to_numpy()
    window = cfg.scoring.window_seconds
    n = len(df)

    counts: Dict[Tuple[int, int], int] = {}

    # Use searchsorted to find the right boundary of each window in O(log n)
    window_ends = np.searchsorted(times, times + window, side="right")

    for i in range(n):
        ev_i = int(evs[i])
        for j in range(i + 1, window_ends[i]):
            key = (ev_i, int(evs[j]))
            if key in pair_set:
                counts[key] = counts.get(key, 0) + 1

    if not counts:
        return pd.DataFrame(columns=[
            "src_event", "dst_event", "pair",
            "count", "weight", "role_discount", "weighted_score", "tactic",
        ])

    # Look up learned discounts for this role, fall back to static
    learned_discounts = {}
    if role_pair_baselines and role_id and str(role_id) in role_pair_baselines:
        learned_discounts = role_pair_baselines[str(role_id)]

    rows = []
    for (a, b), c in counts.items():
        base_w = PAIR_WEIGHTS.get((a, b), _DEFAULT_PAIR_WEIGHT)

        # Apply role-conditioned discount
        # Check learned baselines first, then static
        discount = learned_discounts.get((a, b), 0.0)
        if discount == 0.0 and role_id:
            discount = _get_role_discount(role_id, (a, b))

        effective_w = base_w * (1.0 - discount)

        rows.append({
            "src_event":      a,
            "dst_event":      b,
            "pair":           f"{a}->{b}",
            "count":          c,
            "weight":         round(effective_w, 4),
            "role_discount":  round(discount, 4),
            "weighted_score": c * effective_w,
            "tactic":         _pair_tactic(a, b),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("weighted_score", ascending=False)
        .reset_index(drop=True)
    )


def _pair_tactic(src: int, dst: int) -> str:
    """Return a MITRE tactic label for a known pair, for explainability output."""
    cred_access = {(10,1),(1,10),(10,3),(11,10),(7,10),(8,10),(4768,4769),(4624,10)}
    lateral     = {(4624,7045),(4688,3),(4648,3),(4769,4688)}
    execution   = {(1,11),(11,1),(22,1),(1,6),(1,7),(1,8),(3,1),(4104,1)}
    persistence = {(12,1),(13,1),(7045,1),(11,12),(11,13),(1,7045)}
    c2          = {(1,22),(22,3),(8,3),(4104,3)}
    evasion     = {(1,4624),(4688,4672),(7045,10)}
    recon       = {(1,4798),(11,4798)}

    pair = (src, dst)
    if pair in cred_access: return "credential_access"
    if pair in lateral:     return "lateral_movement"
    if pair in execution:   return "execution"
    if pair in persistence: return "persistence"
    if pair in c2:          return "c2"
    if pair in evasion:     return "defense_evasion"
    if pair in recon:       return "reconnaissance"
    return "unknown"


def correlate_critical_events_by_host(
    df: pd.DataFrame,
    cfg: StrataConfig,
    interesting_pairs: Optional[List[Tuple[int, int]]] = None,
    role_pair_baselines: Optional[Dict[str, Dict[Tuple[int, int], float]]] = None,
) -> pd.DataFrame:
    """
    Run semantic pair correlation across all hosts.

    If role_id is present on the events DataFrame and role_pair_baselines
    is provided, applies role-conditioned weight discounting so that
    pairs expected for a host's role contribute less signal.
    """
    has_role = "role_id" in df.columns
    all_rows = []
    for host, g in df.groupby("host", dropna=False):
        role_id = None
        if has_role:
            roles = g["role_id"].dropna().unique()
            role_id = str(roles[0]) if len(roles) > 0 else None

        pairs = correlate_critical_events_single_host(
            g, cfg=cfg,
            interesting_pairs=interesting_pairs,
            role_id=role_id,
            role_pair_baselines=role_pair_baselines,
        )
        if not pairs.empty:
            pairs.insert(0, "host", str(host))
            all_rows.append(pairs)

    if not all_rows:
        return pd.DataFrame(columns=[
            "host", "src_event", "dst_event", "pair",
            "count", "weight", "role_discount", "weighted_score", "tactic",
        ])
    return pd.concat(all_rows, ignore_index=True)


def compute_pair_stats(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-host pair statistics for the context channel.

    Key output columns:
      n_pairs          — number of distinct known pairs observed
      weighted_score_sum — sum of count × pair_weight (primary signal)
      max_pair_weight  — weight of the highest-confidence pair seen
      n_tactics        — number of distinct MITRE tactics represented
      top_tactic       — most frequently represented tactic (for triage output)
    """
    if pairs_df.empty:
        return pd.DataFrame(columns=[
            "host", "n_pairs", "weighted_score_sum",
            "max_pair_weight", "n_tactics", "top_tactic",
        ])

    stats = pairs_df.groupby("host").agg(
        n_pairs=("pair", "nunique"),
        weighted_score_sum=("weighted_score", "sum"),
        max_pair_weight=("weight", "max"),
        n_tactics=("tactic", "nunique"),
    ).reset_index()

    # Top tactic: which MITRE tactic had the highest total weighted score
    top_tactic = (
        pairs_df.groupby(["host", "tactic"])["weighted_score"]
        .sum()
        .reset_index()
        .sort_values("weighted_score", ascending=False)
        .groupby("host")
        .first()["tactic"]
        .reset_index()
        .rename(columns={"tactic": "top_tactic"})
    )
    stats = stats.merge(top_tactic, on="host", how="left")
    stats["top_tactic"] = stats["top_tactic"].fillna("none")
    return stats
