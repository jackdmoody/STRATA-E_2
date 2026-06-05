# STRATA-(E)
### Structural and Temporal Role-Aware Threat Analytics for Endpoint Telemetry

> A hierarchical, statistically calibrated behavioral anomaly detection framework for Windows Sysmon and Windows Event telemetry. Reduces thousands of endpoint events to a ranked, explainable triage list by combining time-aware sequence modeling, Bayesian peer-role baselines, multi-channel anomaly scoring, and a corroboration gate — answering: **which hosts are behaving abnormally, why, and based on what evidence?**

---

## The Problem

Enterprise endpoint anomaly detection faces four recurring challenges. **Temporal abstraction ambiguity**: coarse time bucketing loses kill-chain velocity signatures while fine-grained bucketing creates unmanageable state spaces. **Transition sparsity**: sparse host windows produce unreliable divergence estimates that inflate false positive rates. **Baseline contamination**: in compromised-by-default environments, global baselines absorb attacker behavior as "normal." **Structural vs. volumetric mismatch**: a host running encoded PowerShell at normal volume looks fine to rate-based detectors; a host running `svchost.exe` at high volume looks anomalous to sequence-based detectors.

STRATA-(E) addresses all four by modeling behavior per-host, per-role, with time-aware transitions, Dirichlet-stabilized baselines, and four independent detection channels that must corroborate before surfacing an alert.

---

## Architectural Overview
<img width="2500" height="1408" alt="strata_arch" src="https://github.com/user-attachments/assets/80a9719d-a1b1-45ec-9c18-ba4b1c824926" />


*Time-aware, role-aware, multi-channel Sysmon behavioral analytic architecture with Bayesian peer baselines and calibrated sequence divergence. Dirichlet shrinkage stabilizes transition estimation; matched-bootstrap effect sizes and leave-one-fold-out conformal p-values calibrate JSD-based sequence anomalies. A parallel validation framework supports attack injection, ablation, calibration checks, and stability analysis.*

---

## How It Works

Each stage builds on the previous, and the four scoring channels answer independent questions about host behavior:

```
Raw Sysmon / Windows Event Telemetry (CSV, Parquet, or JSON)
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1 — Ingest & Preprocessing                               │
│  Flexible column detection (Sysmon, Splunk, Elastic, DARPA TC)  │
│  Canonical schema: ts, host, event_id, image, cmdline, user     │
│  Multi-resolution tokenization (coarse / medium / fine)         │
│  Sessionization with adaptive τ_gap per role                    │
│  Inter-event Δt discretization into time buckets                │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2 — Feature Construction                                  │
│  Time-aware transitions: P(z', β | z)                           │
│  Per-host rate features (proc/script/office/lolbin rates)       │
│  Critical event pair correlation (MITRE ATT&CK-aligned)         │
│  Context flags: encoded cmds, download cradles, LOLBin usage    │
│  TF-IDF command-line novelty scoring                             │
│  Event-level MITRE ATT&CK technique mapping                     │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3 — Bayesian Peer-Role Baselines                          │
│  Host role assignment via hostname pattern matching (11 roles)  │
│  Hierarchical Dirichlet model: θ_r ~ Dir(α₀), θ_h|θ_r ~ Dir(κθ_r) │
│  Shrinkage toward role baseline stabilizes sparse windows       │
│  Role-conditioned pair weight discounting                        │
│  Smoothed peer role baselines: P̂_r(z', β | z)                  │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼  4 independent channels — each answers a different question
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4 — Calibrated Multi-Channel Scoring                      │
│                                                                  │
│  Sequence channel (structural anomalies):                        │
│    S_seq = JS(P̂_h ‖ P̂_r(h)), Dirichlet-smoothed                │
│    Matched bootstrap + LOFO conformal calibration →              │
│      empirical p-value, percentile vs null, ×-null effect size   │
│                                                                  │
│  Frequency channel (volumetric anomalies):                       │
│    S_freq = IsolationForest(rate features)                       │
│    SHAP feature attributions for explainability                   │
│                                                                  │
│  Context channel (fine-grained signals):                         │
│    S_ctx = peer-relative (role median/MAD) rates: content flags, │
│      pair correlation, TF-IDF command novelty, severity prior    │
│                                                                  │
│  Drift channel (behavioral change over time):                    │
│    S_drift = JS(P̂_h^cur ‖ P̂_h^hist)                             │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼  Question: Is the anomaly corroborated across channels?
┌─────────────────────────────────────────────────────────────────┐
│  Stage 5 — Evidence Fusion & Gating                              │
│  Borda rank aggregation across channels                          │
│  Corroboration gate: requires ≥ 2 channels above threshold      │
│  Extreme-channel bypass for single-channel extreme scores        │
│  Result: 79% triage queue reduction on real-world data           │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
  Ranked Host Triage — explainable per-host scores with
  top anomalous transitions, channel breakdowns, calibrated
  percentiles, MITRE ATT&CK technique annotations, SHAP
  feature importance, and suggested next steps
```

### Why this architecture

| Component | Technique | What it answers | Why the alternatives fail |
|---|---|---|---|
| Hostname-pattern roles | Regex-based role inference | What type of host is this? | KMeans clustering requires choosing k, assumes spherical clusters, and produces opaque labels. Hostname patterns yield semantic roles (dc, mail, workstation) directly. |
| Dirichlet shrinkage | Hierarchical Bayesian prior | How do you score a host with 25 events? | MLE on sparse windows produces unstable divergence scores. Shrinkage toward the role baseline stabilizes estimates while preserving genuine deviations. |
| Matched bootstrap + conformal calibration | Per-host multinomial null (host's own per-state counts, same shrinkage estimator, same severity weights) + leave-one-fold-out conformal vs. the host's own benign-window fold scores | Is this divergence significant — vs. sampling noise, and vs. this host's own benign history? | Raw JSD depends on window size, and benign hosts are not iid draws from the pooled role baseline (stable idiosyncrasy). A global parametric null produced 800σ+ z-scores on benign hosts; the matched null fixes the sampling mismatch and the conformal layer gives distribution-free, super-uniform p-values. |
| Peer-role baselines | Hostname-pattern grouping | What's "normal" for this type of host? | Global baselines absorb server behavior into the workstation norm (and vice versa). Role conditioning makes "abnormal" mean "abnormal for your peer group." |
| Role-conditioned pair discounting | Learned + static discounts | Is this event pair suspicious for this role? | A process→network burst (EID 1→3) on a mail server is routine delivery; the same pair on an engineering workstation is notable. Discounts are event-pair-level, learned from each role's baseline plus a static table (e.g., DC logon→LSASS access discounted 0.80). Binary-level role sensitivity (e.g., certutil frequency) is handled separately by the context channel's peer-relative rate scoring, not by pair discounts. |
| Corroboration gate | Multi-channel consensus | Is this really suspicious or just one noisy signal? | Single-channel detectors produce lists dominated by false positives. Requiring independent agreement across structural, volumetric, and contextual channels filters coincidental anomalies. |
| TF-IDF command novelty | Vectorized cosine similarity | Are this host's commands unusual? | Flag-only context scoring misses novel commands that don't match predefined patterns. TF-IDF captures arbitrary command-line anomalies relative to the fleet baseline. |

---

## Real-World Validation

STRATA-(E) has been validated on two independent networks from a military cyber exercise (needs rerun to validate with updated code, this is from APR run): 

| Metric | 152 Network (IT) | 201 Network (ICS) |
|---|---|---|
| Events | 14,874,505 | 30,481,158 |
| Hosts | 57 | 28 |
| Time span | 14 days | 14 days |
| Roles inferred | 11 | 5 |
| Gate pass (triage leads) | 12 (79% filtered) | 7 (75% filtered) |
| Runtime (fast mode) | 19 min | ~10 min |
| Top finding | Selenium automation platform (567 unique processes) | GHOSTS NPC simulation + ICS HMI application |

### Key findings

- **H1 (Dirichlet shrinkage)**: 94.1% JSD variance reduction vs. MLE on real data
- **H2 (Role baselining)**: Sequence channel completely dead under global baseline (S_seq = 0 for all hosts); active under role baseline with clear differentiation. Gate throughput: 9 → 12 hosts
- **H4 (Channel independence)**: 30-70% overlap between fused and single-channel rankings, confirming channels surface different hosts
- **H5 (Corroboration gate)**: 79% host filtering; infrastructure noise replaced by corroborated workstation anomalies
- **Cross-domain generalization**: Consistent 75-79% gate filtering across IT and ICS/OT networks without retraining or reconfiguration
- **SHAP analysis**: script_rate, encoded command rate, and LOLBin rate among top frequency channel drivers — confirming security-relevant behavioral dimensions
- **Reproducibility**: 3 replicate runs produce identical output hashes

---

## Installation

```bash
git clone https://github.com/jackdmoody/STRATA-E.git
cd STRATA-E
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install in editable mode
pip install -e ".[dev,shap]"   # includes pytest + shap (the CLI's SHAP step needs it)

# Optional: SHAP feature importance analysis
pip install shap
```

### Dependencies

Python ≥ 3.10. Core: `pandas`, `numpy`, `scikit-learn`, `scipy`, `networkx`, `matplotlib`, `plotly`, `ipywidgets`, `pyarrow`. Optional: `shap` (for SHAP feature importance analysis).

---

## Usage

### CLI

```bash
# Run on Sysmon data with HTML report
strata --input data/sysmon_export.parquet --output results --report --browser

# Fast mode — 3-10× speedup for iterative development
strata --input data/sysmon_export.parquet --output results --fast --report --browser

# No baseline/scoring split — use fit_score on full dataset
# (required when data spans < 7 days)
strata --input data/events.parquet --output results --fast --no-split --report --browser

# Run with extended evaluation metrics
strata --input data/events.parquet --output results --fast --no-split --report --eval

# Run with full evaluation (adds ablation, sensitivity, scalability — ~25 min extra)
strata --input data/events.parquet --output results --fast --no-split --report --eval-full

# Custom baseline/scoring split
strata --input data/events.parquet --baseline-days 10 --score-days 4 --report --browser

# Run a specific ablation condition
strata --input data/events.parquet --ablation sequence_only

# Suppress matplotlib visualizations (headless)
strata --input data/events.parquet --report --no-plots
```

### Run from Python (recommended for short datasets)

```python
import pandas as pd
from sysmon_pipeline import StrataPipeline, StrataConfig
from sysmon_pipeline.report import ReportContext

df = pd.read_parquet("data/events.parquet")
# Clean placeholder values
for col in ['Image','ParentImage','CommandLine','ParentCommandLine','IntegrityLevel']:
    if col in df.columns:
        df[col] = df[col].replace({'UNKNOWN': pd.NA, '-': pd.NA})

cfg = StrataConfig.fast()
with ReportContext(output_dir="results/report", open_browser=True) as report:
    pipe = StrataPipeline(cfg)
    art = pipe.fit_score(df)
    report.finalise(art)
```

### Run extended evaluation

```bash
# Quick eval: H2-H5 plots, SHAP, workload reduction, explainability, reproducibility
python eval_extended.py --input data/events.parquet --output results --light-plots \
    --skip-ablation --skip-sensitivity --skip-scalability

# Full eval: adds ablation (7 conditions), sensitivity sweeps, scalability benchmarks
python eval_extended.py --input data/events.parquet --output results --light-plots
```

### Separate baseline and scoring windows

```python
from sysmon_pipeline import StrataPipeline, StrataConfig
from sysmon_pipeline.loaders import load_sysmon_csv, split_time_windows

cfg = StrataConfig()
df  = load_sysmon_csv("data/sysmon_30days.csv")

baseline_df, scoring_df = split_time_windows(df, baseline_days=10, score_days=4)

pipe   = StrataPipeline(cfg)
fitted = pipe.fit(baseline_df)
art    = pipe.score(scoring_df, fitted, prior_window_df=baseline_df)
```

### Fast mode from Python

```python
from sysmon_pipeline import StrataPipeline, StrataConfig

cfg = StrataConfig.fast()  # bootstrap=200, IF trees=50, TF-IDF features=300
```

### Debug pipeline (stage-by-stage validation)

The `StrataDebugPipeline` wraps the production pipeline and exposes each stage as a standalone method for notebook-based debugging. The production `StrataPipeline` is not modified.

```python
from sysmon_pipeline.debug import StrataDebugPipeline
from sysmon_pipeline import StrataConfig

dbg = StrataDebugPipeline(StrataConfig.fast())

# Step 1: Validate data compatibility (~seconds)
report = dbg.preprocess_check(raw_df)

# Step 2: Run stages one at a time, inspect each
events = dbg.preprocess(raw_df)
trans, rates = dbg.build_features(events)
fitted = dbg.fit_baselines(events, trans, rates)

# Step 3: Score individual channels
seq  = dbg.score_sequence(trans, fitted)
freq = dbg.score_frequency(rates, fitted)
ctx  = dbg.score_context(events, fitted)

# Step 4: Fuse and triage
triage = dbg.fuse_and_triage(seq, freq, ctx)
```

---

## Data Ingestion

STRATA-(E) accepts data from four sources:

**Generic Sysmon CSV or Parquet** — Any CSV or Parquet export from Sysmon, Splunk, Elastic, or a SIEM. Column names are auto-detected via ordered candidate lists. Format is auto-detected from the file extension (`.csv`, `.csv.gz`, `.parquet`, `.pq`).

| Field | Candidates (first match wins) |
|---|---|
| Timestamp | `_timestamp`, `UtcTime`, `ts`, `timestamp`, `TimeCreated` |
| Host | `host.fqdn`, `Computer`, `Host`, `Hostname`, `host` |
| Event ID | `winlog.event_id`, `EventID`, `EventId`, `event_id` |
| Image | `Image`, `ProcessImage`, `process_image`, `ProcessName` |
| Parent image | `ParentImage`, `ParentProcessName`, `parent_image` |
| Command line | `CommandLine`, `CmdLine`, `cmdline`, `command_line` |
| User | `User`, `UserName`, `SubjectUserName`, `user` |
| Integrity level | `IntegrityLevel`, `integrity_level` |
| Signed | `Signed`, `signed` — parsed string-safe (`"true"`/`"false"`); absent column = unknown, not unsigned |
| Target image (EID 10) | `TargetImage`, `target_image`, `winlog.event_data.TargetImage` — gates LSASS/T1003 severity 1.0 |
| Target object (EID 12/13) | `TargetObject`, `target_object`, `winlog.event_data.TargetObject` — gates Run-key/T1547.001 |
| Ticket encryption (4769) | `TicketEncryptionType`, `ticket_encryption_type` — gates Kerberoasting/T1558.003 |
| Logon type (4624) | `LogonType`, `logon_type` — severity boost for types 9/10 |

Only timestamp, host, and event ID are required. All other columns degrade gracefully if absent.

**DARPA Transparent Computing** — JSON lines from the CADETS, THEIA, FIVEDIRECTIONS, and TRACE datasets. Linux syscall event types are mapped to approximate Sysmon event IDs via `_DARPA_EVENT_MAP`.

**Synthetic data** — Built-in generator for testing and ablation without real data. Use `--synthetic` flag.

---

## HTML Report

The `--report` flag generates an interactive, self-contained HTML dashboard with:

- Summary metric cards: events processed, hosts scored, corroborated hosts, critical findings
- Pipeline flow visualization showing stage-by-stage counts
- **Interactive host investigation dashboard** — click any host to drill down
- Per-host **channel score cards** with calibration detail (conformal p-value, percentile vs. sampling null, ×-null effect size; z-score retained in CSV as an analyst column)
- **Event timeline strip** — color-coded by process category (SCRIPT, LOLBIN, OFFICE, PROC), height by severity level. Hover for event detail including MITRE T-code
- **Top transition bars** — the actual event-to-event sequences behind the sequence channel score
- **Channel radar chart** (Chart.js) — selected host vs. fleet median across all 4 channels
- **MITRE ATT&CK technique summary** — per-host technique frequency with clickable links to attack.mitre.org
- **Pair correlation details** — event pairs, top tactics, weighted scores
- CSV download buttons for all artifact tables
- Diagnostic plot gallery with click-to-zoom

---

## Extended Evaluation Framework

The `eval_extended.py` script produces publication-ready evaluation metrics:

| Section | Output | Description |
|---|---|---|
| Channel contribution | `channel_contribution.csv`, `.png` | Borda rank decomposition per host |
| H2 plot | `h2_seq_comparison.png` | Global vs. role baseline sequence scores |
| H4 plot | `h4_channel_independence.png` | Top-K overlap curves across channels |
| H5 plot | `h5_gate_overlap.png` | Corroboration gate effect on triage ranking |
| Deployment projections | `deployment_projections.csv`, `.png` | Fleet-scale alert volume projections |
| Workload reduction | `workload_reduction.csv`, `.txt`, `.png` | Analyst hours saved at different fleet sizes |
| Per-host explainability | `host_explainability.csv`, `.txt` | Why each gated host was flagged |
| SHAP importance | `shap_importance.csv`, `.png`, `shap_beeswarm.png` | Feature importance for frequency channel |
| Reproducibility | `reproducibility_check.csv` | Determinism verification across replicates |
| Ablation comparison | `ablation_comparison.csv`, `.png` | 7 conditions compared (with `--eval-full`) |
| Stage timing comparison | `stage_timing_comparison.csv`, `.png` | Per-stage timing across ablation conditions |
| Sensitivity sweeps | `sensitivity_sweeps.csv`, `.png` | Gate throughput across parameter ranges |
| Scalability timing | `scalability_timing.csv`, `.png` | Throughput curves at different data sizes |

---

## Scoring Channels

STRATA-(E) scores each host independently across four channels. The corroboration gate requires signal from ≥ 2 channels before surfacing a host as a triage lead.

| Channel | Technique | What it detects | Key output |
|---|---|---|---|
| **Sequence** | Jensen-Shannon divergence from Dirichlet-shrunk peer-role baseline, bootstrap-calibrated | Structural anomalies — novel process chains, unusual transition patterns relative to peer role | `S_seq`, conformal p-value, percentile vs null, ×-null effect size, z (CSV-only), rare transition hits |
| **Frequency** | Isolation Forest on per-host rate features (script_rate, lolbin_rate, proc_rate_total, unique_parents, etc.) | Volumetric anomalies — unusual event volume, rate spikes, abnormal process mix | `S_freq`, SHAP feature attributions |
| **Context** | TF-IDF command-line novelty + role-conditioned event pair correlation | Fine-grained behavioral indicators — novel commands relative to fleet, suspicious parent-child chains with role-aware discounting | `S_ctx`, cmdline novelty score |
| **Drift** | JSD between current and prior-window transition distributions | Behavioral change over time — sustained shifts in host behavior vs. its own history | `S_drift` |

---

## Role Assignment

STRATA-(E) assigns host roles using regex-based hostname pattern matching. This replaces the earlier KMeans clustering approach, which required choosing k, assumed spherical clusters, and produced opaque labels.

| Pattern | Role | Example hostnames |
|---|---|---|
| `dc`, `domain` | dc | asfbn-dc, corp-dc-01 |
| `dns` | dns | asfbn-dns, dns-server |
| `mail`, `exch` | mail | asfbn-mail, exchange-01 |
| `smtp` | mail | asfbn-smtp |
| `sql`, `db` | sql | asfbn-sql, db-server |
| `shrpt`, `sharepoint` | sharepoint | asfbn-shrpt |
| `wec`, `collector` | wec | asfbn-wec |
| `cmd`, `admin`, `jump` | admin | asfbn-cmd-1, jump-box |
| `vpn` | vpn | asfbn-vpn-user |
| `ics`, `scada`, `hmi`, `plc` | ics | l2-eng-win10-4 |
| *(no match)* | workstation | asfbn-s6-1, asfbn-s8-2 |

Roles determine peer groups for sequence channel baselines and pair weight discounting. Hosts with fewer than `min_hosts_for_baseline` (default: 2) peers in their role receive a shrinkage-weighted global baseline.

---



## Ablation Conditions

STRATA-(E) supports structured ablation studies via `AblationConfig` presets:

| Condition | What it disables | Purpose |
|---|---|---|
| `full_pipeline` | Nothing | Full system — all components active |
| `sequence_only` | Context, drift, covariance channels | Isolate structural sequence modeling contribution |
| `no_shrinkage` | Dirichlet shrinkage | Test MLE vs. Bayesian estimation |
| `no_role_baselining` | Role-conditioned baselines | Test role-aware vs. global baseline |
| `no_calibration` | Matched bootstrap + conformal calibration | Test raw vs. calibrated divergence |
| `no_drift` | Drift channel + seq-drift covariance | Test with/without temporal change detection |

---

## Module Reference

| Module | Stage | Description |
|---|---|---|
| `config.py` | — | Typed dataclass configuration. Sub-configs: `IOConfig`, `TimeBucketingConfig`, `BaselineConfig`, `RoleConfig`, `ScoringConfig`, `AblationConfig`. JSON serialization. `StrataConfig.fast()` factory. |
| `schema.py` | 1 | Flexible multi-candidate column detection. Canonical schema normalization. Type coercion. |
| `loaders.py` | 1 | CSV, Parquet, and DARPA TC JSON ingestion. `split_time_windows()`. |
| `mapping.py` | 2 | Multi-resolution token abstraction (coarse/medium/fine). Context flags. Event severity grading. MITRE ATT&CK event-level technique mapping. |
| `sequence.py` | 2 | Sessionization with adaptive τ_gap. Vectorized inter-event Δt bucketing. Transition count extraction. |
| `pairs.py` | 2–3 | Role assignment via hostname patterns. Rate feature computation. Role-conditioned pair weight discounting. Semantic critical event pair correlation. |
| `divergence.py` | 3–4 | Hierarchical Dirichlet peer baselines. JSD scoring. Vectorized bootstrap calibration with `(role, n_bin)` cache. Drift computation. |
| `scoring.py` | 4–5 | Isolation Forest frequency channel. TF-IDF command-line novelty. Context scoring. Borda rank fusion. Corroboration gate. Ranked triage builder. |
| `pipeline.py` | — | `StrataPipeline` orchestrator. `fit()` / `score()` / `fit_score()`. Per-stage timing instrumentation. |
| `cli.py` | — | Full CLI with `--report`, `--eval`, `--eval-full`, `--no-split`, `--fast`, `--browser`, `--ablation`. |
| `report.py` | — | Self-contained HTML report. Interactive host investigation dashboard. Chart.js radar. Event timeline. MITRE technique summary. CSV downloads. |
| `analysis.py` | — | Post-hoc analysis: SHAP feature importance, deployment-prevalence projections, error analysis, latency benchmarking, channel taxonomy. |
| `visuals.py` | — | Static matplotlib and interactive Plotly visualizations. |
| `graph.py` | — | NetworkX graph utilities for transition visualization. |
| `debug.py` | — | Stage-by-stage debug pipeline for notebook-based development. |

---

## Hypotheses Tested

| Hypothesis | Claim | Result (real data) |
|---|---|---|
| **H1** | Dirichlet shrinkage reduces JSD variance vs. MLE | **SUPPORTED**: 94.1% variance reduction |
| **H2** | Peer-role baselines improve detection vs. global baseline | **SUPPORTED**: S_seq = 0 under global, active under role. Gate: 9 → 12 hosts |
| **H3** | Bootstrap-calibrated p-values are uniform under benign data | **NOT SUPPORTED**: p-value pileup artifact from single-window fit/score (requires disjoint split) |
| **H4** | Multi-channel fusion surfaces different hosts than any single channel | **SUPPORTED**: 30-70% overlap confirming channel independence |
| **H5** | Corroboration gating reduces false positives without degrading recall | **SUPPORTED**: 79% host filtering; infrastructure replaced by workstations |

---

## Limitations

- **Batch-oriented** — not streaming-native. Each run processes a fixed time window.
- **Role assignment depends on hostname conventions** — environments with opaque hostnames (e.g., `WIN-A3F2B1`) would need custom patterns or behavioral clustering.
- **Dependent on Sysmon configuration quality** — events not collected by Sysmon cannot be scored.
- **Unsupervised** — no attribution modeling. STRATA identifies anomalous hosts, not specific threat actors.
- **Calibration assumes multinomial null** — the bootstrap null distribution is an approximation.
- **H3 requires disjoint time split** — single-window fit/score produces p-value pileup artifact. A baseline/scoring split (e.g., 10-day baseline, 4-day scoring) is needed for valid calibration.
- **Role assignment limitation** — hosts with the same hostname pattern but different functions (e.g., `asfbn-mail` vs. `asfbn-smtp`) may be grouped into the same peer role despite different behavioral profiles.
- **Memory scaling** — datasets exceeding ~15M events may require per-host subsampling for machines with < 256 GB RAM.

---

## License

MIT
