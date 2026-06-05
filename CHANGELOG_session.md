# STRATA-E Changelog — Detection Quality, Calibration & Channel Independence Release

**Files changed (9):** `config.py`, `schema.py`, `mapping.py`, `sequence.py`,
`divergence.py`, `scoring.py`, `pairs.py`, `pipeline.py`, `report.py`
(all in `sysmon_pipeline/`). 1,020 insertions / 290 deletions vs. prior HEAD.
All changes verified against: repo pytest suite (9/9), 40-case mapping test
suite, end-to-end operational fit/score split on real ASFBN ICS sample
(6.77M events, 28 hosts), 7-condition ablation harness, HTML report generation.

---

## MIGRATION — read before running on full data

1. **Full re-fit required.** Five new LOLBins change the `token_medium`
   vocabulary (`MSBUILD`, `FORFILES`, `ESENTUTL`, `SCRIPTRUNNER`, `MAVINJECT`
   were `PROC:*`, now `LOLBIN:*`). Every persisted `FittedArtifacts` /
   transition baseline is invalid. Run `fit()` fresh; do not score against
   old artifacts.
2. **All channel score distributions shift.** Severity recalibration,
   live S_seq weighting, the S_ctx rewrite, and the Borda default mean new
   outputs are NOT comparable to `entire_data_201/report/` CSVs. Regenerate
   all applied-results numbers; do not diff old vs. new as a correctness check.
3. **Fusion default is now `borda`** (was `weighted_linear`). If any prior
   result was generated under the old default, the paper's method description
   (Eq. 15?) and the implementation must be reconciled — verify which method
   produced reported numbers.
4. **Export fields.** To activate the new conditional gates, the telemetry
   export must include: `Signed`, `TargetImage` (Sysmon 10), `TargetObject`
   (Sysmon 12/13), `TicketEncryptionType` (Security 4769), `LogonType`
   (Security 4624). Absent fields degrade safely to no-ops (verified on the
   ICS sample, which lacks all five).
5. **`severity_label` semantics changed** — it now derives from
   `severity_triage` (flag-boosted), not `severity_score`. Channel inputs
   use `severity_score` only.

---

## mapping.py — severity, MITRE, tokens

### Removed (incorrect mappings)
- 4768 → T1558.003 Kerberoasting (TGT request fires on every domain logon;
  Kerberoasting evidence is 4769 + RC4 ticket encryption)
- 4672 → T1134 Token Manipulation (fires on every admin/SYSTEM logon)
- EID 6 → T1574.002 DLL Side-Loading (category error: side-loading is EID 7)
- Unconditional EID 22 → T1071 (every DNS query tagged C2)
- EVENT_SEVERITY entry 1109 (provider-ambiguous bare event ID)

### Now conditional (was unconditional)
- EID 10 → T1003 only when `TargetImage == lsass.exe`; severity 1.00 only
  then (base 0.55, was flat 1.00)
- EID 12/13 → T1547.001 only on autostart registry paths
  (`CurrentVersion\Run*`, Winlogon Shell/Userinit, IFEO); severity boost 0.85
- 4769 → T1558.003 only with RC4 ticket encryption (0x17/0x18)
- EID 6 → T1014 Rootkit only when driver is KNOWN-unsigned
- EID 7 → T1574.002 only when image is KNOWN-unsigned (low-priority context)
- 4624 severity boost (0.70) for LogonType 9/10

### Added
- Events: 1102 (T1070.001, sev 0.98), 4625, 4698/4702 (T1053.005),
  4720 (T1136.001), 4728/4732/4756 (T1098), 5140/5145, Sysmon 18,
  19/20/21 (T1546.003, 0.90), 23, 25 (T1055.012, 0.95)
- MITRE DB entries: T1105, T1218.004/.007/.008/.009/.013, T1218 (parent),
  T1127.001, T1202, T1003.003, T1055.012, T1014, T1070.001, T1546.003,
  T1053.005, T1136.001, T1098, T1059.006 — plus an import-time self-check
  that raises if any referenced technique is missing from the DB
  (the silent empty-tactic bug class cannot recur)
- `mitre_techniques` multi-label column (semicolon list);
  `mitre_technique` unchanged as highest-priority single label
- LOLBins: msbuild, forfiles, esentutl, scriptrunner, mavinject

### Severity recalibration
- Bulk-volume events lowered: 3 (0.85→0.55), 7 (0.80→0.45), 11 (0.85→0.55),
  22 (0.80→0.40), 4624 (0.70→0.45), 4672 (0.90→0.40), 4768/69 (0.80→0.45),
  4648 (0.85→0.75), 5058 (0.85→0.45), 5061 (0.80→0.35), 4104 (0.95→0.75)
- Table documented as analyst-interest prior, not maliciousness probability

### Severity split (channel-independence)
- `severity_score`: class prior + STRUCTURAL conditionals only (target
  image/object, ticket encryption, logon type, signed). Consumed by channels.
- `severity_triage` (new): adds cmdline content-flag floors (encoded 0.95,
  bypass 0.90, cradle/reflection 0.85, lolbin 0.70). Consumed by report and
  `severity_label` only. Keeps cmdline flags out of channel inputs.

### Regex hardening
- PowerShell parameter-prefix abbreviations covered (`-e`/`-ec`/`-en…`,
  `-ep bypass`, `-exec bypass`, `-w hidden`, `-noni`)
- .NET cradles added (WebClient, Invoke-RestMethod/irm, Start-BitsTransfer,
  `-urlcache`, msxml2/xmlhttp, FromBase64String); curl/wget now require a URL
  in the command line (they ship with Win10+ — bare use is benign)
- Bare `load(` removed from reflection pattern (matched any .NET/Java cmdline)
- All flag regexes converted to non-capturing groups (pandas warning fix)

### Performance / robustness
- Fully vectorized (`np.select`/maps; no `df.apply(axis=1)`): ~9 min
  projected at 45M events vs. hours
- `signed` handled as nullable boolean; unknown ≠ unsigned (see schema.py)
- Optional gating columns guarded — missing columns no-op safely
- token_fine SIG flag is now 3-state (1/0/U)

## schema.py / config.py — ingest

- Canonical schema + `IngestConfig` candidates for `target_image`,
  `target_object`, `ticket_encryption_type`, `logon_type` (Sysmon/WEF and
  ECS-style names)
- **`Signed` string-parsing fix:** Sysmon emits the STRING "true"/"false";
  the old `astype(bool)` read "false" as True. Now parsed
  (true/1/yes, false/0/no), nullable boolean, NA when column absent.
  Without this, an export lacking `Signed` tagged all 452K EID-7 image loads
  in the ICS sample as T1574.002 and every driver load as T1014.
- `fusion_method` default: `weighted_linear` → `borda`
- New: `freq_exclude_cols`, `calib_folds` (default 6), pair-scan cap note

## sequence.py — transitions

- `build_transition_counts` carries `state_severity` (count-weighted mean
  of the source event's severity prior). **This activates the
  severity-weighted JSD, which was a silent no-op** — the column never
  existed, so every state fell to the constant-0.5 fallback and S_seq was an
  unweighted mean despite the documented (and paper-claimed) weighting.
  Verified live by perturbation test (S_seq now responds to severity;
  was bit-identical before).
- Dead `_new_session_mask` removed.

## divergence.py — S_seq calibration (H3)

- **`calibrate_jsd_matched` (new):** per-host parametric bootstrap that
  recomputes the EXACT observed statistic under H0 — host's actual per-state
  sample sizes, same shrinkage estimator, same severity weights. Replaces the
  legacy global null (uniform state allocation, no shrinkage, unweighted),
  which produced 800–1862σ z-scores on real data. Matched z-scores: 2.5–99.
  Legacy function retained for ablation reference, documented as such.
- **`conformal_peer_pvalues` (new):** decision-grade p-values via
  leave-one-fold-out cross-conformal calibration. The baseline window is
  split into `calib_folds` time folds; fold k is scored against role
  baselines fit WITHOUT fold k (out-of-sample), giving each host its own
  benign S_seq distribution; scoring-window p = rank of the duration-matched
  test-fold median among those benign scores, with add-one correction.
  Iteration history (each step fixed a measured failure on real data):
  in-sample fold scoring floored 21/28 benign hosts at the minimum p;
  LOFO eliminated the floor (0/28) and the residual KS deviation is
  CONSERVATIVE (super-uniform: P(p≤t) ≤ t holds) — the formal validity
  property. Cross-host role pools (also LOFO, out-of-sample) remain as
  fallback for hosts with too few folds.
- **`BootstrapNull` corrections:** `empirical_pvalue` uses the add-one form
  (1+k)/(1+B) — can no longer return exactly 0 (the cosmetic "p=0.0000" and
  a mild anti-conservative bias); `empirical_percentile` made consistent
  (caps below 100); new `fold_over_null` effect size (observed/null-mean,
  scale-free, non-saturating); `z_score` retained but documented as
  analyst-only (σ-counts invite Gaussian tail intuition on a right-skewed
  null — see paper-figure note below).

## scoring.py — channels, fusion, gate

- **S_freq channel independence:** cmdline flag rates
  (`has_encoded_rate`, `has_download_cradle_rate`, `has_bypass_rate`)
  excluded from the IsolationForest feature matrix. S_ctx owns content
  flags; sharing them made S_freq and S_ctx corroborate by construction.
- **S_ctx de-saturation (full rewrite of components 2–3 + blend):**
  flag and pair evidence converted to rates per 1,000 events, then scored
  peer-relative via robust z (role median / 1.4826·MAD, global fallback for
  roles < 5 hosts), squashed `1−exp(−z/2)`. Replaces fixed-constant
  saturation (`1−exp(−count/5)`), which pinned 19/28 real hosts within
  ±0.001 of S_ctx=0.852 and scored the most anomalous hosts LOWER.
  After: range 0.07–0.83, std 0.17, zero hosts pinned, no inversion.
  MAD is used here (empirical peer pools can contain the adversary —
  robustness to contamination) vs. empirical quantiles for the simulated
  bootstrap null (skew, no contamination): two threats, two tools.
- **Corroboration gate consistency:** `corroboration_gate` now returns
  `gate_pass` AND `gate_reason` from one percentile-based definition
  (return type: DataFrame, was Series). The hardcoded `>= 0.7` recompute in
  `fuse_scores` (which let gate_pass and gate_reason disagree) is removed.
- TF-IDF baseline sampling seed wired to `cfg.scoring.random_seed`
  (was hardcoded 42 independent of config — reproducibility seam).

## pairs.py

- `(4768, 4769)` "Kerberoasting" pair removed from all five locations
  (DEFAULT_INTERESTING_PAIRS, PAIR_WEIGHTS, static DC discounts,
  `_pair_tactic`, docstrings) — TGT→service-ticket is the normal Kerberos
  sequence on every domain logon; RC4-gated 4769 in mapping.py replaces it.
- **Scan-cap symmetry:** scoring's inner co-occurrence loop now caps at the
  same 50-successor bound used at baseline-learning time. Previously scoring
  was uncapped — O(window_density²) counting, quadratic in volume, and
  asymmetric with the prevalence baseline the discounts were learned from.

## pipeline.py

- `fit()`: LOFO conformal fold computation + benign peer pools persisted in
  `FittedArtifacts` (`self_sseq`, `peer_sseq`, `calib_fold_seconds`);
  TF-IDF vectorizer fit on DEDUPLICATED command lines (was all duplicates —
  the 4GB-OOM cause at 650K cmdlines, and an IDF skew toward chatty hosts;
  consistent with `build_baseline_matrix` which already deduped)
- `score()`: matched bootstrap (z, fold-over-null, percentile) +
  duration-matched conformal p-values replace the legacy calibration loop;
  new columns `S_seq_x_null`, `S_seq_fold_med`

## report.py — HTML

- Calibration strip per host: `p (conformal) · Nth pct vs sampling null ·
  K× null mean`; z-score removed from all reader-facing surfaces
  (retained in triage CSV as analyst column)
- Sequence channel card sublabel: fold-over-null instead of σ
- Verified end-to-end on real data (637KB self-contained report generated
  via `ReportContext.finalise`)

---

## Known remaining items (deliberate, not regressions)

1. **Paper Eq. 15 vs. Borda:** old config comment claimed Eq. 15 is
   weighted_linear; reconcile paper text with the borda default before
   submission.
2. **Calibration power:** conformal p resolution = 1/(calib_folds+1).
   Sweep `calib_folds` (8–12 plausible) against the 2-day baseline; run the
   KS uniformity validation on the full IT network (85 hosts), not the
   28-host ICS slice.
3. **S_ctx novelty component** is still TF-IDF — the planned Markov/graph
   transition-entropy replacement remains the research item (must REPLACE,
   not supplement, per the gate-independence constraint).
4. **Event-ID channel keying:** EVENT_SEVERITY keys on bare integer IDs;
   safe only with loader channel hygiene — verify before OpTC.
5. **debug.py** still uses the LEGACY null in its calibration probe —
   its numbers will not match production; functional, flagged.
6. **run_scripts.py** is a shell-snippet cookbook with a .py extension —
   not importable, pre-existing in HEAD; rename to .sh/.md at leisure.
7. Calibration is now the fit-time cost center (~3× vs no_calibration);
   `calib_folds` is the runtime knob — do not disable calibration.
