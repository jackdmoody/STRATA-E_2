# Changelog

## v2.0.0 (2026-04-08)

### Major Changes
- **Role assignment**: Replaced KMeans clustering with hostname-pattern regex matching. Produces semantic role names (dc, mail, workstation, admin, etc.) instead of opaque cluster IDs. 11 default patterns.
- **Context channel**: Added TF-IDF command-line novelty scoring alongside flag-based context scoring. Role-conditioned pair weight discounting suppresses expected pairs per role.
- **CLI overhaul**: New `sysmon_pipeline/cli.py` with `--no-split`, `--eval`, `--eval-full`, `--light-plots`, `--baseline-days`, `--score-days` flags. Entrypoint changed to `sysmon_pipeline.cli:main`.
- **Extended evaluation framework**: `eval_extended.py` with 11 sections: channel contribution, H2/H4/H5 plots, deployment projections, workload reduction, per-host explainability, SHAP importance, reproducibility, baseline comparison, ablation study, sensitivity sweeps, scalability benchmarks.
- **Analysis module**: New `sysmon_pipeline/analysis.py` with SHAP feature importance, deployment-prevalence projections, error analysis, latency benchmarking, and channel taxonomy.
- **Real-world validation**: Evaluated on 45.4M events across two independent networks from a 14-day military cyber exercise (57-host IT network + 28-host ICS/OT network).

### Performance
- TF-IDF bottleneck fixed: 20+ min → 43 sec via pre-built compact baseline matrix with deduplication and sampling.
- Vectorized bootstrap calibration with (role, n_bin) caching.
- Empirical (distribution-free) p-values replacing Gaussian CDF assumptions.

### Bug Fixes
- NaN handling in `report.py` with safe float/int converters.
- `fit_score()` now stores `_last_fitted` on pipeline instance for CLI access.
- `build_tokens()` import path corrected for eval scripts.

## v1.1.0 (2026-04-04)

### Changes
- Added `--fast` mode (bootstrap=200, IF trees=50, TF-IDF features=300).
- Per-stage timing instrumentation in `score()`.
- Parquet ingestion support.
- Debug pipeline (`StrataDebugPipeline`) for notebook-based development.

## v1.0.0 (2026-03-15)

### Initial Release
- Four-channel behavioral anomaly detection (sequence, frequency, context, drift).
- Hierarchical Dirichlet shrinkage for peer-role baselines.
- Bootstrap-calibrated JSD significance testing.
- Corroboration gate with Borda rank fusion.
- Self-contained HTML report with interactive host investigation dashboard.
- Synthetic data generator for testing.
- DARPA TC dataset loader.
- Five-hypothesis evaluation framework (H1-H5).
