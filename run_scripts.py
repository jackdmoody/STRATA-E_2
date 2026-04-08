# ============================================================
# STRATA-E — Working Python Scripts
# Use these if the CLI has issues. Copy-paste into PowerShell.
# ============================================================


# ────────────────────────────────────────────────────────────
# 1. PIPELINE + HTML REPORT (fit_score on full dataset)
# ────────────────────────────────────────────────────────────
# Use for: any dataset where you want the interactive HTML report
# Time: ~2 min (1M events), ~20 min (15M events)

python -c "
import pandas as pd
from sysmon_pipeline import StrataPipeline, StrataConfig
from sysmon_pipeline.report import ReportContext

df = pd.read_parquet('data/152_windows_all.parquet')
for col in ['Image','ParentImage','CommandLine','ParentCommandLine','IntegrityLevel']:
    if col in df.columns:
        df[col] = df[col].replace({'UNKNOWN': pd.NA, '-': pd.NA})
if 'event_provider' in df.columns:
    df = df[df['event_provider'] != 'Puppet']
if 'host' in df.columns:
    df = df[~df['host'].isin({'unknown', 'UNKNOWN'})].copy()

print('Loaded:', len(df), 'events')

cfg = StrataConfig.fast()
with ReportContext(output_dir='results/report', open_browser=True) as report:
    pipe = StrataPipeline(cfg)
    art = pipe.fit_score(df)
    report.finalise(art)
print('Done - report in results/report/')
"


# ────────────────────────────────────────────────────────────
# 2. PIPELINE + HTML REPORT (with baseline/scoring split)
# ────────────────────────────────────────────────────────────
# Use for: datasets with 10+ days of data
# This activates the drift channel and produces valid H3 results

python -c "
import pandas as pd
from sysmon_pipeline import StrataPipeline, StrataConfig
from sysmon_pipeline.report import ReportContext
from sysmon_pipeline.loaders import split_time_windows
from sysmon_pipeline.schema import normalize_schema

df = pd.read_parquet('data/152_windows_all.parquet')
for col in ['Image','ParentImage','CommandLine','ParentCommandLine','IntegrityLevel']:
    if col in df.columns:
        df[col] = df[col].replace({'UNKNOWN': pd.NA, '-': pd.NA})
if 'event_provider' in df.columns:
    df = df[df['event_provider'] != 'Puppet']

cfg = StrataConfig.fast()
df_norm = normalize_schema(df, cfg)
baseline_df, score_df = split_time_windows(df_norm, ts_col='ts', baseline_days=10, score_days=4)
print('Baseline:', len(baseline_df), 'events')
print('Scoring:', len(score_df), 'events')

pipe = StrataPipeline(cfg)
fitted = pipe.fit(baseline_df)

with ReportContext(output_dir='results_split/report', open_browser=True) as report:
    art = pipe.score(score_df, fitted, prior_window_df=baseline_df)
    report.finalise(art)
print('Done - report in results_split/report/')
"


# ────────────────────────────────────────────────────────────
# 3. EVALUATION METRICS (quick mode)
# ────────────────────────────────────────────────────────────
# Use for: generating H2/H4/H5 plots, SHAP, workload reduction,
#          explainability, reproducibility
# Time: ~10 min (1M events), ~1.5 hours (15M events)

python eval_extended.py --input data\152_windows_all.parquet --output results --skip-ablation --skip-sensitivity --skip-scalability


# ────────────────────────────────────────────────────────────
# 4. EVALUATION METRICS (full mode)
# ────────────────────────────────────────────────────────────
# Use for: ablation study, sensitivity sweeps, scalability,
#          baseline comparison — run overnight
# Time: ~35 min (1M events), ~8 hours (15M events)

python eval_extended.py --input data\152_windows_all.parquet --output results


# ────────────────────────────────────────────────────────────
# 5. PIPELINE + REPORT + EVAL (all in one, same output folder)
# ────────────────────────────────────────────────────────────
# Run these two commands back to back — both write to results/

python -c "
import pandas as pd
from sysmon_pipeline import StrataPipeline, StrataConfig
from sysmon_pipeline.report import ReportContext

df = pd.read_parquet('data/152_windows_all.parquet')
for col in ['Image','ParentImage','CommandLine','ParentCommandLine','IntegrityLevel']:
    if col in df.columns:
        df[col] = df[col].replace({'UNKNOWN': pd.NA, '-': pd.NA})
if 'event_provider' in df.columns:
    df = df[df['event_provider'] != 'Puppet']

cfg = StrataConfig.fast()
with ReportContext(output_dir='results/report', open_browser=True) as report:
    pipe = StrataPipeline(cfg)
    art = pipe.fit_score(df)
    report.finalise(art)
print('Report done')
"

python eval_extended.py --input data\152_windows_all.parquet --output results --skip-ablation --skip-sensitivity --skip-scalability


# ────────────────────────────────────────────────────────────
# 6. LARGE DATASET WITH SUBSAMPLING (30M+ events)
# ────────────────────────────────────────────────────────────
# Use for: datasets where one host has disproportionate events
# Caps each host at 300k events to prevent OOM crashes

python -c "
import pandas as pd
from sysmon_pipeline import StrataPipeline, StrataConfig
from sysmon_pipeline.report import ReportContext

df = pd.read_parquet('data/201_windows_all.parquet')
for col in ['Image','ParentImage','CommandLine','ParentCommandLine','IntegrityLevel']:
    if col in df.columns:
        df[col] = df[col].replace({'UNKNOWN': pd.NA, '-': pd.NA})

print('Loaded:', len(df), 'events,', df['host'].nunique(), 'hosts')
print('Top hosts:')
print(df['host'].value_counts().head(5).to_string())

# Cap each host at 300k events
sampled = []
for h in df['host'].unique():
    hdf = df[df['host'] == h]
    if len(hdf) > 300000:
        hdf = hdf.sample(n=300000, random_state=42)
    sampled.append(hdf)
df = pd.concat(sampled, ignore_index=True)
print('After sampling:', len(df), 'events')

cfg = StrataConfig.fast()
with ReportContext(output_dir='results_201/report', open_browser=True) as report:
    pipe = StrataPipeline(cfg)
    art = pipe.fit_score(df)
    report.finalise(art)
print('Done')
"


# ────────────────────────────────────────────────────────────
# 7. SAVE SAMPLED PARQUET (for eval_extended on large data)
# ────────────────────────────────────────────────────────────
# Creates a smaller parquet file that eval_extended can run
# repeatedly without crashing

python -c "
import pandas as pd
df = pd.read_parquet('data/201_windows_all.parquet')
for col in ['Image','ParentImage','CommandLine','ParentCommandLine','IntegrityLevel']:
    if col in df.columns:
        df[col] = df[col].replace({'UNKNOWN': pd.NA, '-': pd.NA})

# Cap the largest host
top_host = df['host'].value_counts().index[0]
top_count = df['host'].value_counts().iloc[0]
if top_count > 500000:
    others = df[df['host'] != top_host]
    monster = df[df['host'] == top_host].sample(n=500000, random_state=42)
    df = pd.concat([others, monster], ignore_index=True)
    print(f'Capped {top_host} from {top_count} to 500k')

df.to_parquet('data/201_sampled.parquet', index=False)
print('Saved:', len(df), 'events,', df['host'].nunique(), 'hosts')
"

python eval_extended.py --input data\201_sampled.parquet --output results_201 --skip-ablation --skip-sensitivity --skip-scalability


# ────────────────────────────────────────────────────────────
# 8. INVESTIGATE A SPECIFIC HOST
# ────────────────────────────────────────────────────────────
# Replace HOST_NAME and DATA_FILE with your values

python -c "
import pandas as pd
df = pd.read_parquet('data/152_windows_all.parquet')
df['CommandLine'] = df['CommandLine'].replace({'UNKNOWN': pd.NA, '-': pd.NA})

host = df[df['host'] == 'asfbn-s6-1']
print('Events:', len(host))
print()
print('Top processes:')
print(host['Image'].value_counts().head(15).to_string())
print()
print('Top parents:')
print(host['ParentImage'].value_counts().head(10).to_string())
print()
print('Unique processes:', host['Image'].nunique())
print()
print('Sample commands:')
cmds = host['CommandLine'].dropna().drop_duplicates()
print(f'Unique commands: {len(cmds)}')
for c in cmds.head(15):
    print(' ', str(c)[:150])
"


# ────────────────────────────────────────────────────────────
# 9. COMPARE TWO HOSTS (flagged vs normal peer)
# ────────────────────────────────────────────────────────────

python -c "
import pandas as pd
df = pd.read_parquet('data/152_windows_all.parquet')

flagged = df[df['host'] == 'asfbn-s6-1']
normal = df[df['host'] == 'asfbn-s1-1']

print('=== FLAGGED HOST (asfbn-s6-1) ===')
print(f'Events: {len(flagged)}')
print(flagged['Image'].value_counts(normalize=True).head(10).to_string())
print()
print('=== NORMAL PEER (asfbn-s1-1) ===')
print(f'Events: {len(normal)}')
print(normal['Image'].value_counts(normalize=True).head(10).to_string())
"


# ────────────────────────────────────────────────────────────
# 10. H1-H5 HYPOTHESIS TESTS (from eval_asfbn.py)
# ────────────────────────────────────────────────────────────

python eval_asfbn.py --input data\first_sample_3.parquet --output results_h1h5
