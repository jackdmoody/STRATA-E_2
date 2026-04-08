"""
HTML Report Generator
=======================
Captures all matplotlib figures during a pipeline run and assembles them,
along with interactive triage visualizations and CSV downloads, into a
single self-contained HTML file.

Usage
------
    from sysmon_pipeline.report import ReportContext

    with ReportContext(output_dir="results") as report:
        pipe = StrataPipeline(cfg)
        fitted = pipe.fit(df_baseline)
        art    = pipe.score(df_scoring, fitted)
        report.finalise(art)

Report contents
----------------
- Summary metric cards (events, hosts, corroborated, critical)
- Pipeline stage flow visualization
- Interactive host selector with severity indicators
- Per-host channel score cards with calibration detail
- Event timeline strip (color-coded by process category, height by severity)
- Top transition bars (the evidence behind the sequence channel score)
- Channel radar chart comparing selected host vs. fleet median (Chart.js)
- Suspicious command log (encoded/obfuscated executions)
- Embedded CSV downloads for all artifact tables
- Captured matplotlib figure gallery with click-to-zoom
- Run metadata table
"""
from __future__ import annotations

import base64
import io
import json
import logging
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .pipeline import StrataArtifacts

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plot capture context
# ---------------------------------------------------------------------------

class ReportContext:
    """
    Context manager that captures matplotlib figures and assembles an HTML report.

    Usage::

        from sysmon_pipeline import StrataPipeline, StrataConfig
        from sysmon_pipeline.report import ReportContext

        cfg = StrataConfig()
        with ReportContext(output_dir="results", open_browser=True) as report:
            pipe = StrataPipeline(cfg)
            fitted = pipe.fit(df_baseline)
            art    = pipe.score(df_scoring, fitted)
            report.finalise(art)
    """

    def __init__(
        self,
        output_dir:   str | Path = "results",
        open_browser: bool = False,
        title:        str  = "STRATA-E \u2014 Endpoint Behavioral Anomaly Detection Report",
    ) -> None:
        self.output_dir    = Path(output_dir)
        self.open_browser  = open_browser
        self.title         = title
        self._figures:     list[dict] = []
        self._orig_show    = None
        self._orig_backend = None
        self._run_start    = datetime.now(timezone.utc)

    def __enter__(self) -> "ReportContext":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._install_capture_hook()
        return self

    def __exit__(self, *_) -> None:
        self._uninstall_capture_hook()

    def _install_capture_hook(self) -> None:
        import matplotlib
        import matplotlib.pyplot as plt
        self._orig_backend = matplotlib.get_backend()
        matplotlib.use("Agg")
        report_ctx = self

        def _capture_show():
            fig = plt.gcf()
            if not fig.get_axes():
                plt.close(fig)
                return
            label = ""
            if fig._suptitle:
                label = fig._suptitle.get_text().split("\n")[0]
            elif fig.get_axes():
                label = fig.get_axes()[0].get_title().split("\n")[0]
            label = label.strip() or f"Figure {len(report_ctx._figures) + 1}"
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            png_b64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            report_ctx._figures.append({"label": label, "png_b64": png_b64})
            plt.close(fig)

        self._orig_show = plt.show
        plt.show = _capture_show

    def _uninstall_capture_hook(self) -> None:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            if self._orig_show is not None:
                plt.show = self._orig_show
            if self._orig_backend:
                matplotlib.use(self._orig_backend)
        except Exception:
            pass

    def _write_csv(self, df: Optional[pd.DataFrame], name: str) -> tuple[str, str]:
        if df is None or df.empty:
            return "", ""
        path = self.output_dir / name
        df.to_csv(path, index=False)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        b64 = base64.b64encode(csv_bytes).decode("utf-8")
        return name, f"data:text/csv;base64,{b64}"

    def finalise(self, art: "StrataArtifacts") -> Path:
        """Assemble the HTML report from pipeline artifacts and write to disk."""
        run_end = datetime.now(timezone.utc)

        csvs = {}
        for name, df in [
            ("triage.csv",       art.triage),
            ("seq_scores.csv",   art.seq_scores),
            ("freq_scores.csv",  art.freq_scores),
            ("ctx_scores.csv",   art.ctx_scores),
            ("drift_scores.csv", art.drift_scores),
        ]:
            fname, uri = self._write_csv(df, name)
            if fname:
                csvs[fname] = uri

        report_data = _extract_report_data(art)

        n_events = len(art.events) if art.events is not None and not art.events.empty else 0
        n_hosts  = art.events["host"].nunique() if n_events > 0 else 0
        n_roles  = 0
        if art.host_roles is not None and not art.host_roles.empty:
            n_roles = art.host_roles["role_id"].nunique()
        triage = art.triage
        n_gate_pass = int(triage["gate_pass"].sum()) if triage is not None and "gate_pass" in triage.columns else 0

        time_range = ""
        if n_events > 0:
            time_range = f"{art.events['ts'].min()} \u2014 {art.events['ts'].max()}"

        meta = {
            "Run start":        self._run_start.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Run end":          run_end.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Duration":         str(run_end - self._run_start).split(".")[0],
            "Events processed": f"{n_events:,}",
            "Hosts scored":     str(n_hosts),
            "Roles inferred":   str(n_roles),
            "Time range":       time_range,
            "Gate pass":        f"{n_gate_pass} / {len(triage) if triage is not None else 0} hosts",
        }

        html = _render_html(
            title       = self.title,
            meta        = meta,
            figures     = self._figures,
            csvs        = csvs,
            report_data = report_data,
            n_events    = n_events,
            n_hosts     = n_hosts,
            n_roles     = n_roles,
            n_gate_pass = n_gate_pass,
        )

        report_path = self.output_dir / "strata_report.html"
        report_path.write_text(html, encoding="utf-8")
        log.info("Report written to %s", report_path)
        print(f"\n  Report: {report_path.resolve()}")
        if self.open_browser:
            webbrowser.open(report_path.resolve().as_uri())
        return report_path


# ---------------------------------------------------------------------------
# Data extraction for interactive report
# ---------------------------------------------------------------------------

_MAX_TIMELINE_EVENTS = 120
_MAX_TRANSITIONS = 10


def _extract_report_data(art: "StrataArtifacts") -> dict:
    """Extract all data needed for the interactive JS report."""
    data: dict = {"triage": [], "timelines": {}, "transitions": {}}

    if art.triage is None or art.triage.empty:
        return data

    triage = art.triage.copy()
    hosts = list(triage["host"].values)

    # Triage rows
    def _safe_float(val, default=0.0):
        try:
            v = float(val)
            return default if (v != v) else v  # NaN check: NaN != NaN
        except (TypeError, ValueError):
            return default

    def _safe_int(val, default=0):
        try:
            v = float(val)
            return default if (v != v) else int(v)
        except (TypeError, ValueError):
            return default

    triage_records = []
    for _, row in triage.iterrows():
        triage_records.append({
            "host":            str(row.get("host", "")),
            "role_id":         str(row.get("role_id", "default")),
            "score":           round(_safe_float(row.get("score", 0)), 4),
            "gate_pass":       bool(row.get("gate_pass", False)),
            "gate_reason":     str(row.get("gate_reason", "") or ""),
            "S_seq":           round(_safe_float(row.get("S_seq", 0)), 4),
            "S_freq":          round(_safe_float(row.get("S_freq", 0)), 4),
            "S_ctx":           round(_safe_float(row.get("S_ctx", 0)), 4),
            "S_drift":         round(_safe_float(row.get("S_drift", 0)), 4),
            "S_seq_z":         round(_safe_float(row.get("S_seq_z", 0)), 3),
            "S_seq_pvalue":    round(_safe_float(row.get("S_seq_pvalue", 0)), 4),
            "S_seq_percentile": round(_safe_float(row.get("S_seq_percentile", 0)), 1),
            "cmdline_novelty": round(_safe_float(row.get("cmdline_novelty", 0)), 3),
            "n_events":        _safe_int(row.get("n_events", 0)),
            "n_tactics":       _safe_int(row.get("n_tactics", 0)),
            "top_tactic":      str(row.get("top_tactic", "none") or "none"),
            "max_pair_weight": round(_safe_float(row.get("max_pair_weight", 0)), 3),
            "n_pairs":         _safe_int(row.get("n_pairs_y", 0)),
            "evasion_signal":  bool(row.get("evasion_signal", False)),
            "triage_rank":     _safe_int(row.get("triage_rank", 0)),
        })
    data["triage"] = triage_records

    # Per-host event timelines
    if art.events is not None and not art.events.empty:
        for host in hosts:
            hev = art.events[art.events["host"] == host].copy()
            if hev.empty:
                data["timelines"][host] = []
                continue
            hev = hev.sort_values("ts").head(_MAX_TIMELINE_EVENTS)
            t_min = hev["ts"].min()
            events_out = []
            for _, ev in hev.iterrows():
                events_out.append({
                    "t":   round((ev["ts"] - t_min).total_seconds() / 60.0, 2),
                    "ts":  ev["ts"].strftime("%H:%M:%S"),
                    "tc":  str(ev.get("token_coarse", "")),
                    "tm":  str(ev.get("token_medium", "")),
                    "sev": str(ev.get("severity_label", "")),
                    "img": str(ev.get("image", "")).split("\\")[-1],
                    "cmd": str(ev.get("cmdline", ""))[:120],
                    "enc": bool(ev.get("has_encoded", False)),
                    "lol": bool(ev.get("is_lolbin", False)),
                    "mt":  str(ev.get("mitre_technique", "") or ""),
                    "mta": str(ev.get("mitre_tactic", "") or ""),
                    "mtn": str(ev.get("mitre_name", "") or ""),
                })
            data["timelines"][host] = events_out

    # Per-host top transitions
    if art.transition_counts is not None and not art.transition_counts.empty:
        for host in hosts:
            tc = art.transition_counts[art.transition_counts["host"] == host]
            tc_top = tc.sort_values("count", ascending=False).head(_MAX_TRANSITIONS)
            data["transitions"][host] = [
                {
                    "s": str(r["state"]),
                    "n": str(r["next_state"]),
                    "b": str(r.get("dt_bucket", "")),
                    "c": int(r["count"]),
                }
                for _, r in tc_top.iterrows()
            ]

    return data


def _severity(t: dict) -> str:
    if not t.get("gate_pass", False):
        return "LOW"
    if t.get("evasion_signal", False) or (t.get("score", 0) >= 0.95 and t.get("S_ctx", 0) >= 0.7):
        return "CRITICAL"
    if t.get("score", 0) >= 0.85:
        return "HIGH"
    if t.get("score", 0) >= 0.60:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

def _render_html(
    title:       str,
    meta:        dict,
    figures:     list[dict],
    csvs:        dict[str, str],
    report_data: dict,
    n_events:    int,
    n_hosts:     int,
    n_roles:     int,
    n_gate_pass: int,
) -> str:
    triage = report_data.get("triage", [])
    n_critical = sum(1 for t in triage if _severity(t) == "CRITICAL")
    report_json = json.dumps(report_data, separators=(",", ":"))

    pipe_stages = [
        ("Ingest",          f"{n_events:,} events"),
        ("Tokenize",        "coarse / medium / fine"),
        ("Role inference",  f"{n_roles} roles"),
        ("Sessionize",      "time-aware &Delta;t"),
        ("Baselines",       "Dirichlet shrinkage"),
        ("4-channel score", "seq &middot; freq &middot; ctx &middot; drift"),
        ("Fusion + gate",   f"{n_gate_pass} pass"),
    ]
    pipe_html = ""
    for i, (name, count) in enumerate(pipe_stages):
        if i > 0:
            pipe_html += '<span class="pa">&rarr;</span>'
        pipe_html += f'<div class="ps"><div class="pn">{name}</div><div class="pc">{count}</div></div>'

    csv_labels = {
        "triage.csv": "Triage table", "seq_scores.csv": "Sequence scores",
        "freq_scores.csv": "Frequency scores", "ctx_scores.csv": "Context scores",
        "drift_scores.csv": "Drift scores",
    }
    csv_btns = "".join(
        f'<a class="cb" href="{uri}" download="{fname}">&darr; {csv_labels.get(fname, fname)}</a>\n'
        for fname, uri in csvs.items()
    )

    gallery = "".join(
        f'<div class="fc"><div class="ft">{f["label"]}</div>'
        f'<img src="data:image/png;base64,{f["png_b64"]}" alt="{f["label"]}" loading="lazy"'
        f' onclick="openModal(this.src,\'{f["label"]}\')"></div>'
        for f in figures
    )

    meta_rows = "".join(
        f'<tr><td class="mk">{k}</td><td class="mv">{v}</td></tr>'
        for k, v in meta.items()
    )

    return _HTML_TEMPLATE.format(
        title=title,
        run_start=meta.get("Run start", ""),
        n_events=f"{n_events:,}",
        n_hosts=n_hosts,
        n_gate_pass=n_gate_pass,
        n_critical=n_critical,
        pipe_html=pipe_html,
        csv_btns=csv_btns,
        gallery=gallery,
        gallery_count=len(figures),
        meta_rows=meta_rows,
        report_json=report_json,
        gallery_section="" if not figures else (
            '<div class="sc"><div class="sh">&#x1f4ca; Diagnostic plots ('
            + str(len(figures)) + ' figures)</div><div class="sb"><div class="gal">'
            + gallery + '</div></div></div>'
        ),
    )


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
:root{{--bg:#080b12;--sf:#0e1420;--s2:#131926;--bd:#1e2638;--ac:#38bdf8;--dg:#e05c5c;--wn:#f0a500;--ok:#34d399;--tx:#c9d1e0;--mt:#5a6480;--mn:'JetBrains Mono','Consolas',monospace;--sn:'IBM Plex Sans',sans-serif}}
*{{box-sizing:border-box;margin:0;padding:0}}html{{scroll-behavior:smooth}}
body{{background:var(--bg);color:var(--tx);font-family:var(--sn);font-size:13.5px;line-height:1.65}}
a{{color:var(--ac);text-decoration:none}}
.hdr{{background:var(--sf);border-bottom:1px solid var(--bd);padding:16px 36px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}}
.hdr h1{{font-size:16px;font-weight:700;color:#fff;letter-spacing:-.3px}}
.hdr p{{font-size:11px;color:var(--mt);margin-top:1px}}
.mn{{max-width:1440px;margin:0 auto;padding:28px 36px}}
.cds{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:22px}}
.cd{{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:18px 22px;position:relative;overflow:hidden}}
.cd::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--ac)}}
.cd.d::before{{background:var(--dg)}}.cn{{font-size:40px;font-weight:700;color:var(--ac);line-height:1;font-family:var(--mn)}}
.cd.d .cn{{color:var(--dg)}}.cl{{font-size:10px;color:var(--mt);margin-top:6px;text-transform:uppercase;letter-spacing:.6px}}
.sc{{background:var(--sf);border:1px solid var(--bd);border-radius:10px;margin-bottom:18px;overflow:hidden}}
.sh{{padding:12px 20px;border-bottom:1px solid var(--bd);font-weight:600;font-size:13px;color:#fff;display:flex;align-items:center;gap:8px}}
.sb{{padding:20px}}
.pf{{display:flex;align-items:center;flex-wrap:wrap;gap:6px;font-size:11.5px}}
.ps{{background:var(--s2);border:1px solid var(--bd);border-radius:7px;padding:7px 13px;text-align:center;min-width:90px}}
.pn{{font-weight:600;color:#fff;font-size:12px}}.pc{{color:var(--ac);font-size:10.5px;font-family:var(--mn);margin-top:2px}}
.pa{{color:var(--mt);font-size:15px}}
.cb{{display:inline-block;padding:8px 16px;background:transparent;border:1px solid var(--ac);color:var(--ac);border-radius:6px;font-size:12px;font-weight:600;transition:background .15s}}
.cb:hover{{background:rgba(56,189,248,.1)}}
.gal{{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:14px}}
.fc{{background:var(--bg);border:1px solid var(--bd);border-radius:8px;overflow:hidden}}
.ft{{padding:7px 13px;font-size:10px;font-weight:600;color:var(--mt);border-bottom:1px solid var(--bd)}}
.fc img{{width:100%;display:block;cursor:zoom-in}}
.mt{{border-collapse:collapse;width:100%}}.mt td{{padding:7px 12px;border-bottom:1px solid var(--bd)}}
.mk{{color:var(--mt);width:180px;font-size:11px}}.mv{{font-family:var(--mn);font-size:11px}}
.modal{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.9);z-index:1000;align-items:center;justify-content:center;flex-direction:column;gap:10px}}
.modal.open{{display:flex}}.modal img{{max-width:92vw;max-height:88vh;border-radius:6px}}
.modal-close{{position:absolute;top:16px;right:24px;font-size:26px;color:#fff;cursor:pointer}}
@media(max-width:1024px){{.cds{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:640px){{.mn{{padding:16px}}.gal{{grid-template-columns:1fr}}.hdr{{padding:13px 18px}}}}
</style>
</head>
<body>
<div class="hdr">
  <div style="display:flex;align-items:center;gap:14px">
    <span style="font-size:24px">&#x1f6e1;</span>
    <div><h1>{title}</h1><p>Structural and temporal role-aware threat analytics for endpoint telemetry</p></div>
  </div>
  <div style="font-family:var(--mn);font-size:10px;color:var(--mt)">{run_start}</div>
</div>
<div class="mn">
  <div class="cds">
    <div class="cd"><div class="cn">{n_events}</div><div class="cl">Events processed</div></div>
    <div class="cd"><div class="cn">{n_hosts}</div><div class="cl">Hosts scored</div></div>
    <div class="cd"><div class="cn">{n_gate_pass}</div><div class="cl">Corroborated hosts</div></div>
    <div class="cd d"><div class="cn">{n_critical}</div><div class="cl">Critical findings</div></div>
  </div>
  <div class="sc"><div class="sh">&#x1f517; Pipeline flow</div><div class="sb"><div class="pf">{pipe_html}</div></div></div>
  <div class="sc"><div class="sh">&#x1f3af; Host investigation dashboard <span style="font-size:11px;color:var(--mt);font-weight:400;margin-left:auto">Click a host to investigate &mdash; hover timeline marks for event detail</span></div><div class="sb" id="dashboard"></div></div>
  <div class="sc"><div class="sh">&#x1f4e5; Download results</div><div class="sb"><div style="display:flex;flex-wrap:wrap;gap:10px">{csv_btns}</div></div></div>
  {gallery_section}
  <div class="sc"><div class="sh">&#x2139; Run metadata</div><div class="sb"><table class="mt">{meta_rows}</table></div></div>
</div>
<div class="modal" id="modal" onclick="closeModal()">
  <span class="modal-close" onclick="closeModal()">&#x2715;</span>
  <div style="color:var(--mt);font-size:11px" id="modal-title"></div>
  <img id="modal-img" src="" alt="">
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
function openModal(s,t){{document.getElementById('modal-img').src=s;document.getElementById('modal-title').textContent=t;document.getElementById('modal').classList.add('open')}}
function closeModal(){{document.getElementById('modal').classList.remove('open')}}
document.addEventListener('keydown',function(e){{if(e.key==='Escape')closeModal()}});

var D={report_json};
var COL={{SCRIPT:'#7F77DD',LOLBIN:'#D85A30',OFFICE:'#1D9E75',PROC:'#888780',BROWSER:'#378ADD'}};
var SEV={{CRITICAL:'#e05c5c',HIGH:'#f0a500',MEDIUM:'#378ADD',LOW:'#5a6480'}};
var sel=D.triage[0]?D.triage[0].host:'';
var radar=null;

function sev(t){{if(!t.gate_pass)return'LOW';if(t.evasion_signal||(t.score>=0.95&&t.S_ctx>=0.7))return'CRITICAL';if(t.score>=0.85)return'HIGH';if(t.score>=0.60)return'MEDIUM';return'LOW'}}
function esc(s){{return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}}

function render(){{
  var t=D.triage.find(function(r){{return r.host===sel}})||D.triage[0];
  if(!t){{document.getElementById('dashboard').innerHTML='<p style="color:var(--mt)">No triage data.</p>';return}}
  var sv=sev(t);
  var ev=D.timelines[sel]||[];
  var tr=D.transitions[sel]||[];
  var h='';

  h+='<div style="margin-bottom:20px"><div style="display:flex;flex-wrap:wrap;gap:6px">';
  D.triage.forEach(function(r){{
    var s=sev(r),a=r.host===sel;
    h+='<button onclick="sel=\\''+r.host+'\\';render()" style="padding:6px 14px;border-radius:8px;font-size:12px;font-family:var(--mn);cursor:pointer;transition:all .15s;background:'+(a?'rgba(56,189,248,.12)':'var(--s2)')+';border:1px solid '+(a?'var(--ac)':'var(--bd)')+';color:'+(a?'var(--ac)':'var(--tx)')+'"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:'+SEV[s]+';margin-right:6px;vertical-align:middle"></span>'+r.host.replace('.corp.local','')+'&middot;'+r.score.toFixed(2)+(r.gate_pass?' &#10003;':'')+'</button>';
  }});
  h+='</div></div>';

  h+='<div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-bottom:20px">';
  h+='<div style="background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:14px"><div style="font-size:11px;color:var(--mt)">Fused score</div><div style="font-size:26px;font-weight:700;font-family:var(--mn);color:#fff">'+t.score.toFixed(3)+'</div><span style="display:inline-block;margin-top:4px;padding:2px 8px;border-radius:6px;font-size:10px;font-weight:600;background:'+SEV[sv]+'22;color:'+SEV[sv]+'">'+sv+'</span></div>';
  var chans=[['Sequence',t.S_seq,'z = '+(t.S_seq_z||0).toFixed(1)],['Frequency',t.S_freq,'IsolationForest'],['Context',t.S_ctx,'novelty = '+(t.cmdline_novelty||0).toFixed(2)],['Drift',t.S_drift,'vs. prior window']];
  chans.forEach(function(c){{
    h+='<div style="background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:14px"><div style="font-size:11px;color:var(--mt)">'+c[0]+'</div><div style="font-size:26px;font-weight:700;font-family:var(--mn);color:'+(c[1]>=0.5?'var(--dg)':'#fff')+'">'+c[1].toFixed(3)+'</div><div style="font-size:10px;color:var(--mt)">'+c[2]+'</div></div>';
  }});
  h+='</div>';

  h+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px">';
  h+='<div style="background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:12px 16px;font-size:12px"><span style="font-weight:600;color:'+(t.gate_pass?'var(--ok)':'var(--mt)')+'">'+(t.gate_pass?'PASS':'FAIL')+'</span> &mdash; '+esc(t.gate_reason)+(!t.gate_pass?' (needs &ge;2 channels above threshold)':'')+'</div>';
  h+='<div style="background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:12px 16px;font-size:12px;font-family:var(--mn)">z = '+(t.S_seq_z||0).toFixed(2)+' &middot; p = '+(t.S_seq_pvalue||0).toFixed(4)+' &middot; '+(t.S_seq_percentile||0).toFixed(0)+'th percentile for role</div>';
  h+='</div>';

  h+='<div style="font-size:13px;font-weight:600;color:#fff;margin-bottom:8px">Event timeline <span style="font-weight:400;color:var(--mt);font-size:11px">first '+ev.length+' events &mdash; colored by process category, tall = critical severity</span></div>';
  h+='<div style="background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:16px;margin-bottom:20px;overflow-x:auto">';
  if(ev.length){{
    var mx=Math.max.apply(null,ev.map(function(e){{return e.t}}))||1;
    h+='<div style="position:relative;height:80px;min-width:600px">';
    ev.forEach(function(e){{
      var x=e.t/mx*100;
      var c=COL[e.tc]||'#888';
      var cr=e.sev==='critical';
      h+='<div title="'+esc(e.ts+' '+e.tm+' ['+e.sev+']'+(e.mt?' '+e.mt+' '+e.mtn:'')+' '+(e.mta||'')+'\\n'+e.cmd)+'" style="position:absolute;left:'+x+'%;top:'+(cr?8:28)+'px;width:'+(cr?6:4)+'px;height:'+(cr?44:26)+'px;background:'+c+';border-radius:1px;opacity:'+(cr?1:.7)+';cursor:help;transform:translateX(-50%)'+(e.mt?';box-shadow:0 0 0 1px rgba(255,255,255,.3)':'')+'" data-mt="'+(e.mt||'')+'"></div>';
    }});
    h+='<div style="position:absolute;bottom:0;left:0;right:0;height:1px;background:var(--bd)"></div>';
    h+='<div style="position:absolute;bottom:-16px;left:0;font-size:10px;color:var(--mt);font-family:var(--mn)">'+(ev[0]?ev[0].ts:'')+'</div>';
    h+='<div style="position:absolute;bottom:-16px;right:0;font-size:10px;color:var(--mt);font-family:var(--mn)">'+(ev[ev.length-1]?ev[ev.length-1].ts:'')+'</div>';
    h+='</div>';
    h+='<div style="display:flex;gap:16px;margin-top:24px;font-size:11px;color:var(--mt)">';
    Object.keys(COL).forEach(function(k){{h+='<span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:10px;border-radius:2px;background:'+COL[k]+'"></span>'+k+'</span>'}});
    h+='<span style="margin-left:auto;font-style:italic">Hover marks for event detail</span></div>';
  }}else{{h+='<div style="color:var(--mt);font-size:12px">No timeline data</div>'}}
  h+='</div>';

  h+='<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px">';
  h+='<div><div style="font-size:13px;font-weight:600;color:#fff;margin-bottom:8px">Top transitions <span style="font-weight:400;color:var(--mt);font-size:11px">evidence behind sequence score</span></div>';
  h+='<div style="background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:12px">';
  if(tr.length){{
    var mc=Math.max.apply(null,tr.map(function(t){{return t.c}}));
    tr.forEach(function(t){{
      var w=Math.max(8,t.c/mc*100);
      var fc=COL[t.s.split(':')[0]]||'#888';
      var tc=COL[t.n.split(':')[0]]||'#888';
      h+='<div style="display:flex;align-items:center;gap:6px;margin-bottom:5px"><span style="font-family:var(--mn);min-width:105px;font-size:10px;text-align:right;color:'+fc+'">'+esc(t.s)+'</span><span style="color:var(--mt);font-size:9px">&rarr;</span><span style="font-family:var(--mn);min-width:105px;font-size:10px;color:'+tc+'">'+esc(t.n)+'</span><div style="flex:1;height:5px;background:var(--bd);border-radius:3px;overflow:hidden"><div style="width:'+w+'%;height:100%;background:'+fc+';opacity:.6;border-radius:3px"></div></div><span style="font-family:var(--mn);font-size:10px;color:var(--mt);min-width:22px;text-align:right">'+t.c+'</span></div>';
    }});
  }}else{{h+='<div style="color:var(--mt);font-size:12px">No transitions</div>'}}
  h+='</div></div>';

  h+='<div><div style="font-size:13px;font-weight:600;color:#fff;margin-bottom:8px">Channel comparison <span style="font-weight:400;color:var(--mt);font-size:11px">selected vs. fleet median</span></div>';
  h+='<div style="background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:16px"><div style="position:relative;width:100%;height:260px"><canvas id="radarC"></canvas></div></div></div>';
  h+='</div>';

  var susp=ev.filter(function(e){{return e.enc}});
  if(susp.length){{
    h+='<div style="font-size:13px;font-weight:600;color:#fff;margin-bottom:8px">Suspicious commands <span style="font-weight:400;color:var(--mt);font-size:11px">encoded / obfuscated execution ('+susp.length+' found)</span></div>';
    h+='<div style="background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:12px;margin-bottom:20px">';
    susp.slice(0,8).forEach(function(e){{
      h+='<div style="display:flex;gap:8px;padding:5px 0;border-bottom:1px solid var(--bd);font-size:12px"><span style="font-family:var(--mn);color:var(--mt);min-width:60px">'+esc(e.ts)+'</span><span style="font-family:var(--mn);color:var(--dg);word-break:break-all">'+esc(e.cmd)+'</span></div>';
    }});
    h+='</div>';
  }}

  var mitre=ev.filter(function(e){{return e.mt}});
  if(mitre.length){{
    var tcounts={{}};
    mitre.forEach(function(e){{tcounts[e.mt]=(tcounts[e.mt]||{{n:0,name:e.mtn||e.mt,tactic:e.mta||'',tid:e.mt}});tcounts[e.mt].n++}});
    var tlist=Object.values(tcounts).sort(function(a,b){{return b.n-a.n}});
    h+='<div style="font-size:13px;font-weight:600;color:#fff;margin-bottom:8px">MITRE ATT&amp;CK techniques <span style="font-weight:400;color:var(--mt);font-size:11px">'+tlist.length+' techniques across '+mitre.length+' events</span></div>';
    h+='<div style="background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:12px;margin-bottom:20px">';
    tlist.forEach(function(t){{
      var w=Math.max(8,t.n/tlist[0].n*100);
      h+='<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">';
      h+='<a href="https://attack.mitre.org/techniques/'+t.tid.replace('.','/') +'/" target="_blank" rel="noopener" style="font-family:var(--mn);min-width:80px;font-size:11px;color:var(--ac)">'+esc(t.tid)+'</a>';
      h+='<span style="min-width:220px;font-size:11px;color:var(--tx)">'+esc(t.name)+'</span>';
      h+='<span style="min-width:110px;font-size:10px;color:var(--mt)">'+esc(t.tactic)+'</span>';
      h+='<div style="flex:1;height:5px;background:var(--bd);border-radius:3px;overflow:hidden"><div style="width:'+w+'%;height:100%;background:var(--ac);opacity:.5;border-radius:3px"></div></div>';
      h+='<span style="font-family:var(--mn);font-size:10px;color:var(--mt);min-width:30px;text-align:right">'+t.n+'</span>';
      h+='</div>';
    }});
    h+='</div>';
  }}

  if(t.evasion_signal){{
    h+='<div style="background:rgba(224,92,92,.08);border:1px solid rgba(224,92,92,.25);border-radius:8px;padding:14px 18px;margin-bottom:20px;font-size:12px;color:#e07777"><strong>&#x26A0; Evasion signal</strong> &mdash; sudden telemetry drop detected on '+esc(t.host)+'. Verify Sysmon agent health.</div>';
  }}

  document.getElementById('dashboard').innerHTML=h;
  renderRadar(t);
}}

function renderRadar(t){{
  var ctx=document.getElementById('radarC');
  if(!ctx)return;
  if(radar)radar.destroy();
  var med={{}};
  ['S_seq','S_freq','S_ctx','S_drift'].forEach(function(k){{
    var v=D.triage.map(function(r){{return r[k]}}).sort(function(a,b){{return a-b}});
    med[k]=v[Math.floor(v.length/2)];
  }});
  radar=new Chart(ctx,{{
    type:'radar',
    data:{{
      labels:['Sequence','Frequency','Context','Drift'],
      datasets:[
        {{label:sel.replace('.corp.local',''),data:[t.S_seq,t.S_freq,t.S_ctx,t.S_drift],borderColor:'#7F77DD',backgroundColor:'#7F77DD22',pointBackgroundColor:'#7F77DD',borderWidth:2}},
        {{label:'Fleet median',data:[med.S_seq,med.S_freq,med.S_ctx,med.S_drift],borderColor:'#888780',backgroundColor:'#88878011',pointBackgroundColor:'#888780',borderWidth:1,borderDash:[4,4]}}
      ]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      scales:{{r:{{beginAtZero:true,max:1,ticks:{{stepSize:.25,font:{{size:10}},color:'#5a6480'}},grid:{{color:'#1e2638'}},pointLabels:{{font:{{size:12}},color:'#c9d1e0'}}}}}},
      plugins:{{legend:{{labels:{{color:'#c9d1e0',font:{{size:11}}}}}}}}
    }}
  }});
}}

render();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_with_report(
    input_path:    str,
    output_dir:    str = "results",
    config_path:   str | None = None,
    ablation:      str = "full",
    open_browser:  bool = True,
) -> None:
    """
    One-call convenience: load CSV, run pipeline, generate HTML report.

    Usage::

        from sysmon_pipeline.report import run_with_report
        run_with_report("data/sysmon.csv", output_dir="results", open_browser=True)
    """
    from .config import StrataConfig, AblationConfig
    from .pipeline import StrataPipeline

    if config_path:
        cfg = StrataConfig.from_json(config_path)
    else:
        ablation_map = {
            "full":              AblationConfig.full_pipeline,
            "sequence_only":     AblationConfig.sequence_only,
            "no_shrinkage":      AblationConfig.no_shrinkage,
            "no_role_baselining": AblationConfig.no_role_baselining,
            "no_drift":          AblationConfig.no_drift,
        }
        cfg = StrataConfig(ablation=ablation_map.get(ablation, AblationConfig.full_pipeline)())

    cfg.io.input_path = Path(input_path)
    cfg.io.output_dir = Path(output_dir)

    if str(input_path).endswith(".csv"):
        raw = pd.read_csv(input_path, low_memory=False)
    else:
        raw = pd.read_json(input_path, lines=True)

    with ReportContext(output_dir=output_dir, open_browser=open_browser) as report:
        pipe = StrataPipeline(cfg)
        art  = pipe.fit_score(raw)
        report.finalise(art)
