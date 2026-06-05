"""
Token mapping, severity grading, and MITRE ATT&CK enrichment.
=============================================================
Best of:
  - pipeline_updated: multi-resolution tokens (coarse/medium/fine), context flags, MITRE stubs
  - v12_modular:      severity grading (score_to_label, grade_events)

The multi-resolution design is the key architectural decision from the README:
  - token_coarse  -> low sparsity, used as backoff in transition modeling
  - token_medium  -> primary transition state space
  - token_fine    -> context channel only (NOT in transition matrix to avoid sparsity)

IMPORTANT — role of this module in STRATA-E:
  severity_score and the mitre_* columns are TRIAGE ENRICHMENT for report
  output. They are analyst-facing context ("if this host is anomalous,
  here is what the activity would mean"), not detection evidence. They
  must NOT feed the four scoring channels or the Borda fusion — doing so
  would (a) smuggle signature logic into a statistical framework and
  (b) break the independence assumption behind the corroboration gate.

Conditional enrichment columns (all OPTIONAL — used when present):
  target_image            (Sysmon 10)  -> gates LSASS/T1003 mapping
  target_object           (Sysmon 12/13) -> gates Run-key/T1547.001 mapping
  ticket_encryption_type  (Security 4769) -> gates Kerberoasting/T1558.003
  logon_type              (Security 4624) -> severity adjustment

CHANGE LOG vs. prior revision:
  * Removed wrong mappings: 4768->Kerberoasting, 4672->T1134, EID6->T1574.002
  * EID 10 -> T1003 now requires TargetImage == lsass.exe
  * EID 12/13 -> T1547.001 now requires an autostart registry path
  * EID 22 (DNS) no longer maps to T1071 unconditionally
  * EID 6 (driver load) -> T1014 only when unsigned; EID 7 (unsigned image
    load) -> T1574.002 as low-priority context
  * Added: 1102, 4625, 4698/4702, 4720, 4728/4732/4756, 5140/5145,
    Sysmon 18/19/20/21/23/25 to severity table and (where defensible) MITRE
  * Removed 1109 (provider-ambiguous event ID; re-add with channel keying)
  * Recalibrated high-volume events (3, 7, 11, 22, 4624, 4672, 4768/69,
    5058/5061) so the critical/high labels are not dominated by bulk events
  * Severity split: severity_score (class + structural prior, consumed by
    scoring channels) vs. severity_triage (adds cmdline flag floors, consumed
    by report display + severity_label only). Keeps cmdline content out of
    channel inputs to preserve corroboration-gate independence.
  * Multi-label MITRE: mitre_techniques (semicolon list) added;
    mitre_technique remains the single highest-priority label (back-compat)
  * Regexes: PowerShell parameter-prefix abbreviations (-e/-ec/-ep bypass/
    -w hidden/-noni), .NET cradles (WebClient, IRM, BITS, urlcache);
    curl/wget now require a URL in the command line; bare "load(" removed
  * LOLBins added: msbuild, forfiles, esentutl, scriptrunner, mavinject
    (NOTE: this changes token_medium vocabulary — these were PROC:* before.
    Transition baselines must be re-fit after deploying this file.)
  * Vectorized severity + MITRE assignment (no df.apply(axis=1); matters
    at 45M-event scale)

NOTE on event-ID collisions: EVENT_SEVERITY keys on bare integer IDs.
IDs collide across channels/providers (e.g., 1000-series Application-log
crashes vs. other providers). This is safe only if the loader restricts
ingest to known (channel, event_id) pairs — loaders.py does for the
exercise data; verify the same for OpTC before trusting these priors.
"""
from __future__ import annotations

import re
from typing import Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Token mapping tables (extend these for your environment)
# ---------------------------------------------------------------------------

OFFICE_MAP: Dict[str, str] = {
    "winword.exe":   "WORD",
    "excel.exe":     "EXCEL",
    "powerpnt.exe":  "PPT",
    "outlook.exe":   "OUTLOOK",
    "onenote.exe":   "ONENOTE",
    "visio.exe":     "VISIO",
    "msaccess.exe":  "ACCESS",
}

SCRIPT_MAP: Dict[str, str] = {
    "powershell.exe":  "POWERSHELL",
    "pwsh.exe":        "POWERSHELL",
    "wscript.exe":     "WSCRIPT",
    "cscript.exe":     "CSCRIPT",
    "python.exe":      "PYTHON",
    "python3.exe":     "PYTHON",
    "cmd.exe":         "CMD",
}

BROWSER_MAP: Dict[str, str] = {
    "chrome.exe":   "CHROME",
    "msedge.exe":   "EDGE",
    "firefox.exe":  "FIREFOX",
    "iexplore.exe": "IE",
}

LOLBINS: set = {
    "regsvr32.exe", "rundll32.exe", "mshta.exe", "wmic.exe",
    "certutil.exe", "bitsadmin.exe", "msiexec.exe", "odbcconf.exe",
    "regasm.exe", "regsvcs.exe", "installutil.exe", "cmstp.exe",
    "xwizard.exe", "pcalua.exe", "syncappvpublishingserver.exe",
    # added — most commonly abused in current red-team tradecraft
    "msbuild.exe", "forfiles.exe", "esentutl.exe", "scriptrunner.exe",
    "mavinject.exe",
}

# Precomputed binary -> token lookups (vectorized token assignment)
_COARSE_LOOKUP: Dict[str, str] = {}
_MEDIUM_LOOKUP: Dict[str, str] = {}
for _b, _t in OFFICE_MAP.items():
    _COARSE_LOOKUP[_b] = "OFFICE";  _MEDIUM_LOOKUP[_b] = f"OFFICE:{_t}"
for _b, _t in SCRIPT_MAP.items():
    _COARSE_LOOKUP[_b] = "SCRIPT";  _MEDIUM_LOOKUP[_b] = f"SCRIPT:{_t}"
for _b, _t in BROWSER_MAP.items():
    _COARSE_LOOKUP[_b] = "BROWSER"; _MEDIUM_LOOKUP[_b] = f"BROWSER:{_t}"
for _b in LOLBINS:
    _COARSE_LOOKUP[_b] = "LOLBIN"
    _MEDIUM_LOOKUP[_b] = f"LOLBIN:{_b.upper().replace('.EXE', '')}"
del _b, _t

# ---------------------------------------------------------------------------
# Regex patterns for context flags
# ---------------------------------------------------------------------------
# PowerShell accepts any unambiguous parameter prefix, so -e, -ec, -en,
# -enco ... all invoke -EncodedCommand. The payload-anchored alternative
# (-e<prefix> followed by >=16 base64 chars) covers the abbreviations;
# the bare -enc/-encodedcommand alternatives cover truncated cmdlines.
ENCODED_RE = re.compile(
    r"(?i)(?:-encodedcommand\b|-enc\b"
    r"|-e(?:c|n[a-z]*)?\s+[A-Za-z0-9+/=]{16,}"
    r"|frombase64string)"
)

# curl/wget ship with Windows 10+ and are heavily used legitimately, so
# they only count as a cradle when a URL is actually present in the line.
DOWNLOAD_CRADLE_RE = re.compile(
    r"(?i)(?:invoke-webrequest|\biwr\b|invoke-restmethod|\birm\b"
    r"|downloadstring|downloadfile|downloaddata"
    r"|net\.webclient|start-bitstransfer|-urlcache|msxml2|xmlhttp"
    r"|(?:curl|wget)(?:\.exe)?\b[^\r\n]{0,300}?https?://)"
)

BYPASS_RE = re.compile(
    r"(?i)(?:-bypass\b|-nop\b|-noprofile\b|-noni\b|-noninteractive\b"
    r"|-e(?:p|xec(?:utionpolicy)?)\s+bypass|-executionpolicy\s+bypass"
    r"|-w(?:indowstyle)?\s+hidden)"
)

# Tightened: bare "load(" matched any .NET/Java/etc. command line.
REFLECTION_RE = re.compile(
    r"(?i)(?:\[?(?:system\.)?reflection\.assembly\]?::load"
    r"|loadwithpartialname"
    r"|assembly\]::load)"
)

# Autostart registry locations (gates Sysmon 12/13 -> T1547.001)
AUTOSTART_RE = re.compile(
    r"(?i)(?:\\currentversion\\run(?:once)?(?:\\|$)"
    r"|\\currentversion\\policies\\explorer\\run"
    r"|\\winlogon\\(?:shell|userinit)"
    r"|image file execution options)"
)

# ---------------------------------------------------------------------------
# EVENT_SEVERITY
# ---------------------------------------------------------------------------
# Numeric severity scores normalized to [0,1]. These are ANALYST-INTEREST
# PRIORS conditioned only on event class — i.e., "how interesting is this
# event type, on average, before looking at content." They are NOT
# probabilities of maliciousness: most high-severity classes still fire
# benignly. Content-conditional adjustments (LSASS target, autostart
# path, encoded/bypass cmdline) are applied in build_tokens and can move
# an event well above its class prior.
#
# Calibration anchors:
#   0.98 = Event 1102 (audit log cleared) — near-zero benign rate
#   0.95 = Event 8 (CreateRemoteThread), Event 25 (process tampering)
#   1.00 = reserved for conditional boosts (e.g., EID 10 w/ LSASS target)
#   <=0.55 = bulk-volume events (network conns, file creates, logons,
#            DNS, image loads) — individually low signal regardless of
#            how useful the class is in aggregate
# ---------------------------------------------------------------------------
EVENT_SEVERITY: Dict[int, float] = {
    # ---- Sysmon ----
    1:   0.65,  # Process Create
    3:   0.55,  # Network Connection (bulk volume — was 0.85)
    5:   0.30,  # Process Terminate
    6:   0.70,  # Driver Loaded (boosted if unsigned)
    7:   0.45,  # Image Load (bulk volume — was 0.80)
    8:   0.95,  # CreateRemoteThread (code injection)
    10:  0.55,  # Process Access (boosted to 1.0 on LSASS target — was flat 1.0)
    11:  0.55,  # File Create (bulk volume — was 0.85)
    12:  0.45,  # Registry Object Added/Deleted (boosted on autostart path)
    13:  0.45,  # Registry Value Set (boosted on autostart path)
    15:  0.60,  # FileCreateStreamHash (ADS)
    17:  0.60,  # Named Pipe Created
    18:  0.55,  # Named Pipe Connected
    19:  0.90,  # WMI Event Filter (persistence)
    20:  0.90,  # WMI Event Consumer (persistence)
    21:  0.90,  # WMI Filter-to-Consumer Binding (persistence)
    22:  0.40,  # DNS Query (bulk volume — was 0.80; beaconing is CADENCE's job)
    23:  0.45,  # File Delete (archived)
    25:  0.95,  # Process Tampering (hollowing / herpaderping)

    # ---- Windows Security Log ----
    4624: 0.45,  # Successful Logon (bulk; type-conditional boost below)
    4625: 0.55,  # Failed Logon (spray/brute-force signal in aggregate)
    4634: 0.20,  # Logoff
    4648: 0.75,  # Logon w/ Explicit Credentials (runas — common but notable)
    4672: 0.40,  # Special Privileges Assigned (fires on EVERY admin/SYSTEM logon — was 0.90)
    4688: 0.70,  # Process Creation (Security)
    4698: 0.85,  # Scheduled Task Created
    4702: 0.75,  # Scheduled Task Updated
    4720: 0.80,  # User Account Created
    4728: 0.85,  # Member Added to Security-Enabled Global Group
    4732: 0.85,  # Member Added to Security-Enabled Local Group
    4756: 0.85,  # Member Added to Security-Enabled Universal Group
    4768: 0.45,  # Kerberos TGT (every domain logon — was 0.80)
    4769: 0.45,  # Kerberos Service Ticket (boosted on RC4 — was 0.80)
    4776: 0.55,  # NTLM Auth
    4798: 0.70,  # User Group Membership Enumeration
    5140: 0.45,  # Network Share Accessed
    5145: 0.50,  # Network Share Object Checked (detailed)
    1102: 0.98,  # Audit Log Cleared — near-certain signal

    # ---- PowerShell ----
    4103: 0.65,  # Module Logging
    4104: 0.75,  # Script Block Logging (fires per benign script too — was 0.95;
                 #  encoded/bypass content boosts it back to >=0.90)

    # ---- Persistence / Services ----
    7045: 0.90,  # Service Installed

    # ---- Crypto ----
    5058: 0.45,  # Key File Operations (routine TLS/DPAPI volume — was 0.85)
    5061: 0.35,  # Cryptographic Operation (routine — was 0.80)

    # ---- Low Contextual Events ----
    7031: 0.35,  # Service Crashed
    7036: 0.25,  # Service Start/Stop
    1014: 0.25,  # DNS Failure
    1000: 0.25,  # App Crash
    1001: 0.25,  # Bugcheck
    1003: 0.25,  # App Hang
    600:  0.20,  # OS Startup/Shutdown
    # 1109 removed: provider-ambiguous bare ID; re-add under (channel, id) keying
}

# Content-flag severity floors — applied to severity_triage ONLY (never to
# the channel-facing severity_score). If a flag fires, the triage severity is
# raised to at least this value regardless of event class: hostile command
# content is informative independent of which log captured it.
_FLAG_SEVERITY_FLOOR = {
    "has_encoded":         0.95,
    "has_bypass":          0.90,
    "has_download_cradle": 0.85,
    "has_reflection":      0.85,
    "is_lolbin":           0.70,  # modest: msiexec/wmic have heavy benign use
}


def _basename(path) -> str:
    """Extract lowercase filename from a Windows path."""
    if path is None: return ""
    try:
        s = str(path)
        if s in ("nan", "<NA>", "None", ""): return ""
        return s.replace("/", "\\").split("\\")[-1].lower().strip()
    except Exception:
        return ""


def _opt_str_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Optional string column -> lowercase Series ('' where absent/null)."""
    if col in df.columns:
        return df[col].fillna("").astype(str).str.lower()
    return pd.Series("", index=df.index, dtype=str)


def build_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign multi-resolution behavior tokens, context flags, severity, and
    MITRE ATT&CK enrichment to each event row.

    Produces:
      token_coarse  : OFFICE | SCRIPT | BROWSER | LOLBIN | PROC
      token_medium  : OFFICE:WORD | SCRIPT:POWERSHELL | LOLBIN:RUNDLL32 | PROC:CMD.EXE
      token_fine    : token_medium + parent context + cmdline flags + integrity + signed
                      (used in context channel only, NOT in transition state space)

    Context flags:
      has_encoded / has_download_cradle / has_bypass / has_reflection / is_lolbin

    Severity:
      severity_score  : float [0,1] — class prior + STRUCTURAL conditionals only
                        (LSASS target, autostart path, RC4 ticket, logon type,
                        unsigned driver). Consumed by scoring channels.
      severity_triage : float [0,1] — severity_score with cmdline flag floors.
                        Consumed by report display only. NOT a channel input.
      severity_label  : critical / high / medium / low (from severity_triage)

    MITRE (enrichment only — see module docstring):
      mitre_technique  : single highest-priority technique ID (back-compat)
      mitre_techniques : semicolon-joined list of all applicable techniques
      mitre_tactic / mitre_name : looked up from mitre_technique
    """
    out = df.copy()

    img    = out["image"].map(_basename) if "image" in out.columns else pd.Series("", index=out.index)
    pimg   = out["parent_image"].map(_basename) if "parent_image" in out.columns else pd.Series("", index=out.index)
    cmd    = out["cmdline"].fillna("").astype(str) if "cmdline" in out.columns else pd.Series("", index=out.index)
    il     = (out["integrity_level"].fillna("UNK").astype(str).str.upper()
              if "integrity_level" in out.columns else pd.Series("UNK", index=out.index))

    # Signed: nullable boolean (True / False / NA-unknown). The unsigned-driver
    # and unsigned-image-load gates fire ONLY on known-False — an export with
    # no Signed column must not make everything look unsigned.
    if "signed" in out.columns:
        signed_raw = out["signed"].astype("boolean")
    else:
        signed_raw = pd.Series(pd.NA, index=out.index, dtype="boolean")
    signed_false = signed_raw.eq(False).fillna(False).astype(bool)   # known-unsigned
    signed_true  = signed_raw.eq(True).fillna(False).astype(bool)    # known-signed

    # Optional enrichment columns (conditional MITRE / severity gates)
    timg     = _opt_str_col(out, "target_image").map(_basename)
    tobj     = _opt_str_col(out, "target_object")
    tkt_enc  = _opt_str_col(out, "ticket_encryption_type")
    logon_t  = (pd.to_numeric(out["logon_type"], errors="coerce")
                if "logon_type" in out.columns else pd.Series(np.nan, index=out.index))

    # --- Coarse + Medium tokens (vectorized via precomputed lookups) ---
    coarse = img.map(_COARSE_LOOKUP)
    medium = img.map(_MEDIUM_LOOKUP)
    proc_medium = pd.Series(
        np.where(img != "", "PROC:" + img.str.upper(), "PROC:UNKNOWN"),
        index=out.index,
    )
    out["token_coarse"] = coarse.fillna("PROC")
    out["token_medium"] = medium.fillna(proc_medium)

    # --- Context flags ---
    out["has_encoded"]         = cmd.str.contains(ENCODED_RE).astype(bool)
    out["has_download_cradle"] = cmd.str.contains(DOWNLOAD_CRADLE_RE).astype(bool)
    out["has_bypass"]          = cmd.str.contains(BYPASS_RE).astype(bool)
    out["has_reflection"]      = cmd.str.contains(REFLECTION_RE).astype(bool)
    out["is_lolbin"]           = img.isin(LOLBINS)

    # --- Fine token (context channel only) ---
    parent = pimg.replace("", "UNKNOWN").str.upper()
    enc_flag     = out["has_encoded"].astype(int).astype(str)
    dl_flag      = out["has_download_cradle"].astype(int).astype(str)
    bypass_flag  = out["has_bypass"].astype(int).astype(str)
    sig_flag     = pd.Series(
        np.select([signed_true, signed_false], ["1", "0"], default="U"),
        index=out.index,
    )

    out["token_fine"] = (
        out["token_medium"]
        + "|PAR:" + parent
        + "|ENC:" + enc_flag
        + "|DL:"  + dl_flag
        + "|BP:"  + bypass_flag
        + "|IL:"  + il
        + "|SIG:" + sig_flag
    )

    # --- Severity: class prior + structural conditionals (CHANNEL-FACING) ---
    # severity_score is what the scoring channels consume (S_ctx component 1,
    # S_seq state weighting). It depends ONLY on event class and structural
    # event fields (target image/object, ticket encryption, logon type) —
    # never on cmdline content. The cmdline flags are owned by S_ctx
    # component 2; keeping them out of severity_score preserves the
    # independence assumption behind the corroboration gate.
    eid = pd.to_numeric(out["event_id"], errors="coerce") if "event_id" in out.columns \
          else pd.Series(np.nan, index=out.index)

    sev = eid.map(EVENT_SEVERITY).fillna(0.1).astype(float)

    # Structural conditional gates (effective when the optional columns exist)
    lsass_access = (eid == 10) & (timg == "lsass.exe")
    runkey_write = eid.isin([12, 13]) & tobj.str.contains(AUTOSTART_RE)
    rc4_ticket   = (eid == 4769) & tkt_enc.isin(["0x17", "0x18", "23", "24"])
    remote_logon = (eid == 4624) & logon_t.isin([9, 10])  # NewCredentials / RemoteInteractive
    unsigned_drv = (eid == 6) & signed_false

    sev = sev.mask(lsass_access, 1.00)
    sev = sev.mask(runkey_write, np.maximum(sev, 0.85))
    sev = sev.mask(rc4_ticket,   np.maximum(sev, 0.85))
    sev = sev.mask(remote_logon, np.maximum(sev, 0.70))
    sev = sev.mask(unsigned_drv, np.maximum(sev, 0.90))

    out["severity_score"] = sev

    # --- Severity triage (REPORT-FACING): adds cmdline content-flag floors ---
    # severity_triage = max(severity_score, flag floors). It is for analyst
    # display (report.py) and severity_label ONLY. It must NOT be fed into
    # any scoring channel — that would route cmdline flags into S_ctx twice
    # and leak them into S_seq's severity weighting.
    sev_triage = sev.copy()
    for flag, floor in _FLAG_SEVERITY_FLOOR.items():
        sev_triage = sev_triage.mask(out[flag], np.maximum(sev_triage, floor))

    out["severity_triage"] = sev_triage
    out["severity_label"] = np.select(
        [sev_triage >= 0.85, sev_triage >= 0.60, sev_triage >= 0.35],
        ["critical", "high", "medium"],
        default="low",
    )

    # --- MITRE ATT&CK mapping (vectorized, multi-label) ---
    lolbin_t = img.map(_LOLBIN_TECHNIQUE).fillna("")
    script_t = img.map(_SCRIPT_TECHNIQUE).fillna("")
    enc_t    = pd.Series(np.where(out["has_encoded"], "T1027", ""), index=out.index)
    dl_t     = pd.Series(np.where(out["has_download_cradle"], "T1105", ""), index=out.index)

    # Event-semantic mapping. Conditions are checked in priority order;
    # unconditional bulk-event mappings (all DNS -> C2, all reg writes ->
    # persistence, all TGTs -> Kerberoasting, all 4672 -> token manip)
    # were removed — they tagged routine activity with attack techniques.
    event_conditions = [
        eid == 1102,                       # audit log cleared
        lsass_access,                      # ProcessAccess w/ LSASS target ONLY
        eid == 8,                          # CreateRemoteThread
        eid == 25,                         # process tampering
        eid == 7045,                       # service installed
        eid.isin([19, 20, 21]),            # WMI event subscription
        eid == 4698,                       # scheduled task created
        runkey_write,                      # registry autostart path ONLY
        unsigned_drv,                      # unsigned driver load
        eid == 4720,                       # account created
        eid.isin([4728, 4732, 4756]),      # privileged group add
        rc4_ticket,                        # RC4 service ticket ONLY
        eid == 4798,                       # group membership enumeration
        (eid == 7) & signed_false,         # KNOWN-unsigned image load (context-grade)
    ]
    event_choices = [
        "T1070.001", "T1003", "T1055", "T1055.012", "T1543.003",
        "T1546.003", "T1053.005", "T1547.001", "T1014", "T1136.001",
        "T1098", "T1558.003", "T1087", "T1574.002",
    ]
    event_t = pd.Series(np.select(event_conditions, event_choices, default=""),
                        index=out.index)

    # Primary technique: first non-empty by confidence order.
    primary = pd.Series(np.select(
        [event_t != "", lolbin_t != "", enc_t != "", dl_t != "", script_t != ""],
        [event_t, lolbin_t, enc_t, dl_t, script_t],
        default="",
    ), index=out.index)

    # Full applicable set (components are drawn from disjoint vocabularies)
    parts = [event_t, lolbin_t, enc_t, dl_t, script_t]
    joined = parts[0].copy()
    for p in parts[1:]:
        sep = np.where((joined != "") & (p != ""), ";", "")
        joined = joined + sep + p

    out["mitre_technique"]  = primary
    out["mitre_techniques"] = joined
    out["mitre_tactic"]     = primary.map(_TECH_TO_TACTIC).fillna("")
    out["mitre_name"]       = primary.map(_TECH_TO_NAME).fillna("")

    return out


def _score_to_label(score: float) -> str:
    if score >= 0.85: return "critical"
    if score >= 0.60: return "high"
    if score >= 0.35: return "medium"
    return "low"


# ---------------------------------------------------------------------------
# MITRE ATT&CK technique database (endpoint-focused, Sysmon-relevant)
# ---------------------------------------------------------------------------
# Reference: MITRE ATT&CK Enterprise — https://attack.mitre.org/
# Every technique referenced by _LOLBIN_TECHNIQUE, _SCRIPT_TECHNIQUE, or
# the event-semantic mapping MUST have an entry here (enforced by the
# module self-check at the bottom of this file).
# ---------------------------------------------------------------------------

MITRE_TECHNIQUE_DB = {
    # ---- Execution ----
    "T1059.001": {"name": "Command and Scripting Interpreter: PowerShell",
                  "tactic": "Execution",
                  "url": "https://attack.mitre.org/techniques/T1059/001/"},
    "T1059.003": {"name": "Command and Scripting Interpreter: Windows Command Shell",
                  "tactic": "Execution",
                  "url": "https://attack.mitre.org/techniques/T1059/003/"},
    "T1059.005": {"name": "Command and Scripting Interpreter: Visual Basic",
                  "tactic": "Execution",
                  "url": "https://attack.mitre.org/techniques/T1059/005/"},
    "T1059.006": {"name": "Command and Scripting Interpreter: Python",
                  "tactic": "Execution",
                  "url": "https://attack.mitre.org/techniques/T1059/006/"},
    "T1059.007": {"name": "Command and Scripting Interpreter: JavaScript",
                  "tactic": "Execution",
                  "url": "https://attack.mitre.org/techniques/T1059/007/"},
    "T1047":     {"name": "Windows Management Instrumentation",
                  "tactic": "Execution",
                  "url": "https://attack.mitre.org/techniques/T1047/"},
    "T1106":     {"name": "Native API",
                  "tactic": "Execution",
                  "url": "https://attack.mitre.org/techniques/T1106/"},
    "T1204":     {"name": "User Execution",
                  "tactic": "Execution",
                  "url": "https://attack.mitre.org/techniques/T1204/"},
    "T1202":     {"name": "Indirect Command Execution",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1202/"},
    "T1127.001": {"name": "Trusted Developer Utilities Proxy Execution: MSBuild",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1127/001/"},

    # ---- Defense Evasion ----
    "T1218":     {"name": "System Binary Proxy Execution",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1218/"},
    "T1218.003": {"name": "System Binary Proxy Execution: CMSTP",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1218/003/"},
    "T1218.004": {"name": "System Binary Proxy Execution: InstallUtil",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1218/004/"},
    "T1218.005": {"name": "System Binary Proxy Execution: Mshta",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1218/005/"},
    "T1218.007": {"name": "System Binary Proxy Execution: Msiexec",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1218/007/"},
    "T1218.008": {"name": "System Binary Proxy Execution: Odbcconf",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1218/008/"},
    "T1218.009": {"name": "System Binary Proxy Execution: Regsvcs/Regasm",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1218/009/"},
    "T1218.010": {"name": "System Binary Proxy Execution: Regsvr32",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1218/010/"},
    "T1218.011": {"name": "System Binary Proxy Execution: Rundll32",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1218/011/"},
    "T1218.013": {"name": "System Binary Proxy Execution: Mavinject",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1218/013/"},
    "T1140":     {"name": "Deobfuscate/Decode Files or Information",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1140/"},
    "T1027":     {"name": "Obfuscated Files or Information",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1027/"},
    "T1055":     {"name": "Process Injection",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1055/"},
    "T1055.012": {"name": "Process Injection: Process Hollowing",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1055/012/"},
    "T1574.002": {"name": "Hijack Execution Flow: DLL Side-Loading",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1574/002/"},
    "T1014":     {"name": "Rootkit",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1014/"},
    "T1070.001": {"name": "Indicator Removal: Clear Windows Event Logs",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1070/001/"},

    # ---- Credential Access ----
    "T1003":     {"name": "OS Credential Dumping",
                  "tactic": "Credential Access",
                  "url": "https://attack.mitre.org/techniques/T1003/"},
    "T1003.003": {"name": "OS Credential Dumping: NTDS",
                  "tactic": "Credential Access",
                  "url": "https://attack.mitre.org/techniques/T1003/003/"},
    "T1558.003": {"name": "Steal or Forge Kerberos Tickets: Kerberoasting",
                  "tactic": "Credential Access",
                  "url": "https://attack.mitre.org/techniques/T1558/003/"},

    # ---- Persistence ----
    "T1547.001": {"name": "Boot or Logon Autostart Execution: Registry Run Keys",
                  "tactic": "Persistence",
                  "url": "https://attack.mitre.org/techniques/T1547/001/"},
    "T1543.003": {"name": "Create or Modify System Process: Windows Service",
                  "tactic": "Persistence",
                  "url": "https://attack.mitre.org/techniques/T1543/003/"},
    "T1546.003": {"name": "Event Triggered Execution: WMI Event Subscription",
                  "tactic": "Persistence",
                  "url": "https://attack.mitre.org/techniques/T1546/003/"},
    "T1053.005": {"name": "Scheduled Task/Job: Scheduled Task",
                  "tactic": "Persistence",
                  "url": "https://attack.mitre.org/techniques/T1053/005/"},
    "T1136.001": {"name": "Create Account: Local Account",
                  "tactic": "Persistence",
                  "url": "https://attack.mitre.org/techniques/T1136/001/"},
    "T1098":     {"name": "Account Manipulation",
                  "tactic": "Persistence",
                  "url": "https://attack.mitre.org/techniques/T1098/"},

    # ---- Lateral Movement ----
    "T1021":     {"name": "Remote Services",
                  "tactic": "Lateral Movement",
                  "url": "https://attack.mitre.org/techniques/T1021/"},

    # ---- Command and Control ----
    "T1071":     {"name": "Application Layer Protocol",
                  "tactic": "Command and Control",
                  "url": "https://attack.mitre.org/techniques/T1071/"},
    "T1105":     {"name": "Ingress Tool Transfer",
                  "tactic": "Command and Control",
                  "url": "https://attack.mitre.org/techniques/T1105/"},

    # ---- Discovery ----
    "T1087":     {"name": "Account Discovery",
                  "tactic": "Discovery",
                  "url": "https://attack.mitre.org/techniques/T1087/"},

    # ---- Collection ----
    "T1005":     {"name": "Data from Local System",
                  "tactic": "Collection",
                  "url": "https://attack.mitre.org/techniques/T1005/"},
}

# Fast lookup tables for vectorized .map()
_TECH_TO_TACTIC = {t: m["tactic"] for t, m in MITRE_TECHNIQUE_DB.items()}
_TECH_TO_NAME   = {t: m["name"]   for t, m in MITRE_TECHNIQUE_DB.items()}

# LOLBin -> technique mapping
_LOLBIN_TECHNIQUE = {
    "regsvr32.exe":     "T1218.010",
    "rundll32.exe":     "T1218.011",
    "mshta.exe":        "T1218.005",
    "cmstp.exe":        "T1218.003",
    "installutil.exe":  "T1218.004",
    "msiexec.exe":      "T1218.007",
    "odbcconf.exe":     "T1218.008",
    "regasm.exe":       "T1218.009",
    "regsvcs.exe":      "T1218.009",
    "mavinject.exe":    "T1218.013",
    "wmic.exe":         "T1047",
    "certutil.exe":     "T1140",
    "bitsadmin.exe":    "T1105",
    "msbuild.exe":      "T1127.001",
    "forfiles.exe":     "T1202",
    "pcalua.exe":       "T1202",
    "esentutl.exe":     "T1003.003",
    "scriptrunner.exe": "T1218",
    "xwizard.exe":      "T1218",
    "syncappvpublishingserver.exe": "T1218",
}

# Script interpreter -> technique mapping.
# wscript/cscript host both VBScript (T1059.005) and JScript (T1059.007);
# .005 chosen as the more common case. Disambiguate via script extension
# in cmdline if finer resolution is ever needed.
_SCRIPT_TECHNIQUE = {
    "powershell.exe": "T1059.001",
    "pwsh.exe":       "T1059.001",
    "cmd.exe":        "T1059.003",
    "wscript.exe":    "T1059.005",
    "cscript.exe":    "T1059.005",
    "python.exe":     "T1059.006",
    "python3.exe":    "T1059.006",
}


def _map_mitre_technique(row: pd.Series) -> str | None:
    """
    Back-compat single-row mapping (the pipeline path is vectorized inside
    build_tokens; this exists for ad-hoc/notebook use on individual rows).
    Returns the same primary technique build_tokens would assign.
    """
    mapped = build_tokens(pd.DataFrame([row]))
    t = mapped["mitre_technique"].iloc[0]
    return t if t else None


# ---------------------------------------------------------------------------
# Module self-check: every referenced technique must resolve in the DB so
# mitre_tactic / mitre_name can never silently come back empty.
# ---------------------------------------------------------------------------
_EVENT_TECHNIQUES = {
    "T1070.001", "T1003", "T1055", "T1055.012", "T1543.003", "T1546.003",
    "T1053.005", "T1547.001", "T1014", "T1136.001", "T1098", "T1558.003",
    "T1087", "T1574.002", "T1027", "T1105",
}
_missing = (
    (set(_LOLBIN_TECHNIQUE.values()) | set(_SCRIPT_TECHNIQUE.values())
     | _EVENT_TECHNIQUES) - set(MITRE_TECHNIQUE_DB)
)
if _missing:
    raise RuntimeError(
        f"mapping.py: techniques referenced but missing from MITRE_TECHNIQUE_DB: {sorted(_missing)}"
    )
