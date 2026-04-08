"""
Token mapping and severity grading.
=====================================
Best of:
  - pipeline_updated: multi-resolution tokens (coarse/medium/fine), context flags, MITRE stubs
  - v12_modular:      severity grading (score_to_label, grade_events)

The multi-resolution design is the key architectural decision from the README:
  - token_coarse  -> low sparsity, used as backoff in transition modeling
  - token_medium  -> primary transition state space
  - token_fine    -> context channel only (NOT in transition matrix to avoid sparsity)
"""
from __future__ import annotations

import re
from typing import Dict
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
}

# Regex patterns for context flags
ENCODED_RE       = re.compile(r"(?i)(-enc\b|-encodedcommand\b|-e\s+[A-Za-z0-9+/]{20})")
DOWNLOAD_CRADLE_RE = re.compile(r"(?i)(iwr|invoke-webrequest|curl|wget|downloadstring|downloadfile)\b")
BYPASS_RE        = re.compile(r"(?i)(-bypass|-nop\b|-noprofile|-executionpolicy\s+bypass)")
REFLECTION_RE    = re.compile(r"(?i)(reflection\.assembly|loadwithpartialname|load\()")

# ---------------------------------------------------------------------------
# EVENT_SEVERITY
# ---------------------------------------------------------------------------
# Numeric severity scores normalized to [0,1] from original domain-expert
# integer scores (divide by 100). Tune these to match your environment and
# threat model. Higher = more interesting for threat hunting.
#
# Key anchors:
#   1.00 = Event 10 (ProcessAccess / LSASS) — almost always malicious
#   0.95 = Event 8 (CreateRemoteThread) and Event 4104 (PS script block)
#   0.20 = Application crashes — low signal, rarely adversary-driven
# ---------------------------------------------------------------------------
EVENT_SEVERITY: Dict[int, float] = {
    # ---- Sysmon High-Value Events ----
    1:   0.70,  # Sysmon Process Create
    3:   0.85,  # Sysmon Network Connection
    5:   0.40,  # Sysmon Process Terminate
    6:   0.70,  # Sysmon Driver Loaded
    7:   0.80,  # Sysmon Image Load (DLL)
    8:   0.95,  # Sysmon CreateRemoteThread (code injection)
    10:  1.00,  # Sysmon Process Access (e.g., LSASS access)
    11:  0.85,  # Sysmon File Create (payload drop)
    12:  0.75,  # Sysmon Registry Object Added/Deleted
    13:  0.75,  # Sysmon Registry Value Set
    15:  0.60,  # Sysmon FileCreateStreamHash
    17:  0.70,  # Sysmon Named Pipe Events
    22:  0.80,  # Sysmon DNS Query (C2 hunting)

    # ---- Windows Security Log ----
    4624: 0.70,  # Successful Logon
    4634: 0.40,  # Logoff
    4648: 0.85,  # Logon w/ Explicit Credentials
    4672: 0.90,  # Special Privileges Assigned
    4688: 0.80,  # Process Creation (Security)
    4768: 0.80,  # Kerberos TGT
    4769: 0.80,  # Kerberos Service Ticket
    4776: 0.70,  # NTLM Auth
    4798: 0.75,  # User Group Membership Enumeration

    # ---- PowerShell ----
    4103: 0.75,  # PowerShell Module Logging
    4104: 0.95,  # PowerShell Script Block Logging

    # ---- Persistence / Services ----
    7045: 0.90,  # Service Installed

    # ---- Crypto / AD ----
    5058: 0.85,  # Key File Operations
    5061: 0.80,  # Cryptographic Operation
    1109: 0.90,  # Critical Directory Services Event (on DCs)

    # ---- Low Contextual Events ----
    7031: 0.40,  # Service Crashed
    7036: 0.30,  # Service Start/Stop
    1014: 0.30,  # DNS Failure
    1000: 0.25,  # App Crash
    1001: 0.25,  # Bugcheck
    1003: 0.25,  # App Hang
    600:  0.20,  # OS Startup/Shutdown
}


def _basename(path) -> str:
    """Extract lowercase filename from a Windows path."""
    if path is None: return ""
    try:
        s = str(path)
        if s in ("nan", "<NA>", "None", ""): return ""
        return s.replace("/", "\\").split("\\")[-1].lower().strip()
    except:
        return ""


def build_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign multi-resolution behavior tokens and context flags to each event row.

    Produces:
      token_coarse  : OFFICE | SCRIPT | BROWSER | LOLBIN | PROC
      token_medium  : OFFICE:WORD | SCRIPT:POWERSHELL | LOLBIN:RUNDLL32 | PROC:CMD.EXE
      token_fine    : token_medium + parent context + cmdline flags + integrity + signed
                      (used in context channel only, NOT in transition state space)

    Context flags:
      has_encoded           : -enc / -encodedcommand detected
      has_download_cradle   : invoke-webrequest / curl / wget etc.
      has_bypass            : -bypass / -noprofile etc.
      has_reflection        : reflection.assembly / load()
      is_lolbin             : known LOLBin binary

    Severity:
      severity_score  : float [0,1]
      severity_label  : critical / high / medium / low
    """
    out = df.copy()

    img   = out["image"].map(_basename)
    pimg  = out["parent_image"].map(_basename)
    cmd   = out["cmdline"].fillna("").astype(str)
    il    = out["integrity_level"].fillna("UNK").astype(str).str.upper()
    signed = out["signed"].fillna(False).astype(bool)

    # --- Coarse + Medium tokens ---
    coarse_list, medium_list = [], []
    for i in img:
        if i in OFFICE_MAP:
            coarse_list.append("OFFICE")
            medium_list.append(f"OFFICE:{OFFICE_MAP[i]}")
        elif i in SCRIPT_MAP:
            coarse_list.append("SCRIPT")
            medium_list.append(f"SCRIPT:{SCRIPT_MAP[i]}")
        elif i in BROWSER_MAP:
            coarse_list.append("BROWSER")
            medium_list.append(f"BROWSER:{BROWSER_MAP[i]}")
        elif i in LOLBINS:
            coarse_list.append("LOLBIN")
            medium_list.append(f"LOLBIN:{i.upper().replace('.EXE','')}")
        else:
            name = i.upper() if i else "UNKNOWN"
            coarse_list.append("PROC")
            medium_list.append(f"PROC:{name}")

    out["token_coarse"] = coarse_list
    out["token_medium"] = medium_list

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
    sig_flag     = signed.astype(int).astype(str)

    out["token_fine"] = (
        out["token_medium"]
        + "|PAR:" + parent
        + "|ENC:" + enc_flag
        + "|DL:"  + dl_flag
        + "|BP:"  + bypass_flag
        + "|IL:"  + il
        + "|SIG:" + sig_flag
    )

    # --- Severity grading (from v12_modular) ---
    out["severity_score"] = out["event_id"].apply(
        lambda x: EVENT_SEVERITY.get(int(x), 0.1) if pd.notna(x) else 0.1
    )
    out["severity_label"] = out["severity_score"].apply(_score_to_label)

    # --- MITRE ATT&CK mapping (event-level) ---
    out["mitre_technique"] = out.apply(_map_mitre_technique, axis=1)
    out["mitre_tactic"]    = out["mitre_technique"].map(
        lambda t: MITRE_TECHNIQUE_DB.get(t, {}).get("tactic", "") if t else ""
    )
    out["mitre_name"]      = out["mitre_technique"].map(
        lambda t: MITRE_TECHNIQUE_DB.get(t, {}).get("name", "") if t else ""
    )

    return out


def _score_to_label(score: float) -> str:
    if score >= 0.85: return "critical"
    if score >= 0.60: return "high"
    if score >= 0.35: return "medium"
    return "low"


# ---------------------------------------------------------------------------
# MITRE ATT&CK technique database (endpoint-focused, Sysmon-relevant)
# ---------------------------------------------------------------------------
# Reference: MITRE ATT&CK Enterprise v15 — https://attack.mitre.org/
#
# Each entry maps a technique ID to its name, tactic, and ATT&CK URL.
# The _map_mitre_technique function below uses event ID + behavioral
# context flags to select the most specific applicable technique.
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
    "T1059.007": {"name": "Command and Scripting Interpreter: JavaScript",
                  "tactic": "Execution",
                  "url": "https://attack.mitre.org/techniques/T1059/007/"},
    "T1106":     {"name": "Native API",
                  "tactic": "Execution",
                  "url": "https://attack.mitre.org/techniques/T1106/"},
    "T1204":     {"name": "User Execution",
                  "tactic": "Execution",
                  "url": "https://attack.mitre.org/techniques/T1204/"},

    # ---- Defense Evasion ----
    "T1218.010": {"name": "System Binary Proxy Execution: Regsvr32",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1218/010/"},
    "T1218.011": {"name": "System Binary Proxy Execution: Rundll32",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1218/011/"},
    "T1218.005": {"name": "System Binary Proxy Execution: Mshta",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1218/005/"},
    "T1218.003": {"name": "System Binary Proxy Execution: CMSTP",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1218/003/"},
    "T1047":     {"name": "Windows Management Instrumentation",
                  "tactic": "Execution",
                  "url": "https://attack.mitre.org/techniques/T1047/"},
    "T1140":     {"name": "Deobfuscate/Decode Files or Information",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1140/"},
    "T1027":     {"name": "Obfuscated Files or Information",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1027/"},
    "T1055":     {"name": "Process Injection",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1055/"},
    "T1574.002": {"name": "Hijack Execution Flow: DLL Side-Loading",
                  "tactic": "Defense Evasion",
                  "url": "https://attack.mitre.org/techniques/T1574/002/"},

    # ---- Credential Access ----
    "T1003":     {"name": "OS Credential Dumping",
                  "tactic": "Credential Access",
                  "url": "https://attack.mitre.org/techniques/T1003/"},
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

    # ---- Lateral Movement ----
    "T1021":     {"name": "Remote Services",
                  "tactic": "Lateral Movement",
                  "url": "https://attack.mitre.org/techniques/T1021/"},

    # ---- Command and Control ----
    "T1071":     {"name": "Application Layer Protocol",
                  "tactic": "Command and Control",
                  "url": "https://attack.mitre.org/techniques/T1071/"},

    # ---- Discovery ----
    "T1087":     {"name": "Account Discovery",
                  "tactic": "Discovery",
                  "url": "https://attack.mitre.org/techniques/T1087/"},

    # ---- Collection ----
    "T1005":     {"name": "Data from Local System",
                  "tactic": "Collection",
                  "url": "https://attack.mitre.org/techniques/T1005/"},

    # ---- Privilege Escalation ----
    "T1134":     {"name": "Access Token Manipulation",
                  "tactic": "Privilege Escalation",
                  "url": "https://attack.mitre.org/techniques/T1134/"},
}

# LOLBin -> technique mapping
_LOLBIN_TECHNIQUE = {
    "regsvr32.exe":  "T1218.010",
    "rundll32.exe":  "T1218.011",
    "mshta.exe":     "T1218.005",
    "cmstp.exe":     "T1218.003",
    "wmic.exe":      "T1047",
    "certutil.exe":  "T1140",
    "bitsadmin.exe": "T1105",
    "msiexec.exe":   "T1218.007",
}


def _map_mitre_technique(row: pd.Series) -> str | None:
    """
    Map a single event row to the most specific MITRE ATT&CK technique ID.

    Priority order (most specific wins):
      1. Encoded command + PowerShell → T1027 (Obfuscated Files)
      2. Known LOLBin → specific T1218.xxx sub-technique
      3. Script interpreter → T1059.xxx sub-technique
      4. Event ID semantic mapping (injection, LSASS, service install, etc.)
      5. None if no mapping applies
    """
    img = _basename(str(row.get("image", "")))
    eid = row.get("event_id")
    has_enc  = bool(row.get("has_encoded", False))
    has_dl   = bool(row.get("has_download_cradle", False))
    is_lol   = bool(row.get("is_lolbin", False))

    # 1. Encoded/obfuscated command execution
    if has_enc:
        return "T1027"

    # 2. LOLBin proxy execution
    if is_lol and img in _LOLBIN_TECHNIQUE:
        return _LOLBIN_TECHNIQUE[img]

    # 3. Script interpreters
    if img in ("powershell.exe", "pwsh.exe"):
        return "T1059.001"
    if img == "cmd.exe":
        return "T1059.003"
    if img in ("wscript.exe", "cscript.exe"):
        return "T1059.005"

    # 4. Event ID semantic mapping
    try:
        eid_int = int(eid)
    except (TypeError, ValueError):
        return None

    if eid_int == 8:     return "T1055"      # CreateRemoteThread → Process Injection
    if eid_int == 10:    return "T1003"      # ProcessAccess → Credential Dumping (LSASS)
    if eid_int == 7045:  return "T1543.003"  # Service Installed → Persistence
    if eid_int == 6:     return "T1574.002"  # Driver Loaded → DLL Side-Loading
    if eid_int == 4672:  return "T1134"      # Special Privileges → Token Manipulation
    if eid_int == 4768:  return "T1558.003"  # Kerberos TGT → Kerberoasting
    if eid_int == 4798:  return "T1087"      # Group Enum → Account Discovery
    if eid_int == 22:    return "T1071"      # DNS Query → C2 Protocol
    if eid_int in (12, 13):  return "T1547.001"  # Registry → Persistence
    if eid_int == 3 and has_dl: return "T1071"  # Network + download cradle → C2

    return None
