# =============================================================================
# make_readable_tables.py
# ------------------------
# Turn the three raw metric CSVs produced by the heatsink pipeline into
# human-readable, grouped, publication-ready tables:
#
#   simulated_data.py                -> OUTPUT\heatsink_metrics.csv   (SIM)
#   experimental_data.py             -> OUTPUT\exp_outer_metrics.csv   (EXP)
#   comparison.py                    -> OUTPUT\comparison_heatsink_outer\
#                                                comparison_minmax_vs_energy.csv (CMP)
#
# The raw CSVs use terse machine names (`surf_dT_min_peak_K`, `vol_cool_05_end`,
# `area_cool_2_per_J`) and the SIM table has 130 columns.  This script
#   * relabels every column to a human name WITH its unit,
#   * splits each source into logical, scannable sub-tables,
#   * writes Excel + Markdown + CSV + LaTeX (booktabs) copies,
# and NEVER modifies the input CSVs.
#
# House rules (kept identical to make_paper_tables.py):
#   * missing/NaN cells are blank in CSV, "--" in Excel/Markdown/LaTeX;
#   * point probes are reference-only and are NEVER compared with experiment
#     (they live in their own clearly-marked groups).
#
# Usage
#   python make_readable_tables.py                          # all formats
#   python make_readable_tables.py --format excel
#   python make_readable_tables.py --format markdown --out-dir D:\somewhere
#
# Output (default: OUTPUT\readable_tables\)
#   heatsink_readable_tables.xlsx        one sheet per (source, group)
#   readable_<source>_<group>.csv        tidy per-group CSVs
#   readable_<source>_<group>.md         Markdown preview per group
#   latex\readable_<source>_<group>.tex  booktabs snippets
#   index.md                             table of contents
#   coverage_report.txt                  which columns were labelled / grouped
# =============================================================================
import argparse
import os
import re
import sys
import time

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# --- reuse the LaTeX escaper + NA placeholder from the existing tool ---------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from make_paper_tables import _tex, NA_REP  # noqa: E402  (house-style helpers)

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE = _HERE
OUT_ROOT = os.path.join(BASE, "OUTPUT")
TABLES_DIR = os.path.join(OUT_ROOT, "readable_tables")
EXCEL_NAME = "heatsink_readable_tables.xlsx"


# =============================================================================
# COLUMN RELABELLING  (machine name -> "Human name (unit)")
# =============================================================================
PROBE_THERMAL = [
    "dT_cold_peak_K", "dT_hot_peak_K", "dT_cold_end_K", "dT_hot_end_K",
    "dT_cold_60_K", "dT_hot_60_K", "dT_max_peak_K", "dT_min_peak_K",
    "dT_max_end_K", "dT_min_end_K", "dT_max_60_K", "dT_min_60_K",
    "minmax_source", "ratio_cool_heat", "cool_per_A_KpA", "heat_per_A_KpA",
]
TEMPORAL = [
    "t_peak_cool_s", "t_cool63_s", "t_cool90_s", "t_ambient_cross_s",
    "t_back_normal_s", "overshoot_cold_K", "overshoot_time_s", "overshoot_hot_K",
]
ENERGY = [
    "V1_peak_V", "q_peak_Wm2", "W_full_J", "W_drive_J", "E_areal_J_m2",
    "dose_Ks", "K_per_J", "K_per_J_hot", "J_per_K",
]

RELABEL_SIM = {
    "I0_A": "I0 (A)",
    "dT_cold_peak_K": "ΔT cold peak (K)", "dT_hot_peak_K": "ΔT hot peak (K)",
    "dT_cold_end_K": "ΔT cold end (K)", "dT_hot_end_K": "ΔT hot end (K)",
    "dT_cold_60_K": "ΔT cold @60 s (K)", "dT_hot_60_K": "ΔT hot @60 s (K)",
    "dT_max_peak_K": "ΔT max peak (K)", "dT_min_peak_K": "ΔT min peak (K)",
    "dT_max_end_K": "ΔT max end (K)", "dT_min_end_K": "ΔT min end (K)",
    "dT_max_60_K": "ΔT max @60 s (K)", "dT_min_60_K": "ΔT min @60 s (K)",
    "minmax_source": "min/max source", "ratio_cool_heat": "cool/heat ratio (−)",
    "cool_per_A_KpA": "cool per A (K/A)", "heat_per_A_KpA": "heat per A (K/A)",
    "t_peak_cool_s": "t peak cool (s)", "t_cool63_s": "t cool63 (s)",
    "t_cool90_s": "t cool90 (s)", "t_ambient_cross_s": "t ambient cross (s)",
    "t_back_normal_s": "t back to normal (s)",
    "overshoot_cold_K": "overshoot cold (K)", "overshoot_time_s": "overshoot time (s)",
    "overshoot_hot_K": "overshoot hot (K)",
    "V1_peak_V": "V1 peak (V)", "q_peak_Wm2": "q peak (W/m²)",
    "W_full_J": "W full (J)", "W_drive_J": "W drive (J)",
    "E_areal_J_m2": "E areal (J/m²)", "dose_Ks": "dose (K·s)",
    "K_per_J": "K per J", "K_per_J_hot": "K per J (hot)", "J_per_K": "J per K",
    "surf_source": "source", "surf_n_nodes": "n nodes", "surf_n_times": "n times",
    "surf_dT_min_peak_K": "ΔT min peak (K)", "surf_dT_max_peak_K": "ΔT max peak (K)",
    "surf_dT_min_end_K": "ΔT min end (K)", "surf_dT_max_end_K": "ΔT max end (K)",
    "surf_dT_min_60_K": "ΔT min @60 s (K)", "surf_dT_max_60_K": "ΔT max @60 s (K)",
    "surf_dT_mean_end_K": "ΔT mean end (K)", "surf_dT_mean_60_K": "ΔT mean @60 s (K)",
    "surf_spread_peak_K": "spread peak (K)", "surf_spread_end_K": "spread end (K)",
    "surf_std_peak_K": "std peak (K)", "surf_std_end_K": "std end (K)",
    "surf_t_peak_cool_s": "t peak cool (s)", "surf_t_cool63_s": "t cool63 (s)",
    "surf_t_cool90_s": "t cool90 (s)", "surf_t_ambient_cross_s": "t ambient cross (s)",
    "surf_t_back_normal_s": "t back to normal (s)",
    "surf_t_recover63_s": "t recover63 (s)", "surf_t_recover90_s": "t recover90 (s)",
    "surf_recovery_frac_end": "recovery frac end (−)",
    "surf_recovered_by_final": "recovered by final (−)",
    "surf_overshoot_cold_K": "overshoot cold (K)",
    "surf_ratio_cool_heat": "cool/heat ratio (−)",
    "surf_cool_per_A_KpA": "cool per A (K/A)", "surf_heat_per_A_KpA": "heat per A (K/A)",
    "surf_K_per_J": "K per J", "surf_K_per_J_drive": "K per J (drive)",
    "surf_J_per_K": "J per K", "surf_dose_Ks": "dose (K·s)",
}

RELABEL_EXP = {
    "I0 (A)": "I0 (A)", "n_runs": "n runs",
    "dT_cold_peak (C)": "ΔT cold peak (K)", "dT_hot_peak (C)": "ΔT hot peak (K)",
    "t_peak_cool (s)": "t peak cool (s)", "tau_cold (s)": "τ cold (s)",
    "cooling_rate (C/s)": "cooling rate (K/s)",
    "overshoot_cold (C)": "overshoot cold (K)",
    "E_input (J)": "E input (J)", "K_per_J": "K per J",
    "dT_cold_end (C)": "ΔT cold end (K)", "dT_hot_end (C)": "ΔT hot end (K)",
    "E_input_sd (J)": "E input sd (J)", "ratio_cool_heat": "cool/heat ratio (−)",
}

THR = {"05": "0.5", "1": "1", "2": "2", "5": "5"}


def _relabel_volume_area(col):
    """vol_cool_05_end / area_heat_2_per_J -> human label with unit."""
    m = re.match(r"^(vol|area)_(cool|heat)_(05|1|2|5)_(end|60|peak|per_J)$", col)
    if m:
        kind, side, tag, when = m.groups()
        unit = "mm³" if kind == "vol" else "mm²"
        side_t = "cool" if side == "cool" else "heat"
        if when == "per_J":
            return f"{kind} {side_t} @{THR[tag]}K per J ({unit}/J)"
        when_t = {"end": "end", "60": "@60 s", "peak": "peak"}[when]
        return f"{kind} {side_t} @{THR[tag]}K {when_t} ({unit})"
    m = re.match(r"^(vol|area)_ratio_(05|1|2|5)_end$", col)
    if m:
        return f"{m.group(1)} ratio @{THR[m.group(2)]}K end (−)"
    return None


def relabel(source, col):
    """Best human label for `col` given the source; falls back to a tidy name."""
    if source == "SIM" and col in RELABEL_SIM:
        return RELABEL_SIM[col]
    if source == "EXP" and col in RELABEL_EXP:
        return RELABEL_EXP[col]
    lbl = _relabel_volume_area(col)
    if lbl:
        return lbl
    return re.sub(r"_+", " ", col).strip()      # tidy fallback

INPUTS = {
    "SIM": os.path.join(OUT_ROOT, "heatsink_metrics.csv"),
    "EXP": os.path.join(OUT_ROOT, "exp_outer_metrics.csv"),
    "CMP": os.path.join(OUT_ROOT, "comparison_heatsink_outer",
                        "comparison_minmax_vs_energy.csv"),
}

HOUSE_RULE = ("SIM = COMSOL heatsink; EXP = outer-surface IR. Point probes are "
              "reference-only and are NEVER compared with experimental data.")

# =============================================================================
# GROUPING  (which columns belong to which scannable sub-table)
# =============================================================================
def sim_groups(cols):
    g = {}
    g["identity"] = [c for c in ["I0_A"] if c in cols]
    g["probe_thermal"] = [c for c in PROBE_THERMAL if c in cols]
    g["temporal"] = [c for c in TEMPORAL if c in cols]
    g["energy"] = [c for c in ENERGY if c in cols]
    g["volume"] = [c for c in cols if c.startswith("vol_")]
    g["surface_field"] = [c for c in cols if c.startswith("surf_")]
    g["area"] = [c for c in cols if c.startswith("area_")]
    assigned = set(sum(g.values(), []))
    g["other"] = [c for c in cols if c not in assigned]
    return g


def exp_groups(cols):
    g = {}
    g["identity"] = [c for c in ["I0 (A)", "n_runs"] if c in cols]
    g["thermal"] = [c for c in cols if c.startswith("dT_")
                    or c in ("ratio_cool_heat", "overshoot_cold (C)")]
    g["temporal"] = [c for c in cols if c.startswith("t_") or c.startswith("tau_")
                     or c.startswith("cooling_rate")]
    g["energy"] = [c for c in cols if c.startswith("E_input") or c == "K_per_J"]
    g["area"] = [c for c in cols if c.startswith("area_")]
    assigned = set(sum(g.values(), []))
    g["other"] = [c for c in cols if c not in assigned]
    return g


def _ordered_metric_cols(cols, bases):
    """Order one base metric's columns as: exp mean, exp sd, sim, err, relerr."""
    ordered = []
    for base in bases:
        # experimental mean (_exp with optional unit, not _exp_sd, not _sim)
        for c in cols:
            if (base in c and "_exp" in c and "_exp_sd" not in c
                    and "_sim" not in c and "sim_probe" not in c):
                ordered.append(c)
        # experimental sd (_exp_sd)
        for c in cols:
            if base in c and "_exp_sd" in c and "sim_probe" not in c:
                ordered.append(c)
        # simulated (_sim, not sim_probe)
        for c in cols:
            if base in c and "_sim" in c and "sim_probe" not in c:
                ordered.append(c)
        # error
        for c in cols:
            if c == "err_" + base or c.startswith("err_" + base + " "):
                ordered.append(c)
        # relative error
        for c in cols:
            if c == "relerr_" + base + " (%)" or c.startswith("relerr_" + base + " (%)"):
                ordered.append(c)
    return ordered


def cmp_groups(cols):
    g = {}
    g["identity"] = [c for c in ["I0 (A)"] if c in cols]
    g["energy"] = [c for c in ["E_sim_drive (J)", "E_exp (J)", "E_sim_full (J)",
                               "V1_peak_sim (V)"] if c in cols]
    g["thermal_peak"] = [c for c in ["dTmin_peak_sim (C)", "dTmin_peak_exp (C)",
                                      "dTmax_peak_sim (C)", "dTmax_peak_exp (C)"]
                         if c in cols]
    g["thermal_end"] = [c for c in ["dTmin_end_sim (C)", "dTmin_end_exp (C)",
                                     "dTmax_end_sim (C)", "dTmax_end_exp (C)"]
                        if c in cols]
    g["temporal"] = [c for c in ["t_peak_cool_sim (s)", "t_peak_cool_exp (s)",
                                  "tau_cold_exp (s)", "t_cool63_sim (s)",
                                  "t_back_normal_sim (s)", "t_recover63_sim (s)",
                                  "t_recover90_sim (s)", "recovered_at_60s_sim (%)"]
                     if c in cols]
    g["overshoot"] = [c for c in ["overshoot_sim (C)", "overshoot_exp (C)"] if c in cols]
    g["uniformity"] = [c for c in ["spread_peak_sim (C)", "spread_end_sim (C)",
                                   "std_end_sim (C)"] if c in cols]
    # efficiency: SIM-only dose first, then per-metric exp->exp_sd->sim->err->relerr
    g["efficiency"] = ([c for c in ["dose_sim (Ks)"] if c in cols]
                       + _ordered_metric_cols(cols, ["K_per_J",
                                                     "area_cool_2_per_J",
                                                     "area_heat_2_per_J"]))
    # thermal load: per-metric exp->exp_sd->sim->err->relerr
    g["thermal_load"] = _ordered_metric_cols(cols, ["load_cool_peak", "load_cool_end",
                                                    "load_heat_peak", "load_heat_end"])
    # equivalent radius @2K: per-metric exp->exp_sd->sim->err->relerr
    g["radius"] = _ordered_metric_cols(cols, ["r_cool_2_peak", "r_heat_2_peak"])
    # area timing @2K: per-metric exp->exp_sd->sim->err->relerr
    g["area_timing"] = _ordered_metric_cols(cols, ["t_cool_2_50", "t_cool_2_100",
                                                   "t_heat_2_50", "t_heat_2_100"])
    g["probe_reference"] = [c for c in cols if "sim_probe" in c]
    # errors: only those NOT already placed in a specific group
    specific_assigned = set()
    for key in ("efficiency", "thermal_load", "radius", "area_timing"):
        specific_assigned.update(g[key])
    g["errors"] = [c for c in cols
                   if (c.startswith("err_") or c.startswith("relerr_"))
                   and c not in specific_assigned]
    assigned = set(sum(g.values(), []))
    g["other"] = [c for c in cols if c not in assigned]
    return g

GROUP_FUNCS = {"SIM": sim_groups, "EXP": exp_groups, "CMP": cmp_groups}

GROUP_TITLES = {
    ("SIM", "identity"): "Identity",
    ("SIM", "probe_thermal"): "Point-probe thermal (reference only)",
    ("SIM", "temporal"): "Temporal response (probe)",
    ("SIM", "energy"): "Electrical & energy",
    ("SIM", "volume"): "Cooled/heated volume",
    ("SIM", "surface_field"): "Outer-surface field (comparable)",
    ("SIM", "area"): "Cooled/heated surface area",
    ("SIM", "other"): "Other SIM",
    ("EXP", "identity"): "Identity",
    ("EXP", "thermal"): "Thermal",
    ("EXP", "temporal"): "Temporal",
    ("EXP", "energy"): "Energy",
    ("EXP", "area"): "Cooled/heated surface area",
    ("EXP", "other"): "Other EXP",
    ("CMP", "identity"): "Identity",
    ("CMP", "energy"): "Input energy",
    ("CMP", "thermal_peak"): "Thermal peak (sim vs exp)",
    ("CMP", "thermal_end"): "Thermal end (sim vs exp)",
    ("CMP", "temporal"): "Temporal (sim vs exp)",
    ("CMP", "overshoot"): "Overshoot (sim vs exp)",
    ("CMP", "uniformity"): "Uniformity (sim only)",
    ("CMP", "efficiency"): "Efficiency",
    ("CMP", "thermal_load"): "Integrated thermal load (sim vs exp)",
    ("CMP", "radius"): "Equivalent radius @2K (sim vs exp)",
    ("CMP", "area_timing"): "Area timing @2K (sim vs exp)",
    ("CMP", "probe_reference"): "Point-probe reference (NEVER compared)",
    ("CMP", "errors"): "Errors (sim − exp)",
    ("CMP", "other"): "Other comparison",
}

CAPTIONS = {
    "SIM": "COMSOL heatsink simulation metrics (simulated_data.py).",
    "EXP": "Experimental outer-surface IR metrics "
           "(experimental_data.py).",
    "CMP": "Heatsink simulation vs experimental outer surface "
           "(comparison.py).",
}


# =============================================================================
# ROUNDING  (decimals per column, so the printed tables stay readable)
# =============================================================================
def decimals_for(col):
    c = col.lower()
    if any(k in c for k in ("n_nodes", "n_times")) or c in ("surf_source",
                                                            "minmax_source"):
        return 0
    if "per_j" in c or "k_per_j" in c:        # mm³/J, mm²/J, K/J
        return 3
    if "dose" in c:
        return 1
    if c.endswith("_v") or "(v)" in c:
        return 3
    if "frac" in c or "recover" in c or "recovered" in c or "(%)" in c:
        return 2
    if "ratio" in c or "per_a" in c or "kpa" in c:
        return 2
    if "q_peak" in c:
        return 2
    if c.endswith("_k") or "(k)" in c:
        return 2
    if c.endswith("_s") or "(s)" in c:
        return 2
    if "mm" in c or "(j" in c or c.endswith("_j") or "e_areal" in c:
        return 2
    return 3


def _fmt_num(v, dp):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    s = f"{f:.{dp}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s not in ("", "-") else "0"


def rounded(df):
    """Round numeric columns to their per-column precision (numbers stay numeric)."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype(float).round(decimals_for(col))
    return out


def display(df):
    """String frame for Markdown/LaTeX/Excel widths: rounded, NA -> '--'."""
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            dp = decimals_for(col)
            out[col] = [(_fmt_num(v, dp) or NA_REP) for v in df[col]]
        else:
            vals = []
            for v in df[col]:
                if v is None or (not isinstance(v, str) and pd.isna(v)):
                    vals.append(NA_REP)
                else:
                    vals.append(str(v))
            out[col] = vals
    return out


# =============================================================================
# WRITERS
# =============================================================================
def write_csv(df, path):
    df.to_csv(path, index=False)                  # blank keeps cells numeric


def write_excel(tables, path):
    """One sheet per (source, group); numbers stay numeric, NA shown as NA_REP."""
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            sheet = re.sub(r"[\[\]:*?/\\]", "_", name)[:31]
            df.to_excel(writer, sheet_name=sheet, index=False, na_rep=NA_REP)
            ws = writer.sheets[sheet]
            ws.freeze_panes = "A2"
            frame = display(df)
            for i, col in enumerate(frame.columns, start=1):
                width = max([len(str(col))] + [len(str(v)) for v in frame[col]]) + 2
                ws.column_dimensions[get_column_letter(i)].width = min(width, 34)


def write_markdown(df, path, title, caption):
    body = display(df).to_markdown(index=False)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n{caption}\n\n{body}\n")


def df_to_latex(df, caption, label, key_cols):
    keys = [c for c in key_cols if c in df.columns]
    colspec = "l" * len(keys) + "r" * (len(df.columns) - len(keys))
    frame = display(df)
    lab = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
    lines = [r"\begin{table}[htbp]", r"  \centering",
             f"  \\caption{{{_tex(caption)}}}",
             f"  \\label{{tab:{lab}}}",
             f"  \\begin{{tabular}}{{{colspec}}}", r"    \toprule",
             "    " + " & ".join(_tex(c) for c in frame.columns) + r" \\",
             r"    \midrule"]
    for _, row in frame.iterrows():
        lines.append("    " + " & ".join(_tex(v) for v in row) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# =============================================================================
# BUILD
# =============================================================================
def build_all():
    """Read the three CSVs and return {key: {source, group, df, title, cols}}."""
    tables = {}
    id_cols = {"SIM": ["I0_A"], "EXP": ["I0 (A)", "n_runs"], "CMP": ["I0 (A)"]}
    for source, path in INPUTS.items():
        if not os.path.exists(path):
            print(f"[warn] missing input for {source}: {path} (skipped)")
            continue
        raw = pd.read_csv(path)
        raw.columns = [str(c).strip() for c in raw.columns]
        idcols = [c for c in id_cols[source] if c in raw.columns]
        for group, cols in GROUP_FUNCS[source](list(raw.columns)).items():
            if not cols:
                continue
            # prepend the identity key so every sub-table is self-describing
            ordered = idcols + [c for c in cols if c not in idcols]
            sub = raw[ordered].copy()
            sub = sub.rename(columns={c: relabel(source, c) for c in ordered})
            title = GROUP_TITLES.get((source, group),
                                     group.replace("_", " ").title())
            tables[f"{source}_{group}"] = {
                "source": source, "group": group, "df": sub, "title": title,
                "cols": ordered}
    return tables


def coverage_report(tables):
    out = ["Heatsink pipeline -> readable tables coverage report", "=" * 62]
    for source, path in INPUTS.items():
        out.append(f"{source:<4} {os.path.relpath(path, BASE)}  "
                   f"{'OK' if os.path.exists(path) else 'MISSING'}")
    out.append("")
    out.append("grouped tables written:")
    for key, t in tables.items():
        rel = [relabel(t["source"], c) for c in t["cols"]]
        out.append(f"  {key:<22} {t['title']}  ({len(rel)} cols)")
        if t["group"] == "other" and rel:
            out.append(f"      NOTE: ungrouped columns: {', '.join(rel)}")
    out.append("")
    out.append(f"house rule: {HOUSE_RULE}")
    return "\n".join(out) + "\n"


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Readable grouped tables for the heatsink "
                    "SIM/EXP/comparison CSVs.")
    p.add_argument("--format", default="all",
                   choices=["all", "csv", "excel", "markdown", "latex"],
                   help="output format(s); 'all' writes every one (default)")
    p.add_argument("--out-dir", default=TABLES_DIR, help="folder for the tables")
    return p.parse_args(argv)


SRC_SHORT = {"SIM": "sim", "EXP": "exp", "CMP": "cmp"}


def main(argv=None):
    args = parse_args(argv)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    tables = build_all()
    if not tables:
        print("No input CSVs found - run the three pipeline scripts first.")
        return 1

    formats = (["csv", "excel", "markdown", "latex"] if args.format == "all"
               else [args.format])
    written = []
    order = {"SIM": 0, "EXP": 1, "CMP": 2}
    keys = sorted(tables, key=lambda k: (order[tables[k]["source"]],
                                         k.split("_", 1)[1]))

    for k in keys:
        t = tables[k]
        df = rounded(t["df"])
        stem = f"readable_{SRC_SHORT[t['source']]}_{t['group']}"
        caption = f"{CAPTIONS[t['source']]} {t['title']}. {HOUSE_RULE}"
        key_cols = list(t["df"].columns[:1])      # first col = identity key

        if "csv" in formats:
            path = os.path.join(out_dir, stem + ".csv")
            write_csv(df, path)
            written.append(path)
        if "markdown" in formats:
            path = os.path.join(out_dir, stem + ".md")
            write_markdown(df, path, f"{t['source']} — {t['title']}", caption)
            written.append(path)
        if "latex" in formats:
            tex_dir = os.path.join(out_dir, "latex")
            os.makedirs(tex_dir, exist_ok=True)
            path = os.path.join(tex_dir, stem + ".tex")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(df_to_latex(df, caption, stem, key_cols))
            written.append(path)

    if "excel" in formats:
        sheets = {k: rounded(tables[k]["df"]) for k in keys}
        path = os.path.join(out_dir, EXCEL_NAME)
        write_excel(sheets, path)
        written.append(path)

    idx_path = os.path.join(out_dir, "index.md")
    with open(idx_path, "w", encoding="utf-8") as fh:
        fh.write("# Heatsink readable tables\n\n")
        fh.write(f"Generated by `make_readable_tables.py` in "
                 f"{time.time() - t0:.1f} s.\n\n")
        fh.write(HOUSE_RULE + "\n\n## Inputs\n\n")
        for source, path in INPUTS.items():
            fh.write(f"- **{source}**: `{os.path.relpath(path, BASE)}`\n")
        fh.write("\n## Tables\n\n")
        for k in keys:
            t = tables[k]
            stem = f"readable_{SRC_SHORT[t['source']]}_{t['group']}"
            fh.write(f"- `{stem}` — {t['title']} ({len(t['cols'])} cols)\n")
    written.append(idx_path)

    report_path = os.path.join(out_dir, "coverage_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(coverage_report(tables))
    written.append(report_path)

    print(f"Read {len(keys)} grouped tables from "
          f"{sum(1 for p in INPUTS.values() if os.path.exists(p))} input CSVs.")
    print(f"\nFiles written to {out_dir}:")
    for path in written:
        print(f"  {os.path.relpath(path, BASE)}")
    print("\n" + coverage_report(tables))
    return 0


if __name__ == "__main__":
    sys.exit(main())

