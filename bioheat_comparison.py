# =============================================================================
# bioheat_comparison.py
# =============================================================================
# Regular phantom vs. bioheat COMSOL runs, analysed with the same metrics,
# tables and figure style as csv_sensitivity_analysis.py.
#
# INPUT: four CSV files in the Bioheat\ folder
#     regular phantom : <probes file> + <surfaces file>
#     bioheat module  : <probes file> + <surfaces file>
#   Every file holds the drive-current sweep I0 = 0.1 ... 0.5 A.
#   Files are assigned to a run by name: any CSV whose name contains "bioheat"
#   is a bioheat file, every other CSV in the folder is a regular file. The two
#   files of a run (probes + surfaces) are merged on (I0, time); columns that
#   exist in only one file (e.g. Max_surface / Min_surface) are interpolated
#   onto the time grid of the probes file. To use explicit file names, fill in
#   REGULAR_FILES / BIOHEAT_FILES below.
#
# For every current the script computes the same metrics as the sensitivity
# analysis for both runs and reports the change of bioheat vs. regular
# (absolute and %). "Baseline" = regular phantom.
#
# Output (OUTPUT/bioheat_comparison/):
#   TABLES
#     metrics_regular.csv / metrics_bioheat.csv   metrics per current
#     bioheat_comparison.csv        wide: regular / bioheat / abs / % per metric
#     bioheat_comparison_long.csv   one row per (I0, metric) - easy to filter
#     changes_vs_regular.csv        same layout as changes_vs_baseline.csv
#     bioheat_summary.csv           per-metric mean / max |change|, worst I0
#     bioheat_transient_summary.csv legend/table values of the transient plots
#   FIGURES
#     transient_<q>.png             regular (solid) vs bioheat (dashed), all I0,
#                                   legend + value table on the right
#     transient_diff_<q>.png        bioheat - regular vs. time, + table
#     dashboard_I0<x>.png           per-current 2x2 overview
#     heatmap_pct.png               % change, every metric x every current
#     tornado_by_current.png        % change per metric grouped by current
#     tornado_max_pct.png           max |% change| per metric
#     grouped_bars.png, metric_vs_current.png, pct_vs_current.png,
#     abs_vs_current.png            headline metrics
#     cooling_vs_power.png, response_times.png, efficiency_vs_current.png,
#     thermal_footprint.png         surface extremes + cooled/heated volumes
#
# Run:   python bioheat_comparison.py
# =============================================================================

import os
import glob
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sensitivity_analysis import (
    load_csv, compute_metrics, find_col,
    METRIC_LABELS, DEFAULT_METRICS, TRANSIENT_QUANTITIES, TEMP_METRICS,
    IEEE_RC, CBLUE, CORANGE, CGREEN, CRED, CPURPLE, CBROWN,
    new_fig, save_fig, is_cold_metric, _metric_signal, _extreme_value,
)

plt.rcParams.update(IEEE_RC)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
BIOHEAT_DIR = os.path.join(HERE, "Bioheat")
OUTPUT_ROOT = os.path.join(HERE, "OUTPUT", "bioheat_comparison")

# Leave as None for automatic detection, or list the two files of each run, e.g.
#   REGULAR_FILES = ["Config_heatsink_probes.csv", "Config_heatsink_surface.csv"]
#   BIOHEAT_FILES = ["Config_heatsink_bioheat_probes.csv", "Config_heatsink_bioheat_surface.csv"]
REGULAR_FILES = None
BIOHEAT_FILES = None

T_START = 5.0                      # pulse onset (s), as in the sensitivity analysis
METRICS = DEFAULT_METRICS + ["vol_cool2_peak", "vol_heat2_peak",
                             "dT_surf_min", "dT_surf_max"]
HEADLINE = ["dT_surf_min", "dT_surf_max", "W_full_J",
            "vol_cool2_peak", "vol_heat2_peak", "net_power_W"]

# Label overrides for the headline metrics of this script (surface extremes
# taken relative to 20 °C, 2 °C volumes taken at their peak over time instead
# of the 5 °C end value).
_HEADLINE_LABELS = {
    "dT_surf_min":    r"$\Delta T_{\text{surf,min}}$ ($^\circ$C)",
    "dT_surf_max":    r"$\Delta T_{\text{surf,max}}$ ($^\circ$C)",
    "vol_cool2_peak": r"$V_{\text{cool,2C,peak}}$ (mm$^3$)",
    "vol_heat2_peak": r"$V_{\text{heat,2C,peak}}$ (mm$^3$)",
}

REG_COLOR, BIO_COLOR = CBLUE, CORANGE
CUR_PALETTE = [CBLUE, CGREEN, CORANGE, CRED, CPURPLE, CBROWN]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def fmt(x, spec=None):
    if x is None or not np.isfinite(x):
        return "n/a"
    if spec:
        return format(x, spec)
    return f"{x:.3f}" if abs(x) < 1000 else f"{x:.1f}"


def current_colors(currents):
    return {c: CUR_PALETTE[i % len(CUR_PALETTE)] for i, c in enumerate(currents)}


def label(m):
    return _HEADLINE_LABELS.get(m, METRIC_LABELS.get(m, m))


def file_role(path):
    n = os.path.basename(path).lower()
    if "surf" in n:
        return "surfaces"
    if "probe" in n:
        return "probes"
    return "data"


# ---------------------------------------------------------------------------
# File discovery, loading and merging (probes + surfaces -> one table per run)
# ---------------------------------------------------------------------------
def discover_files():
    def resolve(names):
        return [n if os.path.isabs(n) else os.path.join(BIOHEAT_DIR, n) for n in names]

    if REGULAR_FILES and BIOHEAT_FILES:
        return resolve(REGULAR_FILES), resolve(BIOHEAT_FILES)

    reg, bio = [], []
    for p in sorted(glob.glob(os.path.join(BIOHEAT_DIR, "*.csv"))):
        (bio if "bioheat" in os.path.basename(p).lower() else reg).append(p)
    return reg, bio


def merge_run(paths, run_label):
    """Load every file of one run and merge them into a single table."""
    parts = []
    for p in paths:
        df, *_ = load_csv(p)
        parts.append((p, df))
        print(f"  [{run_label}] {os.path.basename(p):45s} ({file_role(p):8s}) "
              f"{len(df)} rows, {len(df.columns)} columns")

    def score(df):  # the file with the probe signals is the time base
        return sum(find_col(df.columns, k) is not None
                   for k in ("Hot", "Cold", "Terminal voltage", "Terminal current"))

    parts.sort(key=lambda x: -score(x[1]))
    base = parts[0][1].copy()
    i0b, tb = find_col(base.columns, "I0"), find_col(base.columns, "Time (s)")
    base["_key"] = base[i0b].round(9)

    for p, df in parts[1:]:
        i0c, tc = find_col(df.columns, "I0"), find_col(df.columns, "Time (s)")
        have = {c.lower() for c in base.columns}
        new_cols = [c for c in df.columns if c not in (i0c, tc) and c.lower() not in have]
        if not new_cols:
            continue
        df = df.copy()
        df["_key"] = df[i0c].round(9)
        for c in new_cols:
            base[c] = np.nan
        for key, g in base.groupby("_key"):
            src = df[df["_key"] == key].sort_values(tc).drop_duplicates(tc)
            if src.empty:
                continue
            for c in new_cols:
                vals = pd.to_numeric(src[c], errors="coerce").to_numpy(float)
                base.loc[g.index, c] = np.interp(g[tb].to_numpy(float),
                                                 src[tc].to_numpy(float), vals)
        print(f"  [{run_label}] merged from {os.path.basename(p)}: {new_cols}")

    return base.drop(columns="_key")


def compute_run_metrics(df, run_label):
    mdf = compute_metrics(df)
    mdf["I0 (A)"] = mdf["I0 (A)"].round(9)
    mdf["run"] = run_label
    # Peak over time of the 2 °C cooled/heated volume (replaces the 5 °C
    # end values in the headline grouped bars).
    i0c = find_col(df.columns, "I0")
    for m, col in (("vol_cool2_peak", "vol_cool2C"), ("vol_heat2_peak", "vol_heat2C")):
        c = find_col(df.columns, col)
        if c is None:
            mdf[m] = np.nan
            continue
        peaks = []
        for i0, g in df.groupby(i0c):
            v = pd.to_numeric(g[c], errors="coerce").dropna().to_numpy(float)
            peaks.append((round(float(i0), 9), float(np.nanmax(v)) if v.size else np.nan))
        mdf = mdf.merge(pd.DataFrame(peaks, columns=["I0 (A)", m]), on="I0 (A)", how="left")
    # Surface extremes relative to 20 °C (replaces absolute surf_min_K /
    # surf_max_K in the headline grouped bars).
    for m, base in (("dT_surf_min", "surf_min_K"), ("dT_surf_max", "surf_max_K")):
        if base in mdf.columns:
            mdf[m] = mdf[base] - 20.0
        else:
            mdf[m] = np.nan
    return mdf


def extract_series(df, t_start=T_START):
    """{I0: (t, {hot, cold, surf_max, surf_min, volt, curr, vol_cool2, vol_heat2})}.
    Time is counted from pulse onset; temperatures are shifted to start at 0 K."""
    i0c, tc = find_col(df.columns, "I0"), find_col(df.columns, "Time (s)")
    cols = {
        "hot": find_col(df.columns, "Hot"),
        "cold": find_col(df.columns, "Cold"),
        "surf_max": find_col(df.columns, "Max_surface"),
        "surf_min": find_col(df.columns, "Min_surface"),
        "volt": find_col(df.columns, "Terminal voltage"),
        "curr": find_col(df.columns, "Terminal current"),
        "vol_cool2": find_col(df.columns, "vol_cool2C"),
        "vol_heat2": find_col(df.columns, "vol_heat2C"),
    }
    out = {}
    for i0, g in df.groupby(i0c):
        g = g.sort_values(tc)
        t_raw = g[tc].to_numpy(float)
        mask = t_raw >= t_start
        t0 = t_start
        if mask.sum() < 2:
            mask, t0 = np.ones_like(t_raw, dtype=bool), t_raw[0]
        y = {}
        for k, c in cols.items():
            if c is None:
                y[k] = None
                continue
            arr = g[c].to_numpy(float)[mask]
            if k in TEMP_METRICS and arr.size:
                fin = arr[np.isfinite(arr)]
                arr = arr - fin[0] if fin.size else arr
            y[k] = arr
        out[round(float(i0), 9)] = (t_raw[mask] - t0, y)
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def build_comparison(reg_mdf, bio_mdf, metrics=METRICS):
    """Wide table: regular / bioheat / abs / % per metric and current."""
    reg, bio = reg_mdf.set_index("I0 (A)"), bio_mdf.set_index("I0 (A)")
    currents = sorted(set(reg.index) & set(bio.index))
    rows = []
    for i0 in currents:
        r, b = reg.loc[i0], bio.loc[i0]
        row = {"I0 (A)": float(i0)}
        for m in metrics:
            rv, bv = r.get(m, np.nan), b.get(m, np.nan)
            ok = np.isfinite(rv) and np.isfinite(bv)
            row[m + "_regular"], row[m + "_bioheat"] = rv, bv
            row[m + "_abs"] = bv - rv if ok else np.nan
            row[m + "_pct"] = (bv - rv) / rv * 100.0 if (ok and abs(rv) > 1e-12) else np.nan
        rows.append(row)
    return pd.DataFrame(rows), currents


def build_long(comp, metrics, currents):
    rows = []
    for i0 in currents:
        r = comp[comp["I0 (A)"] == i0].iloc[0]
        for m in metrics:
            rows.append({"I0 (A)": i0, "metric": m, "metric_label": label(m),
                         "regular": r[m + "_regular"], "bioheat": r[m + "_bioheat"],
                         "abs_change": r[m + "_abs"], "pct_change": r[m + "_pct"]})
    return pd.DataFrame(rows)


def build_changes_vs_regular(comp, metrics):
    """Same layout as changes_vs_baseline.csv of the sensitivity analysis."""
    rows = []
    for _, r in comp.iterrows():
        row = {"I0 (A)": r["I0 (A)"]}
        for m in metrics:
            row[m] = r[m + "_bioheat"]
            row[m + "_base"] = r[m + "_regular"]
            row[m + "_pct"] = r[m + "_pct"]
            row[m + "_abs"] = r[m + "_abs"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary(comp, metrics):
    rows = []
    for m in metrics:
        pct, ab = comp[m + "_pct"], comp[m + "_abs"]
        if ab.dropna().empty:
            continue
        rec = {"metric": m, "metric_label": label(m),
               "mean_abs_change": float(ab.abs().mean()),
               "max_abs_change": float(ab.abs().max())}
        if not pct.dropna().empty:
            rec.update(mean_abs_pct=float(pct.abs().mean()),
                       max_abs_pct=float(pct.abs().max()),
                       mean_pct=float(pct.mean()),
                       worst_I0=float(comp.loc[pct.abs().idxmax(), "I0 (A)"]))
        if is_cold_metric(m):
            rec["most_neg_change"] = float(ab.min())
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.sort_values("max_abs_pct", ascending=False) if "max_abs_pct" in out else out


def transient_records(reg_s, bio_s):
    """Legend/table values of the transient plots (extreme value per quantity & current)."""
    rows = []
    for q, qlabel in TRANSIENT_QUANTITIES:
        for i0 in sorted(set(reg_s) & set(bio_s)):
            rv = _extreme_value(q, _metric_signal(reg_s[i0][1], q))
            bv = _extreme_value(q, _metric_signal(bio_s[i0][1], q))
            ok = np.isfinite(rv) and np.isfinite(bv)
            rows.append({"quantity": q, "I0 (A)": i0, "regular": rv, "bioheat": bv,
                         "delta_abs": bv - rv if ok else np.nan,
                         "delta_pct": (bv - rv) / abs(rv) * 100.0 if (ok and abs(rv) > 1e-12) else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Transient figures (legend + value table on the right, like the sensitivity plots)
# ---------------------------------------------------------------------------
def _table_rows(trec, q, currents):
    rows = []
    for i0 in currents:
        s = trec[(trec["quantity"] == q) & (trec["I0 (A)"] == i0)]
        if s.empty:
            continue
        r = s.iloc[0]
        rows.append([f"{i0:g}", fmt(r["regular"]), fmt(r["bioheat"]),
                     fmt(r["delta_abs"], "+.3f"), fmt(r["delta_pct"], "+.1f")])
    return rows


def _right_column(fig, handles, rows, colors, currents, title_note):
    leg_ax = fig.add_axes([0.66, 0.60, 0.32, 0.30])
    leg_ax.axis("off")
    leg_ax.legend(handles=handles, frameon=False, fontsize=14, loc="upper left",
                  borderaxespad=0.0)
    if not rows:
        return
    tab_ax = fig.add_axes([0.66, 0.12, 0.32, 0.42])
    tab_ax.axis("off")
    tab = tab_ax.table(cellText=rows,
                       colLabels=["I0 (A)", "Regular", "Bioheat", r"$\Delta$", r"$\Delta$ %"],
                       colWidths=[0.15, 0.22, 0.22, 0.21, 0.20],
                       cellLoc="center", loc="upper left")
    tab.auto_set_font_size(False)
    tab.set_fontsize(11)
    tab.scale(1.0, 1.7)
    for (r, c), cell in tab.get_celld().items():
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e8e8e8")
        elif c == 0:
            cell.set_text_props(fontweight="bold", color=colors[currents[r - 1]]
                                if r - 1 < len(currents) else "black")
    fig.text(0.66, 0.07, title_note, fontsize=11, ha="left", va="center")


def plot_transient_compare(q, ylabel, title, reg_s, bio_s, trec, currents, colors, outdir):
    fig = plt.figure(figsize=(14, 6.6))
    ax = fig.add_axes([0.07, 0.13, 0.55, 0.76])
    for i0 in currents:
        for s, ls in ((reg_s, "-"), (bio_s, "--")):
            sig = _metric_signal(s[i0][1], q)
            if sig is not None:
                ax.plot(s[i0][0], sig, color=colors[i0], linestyle=ls, linewidth=2.0)
    ax.set_xlabel("Time from pulse onset (s)", labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_title(title, pad=12)
    handles = [Line2D([], [], color=colors[c], lw=2.2, label=f"I0 = {c:g} A") for c in currents]
    handles += [Line2D([], [], color="black", lw=2.2, ls="-", label="Regular"),
                Line2D([], [], color="black", lw=2.2, ls="--", label="Bioheat")]
    _right_column(fig, handles, _table_rows(trec, q, currents), colors, currents,
                  r"Values at the extreme of each trace; $\Delta$ = bioheat $-$ regular")
    save_fig(fig, os.path.join(outdir, f"transient_{q}.png"))


def plot_transient_diff(q, ylabel, title, reg_s, bio_s, trec, currents, colors, outdir):
    fig = plt.figure(figsize=(14, 6.6))
    ax = fig.add_axes([0.07, 0.13, 0.55, 0.76])
    drawn = False
    for i0 in currents:
        sr, sb = _metric_signal(reg_s[i0][1], q), _metric_signal(bio_s[i0][1], q)
        if sr is None or sb is None:
            continue
        tr, tb = reg_s[i0][0], bio_s[i0][0]
        ax.plot(tr, np.interp(tr, tb, sb) - sr, color=colors[i0], linewidth=2.0)
        drawn = True
    if not drawn:
        plt.close(fig)
        return
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Time from pulse onset (s)", labelpad=10)
    ax.set_ylabel(r"$\Delta$ (" + ylabel + ")", labelpad=10)
    ax.set_title(title + " — Bioheat minus Regular", pad=12)
    handles = [Line2D([], [], color=colors[c], lw=2.2, label=f"I0 = {c:g} A") for c in currents]
    _right_column(fig, handles, _table_rows(trec, q, currents), colors, currents,
                  r"$\Delta$ = bioheat $-$ regular (extreme values)")
    save_fig(fig, os.path.join(outdir, f"transient_diff_{q}.png"))


def plot_dashboard(i0, reg_s, bio_s, outdir):
    """2x2 overview for one current: hot, cold, power, cooled volume."""
    specs = [("hot", r"$\Delta T_{\text{hot}}$ (K)"), ("cold", r"$\Delta T_{\text{cold}}$ (K)"),
             ("power", r"$P$ (W)"), ("vol_cool2", r"$V_{\text{cooled,2C}}$ (mm$^3$)")]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    for ax, (q, yl) in zip(axes.ravel(), specs):
        for s, lab, col, ls in ((reg_s, "Regular", REG_COLOR, "-"), (bio_s, "Bioheat", BIO_COLOR, "--")):
            sig = _metric_signal(s[i0][1], q)
            if sig is None:
                continue
            ax.plot(s[i0][0], sig, color=col, linestyle=ls, linewidth=2.2, label=lab)
        ax.set_xlabel("Time from pulse onset (s)", labelpad=8)
        ax.set_ylabel(yl, labelpad=8)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=10)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle(f"Regular vs bioheat — I0 = {i0:g} A", fontsize=17, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, f"dashboard_I0{i0:g}.png"))


# ---------------------------------------------------------------------------
# Metric figures
# ---------------------------------------------------------------------------
def _grid(n, ncol=3):
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol + 1.2, 3.6 * nrow + 1.2))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


def plot_grouped_bars(ax, comp, metric, currents):
    x, w = np.arange(len(currents)), 0.38
    reg = [comp.loc[comp["I0 (A)"] == c, metric + "_regular"].iloc[0] for c in currents]
    bio = [comp.loc[comp["I0 (A)"] == c, metric + "_bioheat"].iloc[0] for c in currents]
    ax.bar(x - w / 2, reg, w, label="Regular", color=REG_COLOR, edgecolor="black", linewidth=0.6)
    ax.bar(x + w / 2, bio, w, label="Bioheat", color=BIO_COLOR, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:g}" for c in currents])
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlabel("I0 (A)", labelpad=12, fontsize=20)
    ax.set_ylabel(label(metric), labelpad=12, fontsize=20)
    ax.set_title(label(metric), pad=14, fontsize=24)
    ax.legend(frameon=False, fontsize=16)


def plot_metric_vs_current(ax, reg_mdf, bio_mdf, metric):
    for mdf, lab, col, mk in ((reg_mdf, "Regular", REG_COLOR, "o"), (bio_mdf, "Bioheat", BIO_COLOR, "s")):
        d = mdf.sort_values("I0 (A)")
        ax.plot(d["I0 (A)"], d[metric], marker=mk, label=lab, color=col)
    ax.set_xlabel("I0 (A)", labelpad=8)
    ax.set_ylabel(label(metric), labelpad=8)
    ax.set_title(label(metric) + " vs. I0", pad=10, fontsize=12)
    ax.legend(frameon=False, fontsize=9)


def plot_delta_vs_current(ax, comp, metric, currents, kind):
    col = metric + ("_pct" if kind == "pct" else "_abs")
    vals = [float(comp.loc[comp["I0 (A)"] == c, col].iloc[0]) for c in currents]
    ax.plot(currents, vals, marker="o", color=CRED)
    ax.axhline(0, color="black", linewidth=1.0)
    for c, v in zip(currents, vals):
        if np.isfinite(v):
            ax.annotate(f"{v:+.1f}" if kind == "pct" else f"{v:+.3g}", (c, v),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5)
    ax.set_xlabel("I0 (A)", labelpad=8)
    ax.set_ylabel(("change (%)" if kind == "pct" else "change (abs)"), labelpad=8)
    ax.set_title(label(metric) + " — bioheat vs regular", pad=10, fontsize=11.5)
    ax.margins(y=0.2)


def plot_heatmap(comp, metrics, currents, outdir):
    valid = [m for m in metrics if comp[m + "_pct"].notna().any()]
    if not valid:
        return
    mat = np.array([[float(comp.loc[comp["I0 (A)"] == c, m + "_pct"].iloc[0]) for c in currents]
                    for m in valid])
    # clip the colour scale so one huge % (e.g. a time constant) doesn't wash out the rest
    vmax = max(1.0, float(np.nanpercentile(np.abs(mat), 90)))
    fig, ax = plt.subplots(figsize=(2.0 + 1.5 * len(currents), 1.5 + 0.48 * len(valid)))
    im = ax.imshow(mat, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(currents)))
    ax.set_xticklabels([f"{c:g}" for c in currents])
    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels([label(m) for m in valid], fontsize=11)
    ax.grid(False)
    for i in range(len(valid)):
        for j in range(len(currents)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=9.5,
                        color="white" if abs(v) > 0.65 * vmax else "black")
    ax.set_xlabel("I0 (A)", labelpad=8)
    ax.set_title("Bioheat vs regular — % change heatmap", pad=12)
    fig.colorbar(im, ax=ax, label=f"% change (colour clipped at ±{vmax:.0f}%)")
    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, "heatmap_pct.png"))


def plot_tornado_by_current(comp, metrics, currents, colors, outdir):
    valid = [m for m in metrics if comp[m + "_pct"].notna().any()]
    if not valid:
        return
    order = sorted(valid, key=lambda m: comp[m + "_pct"].abs().max())  # biggest on top
    y_base = np.arange(len(order))
    h = 0.8 / len(currents)
    fig, ax = new_fig("Bioheat vs regular — % change by current", figsize=(11.5, max(6, 0.62 * len(order))))
    for k, c in enumerate(currents):
        vals = [float(comp.loc[comp["I0 (A)"] == c, m + "_pct"].iloc[0]) for m in order]
        ypos = y_base + (k - (len(currents) - 1) / 2) * h
        ax.barh(ypos, vals, height=h, color=colors[c], edgecolor="black", linewidth=0.4,
                label=f"I0 = {c:g} A")
        for y, v in zip(ypos, vals):
            if np.isfinite(v) and abs(v) > 1e-3:
                ax.annotate(f"{v:+.1f}", (v, y), xytext=(3 if v >= 0 else -3, 0),
                            textcoords="offset points", va="center",
                            ha="left" if v >= 0 else "right", fontsize=7.5)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(y_base)
    ax.set_yticklabels([label(m) for m in order], fontsize=11)
    ax.set_xlabel("% change (bioheat vs regular)", labelpad=10)
    ax.margins(x=0.12)
    ax.legend(frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save_fig(fig, os.path.join(outdir, "tornado_by_current.png"))


def plot_tornado_max(comp, metrics, outdir):
    labels, pcts = [], []
    for m in metrics:
        s = comp[m + "_pct"].dropna()
        if not s.empty:
            labels.append(m)
            pcts.append(float(s.loc[s.abs().idxmax()]))
    if not labels:
        return
    order = np.argsort(np.abs(pcts))
    labels, pcts = [labels[i] for i in order], [pcts[i] for i in order]
    fig, ax = new_fig("Bioheat vs regular — max |% change| over I0", figsize=(10, max(5, 0.45 * len(labels) + 1.5)))
    ypos = np.arange(len(labels))
    ax.barh(ypos, pcts, color=[CBLUE if p >= 0 else CRED for p in pcts],
            edgecolor="black", height=0.62, linewidth=0.6)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([label(m) for m in labels], fontsize=11)
    ax.set_xlabel("% change (bioheat vs regular)", labelpad=10)
    ax.margins(x=0.15)
    for y, p in zip(ypos, pcts):
        ax.annotate(f"{p:+.1f}%", (p, y), xytext=(4 if p >= 0 else -4, 0), textcoords="offset points",
                    va="center", ha="left" if p >= 0 else "right", fontsize=10, fontweight="bold")
    save_fig(fig, os.path.join(outdir, "tornado_max_pct.png"))


def plot_cooling_vs_power(reg_mdf, bio_mdf, outdir):
    fig, ax = new_fig("Cooling performance vs. power", figsize=(9.5, 5.6))
    for mdf, lab, col, mk in ((reg_mdf, "Regular", REG_COLOR, "o"), (bio_mdf, "Bioheat", BIO_COLOR, "s")):
        d = mdf.sort_values("I0 (A)")
        ax.plot(d["net_power_W"], d["dT_cold_peak_K"], marker=mk, color=col, label=lab)
        for _, r in d.iterrows():
            ax.annotate(f"{r['I0 (A)']:g} A", (r["net_power_W"], r["dT_cold_peak_K"]),
                        textcoords="offset points", xytext=(5, -11), fontsize=8.5, color=col)
    ax.set_xlabel(r"Mean Net Power $P_{\text{net}}$ (W)", labelpad=10)
    ax.set_ylabel(label("dT_cold_peak_K"), labelpad=10)
    ax.legend(frameon=False, loc="best")
    save_fig(fig, os.path.join(outdir, "cooling_vs_power.png"))


def plot_response_times(reg_mdf, bio_mdf, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ax, m in zip(axes, ("t_cool63_s", "t_cool90_s")):
        plot_metric_vs_current(ax, reg_mdf, bio_mdf, m)
    fig.suptitle("Response times — regular vs bioheat", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, "response_times.png"))


def plot_efficiency(reg_mdf, bio_mdf, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    plot_metric_vs_current(axes[0], reg_mdf, bio_mdf, "K_per_J")
    plot_metric_vs_current(axes[1], reg_mdf, bio_mdf, "W_full_J")
    fig.suptitle("Efficiency and energy — regular vs bioheat", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, "efficiency_vs_current.png"))


def plot_footprint(reg_mdf, bio_mdf, outdir):
    """Surface extremes and cooled/heated volumes (the surface-file quantities)."""
    panels = [("Surface minimum", ["surf_min_K"], r"$T_{\text{surf,min}}$ (K)"),
              ("Surface maximum", ["surf_max_K"], r"$T_{\text{surf,max}}$ (K)"),
              ("Cooled volume", ["vol_cool_5_end", "vol_cool_1_end"], r"Volume (mm$^3$)"),
              ("Heated volume", ["vol_heat_5_end", "vol_heat_1_end"], r"Volume (mm$^3$)")]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    for ax, (title, mets, yl) in zip(axes.ravel(), panels):
        for k, m in enumerate(mets):
            ls = "-" if k == 0 else "--"
            tag = "" if len(mets) == 1 else (" (5 °C)" if "_5_" in m else " (1 °C)")
            for mdf, lab, col, mk in ((reg_mdf, "Regular", REG_COLOR, "o"), (bio_mdf, "Bioheat", BIO_COLOR, "s")):
                d = mdf.sort_values("I0 (A)")
                ax.plot(d["I0 (A)"], d[m], marker=mk, color=col, linestyle=ls, label=lab + tag)
        ax.set_xlabel("I0 (A)", labelpad=8)
        ax.set_ylabel(yl, labelpad=8)
        ax.set_title(title, pad=10, fontsize=13)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Thermal footprint — regular vs bioheat", fontsize=17, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, os.path.join(outdir, "thermal_footprint.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_comparison(output_root=OUTPUT_ROOT):
    os.makedirs(output_root, exist_ok=True)
    reg_paths, bio_paths = discover_files()

    print("=" * 78)
    print("  REGULAR PHANTOM vs BIOHEAT COMPARISON")
    print("=" * 78)
    print(f"  Folder : {BIOHEAT_DIR}")
    print(f"  Output : {output_root}\n")

    for lab, paths in (("Regular", reg_paths), ("Bioheat", bio_paths)):
        if not paths or not all(os.path.isfile(p) for p in paths):
            print(f"[!] {lab}: no (or missing) CSV files found: {paths}")
            print("    Check BIOHEAT_DIR or set REGULAR_FILES / BIOHEAT_FILES at the top of this script.")
            return
        if len(paths) != 2:
            print(f"[i] {lab}: expected 2 files (probes + surfaces), found {len(paths)} - merging all of them.")

    reg_df = merge_run(reg_paths, "Regular")
    bio_df = merge_run(bio_paths, "Bioheat")

    reg_mdf = compute_run_metrics(reg_df, "Regular")
    bio_mdf = compute_run_metrics(bio_df, "Bioheat")
    comp, currents = build_comparison(reg_mdf, bio_mdf, METRICS)
    if not currents:
        print("[!] The two runs share no I0 values - nothing to compare.")
        return
    print(f"\n  Common currents: {currents}")

    # missing-signal report (e.g. a column that was not found in either file)
    missing = [m for m in METRICS if reg_mdf[m].isna().all() or bio_mdf[m].isna().all()]
    if missing:
        print(f"  [i] metrics without data in at least one run (skipped in plots): {missing}")
    metrics = [m for m in METRICS if m not in missing]

    # ----- tables -----
    reg_mdf.to_csv(os.path.join(output_root, "metrics_regular.csv"), index=False)
    bio_mdf.to_csv(os.path.join(output_root, "metrics_bioheat.csv"), index=False)
    comp.to_csv(os.path.join(output_root, "bioheat_comparison.csv"), index=False)
    build_long(comp, METRICS, currents).to_csv(
        os.path.join(output_root, "bioheat_comparison_long.csv"), index=False)
    build_changes_vs_regular(comp, METRICS).to_csv(
        os.path.join(output_root, "changes_vs_regular.csv"), index=False)
    sum_df = build_summary(comp, METRICS)
    sum_df.to_csv(os.path.join(output_root, "bioheat_summary.csv"), index=False)

    reg_s, bio_s = extract_series(reg_df), extract_series(bio_df)
    common = [c for c in currents if c in reg_s and c in bio_s]
    trec = transient_records(reg_s, bio_s)
    trec.to_csv(os.path.join(output_root, "bioheat_transient_summary.csv"), index=False)
    colors = current_colors(common)

    # ----- figures -----
    # Transients (all currents) + differences
    titles = {
        "hot": "Hot Side Temperature Shift", "cold": "Cold Side Temperature Shift",
        "power": "Electrical Power", "vol_cool2": "Volume Cooled at 2°C",
        "surf_min": "Surface Minimum Temperature Shift", "vol_heat2": "Volume Heated at 2°C",
        "surf_max": "Surface Maximum Temperature Shift",
    }
    for q, ylabel in TRANSIENT_QUANTITIES:
        if all(_metric_signal(reg_s[c][1], q) is None for c in common):
            continue
        try:
            plot_transient_compare(q, ylabel, f"Regular vs Bioheat — {titles[q]}",
                                   reg_s, bio_s, trec, common, colors, output_root)
            plot_transient_diff(q, ylabel, titles[q], reg_s, bio_s, trec, common, colors, output_root)
        except Exception as exc:
            plt.close("all")
            print(f"  [transient_{q}] skipped: {type(exc).__name__}: {exc}")

    # Per-current dashboards
    for c in common:
        try:
            plot_dashboard(c, reg_s, bio_s, output_root)
        except Exception as exc:
            plt.close("all")
            print(f"  [dashboard I0={c:g}] skipped: {type(exc).__name__}: {exc}")

    headline = [m for m in HEADLINE if m in metrics]

    def headline_figure(fname, title, fn):
        fig, axes = _grid(len(headline))
        for ax, m in zip(axes, headline):
            fn(ax, m)
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        fig.tight_layout()
        save_fig(fig, os.path.join(output_root, fname))

    steps = [
        lambda: headline_figure("grouped_bars.png", "Regular vs bioheat — metric values per current",
                                lambda ax, m: plot_grouped_bars(ax, comp, m, currents)),
        lambda: headline_figure("metric_vs_current.png", "Regular vs bioheat — metric vs. drive current",
                                lambda ax, m: plot_metric_vs_current(ax, reg_mdf, bio_mdf, m)),
        lambda: headline_figure("pct_vs_current.png", "Bioheat vs regular — % change per current",
                                lambda ax, m: plot_delta_vs_current(ax, comp, m, currents, "pct")),
        lambda: headline_figure("abs_vs_current.png", "Bioheat vs regular — absolute change per current",
                                lambda ax, m: plot_delta_vs_current(ax, comp, m, currents, "abs")),
        lambda: plot_heatmap(comp, metrics, currents, output_root),
        lambda: plot_tornado_by_current(comp, metrics, currents, current_colors(currents), output_root),
        lambda: plot_tornado_max(comp, metrics, output_root),
        lambda: plot_cooling_vs_power(reg_mdf, bio_mdf, output_root),
        lambda: plot_response_times(reg_mdf, bio_mdf, output_root),
        lambda: plot_efficiency(reg_mdf, bio_mdf, output_root),
        lambda: plot_footprint(reg_mdf, bio_mdf, output_root),
    ]
    for step in steps:
        try:
            step()
        except Exception as exc:
            plt.close("all")
            print(f"  [figure] skipped: {type(exc).__name__}: {exc}")

    # ----- console summary -----
    cols = [c for c in ["metric", "mean_abs_pct", "max_abs_pct", "worst_I0", "max_abs_change"]
            if c in sum_df.columns]
    print("\n### Bioheat vs regular — summary (sorted by max |% change|)\n")
    try:
        print(sum_df[cols].to_markdown(index=False, floatfmt=".3g"))
    except ImportError:
        print(sum_df[cols].to_string(index=False))

    head_cols = ["I0 (A)"] + [m + s for m in ("dT_cold_peak_K", "W_full_J") for s in ("_regular", "_bioheat", "_pct")]
    print("\n### Headline values per current\n")
    print(comp[[c for c in head_cols if c in comp.columns]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 78)
    print("  COMPARISON COMPLETE")
    print("=" * 78)
    print(f"  Output root: {output_root}")
    return comp, sum_df


if __name__ == "__main__":
    run_comparison()