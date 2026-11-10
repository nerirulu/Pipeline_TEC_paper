# =============================================================================
# processing_experimental_outer_surface.py â€” EXP only, surface (IR).
# Companion to "Processing_file copy.py" (loaders/metrics logic) and
# "reading_comsol_heatsink.py" (H-figure family style). Figures E1..E12 are
# saved as PNGs + metrics CSV + index.md. Run:
#     python processing_experimental_outer_surface.py
# =============================================================================
import os, re, time, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.widgets import RectangleSelector
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.stats import linregress
warnings.filterwarnings("ignore", message="Workbook contains no default style*",
                        category=UserWarning)
t0_run = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = dict(
    SETUPS=[
        (os.path.join(HERE, "Castell22C_7Oct30mm", "Excel files"),
         os.path.join(HERE, "Castell22C_7Oct30mm", "VCs_30mm"), "30mm"),
        (os.path.join(HERE, "Castell22C_7Oct40mm", "Excel files"),
         os.path.join(HERE, "Castell22C_7Oct40mm", "VCs_40mm"), "40mm"),
    ],
    FRAME_RATE_HZ=1.0, BG_FRAC=0.05, DT_PERCENTILE=None,
    INTERACTIVE=False, ROI=None,
    MM_PER_PIXEL={"30mm": 0.0999, "40mm": 0.1119},
    T_ON_S=0.0, T_END_S=25.0, T_FINAL_S=60.0,
    CURRENT_ACTIVE_FRAC=0.90, ENERGY_WINDOW_S=60.0,
    COOL_THRESHOLDS=[0.5, 1.0, 2.0, 5.0],
    TAU_FIT_START_S=0.0, TAU_FIT_END_S=30.0, MIN_FIT_POINTS=5,
    COOL_RATE_WINDOW_S=5.0,
    GRID_CURRENTS="all", FOCUS_CURRENT=0.3, OVERLAY_CURRENTS="all",
    SPATIAL_CURRENTS=[0.1, 0.3, 0.5], MAP_TIMES_S=[2.0, 10.0, 20.0, 25.0, 27.0, 35.0, 60.0],
    PROFILE_TIMES_S=[10.0, 25.0, 35.0], AREA_CURRENTS="all",
    CURRENT_CMAP="viridis",
    OUTPUT_DIR=os.path.join(HERE, "OUTPUT_fixed", "exp_outer_surface"),
    METRICS_CSV=os.path.join(HERE, "OUTPUT_fixed", "exp_outer_metrics.csv"),
    FIGURE_DPI=200, SAVE_FIGURES=True, SHOW_FIGURES=False,
    FAST_TEST=False,  # True -> only 1 run per current (quick smoke test)
    FAST_MAP_TIMES=[2.0, 10.0, 25.0, 35.0],
)
matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 30, "axes.titlesize": 35,
    "axes.labelsize": 32, "xtick.labelsize": 32, "ytick.labelsize": 32,
    "legend.fontsize": 32, "lines.linewidth": 3.2, "axes.linewidth": 1.4,
    "grid.linewidth": 1.0, "grid.alpha": 0.3, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.grid": True,
    "legend.frameon": False})
THR_COLORS = {0.5: "#1b7837", 1.0: "#4dac26", 2.0: "#f1a340", 5.0: "#d01c8b"}
index_entries = []
def save_figure(fig, name, index=None):
    if CONFIG["SAVE_FIGURES"]:
        os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
        fig.savefig(os.path.join(CONFIG["OUTPUT_DIR"], name),
                    dpi=CONFIG["FIGURE_DPI"], bbox_inches="tight")
        print("      saved %s" % name)
        if index is not None: index.append(name)
    if CONFIG["SHOW_FIGURES"]: plt.show()
    plt.close(fig)
def trapz(y, x):
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))
def drive_band(ax, t_on, t_end, color="0.88", alpha=0.55):
    ax.axvspan(t_on, t_end, color=color, alpha=alpha, lw=0, zorder=0)
    ax.axvline(t_on, color="0.55", lw=3.4, ls="--", zorder=1)
    ax.axvline(t_end, color="0.55", lw=3.4, ls="--", zorder=1)
# --- loaders (same parsing as Processing_file copy.py) ---
def load_frame(path):
    path = str(path)
    ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
    df, last_err = None, None
    if ext in ("xlsx", "xls", "xlsm", "xlsb", "odf", "ods"):
        for engine in ("openpyxl", "xlrd", "calamine"):
            try:
                df = pd.read_excel(path, header=None, engine=engine); break
            except Exception as e: last_err = e
    if df is None:
        for sep in (";", ",", "\t"):
            try:
                cand = pd.read_csv(path, header=None, sep=sep)
                if cand.shape[1] > 1: df = cand; break
            except Exception: continue
    if df is None: raise ValueError("Cannot load %s: %s" % (path, last_err))
    return (df.apply(lambda c: c.astype(str)
             .str.replace("Â°C", "", regex=False).str.replace("Â°", "", regex=False)
             .str.replace("°C", "", regex=False).str.replace("°", "", regex=False)
             .str.replace(",", ".", regex=False).str.strip())
            .values.astype(float))
def parse(fn):
    m = re.match(r"^(.+?)_(\d+)\.xlsx$", fn, re.IGNORECASE)
    return (m.group(1), int(m.group(2))) if m else (None, None)
def build_tests(setups):
    tests = {}
    for img_folder, vc_folder, label in setups:
        if not os.path.isdir(img_folder):
            print("WARNING: missing %s" % img_folder); continue
        for fn in sorted(f for f in os.listdir(img_folder) if f.lower().endswith(".xlsx")):
            name, idx = parse(fn)
            if name is None: continue
            key = "%s / %s" % (label, name)
            if key not in tests:
                tests[key] = {"imgs": {}, "label": label, "vc_folder": vc_folder}
            tests[key]["imgs"][idx] = os.path.join(img_folder, fn)
    return tests
def load_vc(path):
    df = pd.read_csv(path); df.columns = df.columns.str.strip()
    cm = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "time_ms": cm[c] = "time_ms"
        elif "time" in cl: cm[c] = "time"
        elif "current" in cl: cm[c] = "current"
        elif "voltage" in cl: cm[c] = "voltage"
        elif "phase" in cl: cm[c] = "phase"
    df = df.rename(columns=cm)
    if "time_ms" in df.columns and "time" not in df.columns: df["time"] = df["time_ms"]
    for c in ["time", "current", "voltage"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time", "current", "voltage"]).sort_values("time")
    df["power"] = df["current"] * df["voltage"]
    span = float(df["time"].iloc[-1] - df["time"].iloc[0])
    df["time_s"] = (df["time"] / 1000.0) if span > 5000 else df["time"].astype(float)
    df["time_s"] = df["time_s"] - df["time_s"].iloc[0]
    df["voltage_corr"] = df["voltage"]; df["power_corr"] = df["power"]
    return df
def vc_protocol(df, config=CONFIG):
    t = df["time_s"].to_numpy(float); i = df["current"].to_numpy(float)
    if "phase" in df.columns:
        on = df["phase"].astype(str).str.upper().eq("ON").to_numpy()
        if on.any(): return float(t[on][0]), float(t[on][-1]), round(float(i[on].max()), 2)
    pk = float(np.nanmax(i))
    on = i >= config["CURRENT_ACTIVE_FRAC"] * pk
    if on.any(): return float(t[on][0]), float(t[on][-1]), round(pk, 2)
    return config["T_ON_S"], config["T_END_S"], round(pk, 2)
# --- stacks (outer surface only) + VC table ---
# Frame cache: first run converts xlsx -> .npz (~8 min once), later runs load in seconds.
CACHE_DIR = os.path.join(HERE, "OUTPUT", "_frame_cache_7oct")
os.makedirs(CACHE_DIR, exist_ok=True)
def cached_frame(path):
    key = ("%s_%s" % (os.path.basename(os.path.dirname(os.path.dirname(path))),
                      os.path.basename(path))).replace(".xlsx", ".npz")
    cp = os.path.join(CACHE_DIR, key)
    if os.path.exists(cp):
        try: return np.load(cp)["a"]
        except Exception: pass
    a = load_frame(path)
    try: np.savez_compressed(cp, a=a)
    except Exception: pass
    return a
tests = build_tests(CONFIG["SETUPS"])
for tn, td in sorted(tests.items()):
    s = sorted(td["imgs"]); print("  [%s] %s: %d..%d (%d)" % (td["label"], tn, s[0], s[-1], len(s)))
setup_refs = {}
for tn, td in tests.items():
    if td["label"] not in setup_refs and 0 in td["imgs"]:
        setup_refs[td["label"]] = (tn, cached_frame(td["imgs"][0]))
mm_per_pixel_map = dict(CONFIG["MM_PER_PIXEL"]); roi = CONFIG["ROI"]
roi_crop = (lambda a: a[roi[0]:roi[1], roi[2]:roi[3]]) if roi else (lambda a: a)
def bg_mask(h, w, frac):
    m = np.zeros((h, w), bool); rb, cb = max(1, int(h*frac)), max(1, int(w*frac))
    m[:rb, :] = m[-rb:, :] = m[:, :cb] = m[:, -cb:] = True; return m
all_dtc = {}
for tn, td in tests.items():
    imgs = td["imgs"]
    if 0 not in imgs: continue
    ref = roi_crop(cached_frame(imgs[0])); idxs = sorted(k for k in imgs if k != 0)
    t_ir = np.asarray(idxs, float) / CONFIG["FRAME_RATE_HZ"]
    stack = np.array([roi_crop(cached_frame(imgs[i])) - ref for i in idxs])
    bm = bg_mask(*stack.shape[1:], CONFIG["BG_FRAC"])
    stack = stack - stack[:, bm].mean(axis=1)[:, None, None]
    all_dtc[tn] = (t_ir, stack, td["label"])
    print("  [%s] %s dT [%.3f, %.3f]" % (td["label"], tn, stack.min(), stack.max()))
vc_data = {}
for tn, td in tests.items():
    bare = tn.split(" / ", 1)[-1] if " / " in tn else tn
    for ext in (".csv", ".CSV"):
        p = os.path.join(td.get("vc_folder") or "", bare + ext)
        if not os.path.exists(p): continue
        df = load_vc(p); t_on, t_end, i0 = vc_protocol(df)
        ts = df["time_s"].to_numpy(float)
        act = df["current"].to_numpy(float) >= CONFIG["CURRENT_ACTIVE_FRAC"] * i0
        win = ts <= CONFIG["ENERGY_WINDOW_S"]; dt = np.diff(ts, prepend=ts[0])
        en = float(np.sum(df["power_corr"].to_numpy(float)[win & act] * dt[win & act]))
        vc_data[tn] = dict(i0=i0, energy_J=en, df=df, t_on=t_on, t_end=t_end)
        print("  VC %s I0=%.2f E=%.3f t[%.1f,%.1f]" % (tn, i0, en, t_on, t_end)); break
runs_by_i0 = {}
for tn, vd in vc_data.items():
    if tn in all_dtc:
        _lst = runs_by_i0.setdefault(round(float(vd["i0"]), 2), [])
        if not (CONFIG.get("FAST_TEST") and _lst):
            _lst.append(tn)
currents = sorted(runs_by_i0); print("Currents:", currents)
# --- traces + metrics (outer only) ---
def _res(sel, default):
    if isinstance(sel, str) and sel.lower() == "all": return list(default)
    return [c for c in sel if any(np.isclose(default, c))] or list(default)
GRID_CURRENTS = _res(CONFIG["GRID_CURRENTS"], currents)
OVERLAY_CURRENTS = _res(CONFIG["OVERLAY_CURRENTS"], currents)
SPATIAL_CURRENTS = _res(CONFIG["SPATIAL_CURRENTS"], currents)
AREA_CURRENTS = _res(CONFIG["AREA_CURRENTS"], currents)
focus = min(currents, key=lambda c: abs(c - CONFIG["FOCUS_CURRENT"])) if currents else None
def frame_extrema(stack, pct=None):
    f = stack.reshape(stack.shape[0], -1)
    if pct is None: return f.min(axis=1), f.max(axis=1)
    return np.percentile(f, pct, axis=1), np.percentile(f, 100-pct, axis=1)
def run_traces(tn):
    t, stack, label = all_dtc[tn]
    dmin, dmax = frame_extrema(stack, CONFIG["DT_PERCENTILE"])
    dmean = stack.reshape(stack.shape[0], -1).mean(axis=1)
    mm = mm_per_pixel_map.get(label, 0.10) or 0.10; px = mm ** 2; ar = {}
    for thr in CONFIG["COOL_THRESHOLDS"]:
        ar[thr] = ((stack <= -thr).sum(axis=(1, 2))*px, (stack >= thr).sum(axis=(1, 2))*px)
    return t, dmin, dmax, dmean, ar
def current_mean_traces(i0):
    tns = runs_by_i0.get(i0, [])
    tu = np.unique(np.concatenate([all_dtc[tn][0] for tn in tns]))
    acc = {"cold": [], "hot": [], "mean": []}
    aa = {thr: ([], []) for thr in CONFIG["COOL_THRESHOLDS"]}
    for tn in tns:
        t, dmin, dmax, dmean, areas = run_traces(tn)
        for k, y in zip(("cold", "hot", "mean"), (dmin, dmax, dmean)):
            acc[k].append(np.interp(tu, t, y))
        for thr in CONFIG["COOL_THRESHOLDS"]:
            for j in (0, 1): aa[thr][j].append(np.interp(tu, t, areas[thr][j]))
    out = {"t": tu, "tns": tns}
    for k in acc: out[k] = np.mean(acc[k], axis=0)
    out["areas"] = {thr: (np.mean(np.vstack(v[0]), axis=0), np.mean(np.vstack(v[1]), axis=0))
                    for thr, v in aa.items()}
    return out
CMAP = plt.get_cmap(CONFIG["CURRENT_CMAP"])
CNORM = plt.Normalize(min(currents), max(currents)) if currents else None
COLORS = {c: CMAP(CNORM(c)) for c in currents} if currents else {}
def exp_decay(t, a, tau, c): return a * (1.0 - np.exp(-t / tau)) + c
def fit_tau(t, y):
    # Cold-spot Î”T goes NEGATIVE: fit -y (positive rise) with tau bounded to run length.
    m = (t >= CONFIG["TAU_FIT_START_S"]) & (t <= CONFIG["TAU_FIT_END_S"])
    if m.sum() < CONFIG["MIN_FIT_POINTS"]: return np.nan, np.nan
    tf, yf = np.asarray(t[m], float), np.asarray(y[m], float)
    yn = -(yf - yf[0])  # positive-going normalised signal
    if not np.isfinite(yn).any() or np.nanmax(yn) <= 0: return np.nan, np.nan
    try:
        po, _ = curve_fit(exp_decay, tf - tf[0], yn, p0=[yn[-1], 5.0, 0.0],
                          bounds=([0, 0.1, -np.inf], [np.inf, 120.0, np.inf]), maxfev=20000)
        return abs(float(po[1])), float(yf[0] - (po[0] + po[2]))
    except Exception: return np.nan, np.nan
def _scalar_from_trace(t, dmin, dmax, ar, energy):
    """Headline scalar metrics for ONE experimental run (used to get per-repeat SD).

    Mirrors the metric definitions in `current_metrics` but on a single run's
    trace, so the spread across repeats can be measured. The mean columns in
    `current_metrics` are left untouched (they stay the metric-of-mean-trace).
    """
    dr = (t >= CONFIG["T_ON_S"]) & (t <= CONFIG["T_END_S"])
    pk_c = float(np.nanmin(dmin[dr])) if dr.any() else np.nan
    pk_h = float(np.nanmax(dmax[dr])) if dr.any() else np.nan
    tpk = float(t[dr][int(np.argmin(dmin[dr]))]) if dr.any() else np.nan
    tau, _ = fit_tau(t, dmin)
    _m2 = (t >= CONFIG["T_ON_S"]) & (t <= CONFIG["T_ON_S"] + CONFIG["COOL_RATE_WINDOW_S"])
    _m2 = _m2 & np.isfinite(dmin)
    rate = float(linregress(t[_m2], dmin[_m2]).slope) if _m2.sum() >= 3 else np.nan
    po = t > CONFIG["T_END_S"]
    ov = float(np.nanmax(dmin[po]) - dmin[dr][-1]) if dr.any() and po.any() else np.nan
    ie = int(np.argmin(np.abs(t - CONFIG["T_END_S"])))
    out = {"dT_cold_peak (C)": pk_c, "dT_hot_peak (C)": pk_h, "t_peak_cool (s)": tpk,
           "tau_cold (s)": tau, "cooling_rate (C/s)": rate, "overshoot_cold (C)": ov,
           "E_input (J)": float(energy), "K_per_J": abs(pk_c) / energy if energy else np.nan,
           "dT_cold_end (C)": float(np.interp(CONFIG["T_END_S"], t, dmin)) if len(t) else np.nan,
           "dT_hot_end (C)": float(np.interp(CONFIG["T_END_S"], t, dmax)) if len(t) else np.nan}
    for thr in CONFIG["COOL_THRESHOLDS"]:
        tag = ("%g" % thr).replace(".", "p")
        co, he = ar[thr]
        out["area_cool_%s_end (mm2)" % tag] = float(co[ie])
        out["area_heat_%s_end (mm2)" % tag] = float(he[ie])
        out["area_cool_%s_peak (mm2)" % tag] = float(np.nanmax(co))
        out["area_heat_%s_peak (mm2)" % tag] = float(np.nanmax(he))
        out["area_cool_%s_per_J (mm2/J)" % tag] = float(np.nanmax(co)) / energy if energy else np.nan
        out["area_heat_%s_per_J (mm2/J)" % tag] = float(np.nanmax(he)) / energy if energy else np.nan
    c05, h05 = out["area_cool_0p5_end (mm2)"], out["area_heat_0p5_end (mm2)"]
    out["ratio_cool_heat"] = c05 / h05 if h05 else np.nan
    return out


def current_run_metrics(i0):
    """Per-run scalar metrics for one current -> list of dicts (one per repeat)."""
    rows = []
    for tn in runs_by_i0.get(i0, []):
        t, dmin, dmax, _dmean, ar = run_traces(tn)
        rows.append(_scalar_from_trace(t, dmin, dmax, ar, vc_data[tn]["energy_J"]))
    return rows


def _sd_name(metric):
    """'dT_cold_peak (C)' -> 'dT_cold_peak_sd (C)'; 'K_per_J' -> 'K_per_J_sd'."""
    if " (" in metric:
        return metric.replace(" (", "_sd (", 1)
    return metric + "_sd"


def current_metrics(i0):
    cm = current_mean_traces(i0); t = cm["t"]
    dr = (t >= CONFIG["T_ON_S"]) & (t <= CONFIG["T_END_S"])
    pk_c = float(np.nanmin(cm["cold"][dr])) if dr.any() else np.nan
    pk_h = float(np.nanmax(cm["hot"][dr])) if dr.any() else np.nan
    tpk = float(t[dr][int(np.argmin(cm["cold"][dr]))]) if dr.any() else np.nan
    tau, _ = fit_tau(t, cm["cold"])
    # initial cooling rate: linear fit over first COOL_RATE_WINDOW_S of DRIVE data
    _m2 = (t >= CONFIG["T_ON_S"]) & (t <= CONFIG["T_ON_S"] + CONFIG["COOL_RATE_WINDOW_S"])
    _m2 = _m2 & np.isfinite(cm["cold"])
    rate = float(linregress(t[_m2], cm["cold"][_m2]).slope) if _m2.sum() >= 3 else np.nan
    po = t > CONFIG["T_END_S"]
    ov = float(np.nanmax(cm["cold"][po]) - cm["cold"][dr][-1]) if dr.any() and po.any() else np.nan
    es = [vc_data[tn]["energy_J"] for tn in cm["tns"]]; em = float(np.mean(es)) if es else np.nan
    row = {"I0 (A)": i0, "n_runs": len(cm["tns"]), "dT_cold_peak (C)": pk_c,
           "dT_hot_peak (C)": pk_h, "t_peak_cool (s)": tpk, "tau_cold (s)": tau,
           "cooling_rate (C/s)": rate, "overshoot_cold (C)": ov,
           "E_input (J)": em, "K_per_J": abs(pk_c)/em if em else np.nan,
           "dT_cold_end (C)": float(np.interp(CONFIG["T_END_S"], t, cm["cold"])) if len(t) else np.nan,
           "dT_hot_end (C)": float(np.interp(CONFIG["T_END_S"], t, cm["hot"])) if len(t) else np.nan}
    for thr in CONFIG["COOL_THRESHOLDS"]:
        tag = ("%g" % thr).replace(".", "p"); co, he = cm["areas"][thr]
        ie = int(np.argmin(np.abs(t - CONFIG["T_END_S"])))
        row["area_cool_%s_end (mm2)" % tag] = float(co[ie])
        row["area_heat_%s_end (mm2)" % tag] = float(he[ie])
        row["area_cool_%s_peak (mm2)" % tag] = float(np.nanmax(co))
        row["area_heat_%s_peak (mm2)" % tag] = float(np.nanmax(he))
        row["area_cool_%s_per_J (mm2/J)" % tag] = float(np.nanmax(co))/em if em else np.nan
        row["area_heat_%s_per_J (mm2/J)" % tag] = float(np.nanmax(he))/em if em else np.nan
    c05, h05 = row["area_cool_0p5_end (mm2)"], row["area_heat_0p5_end (mm2)"]
    row["ratio_cool_heat"] = c05/h05 if h05 else np.nan
    # --- per-repeat standard deviation across the repeats (additive) ----------
    # Adds `<metric>_sd (unit)` columns so the comparison tables can show
    # "EXP mean +/- SD".  The mean columns above are unchanged (metric-of-mean).
    run_rows = current_run_metrics(i0)
    if len(run_rows) > 1:
        for metric in run_rows[0]:
            vals = np.array([r[metric] for r in run_rows], float)
            vals = vals[np.isfinite(vals)]
            row[_sd_name(metric)] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    return row
table = pd.DataFrame([current_metrics(c) for c in currents])
if not table.empty:
    os.makedirs(os.path.dirname(CONFIG["METRICS_CSV"]), exist_ok=True)
    table.to_csv(CONFIG["METRICS_CSV"], index=False); print(table.to_string(index=False))
T_END_REF = float(np.mean([v["t_end"] for v in vc_data.values()])) if vc_data else CONFIG["T_END_S"]
print("Drive end ref: %.1f s" % T_END_REF)
# --- figures E1..E3 (cf. H1..H3) ---
def fig_e1_grid(index, fname="E1_roi_dt_grid.png"):
    fig, axes = plt.subplots(2, 1, figsize=(22.88, 14.56), sharex=True)
    for i0 in GRID_CURRENTS:
        cm = current_mean_traces(i0)
        axes[0].plot(cm["t"], cm["hot"], lw=4.5, color=COLORS[i0], label="I0=%.2f A" % i0)
        axes[1].plot(cm["t"], cm["cold"], lw=4.5, color=COLORS[i0])
    for ax in axes:
        drive_band(ax, CONFIG["T_ON_S"], T_END_REF); ax.set_xlim(0, CONFIG["T_FINAL_S"])
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("$\Delta T_h$ (°C)"); axes[1].set_ylabel("$\Delta T_c$ (°C)")
    axes[0].set_title("(a) $\Delta T_h$ ROI â€” surface (IR)", fontsize=35)
    axes[1].set_title("(b) $\Delta T_c$ ROI â€” surface (IR)", fontsize=35)
    axes[0].legend(fontsize=32, ncol=3)
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout(); save_figure(fig, fname, index)
def fig_e2_focus(index, fname="E2_focus.png"):
    cm = current_mean_traces(focus)
    fig, axes = plt.subplots(2, 2, figsize=(24.96, 14.56))
    axes[0][0].plot(cm["t"], cm["hot"], "r-", lw=5, label="hot-spot")
    axes[0][0].plot(cm["t"], cm["cold"], "b-", lw=5, label="cold-spot")
    axes[0][0].plot(cm["t"], cm["mean"], "k--", lw=3.4, label="ROI mean")
    axes[0][0].set_title("(a) ROI $\Delta T$, $I_0$ = %.2f A" % focus, fontsize=35)
    axes[0][0].set(xlabel="Time (s)", ylabel="$\Delta T$ (°C)"); axes[0][0].legend(fontsize=32)
    for tn in cm["tns"]:
        t, dmin, dmax, _, _ = run_traces(tn)
        axes[0][1].plot(t, dmin, "o-", ms=8.4, lw=3.4, label=tn.split(" / ")[-1])
    axes[0][1].set_title("(b) Repeatability per run", fontsize=35)
    axes[0][1].set(xlabel="Time (s)", ylabel="$\Delta T$ (°C)"); axes[0][1].legend(fontsize=31)
    ax = axes[1][0]
    for thr in CONFIG["COOL_THRESHOLDS"]:
        co, he = cm["areas"][thr]
        ln, = ax.plot(cm["t"], co, lw=4.8, color=THR_COLORS[thr], label="cool %gK" % thr)
        ax.plot(cm["t"], he, lw=3.9, ls="--", color=ln.get_color())
    ax.set_title("(c) Cooled solid / heated dashed", fontsize=35)
    ax.set(xlabel="Time (s)", ylabel="area (mm$^2$)"); ax.legend(fontsize=32, ncol=2)
    ax = axes[1][1]; df = vc_data[cm["tns"][0]]["df"]
    ax.plot(df["time_s"], df["voltage"], lw=4.2)
    ax.set_title("(d) Drive voltage %s" % cm["tns"][0], fontsize=35)
    ax.set(xlabel="Time (s)", ylabel="V (V)")
    for ax in axes.flat:
        drive_band(ax, CONFIG["T_ON_S"], T_END_REF); ax.set_xlim(0, CONFIG["T_FINAL_S"])
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout(); save_figure(fig, fname, index)
def fig_e3_overlay(index, fname="E3_overlay.png"):
    fig, axes = plt.subplots(1, 2, figsize=(24.96, 9.62), sharey=True)
    for i0 in OVERLAY_CURRENTS:
        cm = current_mean_traces(i0)
# --- figures E4..E6 (cf. H4..H6) ---
def fig_e4_normalized(index, fname="E4_normalized.png"):
    fig, ax = plt.subplots(figsize=(18.72, 10.4))
    for i0 in OVERLAY_CURRENTS:
        cm = current_mean_traces(i0); pk = np.nanmin(cm["cold"])
        if np.isfinite(pk) and abs(pk) > 1e-6:
            tau = table.set_index("I0 (A)").loc[i0, "tau_cold (s)"] if not table.empty else np.nan
            ax.plot(cm["t"], cm["cold"]/abs(pk), lw=4.8, color=COLORS[i0],
                    label="%.2f A (tau=%.1f s)" % (i0, tau))
    drive_band(ax, CONFIG["T_ON_S"], T_END_REF)
    ax.set(xlabel="Time (s)", ylabel="$\Delta T_c/|peak|$ (-)", xlim=(0, CONFIG["T_FINAL_S"]))
    ax.set_title("Normalised cold-spot response", fontsize=35); ax.legend(fontsize=32)
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout(); save_figure(fig, fname, index)
def fig_e5_switchoff(index, fname="E5_switchoff.png"):
    fig, axes = plt.subplots(1, 2, figsize=(24.96, 9.62))
    for i0 in OVERLAY_CURRENTS:
        cm = current_mean_traces(i0); m = cm["t"] >= T_END_REF - 5
        axes[0].plot(cm["t"][m]-T_END_REF, cm["cold"][m], lw=4.8, color=COLORS[i0], label="%.2f A" % i0)
    axes[0].axvline(0, color="0.4", lw=3.4, ls=":")
    axes[0].set_title("(a) Switch-off (t-t_end)", fontsize=35)
    axes[0].set(xlabel="Time (s)", ylabel="$\Delta T$c (°C)"); axes[0].legend(fontsize=32)
    if not table.empty:
        axes[1].plot(table["I0 (A)"], table["overshoot_cold (C)"], "o-", lw=5)
        axes[1].set_title("(b) Rebound vs current", fontsize=35)
        axes[1].set(xlabel="Drive current $I_0$ (A)", ylabel="overshoot (°C)")
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout(); save_figure(fig, fname, index)
def fig_e6_check(index, fname="E6_extrema_check.png"):
    cm = current_mean_traces(focus); tn = cm["tns"][0]
    t, stack, _ = all_dtc[tn]
    smin, smax = frame_extrema(stack, None); pmin, pmax = frame_extrema(stack, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(24.96, 9.62), sharex=True)
    axes[0].plot(t, smin, "b-", label="strict min"); axes[0].plot(t, pmin, "b--", label="1st pct")
    axes[1].plot(t, smax, "r-", label="strict max"); axes[1].plot(t, pmax, "r--", label="99th pct")
    for ax in axes:
        drive_band(ax, CONFIG["T_ON_S"], T_END_REF)
        ax.set(xlabel="Time (s)", ylabel="$\Delta T$ (°C)", xlim=(0, CONFIG["T_FINAL_S"])); ax.legend(fontsize=32)
    pass  # suptitle removed (unified style: no long titles)
# --- figures E7..E8: area vs current / balance (cf. H8/H9) ---
def fig_e7_area(index, fname="E7_area_vs_current.png"):
    fig, axes = plt.subplots(2, 2, figsize=(24.96, 16.64))
    panels = [("cool", "end", "(a) Cooled area at drive end"),
              ("heat", "end", "(b) Heated area at drive end"),
              ("cool", "peak", "(c) Peak cooled area"),
              ("heat", "peak", "(d) Peak heated area")]
    for ax, (kind, when, ttl) in zip(axes.flat, panels):
        mk = "o" if when == "end" else "s"
        for thr in CONFIG["COOL_THRESHOLDS"]:
            tag = ("%g" % thr).replace(".", "p")
            col = "area_%s_%s_%s (mm2)" % (kind, tag, when)
            sd_col = "area_%s_%s_%s_sd (mm2)" % (kind, tag, when)
            if col in table.columns:
                yerr = table[sd_col] if sd_col in table.columns else None
                ax.errorbar(table["I0 (A)"], table[col], yerr=yerr,
                            fmt="-%s" % mk, ms=14, lw=5, capsize=8,
                            color=THR_COLORS[thr], label="%g K" % thr)
        ax.set_title(ttl, fontsize=35)
        ax.set(xlabel="Drive current $I_0$ (A)", ylabel="area (mm$^2$)"); ax.set_ylim(bottom=0)
        ax.legend(fontsize=32, ncol=2)
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout(); save_figure(fig, fname, index)
def fig_e8_balance(index, fname="E8_balance.png"):
    fig, axes = plt.subplots(1, 2, figsize=(24.96, 9.62))
    for thr in CONFIG["COOL_THRESHOLDS"]:
        tag = ("%g" % thr).replace(".", "p")
        cc = table["area_cool_%s_end (mm2)" % tag].to_numpy(float)
        hh = table["area_heat_%s_end (mm2)" % tag].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(hh > 0, cc/hh, np.nan)
        axes[0].plot(table["I0 (A)"], r, "-o", ms=14, lw=5, color=THR_COLORS[thr], label="%g K" % thr)
    axes[0].axhline(1.0, color="0.4", lw=3.4, ls=":"); axes[0].set_yscale("log")
    axes[0].set_title("(a) cooled/heated at drive end (log)", fontsize=35)
    axes[0].set(xlabel="Drive current $I_0$ (A)", ylabel="ratio (-)"); axes[0].legend(fontsize=32)
    axes[1].plot(table["I0 (A)"], table["dT_cold_peak (C)"], "bo-", label="cold peak")
    axes[1].plot(table["I0 (A)"], table["dT_hot_peak (C)"], "ro-", label="hot peak")
    axes[1].set_title("(b) Peak $\Delta T$ vs current", fontsize=35)
    axes[1].set(xlabel="Drive current $I_0$ (A)", ylabel="$\Delta T$ (°C)"); axes[1].legend(fontsize=32)
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout(); save_figure(fig, fname, index)
def _map_frame(tn, t_tgt):
    t, stack, _ = all_dtc[tn]; k = int(np.argmin(np.abs(t - t_tgt)))
    return stack[k], float(t[k])
def fig_e9_maps(index, fname="E9_surface_maps_grid.png"):
    times = CONFIG["MAP_TIMES_S"]
    fig, axes = plt.subplots(len(SPATIAL_CURRENTS), len(times)+1,
                             figsize=(3.0*(len(times)+1), 2.9*len(SPATIAL_CURRENTS)), squeeze=False)
    for r, i0 in enumerate(SPATIAL_CURRENTS):
        cm = current_mean_traces(i0); tn = cm["tns"][0]
        mm = mm_per_pixel_map.get(all_dtc[tn][2], 0.10) or 0.10
        vmax = max(1e-6, float(np.nanmax(np.abs(
            np.concatenate([_map_frame(tn, tt)[0].ravel() for tt in times])))))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        for c, tt in enumerate(times):
            ax = axes[r][c]; fr, ta = _map_frame(tn, tt)
            co = float((fr <= -0.5).sum()*mm**2); he = float((fr >= 0.5).sum()*mm**2)
            im = ax.imshow(fr, cmap="seismic", norm=norm, aspect="auto", origin="upper")
            ax.set_title("$I_0$ = %.2f A t=%.0f s\ncool %.1f | heat %.1f mm$^2$" % (i0, ta, co, he), fontsize=32)
            ax.set_xticks([]); ax.set_yticks([])
        ax = axes[r][len(times)]; fr, ta = _map_frame(tn, times[-1])
        im = ax.imshow(fr, cmap="seismic", norm=norm, aspect="auto", origin="upper")
        ax.set_title("last t=%.0f s (row scale)" % ta, fontsize=32)
        ax.set_xticks([]); ax.set_yticks([])
        cax = make_axes_locatable(ax).append_axes("right", size="6%", pad=0.06)
        cb = fig.colorbar(im, cax=cax); cb.set_label("surface (IR)", fontsize=32)
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout(); save_figure(fig, fname, index)

    fig.tight_layout(); save_figure(fig, fname, index)
# --- figures E10/E11 (cf. H11/H12-14) ---
def fig_e10_metrics(index, fname="E10_area_metrics.png"):
    fig, axes = plt.subplots(2, 2, figsize=(27.04, 16.64))
    for ax, (kind, ttl) in zip(axes[0], [("cool", "(a) Cooled at drive end"), ("heat", "(b) Heated at drive end")]):
        for thr in CONFIG["COOL_THRESHOLDS"]:
            tag = ("%g" % thr).replace(".", "p")
            col = "area_%s_%s_end (mm2)" % (kind, tag)
            sd_col = "area_%s_%s_end_sd (mm2)" % (kind, tag)
            yerr = table[sd_col] if sd_col in table.columns else None
            ax.errorbar(table["I0 (A)"], table[col], yerr=yerr,
                        fmt="-o", ms=12.6, lw=5.3, capsize=8,
                        color=THR_COLORS[thr], label="%g K" % thr)
        ax.set_title(ttl, fontsize=35); ax.set(xlabel="Drive current $I_0$ (A)", ylabel="area (mm$^2$)")
        ax.set_ylim(bottom=0); ax.legend(fontsize=32, ncol=2)
    ax = axes[1][0]
    for thr in CONFIG["COOL_THRESHOLDS"]:
        tag = ("%g" % thr).replace(".", "p")
        cc = table["area_cool_%s_end (mm2)" % tag].to_numpy(float)
        hh = table["area_heat_%s_end (mm2)" % tag].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(hh > 0, cc/hh, np.nan)
        ax.plot(table["I0 (A)"], r, "-o", ms=12.6, lw=5.3, color=THR_COLORS[thr], label="%g K" % thr)
    ax.axhline(1.0, color="0.4", lw=3.4, ls=":"); ax.set_yscale("log")
    ax.set_title("(c) Balance (log)", fontsize=35)
    ax.set(xlabel="Drive current $I_0$ (A)", ylabel="cooled/heated (-)"); ax.legend(fontsize=32, ncol=2)
    ax = axes[1][1]
    for i0 in AREA_CURRENTS:
        cm = current_mean_traces(i0); co, he = cm["areas"][0.5]
        ln, = ax.plot(cm["t"], co, lw=4.8, label="I0=%.2f A" % i0)
        ax.plot(cm["t"], he, lw=4.2, ls="--", color=ln.get_color())
    drive_band(ax, CONFIG["T_ON_S"], T_END_REF)
    ax.set_title("(d) Area 0.5K vs time", fontsize=35)
    ax.set(xlabel="Time (s)", ylabel="area (mm$^2$)", xlim=(0, CONFIG["T_FINAL_S"]))
    ax.legend(fontsize=32, ncol=2)
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout(); save_figure(fig, fname, index)
def fig_e11_stats(index, fname="E11_stats_profiles.png"):
    cm = current_mean_traces(focus); tn = cm["tns"][0]
    fig, axes = plt.subplots(2, 2, figsize=(27.04, 16.64))
    ax = axes[0][0]
    for i0 in SPATIAL_CURRENTS:
        c2 = current_mean_traces(i0); tn2 = c2["tns"][0]
        _, st2, _ = all_dtc[tn2]
        sd = [st2[np.argmin(np.abs(c2["t"]-tt))].std() for tt in c2["t"]]
        ax.plot(c2["t"], sd, lw=4.8, color=COLORS[i0], label="%.2f A" % i0)
    drive_band(ax, CONFIG["T_ON_S"], T_END_REF)
    ax.set_title("(a) Spatial std($\Delta T$) vs time", fontsize=35)
    ax.set(xlabel="Time (s)", ylabel="std (°C)", xlim=(0, CONFIG["T_FINAL_S"])); ax.legend(fontsize=32)
    fr0, _ = _map_frame(tn, CONFIG["PROFILE_TIMES_S"][0])
    cy, cx = np.unravel_index(np.argmin(fr0), fr0.shape)
    ax = axes[0][1]
    for tt in CONFIG["PROFILE_TIMES_S"]:
        fr, ta = _map_frame(tn, tt); ax.plot(fr[cy, :], lw=4.8, label="t=%.0f s" % ta)
    ax.set_title("(b) Horiz cut row %d" % cy, fontsize=35)
    ax.set(xlabel="x (px)", ylabel="$\Delta T$ (°C)"); ax.legend(fontsize=32)
    ax = axes[1][0]
    for tt in CONFIG["PROFILE_TIMES_S"]:
        fr, ta = _map_frame(tn, tt); ax.plot(fr[:, cx], lw=4.8, label="t=%.0f s" % ta)
    ax.set_title("(c) Vert cut col %d" % cx, fontsize=35)
    ax.set(xlabel="y (px)", ylabel="$\Delta T$ (°C)"); ax.legend(fontsize=32)
    ax = axes[1][1]; t, stack, _ = all_dtc[tn]; kymo = stack[:, cy, :]
# --- figure E12 energy/efficiency/scorecard (cf. H17-19) + run ---
def v_model(t, V0, Vs, tau): return V0 + Vs*(1.0-np.exp(-t/tau))
def fig_e12_energy(index, fname="E12_energy_efficiency.png"):
    fig, axes = plt.subplots(2, 2, figsize=(27.04, 16.64))
    axes[0][0].plot(table["I0 (A)"], table["E_input (J)"], "ko-", lw=5)
    axes[0][0].set_title("(a) Input energy vs current", fontsize=35)
    axes[0][0].set(xlabel="Drive current $I_0$ (A)", ylabel="E (J)")
    axes[0][1].plot(table["I0 (A)"], table["K_per_J"], "mo-", lw=5)
    axes[0][1].set_title("(b) Cooling per joule", fontsize=35)
    axes[0][1].set(xlabel="Drive current $I_0$ (A)", ylabel="|$\Delta T$c|/E (C/J)")
    ax = axes[1][0]; added = set()
    for tn, vd in vc_data.items():
        df = vd["df"]; on = df["current"] >= CONFIG["CURRENT_ACTIVE_FRAC"]*vd["i0"]
        tt = (df["time_s"][on]-df["time_s"][on].iloc[0]).to_numpy(float)
        vv = df["voltage"][on].to_numpy(float); m = tt <= 10.0
        if m.sum() < 4: continue
        i0r = round(float(vd["i0"]), 2)
        try:
            po, _ = curve_fit(v_model, tt[m], vv[m], p0=[vv[m][0], max(vv[m][-1]-vv[m][0], 1e-4), 5.0], maxfev=5000)
            V0f, Vsf, tauf = po
        except Exception: V0f, Vsf, tauf = np.nan, np.nan, np.nan
        ax.plot(tt[m], vv[m], color=COLORS.get(i0r, "k"), alpha=0.35, lw=3.4)
        if np.isfinite(tauf):
            lb = "I0=%.2f A (tau=%.2f s)" % (i0r, tauf) if i0r not in added else None
            tf = np.linspace(tt[m][0], tt[m][-1], 200)
            ax.plot(tf, v_model(tf, V0f, Vsf, tauf), color=COLORS.get(i0r, "k"), ls="--", lw=5.6, label=lb)
            added.add(i0r)
    ax.set_title("(c) V(t) + exp fit (10 s)", fontsize=35)
    ax.set(xlabel="Time (s)", ylabel="V (V)", xlim=(0, 10)); ax.legend(fontsize=32)
    ax = axes[1][1]; ax.axis("off")
    if not table.empty:
        _keep = ["I0 (A)", "n_runs", "dT_cold_peak (C)", "dT_hot_peak (C)",
                 "E_input (J)", "K_per_J"]
        _keep = [c for c in _keep if c in table.columns]
        ax.text(0.02, 0.95, table[_keep].round(3).to_string(index=False),
                transform=ax.transAxes,
                fontsize=30, family="monospace", va="top", ha="left")
        ax.set_title("(d) Scorecard table", fontsize=35)
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout(); save_figure(fig, fname, index)
# --- figure E13 min/max relative-T vs energy (cf. H20) ---
def fig_e13_minmax_vs_energy(index, fname="E13_minmax_vs_energy.png"):
    df = table.sort_values("I0 (A)")
    fig, axes = plt.subplots(1, 2, figsize=(27.04, 10.01))
    panels = [(("dT_cold_peak (C)", "dT_hot_peak (C)"),
               "(a) Peak in drive window vs input energy"),
              (("dT_cold_end (C)", "dT_hot_end (C)"),
               "(b) End-of-pulse (t = %g s) vs input energy" % CONFIG["T_END_S"])]
    for ax, ((cc, ch), title) in zip(axes, panels):
        ax.errorbar(df["E_input (J)"], df[cc], xerr=df.get("E_input_sd (J)", 0.0),
                    fmt="v", color="#2166ac", ecolor="0.5", capsize=8, ms=19.6,
                    label="min $\Delta T$ (coldest, frame min)")
        ax.errorbar(df["E_input (J)"], df[ch], xerr=df.get("E_input_sd (J)", 0.0),
                    fmt="^", color="#b2182b", ecolor="0.5", capsize=8, ms=19.6,
                    label="max $\Delta T$ (warmest, frame max)")
        for _, r in df.iterrows():
            ax.annotate("%.2f A\n(n=%d)" % (r["I0 (A)"], int(r["n_runs"])),
                        (r["E_input (J)"], r[cc]), fontsize=30, ha="right", va="top")
        ax.axhline(0.0, color="k", lw=3.4)
        ax.set_title(title, fontsize=35)
        ax.set(xlabel="Input energy $E=\\sum V\\cdot I\\,dt$ (J)",
               ylabel="$\Delta T$ (°C)")
        ax.legend(fontsize=32, loc="best")
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout(); save_figure(fig, fname, index)
import sys
if "--probe" in sys.argv:
    # Validate data section only (no figures): exercises loaders/stacks/metrics.
    print("PROBE currents:", currents)
    _cm = current_mean_traces(0.3)
    print("PROBE traces t:", _cm["t"].shape, "tns:", _cm["tns"])
    print(table[["I0 (A)", "dT_cold_peak (C)", "tau_cold (s)",
                 "E_input (J)"]].to_string(index=False))
    print("PROBE OK")
    sys.exit(0)
print("\n[figures] -> %s ..." % CONFIG["OUTPUT_DIR"])
fig_e1_grid(index_entries); fig_e2_focus(index_entries); fig_e3_overlay(index_entries)
fig_e4_normalized(index_entries); fig_e5_switchoff(index_entries); fig_e6_check(index_entries)
fig_e7_area(index_entries); fig_e8_balance(index_entries); fig_e9_maps(index_entries)
fig_e10_metrics(index_entries); fig_e11_stats(index_entries); fig_e12_energy(index_entries)
fig_e13_minmax_vs_energy(index_entries)
with open(os.path.join(CONFIG["OUTPUT_DIR"], "index.md"), "w") as f:
    f.write("# EXP outer-surface report (7th Oct Castell, IR only)\n\n")
    f.write("Drive 0-%.0f s | end %.0f s. dT vs frame 0, border-bg subtracted.\n\n" % (T_END_REF, CONFIG["T_FINAL_S"]))
    f.write("Areas = IR px counts x (mm/px)^2. No COMSOL/inner/volume data.\n\n")
    for n in index_entries: f.write("- %s\n" % n)
print("\nDone in %.1f s â€” %d figs" % (time.time()-t0_run, len(index_entries)))

