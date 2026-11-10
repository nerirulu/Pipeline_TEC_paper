# =============================================================================
# comparison_heatsink_vs_exp_outer.py — part 1/3: config + loaders
# Combines reading_comsol_heatsink.py (SIM, OUTPUT/heatsink_metrics.csv + the
# inner-surface field metrics `surf_*` and OUTPUT/heatsink_surface_traces.csv) with
# processing_experimental_outer_surface.py (EXP outer-surface IR only,
# OUTPUT/exp_outer_metrics.csv). Run: python comparison_heatsink_vs_exp_outer.py
# =============================================================================
import os, sys, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D as _L
from matplotlib.patches import Patch
from openpyxl import load_workbook

t0_run = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# --- shared-module lookup ---------------------------------------------------
# The loaders/plot helpers live in the shared modules, which may sit either next
# to this script or in the "other files" folder, so both locations are made
# importable (the script folder always wins).
_HERE = HERE
for _cand in (os.path.join(_HERE, "other files"), _HERE):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.append(_cand)

CONFIG = dict(
    SIM_METRICS=os.path.join(HERE, "OUTPUT_fixed", "heatsink_metrics.csv"),
    SIM_SURF_TRACES=os.path.join(HERE, "OUTPUT_fixed", "heatsink_surface_traces.csv"),
    EXP_METRICS=os.path.join(HERE, "OUTPUT_fixed", "exp_outer_metrics.csv"),
    OUTPUT_DIR=os.path.join(HERE, "OUTPUT_fixed", "comparison_heatsink_outer"),
    COMP_CSV=os.path.join(HERE, "OUTPUT_fixed", "comparison_heatsink_outer",
                          "comparison_minmax_vs_energy.csv"),
    T_ON_SIM=5.0, T_END_SIM=30.0, T_END_EXP=25.0,
    I0_MATCH_TOL=0.02,
    SETUPS=[(os.path.join(HERE, "Castell22C_7Oct30mm", "Excel files"),
             os.path.join(HERE, "Castell22C_7Oct30mm", "VCs_30mm")),
            (os.path.join(HERE, "Castell22C_7Oct40mm", "Excel files"),
             os.path.join(HERE, "Castell22C_7Oct40mm", "VCs_40mm"))],
    FRAME_RATE_HZ=1.0, BG_FRAC=0.05,
    FIGURE_DPI=200,
    # mm/px per IR setup (must match processing_experimental_outer_surface.py)
    MM_PER_PIXEL={"30mm": 0.0999, "40mm": 0.1119},
    COOL_THRESHOLDS=[0.5, 1.0, 2.0, 5.0],
)
matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 34, "axes.titlesize": 40,
    "axes.labelsize": 36, "xtick.labelsize": 36, "ytick.labelsize": 36,
    "legend.fontsize": 36, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 150, "savefig.dpi": 300, "legend.frameon": False})
CAVEAT = ("SIM: inner-surface FIELD (Config_heatsink_surf.csv, `surf_*` columns) vs "
          "EXP: outer-surface IR frame min/max (rel.) | drive-window energy only. "
          "\nHOUSE RULE: direct probes are NEVER compared with experimental data - the "
          "probe values are kept in the CSV as `*_sim_probe` reference columns only "
          "and are excluded from every error column.")
index_entries = []

def save_figure(fig, name):
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    try:
        bb = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        print("      [size] %s tight-inches %.1f x %.1f -> px %d x %d" % (
            name, bb.bounds[2], bb.bounds[3],
            int(bb.bounds[2] * CONFIG["FIGURE_DPI"]), int(bb.bounds[3] * CONFIG["FIGURE_DPI"])))
    except Exception as e:
        print("      [size] %s bbox-check failed: %s" % (name, e))
    dpi = CONFIG["FIGURE_DPI"]
    try:
        fig.savefig(os.path.join(CONFIG["OUTPUT_DIR"], name),
                    dpi=dpi, bbox_inches="tight")
    except OSError:
        dpi = 110
        print("      [retry] %s at dpi=%d" % (name, dpi))
        fig.savefig(os.path.join(CONFIG["OUTPUT_DIR"], name),
                    dpi=dpi, bbox_inches="tight")
    print("      saved %s" % name); index_entries.append(name); plt.close(fig)

from sim_loaders import load_accumulated_probes, load_area_series
from simulated_data import CFG as HCFG

sim_metrics = pd.read_csv(CONFIG["SIM_METRICS"])
sim_probes = load_accumulated_probes(HCFG["PROBES_CSV"], HCFG["T_AMB_C"])
exp_metrics = pd.read_csv(CONFIG["EXP_METRICS"])
SIM_SURF_CSV = HCFG["SURF_CSV"]          # Config_heatsink_surf.csv (area cache source)

# --- SIM surface traces (the only simulated data compared with the EXP IR) ---
sim_surf_traces = None
if os.path.exists(CONFIG["SIM_SURF_TRACES"]):
    sim_surf_traces = pd.read_csv(CONFIG["SIM_SURF_TRACES"])
    print("SIM surface traces: %s (%d rows)"
          % (CONFIG["SIM_SURF_TRACES"], len(sim_surf_traces)))
else:
    print("[warn] missing %s - re-run reading_comsol_heatsink.py"
          % CONFIG["SIM_SURF_TRACES"])


def sim_surface_trace(i0):
    """(t into pulse, dT_min, dT_max, dT_mean) of the SIM inner-surface field.

    This replaces the point probes on the SIM side of every SIM-vs-EXP figure:
    a surface field is the simulated counterpart of the experimental surface.
    """
    if sim_surf_traces is None:
        return None
    sub = sim_surf_traces[np.isclose(sim_surf_traces["I0_A"], i0, atol=1e-3)]
    if sub.empty:
        return None
    sub = sub.sort_values("time_s")
    return (sub["time_into_pulse_s"].to_numpy(float),
            sub["dT_min_K"].to_numpy(float),
            sub["dT_max_K"].to_numpy(float),
            sub["dT_mean_K"].to_numpy(float))


def sim_area_evolution(i0):
    """SIM threshold-crossing cooled/heated area (mm^2) over time, per threshold.

    Uses the cached Delaunay area series (OUTPUT/_surface_area_cache/*.npz), so
    no re-stream of the surface export is needed.  Returns
    (t_into_pulse, {thr: (cool_mm2, heat_mm2)}) or None when the cache is absent.
    """
    series = load_area_series(SIM_SURF_CSV, i0)
    if series is None or series.get("tri_T") is None:
        return None
    times = series["times"] - CONFIG["T_ON_SIM"]       # -> time into pulse
    area = series["tri_area"]
    out = {}
    for thr in CONFIG["COOL_THRESHOLDS"]:
        cool = np.array([area[row < -thr].sum() for row in series["tri_T"]])
        heat = np.array([area[row > thr].sum() for row in series["tri_T"]])
        out[thr] = (cool, heat)
    return times, out


def sim_thermal_load(i0):
    """SIM integrated thermal load traces: (t_into_pulse, {cool, heat}) in K*mm2.

    cool(t) = sum |dT|*dA over triangles with dT < 0 (per timestep);
    heat(t) = same over dT > 0.  Uses the cached Delaunay area series.
    """
    series = load_area_series(SIM_SURF_CSV, i0)
    if series is None or series.get("tri_T") is None:
        return None
    times = series["times"] - CONFIG["T_ON_SIM"]
    area = series["tri_area"]
    cool = np.array([np.abs(row[row < 0] * area[row < 0]).sum()
                     for row in series["tri_T"]])
    heat = np.array([np.abs(row[row > 0] * area[row > 0]).sum()
                     for row in series["tri_T"]])
    return times, {"cool": cool, "heat": heat}


def exp_thermal_load(i0):
    """EXP integrated thermal load: mean +/- SD over repeats (K*mm2 traces).

    Per repeat: cool(t) = sum |dT|*px_area over px with dT < 0 (frame-0 +
    border-bg subtracted stack); heat(t) over dT > 0.  Returns
    (t_common, cool_mean, cool_sd, heat_mean, heat_sd) or None.
    """
    tns = [tn for tn in vc_info
           if abs(_exp_i0_of(tn) - i0) <= CONFIG["I0_MATCH_TOL"]]
    if not tns:
        return None
    runs = []
    for tn in tns:
        r = exp_run_traces(xf_tests[tn])
        if r is None:
            continue
        t, _dmin, _dmax, _areas = r
        # rebuild dT stack (same recipe as exp_run_traces)
        lst = sorted(xf_tests[tn])
        ref = cached_frame(dict(lst)[0])
        h, w = ref.shape
        bh, bw = max(int(h * CONFIG["BG_FRAC"]), 1), max(int(w * CONFIG["BG_FRAC"]), 1)
        border = np.zeros((h, w), bool)
        border[:bh, :] = True; border[-bh:, :] = True
        border[:, :bw] = True; border[:, -bw:] = True
        idxs = sorted(i for i, _p in lst if i != 0)
        d = dict(lst)
        stk = np.array([cached_frame(d[i]) - ref for i in idxs], float)
        bg = stk[:, border].mean(axis=1)
        stk = stk - (bg - bg[0])[:, None, None]
        mm = _mm_per_px_of(lst[0][1])
        px = mm ** 2
        cool_tr = np.array([np.abs(fr[fr < 0]).sum() * px for fr in stk])
        heat_tr = np.array([np.abs(fr[fr > 0]).sum() * px for fr in stk])
        runs.append((t, cool_tr, heat_tr))
    if not runs:
        return None
    tu = np.unique(np.concatenate([r[0] for r in runs]))
    cm = np.mean([np.interp(tu, r[0], r[1]) for r in runs], axis=0)
    hm = np.mean([np.interp(tu, r[0], r[2]) for r in runs], axis=0)
    cs = (np.std([np.interp(tu, r[0], r[1]) for r in runs], axis=0)
          if len(runs) > 1 else np.zeros_like(tu))
    hs = (np.std([np.interp(tu, r[0], r[2]) for r in runs], axis=0)
          if len(runs) > 1 else np.zeros_like(tu))
    return tu, cm, cs, hm, hs


def _crossing_time(t, y, frac=0.5):
    """First rising-edge time where y >= frac * peak (0 when already above)."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    m = np.isfinite(y)
    if not m.any():
        return np.nan
    pk = float(np.nanmax(y[m]))
    if not np.isfinite(pk) or pk <= 0:
        return np.nan
    tgt = frac * pk
    if y[m][0] >= tgt:
        return float(t[m][0])
    idx = np.where(m & (y >= tgt))[0]
    return float(t[idx[0]]) if idx.size else np.nan


def _sv(row, key):
    """Value of a `surf_*` metric of one SIM row (NaN when unavailable)."""
    v = row.get(key, np.nan) if hasattr(row, "get") else np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def sv(row, key, probe_key=None):
    """`surf_*` metric; falls back to NaN (never to a probe value)."""
    v = _sv(row, key)
    if np.isfinite(v):
        return v
    return _sv(row, probe_key) if probe_key else np.nan

def load_vc(vp):
    # Same recipe as processing_experimental_outer_surface.load_vc: named
    # columns (pulse,phase,sample,time_ms,voltage_V,current_A), phase-based
    # ON window, energy = sum V*I*dt while active.
    d = pd.read_csv(vp)
    d.columns = [str(c).strip() for c in d.columns]
    cm = {}
    for c in d.columns:
        cl = c.lower()
        if cl == "time_ms":
            cm[c] = "time_ms"
        elif "time" in cl:
            cm[c] = "time"
        elif "current" in cl:
            cm[c] = "current"
        elif "voltage" in cl:
            cm[c] = "voltage"
        elif "phase" in cl:
            cm[c] = "phase"
    d = d.rename(columns=cm)
    if "time_ms" in d.columns and "time" not in d.columns:
        d["time"] = d["time_ms"]
    for c in ["time", "current", "voltage"]:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["time", "current", "voltage"]).sort_values("time")
    span = float(d["time"].iloc[-1] - d["time"].iloc[0])
    d["time_s"] = (d["time"] / 1000.0) if span > 5000 else d["time"].astype(float)
    d["time_s"] = d["time_s"] - d["time_s"].iloc[0]
    t = d["time_s"].to_numpy(float)
    i = d["current"].to_numpy(float)
    if "phase" in d.columns:
        on = d["phase"].astype(str).str.upper().eq("ON").to_numpy()
    else:
        on = np.zeros(len(d), bool)
    if on.any():
        t_on, t_end = float(t[on][0]), float(t[on][-1])
        i0 = round(float(i[on].max()), 2)
    else:
        pk = float(np.nanmax(i))
        m = i >= 0.90 * pk
        t_on = float(t[m][0]) if m.any() else 0.0
        t_end = float(t[m][-1]) if m.any() else CONFIG["T_END_EXP"]
        i0 = round(pk, 2)
    act = i >= 0.90 * i0
    win = t <= 60.0
    dt = np.diff(t, prepend=t[0])
    en = float(np.sum((d["voltage"].to_numpy(float) * i)[win & act] * dt[win & act]))
    v_on = d["voltage"].to_numpy(float)[on] if on.any() else d["voltage"].to_numpy(float)[act]
    v_peak = float(np.nanmax(v_on)) if len(v_on) else np.nan
    return dict(i0=i0, t_on=t_on, t_end=t_end, energy_J=en, v_peak_V=v_peak, df=d)

def _clean_frame(path):
    # Proven cleaner from the EXP pipeline: handles "24,9Â°C" locale strings.
    df = pd.read_excel(path, header=None, engine="openpyxl")
    return (df.apply(lambda c: c.astype(str)
             .str.replace("Â°C", "", regex=False).str.replace("Â°", "", regex=False)
             .str.replace(",", ".", regex=False).str.strip())
            .values.astype(float))

_frame_npz_cache = {}
def cached_frame(path):
    cdir = os.path.join(HERE, "OUTPUT", "_frame_cache_7oct")
    os.makedirs(cdir, exist_ok=True)
    key = ("%s_%s" % (os.path.basename(os.path.dirname(os.path.dirname(path))),
                      os.path.basename(path))).replace(".xlsx", ".npz")
    cp = os.path.join(cdir, key)
    if cp in _frame_npz_cache:
        return _frame_npz_cache[cp]
    if os.path.exists(cp):
        try:
            a = np.load(cp)["a"]
            _frame_npz_cache[cp] = a
            return a
        except Exception:
            pass
    a = _clean_frame(path)
    try:
        np.savez_compressed(cp, a=a)
    except Exception:
        pass
    _frame_npz_cache[cp] = a
    return a

def _parse_xf(fn):
    import re as _re
    m = _re.match(r"^(.+?)_(\d+)\.xlsx$", fn, _re.IGNORECASE)
    return (m.group(1), int(m.group(2))) if m else (None, None)

def _mm_per_px_of(path):
    """mm/px for an IR frame path, from the setup folder name (30mm/40mm)."""
    for key, val in CONFIG["MM_PER_PIXEL"].items():
        if key in path:
            return val
    return 0.10


def exp_run_traces(xf_list):
    """Full per-test stack: xf_list = [(idx, path)] of one TESTnnn; returns
    (t, dmin, dmax, areas) with frame-0 baseline + border-bg subtraction.

    `areas` = {threshold: (cooled_mm2_trace, heated_mm2_trace)} — the amount of
    surface area crossing each temperature threshold at every frame (IR px count
    x (mm/px)^2), mirroring the SIM Delaunay area series."""
    xf_list = sorted(xf_list)
    if 0 not in dict(xf_list):
        return None
    ref = cached_frame(dict(xf_list)[0])
    h, w = ref.shape
    bh, bw = max(int(h * CONFIG["BG_FRAC"]), 1), max(int(w * CONFIG["BG_FRAC"]), 1)
    border = np.zeros((h, w), bool)
    border[:bh, :] = True; border[-bh:, :] = True
    border[:, :bw] = True; border[:, -bw:] = True
    idxs = sorted(i for i, _ in xf_list if i != 0)
    d = dict(xf_list)
    stack = np.array([cached_frame(d[i]) - ref for i in idxs], float)
    bg = stack[:, border].mean(axis=1)
    rel = stack - (bg - bg[0])[:, None, None] if len(bg) else stack
    # note: EXP file uses mean (not median) for bg; match it
    t = np.asarray(idxs, float) / CONFIG["FRAME_RATE_HZ"]
    flat = rel.reshape(len(t), -1)
    mm = _mm_per_px_of(xf_list[0][1])
    px = mm ** 2
    areas = {}
    for thr in CONFIG["COOL_THRESHOLDS"]:
        cool = (rel <= -thr).sum(axis=(1, 2)) * px      # cooled area over time
        heat = (rel >= thr).sum(axis=(1, 2)) * px       # heated area over time
        areas[thr] = (cool.astype(float), heat.astype(float))
    return t, np.nanmin(flat, axis=1), np.nanmax(flat, axis=1), areas

# group xlsx frames per TESTnnn + pair with VC logs by test name
vc_info, xf_tests = {}, {}
for xdir, vdir, _label in [
        (os.path.join(HERE, "Castell22C_7Oct30mm", "Excel files"),
         os.path.join(HERE, "Castell22C_7Oct30mm", "VCs_30mm"), "30mm"),
        (os.path.join(HERE, "Castell22C_7Oct40mm", "Excel files"),
         os.path.join(HERE, "Castell22C_7Oct40mm", "VCs_40mm"), "40mm")]:
    CONFIG["SETUPS"] = CONFIG.get("SETUPS", [])
    tests = {}
    if os.path.isdir(xdir):
        for fn in sorted(os.listdir(xdir)):
            if not fn.lower().endswith(".xlsx"):
                continue
            name, idx = _parse_xf(fn)
            if name is None:
                continue
            key = "%s / %s" % (_label, name)
            tests.setdefault(key, []).append((idx, os.path.join(xdir, fn)))
    vcs = {}
    if os.path.isdir(vdir):
        for fn in sorted(os.listdir(vdir)):
            if fn.lower().endswith(".csv"):
                vcs[os.path.splitext(fn)[0].upper()] = os.path.join(vdir, fn)
    for tn, lst in tests.items():
        bare = tn.split(" / ", 1)[-1]
        vp = vcs.get(bare.upper())
        if vp is None:
            for k, p in vcs.items():
                if bare.upper() in k or k in bare.upper():
                    vp = p
                    break
        if vp is None:
            continue
        try:
            vc_info[tn] = load_vc(vp)
            xf_tests[tn] = lst
        except Exception:
            continue

def _exp_i0_of(tn):
    v = vc_info.get(tn)
    return (v["i0"] if isinstance(v, dict) else v) if v is not None else np.nan

def exp_mean_traces(i0):
    tns = [tn for tn in vc_info
           if abs(_exp_i0_of(tn) - i0) <= CONFIG["I0_MATCH_TOL"]]
    if not tns:
        return None
    runs = []
    for tn in tns:
        r = exp_run_traces(xf_tests[tn])
        if r is not None:
            runs.append(r)
    if not runs:
        return None
    tu = np.unique(np.concatenate([r[0] for r in runs]))
    cold = np.mean([np.interp(tu, r[0], r[1]) for r in runs], axis=0)
    hot = np.mean([np.interp(tu, r[0], r[2]) for r in runs], axis=0)
    # spread across repeats (for error bands): std of the interpolated runs
    cs = np.std([np.interp(tu, r[0], r[1]) for r in runs], axis=0) if len(runs) > 1 else np.zeros_like(tu)
    hs = np.std([np.interp(tu, r[0], r[2]) for r in runs], axis=0) if len(runs) > 1 else np.zeros_like(tu)
    # mean threshold-crossing area over time (+ SD across repeats), per threshold
    area_mean, area_sd = {}, {}
    for thr in CONFIG["COOL_THRESHOLDS"]:
        for j, side in enumerate(("cool", "heat")):
            per_run = [np.interp(tu, r[0], r[3][thr][j]) for r in runs]
            m = np.mean(per_run, axis=0)
            s = (np.std(per_run, axis=0) if len(runs) > 1 else np.zeros_like(tu))
            area_mean.setdefault(thr, {})[side] = m
            area_sd.setdefault(thr, {})[side] = s
    return tu, cold, hot, tns, cs, hs, area_mean, area_sd

def exp_vc_overlay(i0):
    """Mean V(t)/I(t) of the EXP runs at this I0 (common t=0 at pulse start)."""
    tns = [tn for tn in vc_info
           if abs(_exp_i0_of(tn) - i0) <= CONFIG["I0_MATCH_TOL"]]
    if not tns:
        return None
    vv, ii = [], []
    tmax = 0.0
    dfs = []
    for tn in tns:
        d = vc_info[tn]["df"]
        t = d["time_s"].to_numpy(float)
        tmax = max(tmax, float(t[-1]))
        dfs.append(d)
    tu = np.arange(0.0, min(tmax, 60.0) + 1e-9, 0.25)
    for d in dfs:
        t = d["time_s"].to_numpy(float)
        vv.append(np.interp(tu, t, d["voltage"].to_numpy(float)))
        ii.append(np.interp(tu, t, d["current"].to_numpy(float)))
    return tu, np.mean(vv, axis=0), np.mean(ii, axis=0), tns

pairs = []
for _, rs in sim_metrics.iterrows():
    d = np.abs(exp_metrics["I0 (A)"] - rs["I0_A"])
    j = int(np.argmin(d))
    if d.iloc[j] <= CONFIG["I0_MATCH_TOL"]:
        pairs.append((float(rs["I0_A"]), rs, exp_metrics.iloc[j]))
print("Matched I0 pairs (tol %.2f A): %s" % (CONFIG["I0_MATCH_TOL"], [p[0] for p in pairs]))
SURF_NEEDED = ["surf_dT_min_peak_K", "surf_dT_max_peak_K", "surf_dT_min_end_K",
              "surf_dT_max_end_K", "surf_t_peak_cool_s", "surf_overshoot_cold_K"]
if not all(c in sim_metrics.columns for c in SURF_NEEDED):
    raise SystemExit("OUTPUT/heatsink_metrics.csv has no `surf_*` columns. Probes are "
                     "never compared with experimental data, so re-run "
                     "`python reading_comsol_heatsink.py` (CFG['RUN_SURF_METRICS']).")
if "--probe" in sys.argv:
    print("PROBE OK"); sys.exit(0)
if not pairs:
    raise SystemExit("No I0 overlap within tolerance.")

# Spatial-area comparison columns.  SIM tag "05" pairs with EXP tag "0p5";
# the other thresholds share the same tag.  Both sides store mm^2 (SIM has no
# unit suffix, EXP does).  Methods differ (Delaunay mesh vs IR px count) so
# these are shape/evolution comparisons, not absolute ones.
def _area_pairs(rs, re_):
    out = {}
    for thr in CONFIG["COOL_THRESHOLDS"]:
        sim_tag = ("%g" % thr).replace(".", "")     # 0.5 -> "05"
        exp_tag = ("%g" % thr).replace(".", "p")    # 0.5 -> "0p5"
        for kind, side in (("cool", "cold"), ("heat", "hot")):
            for when in ("end", "peak"):
                key = "area_%s_%s_%s" % (kind, exp_tag, when)
                sim_col = "area_%s_%s_%s" % (kind, sim_tag, when)
                exp_col = "area_%s_%s_%s (mm2)" % (kind, exp_tag, when)
                out["%s_sim (mm2)" % key] = (float(rs[sim_col])
                                             if sim_col in rs.index else np.nan)
                out["%s_exp (mm2)" % key] = (float(re_[exp_col])
                                             if exp_col in re_.index else np.nan)
    return out

# per-current derived traces for the new metric families (cached per i0)
_new_cache = {}
def _new_metrics(i0):
    if i0 in _new_cache:
        return _new_cache[i0]
    d = {"load_cool_peak_sim": np.nan, "load_cool_end_sim": np.nan,
         "load_heat_peak_sim": np.nan, "load_heat_end_sim": np.nan,
         "load_cool_peak_exp": np.nan, "load_cool_peak_exp_sd": np.nan,
         "load_cool_end_exp": np.nan, "load_cool_end_exp_sd": np.nan,
         "load_heat_peak_exp": np.nan, "load_heat_peak_exp_sd": np.nan,
         "load_heat_end_exp": np.nan, "load_heat_end_exp_sd": np.nan,
         "t_cool_2_50_sim": np.nan, "t_cool_2_100_sim": np.nan,
         "t_heat_2_50_sim": np.nan, "t_heat_2_100_sim": np.nan,
         "t_cool_2_50_exp": np.nan, "t_cool_2_50_exp_sd": np.nan,
         "t_cool_2_100_exp": np.nan, "t_cool_2_100_exp_sd": np.nan,
         "t_heat_2_50_exp": np.nan, "t_heat_2_50_exp_sd": np.nan,
         "t_heat_2_100_exp": np.nan, "t_heat_2_100_exp_sd": np.nan}
    sl = sim_thermal_load(i0)
    if sl is not None:
        ts, ld = sl
        for side, pk, en in (("cool", "load_cool_peak_sim", "load_cool_end_sim"),
                             ("heat", "load_heat_peak_sim", "load_heat_end_sim")):
            y = ld[side]
            d[pk] = float(np.nanmax(y)) if np.isfinite(y).any() else np.nan
            ie = int(np.argmin(np.abs(ts - CONFIG["T_END_EXP"])))
            d[en] = float(y[ie]) if len(y) else np.nan
    el = exp_thermal_load(i0)
    if el is not None:
        tu, cm, cs, hm, hs = el
        ie = int(np.argmin(np.abs(tu - CONFIG["T_END_EXP"])))
        d["load_cool_peak_exp"] = float(np.nanmax(cm))
        d["load_cool_peak_exp_sd"] = float(cs[int(np.nanargmax(cm))])
        d["load_cool_end_exp"] = float(cm[ie])
        d["load_cool_end_exp_sd"] = float(cs[ie])
        d["load_heat_peak_exp"] = float(np.nanmax(hm))
        d["load_heat_peak_exp_sd"] = float(hs[int(np.nanargmax(hm))])
        d["load_heat_end_exp"] = float(hm[ie])
        d["load_heat_end_exp_sd"] = float(hs[ie])
    # area timing at 2 K (SIM traces + EXP per-repeat traces for SD)
    sa = sim_area_evolution(i0)
    if sa is not None:
        ts, areas = sa
        for side, j in (("cool", 0), ("heat", 1)):
            y = areas[2.0][j]
            pk = float(np.nanmax(y)) if np.isfinite(y).any() else np.nan
            d["t_%s_2_50_sim" % side] = _crossing_time(ts, y, 0.5)
            d["t_%s_2_100_sim" % side] = (
                float(ts[int(np.nanargmax(y))]) if np.isfinite(pk) else np.nan)
    em = exp_mean_traces(i0)
    if em is not None:
        tu, _c, _h, tns, _cs, _hs, eam, eas = em
        for side in ("cool", "heat"):
            ym = eam[2.0][side]
            d["t_%s_2_50_exp" % side] = _crossing_time(tu, ym, 0.5)
            d["t_%s_2_100_exp" % side] = (
                float(tu[int(np.nanargmax(ym))]) if np.isfinite(ym).any() else np.nan)
            # SD across per-repeat crossing times
            vals50, vals100 = [], []
            for tn in tns:
                r = exp_run_traces(xf_tests[tn])
                if r is None:
                    continue
                t, _dm, _dx, ar = r
                # per-repeat 2 K area traces
                mm = _mm_per_px_of(xf_tests[tn][0][1])
                _px = mm ** 2
                _lst = sorted(xf_tests[tn])
                _ref = cached_frame(dict(_lst)[0])
                _idxs = sorted(i for i, _p in _lst if i != 0)
                _dd = dict(_lst)
                _stk = np.array([cached_frame(_dd[i]) - _ref for i in _idxs], float)
                _bgb = _stk[:, :max(int(_stk.shape[1] * CONFIG["BG_FRAC"]), 1),
                            :].mean(axis=(1, 2))
                _stk = _stk - (_bgb - _bgb[0])[:, None, None]
                _tt = np.asarray(_idxs, float) / CONFIG["FRAME_RATE_HZ"]
                _cool = (np.array([((fr <= -2.0).sum() * _px) for fr in _stk]))
                _heat = (np.array([((fr >= 2.0).sum() * _px) for fr in _stk]))
                _y = _cool if side == "cool" else _heat
                vals50.append(_crossing_time(_tt, _y, 0.5))
                vals100.append(float(_tt[int(np.nanargmax(_y))])
                               if np.isfinite(_y).any() else np.nan)
            v50 = np.array(vals50, float); v50 = v50[np.isfinite(v50)]
            v100 = np.array(vals100, float); v100 = v100[np.isfinite(v100)]
            d["t_%s_2_50_exp_sd" % side] = float(np.std(v50, ddof=1)) if v50.size > 1 else 0.0
            d["t_%s_2_100_exp_sd" % side] = float(np.std(v100, ddof=1)) if v100.size > 1 else 0.0
    _new_cache[i0] = d
    return d

rows = []
for i0, rs, re_ in pairs:
    _nm = _new_metrics(i0)
    # equivalent radius from peak 2 K areas: r = sqrt(A/pi)
    def _r(a):
        return float(np.sqrt(a / np.pi)) if np.isfinite(a) and a > 0 else (0.0 if a == 0 else np.nan)
    _rc_s = _r(float(rs["area_cool_2_peak"]) if "area_cool_2_peak" in rs.index else np.nan)
    _rh_s = _r(float(rs["area_heat_2_peak"]) if "area_heat_2_peak" in rs.index else np.nan)
    _rc_e = _r(float(re_.get("area_cool_2_peak (mm2)", np.nan)))
    _rh_e = _r(float(re_.get("area_heat_2_peak (mm2)", np.nan)))
    def _r_sd(a, a_sd):
        if not (np.isfinite(a) and np.isfinite(a_sd)) or a <= 0:
            return 0.0 if a == 0 else np.nan
        return float(a_sd / (2.0 * np.sqrt(np.pi * a)))
    rows.append({
        "I0 (A)": i0,
        # --- energy ---
        "E_sim_drive (J)": rs["W_drive_J"], "E_exp (J)": re_["E_input (J)"],
        "E_sim_full (J)": rs["W_full_J"],
        "V1_peak_sim (V)": rs["V1_peak_V"],
        # --- temperature: peaks in drive window (SIM = inner-surface field) ---
        "dTmin_peak_sim (C)": sv(rs, "surf_dT_min_peak_K"),
        "dTmin_peak_exp (C)": re_["dT_cold_peak (C)"],
        "dTmax_peak_sim (C)": sv(rs, "surf_dT_max_peak_K"),
        "dTmax_peak_exp (C)": re_["dT_hot_peak (C)"],
        # --- temperature: end-of-pulse (SIM = inner-surface field) ---
        "dTmin_end_sim (C)": sv(rs, "surf_dT_min_end_K"),
        "dTmin_end_exp (C)": re_.get("dT_cold_end (C)", np.nan),
        "dTmax_end_sim (C)": sv(rs, "surf_dT_max_end_K"),
        "dTmax_end_exp (C)": re_.get("dT_hot_end (C)", np.nan),
        # --- temporal (SIM = inner-surface field) ---
        "t_peak_cool_sim (s)": sv(rs, "surf_t_peak_cool_s") - CONFIG["T_ON_SIM"],
        "t_peak_cool_exp (s)": re_["t_peak_cool (s)"],
        "tau_cold_exp (s)": re_["tau_cold (s)"],
        "t_cool63_sim (s)": sv(rs, "surf_t_cool63_s") - CONFIG["T_ON_SIM"],
        # never reached inside the 60 s window for any current -> kept, NaN, and
        # the fraction-of-drop recovery times below are the usable substitute.
        "t_back_normal_sim (s)": (sv(rs, "surf_t_back_normal_s") - CONFIG["T_END_SIM"]
                                  if np.isfinite(sv(rs, "surf_t_back_normal_s")) else np.nan),
        "t_recover63_sim (s)": (sv(rs, "surf_t_recover63_s") - CONFIG["T_END_SIM"]
                                 if np.isfinite(sv(rs, "surf_t_recover63_s")) else np.nan),
        "t_recover90_sim (s)": (sv(rs, "surf_t_recover90_s") - CONFIG["T_END_SIM"]
                                 if np.isfinite(sv(rs, "surf_t_recover90_s")) else np.nan),
        "recovered_at_60s_sim (%)": 100.0 * sv(rs, "surf_recovery_frac_end"),
        "overshoot_sim (C)": sv(rs, "surf_overshoot_cold_K"),
        "overshoot_exp (C)": re_["overshoot_cold (C)"],
        # --- surface-field shape (SIM only, no EXP counterpart) ---
        "spread_peak_sim (C)": sv(rs, "surf_spread_peak_K"),
        "spread_end_sim (C)": sv(rs, "surf_spread_end_K"),
        "std_end_sim (C)": sv(rs, "surf_std_end_K"),
        "dose_sim (Ks)": sv(rs, "surf_dose_Ks"),
        # --- efficiency (SIM surface |dT| over SIM energy) ---
        "K_per_J_sim": sv(rs, "surf_K_per_J"), "K_per_J_exp": re_["K_per_J"],
        # --- area per joule at 2 K (peak area / drive-window energy) ---
        "area_cool_2_per_J_sim (mm2/J)": (
            float(rs["area_cool_2_peak"]) / float(rs["W_drive_J"])
            if "area_cool_2_peak" in rs.index and np.isfinite(float(rs["W_drive_J"]))
            and float(rs["W_drive_J"]) else np.nan),
        "area_cool_2_per_J_exp (mm2/J)": re_.get("area_cool_2_per_J (mm2/J)", np.nan),
        "area_cool_2_per_J_exp_sd (mm2/J)": re_.get("area_cool_2_per_J_sd (mm2/J)", np.nan),
        "area_heat_2_per_J_sim (mm2/J)": (
            float(rs["area_heat_2_peak"]) / float(rs["W_drive_J"])
            if "area_heat_2_peak" in rs.index and np.isfinite(float(rs["W_drive_J"]))
            and float(rs["W_drive_J"]) else np.nan),
        "area_heat_2_per_J_exp (mm2/J)": re_.get("area_heat_2_per_J (mm2/J)", np.nan),
        "area_heat_2_per_J_exp_sd (mm2/J)": re_.get("area_heat_2_per_J_sd (mm2/J)", np.nan),
        # --- integrated thermal load ∫|dT|dA, peak + end (K*mm2) ---
        "load_cool_peak_sim (Kmm2)": _nm["load_cool_peak_sim"],
        "load_cool_peak_exp (Kmm2)": _nm["load_cool_peak_exp"],
        "load_cool_peak_exp_sd (Kmm2)": _nm["load_cool_peak_exp_sd"],
        "load_cool_end_sim (Kmm2)": _nm["load_cool_end_sim"],
        "load_cool_end_exp (Kmm2)": _nm["load_cool_end_exp"],
        "load_cool_end_exp_sd (Kmm2)": _nm["load_cool_end_exp_sd"],
        "load_heat_peak_sim (Kmm2)": _nm["load_heat_peak_sim"],
        "load_heat_peak_exp (Kmm2)": _nm["load_heat_peak_exp"],
        "load_heat_peak_exp_sd (Kmm2)": _nm["load_heat_peak_exp_sd"],
        "load_heat_end_sim (Kmm2)": _nm["load_heat_end_sim"],
        "load_heat_end_exp (Kmm2)": _nm["load_heat_end_exp"],
        "load_heat_end_exp_sd (Kmm2)": _nm["load_heat_end_exp_sd"],
        # --- equivalent radius of 2 K peak regions, r = sqrt(A/pi) (mm) ---
        "r_cool_2_peak_sim (mm)": _rc_s,
        "r_cool_2_peak_exp (mm)": _rc_e,
        "r_cool_2_peak_exp_sd (mm)": _r_sd(
            float(re_.get("area_cool_2_peak (mm2)", np.nan)),
            float(re_.get("area_cool_2_peak_sd (mm2)", np.nan))),
        "r_heat_2_peak_sim (mm)": _rh_s,
        "r_heat_2_peak_exp (mm)": _rh_e,
        "r_heat_2_peak_exp_sd (mm)": _r_sd(
            float(re_.get("area_heat_2_peak (mm2)", np.nan)),
            float(re_.get("area_heat_2_peak_sd (mm2)", np.nan))),
        # --- time to 50 % / 100 % of 2 K peak area, into-pulse (s) ---
        "t_cool_2_50_sim (s)": _nm["t_cool_2_50_sim"],
        "t_cool_2_50_exp (s)": _nm["t_cool_2_50_exp"],
        "t_cool_2_50_exp_sd (s)": _nm["t_cool_2_50_exp_sd"],
        "t_cool_2_100_sim (s)": _nm["t_cool_2_100_sim"],
        "t_cool_2_100_exp (s)": _nm["t_cool_2_100_exp"],
        "t_cool_2_100_exp_sd (s)": _nm["t_cool_2_100_exp_sd"],
        "t_heat_2_50_sim (s)": _nm["t_heat_2_50_sim"],
        "t_heat_2_50_exp (s)": _nm["t_heat_2_50_exp"],
        "t_heat_2_50_exp_sd (s)": _nm["t_heat_2_50_exp_sd"],
        "t_heat_2_100_sim (s)": _nm["t_heat_2_100_sim"],
        "t_heat_2_100_exp (s)": _nm["t_heat_2_100_exp"],
        "t_heat_2_100_exp_sd (s)": _nm["t_heat_2_100_exp_sd"],
        # --- spatial area (SIM Delaunay mesh vs EXP IR px; methods differ) ---
        # SIM tag "05" pairs with EXP tag "0p5"; the other tags match directly.
        **_area_pairs(rs, re_),
        # --- PROBE REFERENCE ONLY - NEVER COMPARED WITH EXPERIMENTAL DATA ---
        "dTmin_peak_sim_probe (C)": _sv(rs, "dT_min_peak_K"),
        "dTmax_peak_sim_probe (C)": _sv(rs, "dT_max_peak_K"),
        "dTmin_end_sim_probe (C)": _sv(rs, "dT_min_end_K"),
        "dTmax_end_sim_probe (C)": _sv(rs, "dT_max_end_K"),
        "K_per_J_sim_probe": _sv(rs, "K_per_J"),
        "t_cool63_sim_probe (s)": _sv(rs, "t_cool63_s"),
    })
comp = pd.DataFrame(rows)
EXP_PAIRS = [("min_peak", "dTmin_peak_sim (C)", "dTmin_peak_exp (C)"),
                  ("max_peak", "dTmax_peak_sim (C)", "dTmax_peak_exp (C)"),
                  ("min_end", "dTmin_end_sim (C)", "dTmin_end_exp (C)"),
                  ("max_end", "dTmax_end_sim (C)", "dTmax_end_exp (C)"),
                  ("energy", "E_sim_drive (J)", "E_exp (J)"),
                  ("K_per_J", "K_per_J_sim", "K_per_J_exp"),
                  ("area_cool_2_per_J", "area_cool_2_per_J_sim (mm2/J)",
                   "area_cool_2_per_J_exp (mm2/J)"),
                  ("area_heat_2_per_J", "area_heat_2_per_J_sim (mm2/J)",
                   "area_heat_2_per_J_exp (mm2/J)"),
                  ("load_cool_peak", "load_cool_peak_sim (Kmm2)",
                   "load_cool_peak_exp (Kmm2)"),
                  ("load_cool_end", "load_cool_end_sim (Kmm2)",
                   "load_cool_end_exp (Kmm2)"),
                  ("load_heat_peak", "load_heat_peak_sim (Kmm2)",
                   "load_heat_peak_exp (Kmm2)"),
                  ("load_heat_end", "load_heat_end_sim (Kmm2)",
                   "load_heat_end_exp (Kmm2)"),
                  ("r_cool_2_peak", "r_cool_2_peak_sim (mm)",
                   "r_cool_2_peak_exp (mm)"),
                  ("r_heat_2_peak", "r_heat_2_peak_sim (mm)",
                   "r_heat_2_peak_exp (mm)"),
                  ("t_cool_2_50", "t_cool_2_50_sim (s)", "t_cool_2_50_exp (s)"),
                  ("t_cool_2_100", "t_cool_2_100_sim (s)", "t_cool_2_100_exp (s)"),
                  ("t_heat_2_50", "t_heat_2_50_sim (s)", "t_heat_2_50_exp (s)"),
                  ("t_heat_2_100", "t_heat_2_100_sim (s)", "t_heat_2_100_exp (s)"),
                  ("t_peak", "t_peak_cool_sim (s)", "t_peak_cool_exp (s)"),
                  ("overshoot", "overshoot_sim (C)", "overshoot_exp (C)")]
# spatial-area pairs (SIM mesh vs EXP IR px): cool/heat x end/peak x threshold
for _thr in CONFIG["COOL_THRESHOLDS"]:
    _et = ("%g" % _thr).replace(".", "p")
    for _kind in ("cool", "heat"):
        for _when in ("end", "peak"):
            EXP_PAIRS.append(("area_%s_%s_%s" % (_kind, _et, _when),
                              "area_%s_%s_%s_sim (mm2)" % (_kind, _et, _when),
                              "area_%s_%s_%s_exp (mm2)" % (_kind, _et, _when)))
# Only the columns that really have an EXP counterpart are "compared"; the
# recovery times / spread / std metrics are SIM-only (no EXP equivalent).
COMPARED = [sc for _k, sc, ec in EXP_PAIRS if sc in comp.columns and ec in comp.columns]
print("SIM columns compared with EXP: %s" % COMPARED)
for k, sc, ec in EXP_PAIRS:
    comp["err_%s" % k] = comp[ec] - comp[sc]
    with np.errstate(divide="ignore", invalid="ignore"):
        comp["relerr_%s (%%)" % k] = 100.0 * (comp[ec] - comp[sc]) / comp[sc].abs()
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
comp.to_csv(CONFIG["COMP_CSV"], index=False)
print(comp.to_string(index=False))

# -----------------------------------------------------------------------------
# Temporal RMSE + full min/max evolution (SIM outer-surface field vs EXP IR)
# -----------------------------------------------------------------------------
# Both traces already live on a common "time into pulse" axis (0 = pulse start).
# (a) per-current RMSE of the min and max traces over the drive window and the
#     full run, with the bias (mean signed error) to separate offset from scatter;
# (b) the full min/max evolution on a uniform 1 Hz grid, so the tables can show
#     how the extrema evolve over time, not only the peak/end snapshots.
# Additive: existing CSVs and figures are unchanged.
TEMPORAL_DIR = CONFIG["OUTPUT_DIR"]
RMSE_CSV = os.path.join(TEMPORAL_DIR, "temporal_rmse.csv")
EVOL_CSV = os.path.join(TEMPORAL_DIR, "temporal_evolution.csv")
DRIVE_END = CONFIG["T_END_EXP"]           # 25 s into pulse (both sides driven)
GRID_STEP_S = 1.0


def _interp(t_src, y_src, t_dst):
    return np.interp(t_dst, t_src, y_src, left=np.nan, right=np.nan)


def _rmse(d, mask):
    v = d[mask & np.isfinite(d)]
    return float(np.sqrt(np.mean(v ** 2))) if v.size else np.nan


def _bias(d, mask):
    v = d[mask & np.isfinite(d)]
    return float(np.mean(v)) if v.size else np.nan


rmse_rows, evol_frames = [], []
for i0, rs, re_ in pairs:
    str_ = sim_surface_trace(i0)
    em = exp_mean_traces(i0)
    if str_ is None or em is None:
        continue
    ts, smin, smax, _smean = str_
    tu, ecold, ehot, tns, ecs, ehs, *_a = em
    # ---- RMSE on the EXP sampling base (SIM interpolated onto it) ----
    sim_min_e = _interp(ts, smin, tu)
    sim_max_e = _interp(ts, smax, tu)
    dmin = ecold - sim_min_e                      # exp - sim
    dmax = ehot - sim_max_e
    drive = (tu >= 0.0) & (tu <= DRIVE_END)
    allpts = np.ones_like(tu, bool)
    rmse_rows.append({
        "I0 (A)": i0, "n_runs": len(tns),
        "rmse_min_drive (K)": _rmse(dmin, drive),
        "rmse_max_drive (K)": _rmse(dmax, drive),
        "rmse_min_full (K)": _rmse(dmin, allpts),
        "rmse_max_full (K)": _rmse(dmax, allpts),
        "bias_min_drive (K)": _bias(dmin, drive),
        "bias_max_drive (K)": _bias(dmax, drive),
        "n_pts_drive": int(np.sum(drive & np.isfinite(dmin))),
        "n_pts_full": int(np.sum(np.isfinite(dmin))),
    })
    # ---- full evolution on a uniform 1 Hz grid ----
    t_max = float(np.nanmin([np.nanmax(ts), np.nanmax(tu)]))
    grid = np.arange(0.0, np.floor(t_max) + 1.0, GRID_STEP_S)
    evol_frames.append(pd.DataFrame({
        "I0 (A)": i0,
        "time_into_pulse_s": grid,
        "dT_min_sim (K)": _interp(ts, smin, grid),
        "dT_min_exp (K)": _interp(tu, ecold, grid),
        "dT_min_exp_sd (K)": _interp(tu, ecs, grid),
        "dT_max_sim (K)": _interp(ts, smax, grid),
        "dT_max_exp (K)": _interp(tu, ehot, grid),
        "dT_max_exp_sd (K)": _interp(tu, ehs, grid),
    }))

if rmse_rows:
    rmse_df = pd.DataFrame(rmse_rows)
    rmse_df.to_csv(RMSE_CSV, index=False)
    print("\nsaved %s (%d currents)" % (RMSE_CSV, len(rmse_df)))
    print(rmse_df.to_string(index=False))
# per-current RMSE lookup, used to annotate the figure legends/titles below
RMSE_BY_I0 = {}
if rmse_rows:
    for _r in rmse_rows:
        RMSE_BY_I0[round(float(_r["I0 (A)"]), 3)] = _r
if evol_frames:
    evol_df = pd.concat(evol_frames, ignore_index=True)
    evol_df["err_min (K)"] = evol_df["dT_min_exp (K)"] - evol_df["dT_min_sim (K)"]
    evol_df["err_max (K)"] = evol_df["dT_max_exp (K)"] - evol_df["dT_max_sim (K)"]
    evol_df.to_csv(EVOL_CSV, index=False)
    print("saved %s (%d rows, %d currents)"
          % (EVOL_CSV, len(evol_df), evol_df["I0 (A)"].nunique()))

# -----------------------------------------------------------------------------
# Spatial threshold-crossing AREA evolution (SIM vs EXP), slide by slide.
# -----------------------------------------------------------------------------
# For every drive current and every temperature threshold we align, on a common
# 1 Hz "time into pulse" grid, the amount of surface area that crosses that
# threshold:
#   SIM  = Delaunay mesh area of triangles with |dT| > thr (cached area series)
#   EXP  = IR pixel count with |dT| > thr, x (mm/px)^2 (mean +/- SD over repeats)
# Writes area_evolution.csv (full traces) + area_rmse.csv (per-current RMSE).
# Methods differ (mesh vs IR px) so this is a shape/evolution comparison.
AREA_EVOL_CSV = os.path.join(TEMPORAL_DIR, "area_evolution.csv")
AREA_RMSE_CSV = os.path.join(TEMPORAL_DIR, "area_rmse.csv")


def _interp_area(t_src, y_src, t_dst):
    return np.interp(t_dst, t_src, y_src, left=np.nan, right=np.nan)


area_frames, area_rmse_rows = [], []
for i0, rs, re_ in pairs:
    sa = sim_area_evolution(i0)
    em = exp_mean_traces(i0)
    if sa is None or em is None:
        continue
    ts, sareas = sa
    tu, _c, _h, tns, _cs, _hs, earea_mean, earea_sd = em
    t_max = float(np.nanmin([np.nanmax(ts), np.nanmax(tu)]))
    grid = np.arange(0.0, np.floor(t_max) + 1.0, GRID_STEP_S)
    drive = (grid >= 0.0) & (grid <= DRIVE_END)
    for thr in CONFIG["COOL_THRESHOLDS"]:
        sc = _interp_area(ts, sareas[thr][0], grid)   # SIM cooled
        sh = _interp_area(ts, sareas[thr][1], grid)   # SIM heated
        ec = _interp_area(tu, earea_mean[thr]["cool"], grid)
        eh = _interp_area(tu, earea_mean[thr]["heat"], grid)
        ecsd = _interp_area(tu, earea_sd[thr]["cool"], grid)
        ehsd = _interp_area(tu, earea_sd[thr]["heat"], grid)
        area_frames.append(pd.DataFrame({
            "I0 (A)": i0, "threshold_K": thr, "time_into_pulse_s": grid,
            "area_cool_sim (mm2)": sc, "area_cool_exp (mm2)": ec,
            "area_cool_exp_sd (mm2)": ecsd, "err_cool (mm2)": ec - sc,
            "area_heat_sim (mm2)": sh, "area_heat_exp (mm2)": eh,
            "area_heat_exp_sd (mm2)": ehsd, "err_heat (mm2)": eh - sh,
        }))
        allpts = np.ones_like(grid, bool)
        area_rmse_rows.append({
            "I0 (A)": i0, "threshold_K": thr, "n_runs": len(tns),
            "rmse_cool_drive (mm2)": _rmse(ec - sc, drive),
            "rmse_heat_drive (mm2)": _rmse(eh - sh, drive),
            "rmse_cool_full (mm2)": _rmse(ec - sc, allpts),
            "rmse_heat_full (mm2)": _rmse(eh - sh, allpts),
            "bias_cool_drive (mm2)": _bias(ec - sc, drive),
            "bias_heat_drive (mm2)": _bias(eh - sh, drive),
        })

if area_frames:
    area_evol_df = pd.concat(area_frames, ignore_index=True)
    area_evol_df.to_csv(AREA_EVOL_CSV, index=False)
    print("\nsaved %s (%d rows, %d currents, %d thresholds)"
          % (AREA_EVOL_CSV, len(area_evol_df), area_evol_df["I0 (A)"].nunique(),
             area_evol_df["threshold_K"].nunique()))
if area_rmse_rows:
    area_rmse_df = pd.DataFrame(area_rmse_rows)
    area_rmse_df.to_csv(AREA_RMSE_CSV, index=False)
    print("saved %s (%d rows)" % (AREA_RMSE_CSV, len(area_rmse_df)))
    print(area_rmse_df.to_string(index=False))


# ---- C1: peak min/max vs energy (SIM vs EXP overlay) ----
fig, axes = plt.subplots(1, 2, figsize=(27.04, 12.5))
for ax, (cs, ce, title) in zip(axes, [
        ("dTmin_peak_sim (C)", "dTmin_peak_exp (C)",
         "(a) Peak MIN relative-T vs drive energy"),
        ("dTmax_peak_sim (C)", "dTmax_peak_exp (C)",
         "(b) Peak MAX relative-T vs drive energy")]):
    ax.scatter(comp["E_sim_drive (J)"], comp[cs], s=336, marker="v",
               edgecolors="k", linewidths=3, color="#2166ac",
                label="SIM surface", zorder=3)
    ax.scatter(comp["E_exp (J)"], comp[ce], s=288, marker="o",
               edgecolors="k", linewidths=3, color="#b2182b",
               label="EXP surface", zorder=3)
    for _, r in comp.iterrows():
        ax.annotate("%.2f A" % r["I0 (A)"], (r["E_exp (J)"], r[ce]),
                    fontsize=30, ha="right", va="top")
    ax.axhline(0.0, color="k", lw=3.4)
    ax.set_title(title, fontsize=35)
    ax.set(xlabel="Drive-window input energy (J)",
           ylabel="$\Delta T$ (°C)")
    ax.legend(fontsize=26, loc="upper center", bbox_to_anchor=(0.5, -0.24),
              frameon=True, facecolor="white", framealpha=0.95, edgecolor="0.7")
    ax.grid(alpha=0.3)
pass  # suptitle removed (unified style: no long titles)
pass  # footnote removed (unified style: no bottom text)
fig.tight_layout(rect=(0.0, 0.115, 1.0, 1.0)); save_figure(fig, "C1_minmax_peak_vs_energy.png")

# ---- C2: end-of-pulse min/max vs energy ----
fig, axes = plt.subplots(1, 2, figsize=(27.04, 12.5))
for ax, (cs, ce, title) in zip(axes, [
        ("dTmin_end_sim (C)", "dTmin_end_exp (C)",
         "(a) End-of-pulse MIN relative-T vs drive energy"),
        ("dTmax_end_sim (C)", "dTmax_end_exp (C)",
         "(b) End-of-pulse MAX relative-T vs drive energy")]):
    ax.scatter(comp["E_sim_drive (J)"], comp[cs], s=336, marker="v",
               edgecolors="k", linewidths=3, color="#2166ac",
                label="SIM surface", zorder=3)
    ax.scatter(comp["E_exp (J)"], comp[ce], s=288, marker="o",
               edgecolors="k", linewidths=3, color="#b2182b",
               label="EXP surface", zorder=3)
    for _, r in comp.iterrows():
        ax.annotate("%.2f A" % r["I0 (A)"], (r["E_exp (J)"], r[ce]),
                    fontsize=30, ha="right", va="top")
    ax.axhline(0.0, color="k", lw=3.4)
    ax.set_title(title, fontsize=35)
    ax.set(xlabel="Drive-window input energy (J)",
           ylabel="$\Delta T$ (°C)")
    ax.legend(fontsize=26, loc="upper center", bbox_to_anchor=(0.5, -0.24),
              frameon=True, facecolor="white", framealpha=0.95, edgecolor="0.7")
    ax.grid(alpha=0.3)
pass  # suptitle removed (unified style: no long titles)
pass  # footnote removed (unified style: no bottom text)
fig.tight_layout(rect=(0.0, 0.115, 1.0, 1.0)); save_figure(fig, "C2_minmax_end_vs_energy.png")

# ---- C3: per-matched-I0 trace overlays (common time-into-pulse axis) ----
n = len(pairs)
fig, axes = plt.subplots(n, 1, figsize=(20.0, 7 * n), squeeze=False,
                         gridspec_kw={"hspace": 0.5})

for ax, (i0, rs, re_) in zip(axes[:, 0], pairs):
    # SIM side = the inner-surface FIELD (never a point probe).
    tr = sim_surface_trace(i0)
    if tr is not None:
        ts, tc, th, _tm = tr
        ax.plot(ts, tc, color="#2166ac", lw=4.5, label="SIM surface min")
        ax.plot(ts, th, color="#b2182b", lw=4.5, label="SIM surface max")
    em = exp_mean_traces(i0)
    if em is not None:
        tu, cold, hot, tns, cs, hs, *_a = em
        ax.fill_between(tu, cold - cs, cold + cs, color="#2166ac", alpha=0.15, lw=0)
        ax.fill_between(tu, hot - hs, hot + hs, color="#b2182b", alpha=0.15, lw=0)
        ax.plot(tu, cold, color="#2166ac", lw=3.4, ls="--",
                label="EXP min (n=%d)" % len(tns))
        ax.plot(tu, hot, color="#b2182b", lw=3.4, ls="--",
                label="EXP max (n=%d)" % len(tns))
    ax.axvspan(0, CONFIG["T_END_SIM"] - CONFIG["T_ON_SIM"], color="0.88",
               alpha=0.5, lw=0)
    ax.axhline(0.0, color="k", lw=3.4)
    _rm = RMSE_BY_I0.get(round(float(i0), 3))
    _rm_txt = ("  |  RMSE min %.2f / max %.2f °C"
               % (_rm["rmse_min_drive (K)"], _rm["rmse_max_drive (K)"])
               if _rm else "")
    ax.set_title("$I_0$ = %.2f A  |  RMSE min %.2f / max %.2f °C"
                 % ((i0, _rm["rmse_min_drive (K)"], _rm["rmse_max_drive (K)"])
                    if _rm else (i0, float("nan"), float("nan"))), fontsize=30)
    ax.set(xlabel="Time (s)", ylabel="$\Delta T$ (°C)")
    ax.grid(alpha=0.3)
    ax.set_yticks([-5,-2.5, 0, 2.5, 5])

fig.legend(handles=[_L([0], [0], color="#2166ac", lw=4.5, label="SIM surface min"),
                    _L([0], [0], color="#b2182b", lw=4.5, label="SIM surface max"),
                    _L([0], [0], color="#2166ac", lw=3.4, ls="--", label="EXP min"),
                    _L([0], [0], color="#b2182b", lw=3.4, ls="--", label="EXP max")],
           fontsize=26, ncol=4, loc="upper center",
           bbox_to_anchor=(0.5, 0.93), frameon=True, facecolor="white",
           framealpha=0.95, edgecolor="0.7")
pass  # suptitle removed (unified style: no long titles)
pass  # footnote removed (unified style: no bottom text)
fig.tight_layout(rect=(0.0, 0.115, 1.0, 0.90))
fig.savefig(os.path.join(CONFIG["OUTPUT_DIR"], "C3_overlay_traces_matched_I0.png"), dpi=130, bbox_inches="tight")
print("saved C3_overlay_traces_matched_I0.png"); index_entries.append("C3_overlay_traces_matched_I0.png"); plt.close(fig)

# ---- C4: temporal grid mirror (cf. E1/H3): overlay all matched I0 + focus ----
FOCUS = min([p[0] for p in pairs], key=lambda c: abs(c - 0.3))
fig, axes = plt.subplots(2, 1, figsize=(22.88, 16.5), sharex=True)
for i0, rs, re_ in pairs:
    tr = sim_surface_trace(i0)
    _rm = RMSE_BY_I0.get(round(float(i0), 3))
    _lab = ("SIM %.2f A (RMSE %.2f)" % (i0, _rm["rmse_min_drive (K)"])
            if _rm else "SIM %.2f A" % i0)
    if tr is not None:
        ts, tc, th, _tm = tr
        axes[0].plot(ts, th, lw=3.9, label=_lab)
        axes[1].plot(ts, tc, lw=3.9)
    em = exp_mean_traces(i0)
    if em is not None:
        tu, cold, hot, tns, cs, hs, *_a = em
        axes[0].plot(tu, hot, lw=3.4, ls="--")
        axes[1].plot(tu, cold, lw=3.4, ls="--")
for ax in axes:
    ax.axvspan(0, CONFIG["T_END_SIM"] - CONFIG["T_ON_SIM"], color="0.88", alpha=0.5, lw=0)
    ax.axhline(0.0, color="k", lw=3.4); ax.set_xlim(-2, 60)
axes[1].set_xlabel("Time (s)")
axes[0].set_ylabel("$\Delta T_h$ (°C)"); axes[1].set_ylabel("$\Delta T_c$ (°C)")
axes[0].set_title("(a) Hot: SIM solid vs EXP dashed", fontsize=35)
axes[1].set_title("(b) Cold: SIM solid vs EXP dashed", fontsize=35)
fig.legend(handles=[l for l in axes[0].lines if l.get_label() and not l.get_label().startswith("_")],
           labels=[l.get_label() for l in axes[0].lines if l.get_label() and not l.get_label().startswith("_")],
           fontsize=24, ncol=3,
           loc="lower center", bbox_to_anchor=(0.5, 0.075),
           frameon=True, facecolor="white", framealpha=0.95, edgecolor="0.7")
pass  # suptitle removed (unified style: no long titles)
pass  # footnote removed (unified style: no bottom text)
fig.tight_layout(rect=(0.0, 0.165, 1.0, 1.0)); save_figure(fig, "C4_temporal_overlay_all_I0.png")

# ---- C5: normalized dynamics mirror (cf. E4/H4) ----
fig, axes = plt.subplots(1, 2, figsize=(27.04, 13.0))
for i0, rs, re_ in pairs:
    tr = sim_surface_trace(i0)
    if tr is not None:
        ts, dc = tr[0], tr[1]
        pk = np.nanmin(dc[(ts >= 0) & (ts <= 25)])
        if np.isfinite(pk) and abs(pk) > 1e-9:
            axes[0].plot(ts, dc / pk, lw=3.9, label="SIM %.2f A" % i0)
    em = exp_mean_traces(i0)
    if em is not None:
        tu, cold, hot, tns, cs, hs, *_a = em
        pk2 = np.nanmin(cold[(tu >= 0) & (tu <= 25)])
        if np.isfinite(pk2) and abs(pk2) > 1e-9:
            axes[0].plot(tu, cold / pk2, lw=3.4, ls="--")
axes[0].axvspan(0, 25, color="0.88", alpha=0.5, lw=0)
axes[0].set_title("(a) Normalized cold response (peak = 1)", fontsize=35)
axes[0].set(xlabel="Time (s)", ylabel="$\Delta T/\Delta T_{\mathrm{peak}}$ (-)")
axes[0].legend(fontsize=24, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.32), frameon=True, facecolor="white",
              framealpha=0.95, edgecolor="0.7")
axes[1].plot(comp["I0 (A)"], comp["t_peak_cool_sim (s)"], "o-", color="#2166ac",
             label="SIM t_peak (into pulse)")
axes[1].plot(comp["I0 (A)"], comp["t_peak_cool_exp (s)"], "s--", color="#b2182b",
             label="EXP t_peak (into pulse)")
axes[1].set_title("(b) Time to cold peak vs current", fontsize=35)
axes[1].set(xlabel="Drive current $I_0$ (A)", ylabel="Time (s)"); axes[1].legend(fontsize=32)
pass  # suptitle removed (unified style: no long titles)
pass  # footnote removed (unified style: no bottom text)
fig.tight_layout(rect=(0.0, 0.115, 1.0, 1.0)); save_figure(fig, "C5_normalized_dynamics.png")

# ---- C6: switch-off / overshoot mirror (cf. E5/H5) ----
fig, axes = plt.subplots(1, 2, figsize=(27.04, 13.0))
for i0, rs, re_ in pairs:
    tr = sim_surface_trace(i0)
    if tr is not None:
        ts, dc = tr[0], tr[1]
        m = (ts >= 20) & (ts <= 56)
        axes[0].plot(ts[m], dc[m], lw=3.9, label="SIM %.2f A" % i0)
    em = exp_mean_traces(i0)
    if em is not None:
        tu, cold, hot, tns, cs, hs, *_a = em
        m2 = (tu >= 15) & (tu <= 55)
        axes[0].plot(tu[m2], cold[m2], lw=3.4, ls="--")
axes[0].axvline(25, color="k", lw=3.4, ls=":")
axes[0].set_title("(a) Switch-off zoom, $\Delta T_c$ (SIM solid / EXP dashed)", fontsize=35)
axes[0].set(xlabel="Time (s)", ylabel="$\Delta T_c$ (°C)")
axes[0].legend(fontsize=24, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.32), frameon=True, facecolor="white",
              framealpha=0.95, edgecolor="0.7")
axes[1].plot(comp["I0 (A)"], comp["overshoot_sim (C)"], "o-", color="#2166ac",
             label="SIM surface overshoot")
axes[1].plot(comp["I0 (A)"], comp["overshoot_exp (C)"], "s--", color="#b2182b",
             label="EXP overshoot")
axes[1].set_title("(b) Overshoot vs current", fontsize=35)
axes[1].set(xlabel="Drive current $I_0$ (A)", ylabel="Overshoot (°C)"); axes[1].legend(fontsize=32)
pass  # suptitle removed (unified style: no long titles)
pass  # footnote removed (unified style: no bottom text)
fig.tight_layout(rect=(0.0, 0.115, 1.0, 1.0)); save_figure(fig, "C6_switchoff_overshoot.png")

# ---- C7: energy + electrical mirror (cf. E12/H16-H18) ----
fig, axes = plt.subplots(2, 2, figsize=(27.04, 19.5))
axes[0][0].plot(comp["I0 (A)"], comp["E_sim_drive (J)"], "o-", color="#2166ac",
                label="SIM drive (ON->END)")
axes[0][0].plot(comp["I0 (A)"], comp["E_exp (J)"], "s--", color="#b2182b",
                label="EXP (active, <=60 s)")
axes[0][0].set_title("(a) Drive-window input energy vs current", fontsize=35)
axes[0][0].set(xlabel="Drive current $I_0$ (A)", ylabel="E (J)"); axes[0][0].legend(fontsize=32)
axes[0][1].plot(comp["I0 (A)"], comp["K_per_J_sim"], "o-", color="#2166ac", label="SIM")
axes[0][1].plot(comp["I0 (A)"], comp["K_per_J_exp"], "s--", color="#b2182b", label="EXP")
axes[0][1].set_title("(b) Cooling per joule |$\Delta T_{min}$_peak|/E", fontsize=35)
axes[0][1].set(xlabel="Drive current $I_0$ (A)", ylabel="K/J"); axes[0][1].legend(fontsize=32)
ax = axes[1][0]
for i0, rs, re_ in pairs:
    sub = sim_probes[np.isclose(sim_probes["I0"], i0)].sort_values("time_s")
    ts = sub["time_s"].to_numpy(float) - CONFIG["T_ON_SIM"]
    m = (ts >= 0) & (ts <= 10)
    ax.plot(ts[m], sub["voltage_tec1_V"].to_numpy(float)[m].__abs__(), lw=3.9,
            label="SIM %.2f A" % i0)
    vv = exp_vc_overlay(i0)
    if vv is not None:
        tu, vm, im_, tns = vv
        m2 = tu <= 10
        ax.plot(tu[m2], vm[m2], lw=3.4, ls="--")
ax.set_title("(c) $|V(t)|$ first 10 s (SIM solid / EXP dashed)", fontsize=35)
ax.set(xlabel="Time (s)", ylabel="|V| (V)", xlim=(0, 10))
ax.legend(fontsize=24, ncol=3, loc="upper center",
          bbox_to_anchor=(0.5, -0.30), frameon=True, facecolor="white",
          framealpha=0.95, edgecolor="0.7")
ax = axes[1][1]
ax.plot(comp["I0 (A)"], comp["V1_peak_sim (V)"], "o-", color="#2166ac", label="SIM V1 peak")
ax2 = ax.twinx()
for i0 in comp["I0 (A)"]:
    vv = exp_vc_overlay(float(i0))
    if vv is not None:
        ax2.plot([float(i0)], [float(np.nanmax(vv[1][vv[0] <= 10]))], "s", color="#b2182b")
ax2.plot([], [], "s", color="#b2182b", label="EXP V peak (10 s)")
ax.set_title("(d) Peak voltage vs current", fontsize=35)
ax.set(xlabel="Drive current $I_0$ (A)", ylabel="SIM V1 peak (V)")
ax2.set_ylabel("EXP V peak (V)")
from matplotlib.lines import Line2D as _C7L
ax.legend(handles=[_C7L([0], [0], color="#2166ac", marker="o", lw=3.9, label="SIM V1 peak"),
                   _C7L([0], [0], color="#b2182b", marker="s", ls="none", label="EXP V peak (10 s)")],
          fontsize=24, loc="upper center", bbox_to_anchor=(0.5, -0.30),
          frameon=True, facecolor="white", framealpha=0.95, edgecolor="0.7")
pass  # suptitle removed (unified style: no long titles)
pass  # footnote removed (unified style: no bottom text)
fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0)); save_figure(fig, "C7_energy_electrical.png")

# ---- C8: temperature-metrics mirror (cf. E8b/E10d): peaks + end vs current ----
fig, axes = plt.subplots(1, 2, figsize=(27.04, 12.5))
axes[0].plot(comp["I0 (A)"], comp["dTmin_peak_sim (C)"], "v-", color="#2166ac", label="SIM surface min peak")
axes[0].plot(comp["I0 (A)"], comp["dTmin_peak_exp (C)"], "v--", color="#2166ac", label="EXP min peak")
axes[0].plot(comp["I0 (A)"], comp["dTmax_peak_sim (C)"], "^-", color="#b2182b", label="SIM surface max peak")
axes[0].plot(comp["I0 (A)"], comp["dTmax_peak_exp (C)"], "^--", color="#b2182b", label="EXP max peak")
axes[0].axhline(0.0, color="k", lw=3.4)
axes[0].set_title("(a) Peak min/max vs current", fontsize=35)
axes[0].set(xlabel="Drive current $I_0$ (A)", ylabel="$\Delta T$ (°C)")
axes[0].legend(fontsize=26, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, -0.22), frameon=True, facecolor="white",
               framealpha=0.95, edgecolor="0.7")
axes[1].plot(comp["I0 (A)"], comp["dTmin_end_sim (C)"], "v-", color="#2166ac", label="SIM surface min end")
axes[1].plot(comp["I0 (A)"], comp["dTmin_end_exp (C)"], "v--", color="#2166ac", label="EXP min end")
axes[1].plot(comp["I0 (A)"], comp["dTmax_end_sim (C)"], "^-", color="#b2182b", label="SIM surface max end")
axes[1].plot(comp["I0 (A)"], comp["dTmax_end_exp (C)"], "^--", color="#b2182b", label="EXP max end")
axes[1].axhline(0.0, color="k", lw=3.4)
axes[1].set_title("(b) End-of-pulse min/max vs current", fontsize=35)
axes[1].set(xlabel="Drive current $I_0$ (A)", ylabel="$\Delta T$ (°C)")
axes[1].legend(fontsize=26, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, -0.22), frameon=True, facecolor="white",
               framealpha=0.95, edgecolor="0.7")
pass  # suptitle removed (unified style: no long titles)
pass  # footnote removed (unified style: no bottom text)
fig.tight_layout(rect=(0.0, 0.115, 1.0, 1.0)); save_figure(fig, "C8_temperature_vs_current.png")

# ---- C9: spatial area mirror (cf. E7/E10 vs H11) ----
THRS = [(0.5, "05"), (1.0, "1"), (2.0, "2"), (5.0, "5")]
THR_C = {0.5: "#1b7837", 1.0: "#4dac26", 2.0: "#f1a340", 5.0: "#d01c8b"}
fig, axes = plt.subplots(2, 2, figsize=(27.04, 16.64))
xs = sim_metrics.set_index("I0_A"); xe = exp_metrics.set_index("I0 (A)")
for thr, tag in THRS:
    sc = "area_cool_%s_end" % tag
    ec = "area_cool_0p5_end (mm2)" if tag == "05" else "area_cool_%s_end (mm2)" % tag
    if sc in xs.columns and ec in xe.columns:
        axes[0][0].plot(xs.index, xs[sc], "-o", ms=11.2, color=THR_C[thr], label="%gK SIM" % thr)
        axes[0][0].plot(xe.index, xe[ec], "--s", ms=11.2, color=THR_C[thr], label="%gK EXP" % thr)
axes[0][0].set_title("(a) Cooled area at drive end vs current", fontsize=35)
axes[0][0].set(xlabel="Drive current $I_0$ (A)", ylabel="area (mm$^2$)"); axes[0][0].set_ylim(bottom=0)
axes[0][0].legend(fontsize=31, ncol=2)
for thr, tag in THRS:
    sc = "area_cool_%s_peak" % tag
    ec = "area_cool_0p5_peak (mm2)" if tag == "05" else "area_cool_%s_peak (mm2)" % tag
    if sc in xs.columns and ec in xe.columns:
        axes[0][1].plot(xs.index, xs[sc], "-o", ms=11.2, color=THR_C[thr], label="%gK SIM" % thr)
        axes[0][1].plot(xe.index, xe[ec], "--s", ms=11.2, color=THR_C[thr], label="%gK EXP" % thr)
axes[0][1].set_title("(b) Peak cooled area vs current", fontsize=35)
axes[0][1].set(xlabel="Drive current $I_0$ (A)", ylabel="area (mm$^2$)"); axes[0][1].set_ylim(bottom=0)
axes[0][1].legend(fontsize=31, ncol=2)
ax = axes[1][0]
for thr, tag in THRS:
    scc, shh = "area_cool_%s_end" % tag, "area_heat_%s_end" % tag
    ecc = "area_cool_0p5_end (mm2)" if tag == "05" else "area_cool_%s_end (mm2)" % tag
    ehh = "area_heat_0p5_end (mm2)" if tag == "05" else "area_heat_%s_end (mm2)" % tag
    if scc in xs.columns and shh in xs.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            ax.plot(xs.index, np.where(xs[shh] > 0, xs[scc] / xs[shh], np.nan),
                    "-o", ms=11.2, color=THR_C[thr], label="%gK SIM" % thr)
    if ecc in xe.columns and ehh in xe.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            ax.plot(xe.index, np.where(xe[ehh] > 0, xe[ecc] / xe[ehh], np.nan),
                    "--s", ms=11.2, color=THR_C[thr], label="%gK EXP" % thr)
ax.axhline(1.0, color="0.4", lw=3.4, ls=":"); ax.set_yscale("log")
ax.set_title("(c) Cooled/heated balance at drive end (log)", fontsize=35)
ax.set(xlabel="Drive current $I_0$ (A)", ylabel="ratio (-)"); ax.legend(fontsize=31, ncol=2)
ax = axes[1][1]
ax.plot(xs.index, xs["area_cool_05_per_J"], "-o", color="#2166ac", label="SIM 0.5K/J")
ax.plot(xe.index, xe["area_cool_0p5_per_J (mm2/J)"], "--s", color="#b2182b", label="EXP 0.5K/J")
ax.set_title("(d) Peak cooled area per joule", fontsize=35)
ax.set(xlabel="Drive current $I_0$ (A)", ylabel="mm$^2$/J"); ax.legend(fontsize=32)
pass  # suptitle removed (unified style: no long titles)
pass  # footnote removed (unified style: no bottom text)
fig.tight_layout(rect=(0.0, 0.115, 1.0, 1.0)); save_figure(fig, "C9_spatial_areas.png")

# ---- C10: uniformity mirror (cf. H12) + scorecard ----
# SIM side comes from the cached surface traces (no re-stream, no probes).
fig, axes = plt.subplots(2, 2, figsize=(27.04, 16.64))
ax = axes[0][0]
for i0, rs, re_ in pairs:
    tr = sim_surface_trace(i0)
    if tr is None:
        print("      [C10 skip SIM surface I0=%.2f: no traces]" % i0)
        continue
    tt, tc, th, _tm = tr
    ax.plot(tt, th - tc, lw=4.5, label="SIM %.2f A" % i0)
ax.set_title("(a) SIM surface spread (max-min)", fontsize=35)
ax.set(xlabel="Time (s)", ylabel="spread (°C)"); ax.legend(fontsize=31)
ax = axes[0][1]
for i0, rs, re_ in pairs:
    em = exp_mean_traces(i0)
    if em is not None:
        tu, cold, hot, tns, cs, hs, *_a = em
        ax.plot(tu, hot - cold, lw=3.9, ls="--", label="EXP %.2f A" % i0)
ax.set_title("(b) EXP surface spread (frame max-min)", fontsize=35)
ax.set(xlabel="Time (s)", ylabel="spread (°C)"); ax.legend(fontsize=31)
ax = axes[1][0]
for i0, rs, re_ in pairs:
    tr = sim_surface_trace(i0)
    if tr is None:
        continue
    tt, tc, th, tmean = tr
    ax.plot(tt, tmean, lw=4.5, label="SIM %.2f A" % i0)
ax.set_title("(c) SIM surface mean $\Delta T$", fontsize=35)
ax.set(xlabel="Time (s)", ylabel="mean $\Delta T$ (°C)"); ax.legend(fontsize=31)
ax = axes[1][1]; ax.axis("off")
# Only the headline metrics: the full 40+ column table lives in the CSV, and
# dumping it here would wreck the layout (tight_layout cannot fit monospace text).
KEY_COLS = ["I0_A", "dTmin_peak_sim (C)", "dTmin_peak_exp (C)", "err_min_peak",
            "dTmax_peak_sim (C)", "dTmax_peak_exp (C)", "err_max_peak",
            "K_per_J_sim", "K_per_J_exp", "err_K_per_J"]
key = comp[[c for c in KEY_COLS if c in comp.columns]].round(3)
txt = (key.to_string(index=False)
       + "\n\nerr_* columns are probe-free: SIM = inner-surface FIELD only.\n"
       + "Probe values stay in the CSV as *_sim_probe reference columns.")
ax.text(0.01, 0.98, txt, transform=ax.transAxes,
        fontsize=26, family="monospace", va="top", ha="left")
ax.set_title("(d) Scorecard (headline metrics, full table in CSV)", fontsize=35)
pass  # suptitle removed (unified style: no long titles)
pass  # footnote removed (unified style: no bottom text)
# fixed margins: the monospace scorecard makes tight_layout fail with
# "margins cannot be made large enough", so set them explicitly.
fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.115,
                    wspace=0.22, hspace=0.42)
save_figure(fig, "C10_uniformity_scorecard.png")

# ---- C11.1..C11.4: spatial threshold-crossing AREA evolution (SIM vs EXP) ----
# One figure per threshold (0.5/1/2/5 K): cooled + heated area over time, SIM
# solid vs EXP dashed, for every matched current.  Legend labels carry the peak
# area so the slide-by-slide numbers are readable straight off the figure.
if area_frames:
    THRS_C = [(0.5, "#1b7837", "C11.1_area_evolution_0p5K.png"),
              (1.0, "#4dac26", "C11.2_area_evolution_1K.png"),
              (2.0, "#f1a340", "C11.3_area_evolution_2K.png"),
              (5.0, "#d01c8b", "C11.4_area_evolution_5K.png")]
    # precompute SIM + EXP area traces once per current (caches make this cheap)
    c11 = []
    for i0, rs, re_ in pairs:
        sa = sim_area_evolution(i0)
        em = exp_mean_traces(i0)
        if sa is None or em is None:
            continue
        ts, sareas = sa
        tu, _c, _h, tns, _cs, _hs, eam, eas = em
        c11.append((i0, ts, sareas, tu, eam, eas))
    # current-coded colors (viridis): same I0 = same color as H/E figures
    _cmap11 = plt.get_cmap("viridis")
    _i0s11 = [i0 for i0, *_r in c11]
    _norm11 = plt.Normalize(min(_i0s11), max(_i0s11)) if len(_i0s11) > 1 else None
    def _col11(_i0):
        return _cmap11(_norm11(_i0)) if _norm11 is not None else _cmap11(0.5)
    for thr, col, fname_ in THRS_C:
        fig, axes = plt.subplots(1, 2, figsize=(27.04, 10.14))
        for c, side in enumerate(("cool", "heat")):
            ax = axes[c]
            j = 0 if side == "cool" else 1
            for i0, ts, sareas, tu, eam, eas in c11:
                sim_y = sareas[thr][j]
                exp_y = eam[thr][side]
                exp_sd = eas[thr][side]
                cc = _col11(i0)
                ax.plot(ts, sim_y, lw=4.2, color=cc)
                ax.plot(tu, exp_y, lw=3.4, ls="--", color=cc)
                ax.fill_between(tu, exp_y - exp_sd, exp_y + exp_sd,
                                color=cc, alpha=0.12, lw=0)
            ax.axvspan(0, DRIVE_END, color="0.9", alpha=0.6, lw=0)
            ax.set_title("%s area @ %g K" % (side.capitalize(), thr), fontsize=32)
            ax.set(xlabel="Time (s)", ylabel="area (mm$^2$)")
            ax.set_ylim(bottom=0)
        from matplotlib.lines import Line2D as _L11
        fig.legend(handles=[_L11([0], [0], color=_col11(i0), lw=4.2,
                                 label="SIM %.1f A" % i0) for i0, *_r in c11]
                         + [_L11([0], [0], color="k", lw=3.4, ls="--",
                                  label="EXP (dashed)")],
                   fontsize=26, ncol=1, loc="center right",
                   bbox_to_anchor=(0.99, 0.5), frameon=True, facecolor="white",
                   framealpha=0.95, edgecolor="0.7")
        pass  # suptitle removed (unified style: no long titles)
        pass  # footnote removed (unified style: no bottom text)
        # right-side legend: leave room on the right so it never covers data
        fig.tight_layout(rect=(0.0, 0.115, 0.86, 1.0))
        save_figure(fig, fname_)

# ---- C14: cooled/heated AREA balance bars (mirror of H9b), SIM vs EXP ----
# H9(b) draws cooled volume up / heated volume down at drive end, one bar per
# threshold, grouped by drive current.  Areas (mm^2) are the only quantity both
# sides share, so the same bar layout is reused here: (a) SIM surface,
# (b) EXP surface (error bars = SD across repeats).
THRS_B = [(0.5, "05", "0p5"), (1.0, "1", "1"), (2.0, "2", "2"), (5.0, "5", "5")]
cur_b = [p[0] for p in pairs]
fig, axes = plt.subplots(1, 2, figsize=(27.04, 10.14))
xb = np.arange(len(cur_b))
w = 0.2
for pi, (is_sim, ttl) in enumerate(((True, "(a) SIM surface — area at drive end"),
                                    (False, "(b) EXP surface — area at drive end"))):
    ax = axes[pi]
    for k, (thr, stag, etag) in enumerate(THRS_B):
        tag = stag if is_sim else etag
        ccol = ("area_cool_%s_end" % tag) if is_sim else ("area_cool_%s_end (mm2)" % tag)
        hcol = ("area_heat_%s_end" % tag) if is_sim else ("area_heat_%s_end (mm2)" % tag)
        rows_ = [rs if is_sim else re_ for _i0, rs, re_ in pairs]
        if not any(ccol in r.index for r in rows_):
            continue
        cool = np.nan_to_num([r.get(ccol, np.nan) for r in rows_],
                             nan=0.0).astype(float)
        heat = np.nan_to_num([r.get(hcol, np.nan) for r in rows_],
                             nan=0.0).astype(float)
        cerr = herr = None
        if not is_sim:
            csd = ccol.replace(" (mm2)", "_sd (mm2)")
            hsd = hcol.replace(" (mm2)", "_sd (mm2)")
            cerr = np.nan_to_num([r.get(csd, np.nan) for r in rows_], nan=0.0)
            herr = np.nan_to_num([r.get(hsd, np.nan) for r in rows_], nan=0.0)
        ekw = dict(ecolor="0.2", capsize=5, lw=2.4, zorder=6)
        ax.bar(xb + (k - 1.5) * w, cool, w, color=THR_C[thr], edgecolor="k", lw=3,
               label="%g K" % thr, yerr=cerr, error_kw=ekw)
        ax.bar(xb + (k - 1.5) * w, -heat, w, color=THR_C[thr], edgecolor="k", lw=3,
               hatch="///", alpha=0.85, yerr=herr, error_kw=ekw)
    ax.axhline(0, color="0.4", lw=3.4)
    ax.set_xticks(xb)
    ax.set_xticklabels(["%.1f" % c for c in cur_b], fontsize=32)
    ax.set_title(ttl, fontsize=35)
    ax.set_xlabel(r"Drive current $I_0$ (A)")
    ax.set_ylabel("area (mm$^2$)")
    handles = [Patch(facecolor=THR_C[thr], edgecolor="k", label="%g ºC" % thr)
               for thr, _s, _e in THRS_B]
    handles += [Patch(facecolor="0.75", edgecolor="k", label="up = cooled"),
                Patch(facecolor="0.75", edgecolor="k", hatch="///", label="down = heated")]
    ax.legend(handles=handles, fontsize=24, ncol=3, loc="lower left")
# shared y-limits so the 0 line sits at the same height in (a) and (b)
_lo = min(ax.get_ylim()[0] for ax in axes)
_hi = max(ax.get_ylim()[1] for ax in axes)
for ax in axes:
    ax.set_ylim(_lo, _hi)
pass  # suptitle removed (unified style: no long titles)
pass  # footnote removed (unified style: no bottom text)
fig.tight_layout(rect=(0.0, 0.115, 1.0, 1.0))
save_figure(fig, "C14_area_balance_bars.png")
# ---- C15: cooled/heated AREA balance bars at PEAK (mirror of C14), SIM vs EXP ----
fig15, axes15 = plt.subplots(1, 2, figsize=(27.04, 10.14))
xb15 = np.arange(len(cur_b))
for pi, (is_sim, ttl) in enumerate(((True, "(a) SIM surface — peak area"),
                                    (False, "(b) EXP surface — peak area"))):
    ax = axes15[pi]
    for k, (thr, stag, etag) in enumerate(THRS_B):
        tag = stag if is_sim else etag
        ccol = ("area_cool_%s_peak" % tag) if is_sim else ("area_cool_%s_peak (mm2)" % tag)
        hcol = ("area_heat_%s_peak" % tag) if is_sim else ("area_heat_%s_peak (mm2)" % tag)
        rows_ = [rs if is_sim else re_ for _i0, rs, re_ in pairs]
        if not any(ccol in r.index for r in rows_):
            continue
        cool = np.nan_to_num([r.get(ccol, np.nan) for r in rows_],
                             nan=0.0).astype(float)
        heat = np.nan_to_num([r.get(hcol, np.nan) for r in rows_],
                             nan=0.0).astype(float)
        cerr = herr = None
        if not is_sim:
            csd = ccol.replace(" (mm2)", "_sd (mm2)")
            hsd = hcol.replace(" (mm2)", "_sd (mm2)")
            cerr = np.nan_to_num([r.get(csd, np.nan) for r in rows_], nan=0.0)
            herr = np.nan_to_num([r.get(hsd, np.nan) for r in rows_], nan=0.0)
        ekw = dict(ecolor="0.2", capsize=5, lw=2.4, zorder=6)
        ax.bar(xb15 + (k - 1.5) * w, cool, w, color=THR_C[thr], edgecolor="k", lw=3,
               label="%g K" % thr, yerr=cerr, error_kw=ekw)
        ax.bar(xb15 + (k - 1.5) * w, -heat, w, color=THR_C[thr], edgecolor="k", lw=3,
               hatch="///", alpha=0.85, yerr=herr, error_kw=ekw)
    ax.axhline(0, color="0.4", lw=3.4)
    ax.set_xticks(xb15)
    ax.set_xticklabels(["%.1f" % c for c in cur_b], fontsize=32)
    ax.set_title(ttl, fontsize=35)
    ax.set_xlabel(r"Drive current $I_0$ (A)")
    ax.set_ylabel("area (mm$^2$)")
    handles = [Patch(facecolor=THR_C[thr], edgecolor="k", label="%g ºC" % thr)
               for thr, _s, _e in THRS_B]
    handles += [Patch(facecolor="0.75", edgecolor="k", label="up = cooled"),
                Patch(facecolor="0.75", edgecolor="k", hatch="///", label="down = heated")]
    ax.legend(handles=handles, fontsize=24, ncol=3, loc="upper left")
    ax.set_yticks([-100, -75, -50, -25, 0 ,25, 50,75, 100])
# shared y-limits so the 0 line sits at the same height in (a) and (b)
_lo15 = min(ax.get_ylim()[0] for ax in axes15)
_hi15 = max(ax.get_ylim()[1] for ax in axes15)
for ax in axes15:
    ax.set_ylim([-100,100])
fig15.tight_layout(rect=(0.0, 0.115, 1.0, 1.0))
save_figure(fig15, "C15_area_balance_bars_peak.png")
# ---- C16.1..C16.5: SIM-vs-EXP spatial maps per current (binds H10 + E9) ----
# One figure per matched current: 2 rows x 5 columns.
# Top row = SIM field maps (Config_heatsink_surf.csv via load_surface_current +
# grid_from_scatter), bottom row = EXP IR frames (first matched run, frame-0 +
# border-bg subtracted, same recipe as fig_e9_maps).  Columns = 5 timestamps
# into-pulse.  One shared TwoSlopeNorm + coolwarm LUT per figure so colors are
# directly comparable.  Short per-panel titles only (no suptitle, no footnote).
C16_TIMES = [2.0, 10.0, 20.0, 25.0, 40.0]
try:
    from simulated_data import load_surface_current as _c16_sim_loader
    from reading_comsol_dual_tec_graphics import (
        grid_from_scatter as _c16_grid, time_slice as _c16_slice,
        CMAP_T as _CMAP_T)
    from matplotlib.colors import TwoSlopeNorm as _TSN
    from mpl_toolkits.axes_grid1 import make_axes_locatable as _c16_mal
    _c16_ok = True
except Exception as _c16_e:
    print("      [C16 skip] map helpers unavailable: %s" % _c16_e)
    _c16_ok = False
if _c16_ok:
    _c16_cache = {}
    for _c16_k, (_i0, _rs, _re) in enumerate(pairs):
        try:
            _df = _c16_sim_loader(_i0, _c16_cache)
        except Exception as _e:
            print("      [C16 skip I0=%.2f SIM load: %s]" % (_i0, _e))
            continue
        _tns = [tn for tn in vc_info
                if abs(_exp_i0_of(tn) - _i0) <= CONFIG["I0_MATCH_TOL"]]
        if not _tns:
            print("      [C16 skip I0=%.2f: no EXP runs]" % _i0)
            continue
        _lst = sorted(xf_tests[_tns[0]])
        _ref = cached_frame(dict(_lst)[0])
        _h, _w = _ref.shape
        _bh, _bw = max(int(_h * CONFIG["BG_FRAC"]), 1), max(int(_w * CONFIG["BG_FRAC"]), 1)
        _border = np.zeros((_h, _w), bool)
        _border[:_bh, :] = True; _border[-_bh:, :] = True
        _border[:, :_bw] = True; _border[:, -_bw:] = True
        _idxs = sorted(i for i, _p in _lst if i != 0)
        _d = dict(_lst)
        _stk = np.array([cached_frame(_d[i]) - _ref for i in _idxs], float)
        _bg = _stk[:, _border].mean(axis=1)
        _stk = _stk - (_bg - _bg[0])[:, None, None]
        _tt = np.asarray(_idxs, float) / CONFIG["FRAME_RATE_HZ"]
        # SIM panels (gridded field at absolute t = into-pulse + T_ON_SIM)
        _sim_frs, _exp_frs = [], []
        for _tp in C16_TIMES:
            _sub = _c16_slice(_df, _tp + CONFIG["T_ON_SIM"])
            _sim_frs.append(_c16_grid(_sub) if _sub is not None else None)
            _k = int(np.argmin(np.abs(_tt - _tp)))
            _exp_frs.append(_stk[_k])
        _vals = []
        for _fr in _sim_frs:
            if _fr is not None:
                _vals.append(np.nanmax(np.abs(_fr["val"])))
        _vals.append(float(np.nanmax(np.abs(_exp_frs))))
        _vmax = float(np.nanmax(_vals)) if _vals else 1.0
        _vmax = _vmax if np.isfinite(_vmax) and _vmax > 0 else 1.0
        _norm = _TSN(vmin=-_vmax, vcenter=0.0, vmax=_vmax)
        _fig, _axes = plt.subplots(2, len(C16_TIMES),
                                   figsize=(5.4 * len(C16_TIMES), 11.6),
                                   squeeze=False)
        _fig.suptitle("$I_0$ = %.2f A" % _i0, fontsize=38, y=0.99)
        for _c, (_tp, _sfr, _efr) in enumerate(zip(C16_TIMES, _sim_frs, _exp_frs)):
            _ax = _axes[0][_c]
            _ax.grid(False)
            if _sfr is None:
                _ax.axis("off")
                _ax.set_title("t = %.0f s (no data)" % _tp, fontsize=31)
            else:
                _ax.imshow(_sfr["val"], origin="lower", extent=_sfr["extent"],
                           cmap=_CMAP_T, norm=_norm, aspect="auto",
                           interpolation="bilinear")
                try:
                    _z = np.ma.masked_invalid(_sfr["val"])
                    if _z.count() > 2:
                        _ax.contour(_sfr["G1"], _sfr["G2"], _z, levels=8,
                                    colors="0.45", linewidths=2.4, alpha=0.8)
                except Exception:
                    pass
                _ax.set_title("t = %.0f s" % _tp, fontsize=32)
                _ax.set_xticks([]); _ax.set_yticks([])
            if _c == 0:
                _ax.set_ylabel("SIM (°C)", fontsize=36)
            _ax2 = _axes[1][_c]
            _ax2.grid(False)
            _ef = np.flipud(_efr)
            try:
                from scipy.ndimage import gaussian_filter as _gf
                _efd = _gf(_ef, sigma=1.0)
            except Exception:
                _efd = _ef
            _ax2.imshow(_efd, cmap=_CMAP_T, norm=_norm,
                        aspect="auto", origin="upper")
            try:
                _ze = np.ma.masked_invalid(_efd)
                # keep contours out of the +/-0.2 K background-noise band
                _span = float(np.nanmax(np.abs(_efd)))
                _lv = np.linspace(-_span, _span, 10)
                _lv = _lv[np.abs(_lv) >= 0.2]
                if _ze.count() > 2 and len(_lv) >= 2:
                    _ax2.contour(_ze, levels=_lv, colors="0.45",
                                 linewidths=2.4, alpha=0.8)
            except Exception:
                pass
            _ax2.set_xticks([]); _ax2.set_yticks([])
            if _c == 0:
                _ax2.set_ylabel("EXP (°C)", fontsize=36)
        _fig.tight_layout(rect=(0.0, 0.0, 0.92, 0.96))
        _cax = _fig.add_axes((0.93, 0.15, 0.015, 0.7))
        _cb = _fig.colorbar(plt.cm.ScalarMappable(norm=_norm, cmap=_CMAP_T),
                            cax=_cax)
        _cb.set_label(r"$\Delta T$ (°C)", fontsize=32)
        _cb.ax.tick_params(labelsize=32)
        save_figure(_fig, "C16.%d_maps_I0_%.2fA.png" % (_c16_k + 1, _i0))



with open(os.path.join(CONFIG["OUTPUT_DIR"], "index.md"), "w") as f:
    f.write("# Heatsink SIM vs EXP outer-surface comparison\n\n")
    f.write("SIM: OUTPUT/heatsink_metrics.csv (drive %g-%g s). "
            "EXP: OUTPUT/exp_outer_metrics.csv (outer-surface IR only, drive 0-%g s).\n\n"
            % (CONFIG["T_ON_SIM"], CONFIG["T_END_SIM"], CONFIG["T_END_EXP"]))
    f.write("Energy = drive window only. I0 tol %.2f A. %s\n\n"
            % (CONFIG["I0_MATCH_TOL"], CAVEAT))
    f.write("SIM inputs: OUTPUT/heatsink_metrics.csv (`surf_*` columns) + "
            "OUTPUT/heatsink_surface_traces.csv, both produced by "
            "`reading_comsol_heatsink.py` from Config_heatsink_surf.csv.\n"
            "The Cold/Hot/Min/Max point probes are NOT comparable with a surface "
            "measurement: they only appear as `*_sim_probe` columns in the CSV.\n\n")
    for n_ in index_entries:
        f.write("- %s\n" % n_)
print("\nDone in %.1f s — %d figs" % (time.time() - t0_run, len(index_entries)))

