# =============================================================================
# reading_comsol_heatsink.py
# -----------------------------------------------------------------------------
# Single-model report for the Heatsink (Concise) COMSOL export:
#
#     Config_heatsink.csv        accumulated probes (hot/cold, flux, V1,
#                                heated/cooled volume at 0.5/1/2/5 degC)
#     Config_heatsink_surf.csv   inner-surface temperature field
#                                (plane x = 1.4225 mm, y -13..17, z -15..15)
#
# This file ONLY reads the heatsink model and plots everything that the export
# supports: temporal (probe), spatial (volume probes + 2D surface) and energetic
# metrics, one figure family per topic (H1..H19).
#
# Nothing here is dual-TEC specific: the thermally parallel TEC (Probe 3) does
# not exist in this model, so the recovered-voltage section is intentionally
# absent.  Loaders, the Delaunay area engine and the plotting helpers are
# REUSED from the two existing modules so that all three scripts stay
# consistent and share a single implementation.
#
#     python reading_comsol_heatsink.py
# =============================================================================
import os
import sys
import time

# --- shared-module lookup ---------------------------------------------------
# The loaders/plot helpers live in the shared modules, which may sit either next
# to this script or in the "other files" folder, so both locations are made
# importable (the script folder always wins).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_HERE, "other files"), _HERE):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.append(_cand)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sim_loaders import (
    CONFIG as BASE_CONFIG,
    VOL_THRESHOLDS,
    T_ON_S,
    T_END_S,
    T_FINAL_S,
    load_accumulated_probes,
    load_surface_inner,
    nearest_row,
    vol_col,
    has_volume_probes,
    collect_area_series,
    surface_areas_at,
    surface_area_peaks,
)
from sim_graphics import (
    apply_style,
    grid_from_scatter,
    common_map,
    time_slice,
    draw_map,
    frame_areas,
    plane_axes,
    spot_stats,
    check_no_overshoot,
    CMAP_T,
    CMAP_D,
    CMAP_CURRENT,
)

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
CFG = {
    # ---- inputs (single source of truth: the shared CONFIG) ----
    "PROBES_CSV": BASE_CONFIG["HEATSINK_PROBES_CSV"],
    "SURF_CSV": BASE_CONFIG["HEATSINK_INNER_SURF_CSV"],
    "T_AMB_C": BASE_CONFIG["T_AMB_DEFAULT_C"],
    "SURFACE_CHUNK": BASE_CONFIG["SURFACE_CHUNK"],

    # ---- protocol / reference times (25 s drive pulse) ----
    "T_ON_S": T_ON_S,
    "T_END_S": T_END_S,
    "T_FINAL_S": T_FINAL_S,
    "BACK_TO_NORMAL_K": 0.2,          # |Delta T_c| below this = back to normal
    "A_COLD_M2": 12.25e-6,
    "A_HOT_M2": 15.0e-6,
    "COP_P_FLOOR_FRAC": 0.01,
    "RUN_COP": True,          # |Delta T_c| below this = back to normal

    # ---- figure selections ----
    "GRID_CURRENTS": [0.1, 0.2, 0.3, 0.4, 0.5],
    "FOCUS_CURRENT": 0.2,
    "OVERLAY_CURRENTS": [0.1, 0.2, 0.3, 0.4, 0.5],
    "SPATIAL_CURRENTS": [0.1, 0.2, 0.3, 0.4, 0.5],
    "SPATIAL_TIMES_S": [10.0, 20.0, 27.0, 29.0, 45.0, 60.0],
    "MAP_TIMES_S": [10.0, 20.0, 27.0, 29.0, 60.0],
    "STRIDE_TIMES_S": [30.0, 40.0, 50.0, 60.0],
    "PROFILE_TIMES_S": [10.0, 27.0, 45.0],
    "AREA_CURRENTS": [0.1, 0.2, 0.3, 0.4, 0.5],   # or "all"
    "MAP_RESOLUTION": 160,
    "ROI_HALF_WIDTH_MM": 6.0,
    "SPATIAL_TIME_STRIDE": 2,          # keep 1 of every N time steps in memory

    # ---- outputs / switches ----
    "OUTPUT_DIR": os.path.join(os.path.dirname(os.path.abspath(BASE_CONFIG["COMPARISON_CSV"])),
                               "heatsink"),
    "METRICS_CSV": os.path.join(os.path.dirname(os.path.abspath(BASE_CONFIG["COMPARISON_CSV"])),
                                "heatsink_metrics.csv"),
    "SAVE_FIGURES": True,
    "SHOW_FIGURES": False,
    "FIGURE_DPI": 200,
    "RUN_PROBES": True,
    "RUN_SPATIAL_VOLUME": True,
    "RUN_SPATIAL_SURFACE": True,
    "RUN_ENERGY": True,
    "RUN_SCORECARD": True,
    "RUN_SURF_METRICS": True,      # extract the surf_* (surface field) metrics
    "SURF_METRICS_CURRENTS": "all",  # or a list of drive currents
    "SURF_TRACES_CSV": os.path.join(os.path.dirname(os.path.abspath(BASE_CONFIG["COMPARISON_CSV"])),
                                    "heatsink_surface_traces.csv"),
}

C_MAIN = "#b2182b"           # heatsink colour (kept consistent with the comparison script)
THR_COLORS = {0.5: "#1b7837", 1.0: "#4dac26", 2.0: "#f1a340", 5.0: "#d01c8b"}


def save_figure(fig, name, index=None):
    """Save a figure into CFG['OUTPUT_DIR'] (optionally show it), then close."""
    if CFG["SAVE_FIGURES"]:
        os.makedirs(CFG["OUTPUT_DIR"], exist_ok=True)
        fig.savefig(os.path.join(CFG["OUTPUT_DIR"], name),
                    dpi=CFG["FIGURE_DPI"], bbox_inches="tight")
        print("      saved %s" % name)
        if index is not None:
            index.append(name)
    if CFG["SHOW_FIGURES"]:
        plt.show()
    plt.close(fig)


def trapz(y, x):
    """Trapezoidal integral with a fallback for older numpy releases."""
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def sub_current(df, i0):
    """All samples of one drive current, sorted by time and index-compacted."""
    sub = df[np.isclose(df["I0"], i0)].sort_values("time_s")
    return sub.reset_index(drop=True) if not sub.empty else None


def first_cross(t, cond):
    """First time where 'cond' holds (NaN when it never does)."""
    idx = np.where(cond)[0]
    return float(t[idx[0]]) if idx.size else np.nan


def safe_ratio(a, b):
    """a / b with NaN instead of division by zero."""
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0.0:
        return np.nan
    return a / b

# =============================================================================
# 1. METRICS (single model)
# =============================================================================
def current_metrics(df, i0, t_on=None, t_end=None, t_final=None,
                    thresholds=VOL_THRESHOLDS):
    """All temporal, spatial and energetic metrics of one drive current."""
    sub = sub_current(df, i0)
    if sub is None:
        return None
    t_on = CFG["T_ON_S"] if t_on is None else t_on
    t_end = CFG["T_END_S"] if t_end is None else t_end
    t_final = CFG["T_FINAL_S"] if t_final is None else t_final

    t = sub["time_s"].to_numpy(float)
    v1 = np.abs(sub["voltage_tec1_V"].to_numpy(float))
    d_cold = sub["Delta_Cold"].to_numpy(float)
    d_hot = sub["Delta_Hot"].to_numpy(float)
    q = (sub["flux_Wm2"].to_numpy(float) if "flux_Wm2" in sub.columns
         else np.full(t.shape, np.nan))

    i_end, i_fin = nearest_row(sub, t_end), nearest_row(sub, t_final)
    drive, post = (t >= t_on) & (t <= t_end), t > t_end

    d_cold_pk = float(np.nanmin(d_cold[drive]))
    d_hot_pk = float(np.nanmax(d_hot[drive]))
    t_peak_cool = float(t[drive][int(np.argmin(d_cold[drive]))])

    t_cool63 = first_cross(t, drive & (np.abs(d_cold) >= 0.63 * abs(d_cold_pk)))
    t_cool90 = first_cross(t, drive & (np.abs(d_cold) >= 0.90 * abs(d_cold_pk)))
    t_cross = first_cross(t, post & (d_cold > 0.0))
    t_norm = first_cross(t, post & (np.abs(d_cold) < CFG["BACK_TO_NORMAL_K"]))
    if post.any():
        k_ovs = int(np.argmax(d_cold[post]))
        overshoot_cold = float(d_cold[post][k_ovs])
        overshoot_time = float(t[post][k_ovs])
        overshoot_hot = float(np.nanmax(d_hot[post]) - d_hot[i_end])
    else:
        overshoot_cold = overshoot_time = overshoot_hot = np.nan

    W_full = float(i0) * trapz(v1, t)
    W_drive = float(i0) * trapz(v1[drive], t[drive]) if drive.any() else np.nan
    dmin_all = (sub["Delta_Min"].to_numpy(float)
                if "Delta_Min" in sub.columns else np.full(t.shape, np.nan))
    dmax_all = (sub["Delta_Max"].to_numpy(float)
                if "Delta_Max" in sub.columns else np.full(t.shape, np.nan))
    # Effective surface extrema: Delta_Min/Delta_Max probes are dead (~1e-5 K)
    # in the current export (see H6) -> fall back to the Cold/Hot point probes.
    spread_mm = (float(np.nanmax(np.abs(dmin_all[drive]))) if drive.any() else 0.0,
                 float(np.nanmax(np.abs(dmax_all[drive]))) if drive.any() else 0.0)
    alive = (np.isfinite(spread_mm).all()
             and max(spread_mm) > 0.05
             and max(spread_mm) > 0.1 * max(abs(d_cold_pk), d_hot_pk, 1e-9))
    dmin_eff = dmin_all if alive else d_cold
    dmax_eff = dmax_all if alive else d_hot
    row = {
        "I0_A": float(i0),
        "dT_cold_peak_K": d_cold_pk,
        "dT_hot_peak_K": d_hot_pk,
        "dT_cold_end_K": float(d_cold[i_end]),
        "dT_hot_end_K": float(d_hot[i_end]),
        "dT_cold_60_K": float(d_cold[i_fin]),
        "dT_hot_60_K": float(d_hot[i_fin]),
        "dT_max_peak_K": float(np.nanmax(dmax_eff[drive])) if drive.any() else np.nan,
        "dT_min_peak_K": float(np.nanmin(dmin_eff[drive])) if drive.any() else np.nan,
        "dT_max_end_K": float(dmax_eff[i_end]),
        "dT_min_end_K": float(dmin_eff[i_end]),
        "dT_max_60_K": float(dmax_eff[i_fin]),
        "dT_min_60_K": float(dmin_eff[i_fin]),
        "minmax_source": ("probe_MinMax" if alive else "fallback_ColdHot"),
        "ratio_cool_heat": safe_ratio(abs(d_cold_pk), d_hot_pk),
        "cool_per_A_KpA": abs(d_cold_pk) / float(i0),
        "heat_per_A_KpA": d_hot_pk / float(i0),
        "t_peak_cool_s": t_peak_cool,
        "t_cool63_s": t_cool63 - t_on if np.isfinite(t_cool63) else np.nan,
        "t_cool90_s": t_cool90 - t_on if np.isfinite(t_cool90) else np.nan,
        "t_ambient_cross_s": t_cross - t_end if np.isfinite(t_cross) else np.nan,
        "t_back_normal_s": t_norm - t_end if np.isfinite(t_norm) else np.nan,
        "overshoot_cold_K": overshoot_cold,
        "overshoot_time_s": overshoot_time,
        "overshoot_hot_K": overshoot_hot,
        "V1_peak_V": float(np.nanmax(v1[drive])),
        "q_peak_Wm2": float(np.nanmax(np.abs(q[drive]))),
        "W_full_J": W_full,
        "W_drive_J": W_drive,
        "E_areal_J_m2": trapz(np.abs(q[drive]), t[drive]),
        "dose_Ks": trapz(np.abs(d_cold[drive]), t[drive]),
        "K_per_J": safe_ratio(abs(d_cold_pk), W_full),
        "K_per_J_hot": safe_ratio(d_hot_pk, W_full),
        "J_per_K": safe_ratio(W_full, abs(d_cold_pk)),
    }
    for thr in thresholds:
        tag = ("%g" % thr).replace(".", "")
        c_cool, c_heat = vol_col(sub, "cool", thr), vol_col(sub, "heat", thr)
        if c_cool is None or c_heat is None:
            continue
        vc, vh = sub[c_cool].to_numpy(float), sub[c_heat].to_numpy(float)
        row["vol_cool_%s_end" % tag] = float(vc[i_end])
        row["vol_heat_%s_end" % tag] = float(vh[i_end])
        row["vol_cool_%s_60" % tag] = float(vc[i_fin])
        row["vol_heat_%s_60" % tag] = float(vh[i_fin])
        row["vol_cool_%s_peak" % tag] = float(vc.max())
        row["vol_heat_%s_peak" % tag] = float(vh.max())
        row["vol_ratio_%s_end" % tag] = safe_ratio(vc[i_end], vh[i_end])
        row["vol_cool_%s_per_J" % tag] = safe_ratio(vc[i_end], W_full)
    return row


def metrics_table(df, currents=None, thresholds=VOL_THRESHOLDS):
    """Tidy metrics table of the heatsink model (one row per drive current)."""
    if currents is None:
        currents = sorted(set(np.round(df["I0"].unique(), 4)))
    rows = [current_metrics(df, i0, thresholds=thresholds) for i0 in currents]
    return pd.DataFrame([r for r in rows if r])


def val(table, i0, col):
    """Metric value of one current (NaN when the column is missing)."""
    if col not in table.columns:
        return np.nan
    sel = table[np.isclose(table["I0_A"], i0)]
    return float(sel[col].iloc[0]) if len(sel) else np.nan


def series(table, currents, col):
    """Column of the metrics table as a float array over the currents."""
    return np.array([val(table, c, col) for c in currents], dtype=float)


def fmt_scalar(v, prec=2, nan_text="n/a"):
    """Scalar formatting with an explicit placeholder for NaN."""
    return ("%*.*f" % (8, prec, v)) if np.isfinite(v) else "%8s" % nan_text

def print_verdict(table, currents, thresholds=VOL_THRESHOLDS):
    """Per-current console verdict of the heatsink model."""
    tag_low = ("%g" % thresholds[0]).replace(".", "")
    area_ok = "area_cool_%s_end" % tag_low in table.columns
    print("\n" + "=" * 104)
    print("HEATSINK (Concise) - per-current verdict (drive %.0f-%.0f s, run end %.0f s)"
          % (CFG["T_ON_S"], CFG["T_END_S"], CFG["T_FINAL_S"]))
    print("=" * 104)
    for i0 in currents:
        print("-" * 104)
        print("I0 = %.2f A" % i0)
        print("  TEMPERATURE  coldest probe           %s K   (at t = %.1f s)"
              % (fmt_scalar(val(table, i0, "dT_cold_peak_K")),
                 val(table, i0, "t_peak_cool_s")))
        print("               hottest probe           %s K"
              % fmt_scalar(val(table, i0, "dT_hot_peak_K")))
        print("               cooling/heating ratio   %s"
              % fmt_scalar(val(table, i0, "ratio_cool_heat"), 3))
        print("  SPATIAL      cooled volume @%g K       %s mm3   (at run end %s mm3)"
              % (thresholds[0], fmt_scalar(val(table, i0, "vol_cool_%s_end" % tag_low), 1),
                 fmt_scalar(val(table, i0, "vol_cool_%s_60" % tag_low), 1)))
        print("               heated volume @%g K       %s mm3   (at run end %s mm3)"
              % (thresholds[0], fmt_scalar(val(table, i0, "vol_heat_%s_end" % tag_low), 1),
                 fmt_scalar(val(table, i0, "vol_heat_%s_60" % tag_low), 1)))
        print("               volume balance cool/heat %s"
              % fmt_scalar(val(table, i0, "vol_ratio_%s_end" % tag_low), 3))
        if area_ok:
            print("               2D cooled area @%g K      %s mm2"
                  % (thresholds[0],
                     fmt_scalar(val(table, i0, "area_cool_%s_end" % tag_low), 1)))
            print("               2D heated area @%g K      %s mm2"
                  % (thresholds[0],
                     fmt_scalar(val(table, i0, "area_heat_%s_end" % tag_low), 1)))
        print("  TEMPORAL     time to 63%% of cooling    %s s   (from drive start)"
              % fmt_scalar(val(table, i0, "t_cool63_s"), 2))
        print("               ambient crossing        %s s   (after drive end)"
              % fmt_scalar(val(table, i0, "t_ambient_cross_s"), 2))
        print("               back to normal          %s s   (|dTc| < %.1f K)"
              % (fmt_scalar(val(table, i0, "t_back_normal_s"), 2),
                 CFG["BACK_TO_NORMAL_K"]))
        print("               switch-off overshoot    %s K   at t = %s s"
              % (fmt_scalar(val(table, i0, "overshoot_cold_K"), 3),
                 fmt_scalar(val(table, i0, "overshoot_time_s"), 1)))
        print("  ENERGY       W[0-%.0f s]               %s J   (V1 peak %s V)"
              % (CFG["T_FINAL_S"], fmt_scalar(val(table, i0, "W_full_J")),
                 fmt_scalar(val(table, i0, "V1_peak_V"), 3)))
        print("               areal heat energy       %s J/m2"
              % fmt_scalar(val(table, i0, "E_areal_J_m2"), 0))
        print("               cooling per joule       %s K/J   (%s J per K)"
              % (fmt_scalar(val(table, i0, "K_per_J"), 3),
                 fmt_scalar(val(table, i0, "J_per_K"), 2)))
        print("               cooled volume per joule %s mm3/J"
              % fmt_scalar(val(table, i0, "vol_cool_%s_per_J" % tag_low), 2))
    print("=" * 104)


def print_best_points(table, currents, thresholds=VOL_THRESHOLDS):
    """Best operating points of the heatsink model."""
    tag_low = ("%g" % thresholds[0]).replace(".", "")

    def best(col, mode="max"):
        vals = [(i0, val(table, i0, col)) for i0 in currents]
        vals = [(i0, v) for i0, v in vals if np.isfinite(v)]
        if not vals:
            return None
        return (max if mode == "max" else min)(vals, key=lambda p: p[1])

    def show(title, col, prec=2, mode="max"):
        p = best(col, mode)
        if p:
            print("  %-26s I0 = %.1f A   (%s)" % (title, p[0], ("%%.%df" % prec) % p[1]))

    print("\nBEST OPERATING POINTS (heatsink)")
    show("max cooling (coldest probe)", "dT_cold_peak_K", mode="min")
    show("best balance |dTc|/dTh", "ratio_cool_heat", prec=3, mode="max")
    show("least switch-off overshoot", "overshoot_cold_K", prec=3, mode="min")
    show("best cooling per joule", "K_per_J", prec=3, mode="max")
    show("max cooled volume @%g K" % thresholds[0], "vol_cool_%s_end" % tag_low,
         prec=1, mode="max")
    show("best efficiency (mm3/J)", "vol_cool_%s_per_J" % tag_low, prec=2, mode="max")
    print("  note: cooling peaks at a mid current - at 1 A the TEC has to pump "
          "against a larger temperature difference,\n        which reduces both the "
          "cooled volume and the cooling per joule.")

# =============================================================================
# 2. SURFACE LOADER (one current at a time, optional time stride)
# =============================================================================
def load_surface_current(i0, cache=None, stride=None):
    """Stream ONE drive current of the inner-surface export and cache the frame.

    The export holds 401 time steps x 4317 nodes per current (17.3 M rows in
    total), so only one current is kept in memory and - for the plotting stages -
    only every SPATIAL_TIME_STRIDE-th time step.  The area metrics always use the
    full series (they are cached separately).
    """
    stride = CFG["SPATIAL_TIME_STRIDE"] if stride is None else stride
    key = (round(float(i0), 4), int(stride or 1))
    if cache is not None and key in cache:
        return cache[key]

    t0 = time.time()
    df = load_surface_inner(CFG["SURF_CSV"], target_currents=[i0],
                            default_tamb_c=CFG["T_AMB_C"],
                            chunksize=CFG["SURFACE_CHUNK"])
    df["time_s"] = np.round(df["time_s"].to_numpy(float), 4)
    df["I0"] = np.round(df["I0"].to_numpy(float), 4)
    if stride and int(stride) > 1:
        keep = np.unique(df["time_s"].to_numpy(float))[::int(stride)]
        df = df[df["time_s"].isin(keep)].reset_index(drop=True)

    print("      loaded %s @ I0=%.2f A -> %d rows, %d time steps (%.1f s)"
          % (os.path.basename(CFG["SURF_CSV"]), i0, len(df),
             df["time_s"].nunique(), time.time() - t0))
    if cache is not None:
        cache[key] = df
    return df


# =============================================================================
# 2b. INNER-SURFACE FIELD METRICS (Config_heatsink_surf.csv)
# -----------------------------------------------------------------------------
# heatsink_metrics.csv used to carry PROBE temperatures only (hot/cold probes,
# plus the Min/Max probes that are constant in this export and therefore
# silently fall back to them).  The experimental reference is an OUTER-SURFACE
# IR measurement, so the only simulated values that may ever be compared with
# it are the values of the simulated SURFACE FIELD, extracted here in ONE
# streaming pass and stored with a `surf_` prefix.
#
# HOUSE RULE: NEVER COMPARE DIRECT PROBES WITH EXPERIMENTAL DATA.
# =============================================================================
SURF_METRIC_COLS = (
    "surf_source", "surf_n_nodes", "surf_n_times",
    "surf_dT_min_peak_K", "surf_dT_max_peak_K",
    "surf_dT_min_end_K", "surf_dT_max_end_K",
    "surf_dT_min_60_K", "surf_dT_max_60_K",
    "surf_dT_mean_end_K", "surf_dT_mean_60_K",
    "surf_spread_peak_K", "surf_spread_end_K",
    "surf_std_peak_K", "surf_std_end_K",
    "surf_t_peak_cool_s", "surf_t_cool63_s", "surf_t_cool90_s",
    "surf_t_ambient_cross_s", "surf_t_back_normal_s",
    "surf_t_recover63_s", "surf_t_recover90_s", "surf_recovery_frac_end",
    "surf_recovered_by_final", "surf_overshoot_cold_K",
    "surf_ratio_cool_heat", "surf_cool_per_A_KpA", "surf_heat_per_A_KpA",
    "surf_K_per_J", "surf_K_per_J_drive", "surf_J_per_K", "surf_dose_Ks",
)
SURF_TRACE_COLS = ("I0_A", "time_s", "time_into_pulse_s", "dT_min_K", "dT_max_K",
                   "dT_mean_K", "dT_std_K", "spread_K")


def _num(v):
    """float() that returns NaN instead of raising on blank/odd cells."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _surface_header(file_path):
    """Sniff the (semicolon separated) header row of a surface export."""
    with open(file_path, encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            s = line.lstrip("%").strip()
            if s.startswith("x;") or "x;y;z" in s:
                return idx, [c.strip() for c in s.split(";")]
    return 8, ["x", "y", "z", "I0", "t", "T (C)"]


def collect_surface_stats(file_path, currents, t_amb_c=20.0, chunk=None):
    """Per-current statistics of the WHOLE surface export in ONE pass.

    The heatsink export holds 401 time steps x 2665 nodes x 5 currents, so the
    full frames (~1 M rows per current) are never materialised: only
    count/sum/sumsq/min/max are accumulated per (I0, t).  That is everything
    the surface metrics need (frame extrema, mean, std, spread) at a few MB of
    memory, and it keeps the heatsink script independent of any other module.

    Returns {I0: {times, n_nodes, min, max, mean, std, t_amb_c, path, I0}}.
    """
    if not file_path or not os.path.exists(file_path):
        print("      [skip] surface export not found: %s" % file_path)
        return {}
    header_row, cols = _surface_header(file_path)
    c_i0 = next((c for c in cols if c.lower() == "i0"), cols[3])
    c_t = next((c for c in cols if c.lower() in ("t", "time", "time_s")), cols[4])
    c_temp = cols[-1]
    keep = [c_i0, c_t, c_temp]
    acc, t0 = {}, time.time()
    for part in pd.read_csv(file_path, sep=";", skiprows=header_row + 1,
                            header=None, names=cols, chunksize=chunk or CFG["SURFACE_CHUNK"],
                            low_memory=False):
        sub = part[keep].copy()
        for c in keep:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        i0_raw = sub[c_i0].to_numpy(float)
        mask = np.zeros(sub.shape[0], dtype=bool)
        for cur in currents:
            m = np.isclose(i0_raw, cur, atol=1e-3)
            if m.any():
                mask |= m
        if not mask.any():
            continue
        g = pd.DataFrame({"i0": np.round(i0_raw[mask], 4),
                          "t": np.round(sub[c_t].to_numpy(float)[mask], 4),
                          "v": sub[c_temp].to_numpy(float)[mask] - float(t_amb_c)})
        g = g[np.isfinite(g["v"].to_numpy(float))]
        if g.empty:
            continue
        g["sq"] = g["v"].to_numpy(float) ** 2
        agg = (g.groupby(["i0", "t"], sort=True)
                 .agg(n=("v", "size"), s=("v", "sum"), ss=("sq", "sum"),
                      lo=("v", "min"), hi=("v", "max"))
                 .reset_index())
        for i0r, tr, n_, s_, ss_, lo_, hi_ in agg.itertuples(index=False, name=None):
            key = (round(float(i0r), 4), round(float(tr), 4))
            got = acc.get(key)
            if got is None:
                acc[key] = [int(n_), float(s_), float(ss_), float(lo_), float(hi_)]
            else:
                got[0] += int(n_)
                got[1] += float(s_)
                got[2] += float(ss_)
                got[3] = min(got[3], float(lo_))
                got[4] = max(got[4], float(hi_))

    steps = {}
    for (i0r, tr), st in acc.items():
        steps.setdefault(i0r, []).append((tr, st))
    out = {}
    for cur in currents:
        got = None
        for k in steps:
            if np.isclose(k, cur):
                got = steps[k]
                break
        if not got:
            continue
        got.sort(key=lambda p: p[0])
        times = np.array([p[0] for p in got], float)
        a = np.array([p[1] for p in got], float)
        n, s, ss, lo, hi = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
        mean = s / n
        std = np.sqrt(np.maximum(ss / n - mean ** 2, 0.0))
        out[float(cur)] = {"path": file_path, "I0": float(cur),
                           "t_amb_c": float(t_amb_c), "times": times,
                           "n_nodes": n, "min": lo, "max": hi,
                           "mean": mean, "std": std}
    print("      inner-surface stats for currents %s (%d (I0,t) samples, %.1f s)"
          % (sorted(out), len(acc), time.time() - t0))
    return out


def surface_trace(surf_stats, i0):
    """(time, surface min, surface max) of one current (None when unavailable)."""
    if not surf_stats:
        return None
    st = None
    for k, v in surf_stats.items():
        if np.isclose(k, i0):
            st = v
            break
    if st is None:
        return None
    return st["times"], st["min"], st["max"]


def probe_reference(ax, df, i0, col, label=None, color="0.55"):
    """Draw one point probe as a thin dotted REFERENCE line.

    Point probes are never the main curve of a temporal plot: they are not the
    quantity that can be compared with an experimental surface measurement.
    """
    if df is None or col not in df.columns:
        return False
    sub = sub_current(df, i0)
    if sub is None:
        return False
    ax.plot(sub["time_s"], sub[col], color=color, lw=3.4, ls=":", alpha=0.9,
            zorder=2, label=label)
    return True


def _cross_time(t, y, level, above=False, window=None):
    """First time inside `window` where y crosses `level` (NaN when never)."""
    m = np.ones(t.size, dtype=bool) if window is None else np.asarray(window, bool)
    idx = np.where(m)[0]
    if idx.size < 2:
        return np.nan
    y_w, t_w = y[idx], t[idx]
    for j in range(1, y_w.size):
        a, b = y_w[j - 1], y_w[j]
        if (not above and a >= level > b) or (above and a <= level < b):
            f = (a - level) / (a - b) if a != b else 0.0
            return float(t_w[j - 1] + f * (t_w[j] - t_w[j - 1]))
    return np.nan

def surface_current_metrics(st, i0, row=None):
    """Surface-field metrics of one current.

    These are the ONLY simulated values that may be compared with the
    experimental outer-surface IR metrics: the probe values of the same current
    are deliberately ignored here.
    """
    out = {k: np.nan for k in SURF_METRIC_COLS}
    out["surf_source"] = "inner_surface_field"
    if not st:
        out["surf_source"] = "no_surface_data"
        return out
    t, lo, hi = st["times"], st["min"], st["max"]
    mean, std, n = st["mean"], st["std"], st["n_nodes"]
    t_on, t_end, t_fin = float(CFG["T_ON_S"]), float(CFG["T_END_S"]), float(CFG["T_FINAL_S"])
    dr = (t >= t_on - 1e-9) & (t <= t_end + 1e-9)
    post = t > t_end + 1e-9
    k_end = int(np.argmin(np.abs(t - t_end)))
    k_fin = int(np.argmin(np.abs(t - t_fin)))
    out["surf_n_nodes"] = float(np.median(n))
    out["surf_n_times"] = float(t.size)
    if not dr.any():
        return out

    cold_pk = float(lo[dr].min())          # coldest surface point (negative)
    hot_pk = float(hi[dr].max())           # hottest surface point
    cold_end = float(lo[k_end])
    i0f = _num(i0)
    w_full = _num(row.get("W_full_J")) if row is not None else np.nan
    w_drive = _num(row.get("W_drive_J")) if row is not None else np.nan
    out.update({
        "surf_dT_min_peak_K": cold_pk,
        "surf_dT_max_peak_K": hot_pk,
        "surf_dT_min_end_K": cold_end,
        "surf_dT_max_end_K": float(hi[k_end]),
        "surf_dT_min_60_K": float(lo[k_fin]),
        "surf_dT_max_60_K": float(hi[k_fin]),
        "surf_dT_mean_end_K": float(mean[k_end]),
        "surf_dT_mean_60_K": float(mean[k_fin]),
        "surf_spread_peak_K": float((hi - lo)[dr].max()),
        "surf_spread_end_K": float(hi[k_end] - lo[k_end]),
        "surf_std_peak_K": float(std[dr].max()),
        "surf_std_end_K": float(std[k_end]),
        "surf_t_peak_cool_s": float(t[dr][int(np.argmin(lo[dr]))]),
        "surf_overshoot_cold_K": (float(lo[post].max() - cold_end) if post.any() else np.nan),
        "surf_t_back_normal_s": _cross_time(t, lo, -CFG["BACK_TO_NORMAL_K"], above=True, window=post),
        "surf_t_ambient_cross_s": _cross_time(t, lo, 0.0, above=True, window=post),
        "surf_dose_Ks": trapz(np.abs(lo[dr]), t[dr]),
        "surf_t_cool63_s": _cross_time(t, lo, 0.63 * cold_pk, window=dr),
        "surf_t_cool90_s": _cross_time(t, lo, 0.90 * cold_pk, window=dr),
        "surf_t_recover63_s": (_cross_time(t, lo, cold_end * (1.0 - 0.63),
                                            above=True, window=post)
                               if (post.any() and cold_end < 0) else np.nan),
        "surf_t_recover90_s": (_cross_time(t, lo, cold_end * (1.0 - 0.90),
                                            above=True, window=post)
                               if (post.any() and cold_end < 0) else np.nan),
        "surf_recovery_frac_end": (float((lo[k_fin] - cold_end) / (-cold_end))
                                   if (cold_end < 0) else np.nan),
        "surf_recovered_by_final": bool(np.isfinite(out["surf_t_back_normal_s"])),
        "surf_ratio_cool_heat": (abs(cold_pk) / hot_pk if hot_pk else np.nan),
        "surf_cool_per_A_KpA": (abs(cold_pk) / i0f * 1e3 if i0f else np.nan),
        "surf_heat_per_A_KpA": (hot_pk / i0f * 1e3 if i0f else np.nan),
    })
    if np.isfinite(w_full) and cold_pk:
        out["surf_K_per_J"] = abs(cold_pk) / w_full
        out["surf_J_per_K"] = w_full / abs(cold_pk)
    if np.isfinite(w_drive) and cold_pk:
        out["surf_K_per_J_drive"] = abs(cold_pk) / w_drive
    return out


def add_surface_to_table(table, surf_stats):
    """Merge the `surf_*` (inner-surface field) columns into the metrics table."""
    if not surf_stats:
        for c in SURF_METRIC_COLS:
            table[c] = "no_surface_data" if c == "surf_source" else np.nan
        return table
    rows = []
    for _, r in table.iterrows():
        st = None
        for k, v in surf_stats.items():
            if np.isclose(k, _num(r.get("I0_A"))):
                st = v
                break
        rows.append(surface_current_metrics(st, r.get("I0_A"), r))
    extra = pd.DataFrame(rows, columns=list(SURF_METRIC_COLS))
    return pd.concat([table.reset_index(drop=True), extra], axis=1)


def write_surface_traces(surf_stats, path):
    """Dump the surface extrema so no other script has to re-stream the export."""
    if not surf_stats:
        return None
    frames = []
    for i0 in sorted(surf_stats):
        st = surf_stats[i0]
        frames.append(pd.DataFrame({
            "I0_A": float(i0),
            "time_s": st["times"],
            "time_into_pulse_s": st["times"] - float(CFG["T_ON_S"]),
            "dT_min_K": st["min"], "dT_max_K": st["max"],
            "dT_mean_K": st["mean"], "dT_std_K": st["std"],
            "spread_K": st["max"] - st["min"],
        }))
    traces = pd.concat(frames, ignore_index=True)[list(SURF_TRACE_COLS)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    traces.to_csv(path, index=False)
    print("      saved %s (%d rows)" % (path, len(traces)))
    return traces


# =============================================================================
# 3. PROBE / TEMPORAL FIGURES
# =============================================================================
def drive_band(ax, color="0.88", alpha=0.55):
    """Shade the 25 s drive pulse and mark its boundaries."""
    ax.axvspan(CFG["T_ON_S"], CFG["T_END_S"], color=color, alpha=alpha, lw=0, zorder=0)
    ax.axvline(CFG["T_ON_S"], color="0.55", lw=3.4, ls="--", zorder=1)
    ax.axvline(CFG["T_END_S"], color="0.55", lw=3.4, ls="--", zorder=1)


def fig_h1_probes_grid(df, currents, index, surf_stats=None,
                       fname="H1_probes_grid.png"):
    """Surface max/min response of every drive current (small multiples).

    Row (a) is the hot side (inner-surface FIELD max), row (b) the cold side
    (FIELD min).  The $I_0$ titles sit on the top row only and there is no
    figure suptitle.
    """
    rows = [("max", r"$\Delta T_h$ (Â°C)"),
            ("min", r"$\Delta T_c$ (Â°C)")]
    n = len(currents)
    peaks, valleys = [], []
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n + 1.6, 10.2), sharex=True,
                             sharey="row", squeeze=False)
    for j, i0 in enumerate(currents):
        tr = surface_trace(surf_stats, i0)
        for i, (mode, lab) in enumerate(rows):
            ax = axes[i][j]
            drive_band(ax)
            if tr is not None:
                t, mn, mx = tr
                y = mx if mode == "max" else mn
                ax.plot(t, y, color=C_MAIN, lw=5, zorder=3)
                v = np.asarray(y, float)
                k = int(np.nanargmax(v)) if mode == "max" else int(np.nanargmin(v))
                ax.plot(t[k], v[k], "o", ms=12.6, mfc="white",
                        mec=C_MAIN, mew=1.3, zorder=5)
                ax.annotate("%.1f" % v[k], (t[k], v[k]),
                            xytext=(3, 4 if mode == "max" else -10),
                            textcoords="offset points", fontsize=30, color=C_MAIN)
                if mode == "max":
                    peaks.append(float(np.nanmax(v)))
                else:
                    valleys.append(float(np.nanmin(v)))
            ax.axhline(0, color="0.4", lw=3.4, ls=":", zorder=1)
            ax.tick_params(labelsize=32)
            if i == 0:
                ax.set_title("$I_0$ = %.1f A" % i0, fontsize=32)  # top row only
            else:
                ax.set_xlabel("Time (s)", fontsize=32)
            if j == 0:
                ax.set_ylabel(lab, fontsize=32)
    # per-row headroom so the peak (top) / valley (bottom) labels never clip
    if peaks:
        hi = max(peaks)
        axes[0][0].set_ylim(-0.06 * hi, 1.18 * hi)
    if valleys:
        lo = min(valleys)
        axes[1][0].set_ylim(1.18 * lo, -0.06 * lo)
    axes[0][0].set_xlim(0, CFG["T_FINAL_S"])
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 1.0))
    save_figure(fig, fname, index)

def fig_h2_focus(df, i0, index, surf_stats=None, fname="H2_focus_transients.png"):
    """Focus on one current: surface extrema, drive voltage, flux, volumes."""
    sub = sub_current(df, i0)
    if sub is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(26.0, 15.86))

    ax = axes[0][0]
    drive_band(ax)
    tr = surface_trace(surf_stats, i0)
    if tr is not None:
        t, mn, mx = tr
        ax.plot(t, mn, color="#2166ac", lw=5.6, label=r"surface min $\Delta T_c$")
        ax.plot(t, mx, color="#b2182b", lw=5.6, label=r"surface max $\Delta T_h$")
    probe_reference(ax, df, i0, "Delta_Cold", label=r"cold probe (ref.)")
    probe_reference(ax, df, i0, "Delta_Hot", label=r"hot probe (ref.)")
    ax.axhline(0, color="0.4", lw=3.4, ls=":")
    ax.set_title("(a) Surface response (probes = dotted reference)", fontsize=35)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\Delta T$ (Â°C)")
    ax.set_xlim(0, CFG["T_FINAL_S"])
    ax.legend(fontsize=32, loc="best")

    ax = axes[0][1]
    drive_band(ax)
    ax.plot((sub["time_s"]-CFG["T_ON_S"]), np.abs(sub["voltage_tec1_V"]), color="#2ca02c", lw=5.6)
    ax.set_title(r"(b) Driven TEC terminal voltage $|V_1|$", fontsize=35)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_xlim(0, CFG["T_FINAL_S"])

    ax = axes[1][0]
    drive_band(ax)
    ax.plot((sub["time_s"]-CFG["T_ON_S"]), np.abs(sub["flux_Wm2"]), color="#9467bd", lw=5.6)
    ax.set_title(r"(c) Boundary heat flux $|q|$", fontsize=35)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$|q|$ (W/m$^2$)")
    ax.set_xlim(0, CFG["T_FINAL_S"])

    ax = axes[1][1]
    drive_band(ax)
    for thr in VOL_THRESHOLDS:
        c_cool, c_heat = vol_col(sub, "cool", thr), vol_col(sub, "heat", thr)
        if c_cool is None or c_heat is None:
            continue
        ax.plot((sub["time_s"]-CFG["T_ON_S"]), sub[c_cool], color=THR_COLORS[thr], lw=5.3)
        ax.plot((sub["time_s"]-CFG["T_ON_S"]), sub[c_heat], color=THR_COLORS[thr], lw=4.2, ls="--")
        ax.annotate("%g K" % thr, (CFG["T_FINAL_S"], sub[c_cool].iloc[-1]),
                    xytext=(3, -3), textcoords="offset points", fontsize=26,
                    color=THR_COLORS[thr])
    ax.set_title("(d) Cooled (solid) and heated (dashed) volume", fontsize=35)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Volume (mm$^3$)")
    ax.set_xlim(0, CFG["T_FINAL_S"])
    ax.set_ylim(bottom=0)
    handles = [Line2D([], [], color="0.35", lw=5, label="cooled (below ambient)"),
               Line2D([], [], color="0.35", lw=4.5, ls="--", label="heated (above ambient)")]
    ax.legend(handles=handles, fontsize=32, loc="upper left")

    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout()
    save_figure(fig, fname, index)


def fig_h3_overlay(df, currents, index, surf_stats=None,
                       fname="H3_all_currents_overlay.png"):
    """All drive currents overlaid for the four main signals (colour = current)."""
    lo, hi = min(currents), max(currents)
    cmap = plt.get_cmap(CMAP_CURRENT)
    norm = plt.Normalize(lo, hi)
    panels = [("surf_min", r"(a) Surface min $\Delta T_c$ (Â°C)", r"$\Delta T_c$ (Â°C)"),
              ("surf_max", r"(b) Surface max $\Delta T_h$ (Â°C)", r"$\Delta T_h$ (Â°C)"),
              ("abs_flux", r"(c) Heat flux $|q|$ (W/m$^2$)", r"$|q|$ (W/m$^2$)"),
              ("abs_v1", r"(d) Drive voltage $|V_1|$ (V)", "V (V)")]
    fig, axes = plt.subplots(2, 2, figsize=(27.04, 15.86))
    for ax, (col, ttl, ylab) in zip(axes.ravel(), panels):
        drive_band(ax)
        for i0 in currents:
            sub = sub_current(df, i0)
            if sub is None:
                continue
            if col == "abs_flux":
                y = np.abs(sub["flux_Wm2"].to_numpy(float))
            elif col == "abs_v1":
                y = np.abs(sub["voltage_tec1_V"].to_numpy(float))
            elif col in ("surf_min", "surf_max"):
                tr = surface_trace(surf_stats, i0)
                if tr is None:
                    continue
                t, mn, mx = tr
                ax.plot(t, mx if col == "surf_max" else mn,
                        color=cmap(norm(i0)), lw=4.8)
                continue
            else:
                y = sub[col].to_numpy(float)
            ax.plot((sub["time_s"]-CFG["T_ON_S"]), y, color=cmap(norm(i0)), lw=4.8)
        ax.set_title(ttl, fontsize=35)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, CFG["T_FINAL_S"])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                          fraction=0.046, pad=0.02)
        cb.set_label(r"Drive current $I_0$ (A)", fontsize=32)
        cb.ax.tick_params(labelsize=32)
    axes[0][0].axhline(0, color="0.4", lw=3.4, ls=":")
    axes[0][1].axhline(0, color="0.4", lw=3.4, ls=":")
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout()
    save_figure(fig, fname, index)

def fig_h4_normalized_response(df, table, currents, index, surf_stats=None,
                               fname="H4_normalized_response.png"):
    """Dynamics: normalised step response and the response times vs current."""
    fig, axes = plt.subplots(1, 3, figsize=(31.2, 10.01))
    cmap = plt.get_cmap(CMAP_CURRENT)
    norm = plt.Normalize(min(currents), max(currents))

    for kind, sign, ttl, ax in (
            ("min", -1.0, r"(a) Surface min $\Delta T_c/|\Delta T_{c,pk}|$", axes[0]),
            ("max", 1.0, r"(b) Surface max $\Delta T_h/\Delta T_{h,pk}$", axes[1])):
        for i0 in currents:
            tr = surface_trace(surf_stats, i0)
            if tr is None:
                continue
            t, mn, mx = tr
            y = mx if kind == "max" else mn
            pk = float(np.nanmax(np.abs(y)))
            if not np.isfinite(pk) or pk <= 0:
                continue
            ax.plot(t - CFG["T_ON_S"], y / pk, color=cmap(norm(i0)), lw=4.8)
        for level in (0.63, 0.90):
            ax.axhline(sign * level, color="0.55", lw=3.4, ls=":")
        ax.axhline(0, color="0.4", lw=3.4)
        ax.set_xlim(-0.5, 25)
        ax.set_yticks(np.arange(-1.0, 1.01, 0.25))
        ax.set_title(ttl, fontsize=35)
        ax.set_xlabel(r"Time (s)")
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                          fraction=0.046, pad=0.02)
        cb.set_label(r"$I_0$ (A)", fontsize=32)
        cb.ax.tick_params(labelsize=32)
    axes[0].set_ylabel("normalised response (-)")

    ax = axes[2]
    ax.plot(currents, series(table, currents, "surf_t_cool63_s"), "-o", color="#2166ac",
            lw=5.6, ms=14, label=r"$\tau_{63}$ (63 % of surface peak)")
    ax.plot(currents, series(table, currents, "surf_t_cool90_s"), "--s", color="#b2182b",
            lw=5, ms=14, mfc="white", label=r"$\tau_{90}$ (90 % of surface peak)")
    ax.set_title("(c) Cooling response times", fontsize=35)
    ax.set_xlabel(r"Drive current $I_0$ (A)")
    ax.set_ylabel("time from drive start (s)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=32)
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout()
    save_figure(fig, fname, index)

def fig_h5_switch_off(df, table, currents, index, surf_stats=None,
                       fname="H5_switch_off.png"):
    """Switch-off zoom, overshoot versus current and time back to normal."""
    fig, axes = plt.subplots(1, 3, figsize=(31.2, 10.01))
    cmap = plt.get_cmap(CMAP_CURRENT)
    norm = plt.Normalize(min(currents), max(currents))

    ax = axes[0]
    for i0 in currents:
        tr = surface_trace(surf_stats, i0)
        if tr is None:
            continue
        t, mn, mx = tr
        m = (t >= CFG["T_END_S"] - 2) & (t <= CFG["T_FINAL_S"])
        ax.plot(t[m], mn[m], color=cmap(norm(i0)), lw=5)
    ax.axhline(0, color="0.4", lw=3.4)
    ax.axvline(CFG["T_END_S"], color="0.5", lw=3.4, ls="--")
    ax.annotate("drive end", xy=(CFG["T_END_S"] + 0.4, 0.95),
                xycoords=("data", "axes fraction"), fontsize=32, color="0.35")
    ax.set_title(r"(a) Surface min after switch-off (positive = overshoot)", fontsize=35)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\Delta T_c$ (Â°C)")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      fraction=0.046, pad=0.02)
    cb.set_label(r"$I_0$ (A)", fontsize=32)
    cb.ax.tick_params(labelsize=32)

    ax = axes[1]
    dv = series(table, currents, "surf_overshoot_cold_K")
    ax.bar(np.arange(len(currents)), np.nan_to_num(dv), 0.62, color=C_MAIN,
           edgecolor="k", lw=3.4)
    for xi, v in zip(np.arange(len(currents)), dv):
        if np.isfinite(v):
            ax.annotate("%.2f" % v, (xi, v + 0.03), ha="center", fontsize=26, color="0.25")
    ax.set_xticks(np.arange(len(currents)))
    ax.set_xticklabels(["%.1f" % c for c in currents], fontsize=32)
    ax.axhline(0, color="0.4", lw=3.4)
    ax.set_title("(b) Overshoot magnitude", fontsize=35)
    ax.set_xlabel(r"Drive current $I_0$ (A)")
    ax.set_ylabel(r"max $\Delta T_c$ after switch-off (Â°C)")

    ax = axes[2]
    # The absolute crossings (0 K / |dT| < BACK_TO_NORMAL_K) are never reached
    # inside the 60 s window for any current, so the panel uses the fraction of
    # the cold drop that is actually recovered instead - always defined.
    r63 = np.asarray(series(table, currents, "surf_t_recover63_s"), float) - CFG["T_END_S"]
    r90 = np.asarray(series(table, currents, "surf_t_recover90_s"), float) - CFG["T_END_S"]
    frac = np.asarray(series(table, currents, "surf_recovery_frac_end"), float)
    ax.plot(currents, r63, "-o", color="#2ca02c", lw=5.6, ms=14,
            label=r"63 % of $|\Delta T_{c,end}|$ recovered")
    ax.plot(currents, r90, "--s", color="#9467bd", lw=5, ms=14, mfc="white",
            label=r"90 % of $|\Delta T_{c,end}|$ recovered")
    ax.set_title("(c) Recovery after switch-off", fontsize=35)
    ax.set_xlabel(r"Drive current $I_0$ (A)")
    ax.set_ylabel("time after drive end (s)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=32, loc="upper right", framealpha=0.92)
    # second axis: how much of the drop is still missing at t = 60 s
    ax2 = ax.twinx()
    ax2.plot(currents, 100.0 * frac, ":^", color="#2166ac", lw=4.5, ms=14)
    ax2.set_ylabel("recovered at 60 s (%)", fontsize=31, color="#2166ac")
    ax2.tick_params(axis="y", labelcolor="#2166ac", labelsize=32)
    ax2.set_ylim(0, 105)
    note = ("never reaches 0 K nor $|\Delta T_c|$ < %.1f K within %.0f s"
            % (CFG["BACK_TO_NORMAL_K"], CFG["T_FINAL_S"]))
    ax.annotate(note, xy=(0.98, 0.03), xycoords="axes fraction", fontsize=26,
                color="0.35", ha="right", va="bottom", style="italic")
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout()
    save_figure(fig, fname, index)

def fig_h6_extrema_probe_check(df, currents, focus, index,
                               fname="H6_extrema_probe_check.png"):
    """Data-quality check of the T_Max/T_Min surface-extrema probes.

    In the current export these two probes stay constant at the ambient value for
    the whole run, so no other figure is derived from them; this panel documents
    that instead of hiding it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(26.0, 9.62))

    ax = axes[0]
    drive_band(ax)
    for i0 in currents:
        sub = sub_current(df, i0)
        if sub is None:
            continue
        if "T_Max" in sub.columns:
            ax.plot((sub["time_s"]-CFG["T_ON_S"]), sub["T_Max"], lw=3.4, color="0.6")
        if "T_Min" in sub.columns:
            ax.plot((sub["time_s"]-CFG["T_ON_S"]), sub["T_Min"], lw=3.4, color="0.35", ls="--")
    ax.set_title(r"(a) Raw $T_{Max}$ (grey) and $T_{Min}$ (dashed) probes, all currents",
                 fontsize=35)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (Â°C)")
    ax.set_xlim(0, CFG["T_FINAL_S"])
    handles = [Line2D([], [], color="0.6", lw=3.9, label=r"$T_{Max}$ (all currents)"),
               Line2D([], [], color="0.35", lw=3.9, ls="--", label=r"$T_{Min}$ (all currents)")]
    ax.legend(handles=handles, fontsize=32, loc="center left")

    ax = axes[1]
    drive_band(ax)
    sub = sub_current(df, focus)
    if sub is not None:
        if "T_Max" in sub.columns:
            ax.plot((sub["time_s"]-CFG["T_ON_S"]), sub["T_Max"] - CFG["T_AMB_C"], color="0.6", lw=4.5,
                    label=r"$\Delta T_{Max}$ probe")
        if "T_Min" in sub.columns:
            ax.plot((sub["time_s"]-CFG["T_ON_S"]), sub["T_Min"] - CFG["T_AMB_C"], color="0.35", lw=4.5,
                    ls="--", label=r"$\Delta T_{Min}$ probe")
        ax.plot((sub["time_s"]-CFG["T_ON_S"]), sub["Delta_Hot"], color="#b2182b", lw=5,
                label=r"hot probe $\Delta T_h$ (reference)")
        ax.plot((sub["time_s"]-CFG["T_ON_S"]), sub["Delta_Cold"], color="#2166ac", lw=5,
                label=r"cold probe $\Delta T_c$ (reference)")
    ax.axhline(0, color="0.4", lw=3.4, ls=":")
    ax.set_title(r"(b) $I_0$ = %.2f A: the extrema probes do not follow the physics" % focus,
                 fontsize=35)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\Delta T$ (Â°C)")
    ax.set_xlim(0, CFG["T_FINAL_S"])
    ax.legend(fontsize=32, loc="best")
    spread_max = float(df["T_Max"].max() - df["T_Max"].min()) if "T_Max" in df.columns else np.nan
    spread_min = float(df["T_Min"].max() - df["T_Min"].min()) if "T_Min" in df.columns else np.nan
    spread_hot = float(df["T_Hot"].max() - df["T_Hot"].min())
    spread_cold = float(df["T_Cold"].max() - df["T_Cold"].min())
    ax.annotate("DATA QUALITY: the extrema probes are essentially dead here --\n"
                "spread over the whole run: $T_{Max}$ %.4f K, $T_{Min}$ %.4f K,\n"
                "while the hot/cold probes move %.1f / %.1f K.\n"
                "Re-check the COMSOL probes; no figure is derived from them."
                % (spread_max, spread_min, spread_hot, spread_cold),
                xy=(0.03, 0.06), xycoords="axes fraction", fontsize=32, color="#a00000")
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout()
    save_figure(fig, fname, index)

# =============================================================================
# 4. SPATIAL FIGURES - VOLUME PROBES (domain, mm^3)
# =============================================================================
def fig_h7_volume_transients(df, currents, index, fname="H7_volume_transients.png"):
    """Cooled and heated volume versus time for every threshold and current."""
    n = len(currents)
    fig, axes = plt.subplots(2, n, figsize=(3.7 * n + 1.6, 10.6), sharex=True,
                             sharey="row", squeeze=False)
    for j, i0 in enumerate(currents):
        sub = sub_current(df, i0)
        for i, kind in enumerate(("cool", "heat")):
            ax = axes[i][j]
            drive_band(ax)
            if sub is not None:
                for thr in VOL_THRESHOLDS:
                    c = vol_col(sub, kind, thr)
                    if c is None:
                        continue
                    ax.plot((sub["time_s"]-CFG["T_ON_S"]), sub[c], color=THR_COLORS[thr], lw=4.8)
            ax.set_title("$I_0$ = %.1f A" % i0, fontsize=32)
            ax.tick_params(labelsize=32)
            ax.set_ylim(bottom=0)
            if i == 1:
                ax.set_xlabel("Time (s)", fontsize=32)
            if j == 0:
                ax.set_ylabel("cooled volume (mm$^3$)" if i == 0
                              else "heated volume (mm$^3$)", fontsize=32)
                handles = [Line2D([], [], color=THR_COLORS[t], lw=4.8, label="%g K" % t)
                           for t in VOL_THRESHOLDS]
                ax.legend(handles=handles, fontsize=26, ncol=2, loc="upper left")
    axes[0][0].set_xlim(0, CFG["T_FINAL_S"])
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout()
    save_figure(fig, fname, index)

def fig_h8_volume_vs_current(table, currents, index, fname="H8_volume_vs_current.png"):
    """Volume versus current at the reference times (drive end, run end, peak)."""
    fig, axes = plt.subplots(2, 2, figsize=(25, 21))
    specs = [("cool", "end", "(a) Cooled volume at drive end (%.0f s)" % CFG["T_END_S"]),
             ("heat", "end", "(b) Heated volume at drive end (%.0f s)" % CFG["T_END_S"]),
             ("cool", "60", "(c) Cooled volume at run end (%.0f s)" % CFG["T_FINAL_S"]),
             ("heat", "60", "(d) Heated volume at run end (%.0f s)" % CFG["T_FINAL_S"])]
             
    for ax, (kind, when, ttl) in zip(axes.ravel(), specs):
        for thr in VOL_THRESHOLDS:
            tag = ("%g" % thr).replace(".", "")
            y = series(table, currents, "vol_%s_%s_%s" % (kind, tag, when))
            ax.plot(currents, y, "-o", color=THR_COLORS[thr], lw=5.3, ms=12.6,
                    label="%g ºC" % thr)
            
        ax.set_title(ttl, fontsize=35)
        
        # 1. Set axis label (title) font sizes
        ax.set_xlabel(r"Drive current $I_0$ (A)", fontsize=40)
        ax.set_ylabel("volume (mm$^3$)", fontsize=40)
        
        # 2. Set axis numbers (tick label) font sizes
        ax.tick_params(axis='both', which='major', labelsize=40)
        
        ax.set_ylim([0, 100])

    # Grab handles and labels from one of the subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    # 3. Set legend label font size
    fig.legend(handles, labels, fontsize=32, loc='center left', 
               bbox_to_anchor=(1.01, 0.5), frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 1.0])
    save_figure(fig, fname, index)

def fig_h9_volume_balance(table, currents, index, fname="H9_volume_balance.png"):
    """Cooled / heated volume balance per threshold, plus grouped bars."""
    fig, axes = plt.subplots(1, 2, figsize=(27.04, 10.14))

    ax = axes[0]
    for thr in VOL_THRESHOLDS:
        tag = ("%g" % thr).replace(".", "")
        y = series(table, currents, "vol_ratio_%s_end" % tag)
        ax.plot(currents, y, "-o", color=THR_COLORS[thr], lw=5.3, ms=12.6,
                label="%g K" % thr)
    ax.axhline(1.0, color="0.4", lw=3.4, ls=":")
    ax.annotate("break-even: cooled volume = heated volume", xy=(0.03, 0.9),
                xycoords="axes fraction", fontsize=32, color="0.3")
    ax.set_yscale("log")
    ax.set_title("(a) Balance cooled / heated volume at drive end", fontsize=35)
    ax.set_xlabel(r"Drive current $I_0$ (A)")
    ax.set_ylabel("cooled / heated  (-, log)")
    ax.legend(fontsize=32, ncol=2)

    ax = axes[1]
    x = np.arange(len(currents))
    w = 0.2
    for k, thr in enumerate(VOL_THRESHOLDS):
        tag = ("%g" % thr).replace(".", "")
        ax.bar(x + (k - 1.5) * w, np.nan_to_num(series(table, currents, "vol_cool_%s_end" % tag)),
               w, color=THR_COLORS[thr], edgecolor="k", lw=3, label="cooled %g K" % thr)
        ax.bar(x + (k - 1.5) * w, -np.nan_to_num(series(table, currents, "vol_heat_%s_end" % tag)),
               w, color=THR_COLORS[thr], edgecolor="k", lw=3, hatch="///", alpha=0.85)
    ax.axhline(0, color="0.4", lw=3.4)
    ax.set_xticks(x)
    ax.set_xticklabels(["%.1f" % c for c in currents], fontsize=32)
    ax.set_title("(b) Cooled (up) and heated (down) volume at drive end", fontsize=35)
    ax.set_xlabel(r"Drive current $I_0$ (A)")
    ax.set_ylabel("volume (mm$^3$)")
    handles = [Patch(facecolor=THR_COLORS[t], edgecolor="k", label="%g K" % t)
               for t in VOL_THRESHOLDS]
    handles += [Patch(facecolor="0.75", edgecolor="k", label="up = cooled"),
                Patch(facecolor="0.75", edgecolor="k", hatch="///", label="down = heated")]
    ax.legend(handles=handles, fontsize=26, ncol=3, loc="lower left")
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout()
    save_figure(fig, fname, index)

# =============================================================================
# 5. SPATIAL FIGURES - 2D INNER SURFACE (mm^2)
# =============================================================================
def area_series_for(area_map, i0):
    """The cached area series of one current (tolerant float-key lookup)."""
    if not area_map:
        return None
    for k, v in area_map.items():
        if np.isclose(k, i0):
            return v
    return None


def area_value(area_map, i0, thr, kind, when):
    """Cooled/heated area (mm^2) of one current at a reference time or peak."""
    s = area_series_for(area_map, i0)
    if s is None:
        return np.nan
    if when == "end":
        vals = surface_areas_at(s, CFG["T_END_S"])
        return vals.get(thr, (np.nan, np.nan))[0 if kind == "cool" else 1]
    if when == "60":
        vals = surface_areas_at(s, CFG["T_FINAL_S"])
        return vals.get(thr, (np.nan, np.nan))[0 if kind == "cool" else 1]
    pk = surface_area_peaks(s).get(thr)
    if not pk:
        return np.nan
    return pk["cool_peak"] if kind == "cool" else pk["heat_peak"]


def fig_h10_surface_maps_grid(cache, currents, times, index,
                              fname="H10_surface_maps_grid.png"):
    """Inner-surface temperature maps: one row per current, plus an ROI zoom."""
    hw = CFG["ROI_HALF_WIDTH_MM"]
    n_t = len(times)
    fig, axes = plt.subplots(len(currents), n_t + 1,
                             figsize=(5.0 * (n_t + 1), 4.8 * len(currents)), squeeze=False)
    for r, i0 in enumerate(currents):
        df = load_surface_current(i0, cache)
        vmax = float(np.nanmax(np.abs(df["Delta_T"])))
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        last_fr, last_areas = None, None
        for c, t_tgt in enumerate(times):
            ax = axes[r][c]
            sub = time_slice(df, t_tgt)
            if sub is None:
                ax.grid(False)
                ax.axis("off")
                ax.set_title("$t$ = %.0f s (no data)" % t_tgt, fontsize=31)
                continue
            fr = grid_from_scatter(sub)
            areas = frame_areas(sub)
            last_fr, last_areas = fr, areas
            head = "$I_0$ = %.1f A,  " % i0
            draw_map(ax, fr, norm,
                     title="%s$t$ = %.0f s\ncool %.1f | heat %.1f mm$^2$"
                           % (head, t_tgt, areas[0.5][0], areas[0.5][1]))
            check_no_overshoot(fr, sub, tag="I0=%.1f t=%.0f" % (i0, t_tgt))

        ax = axes[r][n_t]
        if last_fr is not None:
            st = spot_stats(last_fr["val"], last_fr["g1"], last_fr["g2"])
            c1, c2 = ((0.5 * (st["cold"][0] + st["hot"][0]),
                       0.5 * (st["cold"][1] + st["hot"][1])) if st else (0.0, 0.0))
            bounds = ((max(c1 - hw, last_fr["g1"][0]), min(c1 + hw, last_fr["g1"][-1])),
                      (max(c2 - hw, last_fr["g2"][0]), min(c2 + hw, last_fr["g2"][-1])))
            fr_roi = grid_from_scatter(time_slice(df, times[-1]), bounds=bounds)
            im = draw_map(ax, fr_roi, norm,
                          title="ROI zoom (cold & hot spot)\n$t$ = %.0f s" % times[-1])
            cax = make_axes_locatable(ax).append_axes("right", size="6%", pad=0.06)
            cb = fig.colorbar(im, cax=cax)
            cb.set_label(r"Surface $\Delta T$ (Â°C)", fontsize=32)
            cb.ax.tick_params(labelsize=32)
            if last_areas:
                ax.annotate("cooled %.1f mm$^2$\nheated %.1f mm$^2$"
                            % (last_areas[0.5][0], last_areas[0.5][1]),
                            xy=(0.02, 0.02), xycoords="axes fraction", fontsize=26,
                            color="0.2")
        else:
            ax.grid(False)
            ax.axis("off")
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout()
    save_figure(fig, fname, index)

def fig_h11_area_metrics(area_map, currents, table, index,
                         fname="H11_area_metrics.png"):
    """2D surface areas: versus current, the balance, and versus time."""
    if not area_map:
        print("      [skip] no area series available")
        return
    have = [i0 for i0 in currents if area_series_for(area_map, i0) is not None]
    fig, axes = plt.subplots(2, 2, figsize=(27.04, 16.64))

    specs = [("cool", "(a) Cooled surface area at drive end (%.0f s)" % CFG["T_END_S"]),
             ("heat", "(b) Heated surface area at drive end (%.0f s)" % CFG["T_END_S"])]
    for ax, (kind, ttl) in zip(axes[0], specs):
        for thr in VOL_THRESHOLDS:
            y = np.array([area_value(area_map, i0, thr, kind, "end") for i0 in have])
            ax.plot(have, y, "-o", color=THR_COLORS[thr], lw=5.3, ms=12.6,
                    label="%g K" % thr)
        ax.set_title(ttl, fontsize=35)
        ax.set_xlabel(r"Drive current $I_0$ (A)")
        ax.set_ylabel("area (mm$^2$)")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=32, ncol=2)

    ax = axes[1][0]
    for thr in VOL_THRESHOLDS:
        cc = np.array([area_value(area_map, i0, thr, "cool", "end") for i0 in have])
        hh = np.array([area_value(area_map, i0, thr, "heat", "end") for i0 in have])
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(hh > 0, cc / hh, np.nan)
        ax.plot(have, r, "-o", color=THR_COLORS[thr], lw=5.3, ms=12.6, label="%g K" % thr)
    ax.axhline(1.0, color="0.4", lw=3.4, ls=":")
    ax.set_yscale("log")
    ax.set_title("(c) Balance cooled / heated area at drive end", fontsize=35)
    ax.set_xlabel(r"Drive current $I_0$ (A)")
    ax.set_ylabel("cooled / heated  (-, log)")
    ax.legend(fontsize=32, ncol=2)

    ax = axes[1][1]
    for i0 in have:
        s = area_series_for(area_map, i0)
        if s is None:
            continue
        t = s["times"]
        area = s["tri_area"]
        cool = np.array([area[row < -0.5].sum() for row in s["tri_T"]])
        heat = np.array([area[row > 0.5].sum() for row in s["tri_T"]])
        line, = ax.plot(t, cool, lw=4.8, label=r"$I_0$ = %.1f A" % i0)
        ax.plot(t, heat, lw=4.2, ls="--", color=line.get_color())
    drive_band(ax)
    ax.axhline(0, color="0.4", lw=3.4)
    ax.set_title("(d) Area at 0.5 K vs time (solid = cooled, dashed = heated)",
                 fontsize=35)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("area (mm$^2$)")
    ax.set_xlim(0, CFG["T_FINAL_S"])
    ax.legend(fontsize=32, ncol=2)

    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout()
    save_figure(fig, fname, index)


def surface_stats_over_time(df, strides=1):
    """min/max/mean/std of the surface Delta-T at every loaded time step."""
    out = {"t": [], "min": [], "max": [], "mean": [], "std": []}
    for tt in np.unique(df["time_s"].to_numpy(float)):
        sub = df[df["time_s"] == tt]
        v = sub["Delta_T"].to_numpy(float)
        out["t"].append(tt)
        out["min"].append(np.nanmin(v))
        out["max"].append(np.nanmax(v))
        out["mean"].append(np.nanmean(v))
        out["std"].append(np.nanstd(v))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def fig_h12_surface_stats(cache, currents, index, fname="H12_surface_stats.png"):
    """Uniformity: surface min/max/mean and spread versus time for every current."""
    fig, axes = plt.subplots(2, 2, figsize=(27.04, 16.64))
    cmap = plt.get_cmap(CMAP_CURRENT)
    norm = plt.Normalize(min(currents), max(currents))
    for i0 in currents:
        df = load_surface_current(i0, cache)
        st = surface_stats_over_time(df)
        c = cmap(norm(i0))
        axes[0][0].plot(st["t"], st["min"], color=c, lw=4.8)
        axes[0][1].plot(st["t"], st["max"], color=c, lw=4.8)
        axes[1][0].plot(st["t"], st["mean"], color=c, lw=4.8)
        axes[1][1].plot(st["t"], st["std"], color=c, lw=4.8)
    titles = ["(a) Coldest point on the surface", "(b) Hottest point on the surface",
              "(c) Mean surface temperature", "(d) Spatial spread (std)"]
    ylabs = [r"min $\Delta T$ (Â°C)", r"max $\Delta T$ (Â°C)", r"mean $\Delta T$ (Â°C)",
             r"std $\Delta T$ (Â°C)"]
    for ax, ttl, ylab in zip(axes.ravel(), titles, ylabs):
        drive_band(ax)
        ax.axhline(0, color="0.4", lw=3.4, ls=":")
        ax.set_title(ttl, fontsize=35)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, CFG["T_FINAL_S"])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                          fraction=0.046, pad=0.02)
        cb.set_label(r"$I_0$ (A)", fontsize=32)
        cb.ax.tick_params(labelsize=32)
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout()
    save_figure(fig, fname, index)

def fig_h13_profiles(cache, i0, times, index, fname="H13_profiles.png"):
    """Cross-sections through the hot/cold spots at several times."""
    df = load_surface_current(i0, cache)
    fig, axes = plt.subplots(2, len(times), figsize=(5.5 * len(times), 10.6),
                             squeeze=False, sharey="row")
    for c, t_tgt in enumerate(times):
        sub = time_slice(df, t_tgt)
        if sub is None:
            continue
        fr = grid_from_scatter(sub)
        st = spot_stats(fr["val"], fr["g1"], fr["g2"])
        # val has shape (len(g2), len(g1)): col -> g1 (a1, x), row -> g2 (a2, y).
        row_hot = int(np.nanargmax(fr["val"])) // fr["val"].shape[1]
        col_hot = int(np.nanargmax(fr["val"])) % fr["val"].shape[1]
        for r, (label, prof, coord) in enumerate(
                (("along %s" % fr["a1"], fr["val"][row_hot, :], fr["g1"]),
                 ("along %s" % fr["a2"], fr["val"][:, col_hot], fr["g2"]))):
            ax = axes[r][c]
            ax.plot(coord, prof, color=C_MAIN, lw=5.3)
            ax.axhline(0, color="0.4", lw=3.4, ls=":")
            ax.set_title("$t$ = %.0f s  (%s through hottest)" % (t_tgt, label),
                         fontsize=31)
            ax.set_xlabel("%s (mm)" % (fr["a1"] if r == 0 else fr["a2"]), fontsize=30)
            if c == 0:
                ax.set_ylabel(r"$\Delta T$ (Â°C)", fontsize=31)
            if st:
                ax.annotate("cold %.1f | hot %.1f" % (st["cold"][2], st["hot"][2]),
                            xy=(0.03, 0.06), xycoords="axes fraction", fontsize=26,
                            color="0.25")
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout()
    save_figure(fig, fname, index)


def kymograph(df, axis, nbins=150):
    """Mean Delta-T along one in-plane axis versus time (space-time strip)."""
    coord = df[axis].to_numpy(float)
    times = np.unique(df["time_s"].to_numpy(float))
    t_all = df["time_s"].to_numpy(float)
    dT = df["Delta_T"].to_numpy(float)
    edges = np.linspace(coord.min(), coord.max(), nbins + 1)
    bins = np.clip(np.digitize(coord, edges) - 1, 0, nbins - 1)
    M = np.full((times.size, nbins), np.nan)
    for k, tt in enumerate(times):
        m = t_all == tt
        b, v = bins[m], dT[m]
        cnt = np.bincount(b, minlength=nbins).astype(float)
        tot = np.bincount(b, weights=v, minlength=nbins)
        nz = cnt > 0
        M[k, nz] = tot[nz] / cnt[nz]
    centres = 0.5 * (edges[:-1] + edges[1:])
    return times, centres, M


def fig_h14_kymograph(cache, i0, index, fname="H14_kymograph.png"):
    """Space-time kymographs of the inner surface at one drive current."""
    df = load_surface_current(i0, cache)
    a1, a2 = plane_axes(df)
    fig, axes = plt.subplots(1, 2, figsize=(27.04, 10.01))
    for ax, axis in zip(axes, (a1, a2)):
        times, centres, M = kymograph(df, axis)
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=CMAP_T,
                       vmin=-np.nanmax(np.abs(M)), vmax=np.nanmax(np.abs(M)),
                       extent=[times[0], times[-1], centres[0], centres[-1]])
        ax.axvline(CFG["T_ON_S"], color="w", lw=3.4, ls="--")
        ax.axvline(CFG["T_END_S"], color="w", lw=3.4, ls="--")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(r"mean $\Delta T$ (Â°C)", fontsize=32)
        cb.ax.tick_params(labelsize=32)
        ax.set_title("(a) along %s" % axis if axis == a1 else "(b) along %s" % axis,
                     fontsize=35)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("%s (mm)" % axis)
        ax.grid(False)
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout()
    save_figure(fig, fname, index)


def fig_h15_current_difference(cache, i0_low, i0_high, t_target, index,
                               fname="H15_current_difference.png"):
    """Maps of two drive currents and their difference at one time."""
    d_low = time_slice(load_surface_current(i0_low, cache), t_target)
    d_high = time_slice(load_surface_current(i0_high, cache), t_target)
    if d_low is None or d_high is None:
        print("      [skip] current-difference map: missing slice")
        return
    cm = common_map(d_low, d_high)
    va, vb = cm["va"], cm["vb"]
    diff = vb - va
    lim = float(np.nanmax(np.abs(np.concatenate([va.ravel(), vb.ravel()]))))
    dlim = float(np.nanmax(np.abs(diff)))
    lim = lim if np.isfinite(lim) and lim > 0 else 1.0
    dlim = dlim if np.isfinite(dlim) and dlim > 0 else 1.0

    ar_low, ar_high = frame_areas(d_low), frame_areas(d_high)
    frames = [dict(cm, val=va), dict(cm, val=vb), dict(cm, val=diff)]
    norms = [TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
             TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
             TwoSlopeNorm(vmin=-dlim, vcenter=0.0, vmax=dlim)]
    titles = ["$I_0$ = %.1f A\ncool %.1f | heat %.1f mm$^2$"
              % (i0_low, ar_low[0.5][0], ar_low[0.5][1]),
              "$I_0$ = %.1f A\ncool %.1f | heat %.1f mm$^2$"
              % (i0_high, ar_high[0.5][0], ar_high[0.5][1]),
              "difference (%.1f - %.1f A)\nmax %.2f K | RMS %.2f K"
              % (i0_high, i0_low, float(np.nanmax(diff)),
                 float(np.sqrt(np.nanmean(diff ** 2))))]
    fig, axes = plt.subplots(1, 3, figsize=(32.24, 9.75))
    for ax, fr, norm, ttl, cmap in zip(axes, frames, norms, titles,
                                       (CMAP_T, CMAP_T, CMAP_D)):
        im = draw_map(ax, fr, norm, title=ttl, cmap=cmap, annotate=(cmap != CMAP_D))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(r"$\Delta T$ (Â°C)" if cmap != CMAP_D else r"$\Delta$ (Â°C)", fontsize=32)
        cb.ax.tick_params(labelsize=32)
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout()
    save_figure(fig, fname, index)

# =============================================================================
# 6. ENERGY FIGURES
# =============================================================================
def cumulative_integral(y, x):
    """Cumulative trapezoidal integral starting at zero."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    out = np.zeros(y.shape, dtype=float)
    if y.size > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def fig_h16_flux_cumulative(df, currents, index, fname="H16_flux_cumulative.png"):
    """Heat flux transient and the cumulative areal heat energy."""
    cmap = plt.get_cmap(CMAP_CURRENT)
    norm = plt.Normalize(min(currents), max(currents))
    fig, axes = plt.subplots(1, 2, figsize=(27.04, 10.14))
    for i0 in currents:
        sub = sub_current(df, i0)
        if sub is None:
            continue
        t = (sub["time_s"]-CFG["T_ON_S"]).to_numpy(float)
        c = cmap(norm(i0))
        axes[0].plot(t, np.abs(sub["flux_Wm2"].to_numpy(float)), color=c, lw=4.8)
        cum = cumulative_integral(np.abs(sub["flux_Wm2"].to_numpy(float)), t)
        axes[1].plot(t, cum / 1e3, color=c, lw=4.8)
        axes[1].plot([t[-1]], [cum[-1] / 1e3], "o", ms=9.8, color=c)
    drive_band(axes[0])
    drive_band(axes[1])
    axes[0].set_title(r"(a) Boundary heat-flux magnitude $|q|(t)$", fontsize=35)
    axes[1].set_title(r"(b) Cumulative areal heat energy $\int|q|\,\mathrm{d}t$",
                      fontsize=35)
    for ax, ylab in zip(axes, (r"$|q|$ (W/m$^2$)", r"cumulative energy (kJ/m$^2$)")):
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, CFG["T_FINAL_S"])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                          fraction=0.046, pad=0.02)
        cb.set_label(r"$I_0$ (A)", fontsize=32)
        cb.ax.tick_params(labelsize=32)
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout()
    save_figure(fig, fname, index)

def fig_h17_energy_vs_current(df, table, currents, index,
                              fname="H17_energy_vs_current.png"):
    """Consumption and energetic metrics versus drive current."""
    cmap = plt.get_cmap(CMAP_CURRENT)
    norm = plt.Normalize(min(currents), max(currents))
    fig, axes = plt.subplots(2, 2, figsize=(27.04, 16.64))

    ax = axes[0][0]
    for i0 in currents:
        sub = sub_current(df, i0)
        if sub is None:
            continue
        t = (sub["time_s"]-CFG["T_ON_S"]).to_numpy(float)
        ax.plot(t, np.abs(sub["voltage_tec1_V"]) * i0, color=cmap(norm(i0)), lw=4.8)
    drive_band(ax)
    ax.set_title(r"(a) Electrical power $|V_1|\cdot I_0$", fontsize=35)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("power (W)")
    ax.set_xlim(0, CFG["T_FINAL_S"])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      fraction=0.046, pad=0.02)
    cb.set_label(r"$I_0$ (A)", fontsize=32)
    cb.ax.tick_params(labelsize=32)

    panels = [("W_full_J", r"(b) Electrical energy $W = I_0\!\int_0^{60}|V_1|\,\mathrm{d}t$  (J)",
               "energy (J)"),
              ("E_areal_J_m2", r"(c) Areal heat energy $\int|q|\,\mathrm{d}t$  (J/m$^2$)",
               r"energy (J/m$^2$)"),
              ("dose_Ks", r"(d) Thermal dose $\int|\Delta T_c|\,\mathrm{d}t$  (K$\cdot$s)",
               r"dose (K$\cdot$s)")]
    for ax, (col, ttl, ylab) in zip(axes.ravel()[1:], panels):
        ax.plot(currents, series(table, currents, col), "-o", color=C_MAIN, lw=5.6, ms=14)
        ax.set_title(ttl, fontsize=35)
        ax.set_xlabel(r"Drive current $I_0$ (A)")
        ax.set_ylabel(ylab)
        ax.set_ylim(bottom=0)
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout()
    save_figure(fig, fname, index)

def fig_h18_efficiency(table, currents, index, fname="H18_efficiency.png"):
    """Scale-free efficiencies and the cooling-versus-overheating trade-off."""
    tag_low = ("%g" % VOL_THRESHOLDS[0]).replace(".", "")
    fig, axes = plt.subplots(1, 3, figsize=(31.2, 10.01))

    ax = axes[0]
    ax.plot(currents, series(table, currents, "K_per_J"), "-o", color="#2166ac",
            lw=5.6, ms=14, label="cooling per joule (K/J)")
    ax.set_xlabel(r"Drive current $I_0$ (A)")
    ax.set_ylabel("K / J", color="#2166ac")
    ax.tick_params(axis="y", colors="#2166ac")
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.plot(currents, series(table, currents, "J_per_K"), "--s", color="#b2182b",
             lw=5, ms=14, mfc="white", label="energy per K (J/K)")
    ax2.set_ylabel("J / K", color="#b2182b")
    ax2.tick_params(axis="y", colors="#b2182b")
    ax.set_title("(a) Energy cost of cooling", fontsize=35)
    ax.legend(fontsize=32, loc="upper right")
    ax2.legend(fontsize=32, loc="lower left")

    ax = axes[1]
    ax.plot(currents, series(table, currents, "vol_cool_%s_per_J" % tag_low), "-o",
            color="#2ca02c", lw=5.6, ms=14, label=r"cooled volume per joule (mm$^3$/J)")
    ax.plot(currents, series(table, currents, "K_per_J_hot"), "--s", color="#9467bd",
            lw=5, ms=14, mfc="white", label="overheating per joule (K/J)")
    ax.set_title("(b) Useful cooling vs unwanted heating per joule", fontsize=35)
    ax.set_xlabel(r"Drive current $I_0$ (A)")
    ax.set_ylabel("efficiency")
    ax.legend(fontsize=32)

    ax = axes[2]
    cx = np.abs(series(table, currents, "dT_cold_peak_K"))
    hy = series(table, currents, "dT_hot_peak_K")
    ax.plot(cx, hy, "-", color=C_MAIN, lw=3.4, alpha=0.6)
    ax.scatter(cx, hy, s=216, color=C_MAIN, edgecolor="k", lw=3, zorder=4)
    for xx, yy, i0 in zip(cx, hy, currents):
        if np.isfinite(xx) and np.isfinite(yy):
            ax.annotate("%.1f" % i0, (xx, yy), xytext=(4, 4),
                        textcoords="offset points", fontsize=30, color=C_MAIN)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(hy > 0, cx / hy, np.nan)
    if np.isfinite(ratio).any():
        k = int(np.nanargmax(ratio))
        ax.scatter([cx[k]], [hy[k]], s=760, marker="*", color="#ffbf00", edgecolor="k",
                   lw=3.4, zorder=5)
        ax.annotate("best balance\n($I_0$ = %.1f A)" % currents[k], (cx[k], hy[k]),
                    xytext=(6, -14), textcoords="offset points", fontsize=26, color="0.25")
    ax.set_title("(c) Cooling vs overheating (star = best balance)", fontsize=35)
    ax.set_xlabel(r"cooling $|\Delta T_c|$ peak (°C)")
    ax.set_ylabel(r"overheating $\Delta T_h$ peak (°C)")
    ax.annotate("better\n(more cooling,\nless overheating)", xy=(0.75, 0.1),
                xycoords="axes fraction", ha="center", fontsize=32, color="0.3")
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout()
    save_figure(fig, fname, index)

# =============================================================================
# 7. SCORECARD
# =============================================================================
def fig_h19_scorecard(table, currents, index, fname="H19_scorecard.png"):
    """Numeric scorecard: every headline metric against every drive current."""
    tag_low = ("%g" % VOL_THRESHOLDS[0]).replace(".", "")
    rows = [("Coldest probe (Â°C)", "dT_cold_peak_K", "%.2f", False),
            ("Hottest probe (Â°C)", "dT_hot_peak_K", "%.2f", False),
            ("Cool/heat ratio (-)", "ratio_cool_heat", "%.2f", False),
            ("Cooled vol @%gÂ°C (mm3)" % VOL_THRESHOLDS[0], "vol_cool_%s_end" % tag_low,
             "%.1f", False),
            ("Heated vol @%gÂ°C (mm3)" % VOL_THRESHOLDS[0], "vol_heat_%s_end" % tag_low,
             "%.1f", False),
            ("tau63 cooling (s)", "t_cool63_s", "%.2f", False),
            ("Back to normal (s)", "t_back_normal_s", "%.2f", False),
            ("Switch-off overshoot (Â°C)", "overshoot_cold_K", "%.2f", False),
            ("Electrical energy (J)", "W_full_J", "%.2f", False),
            ("Areal heat energy (J/m2)", "E_areal_J_m2", "%.0f", False),
            ("Cooling per joule (Â°C/J)", "K_per_J", "%.3f", False),
            ("Cooled mm3 per joule", "vol_cool_%s_per_J" % tag_low, "%.2f", False)]
    if "area_cool_%s_end" % tag_low in table.columns:
        rows.insert(5, ("Cooled area @%gÂ°C (mm2)" % VOL_THRESHOLDS[0],
                        "area_cool_%s_end" % tag_low, "%.1f", False))
        rows.insert(6, ("Heated area @%gÂ°C (mm2)" % VOL_THRESHOLDS[0],
                        "area_heat_%s_end" % tag_low, "%.1f", False))

    values = np.full((len(rows), len(currents)), np.nan)
    text = [["-" for _ in currents] for _ in rows]
    for r, (_, col, fmt, _) in enumerate(rows):
        for c, i0 in enumerate(currents):
            v = val(table, i0, col)
            values[r, c] = v
            text[r][c] = fmt % v if np.isfinite(v) else "-"

    fig, ax = plt.subplots(figsize=(1.7 * len(currents) + 8.8, 0.7 * len(rows) + 3.5))
    ax.axis("off")
    tb = ax.table(cellText=text, rowLabels=[r[0] for r in rows],
                  colLabels=["%.1f A" % c for c in currents], loc="center",
                  cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(22)
    tb.scale(1.0, 1.25)

    # shade every row by its own normalised magnitude so the best cell stands out
    cmap = plt.get_cmap("YlGnBu")
    for r in range(len(rows)):
        row = values[r]
        finite = np.isfinite(row)
        if not finite.any():
            continue
        lo, hi = np.nanmin(row), np.nanmax(row)
        span = hi - lo if hi > lo else 1.0
        for c in range(len(currents)):
            if not finite[c]:
                continue
            ax.table  # noqa - keep the reference explicit
            cell = tb[(r + 1, c)]
            cell.set_facecolor(cmap(0.15 + 0.7 * (row[c] - lo) / span))
    for (r, c), cell in tb.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_text_props(weight="bold")
        if c == -1:
            cell.set_text_props(ha="left")
    ax.set_title("Heatsink (Concise) - scorecard per drive current\n"
                 "each row is shaded from its own minimum (light) to maximum (dark)",
                 fontsize=35, weight="bold", pad=16)
    save_figure(fig, fname, index)


def write_index(index, extra, fname="figure_index.md"):
    """Write the figure index / provenance file."""
    path = os.path.join(CFG["OUTPUT_DIR"], fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Heatsink (Concise) - figure index\n\n")
        f.write("Inputs:\n\n")
        f.write("- probes: `%s`\n" % CFG["PROBES_CSV"])
        f.write("- surface: `%s`\n\n" % CFG["SURF_CSV"])
        for line in extra:
            f.write("%s\n" % line)
        f.write("\n## Figures\n\n")
        for nm in index:
            f.write("- `%s`\n" % nm)
    return path

def add_area_to_table(table, area_map, thresholds=VOL_THRESHOLDS):
    """Merge the 2D inner-surface area columns into the metrics table."""
    if not area_map:
        return table
    out = table.copy()
    for thr in thresholds:
        tag = ("%g" % thr).replace(".", "")
        for when in ("end", "60", "peak"):
            out["area_cool_%s_%s" % (tag, when)] = [
                area_value(area_map, float(r["I0_A"]), thr, "cool", when)
                for _, r in out.iterrows()]
            out["area_heat_%s_%s" % (tag, when)] = [
                area_value(area_map, float(r["I0_A"]), thr, "heat", when)
                for _, r in out.iterrows()]
        out["area_ratio_%s_end" % tag] = [
            safe_ratio(a, b) for a, b in
            zip(out["area_cool_%s_end" % tag], out["area_heat_%s_end" % tag])]
        out["area_cool_%s_per_J" % tag] = [
            safe_ratio(a, w) for a, w in
            zip(out["area_cool_%s_end" % tag], out["W_full_J"])]
    return out


# =============================================================================
# H20. MIN/MAX RELATIVE-T vs ENERGY (drive-window peaks + end values)
# =============================================================================
def fig_h20_minmax_vs_energy(table, index, fname="H20_minmax_vs_energy.png"):
    """Peak & end-of-pulse surface Delta-T extrema vs drive-window energy.

    x = W_drive_J (I0*int|V1| ON->END); y = dT_min/dT_max (peak in drive
    window on the left, value at t_end on the right).  One point per I0.
    """
    df = table.dropna(subset=["W_drive_J"]).sort_values("I0_A")
    fig, axes = plt.subplots(1, 2, figsize=(27.04, 10.01))
    panels = [(("dT_min_peak_K", "dT_max_peak_K"),
               "(a) Peak in drive window vs drive energy"),
              (("dT_min_end_K", "dT_max_end_K"),
               "(b) End-of-pulse (t = %g s) vs drive energy" % CFG["T_END_S"])]
    for ax, ((cmin, cmax), title) in zip(axes, panels):
        ax.scatter(df["W_drive_J"], df[cmin], c=df["I0_A"], cmap=CMAP_CURRENT,
                   s=336, marker="v", edgecolors="k", linewidths=3,
                   label="min $\Delta T$ (coldest)", zorder=3)
        ax.scatter(df["W_drive_J"], df[cmax], c=df["I0_A"], cmap=CMAP_CURRENT,
                   s=336, marker="^", edgecolors="k", linewidths=3,
                   label="max $\Delta T$ (warmest)", zorder=3)
        for _, r in df.iterrows():
            ax.annotate("%.1f A" % r["I0_A"], (r["W_drive_J"], r[cmin]),
                        fontsize=30, ha="right", va="top")
        ax.axhline(0.0, color="k", lw=3.4)
        ax.set_title(title, fontsize=35)
        ax.set_xlabel("Drive energy $W_{drive}$ = $I_0\\int|V_1|dt$ (J)")
        ax.set_ylabel("$\Delta T$ (Â°C)")
        ax.legend(fontsize=32, loc="best")
        ax.grid(alpha=0.3)
    pass  # suptitle removed (unified style: no long titles)
    fig.tight_layout()
    save_figure(fig, fname, index)


# =============================================================================
# 8. PIPELINE
# =============================================================================
def main():
    """Read the heatsink export and produce every supported figure + metrics."""
    apply_style()
    t0 = time.time()
    index, cache = [], {}
    os.makedirs(CFG["OUTPUT_DIR"], exist_ok=True)
    print("=" * 100)
    print("HEATSINK (Concise) - standalone report: temporal, spatial and energy plots")
    print("=" * 100)

    print("\n[1/6] Loading probe table ...")
    df = load_accumulated_probes(CFG["PROBES_CSV"], CFG["T_AMB_C"])
    currents = sorted(set(np.round(df["I0"].unique(), 4)))
    focus = CFG["FOCUS_CURRENT"] if any(np.isclose(currents, CFG["FOCUS_CURRENT"])) \
        else currents[len(currents) // 2]
    print("      %d rows | %d currents %s | dt = %.2f s | volume probes: %s"
          % (len(df), len(currents), currents,
             float(np.min(np.diff(np.sort(df["time_s"].unique())))),
             has_volume_probes(df)))
    if "T_Max" in df.columns:
        spread_max = float(df["T_Max"].max() - df["T_Max"].min())
        spread_min = float(df["T_Min"].max() - df["T_Min"].min())
        spread_hot = float(df["T_Hot"].max() - df["T_Hot"].min())
        spread_cold = float(df["T_Cold"].max() - df["T_Cold"].min())
        if max(spread_max, spread_min) < 0.05 and max(spread_hot, spread_cold) > 1.0:
            print("      WARNING: the surface-extrema probes are essentially dead in this "
                  "export (spread T_Max = %.4f K, T_Min = %.4f K, while the hot/cold probes "
                  "move %.1f/%.1f K) - see H6"
                  % (spread_max, spread_min, spread_hot, spread_cold))

    # One streaming pass over the surface export, reused by the temporal figures
    # and by the metrics table (never re-read).
    surf_stats = {}
    if CFG["RUN_SURF_METRICS"]:
        print("\n[1b/6] Inner-surface field (Config_heatsink_surf) ...")
        wanted_s = CFG["SURF_METRICS_CURRENTS"]
        if isinstance(wanted_s, str) and wanted_s.lower() == "all":
            wanted_s = currents
        wanted_s = [c for c in wanted_s if any(np.isclose(currents, c))] or currents
        surf_stats = collect_surface_stats(CFG["SURF_CSV"], wanted_s, CFG["T_AMB_C"])
        if not surf_stats:
            print("      [warn] no inner-surface data - the temporal figures fall back "
                  "to the probes")

    print("\n[2/6] Metrics table ...")
    table = metrics_table(df, currents)
    # Merge the inner-surface columns NOW: the temporal figures and the 2D-area
    # step both read `table`, so the surf_* values must be there from the start.
    if surf_stats:
        table = add_surface_to_table(table, surf_stats)
        write_surface_traces(surf_stats, CFG["SURF_TRACES_CSV"])
    table.to_csv(CFG["METRICS_CSV"], index=False)
    print("      saved %s (%d rows x %d columns, %d surf_* inner-surface columns)"
          % (CFG["METRICS_CSV"], table.shape[0], table.shape[1],
             len([c for c in table.columns if c.startswith("surf_")])))

    if CFG["RUN_PROBES"]:
        print("\n[3/6] Probe / temporal figures ...")
        fig_h1_probes_grid(df, CFG["GRID_CURRENTS"], index, surf_stats)
        fig_h2_focus(df, focus, index, surf_stats)
        fig_h3_overlay(df, CFG["OVERLAY_CURRENTS"], index, surf_stats)
        fig_h4_normalized_response(df, table, currents, index, surf_stats)
        fig_h5_switch_off(df, table, currents, index, surf_stats)
        fig_h6_extrema_probe_check(df, currents, focus, index)

    if CFG["RUN_SPATIAL_VOLUME"]:
        print("\n[4/6] Volume (spatial probe) figures ...")
        fig_h7_volume_transients(df, CFG["GRID_CURRENTS"], index)
        fig_h8_volume_vs_current(table, currents, index)
        fig_h9_volume_balance(table, currents, index)

    area_map = {}
    if CFG["RUN_SPATIAL_SURFACE"]:
        print("\n[5/6] 2D inner-surface figures (streaming the surface export) ...")
        if not os.path.exists(CFG["SURF_CSV"]):
            print("      [skip] surface export not found: %s" % CFG["SURF_CSV"])
        else:
            wanted = CFG["AREA_CURRENTS"]
            if isinstance(wanted, str) and wanted.lower() == "all":
                wanted = currents
            wanted = [c for c in wanted if any(np.isclose(currents, c))] or [focus]
            area_map = collect_area_series(CFG["SURF_CSV"], wanted, CFG["T_AMB_C"])
            if area_map:
                hull = list(area_map.values())[0]["hull_mm2"]
                print("      area series for currents %s (hull %.1f mm^2)"
                      % (sorted(area_map), hull))
                table = add_area_to_table(table, area_map)
                table.to_csv(CFG["METRICS_CSV"], index=False)
                print("      metrics CSV refreshed with the 2D area columns")

            fig_h10_surface_maps_grid(cache, CFG["SPATIAL_CURRENTS"],
                                      CFG["MAP_TIMES_S"], index)
            fig_h11_area_metrics(area_map, currents, table, index)
            fig_h12_surface_stats(cache, CFG["SPATIAL_CURRENTS"], index)
            fig_h13_profiles(cache, focus, CFG["PROFILE_TIMES_S"], index)
            fig_h14_kymograph(cache, focus, index)
            if len(CFG["OVERLAY_CURRENTS"]) >= 2:
                fig_h15_current_difference(cache, CFG["OVERLAY_CURRENTS"][0],
                                           CFG["OVERLAY_CURRENTS"][-1],
                                           CFG["MAP_TIMES_S"][-2], index)

    if CFG["RUN_ENERGY"]:
        print("\n[6/6] Energy figures ...")
        fig_h16_flux_cumulative(df, CFG["GRID_CURRENTS"], index)
        fig_h17_energy_vs_current(df, table, currents, index)
        fig_h18_efficiency(table, currents, index)
        fig_h20_minmax_vs_energy(table, index)

    if CFG["RUN_SCORECARD"]:
        fig_h19_scorecard(table, currents, index)
        print_verdict(table, currents)
        print_best_points(table, currents)

    path = write_index(index, [
        "Generated by `reading_comsol_heatsink.py` in %.1f s." % (time.time() - t0),
        "Protocol: drive %.0f-%.0f s | run end %.0f s | T_amb = %.1f Â°C"
        % (CFG["T_ON_S"], CFG["T_END_S"], CFG["T_FINAL_S"], CFG["T_AMB_C"]),
        "2D areas come from a Delaunay triangulation of the exported surface mesh "
        "(thresholds 0.5/1/2/5 Â°C).",
        "Note: T_Max/T_Min probes are constant in the current export (see H6).",
        "The `surf_*` columns come from the inner-surface field "
        "(Config_heatsink_surf.csv) and are the only simulated values that may be "
        "compared with experimental data: point probes are never compared with "
        "experiments.",
        "The temporal figures (H1-H5) plot the surface min/max; the hot/cold point "
        "probes survive only as thin dotted reference lines (H6 keeps the probe "
        "diagnostic).",
        "Surface traces: OUTPUT/heatsink_surface_traces.csv.",
    ])
    print("\n      %d figures written to %s" % (len(index), CFG["OUTPUT_DIR"]))
    print("      index: %s" % path)
    print("\nDone in %.1f s." % (time.time() - t0))


if __name__ == "__main__":
    main()
