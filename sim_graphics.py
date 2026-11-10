# =============================================================================
# reading_comsol_dual_tec_graphics.py
# -----------------------------------------------------------------------------
# Improved graphics suite comparing the thermally-parallel Dual-TEC model
# (Config_Castell) against the Heatsink reference model (Config_heatsink).
#
# Three metric families are visualised side by side:
#   * TEMPORAL  -> probe transients, normalised step response, per-current trends
#   * ENERGETIC -> per-area heat energy, thermal dose, drive work and scale-free
#                  efficiency surrogates.  NO load-resistance assumption is made.
#   * SPATIAL   -> inner-surface maps, maps of the difference, space-time
#                  kymographs, spatial statistics and cross-section profiles
# plus a dedicated analysis of the OPEN-CIRCUIT recovered voltage of the
# thermally parallel TEC (Castell Global Variable Probe 3), which the Heatsink
# model does not expose.
#
# The parsing helpers are reused from reading_comsol_dual_tec.py so that both
# scripts share a single loader implementation:
#
#     python reading_comsol_dual_tec_graphics.py
# =============================================================================
import os
import time

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import LinearNDInterpolator

from sim_loaders import (
    CONFIG as BASE_CONFIG,
    load_accumulated_probes,
    load_surface_inner,
)

# =============================================================================
# 0. CONFIGURATION, PALETTE & FIGURE STYLE
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "OUTPUT", "dual_tec_graphics")

SAVE_FIGURES = True
SHOW_FIGURES = False
FIGURE_DPI = 200

CFG = {
    # ---- inputs (shared with reading_comsol_dual_tec.py) ----
    "DUAL_PROBES_CSV": BASE_CONFIG["DUAL_TEC_PROBES_CSV"],
    "SINK_PROBES_CSV": BASE_CONFIG["HEATSINK_PROBES_CSV"],
    "DUAL_SURF_CSV": BASE_CONFIG["DUAL_TEC_INNER_SURF_CSV"],
    "SINK_SURF_CSV": BASE_CONFIG["HEATSINK_INNER_SURF_CSV"],
    "T_AMB_C": BASE_CONFIG["T_AMB_DEFAULT_C"],
    "SURFACE_CHUNK": BASE_CONFIG["SURFACE_CHUNK"],

    # ---- metric definitions ----
    # One fixed drive window (s) is used for EVERY integrated metric so that the
    # two models are integrated over exactly the same interval.  The window that
    # a 2%-of-peak threshold would produce is reported next to it for
    # transparency (it drifts by +-10 s between currents, which would make the
    # integrated energies incomparable).
    "DRIVE_WINDOW_S": (4.0, 30.0),
    "WINDOW_THRESHOLD_FRAC": 0.02,

    # ---- figure selections ----
    "GRID_CURRENTS": [0.1, 0.2, 0.4, 0.5, 0.7, 0.9, 1.0],
    "FOCUS_CURRENT": 0.5,
    "SPATIAL_CURRENTS": [0.1, 0.5, 1.0],
    "SPATIAL_TIMES_S": [10.0, 20.0, 27.0],
    "MAP_RESOLUTION": 160,
    "ROI_HALF_WIDTH_MM": 6.0,

    # ---- run switches ----
    "RUN_TEMPORAL": True,
    "RUN_ENERGETIC": True,
    "RUN_RECOVERED_VOLTAGE": True,
    "RUN_SPATIAL": True,
    "RUN_SCORECARD": True,
}

C_DUAL = "#1f77b4"
C_SINK = "#d62728"
LBL_DUAL = "Dual-TEC (Castell)"
LBL_SINK = "Heatsink"

CMAP_T = "coolwarm"        # signed temperature fields
CMAP_D = "RdBu_r"          # signed differences
CMAP_CURRENT = "viridis"   # families of curves parameterised by drive current


def apply_style():
    """Consistent, publication-friendly matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": FIGURE_DPI,
        "savefig.bbox": "tight",
        "font.size": 26,
        "axes.titlesize": 30,
        "axes.titleweight": "normal",
        "axes.labelsize": 26,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.55,
        "legend.fontsize": 28,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save_fig(fig, name, index):
    """Save one figure into OUTPUT_DIR (optionally also show it), then close."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_FIGURES:
        fig.savefig(os.path.join(OUTPUT_DIR, name))
        print(f"      saved {name}")
    index.append(name)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

# =============================================================================
# 1. METRIC LAYER - temporal, energetic and open-circuit recovered voltage
# =============================================================================
def trapz(y, x):
    """Trapezoidal integral with a fallback for older numpy releases."""
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def detect_drive_window(t_s, v1, frac=0.02):
    """Active drive interval inferred from |V1|: (t_on, t_off, t_peak, |V1|peak)."""
    t_s = np.asarray(t_s, dtype=float)
    absv = np.abs(np.asarray(v1, dtype=float))
    if absv.size == 0 or not np.isfinite(absv).any():
        return float(t_s[0]), float(t_s[-1]), float(t_s[0]), 0.0
    peak = float(np.nanmax(absv))
    active = absv > frac * peak if peak > 0 else np.zeros(absv.shape, dtype=bool)
    if not active.any():
        return float(t_s[0]), float(t_s[-1]), float(t_s[0]), peak
    return (float(t_s[active].min()), float(t_s[active].max()),
            float(t_s[int(np.nanargmax(absv))]), peak)


def crossing_time(t, y, t0, target):
    """Delay from t0 until |y| first reaches 'target' (NaN when never reached)."""
    idx = np.where((t >= t0) & (np.abs(y) >= target))[0]
    return float(t[idx[0]] - t0) if idx.size else np.nan


def window_mask(t, window):
    """Boolean mask of the samples inside a (t_on, t_off) window."""
    return (t >= window[0]) & (t <= window[1])


def slice_current(df, i0):
    """All time samples of one drive current, sorted by time (None if absent)."""
    if df is None:
        return None
    sub = df[np.isclose(df["I0"], i0)].sort_values("time_s")
    return sub if not sub.empty else None


def has_v2_column(df):
    """True when the loaded table carries a usable parallel-TEC voltage."""
    return df is not None and "voltage_tec2_V" in df.columns and df["voltage_tec2_V"].notna().any()


def run_metrics(df, i0, label, t_amb_c=20.0, window=None, frac=0.02):
    """Temporal + energetic metrics of one model at one drive current.

    No load resistance is assumed anywhere: the recovered voltage of the
    parallel TEC is treated strictly as an open-circuit quantity (peak value,
    voltage-time integral, ratio to V1 and its slope versus the cold-side
    temperature difference).
    """
    sub = slice_current(df, i0)
    if sub is None:
        return None

    t = sub["time_s"].to_numpy(float)
    v1 = sub["voltage_tec1_V"].to_numpy(float)
    v2 = sub["voltage_tec2_V"].to_numpy(float) if has_v2_column(df) else np.full(t.shape, np.nan)
    d_hot = sub["Delta_Hot"].to_numpy(float)
    d_cold = sub["Delta_Cold"].to_numpy(float)
    q = sub["flux_Wm2"].to_numpy(float)

    auto = detect_drive_window(t, v1, frac)
    t_on, t_off = window if window is not None else auto[:2]
    m = window_mask(t, (t_on, t_off))
    if not m.any():
        return None

    d_cold_pk = float(np.nanmin(d_cold[m]))            # most negative = coldest
    d_hot_pk = float(np.nanmax(d_hot[m]))
    q_pk = float(np.nanmax(np.abs(q[m])))
    v1_pk = float(np.nanmax(np.abs(v1[m])))
    x_cold = np.abs(d_cold[m])
    v2_ok = has_v2_column(df) and np.isfinite(v2[m]).any()
    v2_pk = float(np.nanmax(np.abs(v2[m]))) if v2_ok else np.nan

    slope_mV_K = np.nan
    if v2_ok:
        denom = float(np.sum(x_cold ** 2))
        if denom > 0:
            slope_mV_K = float(np.sum(np.abs(v2[m]) * x_cold) / denom * 1e3)

    return {
        "model": label,
        "I0_A": float(i0),
        "t_on_s": float(t_on),
        "t_off_s": float(t_off),
        "t_on_auto_s": auto[0],
        "t_off_auto_s": auto[1],
        "dT_cold_peak_K": d_cold_pk,
        "dT_hot_peak_K": d_hot_pk,
        "dT_ratio": abs(d_cold_pk) / d_hot_pk if d_hot_pk > 0 else np.nan,
        "dT_cold_per_A_KpA": abs(d_cold_pk) / float(i0),
        "t63_cold_s": crossing_time(t, d_cold, t_on, 0.63 * abs(d_cold_pk)),
        "t90_cold_s": crossing_time(t, d_cold, t_on, 0.90 * abs(d_cold_pk)),
        "t63_hot_s": crossing_time(t, d_hot, t_on, 0.63 * max(d_hot_pk, 0.0)),
        "q_peak_Wm2": q_pk,
        "q_mean_active_Wm2": float(np.nanmean(np.abs(q[m]))),
        "E_areal_J_m2": trapz(np.abs(q[m]), t[m]),
        "dose_Ks": trapz(x_cold, t[m]),
        "W_drive_J": float(i0) * trapz(np.abs(v1[m]), t[m]),
        "V1_peak_V": v1_pk,
        "V1_mean_active_V": float(np.nanmean(np.abs(v1[m]))),
        "V2_peak_V": v2_pk,
        "V2_ratio_peak": v2_pk / v1_pk if (v2_ok and v1_pk > 0) else np.nan,
        "V2_int_Vs": trapz(np.abs(v2[m]), t[m]) if v2_ok else np.nan,
        "V2_per_dT_mV_K": slope_mV_K,
        "V2_per_A_VpA": v2_pk / float(i0) if v2_ok else np.nan,
        "cool_per_W_KpW": abs(d_cold_pk) / (v1_pk * float(i0)) if v1_pk > 0 else np.nan,
        "dT_cold_end_K": float(d_cold[-1]),
    }


def build_metrics_table(df_dual, df_sink, currents, window, frac=0.02, t_amb_c=20.0):
    """Tidy metrics table: one row per (model, drive current)."""
    rows = []
    for i0 in currents:
        r = run_metrics(df_dual, i0, LBL_DUAL, t_amb_c, window, frac)
        if r:
            rows.append(r)
    for i0 in currents:
        r = run_metrics(df_sink, i0, LBL_SINK, t_amb_c, window, frac)
        if r:
            rows.append(r)
    return pd.DataFrame(rows)


def metric_row(table, model, i0, col):
    """Single metric value of one model/current (NaN when not available)."""
    sel = table[(table["model"] == model) & np.isclose(table["I0_A"], i0)]
    return float(sel[col].iloc[0]) if len(sel) and col in sel else np.nan


def shade_drive_window(ax, df, i0, frac=0.02):
    """Shade the active drive interval behind the curves of one current."""
    sub = slice_current(df, i0)
    if sub is None:
        return None
    t = sub["time_s"].to_numpy(float)
    t_on, t_off, _, _ = detect_drive_window(t, sub["voltage_tec1_V"].to_numpy(float), frac)
    ax.axvspan(t_on, t_off, color="0.88", alpha=0.55, lw=0, zorder=0)
    ax.axvline(t_on, color="0.55", lw=2.4, ls="--", zorder=1)
    ax.axvline(t_off, color="0.55", lw=2.4, ls="--", zorder=1)
    return t_on, t_off


def legend_models(ax, **kwargs):
    """Two-entry legend describing the solid/dashed model convention."""
    handles = [Line2D([], [], color=C_DUAL, lw=4.4, label=LBL_DUAL),
               Line2D([], [], color=C_SINK, lw=4, ls="--", label=LBL_SINK)]
    ax.legend(handles=handles, **kwargs)


def annotate_marker(ax, x, y, label, color, marker="o", dx=1.2, dy=0.0, ha="left"):
    """Mark an extreme value and write its label next to it."""
    ax.plot([x], [y], marker=marker, ms=10, mfc="white", mec=color, mew=1.4, zorder=5)
    ax.annotate(label, (x, y), xytext=(x + dx, y + dy), fontsize=7.5, color=color,
                ha=ha, va="center", zorder=6)


def ddelta_curves(df_dual, df_sink, i0):
    """Castell-minus-Heatsink difference of hot/cold Delta-T on the common grid."""
    d = slice_current(df_dual, i0)
    s = slice_current(df_sink, i0)
    if d is None or s is None:
        return None
    t_common = np.intersect1d(np.round(d["time_s"].to_numpy(float), 3),
                              np.round(s["time_s"].to_numpy(float), 3))
    if t_common.size == 0:
        return None
    di = d.assign(_t=np.round(d["time_s"].to_numpy(float), 3)).set_index("_t")
    si = s.assign(_t=np.round(s["time_s"].to_numpy(float), 3)).set_index("_t")
    return {
        "t": t_common,
        "hot": di.loc[t_common, "Delta_Hot"].to_numpy(float) - si.loc[t_common, "Delta_Hot"].to_numpy(float),
        "cold": di.loc[t_common, "Delta_Cold"].to_numpy(float) - si.loc[t_common, "Delta_Cold"].to_numpy(float),
        "flux": di.loc[t_common, "flux_Wm2"].to_numpy(float) - si.loc[t_common, "flux_Wm2"].to_numpy(float),
    }

# =============================================================================
# 2. TEMPORAL GRAPHICS
# =============================================================================
def fig_temporal_grid(df_dual, df_sink, currents, index, fname="T1_temporal_grid.png"):
    """Per-current small multiples of the hot and cold probe response."""
    rows = [("Delta_Hot", r"(a) Hot probe   $\Delta T$ (Â°C)", "max"),
            ("Delta_Cold", r"(b) Cold probe   $\Delta T$ (Â°C)", "min")]
    n = len(currents)
    fig, axes = plt.subplots(2, n, figsize=(2.25 * n + 1.0, 6.4),
                             sharex=True, sharey="row", squeeze=False)

    for j, i0 in enumerate(currents):
        for i, (col, lab, mode) in enumerate(rows):
            ax = axes[i][j]
            d = slice_current(df_dual, i0)
            s = slice_current(df_sink, i0)
            if d is not None:
                shade_drive_window(ax, df_dual, i0)
                ax.plot(d["time_s"], d[col], color=C_DUAL, lw=3.8, zorder=3)
                k = int(np.nanargmax(d[col].to_numpy(float))) if mode == "max" \
                    else int(np.nanargmin(d[col].to_numpy(float)))
                ax.plot(d["time_s"].iloc[k], d[col].iloc[k], "o", ms=9, mfc="white",
                        mec=C_DUAL, mew=1.3, zorder=5)
            if s is not None:
                ax.plot(s["time_s"], s[col], color=C_SINK, lw=3.4, ls="--", zorder=3)
                k = int(np.nanargmax(s[col].to_numpy(float))) if mode == "max" \
                    else int(np.nanargmin(s[col].to_numpy(float)))
                ax.plot(s["time_s"].iloc[k], s[col].iloc[k], "s", ms=8, mfc="white",
                        mec=C_SINK, mew=1.3, zorder=5)

            ax.axhline(0, color="0.4", lw=2.4, ls=":", zorder=1)
            ax.set_title(f"$I_0$ = {i0:.1f} A", fontsize=9.5)
            ax.tick_params(labelsize=8)
            if i == 1:
                ax.set_xlabel("Time (s)", fontsize=9.5)
            if j == 0:
                ax.set_ylabel(lab, fontsize=9.5)
                legend_models(ax, loc="best", fontsize=7, framealpha=0.9)
    axes[0][0].set_xlim(0, 60)

    fig.suptitle("Temporal metric 1 - probe response of every drive current\n"
                 "(solid = Dual-TEC Castell, dashed = Heatsink, band = active drive, "
                 "markers = window peak)", fontsize=11.5, weight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, fname, index)


def fig_temporal_focus(df_dual, df_sink, i0, index, fname="T2_temporal_focus.png"):
    """Four-panel focus on one current: probes, drive voltage, flux, differences."""
    d = slice_current(df_dual, i0)
    s = slice_current(df_sink, i0)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4))
    ax = axes[0][0]
    shade_drive_window(ax, df_dual, i0)
    if d is not None:
        ax.plot(d["time_s"], d["Delta_Hot"], color=C_DUAL, lw=4, label=f"{LBL_DUAL} - hot")
        ax.plot(d["time_s"], d["Delta_Cold"], color=C_DUAL, lw=4, ls=":",
                label=f"{LBL_DUAL} - cold")
    if s is not None:
        ax.plot(s["time_s"], s["Delta_Hot"], color=C_SINK, lw=3.6, ls="--",
                label=f"{LBL_SINK} - hot")
        ax.plot(s["time_s"], s["Delta_Cold"], color=C_SINK, lw=3.6, ls=(0, (5, 1, 1, 1)),
                label=f"{LBL_SINK} - cold")
    ax.axhline(0, color="0.4", lw=2.4, ls=":")
    ax.set_title(r"(a) Probe response $\Delta T$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\Delta T$ (Â°C)")
    ax.legend(fontsize=7.5, ncol=2)

    ax = axes[0][1]
    shade_drive_window(ax, df_dual, i0)
    if d is not None:
        ax.plot(d["time_s"], d["voltage_tec1_V"], color=C_DUAL, lw=4, label=LBL_DUAL)
    if s is not None:
        ax.plot(s["time_s"], s["voltage_tec1_V"], color=C_SINK, lw=3.6, ls="--", label=LBL_SINK)
    ax.set_title(r"(b) Primary TEC terminal voltage $V_1$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.legend(fontsize=8)

    ax = axes[1][0]
    shade_drive_window(ax, df_dual, i0)
    if d is not None:
        ax.plot(d["time_s"], d["flux_Wm2"], color=C_DUAL, lw=4, label=LBL_DUAL)
    if s is not None:
        ax.plot(s["time_s"], s["flux_Wm2"], color=C_SINK, lw=3.6, ls="--", label=LBL_SINK)
    ax.axhline(0, color="0.4", lw=2.4, ls=":")
    ax.set_title(r"(c) Boundary heat flux (negative = cooling of the surface)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$q$ (W/m$^2$)")
    ax.legend(fontsize=8)

    ax = axes[1][1]
    shade_drive_window(ax, df_dual, i0)
    dfl = ddelta_curves(df_dual, df_sink, i0)
    if dfl is not None:
        ax.plot(dfl["t"], dfl["cold"], color="#2ca02c", lw=4,
                label=r"$\Delta T_{cold}$ (Castell - Heatsink)")
        ax.plot(dfl["t"], dfl["hot"], color="#9467bd", lw=4,
                label=r"$\Delta T_{hot}$ (Castell - Heatsink)")
    ax.axhline(0, color="0.4", lw=2.4, ls="-")
    ax.set_title("(d) Differential metrics (positive = Castell higher)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\Delta(\Delta T)$ (Â°C)")
    ax.legend(fontsize=8)

    for ax in axes.ravel():
        ax.set_xlim(0, 60)
    fig.suptitle(f"Temporal metric 2 - dual-view focus at $I_0$ = {i0:.2f} A",
                 fontsize=12.5, weight="bold", y=1.0)
    fig.tight_layout()
    save_fig(fig, fname, index)


def fig_normalized_response(df_dual, df_sink, i0, index, fname="T3_normalized_response.png"):
    """Dynamics comparison: window-normalised step response of both models."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for ax, (col, sign, ttl) in zip(
            axes,
            [("Delta_Cold", -1.0, r"(a) Cold probe:  $\Delta T/|\Delta T_{peak}|$"),
             ("Delta_Hot", 1.0, r"(b) Hot probe:  $\Delta T/|\Delta T_{peak}|$")]):
        for df, color, name, ls in [(df_dual, C_DUAL, LBL_DUAL, "-"),
                                    (df_sink, C_SINK, LBL_SINK, "--")]:
            sub = slice_current(df, i0)
            if sub is None:
                continue
            t = sub["time_s"].to_numpy(float)
            y = sub[col].to_numpy(float)
            t_on, t_off, _, _ = detect_drive_window(t, sub["voltage_tec1_V"].to_numpy(float))
            pk = np.nanmax(np.abs(y))
            if not np.isfinite(pk) or pk <= 0:
                continue
            ax.plot(t - t_on, y / pk, color=color, lw=4, ls=ls, label=name)
            lag = crossing_time(t, y, t_on, 0.63 * pk)
            ax.plot([lag], [sign * 0.63], marker="o", ms=11, mfc="white",
                    mec=color, mew=1.5, zorder=5)
            ax.annotate(f"$\\tau_{{63}}$ = {lag:.1f} s", (lag, sign * 0.63),
                        xytext=(lag + 1.0, sign * (0.63 - 0.30 * (1 if sign > 0 else -1))),
                        fontsize=8, color=color)
        for level in (0.63, 0.90):
            ax.axhline(sign * level, color="0.55", lw=2.4, ls=":")
        ax.axhline(0, color="0.4", lw=2.4)
        ax.set_xlim(-0.5, 25)
        ax.set_yticks(np.arange(-1.0, 1.01, 0.25))
        ax.set_title(ttl, fontsize=10.5)
        ax.set_xlabel(r"Time since drive start  $t - t_{on}$ (s)")
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel(r"Normalised $\Delta T$ (-)")
    fig.suptitle(f"Temporal metric 3 - normalised step response at $I_0$ = {i0:.2f} A "
                 "(dotted lines: 63 % and 90 % of peak)",
                 fontsize=11.5, weight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, fname, index)


def fig_metrics_vs_current(table, index, fname="T4_metrics_vs_current.png"):
    """Trends of the key temporal and energetic metrics versus drive current."""
    panels = [
        ("dT_cold_peak_K", r"(a) Cold probe peak $|\Delta T_c|$  (°C)"),
        ("dT_hot_peak_K", r"(b) Hot probe peak $\Delta T_h$  (°C)"),
        ("q_peak_Wm2", r"(c) Peak heat flux  (W/m$^2$)"),
        ("dT_ratio", r"(d) Cooling / heating ratio $|\Delta T_c|/\Delta T_h$"),
        ("t63_cold_s", r"(e) Cold-side response time $\tau_{63}$  (s)"),
        ("E_areal_J_m2", r"(f) Areal heat energy $\int|q|\,\mathrm{d}t$  (J/m$^2$)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.4))
    for ax, (col, ttl) in zip(axes.ravel(), panels):
        for label, color, ls, mk in [(LBL_DUAL, C_DUAL, "-", "o"), (LBL_SINK, C_SINK, "--", "s")]:
            sub = table[table["model"] == label].sort_values("I0_A")
            if sub.empty or col not in sub.columns:
                continue
            ax.plot(sub["I0_A"], sub[col], color=color, ls=ls, marker=mk, ms=9,
                    lw=3.8, label=label)
            if col == "dT_cold_peak_K":
                k = int(sub[col].abs().idxmax())
                ax.plot([sub["I0_A"].loc[k]], [sub[col].loc[k]], marker="*", ms=22,
                        color=color, zorder=5)
        if col == "t63_cold_s":
            ax.set_ylim(bottom=0)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel(r"Drive current $I_0$ (A)")
        ax.legend(fontsize=8)
    fig.suptitle("Temporal metric 4 - how the metrics scale with the drive current "
                 "(stars = current of maximum cooling)", fontsize=12, weight="bold", y=1.0)
    fig.tight_layout()
    save_fig(fig, fname, index)

# =============================================================================
# 3. ENERGETIC GRAPHICS (no load-resistance assumption anywhere)
# =============================================================================
def cumulative_integral(y, x):
    """Cumulative trapezoidal integral, starting at zero."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros(y.shape, dtype=float)
    if y.size > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def fig_energy_budget(table, index, fname="E1_energy_budget.png"):
    """Grouped bars of the energetic metrics, dual TEC versus heatsink."""
    metrics = [("E_areal_J_m2", r"(a) Areal heat energy $\int|q|\,\mathrm{d}t$  (J/m$^2$)"),
               ("dose_Ks", r"(b) Thermal dose $\int|\Delta T_c|\,\mathrm{d}t$  (K$\cdot$s)"),
               ("W_drive_J", r"(c) Drive work estimate $I_0\!\int|V_1|\,\mathrm{d}t$  (J)")]
    currents = sorted(table["I0_A"].unique())
    x = np.arange(len(currents))
    w = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))

    for ax, (col, ttl) in zip(axes, metrics):
        dv = np.array([metric_row(table, LBL_DUAL, c, col) for c in currents])
        sv = np.array([metric_row(table, LBL_SINK, c, col) for c in currents])
        ax.bar(x - w / 2, np.nan_to_num(dv), w, color=C_DUAL, edgecolor="k", lw=1.2,
               label=LBL_DUAL)
        ax.bar(x + w / 2, np.nan_to_num(sv), w, color=C_SINK, edgecolor="k", lw=1.2,
               label=LBL_SINK)
        for xi, a, b in zip(x, dv, sv):
            if np.isfinite(a) and np.isfinite(b) and b != 0:
                ymax = max(abs(a), abs(b))
                ax.annotate(f"{(a - b) / b * 100:+.0f}%", (xi, ymax * 1.03),
                            ha="center", fontsize=7.5, color="0.25")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c:.1f}" for c in currents], fontsize=8)
        ax.set_xlabel(r"Drive current $I_0$ (A)")
        ax.set_ylim(0, max(1e-9, np.nanmax([dv, sv])) * 1.18)
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Energetic metric 1 - window-integrated energy quantities "
                 "(labels: dual TEC relative to heatsink)",
                 fontsize=12, weight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, fname, index)


def fig_flux_cumulative(df_dual, df_sink, currents, index, fname="E2_flux_cumulative.png"):
    """Heat-flux transients and the cumulative areal energy of both models."""
    currents = sorted(currents)
    cmap = plt.get_cmap(CMAP_CURRENT)
    lo, hi = min(currents), max(currents)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))

    ax = axes[0]
    shade_drive_window(ax, df_dual, currents[0])
    for i0 in currents:
        c = cmap((i0 - lo) / (hi - lo) if hi > lo else 0.5)
        d = slice_current(df_dual, i0)
        s = slice_current(df_sink, i0)
        if d is not None:
            ax.plot(d["time_s"], np.abs(d["flux_Wm2"]), color=c, lw=3.6)
        if s is not None:
            ax.plot(s["time_s"], np.abs(s["flux_Wm2"]), color=c, lw=3, ls="--")
    ax.set_title(r"(a) Heat-flux magnitude $|q|(t)$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$|q|$ (W/m$^2$)")
    ax.set_xlim(0, 60)
    ax.annotate("solid = Castell\ndashed = Heatsink", xy=(0.62, 0.06),
                xycoords="axes fraction", fontsize=8, color="0.25")
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(lo, hi), cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"Drive current $I_0$ (A)")

    ax = axes[1]
    shade_drive_window(ax, df_dual, currents[0])
    for i0 in currents:
        c = cmap((i0 - lo) / (hi - lo) if hi > lo else 0.5)
        for df, ls, lw in [(df_dual, "-", 1.8), (df_sink, "--", 1.5)]:
            sub = slice_current(df, i0)
            if sub is None:
                continue
            t = sub["time_s"].to_numpy(float)
            cum = cumulative_integral(np.abs(sub["flux_Wm2"].to_numpy(float)), t)
            ax.plot(t, cum / 1e3, color=c, lw=lw, ls=ls)
            if ls == "-":
                ax.plot([t[-1]], [cum[-1] / 1e3], "o", ms=7, color=c)
    ax.set_title(r"(b) Cumulative areal heat energy $\int|q|\,\mathrm{d}t$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Cumulative energy (kJ/m$^2$)")
    ax.set_xlim(0, 60)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(lo, hi), cmap=cmap),
                      ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"Drive current $I_0$ (A)")

    fig.suptitle("Energetic metric 2 - areal heat pumping: transient and accumulated",
                 fontsize=12, weight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, fname, index)


def recovered_samples(df_dual, currents, window):
    """(I0, |dT_cold|, |V2|) samples of the parallel TEC inside the drive window."""
    i0s, xs, ys = [], [], []
    for i0 in currents:
        sub = slice_current(df_dual, i0)
        if sub is None or not has_v2_column(df_dual):
            continue
        t = sub["time_s"].to_numpy(float)
        m = window_mask(t, window)
        xs.append(np.abs(sub["Delta_Cold"].to_numpy(float)[m]))
        ys.append(np.abs(sub["voltage_tec2_V"].to_numpy(float)[m]))
        i0s.append(np.full(int(m.sum()), float(i0)))
    if not xs:
        return None
    return np.concatenate(i0s), np.concatenate(xs), np.concatenate(ys)


def fig_recovered_voltage(df_dual, table, window, index,
                          fname="E3_recovered_voltage.png"):
    """Open-circuit analysis of the recovered voltage of the parallel TEC."""
    if not has_v2_column(df_dual):
        print("      [skip] no parallel-TEC voltage in this export")
        return
    currents = sorted(df_dual["I0"].unique())
    cmap = plt.get_cmap(CMAP_CURRENT)
    norm = plt.Normalize(min(currents), max(currents))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))

    # (a) transients -------------------------------------------------------
    ax = axes[0][0]
    shade_drive_window(ax, df_dual, currents[0])
    for i0 in currents:
        sub = slice_current(df_dual, i0)
        if sub is None:
            continue
        t = sub["time_s"].to_numpy(float)
        v2 = np.abs(sub["voltage_tec2_V"].to_numpy(float))
        c = cmap(norm(i0))
        ax.plot(t, v2 * 1e3, color=c, lw=3.4)
        m = window_mask(t, window)
        if m.any():
            k = int(np.nanargmax(v2 * m))
            ax.plot(t[k], v2[k] * 1e3, "o", ms=8, mfc="white", mec=c, mew=1.2, zorder=5)
    ax.set_title(r"(a) Recovered (open-circuit) voltage $V_2(t)$", fontsize=10.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$V_2$ (mV)")
    ax.set_xlim(0, 60)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      fraction=0.046, pad=0.02)
    cb.set_label(r"Drive current $I_0$ (A)")

    # (b) peak voltage versus drive current --------------------------------
    ax = axes[0][1]
    i0s = np.array([metric_row(table, LBL_DUAL, c, "I0_A") for c in currents])
    v2p = np.array([metric_row(table, LBL_DUAL, c, "V2_peak_V") for c in currents])
    v1p = np.array([metric_row(table, LBL_DUAL, c, "V1_peak_V") for c in currents])
    ok = np.isfinite(v2p)
    ax.plot(i0s[ok], v2p[ok] * 1e3, "o-", color="#2ca02c", lw=3.8, ms=11,
            label=r"recovered $V_2$ peak")
    ratio = v2p[ok] / v1p[ok]
    if ok.any() and np.sum(i0s[ok] ** 2) > 0:
        k_fit = float(np.sum(v2p[ok] * i0s[ok]) / np.sum(i0s[ok] ** 2))
        ss_res = float(np.sum((v2p[ok] - k_fit * i0s[ok]) ** 2))
        ss_tot = float(np.sum((v2p[ok] - v2p[ok].mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        ax.plot(i0s[ok], k_fit * i0s[ok] * 1e3, "--", color="0.35", lw=2.6,
                label=f"fit $V_2 = {k_fit:.3f}\\,I_0$  ($R^2$={r2:.3f})")
    ax.set_xlabel(r"Drive current $I_0$ (A)")
    ax.set_ylabel(r"Recovered voltage $V_2$ (mV)", color="#2ca02c")
    ax.tick_params(axis="y", colors="#2ca02c")
    ax.set_title("(b) Recovered voltage versus drive current", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")

    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.plot(i0s, v1p, "s--", color=C_DUAL, lw=3.2, ms=9, label=r"drive $V_1$ peak")
    ax2.set_ylabel(r"Drive voltage $V_1$ peak (V)", color=C_DUAL)
    ax2.tick_params(axis="y", colors=C_DUAL)
    ax2.legend(fontsize=8, loc="lower right")
    if ratio.size:
        ax.annotate(f"$V_2/V_1$ = {ratio[0]*100:.1f} % ... {ratio[-1]*100:.1f} %",
                    xy=(0.04, 0.62), xycoords="axes fraction", fontsize=8.5, color="0.25")

    # (c) coupling with the cold-side temperature -------------------------
    ax = axes[1][0]
    smp = recovered_samples(df_dual, currents, window)
    if smp is not None:
        i0s_s, xs, ys = smp
        sc = ax.scatter(xs, ys * 1e3, c=i0s_s, cmap=cmap, norm=norm, s=36, alpha=0.75)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(r"Drive current $I_0$ (A)")
        denom = float(np.sum(xs ** 2))
        if denom > 0:
            slope = float(np.sum(ys * xs) / denom)
            xx = np.linspace(0.0, float(np.nanmax(xs)), 50)
            ax.plot(xx, slope * xx * 1e3, "--", color="0.3", lw=2.8,
                    label=f"slope = {slope*1e3:.2f} mV/K")
            ax.legend(fontsize=8, loc="upper left")
    ax.set_title("(c) Thermoelectric coupling of the parallel TEC", fontsize=10.5)
    ax.set_xlabel(r"Cold-side temperature difference $|\Delta T_c|$ (°C)")
    ax.set_ylabel(r"$V_2$ (mV)")

    # (d) normalised recovered voltage ------------------------------------
    ax = axes[1][1]
    per_a = np.array([metric_row(table, LBL_DUAL, c, "V2_per_A_VpA") for c in currents])
    ax.plot(i0s[ok], per_a[ok] * 1e3, "o-", color="#9467bd", lw=3.8, ms=11,
            label=r"$V_2/I_0$  (mV/A)")
    ax.set_xlabel(r"Drive current $I_0$ (A)")
    ax.set_ylabel(r"$V_2/I_0$  (mV/A)", color="#9467bd")
    ax.tick_params(axis="y", colors="#9467bd")
    ax.set_title("(d) Recovered voltage per ampere and voltage ratio", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")

    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.plot(i0s[ok], ratio * 100, "s--", color="0.35", lw=3.2, ms=9,
             label=r"$V_2/V_1$  (%)")
    ax2.set_ylabel(r"$V_2/V_1$  (%)", color="0.35")
    ax2.tick_params(axis="y", colors="0.35")
    ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle("Recovered voltage of the thermally parallel TEC (Castell Global Variable "
                 f"Probe 3) - open-circuit view, window {window[0]:.0f}-{window[1]:.0f} s",
                 fontsize=12.5, weight="bold", y=1.0)
    fig.tight_layout()
    save_fig(fig, fname, index)

# =============================================================================
# 4. SPATIAL GRAPHICS - inner surface
# =============================================================================
def load_surface_current(path, i0, cache, t_amb_c=20.0):
    """Load every time step of one drive current from a surface export (cached).

    The surface exports are written point-by-point per time step, so the loader
    streams the whole file once per (file, current) and the frame is cached: a
    full Castell current is ~0.94 M rows / 45 MB / ~6 s.
    """
    key = (path, round(float(i0), 4))
    if key not in cache:
        print(f"      loading {os.path.basename(path)} @ I0={i0:.2f} A ...")
        df = load_surface_inner(path, target_currents=[i0], default_tamb_c=t_amb_c,
                                chunksize=CFG["SURFACE_CHUNK"])
        df["time_s"] = df["time_s"].round(4)
        df["I0"] = df["I0"].round(4)
        cache[key] = df
    return cache[key]


def plane_axes(df):
    """The two in-plane coordinates of a surface frame (largest spans)."""
    spans = [(c, float(df[c].max() - df[c].min())) for c in ("x", "y", "z")]
    spans.sort(key=lambda t: -t[1])
    return spans[0][0], spans[1][0]


def time_slice(df, t_target, atol=None):
    """All mesh points of the SINGLE time step closest to t_target.

    The tolerance is derived from the actual time grid: the new exports are
    0.15 s apart, so a fixed +-0.26 s window would merge four steps and thereby
    duplicate every mesh node (which corrupts maps AND the area metric).
    """
    t = df["time_s"].to_numpy(float)
    uniq = np.unique(t)
    if uniq.size == 0:
        return None
    k = int(np.argmin(np.abs(uniq - t_target)))
    dt = np.diff(uniq)
    step = float(np.min(dt)) if dt.size else 0.5
    tol = atol if atol is not None else max(1e-6, 0.4 * step)
    if abs(uniq[k] - t_target) > max(1.5 * step, tol):
        return None
    sub = df[np.isclose(t, uniq[k], atol=1e-6)]
    return sub if not sub.empty else None


def grid_from_scatter(df, res=None, bounds=None):
    """Interpolate a surface frame onto a regular grid.

    Delaunay-based LINEAR interpolation is used on purpose: the scipy
    griddata 'cubic' mode was measured to overshoot by more than +30 degC on
    these meshes, which would corrupt both the maps and their differences.
    Returns a dict with the coordinate vectors, the meshgrids, the field and a
    ready-to-use imshow extent.
    """
    a1, a2 = plane_axes(df)
    res = int(res or CFG["MAP_RESOLUTION"])
    if bounds is None:
        b1 = (float(df[a1].min()), float(df[a1].max()))
        b2 = (float(df[a2].min()), float(df[a2].max()))
    else:
        b1, b2 = bounds
    g1 = np.linspace(b1[0], b1[1], res)
    g2 = np.linspace(b2[0], b2[1], res)
    G1, G2 = np.meshgrid(g1, g2)
    pts = np.column_stack([df[a1].to_numpy(float), df[a2].to_numpy(float)])
    val = LinearNDInterpolator(pts, df["Delta_T"].to_numpy(float))(G1, G2)
    # Display convention: x = a1 (g1), y = a2 (g2), so the raw array maps
    # directly with extent [g1, g2] (90 deg transposed vs the earlier
    # x = a2 / y = a1 layout).
    return {"a1": a1, "a2": a2, "g1": g1, "g2": g2, "G1": G1, "G2": G2,
            "val": val, "extent": [g1[0], g1[-1], g2[0], g2[-1]]}


def common_map(df_a, df_b, res=None, bounds=None):
    """Both models on one shared grid (intersection of their footprints)."""
    a1, a2 = plane_axes(df_a)
    if bounds is None:
        b1 = (max(float(df_a[a1].min()), float(df_b[a1].min())),
              min(float(df_a[a1].max()), float(df_b[a1].max())))
        b2 = (max(float(df_a[a2].min()), float(df_b[a2].min())),
              min(float(df_a[a2].max()), float(df_b[a2].max())))
    else:
        b1, b2 = bounds
    res = int(res or CFG["MAP_RESOLUTION"])
    g1 = np.linspace(b1[0], b1[1], res)
    g2 = np.linspace(b2[0], b2[1], res)
    G1, G2 = np.meshgrid(g1, g2)

    def interp(df):
        pts = np.column_stack([df[a1].to_numpy(float), df[a2].to_numpy(float)])
        return LinearNDInterpolator(pts, df["Delta_T"].to_numpy(float))(G1, G2)

    return {"a1": a1, "a2": a2, "g1": g1, "g2": g2, "G1": G1, "G2": G2,
            "va": interp(df_a), "vb": interp(df_b),
            "extent": [g1[0], g1[-1], g2[0], g2[-1]]}


def spot_stats(val, g1, g2):
    """Coldest/hottest pixel of a gridded frame: (coord1, coord2, value).

    Grid convention (see grid_from_scatter): with the default ``xy`` meshgrid,
    ``val`` has shape ``(len(g2), len(g1))``, i.e. rows run along ``g2`` and
    columns along ``g1``.  The unravelled ``(row, col)`` index therefore maps
    to ``(a1=g1[col], a2=g2[row])``.
    """
    if not np.isfinite(val).any():
        return None
    i_cold = np.unravel_index(np.nanargmin(val), val.shape)
    i_hot = np.unravel_index(np.nanargmax(val), val.shape)
    return {"cold": (float(g1[i_cold[1]]), float(g2[i_cold[0]]), float(val[i_cold])),
            "hot": (float(g1[i_hot[1]]), float(g2[i_hot[0]]), float(val[i_hot]))}


def check_no_overshoot(fr, scatter_df, tag=""):
    """Verify that interpolation stayed inside the measured data range."""
    lo, hi = float(scatter_df["Delta_T"].min()), float(scatter_df["Delta_T"].max())
    glo, ghi = float(np.nanmin(fr["val"])), float(np.nanmax(fr["val"]))
    ok = (glo >= lo - 1e-6) and (ghi <= hi + 1e-6)
    if not ok:
        print(f"      [check] {tag}: interpolated {glo:.2f}..{ghi:.2f} vs data {lo:.2f}..{hi:.2f}")
    return ok


# =============================================================================
# 4b. SPATIAL FIGURES - inner-surface maps and model-to-model difference
# =============================================================================
from scipy.spatial import Delaunay

from sim_loaders import VOL_THRESHOLDS, T_END_S


def frame_areas(sub, thresholds=None):
    """Cooled/heated inner-surface AREA (mm^2) of one loaded time slice.

    Delaunay triangulation of the in-plane point cloud; every triangle is
    classified by the mean Delta-T of its three vertices.  Validated against the
    Castell export, whose hull is exactly 900 mm^2 (30 x 30 mm).
    """
    thresholds = thresholds or VOL_THRESHOLDS
    a1, a2 = plane_axes(sub)
    pts = np.column_stack([sub[a1].to_numpy(float), sub[a2].to_numpy(float)])
    tri = Delaunay(pts)
    s = tri.simplices
    p0, p1, p2 = pts[s[:, 0]], pts[s[:, 1]], pts[s[:, 2]]
    area = 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) -
                        (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
    t_mean = sub["Delta_T"].to_numpy(float)[s].mean(axis=1)
    return {thr: (float(area[t_mean < -thr].sum()), float(area[t_mean > thr].sum()))
            for thr in thresholds}


def draw_map(ax, fr, norm, title=None, contour_levels=8, cmap=CMAP_T, annotate=True):
    """Heat-map panel: interpolated field + contours + cold/hot spot markers.

    Display convention: x = a1 (g1), y = a2 (g2); ``fr["val"]`` has shape
    ``(len(g2), len(g1))`` (default ``xy`` meshgrid) so it maps directly with
    ``extent=[g1, g2]``.  This is 90 deg transposed vs the earlier
    x = a2 / y = a1 layout; image, contours and markers all share it.
    """
    ax.grid(False)
    im = ax.imshow(fr["val"], origin="lower", extent=fr["extent"], cmap=cmap,
                   norm=norm, aspect="auto", interpolation="bilinear")
    z = np.ma.masked_invalid(fr["val"])
    if z.count() > 2:
        try:
            ax.contour(fr["G1"], fr["G2"], z, levels=contour_levels, colors="k",
                       linewidths=2.4, alpha=0.45)
        except Exception:
            pass
    if annotate:
        st = spot_stats(fr["val"], fr["g1"], fr["g2"])
        if st:
            for key, mk, col in (("cold", "v", "#08306b"), ("hot", "^", "#67000d")):
                c1, c2, v = st[key]
                ax.plot([c1], [c2], marker=mk, ms=10, mfc="white", mec="k", mew=0.8,
                        zorder=5)
                ax.annotate("%+.1f" % v, (c1, c2), xytext=(c1 + 0.7, c2 + 0.7),
                            fontsize=7, color=col, zorder=6)
    ax.set_xlabel("%s (mm)" % fr["a1"], fontsize=8.5)
    ax.set_ylabel("%s (mm)" % fr["a2"], fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    if title:
        ax.set_title(title, fontsize=9)
    return im


def fig_spatial_maps_grid(cache, currents, times, index,
                          fname="S1_spatial_maps_grid.png"):
    """Dual-TEC inner-surface maps: one row per current plus a zoom on the spots.

    Every panel is annotated with the 2D cooled/heated area (mm^2) obtained from
    the same mesh, so the maps and the numbers of the volume probes can be read
    together.
    """
    n_t = len(times)
    hw = CFG["ROI_HALF_WIDTH_MM"]
    fig, axes = plt.subplots(len(currents), n_t + 1,
                             figsize=(3.2 * (n_t + 1), 3.1 * len(currents)), squeeze=False)

    for r, i0 in enumerate(currents):
        df = load_surface_current(CFG["DUAL_SURF_CSV"], i0, cache, CFG["T_AMB_C"])
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
                ax.set_title("$t$ = %.0f s (no data)" % t_tgt, fontsize=9)
                continue
            fr = grid_from_scatter(sub)
            areas = frame_areas(sub)
            last_fr, last_areas = fr, areas
            head = ("$I_0$ = %.1f A,  " % i0) if r == 0 else ""
            title = ("%s$t$ = %.0f s\ncool %.0f | heat %.0f mm$^2$ @0.5 °C"
                     % (head, t_tgt, areas[0.5][0], areas[0.5][1]))
            draw_map(ax, fr, norm, title=title)
            check_no_overshoot(fr, sub, tag="I0=%.1f t=%.0f" % (i0, t_tgt))

        ax = axes[r][n_t]
        if last_fr is not None:
            st = spot_stats(last_fr["val"], last_fr["g1"], last_fr["g2"])
            c1, c2 = (0.5 * (st["cold"][0] + st["hot"][0]),
                      0.5 * (st["cold"][1] + st["hot"][1])) if st else (0.0, 0.0)
            bounds = ((max(c1 - hw, last_fr["g1"][0]), min(c1 + hw, last_fr["g1"][-1])),
                      (max(c2 - hw, last_fr["g2"][0]), min(c2 + hw, last_fr["g2"][-1])))
            sub_last = time_slice(df, times[-1])
            fr_roi = grid_from_scatter(sub_last, bounds=bounds)
            im = draw_map(ax, fr_roi, norm,
                          title="ROI zoom (cold & hot spot)\n$t$ = %.0f s" % times[-1])
            cax = make_axes_locatable(ax).append_axes("right", size="6%", pad=0.06)
            cb = fig.colorbar(im, cax=cax)
            cb.set_label(r"Inner-surface $\Delta T$ (Â°C)", fontsize=8)
            cb.ax.tick_params(labelsize=7.5)
            if last_areas:
                ax.annotate("cooled %.1f mm$^2$\nheated %.1f mm$^2$" %
                            (last_areas[0.5][0], last_areas[0.5][1]),
                            xy=(0.02, 0.02), xycoords="axes fraction", fontsize=7.5,
                            color="0.2")
        else:
            ax.grid(False)
            ax.axis("off")

    fig.suptitle("Spatial metric - Dual-TEC (Castell) inner-surface maps\n"
                 "shared colour scale per row; markers: coldest (v) / hottest (^); "
                 "title numbers = 2D cooled/heated area",
                 fontsize=12, weight="bold", y=1.0)
    fig.tight_layout()
    save_fig(fig, fname, index)


def fig_spatial_diff(cache, i0, t_target, index, fname="S2_spatial_diff.png"):
    """Castell | Heatsink | difference of the inner surface on the common footprint.

    The two models have different in-plane extents (Castell reaches y = 17 mm,
    the heatsink only 3.4 mm), so both fields are re-interpolated on the
    INTERSECTION of their footprints before subtracting.
    """
    dd = load_surface_current(CFG["DUAL_SURF_CSV"], i0, cache, CFG["T_AMB_C"])
    sd = load_surface_current(CFG["SINK_SURF_CSV"], i0, cache, CFG["T_AMB_C"])
    sub_d, sub_s = time_slice(dd, t_target), time_slice(sd, t_target)
    if sub_d is None or sub_s is None:
        print("      [skip] no surface comparison data for I0=%.2f A, t=%.0f s" % (i0, t_target))
        return

    cm = common_map(sub_d, sub_s)
    va, vb = cm["va"], cm["vb"]
    diff = va - vb
    lim = float(np.nanmax(np.abs(np.concatenate([va.ravel(), vb.ravel()]))))
    dlim = float(np.nanmax(np.abs(diff)))
    lim = lim if np.isfinite(lim) and lim > 0 else 1.0
    dlim = dlim if np.isfinite(dlim) and dlim > 0 else 1.0

    ar_d, ar_s = frame_areas(sub_d), frame_areas(sub_s)
    frames = [dict(cm, val=va), dict(cm, val=vb), dict(cm, val=diff)]
    norms = [TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
             TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
             TwoSlopeNorm(vmin=-dlim, vcenter=0.0, vmax=dlim)]
    titles = ["Dual-TEC (Castell)\ncool %.1f | heat %.1f mm$^2$ @0.5 °C" % ar_d[0.5],
              "Heatsink\ncool %.1f | heat %.1f mm$^2$ @0.5 °C" % ar_s[0.5],
              "difference (Castell - Heatsink)\nmax %.2f K | RMS %.2f K"
              % (float(np.nanmax(diff)), float(np.sqrt(np.nanmean(diff ** 2))))]
    cmaps = [CMAP_T, CMAP_T, CMAP_D]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    for ax, fr, norm, ttl, cmap in zip(axes, frames, norms, titles, cmaps):
        im = draw_map(ax, fr, norm, title=ttl, cmap=cmap, annotate=(cmap != CMAP_D))
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(r"$\Delta T$ (Â°C)" if cmap != CMAP_D else r"$\Delta$ (Â°C)", fontsize=8)
        cb.ax.tick_params(labelsize=7.5)

    overlap = 100.0 * np.isfinite(diff).mean()
    fig.suptitle("Spatial comparison at $I_0$ = %.2f A, $t$ = %.0f s\n"
                 "common footprint in (%s, %s) [mm], %.0f %% of the grid covered by both models"
                 % (i0, t_target, cm["a1"], cm["a2"], overlap),
                 fontsize=12, weight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, fname, index)


# =============================================================================
# 5. PIPELINE
# =============================================================================
def write_index(index, extra_lines, fname="figure_index.md"):
    """Write a small table of contents for the generated figures."""
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Dual-TEC (Castell) vs Heatsink - figure index\n\n")
        for line in extra_lines:
            f.write("%s\n" % line)
        f.write("\n## Figures\n\n")
        for nm in index:
            f.write("- `%s`\n" % nm)
    return path


def main():
    """Generate the temporal, energetic and spatial graphics of both models."""
    apply_style()
    t0 = time.time()
    index = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 100)
    print("Dual-TEC (Castell) vs Heatsink - improved temporal / energetic / spatial graphics")
    print("=" * 100)

    print("\n[1/6] Loading probe tables ...")
    df_dual = load_accumulated_probes(CFG["DUAL_PROBES_CSV"], CFG["T_AMB_C"])
    df_sink = load_accumulated_probes(CFG["SINK_PROBES_CSV"], CFG["T_AMB_C"])
    currents = sorted(set(np.round(df_dual["I0"].unique(), 4)))
    print("      %-9s %d rows | %d currents | probe 3 present: %s"
          % ("Dual-TEC", len(df_dual), len(currents),
             bool(df_dual["voltage_tec2_V"].notna().any())))
    print("      %-9s %d rows | %d currents" % ("Heatsink", len(df_sink),
                                                len(set(np.round(df_sink["I0"].unique(), 4)))))

    print("\n[2/6] Metrics table (window %.1f-%.1f s) ..." % CFG["DRIVE_WINDOW_S"])
    table = build_metrics_table(df_dual, df_sink, currents,
                                tuple(CFG["DRIVE_WINDOW_S"]),
                                CFG["WINDOW_THRESHOLD_FRAC"], CFG["T_AMB_C"])
    csv_path = os.path.join(OUTPUT_DIR, "metrics_table.csv")
    table.to_csv(csv_path, index=False)
    print("      saved metrics_table.csv (%d rows x %d columns)" % table.shape)
    cols = [c for c in ("model", "I0_A", "dT_cold_peak_K", "dT_hot_peak_K", "q_peak_Wm2",
                        "E_areal_J_m2", "V1_peak_V", "V2_peak_V", "t63_cold_s")
            if c in table.columns]
    print(table[cols].to_string(index=False))

    cache = {}
    grid_cur = [c for c in CFG["GRID_CURRENTS"] if any(np.isclose(currents, c))] or currents
    focus = CFG["FOCUS_CURRENT"]
    if focus not in currents:
        focus = currents[len(currents) // 2]

    if CFG["RUN_TEMPORAL"]:
        print("\n[3/6] Temporal figures ...")
        fig_temporal_grid(df_dual, df_sink, grid_cur, index)
        fig_temporal_focus(df_dual, df_sink, focus, index)
        fig_normalized_response(df_dual, df_sink, focus, index)
        fig_metrics_vs_current(table, index)

    if CFG["RUN_ENERGETIC"]:
        print("\n[4/6] Energetic figures ...")
        fig_energy_budget(table, index)
        fig_flux_cumulative(df_dual, df_sink, grid_cur, index)

    if CFG["RUN_RECOVERED_VOLTAGE"]:
        print("\n[5/6] Recovered-voltage analysis (thermally parallel TEC) ...")
        fig_recovered_voltage(df_dual, table, tuple(CFG["DRIVE_WINDOW_S"]), index)

    if CFG["RUN_SPATIAL"]:
        print("\n[6/6] Spatial figures (streaming the inner-surface exports) ...")
        spatial_cur = [c for c in CFG["SPATIAL_CURRENTS"]
                       if any(np.isclose(currents, c))] or [focus]
        fig_spatial_maps_grid(cache, spatial_cur, CFG["SPATIAL_TIMES_S"], index)
        if os.path.exists(CFG["SINK_SURF_CSV"]):
            fig_spatial_diff(cache, focus, CFG["SPATIAL_TIMES_S"][-1], index)
        else:
            print("      [skip] heatsink surface export not found: %s" % CFG["SINK_SURF_CSV"])

    path = write_index(index, [
        "Generated by `reading_comsol_dual_tec_graphics.py` in %.1f s." % (time.time() - t0),
        "Metric window: %.1f-%.1f s | T_amb = %.1f Â°C" % (CFG["DRIVE_WINDOW_S"][0],
                                                          CFG["DRIVE_WINDOW_S"][1],
                                                          CFG["T_AMB_C"]),
        "Recovered voltage is treated strictly as an open-circuit quantity "
        "(no load resistance is assumed).",
    ])
    print("\n      %d figures written to %s" % (len(index), OUTPUT_DIR))
    print("      index: %s" % path)
    print("\nDone in %.1f s." % (time.time() - t0))


if __name__ == "__main__":
    main()
