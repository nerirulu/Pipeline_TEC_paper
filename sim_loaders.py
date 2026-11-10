# =============================================================================
# reading_comsol_dual_tec.py
# ---------------------------
# Reads the COMSOL exports of the two configurations and compares them per
# drive current:
#
#   1) Dual-TEC (Castell) : Config_Castell.csv  + Config_Castell_surf.csv
#   2) Heatsink (Concise) : Config_heatsink.csv + Config_heatsink_surf.csv
#
# Content of the 30/09/2026 exports:
#   * hot/cold temperature probes, boundary heat flux, terminal voltages
#     (Probe 1 = driven TEC, Probe 3 = thermally parallel TEC, Castell only)
#   * heated/cooled VOLUME probes at 0.5, 1, 2 and 5 degC (domain probes, mm^3);
#     the threshold is parsed from the probe NAME because the two files order
#     these columns differently
#   * inner-surface temperature fields (point clouds on one plane, x = 1.4225 mm)
#
# Section 9 adds the per-current comparison of four metric families
# (temperature / spatial volume+area / temporal / energy) plus its figures:
#     python reading_comsol_dual_tec.py            -> main_comparison()
#     python reading_comsol_dual_tec_graphics.py   -> extended graphics suite
# =============================================================================
import os
import io
import csv
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
_BASE_FINAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Final files")


def _first_existing(path):
    """Return `path` as-is when it exists, otherwise try with/without ".csv".

    COMSOL's "Export > Data" dialog drops the extension for some exports, so
    Config_heatsink_surf exists on disk without the .csv suffix while the rest
    of the pipeline (and the figure index) refers to Config_heatsink_surf.csv.
    Without this, os.path.exists() fails and the whole 2D area step is silently
    skipped, which leaves the area_* columns out of the metrics CSV.
    """
    if os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    alt = base if ext else path + ".csv"
    if os.path.exists(alt):
        return alt
    return path


CONFIG = {
    # Dual-TEC (Castell) files
    "DUAL_TEC_PROBES_CSV": os.path.join(_BASE_FINAL, "Probes_auto.csv"),
    "DUAL_TEC_INNER_SURF_CSV": _first_existing(
        os.path.join(_BASE_FINAL, "Surface_auto.csv")),
    
    # Comparison heatsink reference files (Concise model)
    "HEATSINK_PROBES_CSV": os.path.join(_BASE_FINAL, "Probes_auto.csv"),
    "HEATSINK_INNER_SURF_CSV": _first_existing(
        os.path.join(_BASE_FINAL, "Surface_auto.csv")),
    
    # Probes settings
    "T_AMB_DEFAULT_C": 20.0,
    "COMPARE_CURRENTS": [0.1, 0.2, 0.3, 0.4, 0.5],        # Shared currents between Dual TEC and Heatsink
    "SURFACE_CURRENTS": [0.1, 0.2, 0.3, 0.4, 0.5],   # Currents to inspect for surface maps
    # Currents for the 2D area metrics.  The first run streams the surface export
    # ONCE for all of them (~20 s for the 429 MB Castell file) and caches the
    # Delaunay result in OUTPUT/_surface_area_cache, so later runs are fast.
    "AREA_CURRENTS": [0.1, 0.2, 0.3, 0.4, 0.5],
    "SURFACE_TIMES_S": [5.0, 15.0, 25.0, 40.0, 55.0],
    "SURFACE_RESOLUTION": 150,
    "SURFACE_CHUNK": 500_000,
    
    # Flags to control execution
    "RUN_PROBES_ANALYSIS": True,
    "RUN_ELECTRICAL_ANALYSIS": True,
    "RUN_HEATSINK_COMPARISON": True,
    "RUN_SURFACE_MAPS": True,
    "RUN_SURFACE_COMPARISON": True,

    # ---- per-current comparison (30/09/2026 exports with volume probes) ----
    "RUN_CURRENT_COMPARISON": True,
    "COMPARE_ALL_CURRENTS": True,      # else: only CONFIG["COMPARE_CURRENTS"]
    "COMPARISON_CSV": os.path.join(_BASE_FINAL, "..", "OUTPUT_auto",
                                   "comparison_per_current.csv"),
    "COMPARISON_FIG_DIR": os.path.join(_BASE_FINAL, "..", "OUTPUT_auto", "comparison"),
    "USE_AREA_CACHE": True,            # cache the Delaunay area series (.npz) per export/current
    "SAVE_FIGURES": True,
    "FIGURE_DPI": 200,
    "SHOW_FIGURES": False,
}

# Volume-threshold probes of the 30/09/2026 exports:
#   vol_heat<Thr>C / vol_cool<Thr>C   [mm^3]  -> heated / cooled domain volume
VOL_THRESHOLDS = (0.5, 1.0, 2.0, 5.0)

# The threshold is parsed from the probe NAME and never from its position,
# because the two exports order these columns differently (Castell lists
# 0.5, 2, 1, 5 while the heatsink lists 0.5, 1, 2, 5).  IGNORECASE is required:
# the header carries an upper-case "C" ("vol_heat0.5C") that is lower-cased.
_VOL_RE = re.compile(r"vol_(heat|cool)([0-9.]+)c", re.IGNORECASE)

# Protocol phases of the 25 s drive pulse used by every comparison metric.
# They are given explicitly because the applied current CANNOT be recovered from
# V1: after switch-off the TEC keeps generating a Seebeck voltage, so a
# |V1|-threshold "window" drifts from ~30 s (1 A) to ~40 s (0.1 A).
T_ON_S = 4.0        # current switched on
T_END_S = 29.0      # end of the 25 s drive pulse
T_FINAL_S = 60.0    # end of the exported window

# =============================================================================
# 1. PARSING HELPERS
# =============================================================================
def _to_numeric(df):
    """Coerce every column to numeric, drop rows with NaN, compact index."""
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().reset_index(drop=True)


def parse_header_line(header_line):
    """Parse comma-separated header line, handling COMSOL quoting patterns."""
    first = list(csv.reader([header_line]))[0]
    if len(first) == 1 and "," in first[0]:
        first = list(csv.reader([first[0]]))[0]
    return [c.lstrip("%").strip() for c in first]

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata

warnings.filterwarnings("ignore", message=".*mixed types.*", category=UserWarning)

# =============================================================================
# 2. ACCUMULATED PROBES LOADER (Dual TEC & Heatsink Compatible)
# =============================================================================
def load_accumulated_probes(file_path, default_tamb_c=20.0):
    """Load an accumulated-probes CSV.
    Supports single or dual TEC voltage probes.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Probes file not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        raw = f.read().splitlines()

    lines = [ln.strip() for ln in raw
             if ln.strip() and not ln.strip().startswith(("% Version", "% Date", "% Table", "% Dimension", "% Nodes"))]
    if not lines:
        raise ValueError(f"No data rows found in {file_path}")

    header_idx = None
    for idx, ln in enumerate(lines):
        ln_clean = ln.lstrip("%").strip()
        if "I0" in ln_clean and ("Time" in ln_clean or "t" in ln_clean):
            header_idx = idx
            break

    if header_idx is None:
        raise ValueError(f"Could not locate table header in {file_path}")

    cols = parse_header_line(lines[header_idx])
    payload = io.StringIO("\n".join(lines[header_idx + 1:]))
    df = pd.read_csv(payload, sep=",", header=None, names=cols, engine="python")
    df = _to_numeric(df)

    def find_col(predicate):
        for c in df.columns:
            if predicate(c):
                return c
        return None

    c_i0 = find_col(lambda c: "i0" in c.lower())
    c_time = find_col(lambda c: "time" in c.lower() or c.lower() == "t")
    c_tamb = find_col(lambda c: "t_amb" in c.lower())
    c_final = find_col(lambda c: "final_time" in c.lower())
    c_dist = find_col(lambda c: "dist" in c.lower())
    c_flux = find_col(lambda c: "flux" in c.lower())
    c_curr = find_col(lambda c: "current" in c.lower() and "probe" in c.lower())
    
    # Identify voltage probes
    v_cols = [c for c in df.columns if "voltage" in c.lower()]
    v1_col, v2_col = None, None
    for vc in v_cols:
        vc_low = vc.lower()
        if "probe 1" in vc_low or "global variable probe 1" in vc_low:
            v1_col = vc
        elif "probe 3" in vc_low or "global variable probe 3" in vc_low:
            v2_col = vc
    if v1_col is None and len(v_cols) >= 1:
        v1_col = v_cols[0]
    if v2_col is None and len(v_cols) >= 2:
        v2_col = v_cols[1]

    # Temperature probes
    c_hot = find_col(lambda c: "temperature" in c.lower() and "hot" in c.lower())
    c_cold = find_col(lambda c: "temperature" in c.lower() and "cold" in c.lower())
    c_max = find_col(lambda c: "temperature" in c.lower() and "max" in c.lower())
    c_min = find_col(lambda c: "temperature" in c.lower() and "min" in c.lower())

    renames = {}
    if c_i0: renames[c_i0] = "I0"
    if c_time: renames[c_time] = "time_s"
    if c_tamb: renames[c_tamb] = "T_amb_K"
    if c_final: renames[c_final] = "final_time_s"
    if c_dist: renames[c_dist] = "dist_mm"
    if c_flux: renames[c_flux] = "flux_Wm2"
    if c_curr: renames[c_curr] = "current_A"
    if v1_col: renames[v1_col] = "voltage_tec1_V"
    if v2_col: renames[v2_col] = "voltage_tec2_V"
    if c_hot: renames[c_hot] = "T_Hot"
    if c_cold: renames[c_cold] = "T_Cold"
    if c_max: renames[c_max] = "T_Max"
    if c_min: renames[c_min] = "T_Min"

    df = df.rename(columns=renames)

    # ---- volume probes (vol_heat<Thr>C / vol_cool<Thr>C, mm^3) ------------
    vol_renames = {}
    for c in df.columns:
        m = _VOL_RE.search(str(c).lower())
        if m:
            vol_renames[c] = "vol_%s_%s" % (m.group(1), m.group(2).replace(".", ""))
    if vol_renames:
        df = df.rename(columns=vol_renames)

    # Ambient temperature normalization
    if "T_amb_K" in df.columns:
        df["T_amb_C"] = df["T_amb_K"] - 273.15
    else:
        df["T_amb_C"] = default_tamb_c
        df["T_amb_K"] = default_tamb_c + 273.15

    # Secondary parallel TEC voltage column if not present
    if "voltage_tec2_V" not in df.columns:
        df["voltage_tec2_V"] = np.nan

    # Delta T calculations
    if "T_Hot" in df.columns:
        df["Delta_Hot"] = df["T_Hot"] - df["T_amb_C"]
    if "T_Cold" in df.columns:
        df["Delta_Cold"] = df["T_Cold"] - df["T_amb_C"]
    if "T_Max" in df.columns:
        df["Delta_Max"] = df["T_Max"] - df["T_amb_C"]
    if "T_Min" in df.columns:
        df["Delta_Min"] = df["T_Min"] - df["T_amb_C"]

    # Final time normalization
    if "final_time_s" not in df.columns and "time_s" in df.columns:
        df["final_time_s"] = df["time_s"].max()

    return df

# =============================================================================
# 3. SURFACE MAPS LOADER (Inner Surface)
# =============================================================================
def _find_surface_header_and_cols(file_path):
    """Sniff header line in surface map csv."""
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            s = line.lstrip("%").strip()
            if s.startswith("x;") or "x;y;z" in s:
                cols = [c.strip() for c in s.split(";")]
                return idx, cols
    return 8, ["x", "y", "z", "I0", "t", "T (C)"]


def load_surface_inner(file_path, target_currents, default_tamb_c=20.0, chunksize=500_000):
    """Stream-load only requested currents from Inner_surface CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Surface file not found: {file_path}")

    header_row, cols = _find_surface_header_and_cols(file_path)
    
    # Identify key column names
    c_i0 = next((c for c in cols if c.lower() == "i0"), cols[3])
    c_t = next((c for c in cols if c.lower() in ("t", "time", "time_s")), cols[4])
    c_temp = cols[-1]

    frames = []
    matched_currents = set()

    for chunk in pd.read_csv(file_path, sep=";", skiprows=header_row + 1,
                             header=None, names=cols, chunksize=chunksize,
                             low_memory=False):
        chunk = _to_numeric(chunk)
        # Filter by requested currents within tolerance
        mask = np.zeros(len(chunk), dtype=bool)
        for cur in target_currents:
            cur_mask = np.isclose(chunk[c_i0], cur, atol=1e-3)
            if cur_mask.any():
                matched_currents.add(cur)
                mask |= cur_mask

        if mask.any():
            sub = chunk[mask].copy()
            frames.append(sub)

    if not frames:
        raise ValueError(f"No rows matching {target_currents} found in {file_path}")

    df = pd.concat(frames, ignore_index=True)
    renames = {
        c_i0: "I0",
        c_t: "time_s",
        c_temp: "T_degC"
    }
    df = df.rename(columns=renames)
    df["T_amb_C"] = default_tamb_c
    df["Delta_T"] = df["T_degC"] - df["T_amb_C"]

    return df

# =============================================================================
# 4. 2D INTERPOLATION HELPERS
# =============================================================================
def _map_axes(df):
    spans = []
    for c in ("x", "y", "z"):
        if c in df.columns:
            lo, hi = df[c].min(), df[c].max()
            if np.isfinite(lo) and np.isfinite(hi):
                spans.append((c, float(hi - lo)))
    spans.sort(key=lambda t: -t[1])
    if len(spans) >= 2:
        return spans[0][0], spans[1][0]
    return "y", "z"


def _safe_griddata(points, values, grid):
    # NOTE: "linear" is deliberately tried first. Cubic interpolation of
    # scattered COMSOL surface exports overshoots badly (measured +43.8 degC
    # interpolated where the data maximum was +31.0 degC), which corrupts maps
    # and, even worse, maps of the difference between two models.
    for method in ("linear", "cubic", "nearest"):
        try:
            return griddata(points, values, grid, method=method)
        except Exception:
            continue
    return np.full(grid[0].shape, np.nan)


def create_surface_grid(df_sub, data_col="Delta_T", resolution=150):
    c1, c2 = _map_axes(df_sub)
    x = df_sub[c1].values
    y = df_sub[c2].values
    val = df_sub[data_col].values
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    Vi = _safe_griddata((x, y), val, (Xi, Yi))
    return c1, c2, Xi, Yi, Vi


def grid_compare_surfaces(df_a, df_b, data_col="Delta_T", resolution=150):
    c1, c2 = _map_axes(df_a)
    x = np.concatenate([df_a[c1].values, df_b[c1].values])
    y = np.concatenate([df_a[c2].values, df_b[c2].values])
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    Va = _safe_griddata((df_a[c1].values, df_a[c2].values), df_a[data_col].values, (Xi, Yi))
    Vb = _safe_griddata((df_b[c1].values, df_b[c2].values), df_b[data_col].values, (Xi, Yi))
    return c1, c2, Xi, Yi, Va, Vb

# =============================================================================
# 5. PLOTTING FUNCTIONS - DUAL TEC PROBES & ELECTRICAL
# =============================================================================
def plot_dual_tec_probes(df_dual):
    """Plot Hot & Cold probe Delta-T evolution across all drive currents."""
    currents = sorted(df_dual["I0"].unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    for i0 in currents:
        sub = df_dual[np.isclose(df_dual["I0"], i0)].sort_values("time_s")
        ax1.plot(sub["time_s"], sub["Delta_Hot"], lw=1.8, label=f"$I_0={i0:.2f}$ A")
        ax2.plot(sub["time_s"], sub["Delta_Cold"], lw=1.8, label=f"$I_0={i0:.2f}$ A")

    ax1.set_title("Dual TEC - Hot Probe Evolution $\\Delta T_{Hot}$", fontsize=11, weight="bold")
    ax2.set_title("Dual TEC - Cold Probe Evolution $\\Delta T_{Cold}$", fontsize=11, weight="bold")
    for ax in (ax1, ax2):
        ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel(r"$\Delta T$ (°C)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    ax1.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.show()


def plot_dual_tec_electrical(df_dual):
    """Plot electrical transients: Primary TEC 1 voltage, Parallel TEC 2 voltage, and Heat flux."""
    currents = sorted(df_dual["I0"].unique())
    has_tec2 = "voltage_tec2_V" in df_dual.columns and df_dual["voltage_tec2_V"].notna().any()

    ncols = 3 if has_tec2 else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.8))

    ax_v1 = axes[0]
    ax_v2 = axes[1] if has_tec2 else None
    ax_flux = axes[-1]

    for i0 in currents:
        sub = df_dual[np.isclose(df_dual["I0"], i0)].sort_values("time_s")
        ax_v1.plot(sub["time_s"], sub["voltage_tec1_V"], lw=1.8, label=f"$I_0={i0:.2f}$ A")
        if has_tec2:
            ax_v2.plot(sub["time_s"], sub["voltage_tec2_V"], lw=1.8, label=f"$I_0={i0:.2f}$ A")
        if "flux_Wm2" in sub.columns:
            ax_flux.plot(sub["time_s"], sub["flux_Wm2"], lw=1.8, label=f"$I_0={i0:.2f}$ A")

    ax_v1.set_title("Primary TEC 1 Voltage (V)", fontsize=11, weight="bold")
    ax_v1.set_ylabel("Voltage (V)", fontsize=10)

    if has_tec2:
        ax_v2.set_title("Thermally Parallel TEC 2 Voltage (V)", fontsize=11, weight="bold")
        ax_v2.set_ylabel("Voltage (V)", fontsize=10)

    ax_flux.set_title("Boundary Heat Flux ($W/m^2$)", fontsize=11, weight="bold")
    ax_flux.set_ylabel("Heat Flux ($W/m^2$)", fontsize=10)

    for ax in axes:
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    ax_v1.legend(fontsize=8, ncol=2)
    fig.suptitle("Dual TEC Electrical & Heat Flux Transients (Config Castell)", fontsize=13, weight="bold", y=1.02)
    plt.tight_layout()
    plt.show()
# =============================================================================
# 6. COMPARISON: DUAL TEC (CASTELL) VS HEATSINK MODEL
# =============================================================================
def plot_comparison_probes(df_dual, df_sink, compare_currents):
    """Compare Hot and Cold probe transients between Dual TEC and Heatsink."""
    valid_currents = []
    for cur in compare_currents:
        if any(np.isclose(df_dual["I0"], cur)) and any(np.isclose(df_sink["I0"], cur)):
            valid_currents.append(cur)

    if not valid_currents:
        print(f"[Warning] No overlapping currents found in {compare_currents}")
        return

    n_cur = len(valid_currents)
    fig, axes = plt.subplots(2, n_cur, figsize=(4.8 * n_cur + 1, 8), squeeze=False, sharex=True)

    for col_idx, cur in enumerate(valid_currents):
        sub_dual = df_dual[np.isclose(df_dual["I0"], cur)].sort_values("time_s")
        sub_sink = df_sink[np.isclose(df_sink["I0"], cur)].sort_values("time_s")

        # Row 0: Hot & Cold probe comparison
        ax0 = axes[0][col_idx]
        ax0.plot(sub_dual["time_s"], sub_dual["Delta_Hot"], "r-", lw=2, label="Dual TEC (Hot)")
        ax0.plot(sub_dual["time_s"], sub_dual["Delta_Cold"], "b-", lw=2, label="Dual TEC (Cold)")
        ax0.plot(sub_sink["time_s"], sub_sink["Delta_Hot"], "r--", lw=1.8, label="Heatsink (Hot)")
        ax0.plot(sub_sink["time_s"], sub_sink["Delta_Cold"], "b--", lw=1.8, label="Heatsink (Cold)")
        ax0.set_title(f"Drive Current $I_0 = {cur:.2f}$ A", fontsize=11, weight="bold")
        ax0.set_ylabel(r"$\Delta T$ (°C)", fontsize=10)
        ax0.grid(True, linestyle=":", alpha=0.6)
        if col_idx == 0:
            ax0.legend(fontsize=8, loc="best")

        # Row 1: Difference (Dual TEC - Heatsink)
        ax1 = axes[1][col_idx]
        t_common = np.intersect1d(sub_dual["time_s"].round(3), sub_sink["time_s"].round(3))
        dual_indexed = sub_dual.set_index(sub_dual["time_s"].round(3))
        sink_indexed = sub_sink.set_index(sub_sink["time_s"].round(3))
        
        diff_hot = dual_indexed.loc[t_common, "Delta_Hot"] - sink_indexed.loc[t_common, "Delta_Hot"]
        diff_cold = dual_indexed.loc[t_common, "Delta_Cold"] - sink_indexed.loc[t_common, "Delta_Cold"]
        
        ax1.plot(t_common, diff_hot, "r-", lw=1.8, label=r"$\Delta T_{Hot}^{Dual} - \Delta T_{Hot}^{Sink}$")
        ax1.plot(t_common, diff_cold, "b-", lw=1.8, label=r"$\Delta T_{Cold}^{Dual} - \Delta T_{Cold}^{Sink}$")
        ax1.axhline(0, color="gray", ls="--", lw=0.8)
        ax1.set_xlabel("Time (s)", fontsize=10)
        ax1.set_ylabel(r"Difference $\Delta(\Delta T)$ (°C)", fontsize=10)
        ax1.grid(True, linestyle=":", alpha=0.6)
        if col_idx == 0:
            ax1.legend(fontsize=8, loc="best")

    fig.suptitle("Comparison: Dual TEC (Config Castell) vs Heatsink Model", fontsize=13, weight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_comparison_electrical(df_dual, df_sink, compare_currents):
    """Compare primary voltage and heat flux between Dual TEC and Heatsink."""
    valid_currents = [cur for cur in compare_currents
                      if any(np.isclose(df_dual["I0"], cur)) and any(np.isclose(df_sink["I0"], cur))]
    if not valid_currents:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    for cur in valid_currents:
        sub_dual = df_dual[np.isclose(df_dual["I0"], cur)].sort_values("time_s")
        sub_sink = df_sink[np.isclose(df_sink["I0"], cur)].sort_values("time_s")

        # Voltage TEC 1
        ax1.plot(sub_dual["time_s"], sub_dual["voltage_tec1_V"], lw=2,
                 label=f"Dual TEC $I_0={cur:.2f}$A")
        ax1.plot(sub_sink["time_s"], sub_sink["voltage_tec1_V"], lw=1.8, ls="--",
                 label=f"Heatsink $I_0={cur:.2f}$A")

        # Heat flux
        if "flux_Wm2" in sub_dual.columns and "flux_Wm2" in sub_sink.columns:
            ax2.plot(sub_dual["time_s"], sub_dual["flux_Wm2"], lw=2,
                     label=f"Dual TEC $I_0={cur:.2f}$A")
            ax2.plot(sub_sink["time_s"], sub_sink["flux_Wm2"], lw=1.8, ls="--",
                     label=f"Heatsink $I_0={cur:.2f}$A")

    ax1.set_title("Primary TEC 1 Terminal Voltage Comparison", fontsize=11, weight="bold")
    ax1.set_ylabel("Voltage (V)", fontsize=10)
    ax2.set_title("Boundary Heat Flux Comparison", fontsize=11, weight="bold")
    ax2.set_ylabel("Heat Flux (W/m$^2$)", fontsize=10)

    for ax in (ax1, ax2):
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    ax1.legend(fontsize=8, ncol=2)
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(fontsize=8, ncol=2)

    fig.suptitle("Comparison: Primary Voltage & Heat Flux (Dual TEC vs Heatsink)",
                 fontsize=13, weight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 7. INNER SURFACE MAPS & SPATIAL COMPARISON
# =============================================================================
def plot_surface_temporal_inner(df_surf, currents, time_steps, resolution=150):
    """Plot 2D temperature maps for the Inner Surface only across time steps."""
    for cur in currents:
        sub_cur = df_surf[np.isclose(df_surf["I0"], cur)]
        if sub_cur.empty:
            continue

        fig, axes = plt.subplots(1, len(time_steps), figsize=(3.8 * len(time_steps), 4), sharey=True)
        if len(time_steps) == 1:
            axes = [axes]

        vmax = max(abs(sub_cur["Delta_T"].min()), abs(sub_cur["Delta_T"].max()), 1.0)
        last_im = None

        for ax, t_step in zip(axes, time_steps):
            sub_t = sub_cur[np.isclose(sub_cur["time_s"], t_step, atol=0.05)]
            if sub_t.empty:
                ax.set_title(f"$t={t_step:.0f}$ s\n(No data)", fontsize=9)
                continue

            c1, c2, Xi, Yi, Ti = create_surface_grid(sub_t, "Delta_T", resolution)
            last_im = ax.imshow(Ti, extent=[Xi.min(), Xi.max(), Yi.min(), Yi.max()],
                                origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(f"$t={t_step:.0f}$ s", fontsize=10)
            ax.set_xlabel(f"{c1} (mm)", fontsize=9)
            if ax is axes[0]:
                ax.set_ylabel(f"{c2} (mm)", fontsize=9)
            ax.grid(False)

        if last_im is not None:
            cax = inset_axes(axes[-1], width="8%", height="100%", loc="center left",
                             bbox_to_anchor=(1.08, 0, 1, 1),
                             bbox_transform=axes[-1].transAxes, borderpad=0)
            fig.colorbar(last_im, cax=cax).set_label(r"Inner Surface $\Delta T$ (°C)", fontsize=10)

        fig.suptitle(f"Inner Surface Thermal Evolution (Dual TEC - Castell) - $I_0 = {cur:.2f}$ A",
                     fontsize=12, weight="bold", y=1.02)
        plt.tight_layout()
        plt.show()


def plot_surface_comparison_inner(df_dual_surf, df_sink_surf, current, time_s, resolution=150):
    """Spatial 2D comparison between Dual TEC and Heatsink on Inner Surface at a fixed (I0, t)."""
    sub_d = df_dual_surf[np.isclose(df_dual_surf["I0"], current) & np.isclose(df_dual_surf["time_s"], time_s, atol=0.25)]
    sub_s = df_sink_surf[np.isclose(df_sink_surf["I0"], current) & np.isclose(df_sink_surf["time_s"], time_s, atol=0.25)]

    if sub_d.empty or sub_s.empty:
        print(f"[Skip] Missing surface comparison data for I0={current}A, t={time_s}s")
        return

    c1, c2, Xi, Yi, T_dual, T_sink = grid_compare_surfaces(sub_d, sub_s, "Delta_T", resolution)
    T_diff = T_dual - T_sink

    v_abs_max = max(np.nanmax(np.abs(T_dual)), np.nanmax(np.abs(T_sink)), 1.0)
    diff_max = np.nanmax(np.abs(T_diff)) if np.isfinite(T_diff).any() else 1.0

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    im1 = ax1.imshow(T_dual, extent=[Xi.min(), Xi.max(), Yi.min(), Yi.max()],
                     origin="lower", cmap="coolwarm", vmin=-v_abs_max, vmax=v_abs_max)
    ax1.set_title("Dual TEC (Castell)", fontsize=11, weight="bold")

    im2 = ax2.imshow(T_sink, extent=[Xi.min(), Xi.max(), Yi.min(), Yi.max()],
                     origin="lower", cmap="coolwarm", vmin=-v_abs_max, vmax=v_abs_max)
    ax2.set_title("Heatsink Model", fontsize=11, weight="bold")

    im3 = ax3.imshow(T_diff, extent=[Xi.min(), Xi.max(), Yi.min(), Yi.max()],
                     origin="lower", cmap="seismic", vmin=-diff_max, vmax=diff_max)
    ax3.set_title("Difference (Dual TEC - Heatsink)", fontsize=11, weight="bold")

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel(f"{c1} (mm)", fontsize=9)
    ax1.set_ylabel(f"{c2} (mm)", fontsize=9)

# =============================================================================
# 8. MAIN EXECUTION PIPELINE
# =============================================================================
def main():
    print("=" * 75)
    print("COMSOL DUAL TEC & HEATSINK COMPARISON PIPELINE")
    print("=" * 75)

    # 1. Load Dual TEC probes
    print("\n[1/5] Loading Dual TEC Probes (Config_Castell.csv)...")
    df_dual_probes = load_accumulated_probes(CONFIG["DUAL_TEC_PROBES_CSV"],
                                             default_tamb_c=CONFIG["T_AMB_DEFAULT_C"])
    dual_currents = sorted(df_dual_probes["I0"].unique())
    print(f"      Loaded {len(df_dual_probes)} rows, currents: {dual_currents}")
    if "voltage_tec2_V" in df_dual_probes.columns and df_dual_probes["voltage_tec2_V"].notna().any():
        print("      Detected secondary parallel TEC voltage (Probe 3).")

    # 2. Load Heatsink reference probes
    df_sink_probes = None
    if CONFIG["RUN_HEATSINK_COMPARISON"]:
        print("\n[2/5] Loading Heatsink Probes for comparison...")
        df_sink_probes = load_accumulated_probes(CONFIG["HEATSINK_PROBES_CSV"],
                                                 default_tamb_c=CONFIG["T_AMB_DEFAULT_C"])
        sink_currents = sorted(df_sink_probes["I0"].unique())
        print(f"      Loaded {len(df_sink_probes)} rows, currents: {sink_currents}")

    # 3. Plot Dual TEC Probes & Electrical
    if CONFIG["RUN_PROBES_ANALYSIS"]:
        print("\n[3/5] Generating Dual TEC probe & electrical transient plots...")
        plot_dual_tec_probes(df_dual_probes)
        if CONFIG["RUN_ELECTRICAL_ANALYSIS"]:
            plot_dual_tec_electrical(df_dual_probes)

    # 4. Compare with Heatsink Probes
    if CONFIG["RUN_HEATSINK_COMPARISON"] and df_sink_probes is not None:
        print("\n[4/5] Generating Dual TEC vs Heatsink probe comparison plots...")
        plot_comparison_probes(df_dual_probes, df_sink_probes, CONFIG["COMPARE_CURRENTS"])
        plot_comparison_electrical(df_dual_probes, df_sink_probes, CONFIG["COMPARE_CURRENTS"])

    # 5. Inner Surface Map processing (Dual TEC and comparison)
    if CONFIG["RUN_SURFACE_MAPS"]:
        print("\n[5/5] Streaming Inner Surface Map (Config_Castell_surf.csv)...")
        df_dual_surf = load_surface_inner(CONFIG["DUAL_TEC_INNER_SURF_CSV"],
                                          target_currents=CONFIG["SURFACE_CURRENTS"],
                                          default_tamb_c=CONFIG["T_AMB_DEFAULT_C"],
                                          chunksize=CONFIG["SURFACE_CHUNK"])
        print(f"      Loaded {len(df_dual_surf):,} points for currents: {sorted(df_dual_surf['I0'].unique())}")
        
        plot_surface_temporal_inner(df_dual_surf,
                                   currents=CONFIG["SURFACE_CURRENTS"],
                                   time_steps=CONFIG["SURFACE_TIMES_S"],
                                   resolution=CONFIG["SURFACE_RESOLUTION"])

        if CONFIG["RUN_SURFACE_COMPARISON"] and os.path.exists(CONFIG["HEATSINK_INNER_SURF_CSV"]):
            print("      Loading Heatsink Inner Surface Map for 2D comparison...")
            sink_compare_cur = [c for c in CONFIG["COMPARE_CURRENTS"] if c in CONFIG["SURFACE_CURRENTS"]]
            if not sink_compare_cur:
                sink_compare_cur = [0.1]
            df_sink_surf = load_surface_inner(CONFIG["HEATSINK_INNER_SURF_CSV"],
                                              target_currents=sink_compare_cur,
                                              default_tamb_c=CONFIG["T_AMB_DEFAULT_C"],
                                              chunksize=CONFIG["SURFACE_CHUNK"])
            for cur in sink_compare_cur:
                plot_surface_comparison_inner(df_dual_surf, df_sink_surf,
                                              current=cur, time_s=25.0,
                                              resolution=CONFIG["SURFACE_RESOLUTION"])

    print("\nPipeline execution complete.")


# =============================================================================
# 9. PER-CURRENT COMPARISON OF THE TWO CONFIGURATIONS
# -----------------------------------------------------------------------------
# Four metric families are computed for every drive current, for both models:
#
#   TEMPERATURE  which cools more / which heats more and by which factor
#   SPATIAL      cooled & heated VOLUME (domain probes, mm^3) and cooled &
#                heated inner-surface AREA (surface meshes, mm^2) + ratios
#   TEMPORAL     how long it takes to cool, to come back to normal, and the
#                switch-off overshoot that appears at the larger currents
#   ENERGY       full-run electrical energy and - above all - the SCALE-FREE
#                efficiencies (K/J, mm^3/J, mm^2/J), which stay meaningful even
#                though the two exports ramp the current down differently
# =============================================================================
from scipy.spatial import Delaunay

LABEL_DUAL = "Dual-TEC (Castell)"
LABEL_SINK = "Heatsink"
COLOR_DUAL = "#1f77b4"
COLOR_SINK = "#d62728"


def _sub_current(df, i0):
    """All samples of one drive current, sorted by time (None when absent)."""
    if df is None:
        return None
    sub = df[np.isclose(df["I0"], i0)].sort_values("time_s")
    return sub.reset_index(drop=True) if not sub.empty else None


def nearest_row(df, t_ref):
    """Index of the sample closest to t_ref (the 0.15 s grid rarely hits it)."""
    return int(np.argmin(np.abs(df["time_s"].to_numpy(float) - t_ref)))


def vol_col(df, kind, thr):
    """Tidy name of a volume-probe column ('heat'/'cool' + threshold)."""
    key = "vol_%s_%s" % (kind, ("%g" % thr).replace(".", ""))
    return key if key in df.columns else None


def has_volume_probes(df):
    """True when the export carries vol_heat*/vol_cool* domain probes."""
    if df is None:
        return False
    return any(vol_col(df, k, t) for k in ("heat", "cool") for t in VOL_THRESHOLDS)


def _trapz(y, x):
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def _ratio(a, b):
    """a / b guarding against zero or non-finite inputs (NaN when undefined).

    Scalars are accepted as well as single-element pandas objects, because the
    merge helpers pass table cells (`Series`) straight through.
    """
    try:
        a = float(np.asarray(a, dtype=float).reshape(-1)[0])
        b = float(np.asarray(b, dtype=float).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return np.nan
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0.0:
        return np.nan
    return a / b


def _first_cross(t, cond):
    """First time where 'cond' holds (NaN when it never does)."""
    idx = np.where(cond)[0]
    return float(t[idx[0]]) if idx.size else np.nan


def _delta_pct(a, b):
    """Relative difference of a with respect to b, in percent."""
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0.0:
        return np.nan
    return (a - b) / b * 100.0


def knee_time(sub):
    """Diagnostic: time at which the drive ramp-down ends (|V1| slope collapses).

    The two exports currently disagree here (Castell ramps down in ~1 s while
    the heatsink needs ~6 s at 1 A), which is precisely why the energy family
    below relies on scale-free ratios instead of absolute joules.
    """
    t = sub["time_s"].to_numpy(float)
    v1 = np.abs(sub["voltage_tec1_V"].to_numpy(float))
    if t.size < 5:
        return np.nan
    dv = np.gradient(v1, t)
    k = int(np.argmin(dv))
    slope = abs(float(dv[k]))
    if slope <= 0:
        return np.nan
    after = np.where((t > t[k]) & (np.abs(dv) < 0.05 * slope))[0]
    return float(t[after[0]]) if after.size else float(t[k])


def compare_current_metrics(df, i0, label, t_on=T_ON_S, t_end=T_END_S,
                            t_final=T_FINAL_S, thresholds=VOL_THRESHOLDS):
    """One tidy row of per-current comparison metrics for one model."""
    sub = _sub_current(df, i0)
    if sub is None:
        return None

    t = sub["time_s"].to_numpy(float)
    v1 = np.abs(sub["voltage_tec1_V"].to_numpy(float))
    d_cold = sub["Delta_Cold"].to_numpy(float)
    d_hot = sub["Delta_Hot"].to_numpy(float)
    q = sub["flux_Wm2"].to_numpy(float) if "flux_Wm2" in sub.columns else np.full(t.shape, np.nan)

    i_end = nearest_row(sub, t_end)
    i_fin = nearest_row(sub, t_final)
    drive = (t >= t_on) & (t <= t_end)
    post = t > t_end

    d_cold_pk = float(np.nanmin(d_cold[drive]))      # most negative = coldest
    d_hot_pk = float(np.nanmax(d_hot[drive]))
    t_peak_cool = float(t[drive][int(np.argmin(d_cold[drive]))])

    # ---- temporal: cooling speed, return to normal, switch-off overshoot ---
    t_cool63 = _first_cross(t, drive & (np.abs(d_cold) >= 0.63 * abs(d_cold_pk)))
    t_cool90 = _first_cross(t, drive & (np.abs(d_cold) >= 0.90 * abs(d_cold_pk)))
    t_cross = _first_cross(t, post & (d_cold > 0.0))
    t_norm = _first_cross(t, post & (np.abs(d_cold) < 0.2))
    if post.any():
        k_ovs = int(np.argmax(d_cold[post]))
        overshoot_cold = float(d_cold[post][k_ovs])
        overshoot_time = float(t[post][k_ovs])
        overshoot_hot = float(np.nanmax(d_hot[post]) - d_hot[i_end])
    else:
        overshoot_cold = overshoot_time = overshoot_hot = np.nan

    # ---- energy: full run (as agreed) + scale-free efficiencies -----------
    W_full = float(i0) * _trapz(v1, t)
    W_drive = float(i0) * _trapz(v1[drive], t[drive])
    E_areal = _trapz(np.abs(q[drive]), t[drive]) if np.isfinite(q).any() else np.nan

    row = {
        "model": label,
        "I0_A": float(i0),
        "knee_s": knee_time(sub),
        "dT_cold_peak_K": d_cold_pk,
        "dT_hot_peak_K": d_hot_pk,
        "dT_cold_end_K": float(d_cold[i_end]),
        "dT_hot_end_K": float(d_hot[i_end]),
        "ratio_cool_heat": _ratio(abs(d_cold_pk), d_hot_pk),
        "t_peak_cool_s": t_peak_cool,
        "t_cool63_s": t_cool63 - t_on if np.isfinite(t_cool63) else np.nan,
        "t_cool90_s": t_cool90 - t_on if np.isfinite(t_cool90) else np.nan,
        "t_ambient_cross_s": t_cross - t_end if np.isfinite(t_cross) else np.nan,
        "t_back_normal_s": t_norm - t_end if np.isfinite(t_norm) else np.nan,
        "overshoot_cold_K": overshoot_cold,
        "overshoot_time_s": overshoot_time,
        "overshoot_hot_K": overshoot_hot,
        "W_full_J": W_full,
        "W_drive_J": W_drive,
        "E_areal_J_m2": E_areal,
        "K_per_J": _ratio(abs(d_cold_pk), W_full),
        "K_per_J_hot": _ratio(d_hot_pk, W_full),
    }

    # ---- spatial: cooled / heated volumes at the reference times ----------
    for thr in thresholds:
        tag = ("%g" % thr).replace(".", "")
        c_cool, c_heat = vol_col(sub, "cool", thr), vol_col(sub, "heat", thr)
        if c_cool is None or c_heat is None:
            continue
        vc = sub[c_cool].to_numpy(float)
        vh = sub[c_heat].to_numpy(float)
        row["vol_cool_%s_end" % tag] = float(vc[i_end])
        row["vol_heat_%s_end" % tag] = float(vh[i_end])
        row["vol_cool_%s_60" % tag] = float(vc[i_fin])
        row["vol_heat_%s_60" % tag] = float(vh[i_fin])
        row["vol_cool_%s_peak" % tag] = float(vc.max())
        row["vol_heat_%s_peak" % tag] = float(vh.max())
        row["vol_ratio_%s_end" % tag] = _ratio(vc[i_end], vh[i_end])
        row["vol_cool_%s_per_J" % tag] = _ratio(vc[i_end], W_full)
        row["vol_heat_growth_%s" % tag] = _ratio(vh[i_fin], vh[i_end])
    return row


def build_comparison_table(df_dual, df_sink, currents=None, thresholds=VOL_THRESHOLDS):
    """Per-current comparison table for both configurations (tidy DataFrame)."""
    if currents is None:
        currents = sorted(set(np.round(df_dual["I0"].unique(), 4)))
    rows = []
    for i0 in currents:
        for df, label in ((df_dual, LABEL_DUAL), (df_sink, LABEL_SINK)):
            r = compare_current_metrics(df, i0, label, thresholds=thresholds)
            if r:
                rows.append(r)
    return pd.DataFrame(rows)


def add_surface_area_metrics(table, area_series, label, thresholds=VOL_THRESHOLDS,
                             t_end=T_END_S, t_final=T_FINAL_S):
    """Merge the 2D inner-surface area metrics into the comparison table."""
    if not area_series:
        return table
    for i0, series in area_series.items():
        mask = (table["model"] == label) & np.isclose(table["I0_A"], i0)
        if not mask.any():
            continue
        end = surface_areas_at(series, t_end, thresholds)
        fin = surface_areas_at(series, t_final, thresholds)
        peaks = surface_area_peaks(series, thresholds)
        for thr in thresholds:
            tag = ("%g" % thr).replace(".", "")
            idx = table.index[mask]
            table.loc[idx, "area_cool_%s_end" % tag] = end[thr][0]
            table.loc[idx, "area_heat_%s_end" % tag] = end[thr][1]
            table.loc[idx, "area_cool_%s_60" % tag] = fin[thr][0]
            table.loc[idx, "area_heat_%s_60" % tag] = fin[thr][1]
            table.loc[idx, "area_cool_%s_peak" % tag] = peaks[thr]["cool_peak"]
            table.loc[idx, "area_heat_%s_peak" % tag] = peaks[thr]["heat_peak"]
            table.loc[idx, "area_ratio_%s_end" % tag] = _ratio(end[thr][0], end[thr][1])
            table.loc[idx, "area_cool_%s_per_J" % tag] = _ratio(
                end[thr][0], table.loc[idx, "W_full_J"])
    return table


# -----------------------------------------------------------------------------
# 2D inner-surface AREA metrics (mm^2) - the "2D from the surface" companion of
# the volumetric domain probes above
# -----------------------------------------------------------------------------
_AREA_CACHE = {}


def _area_cache_dir():
    """Directory holding the cached Delaunay area series (one file per series)."""
    path = os.path.join(os.path.dirname(os.path.abspath(CONFIG["COMPARISON_CSV"])),
                        "_surface_area_cache")
    os.makedirs(path, exist_ok=True)
    return path


def _area_cache_path(file_path, i0):
    """Cache file for one (export, current); mtime/size invalidate it on re-export."""
    st = os.stat(file_path)
    base = os.path.splitext(os.path.basename(file_path))[0]
    tag = "%s_I%s_%d_%d" % (base, ("%g" % i0).replace(".", "p"),
                            int(st.st_mtime), int(st.st_size))
    return os.path.join(_area_cache_dir(), tag + ".npz")


def save_area_series(series):
    """Persist a computed area series as float32 (tri_T is 121 x ~15k)."""
    if not series or series.get("tri_T") is None:
        return None
    path = _area_cache_path(series["path"], series["I0"])
    np.savez_compressed(path, times=series["times"], tri_area=series["tri_area"],
                        tri_T=series["tri_T"].astype(np.float32),
                        hull=np.array([series["hull_mm2"]]),
                        axes=np.array(series["axes"], dtype="<U8"))
    return path


def load_area_series(file_path, i0):
    """Read a cached area series (None when absent or disabled)."""
    if not CONFIG.get("USE_AREA_CACHE", True) or not os.path.exists(file_path):
        return None
    path = _area_cache_path(file_path, i0)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as z:
            return {"path": file_path, "I0": float(i0),
                    "axes": tuple(str(a) for a in z["axes"]),
                    "times": z["times"], "tri_area": z["tri_area"],
                    "tri_T": z["tri_T"].astype(float), "hull_mm2": float(z["hull"][0])}
    except Exception as exc:
        print("      [warn] unreadable area cache %s (%s)"
              % (os.path.basename(path), exc))
        return None


def compute_area_series(file_path, i0, df, use_cache=True):
    """Area series of ONE current from an already loaded frame.

    Only one triangulation is needed per current because the exported mesh nodes
    keep their order at every time step (verified: the Castell hull is exactly
    900.0 mm^2 = 30 x 30 mm with 7795 nodes per step).
    """
    a1, a2 = _map_axes(df)
    t_all = df["time_s"].to_numpy(float)
    order = np.argsort(t_all, kind="stable")
    dT = df["Delta_T"].to_numpy(float)[order]
    c1 = df[a1].to_numpy(float)[order]
    c2 = df[a2].to_numpy(float)[order]
    times, starts = np.unique(t_all[order], return_index=True)
    bounds = np.append(starts, t_all.size)

    pts = np.column_stack([c1[bounds[0]:bounds[1]], c2[bounds[0]:bounds[1]]])
    simpl = Delaunay(pts).simplices
    p0, p1, p2 = pts[simpl[:, 0]], pts[simpl[:, 1]], pts[simpl[:, 2]]
    tri_area = 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) -
                            (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))

    tri_T = np.full((times.size, tri_area.size), np.nan)
    for k in range(times.size):
        chunk = dT[bounds[k]:bounds[k + 1]]
        if chunk.size != pts.shape[0]:
            print("      [warn] mesh changed at t = %.2f s - current skipped" % times[k])
            return None
        tri_T[k] = chunk[simpl].mean(axis=1)

    series = {"path": file_path, "I0": float(i0), "axes": (a1, a2), "times": times,
              "tri_area": tri_area, "tri_T": tri_T, "hull_mm2": float(tri_area.sum())}
    if use_cache:
        save_area_series(series)
    return series


def surface_area_series(file_path, i0, t_amb_c=20.0, chunk=None):
    """Area series of one current: memory cache -> disk cache -> one streamed pass."""
    key = (file_path, round(float(i0), 4))
    if key in _AREA_CACHE:
        return _AREA_CACHE[key]
    if not os.path.exists(file_path):
        print("      [skip] surface file not found: %s" % file_path)
        _AREA_CACHE[key] = None
        return None

    series = load_area_series(file_path, i0)
    if series is None:
        print("      areas: streaming %s @ I0=%.2f A ..."
              % (os.path.basename(file_path), i0))
        df = load_surface_inner(file_path, target_currents=[i0],
                               default_tamb_c=t_amb_c,
                               chunksize=chunk or CONFIG["SURFACE_CHUNK"])
        series = compute_area_series(file_path, i0, df)
    _AREA_CACHE[key] = series
    return series


def surface_areas_at(series, t_ref, thresholds=VOL_THRESHOLDS):
    """{threshold: (cooled_area, heated_area)} in mm^2 at the closest time step."""
    out = {}
    if series is None or series.get("tri_T") is None:
        return out
    k = int(np.argmin(np.abs(series["times"] - t_ref)))
    row = series["tri_T"][k]
    area = series["tri_area"]
    for thr in thresholds:
        out[thr] = (float(area[row < -thr].sum()), float(area[row > thr].sum()))
    return out


def surface_area_peaks(series, thresholds=VOL_THRESHOLDS):
    """Peak cooled/heated area and the time it occurs, per threshold."""
    out = {}
    if series is None or series.get("tri_T") is None:
        return out
    area = series["tri_area"]
    times = series["times"]
    for thr in thresholds:
        cool = np.array([area[row < -thr].sum() for row in series["tri_T"]])
        heat = np.array([area[row > thr].sum() for row in series["tri_T"]])
        out[thr] = {"cool_peak": float(cool.max()) if cool.size else np.nan,
                    "cool_peak_t": float(times[int(cool.argmax())]) if cool.size else np.nan,
                    "heat_peak": float(heat.max()) if heat.size else np.nan,
                    "heat_peak_t": float(times[int(heat.argmax())]) if heat.size else np.nan}
    return out


def collect_area_series(file_path, currents, t_amb_c=20.0):
    """{current: area series} for the requested currents.

    Reading the 429 MB Castell export costs ~20 s per pass, so the currents that
    are not cached yet are streamed TOGETHER in a single pass, and every result
    is written to the small .npz cache so that later runs are instant.
    """
    if not os.path.exists(file_path):
        print("      [skip] surface file not found: %s" % file_path)
        return {}
    out, missing = {}, []
    for i0 in currents:
        key = (file_path, round(float(i0), 4))
        series = _AREA_CACHE.get(key)
        if series is None:
            series = load_area_series(file_path, i0)
        if series is not None:
            _AREA_CACHE[key] = series
            out[float(i0)] = series
        else:
            missing.append(i0)

    if missing:
        print("      areas: streaming %s for currents %s ..."
              % (os.path.basename(file_path), missing))
        try:
            df = load_surface_inner(file_path, target_currents=missing,
                                    default_tamb_c=t_amb_c,
                                    chunksize=CONFIG["SURFACE_CHUNK"])
        except Exception as exc:                      # keep the pipeline alive
            print("      [warn] could not load %s (%s)" % (file_path, exc))
            return out
        for i0 in missing:
            sub = df[np.isclose(df["I0"], i0)]
            if sub.empty:
                print("      [warn] no rows for I0=%.2f A in %s"
                      % (i0, os.path.basename(file_path)))
                continue
            try:
                series = compute_area_series(file_path, i0, sub)
            except Exception as exc:
                print("      [warn] area computation failed for I0=%.2f A: %s" % (i0, exc))
                series = None
            _AREA_CACHE[(file_path, round(float(i0), 4))] = series
            if series is not None:
                out[float(i0)] = series
    return out


def row_of(table, model, i0):
    """Metric row of one model/current as a dict ({} when missing)."""
    sel = table[(table["model"] == model) & np.isclose(table["I0_A"], i0)]
    return sel.iloc[0].to_dict() if len(sel) else {}


def _g(row, key):
    """Metric value as float (NaN when the column is missing)."""
    v = row.get(key, np.nan)
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _verdict(a, b, magnitude=True, eps=1e-9):
    """'X% more/less' phrasing of a against b (magnitudes by default).

    A zero baseline makes the percentage meaningless (the heatsink has no
    heated volume at all at 0.1 A), so it is reported explicitly instead.
    """
    if magnitude:
        a, b = abs(a), abs(b)
    if abs(b) <= eps:
        return "vs ~0 baseline" if abs(a) > eps else "both ~0"
    pct = _delta_pct(a, b)
    if not np.isfinite(pct):
        return "n/a"
    return "%.0f%% %s" % (abs(pct), "more" if pct > 0 else "less")


def _fmt_pair(ra, rb, key, unit="", prec=2, nan_text="n/a"):
    """'Castell x  Heatsink y' string; NaN renders as nan_text without a unit."""

    def fmt(row):
        v = _g(row, key)
        if not np.isfinite(v):
            return "%7s" % nan_text, ""
        return "%*.*f" % (7, prec, v), unit

    ta, ua = fmt(ra)
    tb, ub = fmt(rb)
    return "Castell %s %s  Heatsink %s %s" % (ta, ua, tb, ub)


def print_comparison_summary(table, currents=None, thresholds=VOL_THRESHOLDS):
    """Compact per-current verdict of the four metric families."""
    if currents is None:
        currents = sorted(table["I0_A"].unique())
    t_mid = thresholds[len(thresholds) // 2]
    tag_mid = ("%g" % t_mid).replace(".", "")
    print("\n" + "=" * 118)
    print("PER-CURRENT VERDICT (reference times: drive end = %.0f s, end of run = %.0f s)"
          % (T_END_S, T_FINAL_S))
    print("=" * 118)
    for i0 in currents:
        ra, rb = row_of(table, LABEL_DUAL, i0), row_of(table, LABEL_SINK, i0)
        if not ra or not rb:
            continue
        print("-" * 118)
        print("I0 = %.2f A" % i0)
        print("  TEMPERATURE  cooling |dTc| peak   %s   -> Castell cools %s"
              % (_fmt_pair(ra, rb, "dT_cold_peak_K", "K"), _verdict(_g(ra, "dT_cold_peak_K"), _g(rb, "dT_cold_peak_K"))))
        print("               heating  dTh peak   %s   -> Castell heats %s"
              % (_fmt_pair(ra, rb, "dT_hot_peak_K", "K"), _verdict(_g(ra, "dT_hot_peak_K"), _g(rb, "dT_hot_peak_K"))))
        print("               ratio |dTc|/dTh      %s"
              % _fmt_pair(ra, rb, "ratio_cool_heat", " ", 3))
        print("  SPATIAL      cooled vol @%.1fK     %s   -> Castell cools %s volume"
              % (thresholds[0],
                 _fmt_pair(ra, rb, "vol_cool_%s_end" % ("%g" % thresholds[0]).replace(".", ""), "mm3", 1),
                 _verdict(_g(ra, "vol_cool_%s_end" % ("%g" % thresholds[0]).replace(".", "")),
                          _g(rb, "vol_cool_%s_end" % ("%g" % thresholds[0]).replace(".", "")))))
        print("               heated vol @%.1fK     %s   -> Castell heats %s volume"
              % (thresholds[0],
                 _fmt_pair(ra, rb, "vol_heat_%s_end" % ("%g" % thresholds[0]).replace(".", ""), "mm3", 1),
                 _verdict(_g(ra, "vol_heat_%s_end" % ("%g" % thresholds[0]).replace(".", "")),
                          _g(rb, "vol_heat_%s_end" % ("%g" % thresholds[0]).replace(".", "")))))
        print("               heated vol @%gK       %s   -> Castell heats %s hot volume"
              % (t_mid,
                 _fmt_pair(ra, rb, "vol_heat_%s_end" % tag_mid, "mm3", 1),
                 _verdict(_g(ra, "vol_heat_%s_end" % tag_mid), _g(rb, "vol_heat_%s_end" % tag_mid))))
        if "area_cool_05_end" in table.columns:
            print("               2D cooled area @%.1fK %s   -> Castell cools %s area"
                  % (thresholds[0],
                     _fmt_pair(ra, rb, "area_cool_%s_end" % ("%g" % thresholds[0]).replace(".", ""), "mm2", 1),
                     _verdict(_g(ra, "area_cool_%s_end" % ("%g" % thresholds[0]).replace(".", "")),
                              _g(rb, "area_cool_%s_end" % ("%g" % thresholds[0]).replace(".", "")))))
            print("               2D heated area @%.1fK %s   -> Castell heats %s area"
                  % (thresholds[0],
                     _fmt_pair(ra, rb, "area_heat_%s_end" % ("%g" % thresholds[0]).replace(".", ""), "mm2", 1),
                     _verdict(_g(ra, "area_heat_%s_end" % ("%g" % thresholds[0]).replace(".", "")),
                              _g(rb, "area_heat_%s_end" % ("%g" % thresholds[0]).replace(".", "")))))
        print("  TEMPORAL     time to 63%% cooling    %s   (from drive start)"
              % _fmt_pair(ra, rb, "t_cool63_s", "s", 1))
        print("               back to normal        %s   (after switch-off, 'never' = stays away)"
              % _fmt_pair(ra, rb, "t_back_normal_s", "s", 1, nan_text="never"))
        print("               switch-off overshoot  %s   at t = %.1f s / %.1f s"
              % (_fmt_pair(ra, rb, "overshoot_cold_K", "K", 2),
                 _g(ra, "overshoot_time_s"), _g(rb, "overshoot_time_s")))
        print("  ENERGY       W[0-%.0fs]             %s   (not cross-model fair: ramps differ)"
              % (T_FINAL_S, _fmt_pair(ra, rb, "W_full_J", "J", 2)))
        print("               cooling per joule     %s   Castell is %s efficient"
              % (_fmt_pair(ra, rb, "K_per_J", "K/J", 3),
                 _verdict(_g(ra, "K_per_J"), _g(rb, "K_per_J"))))
        print("               cooled vol per joule  %s   Castell is %s efficient"
              % (_fmt_pair(ra, rb, "vol_cool_%s_per_J" % ("%g" % thresholds[0]).replace(".", ""), "mm3/J", 2),
                 _verdict(_g(ra, "vol_cool_%s_per_J" % ("%g" % thresholds[0]).replace(".", "")),
                          _g(rb, "vol_cool_%s_per_J" % ("%g" % thresholds[0]).replace(".", "")))))
    print("=" * 118)


# -----------------------------------------------------------------------------
# Figures: one per metric family (simple, grouped per drive current)
# -----------------------------------------------------------------------------
from matplotlib.lines import Line2D

THR_COLORS = {0.5: "#1b7837", 1.0: "#4dac26", 2.0: "#f1a340", 5.0: "#d01c8b"}


def _fig_dir():
    """Absolute output directory of the comparison figures (created if needed)."""
    path = os.path.abspath(CONFIG["COMPARISON_FIG_DIR"])
    os.makedirs(path, exist_ok=True)
    return path


def _save_figure(fig, name, index=None):
    """Save a figure into CONFIG['COMPARISON_FIG_DIR'] (optionally show it)."""
    if CONFIG.get("SAVE_FIGURES", True):
        fig.savefig(os.path.join(_fig_dir(), name),
                    dpi=CONFIG.get("FIGURE_DPI", 200), bbox_inches="tight")
        print(f"      saved {name}")
        if index is not None:
            index.append(name)
    if CONFIG.get("SHOW_FIGURES", False):
        plt.show()
    plt.close(fig)


def _series(table, model, currents, col):
    """Column of the comparison table as a float array over the currents."""
    return np.array([_g(row_of(table, model, c), col) for c in currents], dtype=float)


def _model_style_legend(ax, **kw):
    handles = [Line2D([], [], color=COLOR_DUAL, lw=2.0, marker="o", ms=4, label=LABEL_DUAL),
               Line2D([], [], color=COLOR_SINK, lw=2.0, ls="--", marker="s", ms=4,
                      mfc="white", label=LABEL_SINK)]
    ax.legend(handles=handles, **kw)


def fig_temperature_summary(table, currents, fname="temperature_summary.png"):
    """Temperature family: who cools more, who heats more, and the ratios."""
    if not currents:
        return
    x = np.arange(len(currents))
    w = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7))

    panels = [("dT_cold_peak_K", r"(a) Cooling: cold-probe peak $|\Delta T_c|$"),
              ("dT_hot_peak_K", r"(b) Heating: hot-probe peak $\Delta T_h$")]
    for ax, (col, ttl) in zip(axes[:2], panels):
        dv = np.abs(_series(table, LABEL_DUAL, currents, col))
        sv = np.abs(_series(table, LABEL_SINK, currents, col))
        ax.bar(x - w / 2, np.nan_to_num(dv), w, color=COLOR_DUAL, edgecolor="k", lw=0.5,
               label=LABEL_DUAL)
        ax.bar(x + w / 2, np.nan_to_num(sv), w, color=COLOR_SINK, edgecolor="k", lw=0.5,
               label=LABEL_SINK)
        for xi, a, b in zip(x, dv, sv):
            if np.isfinite(a) and np.isfinite(b):
                ax.annotate(_verdict(a, b), (xi, max(a, b) * 1.03), ha="center",
                            fontsize=7, color="0.25")
        ax.set_title(ttl, fontsize=10.5)
        ax.set_ylabel("K")
        ax.set_xticks(x)
        ax.set_xticklabels(["%.1f" % c for c in currents], fontsize=8)
        ax.set_xlabel(r"Drive current $I_0$ (A)")
        ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(currents, _series(table, LABEL_DUAL, currents, "ratio_cool_heat"), "-o",
            color=COLOR_DUAL, lw=2.0, ms=5, label=LABEL_DUAL)
    ax.plot(currents, _series(table, LABEL_SINK, currents, "ratio_cool_heat"), "--s",
            color=COLOR_SINK, lw=2.0, ms=5, mfc="white", label=LABEL_SINK)
    ax.axhline(1.0, color="0.4", lw=1.0, ls=":")
    ax.annotate("break-even (cooling = heating)", xy=(0.02, 0.76), xycoords="axes fraction",
                fontsize=8, color="0.3")
    ax.set_title(r"(c) Balance: $|\Delta T_c|/\Delta T_h$", fontsize=10.5)
    ax.set_xlabel(r"Drive current $I_0$ (A)")
    ax.set_ylabel("-")
    ax.legend(fontsize=8)

    fig.suptitle("Temperature metrics per current - cooling, heating and their ratio",
                 fontsize=12.5, weight="bold", y=1.02)
    fig.tight_layout()
    _save_figure(fig, fname)


def _thr_family_lines(ax, table, currents, col_fmt, thresholds):
    """One threshold family: colour = threshold, solid = Castell, dashed = Heatsink."""
    for thr in thresholds:
        tag = ("%g" % thr).replace(".", "")
        dv = _series(table, LABEL_DUAL, currents, col_fmt % tag)
        sv = _series(table, LABEL_SINK, currents, col_fmt % tag)
        if not (np.isfinite(dv).any() or np.isfinite(sv).any()):
            continue
        ax.plot(currents, dv, "-o", color=THR_COLORS[thr], lw=1.8, ms=4)
        ax.plot(currents, sv, "--s", color=THR_COLORS[thr], lw=1.5, ms=4, mfc="white")
        last = dv if np.isfinite(dv).any() else sv
        ax.annotate("%gK" % thr, (currents[-1], last[-1]), xytext=(3, -3),
                    textcoords="offset points", fontsize=7.5, color=THR_COLORS[thr])


def fig_spatial_summary(table, currents, thresholds=VOL_THRESHOLDS,
                        fname="spatial_summary.png"):
    """Spatial family: cooled/heated volume (mm^3) and inner-surface area (mm^2)."""
    if not currents:
        return
    has_area = ("area_cool_05_end" in table.columns and
                np.isfinite(pd.to_numeric(table["area_cool_05_end"], errors="coerce")).any())
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))

    specs = [("vol_cool", axes[0][0], "(a) Cooled VOLUME (domain probes)", "mm$^3$", False),
             ("vol_heat", axes[0][1], "(b) Heated VOLUME (domain probes)", "mm$^3$", False),
             ("area_cool", axes[1][0], "(d) Cooled AREA (inner surface, 2D)", "mm$^2$", True),
             ("area_heat", axes[1][1], "(e) Heated AREA (inner surface, 2D)", "mm$^2$", True)]
    for fam, ax, ttl, unit, needs_area in specs:
        if needs_area and not has_area:
            ax.axis("off")
            ax.set_title(ttl + "\n(no surface data available)", fontsize=10.5)
            continue
        _thr_family_lines(ax, table, currents, fam + "_%s_end", thresholds)
        ax.set_title(ttl, fontsize=10.5)
        ax.set_ylabel(unit)
        ax.set_xlabel(r"Drive current $I_0$ (A)")
        ax.set_ylim(bottom=0)
        _model_style_legend(ax, fontsize=8, loc="upper left")
        ax.annotate("colour = threshold", xy=(0.98, 0.03), xycoords="axes fraction",
                    ha="right", fontsize=7.5, color="0.35")

    for fam, ax, ttl, needs_area in [("vol", axes[0][2], "(c) Balance: cooled / heated VOLUME", False),
                                     ("area", axes[1][2], "(f) Balance: cooled / heated AREA", True)]:
        if needs_area and not has_area:
            ax.axis("off")
            ax.set_title(ttl + "\n(no surface data available)", fontsize=10.5)
            continue
        _thr_family_lines(ax, table, currents, fam + "_ratio_%s_end", thresholds)
        ax.axhline(1.0, color="0.4", lw=1.0, ls=":")
        ax.set_title(ttl, fontsize=10.5)
        ax.set_ylabel("cooled / heated  (-)")
        ax.set_xlabel(r"Drive current $I_0$ (A)")
        _model_style_legend(ax, fontsize=8, loc="upper right")

    fig.suptitle("Spatial metrics per current - how much volume (mm$^3$, probes) and inner-surface "
                 "area (mm$^2$, 2D) each configuration cools and heats\n"
                 "reference time = drive end (%.0f s); values < 1 mean less cooling than heating" % T_END_S,
                 fontsize=12, weight="bold", y=1.0)
    fig.tight_layout()
    _save_figure(fig, fname)


def _inv(v):
    """Element-wise 1/v with NaN for zeros (keeps ratios honest)."""
    v = np.asarray(v, dtype=float)
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v) & (v != 0)
    out[ok] = 1.0 / v[ok]
    return out


def fig_temporal_summary(table, currents, fname="temporal_summary.png"):
    """Temporal family: cooling time, return to normal and the switch-off overshoot."""
    if not currents:
        return
    x = np.arange(len(currents))
    w = 0.38
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.4))

    ax = axes[0][0]
    dv = _series(table, LABEL_DUAL, currents, "overshoot_cold_K")
    sv = _series(table, LABEL_SINK, currents, "overshoot_cold_K")
    ax.bar(x - w / 2, np.nan_to_num(dv), w, color=COLOR_DUAL, edgecolor="k", lw=0.5,
           label=LABEL_DUAL)
    ax.bar(x + w / 2, np.nan_to_num(sv), w, color=COLOR_SINK, edgecolor="k", lw=0.5,
           label=LABEL_SINK)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_title("(a) Switch-off overshoot of the cold side", fontsize=10.5)
    ax.set_ylabel(r"max $\Delta T_c$ after switch-off (K)")
    ax.set_xticks(x)
    ax.set_xticklabels(["%.1f" % c for c in currents], fontsize=8)
    ax.set_xlabel(r"Drive current $I_0$ (A)")
    ax.legend(fontsize=8, loc="upper left")
    ax2 = ax.twinx()
    ax2.grid(False)
    for model, color, mk in ((LABEL_DUAL, COLOR_DUAL, "o"), (LABEL_SINK, COLOR_SINK, "s")):
        t_ovs = _series(table, model, currents, "overshoot_time_s")
        ax2.plot(currents, t_ovs - T_END_S, mk + "--", color=color, lw=1.3, ms=4, alpha=0.8)
    ax2.set_ylabel("time of overshoot after drive end (s)", fontsize=8.5)

    for ax, cols, ttl, ylab in [
            (axes[0][1], [("t_cool63_s", "-o"), ("t_cool90_s", "--s")],
             "(b) Cooling dynamics (from drive start)", "time (s)"),
            (axes[1][0], [("t_ambient_cross_s", "-o"), ("t_back_normal_s", "--s")],
             r"(c) Return to normal after switch-off ($|\Delta T_c|<0.2$ K)", "time after drive end (s)")]:
        for model, color in ((LABEL_DUAL, COLOR_DUAL), (LABEL_SINK, COLOR_SINK)):
            for col, style in cols:
                ax.plot(currents, _series(table, model, currents, col), style, color=color,
                        lw=1.9, ms=4.5, mfc="white" if style.startswith("--") else color)
        ax.set_title(ttl, fontsize=10.5)
        ax.set_xlabel(r"Drive current $I_0$ (A)")
        ax.set_ylabel(ylab)
        ax.set_ylim(bottom=0)
        handles = [Line2D([], [], color="0.35", lw=1.9, marker="o", ms=4, label="63 % / ambient crossing"),
                   Line2D([], [], color="0.35", lw=1.9, ls="--", marker="s", ms=4, mfc="white",
                          label="90 % / back to normal")]
        ax.legend(handles=handles, fontsize=8, loc="upper left")

    ax = axes[1][1]
    ax.axvline(T_END_S, color="0.5", lw=1.0, ls="--")
    ax.annotate("drive end", xy=(T_END_S + 0.6, 0.06), fontsize=8, color="0.35")
    for model, color, dy, mk in ((LABEL_DUAL, COLOR_DUAL, 0.16, "o"),
                                 (LABEL_SINK, COLOR_SINK, -0.16, "s")):
        for row, i0 in enumerate(currents):
            y = row + dy
            ax.plot([0, 60], [y, y], color="0.9", lw=0.6, zorder=0)
            ev = [(_g(row_of(table, model, i0), "t_peak_cool_s"), "*"),
                  (_g(row_of(table, model, i0), "t_ambient_cross_s") + T_END_S, "v"),
                  (_g(row_of(table, model, i0), "overshoot_time_s"), "D")]
            for tv, mk2 in ev:
                if np.isfinite(tv):
                    ax.plot([tv], [y], marker=mk2, ms=6, color=color, mfc="white", mew=1.2,
                            label="_nolegend_")
    ax.set_yticks(range(len(currents)))
    ax.set_yticklabels(["%.1f A" % c for c in currents], fontsize=8)
    ax.set_xlim(0, 62)
    ax.set_xlabel("Time (s)")
    ax.set_title("(d) Event timeline (v = ambient crossing, D = overshoot peak, * = coldest)",
                 fontsize=10.5)
    _model_style_legend(ax, fontsize=8, loc="lower right")

    fig.suptitle("Temporal metrics per current - how fast it cools, how long it takes to recover "
                 "and the overshoot when the current is switched off",
                 fontsize=12.5, weight="bold", y=1.0)
    fig.tight_layout()
    _save_figure(fig, fname)


def fig_energy_summary(table, currents, thresholds=VOL_THRESHOLDS,
                       fname="energy_summary.png"):
    """Energy family: consumption, scale-free efficiencies and the best compromise."""
    if not currents:
        return
    tag_low = ("%g" % thresholds[0]).replace(".", "")
    x = np.arange(len(currents))
    w = 0.38
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.4))

    ax = axes[0][0]
    dv = _series(table, LABEL_DUAL, currents, "W_full_J")
    sv = _series(table, LABEL_SINK, currents, "W_full_J")
    ax.bar(x - w / 2, np.nan_to_num(dv), w, color=COLOR_DUAL, edgecolor="k", lw=0.5,
           label=LABEL_DUAL)
    ax.bar(x + w / 2, np.nan_to_num(sv), w, color=COLOR_SINK, edgecolor="k", lw=0.5,
           label=LABEL_SINK)
    for xi, a, b in zip(x, dv, sv):
        if np.isfinite(a) and np.isfinite(b):
            ax.annotate(_verdict(a, b), (xi, max(a, b) * 1.03), ha="center", fontsize=7,
                        color="0.25")
    ax.set_title(r"(a) Electrical energy of the whole run $W = I_0\!\int_0^{60}|V_1|\,\mathrm{d}t$",
                 fontsize=10.5)
    ax.set_ylabel("J")
    ax.set_xticks(x)
    ax.set_xticklabels(["%.1f" % c for c in currents], fontsize=8)
    ax.set_xlabel(r"Drive current $I_0$ (A)")
    ax.legend(fontsize=8, loc="upper left")
    ax.annotate("absolute joules are not directly comparable:\nthe two exports ramp the current "
                "down differently", xy=(0.03, 0.62), xycoords="axes fraction", fontsize=7.5,
                color="0.35")

    eff_panels = [("K_per_J", "(b) Cooling per joule", "K / J"),
                  ("vol_cool_%s_per_J" % tag_low,
                   "(c) Cooled volume per joule (@%g K)" % thresholds[0], "mm$^3$ / J")]
    for ax, (col, ttl, ylab) in zip([axes[0][1], axes[1][0]], eff_panels):
        for model, color, style in ((LABEL_DUAL, COLOR_DUAL, "-o"),
                                    (LABEL_SINK, COLOR_SINK, "--s")):
            ax.plot(currents, _series(table, model, currents, col), style, color=color,
                    lw=1.9, ms=5, mfc="white" if style.startswith("--") else color, label=model)
        ax.set_title(ttl, fontsize=10.5)
        ax.set_ylabel(ylab)
        ax.set_xlabel(r"Drive current $I_0$ (A)")
        ax.legend(fontsize=8, loc="upper right")
    d_kj = _series(table, LABEL_DUAL, currents, "K_per_J")[-1]
    s_kj = _series(table, LABEL_SINK, currents, "K_per_J")[-1]
    axes[0][1].annotate("at %.1f A: Castell is %s efficient" % (currents[-1], _verdict(d_kj, s_kj)),
                        xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8, color="0.25")

    ax = axes[1][1]
    for model, color, mk in ((LABEL_DUAL, COLOR_DUAL, "o"), (LABEL_SINK, COLOR_SINK, "s")):
        cx = np.abs(_series(table, model, currents, "dT_cold_peak_K"))
        hy = _series(table, model, currents, "dT_hot_peak_K")
        ratio = np.divide(cx, hy, out=np.full_like(cx, np.nan), where=(hy > 0))
        ax.plot(cx, hy, "-", color=color, lw=1.1, alpha=0.6)
        ax.scatter(cx, hy, s=45, marker=mk, color=color, edgecolor="k", lw=0.4,
                   label=model, zorder=4)
        for xx, yy, i0 in zip(cx, hy, currents):
            if np.isfinite(xx) and np.isfinite(yy):
                ax.annotate("%.1f" % i0, (xx, yy), xytext=(4, 4), textcoords="offset points",
                            fontsize=7, color=color)
        if np.isfinite(ratio).any():
            k = int(np.nanargmax(ratio))
            ax.scatter([cx[k]], [hy[k]], s=190, marker="*", color=color, edgecolor="k",
                       lw=0.5, zorder=5, label="_nolegend_")
            ax.annotate("best balance\n($I_0$=%.1f A)" % currents[k],
                        (cx[k], hy[k]), xytext=(6, -14), textcoords="offset points",
                        fontsize=7.5, color=color)
    ax.set_title("(d) Where each configuration is best (star = best cooling/heating balance)",
                 fontsize=10.5)
    ax.set_xlabel(r"cooling $|\Delta T_c|$ peak (K)")
    ax.set_ylabel(r"overheating $\Delta T_h$ peak (K)")
    ax.annotate("better\n(more cooling,\nless overheating)", xy=(0.72, 0.06),
                xycoords="axes fraction", fontsize=8, color="0.3", ha="center")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Energy metrics per current - consumption, scale-free efficiencies "
                 "(K/J, mm$^3$/J) and the lowest-overheating best configuration",
                 fontsize=12.5, weight="bold", y=1.0)
    fig.tight_layout()
    _save_figure(fig, fname)


def _verdict_rows(table, thresholds):
    """(label, column, format) rows of the condensed verdict table."""
    tag_low = ("%g" % thresholds[0]).replace(".", "")
    tag_mid = ("%g" % thresholds[len(thresholds) // 2]).replace(".", "")
    rows = [("Cooling |dTc| (K)", "dT_cold_peak_K", "%.2f"),
            ("Heating dTh (K)", "dT_hot_peak_K", "%.2f"),
            ("|dTc| / dTh", "ratio_cool_heat", "%.2f"),
            ("Cooled vol @%gK (mm3)" % thresholds[0], "vol_cool_%s_end" % tag_low, "%.1f"),
            ("Heated vol @%gK (mm3)" % thresholds[0], "vol_heat_%s_end" % tag_low, "%.1f"),
            ("Heated vol @%gK (mm3)" % thresholds[len(thresholds) // 2],
             "vol_heat_%s_end" % tag_mid, "%.1f")]
    area_cols = [c for c in table.columns if c.startswith("area_")]
    if area_cols:
        rows += [("Cooled area @%gK (mm2)" % thresholds[0], "area_cool_%s_end" % tag_low, "%.1f"),
                 ("Heated area @%gK (mm2)" % thresholds[0], "area_heat_%s_end" % tag_low, "%.1f")]
    rows += [("t to 63%% cooling (s)", "t_cool63_s", "%.1f"),
             ("Back to normal (s)", "t_back_normal_s", "%.1f"),
             ("Switch-off overshoot (K)", "overshoot_cold_K", "%.2f"),
             ("W[0-60] (J)", "W_full_J", "%.2f"),
             ("K per J", "K_per_J", "%.3f"),
             ("Cooled mm3 per J", "vol_cool_%s_per_J" % tag_low, "%.2f")]
    return rows


def fig_verdict_table(table, currents, thresholds=VOL_THRESHOLDS, fname="verdict_table.png"):
    """Condensed numeric table: Castell, Heatsink and their ratio, per current."""
    if not currents:
        return
    rows = _verdict_rows(table, thresholds)
    labels = [r[0] for r in rows]
    cols = ["%.1f A" % c for c in currents]

    data_dual, data_sink, data_ratio = [], [], []
    for _, col, fmt in rows:
        dv = _series(table, LABEL_DUAL, currents, col)
        sv = _series(table, LABEL_SINK, currents, col)
        data_dual.append([fmt % v if np.isfinite(v) else "-" for v in dv])
        data_sink.append([fmt % v if np.isfinite(v) else "-" for v in sv])
        r = [_ratio(a, b) for a, b in zip(dv, sv)]
        data_ratio.append(["%.2f" % v if np.isfinite(v) else "-" for v in r])

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 0.42 * len(rows) + 2.2))
    for ax, data, ttl in ((axes[0], data_dual, LABEL_DUAL),
                          (axes[1], data_sink, LABEL_SINK),
                          (axes[2], data_ratio, "ratio  Castell / Heatsink")):
        ax.axis("off")
        tb = ax.table(cellText=data, rowLabels=labels, colLabels=cols, loc="center",
                      cellLoc="center")
        tb.auto_set_font_size(False)
        tb.set_fontsize(7.5)
        tb.scale(1.0, 1.25)
        for (r, c), cell in tb.get_celld().items():
            cell.set_linewidth(0.4)
            if r == 0:
                cell.set_text_props(weight="bold")
            if c == -1:
                cell.set_text_props(ha="left")
        ax.set_title(ttl, fontsize=10.5, weight="bold", pad=14)

    fig.suptitle("Condensed verdict per drive current - reference: drive end %.0f s, run end %.0f s "
                 "(ratio > 1 means the Dual-TEC has more of that quantity)"
                 % (T_END_S, T_FINAL_S), fontsize=12, weight="bold", y=1.02)
    _save_figure(fig, fname)


def print_best_operating_points(table, currents, thresholds=VOL_THRESHOLDS):
    """Which current is best for cooling, for balance and for lowest overheating."""
    tag_low = ("%g" % thresholds[0]).replace(".", "")
    print("\nBEST OPERATING POINTS")
    for model in (LABEL_DUAL, LABEL_SINK):
        rows = [row_of(table, model, c) for c in currents]
        rows = [r for r in rows if r]
        if not rows:
            continue
        c_cool = max(rows, key=lambda r: abs(_g(r, "dT_cold_peak_K")))

        def _safe_max(rs, key):
            rs2 = [r for r in rs if np.isfinite(_g(r, key))]
            return max(rs2, key=lambda r: _g(r, key)) if rs2 else None

        def _safe_min(rs, key):
            rs2 = [r for r in rs if np.isfinite(_g(r, key))]
            return min(rs2, key=lambda r: _g(r, key)) if rs2 else None

        c_bal = _safe_max(rows, "ratio_cool_heat")
        c_hot = _safe_min(rows, "overshoot_cold_K")
        c_eff = _safe_max(rows, "vol_cool_%s_per_J" % tag_low)
        print("  %-18s max cooling   : I0 = %.1f A  (|dTc| = %.2f K)"
              % (model, _g(c_cool, "I0_A"), abs(_g(c_cool, "dT_cold_peak_K"))))
        if c_bal:
            print("  %-18s best balance  : I0 = %.1f A  (|dTc|/dTh = %.2f)"
                  % ("", _g(c_bal, "I0_A"), _g(c_bal, "ratio_cool_heat")))
        if c_hot:
            print("  %-18s least overshoot: I0 = %.1f A  (%.2f K)"
                  % ("", _g(c_hot, "I0_A"), _g(c_hot, "overshoot_cold_K")))
        if c_eff:
            print("  %-18s best efficiency: I0 = %.1f A  (%.2f mm3/J)"
                  % ("", _g(c_eff, "I0_A"), _g(c_eff, "vol_cool_%s_per_J" % tag_low)))


def main_comparison():
    """Per-current comparison of the Dual-TEC (Castell) and Heatsink configurations.

    Produces (a) a tidy CSV with every metric of the four families, (b) a
    console verdict per drive current and (c) five summary figures.
    """
    print("=" * 118)
    print("PER-CURRENT COMPARISON - Dual-TEC (Castell) vs Heatsink  "
          "(25 s drive pulse, room temperature)")
    print("=" * 118)

    print("\n[1/5] Loading probe tables ...")
    df_dual = load_accumulated_probes(CONFIG["DUAL_TEC_PROBES_CSV"],
                                      CONFIG["T_AMB_DEFAULT_C"])
    df_sink = load_accumulated_probes(CONFIG["HEATSINK_PROBES_CSV"],
                                      CONFIG["T_AMB_DEFAULT_C"])
    currents = sorted(set(np.round(df_dual["I0"].unique(), 4)))
    if not CONFIG.get("COMPARE_ALL_CURRENTS", True):
        currents = [c for c in CONFIG["COMPARE_CURRENTS"]
                    if any(np.isclose(df_dual["I0"], c))]
    print(f"      Dual-TEC : {len(df_dual):,} rows | currents {currents}")
    print(f"      Heatsink : {len(df_sink):,} rows | currents "
          f"{sorted(set(np.round(df_sink['I0'].unique(), 4)))}")
    print(f"      Volume probes: Castell={has_volume_probes(df_dual)} | "
          f"Heatsink={has_volume_probes(df_sink)}")
    print(f"      Parallel-TEC voltage (Probe 3): "
          f"{bool(df_dual['voltage_tec2_V'].notna().any())}")

    print("\n[2/5] Per-current metrics from the probes ...")
    table = build_comparison_table(df_dual, df_sink, currents)
    print(f"      {len(table)} rows computed")

    print("\n[3/5] 2D inner-surface area metrics (Delaunay, mm^2) ...")
    area_currents = CONFIG.get("AREA_CURRENTS") or CONFIG["SURFACE_CURRENTS"]
    area_currents = [c for c in area_currents if any(np.isclose(currents, c))]
    for key, label in (("DUAL_TEC_INNER_SURF_CSV", LABEL_DUAL),
                       ("HEATSINK_INNER_SURF_CSV", LABEL_SINK)):
        path = CONFIG[key]
        if not os.path.exists(path):
            print(f"      [skip] {path} not found")
            continue
        series = collect_area_series(path, area_currents, CONFIG["T_AMB_DEFAULT_C"])
        table = add_surface_area_metrics(table, series, label)
        if series:
            hull = list(series.values())[0]["hull_mm2"]
            print(f"      {label}: areas for currents {sorted(series)} | hull = {hull:.1f} mm^2")

    print("\n[4/5] Writing the comparison table ...")
    csv_path = os.path.abspath(CONFIG["COMPARISON_CSV"])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    table.to_csv(csv_path, index=False)
    print(f"      saved {csv_path}  ({table.shape[0]} rows x {table.shape[1]} columns)")
    print_comparison_summary(table, currents)
    print_best_operating_points(table, currents)

    print("\n[5/5] Summary figures ...")
    fig_temperature_summary(table, currents)
    fig_spatial_summary(table, currents)
    fig_temporal_summary(table, currents)
    fig_energy_summary(table, currents)
    fig_verdict_table(table, currents)
    print(f"      figures written to {_fig_dir()}")
    print("\nComparison pipeline complete.")

if __name__ == "__main__":
    main_comparison()

