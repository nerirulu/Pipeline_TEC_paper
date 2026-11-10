# =============================================================================
# csv_sensitivity_analysis.py
# =============================================================================
import os
import re
import glob
import csv
from io import StringIO

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IEEE_RC = {
    # NOTE: "savefig.bbox": "tight" is intentionally NOT set here. With "tight",
    # any artist outside the axes (e.g. bar labels placed in data coordinates
    # far beyond the axis limits) inflates the saved canvas to absurd sizes.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 25,
    "axes.titlesize": 25,
    "axes.labelsize": 25,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "legend.fontsize": 25,
    "legend.title_fontsize": 25,
    "figure.titlesize": 25,
    "figure.titleweight": "normal",
    "lines.linewidth": 2.2,
    "lines.markersize": 18,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.55,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.15,
}
plt.rcParams.update(IEEE_RC)

CBLUE   = "#1f77b4"
CORANGE = "#ff7f0e"
CGREEN  = "#2ca02c"
CRED    = "#d62728"
CPURPLE = "#9467bd"
CBROWN  = "#8c564b"

METRIC_LABELS = {
    "dT_cold_peak_K":   r"$\Delta T_{\text{cold,peak}}$ (K)",
    "dT_hot_peak_K":    r"$\Delta T_{\text{hot,peak}}$ (K)",
    "t_cool63_s":       r"$t_{\text{cool,63\%}}$ (s)",
    "t_cool90_s":       r"$t_{\text{cool,90\%}}$ (s)",
    "t_hot_peak_s":     r"$t_{\text{hot,peak}}$ (s)",
    "cold_T_final_C":   r"$T_{\text{cold,final}}$ ($^\circ$C)",
    "hot_T_final_C":    r"$T_{\text{hot,final}}$ ($^\circ$C)",
    "surf_min_K":       r"$T_{\text{surf,min}}$ (K)",
    "surf_max_K":       r"$T_{\text{surf,max}}$ (K)",
    "vol_cool_5_end":   r"$V_{\text{cool,5C,end}}$ (mm$^3$)",
    "vol_cool_1_end":   r"$V_{\text{cool,1C,end}}$ (mm$^3$)",
    "vol_heat_5_end":   r"$V_{\text{heat,5C,end}}$ (mm$^3$)",
    "vol_heat_1_end":   r"$V_{\text{heat,1C,end}}$ (mm$^3$)",
    "V1_peak_V":        r"$V_{1,\text{peak}}$ (V)",
    "net_power_W":      r"$P_{\text{net}}$ (W)",
    "W_full_J":         r"$W_{\text{full}}$ (J)",
    "K_per_J":          r"$K/J$ (K/J)",
    "E_areal_J_m2":     r"$E_{\text{areal}}$ (J/m$^2$)",
}

DEFAULT_METRICS = [
    "dT_cold_peak_K", "dT_hot_peak_K", "t_cool63_s", "t_cool90_s",
    "t_hot_peak_s", "cold_T_final_C", "hot_T_final_C",
    "surf_min_K", "surf_max_K",
    "vol_cool_5_end", "vol_cool_1_end",
    "vol_heat_5_end", "vol_heat_1_end",
    "V1_peak_V", "net_power_W", "W_full_J", "K_per_J", "E_areal_J_m2",
]


def is_cold_metric(metric):
    """Return True for cold-side metrics (most negative value)."""
    return "cold" in metric or "surf_min" in metric
TEMP_METRICS = {"hot", "cold", "surf_min", "surf_max"}

# File names start with "Probes auto" (any case / separators); it is never shown in labels.
_PROBES_AUTO_PREFIX = re.compile(r"^\s*probes[\s_\-]*auto[\s_\-:.]*", re.IGNORECASE)

def strip_probes_auto(name):
    """Remove the leading 'Probes auto' from a file/case name (falls back to the original if nothing is left)."""
    if not isinstance(name, str):
        return name
    stripped = _PROBES_AUTO_PREFIX.sub("", name)
    return stripped if stripped.strip() else name

def format_name(name):
    if not isinstance(name, str):
        return str(name)
    name = strip_probes_auto(name)
    parts = name.split("(")
    base = parts[0].replace("_", " ").strip().title()
    if len(parts) > 1:
        return f"{base} ({parts[1]}"
    return base

def find_col(cols, substr):
    sl = substr.lower()
    for c in cols:
        if sl in c.lower():
            return c
    return None

def load_csv(path):
    meta, data_lines = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("%"):
                meta.append(line)
            elif line.strip():
                data_lines.append(line)

    header = None
    for m in meta:
        if m.lstrip("%").strip().lower().startswith("i0") or m.startswith("% I0"):
            header = [h.strip().strip('"') for h in next(csv.reader([m[2:]]))]
            break
    if header is None and data_lines:
        header = [h.strip().strip('"') for h in next(csv.reader([data_lines[0]]))]
        data_lines = data_lines[1:]

    df = pd.read_csv(StringIO("\n".join(data_lines)), header=None, names=header or [])
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except (ValueError, TypeError):
            pass

    param_col = df.columns[1] if len(df.columns) >= 2 else None
    return df, meta, header, param_col

def _safe(x, default=np.nan):
    try:
        x = float(x)
        if np.isfinite(x):
            return x
    except (TypeError, ValueError):
        pass
    return default

def compute_metrics(df):
    i0_col = find_col(df.columns, "I0")
    t_col  = find_col(df.columns, "Time (s)")
    hot_c  = find_col(df.columns, "Hot")
    cold_c = find_col(df.columns, "Cold")
    maxs   = find_col(df.columns, "Max_surface")
    mins   = find_col(df.columns, "Min_surface")
    volt_c = find_col(df.columns, "Terminal voltage")
    curr_c = find_col(df.columns, "Terminal current")

    vol_cool = {thr: find_col(df.columns, f"vol_cool{thr}C") for thr in (0.5, 1, 2, 5)}
    vol_heat = {thr: find_col(df.columns, f"vol_heat{thr}C") for thr in (0.5, 1, 2, 5)}

    records = []
    for i0, g in df.groupby(i0_col):
        g = g.sort_values(t_col)
        t = g[t_col].to_numpy(dtype=float)

        hot  = g[hot_c].to_numpy(dtype=float) if hot_c else np.full_like(t, np.nan)
        cold = g[cold_c].to_numpy(dtype=float) if cold_c else np.full_like(t, np.nan)
        ts_max = g[maxs].to_numpy(dtype=float) if maxs else np.full_like(t, np.nan)
        ts_min = g[mins].to_numpy(dtype=float) if mins else np.full_like(t, np.nan)

        v = g[volt_c].to_numpy(dtype=float) if volt_c else np.zeros_like(t)
        c = g[curr_c].to_numpy(dtype=float) if curr_c else np.zeros_like(t)
        p = v * c

        cold_init, cold_min, cold_max = _safe(cold[0]), _safe(np.nanmin(cold)), _safe(np.nanmax(cold))
        hot_min, hot_max = _safe(np.nanmin(hot)), _safe(np.nanmax(hot))

        dT_cold_peak = cold_max - cold_min
        dT_hot_peak  = hot_max - hot_min

        def time_to_frac(signal, init, extremum, frac, direction):
            if not np.isfinite(extremum) or not np.isfinite(init):
                return np.nan
            span = extremum - init
            if abs(span) < 1e-12:
                return np.nan
            target = init + direction * frac * span
            idx = np.where(direction * (signal - target) >= 0)[0]
            return float(t[idx[0]]) if idx.size else np.nan

        t_cool63 = time_to_frac(cold, cold_init, cold_min, 0.63, -1)
        t_cool90 = time_to_frac(cold, cold_init, cold_min, 0.90, -1)
        t_hot_peak = float(t[int(np.nanargmax(hot))]) if np.any(np.isfinite(hot)) else np.nan

        def vol_end(thr, kind):
            col = (vol_cool if kind == "cool" else vol_heat).get(thr)
            if col is None:
                return np.nan
            vals = g[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            return float(vals[-1]) if vals.size else np.nan

        _trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        W_full = float(_trapezoid(p, t)) if t.size > 1 else 0.0
        net_power = float(np.nanmean(p)) if p.size else np.nan
        V1_peak = float(np.nanmax(np.abs(v))) if v.size else np.nan
        K_per_J = (dT_cold_peak / W_full) if (np.isfinite(dT_cold_peak) and W_full > 0) else np.nan

        records.append({
            "I0 (A)": float(i0),
            "dT_cold_peak_K": dT_cold_peak,
            "dT_hot_peak_K": dT_hot_peak,
            "t_cool63_s": t_cool63,
            "t_cool90_s": t_cool90,
            "t_hot_peak_s": t_hot_peak,
            "cold_T_final_C": float(cold[-1]) if cold.size else np.nan,
            "hot_T_final_C": float(hot[-1]) if hot.size else np.nan,
            "surf_min_K": float(np.nanmin(ts_min)) if ts_min.size else np.nan,
            "surf_max_K": float(np.nanmax(ts_max)) if ts_max.size else np.nan,
            "vol_cool_5_end": vol_end(5, "cool"),
            "vol_cool_1_end": vol_end(1, "cool"),
            "vol_heat_5_end": vol_end(5, "heat"),
            "vol_heat_1_end": vol_end(1, "heat"),
            "V1_peak_V": V1_peak,
            "net_power_W": net_power,
            "W_full_J": W_full,
            "K_per_J": K_per_J,
            "E_areal_J_m2": W_full,
        })

    out = pd.DataFrame(records)
    return out.sort_values("I0 (A)").reset_index(drop=True) if not out.empty else out

def new_fig(title=None, figsize=(10.5, 5.6)):
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        ax.set_title(title, pad=14)
    return fig, ax

def save_fig(fig, path, max_width_inches=14, max_height_inches=14):
    """Save a figure, limiting size to avoid renderer errors.

    Matplotlib's Agg backend cannot render images larger than 2^16 pixels
    in either dimension.  If a figure is too large (e.g. lots of parameter
    values driving the height), cap its pixel size here.
    """
    max_px = 2**16 - 1  # 65535

    # Cap figure dimensions in inches so the canvas cannot explode.
    w = fig.get_figwidth()
    h = fig.get_figheight()
    if w > max_width_inches or h > max_height_inches:
        scale = min(max_width_inches / w, max_height_inches / h)
        fig.set_figwidth(w * scale)
        fig.set_figheight(h * scale)
        print(f"  [save_fig] Resized figure to fit within "
              f"{max_width_inches}x{max_height_inches} inches "
              f"(was {w:.1f}x{h:.1f} inches)")

    # Determine a safe DPI so pixel dimensions stay under 2^16.
    base_dpi = fig.get_dpi() if fig.get_dpi() else 100.0
    w_in = fig.get_figwidth()
    h_in = fig.get_figheight()
    max_dpi_w = max_px / w_in if w_in > 0 else base_dpi
    max_dpi_h = max_px / h_in if h_in > 0 else base_dpi
    safe_dpi = min(base_dpi, max_dpi_w, max_dpi_h, 300.0)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Try saving; if the renderer still complains (e.g. very tall figures
    # with legends outside the axes), retry with progressively smaller
    # dimensions and DPI.
    def _pick_bbox(dpi):
        """Use a tight bbox (keeps outside legends/tables) unless it would be absurdly large."""
        try:
            fig.canvas.draw()
            tb = fig.get_tightbbox(fig.canvas.get_renderer())
            if tb is not None and (tb.width * dpi > max_px * 0.9 or tb.height * dpi > max_px * 0.9
                                   or tb.width > 3 * max_width_inches or tb.height > 3 * max_height_inches):
                print("  [save_fig] Tight bbox is unreasonably large; saving the full canvas instead")
                return None
        except Exception:
            return None
        return "tight"

    attempt = 0
    while True:
        attempt += 1
        try:
            fig.savefig(path, dpi=safe_dpi, bbox_inches=_pick_bbox(safe_dpi), pad_inches=0.1)
            plt.close(fig)
            return
        except ValueError as exc:
            plt.close(fig)
            if attempt >= 5 or "too large" not in str(exc).lower():
                raise
            # Shrink further and try again.
            scale = 0.7
            fig.set_figwidth(w_in * scale)
            fig.set_figheight(h_in * scale)
            w_in *= scale
            h_in *= scale
            safe_dpi = min(base_dpi, max_px / w_in, max_px / h_in, 300.0)
            print(f"  [save_fig] Retry {attempt}: shrinking to "
                  f"{w_in:.1f}x{h_in:.1f} inches at {safe_dpi:.0f} dpi")

def extract_time_series(df, t_start=5.0):
    i0_col = find_col(df.columns, "I0")
    param_col = df.columns[1] if len(df.columns) >= 2 else None
    t_col = find_col(df.columns, "Time (s)")
    cols_needed = {
        "hot": find_col(df.columns, "Hot"),
        "cold": find_col(df.columns, "Cold"),
        "surf_max": find_col(df.columns, "Max_surface"),
        "surf_min": find_col(df.columns, "Min_surface"),
        "volt": find_col(df.columns, "Terminal voltage"),
        "curr": find_col(df.columns, "Terminal current"),
        "vol_cool2": find_col(df.columns, "vol_cool2C"),
        "vol_heat2": find_col(df.columns, "vol_heat2C"),
    }
    series = {}
    for pval, g in df.groupby(param_col):
        for i0, sub in g.groupby(i0_col):
            sub = sub.sort_values(t_col)
            t_raw = sub[t_col].to_numpy(dtype=float)
            
            mask = t_raw >= t_start
            t = t_raw[mask] - t_start
            
            y = {}
            for k, c in cols_needed.items():
                if c is not None:
                    arr = sub[c].to_numpy(dtype=float)[mask]
                    if k in TEMP_METRICS and arr.size > 0:
                        arr = arr - arr[0]
                    y[k] = arr
                else:
                    y[k] = None
                    
            series[(round(float(pval), 9), round(float(i0), 9))] = (t, y)
    return series

def _metric_signal(y, metric):
    """Return the plotted signal for a metric (power = V*I), or None if unavailable."""
    if metric == "power":
        if y.get("volt") is None or y.get("curr") is None:
            return None
        return y["volt"] * y["curr"]
    return y.get(metric)


def _extreme_value(metric, sig):
    """Value shown in the legend: most negative for cold-side metrics, else max |value| (signed)."""
    if sig is None:
        return float("nan")
    sig = np.asarray(sig, dtype=float)
    finite = sig[np.isfinite(sig)]
    if finite.size == 0:
        return float("nan")
    if is_cold_metric(metric):
        return float(np.min(finite))
    return float(sig[int(np.nanargmax(np.abs(np.where(np.isfinite(sig), sig, np.nan))))])


def _style_maps(series, baseline_param):
    param_vals = sorted({k[0] for k in series.keys()})
    currents = sorted({k[1] for k in series.keys()})
    palette = [CBLUE, CGREEN, CORANGE, CRED, CPURPLE, CBROWN]
    current_colors = {i0: palette[i % len(palette)] for i, i0 in enumerate(currents)}
    linestyle_cycle = ['--', ':', '-.', (0, (3, 1, 1, 1))]
    pval_ls = {baseline_param: '-'}
    for i, pv in enumerate([p for p in param_vals if p != baseline_param]):
        pval_ls[pv] = linestyle_cycle[i % len(linestyle_cycle)]
    return param_vals, currents, current_colors, pval_ls


def _pct_records(series, baseline_param, metric, param_vals, currents):
    """Numeric sensitivity records: legend (final) value, baseline value, differences and S per case."""
    recs = []
    for pv in param_vals:
        if pv == baseline_param:
            continue
        for i0 in currents:
            key, base_key = (pv, i0), (baseline_param, i0)
            if key not in series or base_key not in series:
                continue
            val = _extreme_value(metric, _metric_signal(series[key][1], metric))
            base = _extreme_value(metric, _metric_signal(series[base_key][1], metric))
            ok = np.isfinite(val) and np.isfinite(base)
            d_abs = (val - base) if ok else np.nan
            pct = (val - base) / abs(base) * 100.0 if (ok and abs(base) > 1e-12) else np.nan
            dp_pct = (pv - baseline_param) / abs(baseline_param) * 100.0 if abs(baseline_param) > 1e-12 else np.nan
            # Normalised sensitivity: % change of the output per % change of the varied parameter
            sens = pct / dp_pct if (np.isfinite(pct) and np.isfinite(dp_pct) and abs(dp_pct) > 1e-12) else np.nan
            recs.append({"I0 (A)": i0, "parameter_value": pv, "baseline_parameter": baseline_param,
                         "d_param_pct": dp_pct, "value": val, "base": base,
                         "delta_abs": d_abs, "delta_pct": pct, "S": sens})
    return recs


def _pct_rows(series, baseline_param, metric, param_vals, currents, param_name):
    """Formatted rows for the on-figure table."""
    def f(x, spec):
        return format(x, spec) if np.isfinite(x) else "n/a"
    return [[f"{r['I0 (A)']:g} A, {r['parameter_value']:g}", f(r["value"], ".3f"), f(r["base"], ".3f"),
             f(r["delta_abs"], ".3f"), f(r["delta_pct"], ".1f"), f(r["S"], ".2f")]
            for r in _pct_records(series, baseline_param, metric, param_vals, currents)]


def _legend_and_table(fig, ax, rows, param_label="param", legend_fs=19, table_fs=14):
    """Legend in the top-right column, sensitivity table directly beneath it (bottom right)."""
    handles, labels = ax.get_legend_handles_labels()
    n_leg = len(labels)
    n_rows = len(rows)

    # Size the figure so legend + table both fit in the right-hand column.
    leg_h = 0.0195 * legend_fs * n_leg + 0.20
    tab_h = 0.024 * table_fs * (n_rows + 1) + 0.15 if n_rows else 0.0
    fig_w = 14.0
    fig_h = max(5.6, leg_h + tab_h + 2.3)
    fig.set_size_inches(fig_w, fig_h)

    left_frac = 0.08
    main_right = 0.60
    fig.subplots_adjust(left=left_frac, right=main_right, bottom=0.12, top=0.90)

    col_x = main_right + 0.03
    col_w = 0.99 - col_x
    leg_y = 1 - 0.10 - leg_h / fig_h
    tab_hf = min(tab_h / fig_h, leg_y - 0.12 - 0.02)

    leg_ax = tab_ax = None
    if handles:
        leg_ax = fig.add_axes([col_x, leg_y, col_w, leg_h / fig_h])
        leg_ax.axis("off")
        leg = leg_ax.legend(handles, labels, frameon=False, fontsize=legend_fs,
                            loc="upper left", bbox_to_anchor=(0.0, 1.0), borderaxespad=0.0)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        # Measure the real legend width so the table can match it exactly.
        fig.canvas.draw()
        bb = leg.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        leg_w = min(bb.width, col_w)
    else:
        leg_w = col_w

    # Give the freed space back to the main plot; right column is exactly legend-wide.
    main_right = 0.99 - leg_w - 0.03
    fig.subplots_adjust(left=left_frac, right=main_right, bottom=0.12, top=0.90)
    col_x = main_right + 0.03
    if leg_ax is not None:
        leg_ax.set_position([col_x, leg_y, leg_w, leg_h / fig_h])

    if rows:
        tab_ax = fig.add_axes([col_x, 0.12, leg_w, tab_hf])
        tab_ax.axis("off")
        tab = tab_ax.table(cellText=rows,
                           colLabels=["I0, Parameter", "Value", "Base", r"$\Delta$", r"$\Delta$ %", "S"],
                           colWidths=[0.25, 0.17, 0.17, 0.17, 0.13, 0.13], cellLoc="center", loc="upper left")
        tab.auto_set_font_size(False)
        tab.set_fontsize(table_fs)
        tab.scale(1.0, 1.6)
        fig.text(col_x, 0.07, r"$S=\Delta\%\,/\,\Delta\%_{param}$ (normalised sensitivity)",
                 fontsize=table_fs, ha="left", va="center")
        for (r, c), cell in tab.get_celld().items():
            cell.set_linewidth(0.5)
            if r == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#e8e8e8")


def plot_transient_multi(ax, series, baseline_param, metric, ylabel, title, param_name="param"):
    baseline_param = round(float(baseline_param), 9)
    param_vals, currents, current_colors, pval_ls = _style_maps(series, baseline_param)

    for pv in param_vals:
        for i0 in currents:
            key = (pv, i0)
            if key not in series:
                continue
            t, y = series[key]
            y_plot = _metric_signal(y, metric)
            if y_plot is None:
                continue

            if pv == baseline_param:
                tag = " [baseline]"
            else:
                val = _extreme_value(metric, y_plot)
                tag = f"  ({val:.3f})" if np.isfinite(val) else "  (n/a)"

            ax.plot(t, y_plot, color=current_colors.get(i0, "#555555"), linestyle=pval_ls[pv], linewidth=2.0,
                    label=f"I0={i0:g} A, {param_name}={pv:g}{tag}")

    ax.set_xlabel("Time from pulse onset (s)", labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_title(title, pad=12)
    ax.set
    rows = _pct_rows(series, baseline_param, metric, param_vals, currents, param_name)
    _legend_and_table(ax.figure, ax, rows, param_label=param_name)


def plot_transient_diff_multi(ax, series, baseline_param, metric, ylabel, title, param_name="param"):
    baseline_param = round(float(baseline_param), 9)
    param_vals, currents, current_colors, pval_ls = _style_maps(series, baseline_param)

    for pv in param_vals:
        if pv == baseline_param:
            continue
        for i0 in currents:
            key, base_key = (pv, i0), (baseline_param, i0)
            if key not in series or base_key not in series:
                continue
            t, y = series[key]
            bt, by = series[base_key]
            y_plot, by_plot = _metric_signal(y, metric), _metric_signal(by, metric)
            if y_plot is None or by_plot is None:
                continue

            min_len = min(len(t), len(bt), len(y_plot), len(by_plot))
            diff = y_plot[:min_len] - by_plot[:min_len]
            val = _extreme_value(metric, y_plot)
            tag = f"  ({val:.3f})" if (diff.size > 0 and np.any(np.isfinite(diff)) and np.isfinite(val)) else "  (n/a)"

            ax.plot(t[:min_len], diff, color=current_colors.get(i0, "#555555"), linestyle=pval_ls[pv], linewidth=2.0,
                    label=f"I0={i0:g} A, {param_name}={pv:g}{tag}")

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Time from pulse onset (s)", labelpad=10)
    ax.set_ylabel(r"$\Delta$ (" + ylabel + ")", labelpad=10)
    ax.set_title(title, pad=12)

    rows = _pct_rows(series, baseline_param, metric, param_vals, currents, param_name)
    _legend_and_table(ax.figure, ax, rows, param_label=param_name)


def plot_tornado_grouped_by_current(changes_df, case_dir, name, param_display_name, chosen_metric="dT_cold_peak_K"):
    """Per-case tornado chart grouped by current (I0)."""
    if changes_df.empty or (chosen_metric + "_abs") not in changes_df.columns:
        return

    currents = sorted(changes_df["I0 (A)"].unique())
    param_vals = sorted(changes_df["parameter"].unique())
    col_abs = chosen_metric + "_abs"

    fig, ax = new_fig(f"{format_name(name)} — Sensitivity by Current ({chosen_metric})", figsize=(10.5, max(5, len(param_vals) * 1.2)))
    palette = [CBLUE, CGREEN, CORANGE, CRED, CPURPLE, CBROWN]
    current_colors = {i0: palette[i % len(palette)] for i, i0 in enumerate(currents)}

    y_base = np.arange(len(param_vals))
    bar_height = 0.8 / len(currents)

    # Extreme value across currents for each parameter value (for legend/y-tick labels)
    if is_cold_metric(chosen_metric):
        extreme_vals = {pv: float(changes_df[changes_df["parameter"] == pv][col_abs].min()) for pv in param_vals}
    else:
        extreme_vals = {pv: float(changes_df[changes_df["parameter"] == pv][col_abs].abs().max()) for pv in param_vals}

    for i, i0 in enumerate(currents):
        sub = changes_df[changes_df["I0 (A)"] == i0]
        vals = [sub[sub["parameter"] == pv][col_abs].iloc[0] if not sub[sub["parameter"] == pv].empty else 0.0 for pv in param_vals]
        y_pos = y_base + (i - (len(currents) - 1) / 2) * bar_height

        ax.barh(y_pos, vals, height=bar_height, color=current_colors[i0], edgecolor="black", linewidth=0.5, label=f"I0 = {i0:g} A")

        for y, v in zip(y_pos, vals):
            if np.isfinite(v) and abs(v) > 1e-3:
                ax.annotate(f"{v:.3f}", xy=(v, y), xytext=(4 if v >= 0 else -4, 0),
                            textcoords="offset points", va="center",
                            ha="left" if v >= 0 else "right", fontsize=17)

    ax.axvline(0, color="black", linewidth=1.0)
    y_labels = [f"{param_display_name} = {pv:g}\nMax: {extreme_vals[pv]:.3f}" for pv in param_vals]
    ax.set_yticks(y_base)
    ax.set_yticklabels(y_labels, fontsize=24)
    ax.tick_params(axis="x", labelsize=24)
    ax.set_xlabel(f"Absolute change in {METRIC_LABELS.get(chosen_metric, chosen_metric)}", labelpad=10, fontsize=30)
    ax.set_title(f"Tornado Grouped by Current — {format_name(name)}", pad=12, fontsize=34)

    legend_labels = []
    for i0 in currents:
        entries = [f"I0 = {i0:g} A"]
        for pv in param_vals:
            entries.append(f"{param_display_name}={pv:g}: {extreme_vals[pv]:.3f}")
        legend_labels.append("\n".join(entries))

    ax.legend(labels=legend_labels, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    save_fig(fig, os.path.join(case_dir, f"tornado_by_current_{chosen_metric}.png"))

TRANSIENT_QUANTITIES = [
    ("hot",       r"$\Delta T_{\text{hot}}$ (K)"),
    ("cold",      r"$\Delta T_{\text{cold}}$ (K)"),
    ("power",     r"$P$ (W)"),
    ("vol_cool2", r"$V_{\text{cooled,2C}}$ (mm$^3$)"),
    ("surf_min",  r"$\Delta T_{\text{surf,min}}$ (K)"),
    ("vol_heat2", r"$V_{\text{heated,2C}}$ (mm$^3$)"),
    ("surf_max",  r"$\Delta T_{\text{surf,max}}$ (K)"),
]


def plot_global_summary_tornado_S(output_root, quantity, label):
    """Global tornado of the normalised sensitivity S for one transient quantity.

    One bar per (case, I0): the S with the largest |S| over the varied parameter values (sign kept).
    Cases are sorted by their largest |S|, biggest on top.
    """
    csv_path = os.path.join(output_root, "all_cases_S_values_transient.csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    df = df[(df["quantity"] == quantity) & np.isfinite(df["S"])]
    if df.empty:
        print(f"  [global tornado S] {quantity}: no finite data, skipped")
        return

    recs = []
    for (case, i0), sub in df.groupby(["case", "I0 (A)"]):
        recs.append({"case": case, "I0 (A)": float(i0), "S": float(sub.loc[sub["S"].abs().idxmax(), "S"])})
    d = pd.DataFrame(recs)
    cases = d.groupby("case")["S"].apply(lambda x: x.abs().max()).sort_values(ascending=True).index.tolist()
    currents = sorted(d["I0 (A)"].unique())

    fig, ax = new_fig(f"Global Sensitivity $S$ — {label}", figsize=(11, max(5, len(cases) * 1.0)))
    palette = [CBLUE, CGREEN, CORANGE, CRED, CPURPLE, CBROWN]
    current_colors = {i0: palette[i % len(palette)] for i, i0 in enumerate(currents)}
    y_base = np.arange(len(cases))
    bar_height = 0.8 / len(currents)

    for i, i0 in enumerate(currents):
        sub = d[d["I0 (A)"] == i0]
        vals = [float(sub[sub["case"] == c]["S"].iloc[0]) if not sub[sub["case"] == c].empty else 0.0 for c in cases]
        y_pos = y_base + (i - (len(currents) - 1) / 2) * bar_height
        ax.barh(y_pos, vals, height=bar_height, color=current_colors[i0], edgecolor="black",
                linewidth=0.5, label=f"I0 = {i0:g} A")
        for y, v in zip(y_pos, vals):
            if np.isfinite(v) and abs(v) > 1e-6:
                ax.annotate(f"{v:.2f}", xy=(v, y), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                            va="center", ha="left" if v >= 0 else "right", fontsize=8.5)

    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(y_base)
    ax.set_yticklabels(cases, fontsize=10)
    ax.set_xlabel(r"$S=\Delta\%\,/\,\Delta\%_{param}$ (largest $|S|$ over parameter values)", labelpad=10)
    ax.set_title(f"Global Sensitivity $S$ by Current, All Cases — {label}", pad=12)
    ax.margins(x=0.12)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save_fig(fig, os.path.join(output_root, f"global_summary_tornado_S_{quantity}.png"))

def plot_heatmap_pct(changes_df, metrics, case_dir, name, param_display_name):
    if changes_df.empty: return
    currents, param_vals = sorted(changes_df["I0 (A)"].unique()), sorted(changes_df["parameter"].unique())
    valid_metrics = [m for m in metrics if (m + "_pct") in changes_df.columns]
    if not valid_metrics or not param_vals or not currents: return

    fig, axes = plt.subplots(len(currents), 1, figsize=(9, 4.5 * len(currents)), squeeze=False)
    for idx, i0 in enumerate(currents):
        ax = axes[idx, 0]
        sub = changes_df[changes_df["I0 (A)"] == i0]
        matrix = [[float(sub[sub["parameter"] == pv][m + "_pct"].iloc[0]) if not sub[sub["parameter"] == pv].empty else np.nan for pv in param_vals] for m in valid_metrics]
        
        mat = np.array(matrix)
        vmax = np.nanmax(np.abs(mat)) if np.isfinite(np.nanmax(mat)) else 1.0
        im = ax.imshow(mat, cmap="coolwarm", aspect="auto", vmin=-vmax, vmax=vmax)
        
        ax.set_xticks(np.arange(len(param_vals)))
        ax.set_xticklabels([str(pv) for pv in param_vals], fontsize=10)
        ax.set_yticks(np.arange(len(valid_metrics)))
        ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in valid_metrics], fontsize=10)
        ax.set_xlabel(param_display_name, labelpad=8)
        ax.set_title(f"I0 = {i0:g} A — % Change Heatmap", pad=10, fontsize=12)
        fig.colorbar(im, ax=ax, label="% change")
        
    fig.suptitle(f"{format_name(name)} — Heatmap of % Change vs Baseline", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, os.path.join(case_dir, "heatmap_pct.png"))

def plot_cooling_vs_power(metrics_per_param, param_display_name, case_dir, name):
    fig, ax = new_fig(f"{format_name(name)} — Cooling vs Power Trade-off", figsize=(9.5, 5.2))
    for val, mdf in metrics_per_param.items():
        if "dT_cold_peak_K" in mdf.columns and "net_power_W" in mdf.columns:
            ax.plot(mdf["net_power_W"], mdf["dT_cold_peak_K"], marker="o", label=f"{param_display_name}={val:g}", linewidth=2, markersize=6)
            
    ax.set_xlabel(r"Mean Net Power $P_{\text{net}}$ (W)", labelpad=10)
    ax.set_ylabel(METRIC_LABELS.get("dT_cold_peak_K", "dT_cold_peak_K"), labelpad=10)
    ax.set_title(f"{format_name(name)} — Cooling Performance vs. Power", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save_fig(fig, os.path.join(case_dir, "cooling_vs_power.png"))

def plot_response_times(metrics_per_param, param_display_name, param_unit, case_dir, name):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    for idx, (m, ylabel) in enumerate([("t_cool63_s", "63% Cooling Time (s)"), ("t_cool90_s", "90% Cooling Time (s)")]):
        ax = axes[idx]
        series_by_i0 = {}
        for val, mdf in metrics_per_param.items():
            if m in mdf.columns:
                for _, r in mdf.iterrows():
                    series_by_i0.setdefault(r["I0 (A)"], [[], []])
                    series_by_i0[r["I0 (A)"]][0].append(val)
                    series_by_i0[r["I0 (A)"]][1].append(r[m])
                    
        for i0, (pv, mv) in series_by_i0.items():
            order = np.argsort(pv)
            ax.plot(np.array(pv)[order], np.array(mv)[order], marker="o", label=f"I0={i0:g} A", linewidth=2, markersize=6)
            
        ax.set_xlabel(f"{param_display_name} ({param_unit})" if param_unit else param_display_name, labelpad=10)
        ax.set_ylabel(ylabel, labelpad=10)
        ax.set_title(ylabel, pad=10, fontsize=13)
        ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        
    fig.suptitle(f"{format_name(name)} — Response Times vs. {param_display_name}", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, os.path.join(case_dir, "response_times.png"))

def plot_efficiency_vs_param(metrics_per_param, param_display_name, param_unit, case_dir, name):
    fig, ax = new_fig(f"{format_name(name)} — Efficiency ($K/J$) vs {param_display_name}", figsize=(9.5, 5.2))
    series_by_i0 = {}
    for val, mdf in metrics_per_param.items():
        if "K_per_J" in mdf.columns:
            for _, r in mdf.iterrows():
                series_by_i0.setdefault(r["I0 (A)"], [[], []])
                series_by_i0[r["I0 (A)"]][0].append(val)
                series_by_i0[r["I0 (A)"]][1].append(r["K_per_J"])
                
    for i0, (pv, mv) in series_by_i0.items():
        order = np.argsort(pv)
        ax.plot(np.array(pv)[order], np.array(mv)[order], marker="o", label=f"I0={i0:g} A", linewidth=2, markersize=6)
        
    ax.set_xlabel(f"{param_display_name} ({param_unit})" if param_unit else param_display_name, labelpad=10)
    ax.set_ylabel(METRIC_LABELS.get("K_per_J", "K/J (K/J)"), labelpad=10)
    ax.set_title(f"Efficiency ($K/J$) vs. {param_display_name}", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save_fig(fig, os.path.join(case_dir, "efficiency_vs_param.png"))

def plot_global_summary_heatmap(output_root, metrics=DEFAULT_METRICS):
    """Standalone global summary heatmap across all cases and metrics."""
    sum_csv = os.path.join(output_root, "all_cases_summary.csv")
    if not os.path.exists(sum_csv):
        return
    df = pd.read_csv(sum_csv)
    if df.empty or "case" not in df.columns:
        return

    valid_metrics = [m for m in metrics if (m + "_max_abs_pct") in df.columns]
    if not valid_metrics:
        return

    matrix = []
    cases = df["case"].tolist()
    for _, row in df.iterrows():
        row_vals = [row.get(m + "_max_abs_pct", np.nan) for m in valid_metrics]
        matrix.append(row_vals)

    mat = np.array(matrix)
    if mat.size == 0 or np.all(np.isnan(mat)):
        return

    fig, ax = plt.subplots(figsize=(max(8, len(valid_metrics) * 0.9), max(5, len(cases) * 0.6)))
    im = ax.imshow(mat, cmap="Reds", aspect="auto")

    ax.set_xticks(np.arange(len(valid_metrics)))
    ax.set_xticklabels([METRIC_LABELS.get(m, m).split(" ")[0] for m in valid_metrics], rotation=45, ha="right", fontsize=11)
    ax.set_yticks(np.arange(len(cases)))
    ax.set_yticklabels(cases, fontsize=11)
    ax.set_xlabel("Performance Metrics", labelpad=10, fontweight="bold")
    ax.set_ylabel("Sweep Cases", labelpad=10, fontweight="bold")
    ax.set_title("Global Summary — Maximum Absolute % Change Across All Cases", pad=15, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Max |% change|")

    fig.tight_layout()
    save_fig(fig, os.path.join(output_root, "global_summary_heatmap.png"))

def run_sensitivity(sweep_folder=None, output_root=None, metrics=None):
    metrics = metrics or DEFAULT_METRICS
    here = os.path.dirname(os.path.abspath(__file__))
    sweep_folder = sweep_folder or next((c for c in [os.path.join(here, "csv"), os.path.join(here, "Sensitivity analysis auto")] if os.path.isdir(c)), None)
    output_root = output_root or os.path.join(here, "OUTPUT_auto", "csv_sensitivity")
    os.makedirs(output_root, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(sweep_folder, "*.csv"))) if sweep_folder else []
    if not csv_files:
        return output_root

    all_summaries = []
    s_transient_rows = []   # S of the transient quantities (as shown in the figure tables)
    case_names = []
    for path in csv_files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            df, meta, header, param_col = load_csv(path)
        except Exception:
            continue

        if param_col is None: continue
        case_names.append(name)

        param_unit = param_col.split("(")[1].split(")")[0].strip() if "(" in param_col and ")" in param_col else ""
        param_display_name = format_name(param_col.split("(")[0].strip() if "(" in param_col else param_col)
        
        groups = {float(val): sub for val, sub in df.groupby(param_col)}
        param_values = sorted(groups.keys())

        baseline_val = param_values[1] if len(param_values) > 1 else param_values[0]

        metrics_per_param = {val: compute_metrics(groups[val]).assign(parameter=val) for val in param_values}
        base_mdf = metrics_per_param[baseline_val]

        long_rows = []
        for val in param_values:
            for _, r in metrics_per_param[val].iterrows():
                long_rows.append({"parameter": val, "I0 (A)": r["I0 (A)"], **{m: r.get(m, np.nan) for m in metrics}})
        
        case_dir = os.path.join(output_root, name)
        os.makedirs(case_dir, exist_ok=True)
        pd.DataFrame(long_rows).to_csv(os.path.join(case_dir, "metrics_per_param.csv"), index=False)

        change_rows = []
        for val in param_values:
            if val == baseline_val: continue
            d_param_pct = ((val - baseline_val) / abs(baseline_val) * 100.0) if abs(baseline_val) > 1e-12 else np.nan

            for _, r in metrics_per_param[val].iterrows():
                i0 = r["I0 (A)"]
                brow = base_mdf[base_mdf["I0 (A)"] == i0]
                if brow.empty: continue
                b = brow.iloc[0]
                row = {"parameter": val, "baseline_parameter": baseline_val, "I0 (A)": i0}
                for m in metrics:
                    bv, cv = b.get(m, np.nan), r.get(m, np.nan)
                    row[m], row[m + "_base"] = cv, bv
                    if np.isfinite(bv) and abs(bv) > 1e-12 and np.isfinite(cv):
                        pct = (cv - bv) / bv * 100.0
                        row[m + "_pct"], row[m + "_abs"] = pct, cv - bv
                        row[m + "_norm_sens"] = (pct / d_param_pct) if (np.isfinite(d_param_pct) and abs(d_param_pct) > 1e-12) else np.nan
                    else:
                        row[m + "_pct"], row[m + "_abs"], row[m + "_norm_sens"] = np.nan, (cv - bv) if (np.isfinite(cv) and np.isfinite(bv)) else np.nan, np.nan
                change_rows.append(row)

        changes_df = pd.DataFrame(change_rows)
        changes_df.to_csv(os.path.join(case_dir, "changes_vs_baseline.csv"), index=False)

        summary = {"case": format_name(name), "parameter": param_display_name, "baseline": baseline_val}
        for m in metrics:
            if (m + "_abs") in changes_df.columns:
                s = changes_df[m + "_abs"].dropna()
                if not s.empty:
                    summary[m + "_max_abs_change"] = float(s.abs().max())
                    sp = changes_df[m + "_pct"].dropna() if (m + "_pct") in changes_df.columns else s.iloc[0:0]
                    if not sp.empty:
                        summary[m + "_max_abs_pct"] = float(sp.abs().max())
                    summary[m + "_mean_abs"] = float(s.mean())
                    if is_cold_metric(m):
                        summary[m + "_most_neg_change"] = float(s.min())
        all_summaries.append(summary)


        plot_tornado_grouped_by_current(changes_df, case_dir, name, param_display_name, chosen_metric="dT_cold_peak_K")
        plot_heatmap_pct(changes_df, metrics, case_dir, name, param_display_name)
        plot_cooling_vs_power(metrics_per_param, param_display_name, case_dir, name)
        plot_response_times(metrics_per_param, param_display_name, param_unit, case_dir, name)
        plot_efficiency_vs_param(metrics_per_param, param_display_name, param_unit, case_dir, name)

        series = extract_time_series(df, t_start=5.0)
        _bl = round(float(baseline_val), 9)
        _pvs = sorted({k[0] for k in series.keys()})
        _cur = sorted({k[1] for k in series.keys()})
        for _tm in ["hot", "cold", "power", "vol_cool2", "surf_min", "vol_heat2", "surf_max"]:
            for rec in _pct_records(series, _bl, _tm, _pvs, _cur):
                s_transient_rows.append({"case": format_name(name), "parameter": param_display_name,
                                         "quantity": _tm, **rec})
        transient_specs = [
            ("hot", r"$\Delta T_{\text{hot}}$ (K)", f"{format_name(name)} — Hot Side Temperature Shift"),
            ("cold", r"$\Delta T_{\text{cold}}$ (K)", f"{format_name(name)} — Cold Side Temperature Shift"),
            ("power", r"$P$ (W)", f"{format_name(name)} — Electrical Power"),
            ("vol_cool2", r"$V_{\text{cooled,2C}}$ (mm$^3$)", f"{format_name(name)} — Volume Cooled at 2°C"),
            ("surf_min", r"$\Delta T_{\text{surf,min}}$ (K)", f"{format_name(name)} — Surface Minimum Temperature Shift"),
            ("vol_heat2", r"$V_{\text{heated,2C}}$ (mm$^3$)", f"{format_name(name)} — Volume Heated at 2°C"),
            ("surf_max", r"$\Delta T_{\text{surf,max}}$ (K)", f"{format_name(name)} — Surface Maximum Temperature Shift"),
        ]
        for metric, ylabel, title in transient_specs:
            try:
                fig, ax = new_fig(title, figsize=(11.5, 5.2))
                plot_transient_multi(ax, series, baseline_val, metric, ylabel, title, param_name=param_display_name)
                save_fig(fig, os.path.join(case_dir, f"transient_{metric}.png"))

                fig, ax = new_fig(f"{title} — Difference", figsize=(11.5, 5.2))
                plot_transient_diff_multi(ax, series, baseline_val, metric, ylabel, f"{title} — Difference to Baseline", param_name=param_display_name)
                save_fig(fig, os.path.join(case_dir, f"transient_diff_{metric}.png"))
            except Exception as exc:
                plt.close("all")
                print(f"  [transient_{metric}] skipped: {type(exc).__name__}: {exc}")

    if all_summaries:
        sum_df = pd.DataFrame(all_summaries)
        sum_summary_path = os.path.join(output_root, "all_cases_summary.csv")
        sum_df.to_csv(sum_summary_path, index=False)

        if s_transient_rows:
            pd.DataFrame(s_transient_rows).to_csv(os.path.join(output_root, "all_cases_S_values_transient.csv"), index=False)

        # Generate global summary tornado grouped by current across all cases
        # Global tornado of S for every transient quantity (global_summary_tornado_S_<quantity>.png)
        for q, qlabel in TRANSIENT_QUANTITIES:
            try:
                plot_global_summary_tornado_S(output_root, q, qlabel)
            except Exception as exc:
                plt.close("all")
                print(f"  [global tornado S {q}] skipped: {type(exc).__name__}: {exc}")
        # Generate standalone global heatmap across all cases
        plot_global_summary_heatmap(output_root, metrics=metrics)

        cols = [c for c in ["case", "parameter", "baseline", "dT_cold_peak_K_max_abs_pct", "W_full_J_max_abs_pct", "net_power_W_max_abs_pct"] if c in sum_df.columns]
        print("\n### Data Summary\n")
        try:
            print(sum_df[cols].to_markdown(index=False))
        except ImportError:
            print(sum_df[cols].to_string(index=False))

    return output_root

if __name__ == "__main__":
    run_sensitivity()