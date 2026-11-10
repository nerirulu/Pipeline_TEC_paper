# Heatsink analysis pipeline — what the three processing scripts do

This document explains the three Python scripts that turn raw simulation and
experiment data into figures and metrics for the **Heatsink (Concise)** study,
and how they fit together.

The pipeline has two independent data sources that are only brought together at
the very end:

```
   SIMULATION (COMSOL)                         EXPERIMENT (IR camera + VC log)
   ─────────────────────                       ────────────────────────────────
   simulated_data.py                           experimental_data.py
        │  reads Config_heatsink.csv                 │  reads IR frames (.xlsx) + VC logs (.csv)
        │  reads Config_heatsink_surf.csv            │  subtracts frame-0 + border background
        │                                             │
        ▼                                             ▼
   OUTPUT/heatsink_metrics.csv                 OUTPUT/exp_outer_metrics.csv
   OUTPUT/heatsink_surface_traces.csv          (figures E1..E13 in OUTPUT/exp_outer_surface/)
   (figures H1..H20 in OUTPUT/heatsink/)                │
        │                                                │
        └───────────────────┬────────────────────────────┘
                            ▼
              comparison.py
              (figures C1..C10 + comparison_minmax_vs_energy.csv)
                            │
                            ▼
              make_readable_tables.py  ->  readable grouped tables
                                        (xlsx / csv / md / latex)
```

| # | Script | Data | Role |
|---|--------|------|------|
| 1 | `simulated_data.py` | **SIM** (COMSOL) | Single-model report: reads the heatsink export and plots/computes everything it supports. |
| 2 | `experimental_data.py` | **EXP** (IR) | Experimental-only report of the **outer** surface from the thermal camera. |
| 3 | `comparison.py` | **SIM vs EXP** | Overlays the two, computes errors, and writes the comparison table. |

Run them **in order** (1 and 2 can be run in parallel; 3 needs both):

```bash
python simulated_data.py                  # SIM  -> heatsink_metrics.csv + H1..H20
python experimental_data.py               # EXP  -> exp_outer_metrics.csv + E1..E13
python comparison.py                      # CMP  -> comparison_minmax_vs_energy.csv + C1..C10
python make_readable_tables.py                    # optional: readable grouped tables
```

---

## 1. `simulated_data.py` — the SIMULATION side

**Purpose.** A single-model report for the Heatsink (Concise) COMSOL export. It
reads the simulation results and produces every figure and metric the export
supports — temporal, spatial (volume + 2D surface) and energetic — organised
into figure families **H1 … H20**.

**Inputs (from `Final files/`):**
- `Config_heatsink.csv` — accumulated time probes: hot/cold point temperatures,
  boundary heat-flux, TEC voltage `V1`, and the heated/cooled **volume** at
  thresholds 0.5 / 1 / 2 / 5 °C.
- `Config_heatsink_surf.csv` — the **outer-surface** temperature field on a
  plane (x = 1.4225 mm, y −13…17, z −15…15). This is a *field*, not a point.
  (The original script comments call it "inner", but the exported plane is the
  **outer** surface — the one the IR camera also sees.)

**What it computes / produces:**
- **Temporal (probe) metrics** — peak/end/60 s ΔT, peak time, cooling time
  constants (τ63, τ90), back-to-normal time, overshoot.
- **Volume (spatial probe) metrics** — cooled/heated domain volume at each
  ΔT threshold (figures H7–H9).
- **Outer-surface field metrics (`surf_*`)** — field min/max/mean, spread, std,
  recovery fractions. A **Delaunay triangulation** of the exported surface mesh
  turns the field into 2D **areas** (mm²) at the 0.5/1/2/5 °C thresholds.
- **Energy metrics** — electrical energy `W_full` / `W_drive`, areal heat energy,
  dose, and efficiencies (K per J, J per K).
- **Surface traces** — the full per-time ΔT field traces written to
  `OUTPUT/heatsink_surface_traces.csv`.

**Outputs:**
- `OUTPUT/heatsink_metrics.csv` — one row per drive current (0.1–0.5 A),
  ~130 columns (probe + `surf_*` + `vol_*` + `area_*` blocks).
- `OUTPUT/heatsink_surface_traces.csv` — the outer-surface min/max/mean vs time.
- `OUTPUT/heatsink/H1 … H20*.png` + `figure_index.md`.

> **House rule.** The Cold/Hot/Min/Max *point probes* are reference-only. They
> are **never** compared with experimental data; only the outer-surface *field*
> (`surf_*`) is comparable with a surface measurement.

---

## 2. `processing_experimental_outer_surface.py` — the EXPERIMENT side

**Purpose.** The experimental-only counterpart. It analyses the **outer surface**
of the device from the thermal (IR) camera footage and produces figure family
**E1 … E13** plus a metrics table. It deliberately contains **no** COMSOL /
simulation data — it is the pure experimental counterpart.

**Inputs:**
- IR frames — `.xlsx` grids of temperatures, one file per frame
  (`Castell22C_7Oct{30,40}mm/Excel files/…`), grouped per test `TESTnnn`.
- VC (voltage/current) logs — `.csv` per test, giving the electrical drive and
  therefore the **input energy** `E = Σ V·I·dt`.

**What it does, step by step:**
1. **Frame cache** — converts the `.xlsx` frames to `.npz` once (~8 min), then
   reloads in seconds (`OUTPUT/_frame_cache_7oct/`).
2. **Background subtraction** — every frame is differenced against frame 0, then
   a border-region background is removed (`BG_FRAC = 0.05`).
3. **Drive window** — from the VC log, the ON window and current `I0` are found
   (phase-based, or current ≥ 90 % of peak). **Energy** is integrated over the
   active drive window.
4. **Metrics per current** (mean over the repeated runs): peak/end ΔT (cold =
   frame min, hot = frame max), peak-cooling time, time constant τ (exponential
   fit), initial cooling rate (linear fit), overshoot, input energy ± sd,
   **cooled/heated area** at each threshold (IR pixel count × (mm/px)²), and
   cool/heat ratio.
5. **Figures E1 … E13** — ROI grids, focus/overlay transients, normalised
   response, switch-off, area-vs-current, balance, surface maps, area metrics,
   stats, energy/efficiency, and min/max-vs-energy.

**Outputs:**
- `OUTPUT/exp_outer_metrics.csv` — one row per drive current (0.1–0.5 A),
  ~30 columns.
- `OUTPUT/exp_outer_surface/E1 … E13*.png` + `index.md`.

**Key configuration** (top of the file): drive 0–25 s, run to 60 s; mm/px =
0.0999 (30 mm) / 0.1119 (40 mm); thresholds 0.5/1/2/5 °C; frame rate 1 Hz.


---

## 3. `comparison.py` — SIM vs EXP

**Purpose.** Bring the two independent sources together: overlay the simulation
**outer-surface field** against the experiment **outer surface** (IR), compute
agreement errors, and write the master comparison table. Figure family
**C1 … C10**.

**Inputs (produced by scripts 1 and 2):**
- `OUTPUT/heatsink_metrics.csv` (`surf_*` columns) + `heatsink_surface_traces.csv`
- `OUTPUT/exp_outer_metrics.csv`

**What it does:**
1. **Matches currents** — pairs each SIM current `I0_A` with the nearest EXP
   `I0 (A)` (tolerance 0.02 A).
2. **Aligns times** — both sides are put on a common "time into pulse" axis.
   SIM is compared using the outer-surface **field** traces (`sim_surface_trace`);
   EXP uses the mean of its repeated IR runs with a ±std band.
3. **Computes errors** — for each headline metric it writes `sim`, `exp`,
   `err = exp − sim` and `relerr (%)`. The comparison **energy** is the
   drive-window energy on both sides (`E_sim_drive` vs `E_exp`).
4. **Figures C1 … C10** — min/max vs energy (peak & end), per-current trace
   overlays, temporal overlay, normalised dynamics, switch-off/overshoot,
   energy/electrical, temperature-vs-current, spatial areas, and a
   uniformity + numeric scorecard.

**Outputs:**
- `OUTPUT/comparison_heatsink_outer/comparison_minmax_vs_energy.csv`
  (~51 columns: sim | exp | err | relerr blocks + `*_sim_probe` references).
- `OUTPUT/comparison_heatsink_outer/C1 … C10*.png` + `index.md`.

> **House rule (repeated on every figure).** SIM = outer-surface *field*;
> EXP = outer-surface IR frame min/max; energy = drive window only. The direct
> point probes are **never** compared with the experiment — they survive only as
> `*_sim_probe` reference columns and are excluded from every error column.


---

## How the three connect

- **Scripts 1 and 2 are independent** — one is simulation, one is experiment.
  They only share the drive protocol and the ΔT thresholds, so their outputs can
  be produced in any order (or in parallel).
- **Script 3 depends on both** — it reads the two metrics CSVs, so re-run
  scripts 1/2 before it if their inputs change.
- **The comparable quantity is the surface field.** SIM's `surf_*` (outer
  surface) is the only simulated block allowed to meet EXP's outer-surface IR
  min/max — both sides are the **same outer surface**. Everything else (point
  probes, volume) stays on its own side.
- **`make_readable_tables.py`** (added later) reads all three metrics CSVs and
  re-labels/groups them into publication-ready tables in
  `OUTPUT/readable_tables/` — it does not re-run any analysis.

---

## Quick reference — files produced

| Script | Figures | Main CSV |
|--------|---------|----------|
| `simulated_data.py` | `OUTPUT/heatsink/H1..H20.png` | `OUTPUT/heatsink_metrics.csv` (+ `heatsink_surface_traces.csv`) |
| `experimental_data.py` | `OUTPUT/exp_outer_surface/E1..E13.png` | `OUTPUT/exp_outer_metrics.csv` |
| `comparison.py` | `OUTPUT/comparison_heatsink_outer/C1..C10.png` | `OUTPUT/comparison_heatsink_outer/comparison_minmax_vs_energy.csv` |
| `make_readable_tables.py` *(helper)* | — | `OUTPUT/readable_tables/…` (xlsx/csv/md/latex) |

