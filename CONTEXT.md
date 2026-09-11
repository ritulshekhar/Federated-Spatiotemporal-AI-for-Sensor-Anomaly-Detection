# CONTEXT.md — running build log

**Append-only.** Nothing in this file is ever edited or deleted, including
entries that later turn out to be wrong. When something changes, add a new
entry that says so and links back. The value of this file is that it records
what was believed *at the time*, not a tidied-up final story.

Format: one `## [YYYY-MM-DD] entry N — title` block per change, newest at the
bottom. Each entry says **what changed**, **why**, and **what it means for
whoever picks this up next**.

Companion documents:
- `CLAUDE.md` — the design contract. LOCKED decisions live there, not here.
- `README.md` — how to run things.

---

## [2026-09-09] entry 1 — repository scaffolding, configs, and the data layer

### What changed

Created the repository from scratch against `CLAUDE.md` §4. Directory layout
follows the spec exactly, with two additions noted under "Deviations" below.

**Environment**
- `.venv/` created with Python 3.13.7, gitignored.
- `requirements.txt` — numpy, pandas, pyarrow, pyyaml, requests, torch (CPU
  wheel via the PyTorch CPU index), scikit-learn, scipy, tqdm, pytest.
- `requirements-optional.txt` — `airbase` and `matplotlib`, split out so a
  failure to install the EEA download client does not block the core pipeline.

**Configs** (`configs/`)
- `data.yaml` — paths, EEA source settings, the 10-station Barcelona node set,
  every §6 preprocessing parameter, and a disabled `meteo:` block.
- `model.yaml` — node GRU, centralised GRU baseline, and (Phase 2) server GNN
  and decision-layer hyperparameters.
- `federation.yaml` — rounds, `aggregation.scheme: region_first` as the locked
  default per §5.7, and a machine-readable `privacy:` block listing exactly
  what may and may not leave a node.

**Utilities** (`src/utils/`)
- `paths.py` — repo-root resolution, cached YAML config loading, and
  `station_dir()`, which forces per-station output directories so "just
  concatenate the folder" is never the path of least resistance (§2.2).
- `logging_setup.py`, `seeding.py`.

**Data layer** (`src/data/`)
- `station_metadata.py` — downloads EEA's `PanEuropean_metadata.csv`, filters
  Barcelona, maps EEA area×type to our `station_class` vocabulary, and diffs
  the registry against `configs/data.yaml`. Also supplies station coordinates
  for Phase 2 distance-based graph edges.
- `fetch_eea.py` — three acquisition routes (airbase → local directory tree →
  demo), column-alias normalisation for EEA's newer Parquet schema, and a hard
  refusal to fetch anything but E1a.
- `harmonize.py` — the §5.1 common schema, `label_from_validity` verbatim from
  §3.1, unit canonicalisation that *raises* on a ug/m3 vs mg/m3 mismatch rather
  than coercing, and `real_fault_rate()` for §5.5 rate matching.
- `demo_source.py` — a simulated EEA-schema generator, described below.
- `fetch_meteo.py` — Open-Meteo ERA5 covariates, opt-in. See "Open items".
- `fetch_cpcb.py` — deliberately raises. Phase 3 only.

### Why

Phase 1 in `CLAUDE.md` §9 is "data & baselines, all of it, before any
federated/graph code". This entry is the data half.

Three choices are worth recording because they are judgement calls, not
transcriptions of the spec:

1. **Station EoI codes are marked UNVERIFIED in `configs/data.yaml`.** The ten
   Barcelona codes were written from general knowledge and have *not* been
   checked against a real EEA registry download. Rather than assert them as
   fact, `station_metadata.py` downloads the authoritative registry and
   `verify_configured_stations()` logs an error naming any code that does not
   exist. Run `python -m src.data.station_metadata --refresh --print` and
   correct the config before any reported number depends on it.

2. **`harmonize_eea` gives unparseable-Validity rows `label = None`, not
   `"trustworthy"`.** `label_from_validity` as written in §3.1 returns
   `"trustworthy"` for anything that is not -1 or -99, which would turn a
   missing flag into a positive assertion of validity. Missing ground truth is
   modelled as missing.

3. **Unit mismatches raise.** §3.1 calls the CO mg/m3 case a "unit trap".
   Coercing would hide it, so `_canonicalise_units` refuses.

### The demo source, and why it is safe

`src/data/demo_source.py` generates simulated station data in the EEA schema so
the pipeline and baseline ladder are runnable and testable without a download.
It is guarded three independent ways, all of which the reporting layer groups
on, so a simulated number cannot be averaged into a real one:

- `region = "demo"` (never `"catalonia"`)
- `label_source = "simulated"` (a **third** value, distinct from both `"real"`
  and `"synthetic"`)
- every station id is prefixed `DEMO-`

It reproduces only the structure the pipeline must survive: hourly cadence with
gaps, diurnal/weekly/seasonal cycles, a shared regional driver that creates
genuine cross-station correlation for the Phase 2 graph layer to find, real
high-pollution *episodes* that are not faults (the §8.2 false-positive control),
and Validity-flagged faults whose `Concentration` is still present — the
property that makes real EEA labels usable at all.

### Deviations from CLAUDE.md §4

- Added `src/utils/` (paths, logging, seeding) — not in the §4 tree.
- Added `scripts/` for runnable entry points, and `data/interim/` is gitignored
  but unused so far.
- Added `src/data/demo_source.py` and `src/data/fetch_meteo.py`, neither in the
  §4 tree. Both are documented above.

### Open items — flagged, not resolved (per §11)

1. **Station codes unverified.** As above. Highest-priority item.
2. **EEA has no meteorological variables.** Objective 3 asks for "pollutant +
   meteorological variables", but the E1a delivery contains pollutant
   concentrations and QA flags only. `fetch_meteo.py` sources them from
   Open-Meteo's ERA5 reanalysis instead, **opt-in and disabled by default**.
   Caveat worth carrying forward: ERA5 is a ~25 km grid, so all ten Barcelona
   stations fall in one or two cells. Met features will help the temporal
   forecast and will contribute essentially nothing to *spatial*
   discrimination. This is a real limitation of the objective as written, not
   of the implementation.
3. **`year_to: 2024`** assumes E1a for 2024 was published by 2025-09-30.
   Confirm against the EEA portal.
4. **CPCB schema unverified** — unchanged from §11; `harmonize_cpcb` and
   `fetch_cpcb.fetch` both raise.

---

## [2026-09-11] entry 2 — preprocessing pipeline, evaluation layer, node GRU, baseline ladder rungs 1–3

### What changed

**Preprocessing** (`src/preprocessing/`, all of §6)
- `indexing.py` (§6.1): long common schema → one contiguous hourly wide frame
  per station, with a per-pollutant gap report. Fixes the column convention
  every later stage uses (`<POL>`, `<POL>__validity`, `__verification`,
  `__label`, then `__is_imputed`, `__physical_flag`).
- `imputation.py` (§6.2): `is_imputed` mask built *before* filling. ≤3h gaps get
  linear interpolation, longer ones get seasonal-naive (same hour, nearest
  observed earlier day, then later day), and gaps over 168h stay NaN on purpose
  so windowing drops them.
- `validity_screen.py` (§6.3): negatives, out-of-range, flatline runs over 6,
  and PM2.5 > PM10. It only flags rows and never deletes them. It also logs how
  much the flags overlap with the real Validity faults, which gives an
  unsupervised floor to compare detectors against.
- `features.py` (§6.4): sin/cos calendar encodings, a weekend flag, lags
  t-1/t-24/t-168, and rolling mean/std over 6h and 24h. Everything is causal.
  Wind direction takes either degrees or a 16-point compass label.
- `normalize.py` (§6.5): `LocalScaler`, fitted on the train window only and
  saved to the station's own directory. It carries a `NEVER_TRANSMIT` marker,
  and `contains_normalization_statistics()` is the recursive check the Phase 2
  federated client will run against it.
- `windowing.py` (§6.6): chronological 70/15/15 boundaries, windows of 24h in
  and 6h out at stride 1. A window whose full span (inputs and targets) doesn't
  sit inside one split is dropped. Each window carries its target-side imputed
  mask, label, physical flag and Verification code.
- `fault_labels.py` (§6.7): rate-matched synthetic injection covering
  stuck_at, drift, spike, noise, dropout and bias. `target_rate` has no default.
  Injection never touches rows that already have real faults or were imputed,
  and it writes to separate `__synthetic_label` columns so the real labels are
  never overwritten.
- `pipeline.py`: runs the whole chain for each station separately and writes
  `data/processed/<station>/{train,val,test}.npz`, `scaler.json` and
  `manifest.json`, plus a run manifest per region. A station is excluded if
  less than 60% of its target was actually observed.

**Stage-order deviation (minimal, documented in `pipeline.py`):** §6 lists
normalization (6.5) before windowing/split (6.6), but fitting "on the training
window" requires knowing where that window ends. So split *boundaries* are now
computed between 6.4 and 6.5, and windowing itself still runs after 6.5.
Nothing else in the order moved.

**Evaluation** (`src/evaluation/`)
- `metrics.py`: RMSE/MAE/MAPE/bias in physical units. MAPE skips |y| < 1 ug/m3
  and counts how many points it skipped. Also precision/recall/F1/AUC-PR plus
  `fpr_on_extremes`, the Gupchup (2019) control. `anomaly_metrics` can't be
  called without the extreme-reading mask.
- `report.py`: every aggregation runs through `_grouped`, which raises
  `PoolingError` if `(region, station_class, label_source)` is missing from the
  grouping. No function returns a scalar across stations. `spread_table`
  reports min/median/max over stations and never a mean.
- Thresholds are keyed by **(model, station)**, not just station. The first
  draft keyed by station only, which would have scored every rung at
  persistence's operating point. Each rung produces residuals on a different
  scale, so each one needs its own threshold, calibrated on validation.

**Model** (`src/models/node_gru.py`): a GRU encoder with a GRU decoder (not a
dense head), teacher forcing annealed from 0.5 to 0, early stopping on val.
`encode()` / `embed()` are the only way hidden states get out, and that is the
Phase 2 split-learning hook. `payload_bytes()` supplies the §8.2 comms metric.

**Baselines** (`src/baselines/ladder.py`): rungs 1 (persistence), 2 (local GRU)
and 3 (centralised GRU). Rung 3 pools *windows* and shares weights across
stations, but normalization and reporting stay per station. Stations can report
different pollutants, so rung 3 uses the feature columns they all share rather
than zero-filling the missing ones. Rungs 4–6 raise `NotImplementedError`.
Every rung goes through one scoring path (`score_station`).

**Demo generator fix** (`src/data/demo_source.py`): PM2.5 is now derived as a
bounded fraction of PM10 instead of being drawn separately. The separate draw
had put PM2.5 above PM10 in ~10.5% of hours, so §6.3 was mostly flagging
generator artefacts. It's now ~3%, and all of that comes from pollution
episodes applied to each pollutant on its own. I also added multi-day outages
so the seasonal-naive and left-NaN imputation branches actually run.

### Verified vs. not yet verified

- **Verified on the demo region (simulated, not reportable):** harmonize → full
  §6 pipeline on all 10 stations, and rung 1 end to end with the grouped
  forecast and anomaly tables written to `outputs/phase1_demo/`. The linear,
  seasonal and left-NaN imputation branches all ran.
- **Not yet run end to end: rungs 2 and 3.** A 2-epoch smoke test failed before
  it reached any project code (`ModuleNotFoundError: src`) because the script
  ran from outside the repo root. That was an invocation problem, not a code
  result, so the GRU training path hasn't been exercised yet. Run
  `python -m src.baselines.ladder --region demo` from the repo root next.
- **Still no tests** under `tests/`.

### Open items (carried forward, plus new)

1. Station EoI codes unverified (entry 1, still the top priority).
2. EEA has no meteorology (entry 1).
3. **New:** the demo test fault rate differs a lot between stations (2.8%–6.8%),
   because chronological splits land on different fault episodes. This is
   expected. It's also a reminder that anomaly metrics from one test window are
   noisy, so real results will need confidence intervals, for example a block
   bootstrap over time.
4. **New:** rung 1 on the demo already gives `fpr_on_extremes` > 0 at 6h for
   several stations. That is the naive residual detector flagging real
   episodes, the exact behaviour rung 6 has to beat. The number itself doesn't
   mean anything yet (simulated data), but it confirms the control column
   works.
