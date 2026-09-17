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

---

## [2026-09-16] entry 3 — real EEA data, and the India route

Pushed to GitHub for the first time:
`https://github.com/Skandy1212/Federated-Spatiotemporal-AI-for-sensor-anomaly-detection`
(branch `master`).

### Station codes: verified, and several classes were wrong

`python -m src.data.station_metadata --refresh --print` downloaded the EEA
registry (26.9 MB). **All ten configured EoI codes are real**, which was lucky
rather than earned. But the registry contradicted five of my `station_class`
guesses, and those are now corrected from its own area × type:

| Station | I guessed | Registry says |
|---|---|---|
| ES1396A | urban_traffic | **urban_background** |
| ES1856A | urban_traffic | **urban_background** |
| ES1480A | urban_background | **urban_traffic** |
| ES1992A | suburban_background | **urban_background** |
| ES0559A | urban_industrial | **urban_traffic** |

Two things the registry export does *not* have: a station-name column (so the
`name:` fields in `configs/data.yaml` stay unverified labels) and any notion of
municipality except through `AirQualityStationNatCode`, whose first five digits
are the INE code — 08019 Barcelona, 08015 Badalona, 08211 Sant Adrià. My
`stations_in_city()` had been matching the string "Barcelona" against a numeric
column, which is why it returned zero rows while every code was in fact present.

### Only 7 of 10 stations are usable

ES2027A, ES1362A and ES0559A report no NO2 (the configured target), and
ES1362A's sampling point closed on 2017-12-31. They are also outside EEA's
"Barcelona" city filter. The working node set is therefore **7 stations**, all
in Barcelona municipality. Switching `target_pollutant` to PM10 would give 9,
which is worth revisiting when the graph layer wants more nodes.

### airbase 1.0.0 is not the API CLAUDE.md documents

§3.1 records the 0.x call. The installed 1.0.0 is
`client.request("Verified", "ES", poll=[...])` → `r.download(dir=...)`, and its
dataset vocabulary matters a great deal:

- `Historical` — data delivered **2002–2012**. My 1.x fallback was passing this,
  which would have quietly fetched the wrong era entirely.
- `Verified` — **E1a**. The only acceptable value.
- `Unverified` — E2a, LOCKED out by §2/§3.1.

`request()` takes no year arguments, so `year_from`/`year_to` are now a
post-download filter (`harmonize.filter_years`). Restricting to the Barcelona
urban area cut the download from ~2.2 GB to ~58 MB (61 files).

### EEA's Parquet download has no EoI code

The real schema is `Samplingpoint, Pollutant, Start, End, Value, Unit, AggType,
Validity, Verification, ResultTime, DataCapture, FkObservationLog`.
`Samplingpoint` looks like `ES/SP_08019043_10_47`, where `08019043` is the
*Spanish national* code. It relates to the EoI code only through the registry,
so `fetch_eea.sampling_point_to_eoi()` joins the two — parsing the string would
not work. Resolves 61/61 files.

### The finding that matters most: labels start in 2017

`label_coverage_by_year()` was written to check this and immediately earned its
place. EEA's Catalonia delivery carries **zero negative Validity codes before
2017** — 477,114 rows, every one flagged valid — then 2.3–4.3% faults every
year from 2017 onward. That is a reporting-practice change, not four
fault-free years.

Consequences, now baked into `configs/data.yaml` with the reasoning inline:
- `year_from: 2017` (was 2013). Including the earlier years drops the measured
  fault rate from **2.805% to 1.912%** and would teach the model that older
  data never fails.
- `year_to: 2024`. The service actually serves into 2025 — airbase's own docs
  claiming 2013–2023 are stale — but only 2 of the 7 stations have 2025
  coverage. 2024 is the last year complete across the node set.

Other measured facts: every row is `Verification=1`, and the only Validity
codes present are `1` and `-1`. **There is no `-99` anywhere in this node set**,
so §3.1's "single most valuable label" — government-recorded maintenance —
simply does not exist here. Any claim about distinguishing maintenance faults
specifically cannot be made from this data.

**Real fault rate = 0.02805.** This is the §5.5 target for synthetic injection
and the first genuinely novel measurement in the project. Per station it ranges
1.6% (ES1679A) to 4.6% (ES1992A).

### A pipeline bug real data exposed

ES1438A produced **zero** train and val windows. Cause: it added PM10 and PM2.5
instruments in 2024, and since a window requires every feature column finite,
all 48,892 pre-2024 training windows were silently dropped. The demo data could
never have surfaced this because every simulated station reports everything
from day one.

Fix: `_prune_sparse_features()` drops an *exogenous* pollutant whose coverage
inside the station's own training window is below `min_feature_coverage` (0.80).
A pollutant that was not being measured during training cannot be a training
feature. The target is never dropped — a station that cannot supply its own
target is excluded outright instead. ES1438A now has its full 48,892 windows.

### Physical screening on real data

Worth recording as a diagnostic: **every negative reading is also a
Validity-flagged fault** (984/984 at ES1679A, 1568/1568 at ES1856A, and so on).
That confirms §3.1's claim that `Concentration` is retained when `Validity` is
negative, which is the property the whole real-label approach depends on.

Also: the flatline rule flags **75% of SO2** at ES1856A and 72% at ES1992A.
Barcelona SO2 sits near the detection limit, so long runs of identical values
are normal instrument behaviour, not a fault. The `flatline_run_length: 6` rule
is miscalibrated for near-detection-limit pollutants. Not yet fixed — flagged
here because it would badly distort any SO2-based result.

### India / CPCB (out of §9 order, at the user's request)

§9 puts this in Phase 3, after Phases 1 and 2 report numbers. Done early
because it was asked for. It is acquisition and schema verification only —
nothing is trained on it, and no LOCKED constraint is relaxed.

Route: **OpenAQ**, which ingests CPCB as its India source. Of the §3.2 options
it is the only one giving multi-year *hourly* per-pollutant data without
scraping — `app.cpcbccr.com` is an undocumented internal API capped at about a
week per request, and `data.gov.in` serves AQI-level min/max/avg, which is
coarser than the structure this project needs.

**Schema VERIFIED against a real archive object**, closing the §11 open risk:
`location_id, sensors_id, location, datetime, lat, lon, parameter, units, value`.
Two corrections to the §5.1 sketch, which had assumed the data.gov.in export:
`datetime` already carries a UTC offset (localising it to Asia/Kolkata as §5.1
does would shift every reading by **+5:30**), and `parameter` is a lowercase
name, not a numeric id. `harmonize_cpcb` is now implemented rather than raising.

The unit guard earned its keep here too: `read_archive_object` was decoding the
gzip stream with the platform default (cp1252 on Windows) instead of UTF-8,
turning `µg/m³` into mojibake. `_canonicalise_units` refused it rather than
coercing — exactly what §3.1's unit trap is for.

**Blocked on one thing:** a free `OPENAQ_API_KEY`
(https://explore.openaq.org/register), needed *once* to map country → location
ids. The bulk history download afterwards needs no credentials. Every keyless
route was tested and failed: OpenAQ v3 returns 401, data.gov.in 400, OpenAQ v2
is retired (410), and WAQI's demo token ignores the requested city and returns
Shanghai regardless.

### Open items

1. ~~Station codes unverified~~ — **closed**, all 10 verified.
2. ~~CPCB schema unverified~~ — **closed** for the OpenAQ route.
3. EEA has no meteorology (entries 1–2, unchanged). `meteo.enabled` still false.
4. **New:** no `-99` maintenance labels exist in this node set.
5. **New:** the flatline screen is miscalibrated for near-detection-limit SO2.
6. **New:** `target_pollutant: NO2` gives 7 stations; PM10 would give 9.
7. **New:** Indian values need an OpenAQ API key.

---

## [2026-09-16] entry 4 — the −999 sentinel, and why Objective 4 cannot be evaluated on real EEA labels

**This is the most consequential finding so far. It contradicts a premise
CLAUDE.md states as fact, and it changes what Phase 2 can claim.**

### The claim, and the measurement

CLAUDE.md §3.1, on the `Concentration` column:

> "Present even when Validity is negative — **this is what makes real fault
> labels possible.**"

That is false for the Catalonia E1a delivery. Measured across 2017–2024:

| Pollutant | Validity=−1 rows | fraction equal to −999.0 |
|---|---|---|
| NO2 | 15,263 | **100.00%** |
| SO2 | 7,205 | **100.00%** |
| O3 | 5,109 | **100.00%** |
| PM10 | 103 | **100.00%** |
| PM2.5 | 103 | **100.00%** |

Every invalid row carries the no-data sentinel −999.0. **Not one carries a real
reading.** The relationship is also exactly biconditional: every sentinel is a
Validity fault, and every Validity fault is a sentinel.

### Consequence 1 (mechanical): every earlier real-data number was corrupt

−999 was flowing through as if it were a concentration. It dragged the NO2
scaler to **mean −1.41 / std 180.03** against a true **30.60 / 22.12** — so
every normalized feature, every lag, and every rolling statistic was distorted,
and the model spent its capacity trying to predict −999 spikes. That is why the
first run's RMSE (70–160 µg/m³) sat at roughly 6× its MAE (13–42) for a
pollutant whose clean mean is 30.6 and whose clean maximum is 233.

`harmonize._mask_sentinels` now converts the sentinel to missing, which routes
it into §6.2's imputation and marks it `is_imputed`. The Validity-derived label
is deliberately preserved: the row really was flagged faulty, we simply have no
reading to adjudicate. **The first `phase1_catalonia` run is superseded and
none of its numbers should be quoted.**

### Consequence 2 (conceptual): the central objective is not evaluable here

Objective 4 — the one §1 calls CENTRAL, the one the rest of the system exists
to serve — is to take an *anomalous reading* and decide whether it is a real
event or a sensor fault. That presupposes a reading exists to judge.

With this data there is none. Checked directly across all 7 stations:

| | test windows | fault targets | fault **and** imputed | fault **and** evaluable |
|---|---|---|---|---|
| all 7 stations | 69,527 | 7,106 | 7,106 | **0** |

Since §6.2 excludes imputed points from anomaly scoring — correctly, so the
detector never flags its own interpolation — **zero fault labels remain
evaluable**. The anomaly half of the report is necessarily empty.

And before the fix, when the sentinels were still present, the task was not
hard but vacuous: `value == -999` would score perfect precision and recall,
whereas the residual detector managed AUC-PR 0.67–0.83. **A trivial equality
check beat the model, because the task as posed was trivial.** Those apparently
respectable AUC-PR numbers were detecting a sentinel, not a fault.

### What this means

EEA's `Validity` flag marks *"data absent or invalid"*, not *"data present but
wrong"*. The interesting case — a plausible-looking reading that is
nevertheless incorrect, which is precisely what Ferrer-Cid (2022) and Gupchup
(2019) are about — **is not represented in these labels at all**.

This collides with LOCKED decision §2.1, which makes real `Validity` labels the
primary evaluation ground truth and demotes synthetic injection to "a
supplementary stress test, never the headline metric". On this dataset that
ordering cannot be honoured, because the primary ground truth contains no
evaluable positives.

Not resolved unilaterally — it touches a LOCKED decision, so it is the user's
call. The options, as put to them:

- **A.** Make rate-matched synthetic injection the primary evaluation for
  Objective 4, with the real `Validity` labels kept as an honest
  data-availability baseline. The injector already exists and is anchored to
  the *measured* real rate (0.02805), so this is not an arbitrary simulation.
  Requires amending §2.1.
- **B.** Source real labels that do contain plausible-but-wrong readings —
  co-located reference-vs-low-cost-sensor datasets are the usual route, and are
  what Ferrer-Cid (2022) used.
- **C.** Reframe Objective 4 around what EEA labels genuinely support.

Worth noting for whoever picks this up: the `demo_source.py` generator written
in entry 1 produces faults that *do* carry plausible corrupted values (stuck,
spike, bias, noise). That is the regime the project needs and, as it turns out,
the regime the real data does not supply.

### What survives

The forecasting half is unaffected by the conceptual problem and is now
computed on clean data. The federated architecture, the privacy contract, the
per-station discipline and the no-pooling reporting are all untouched. What
needs rethinking is specifically how Objective 4 gets evaluated.
