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

---

## [2026-10-06] entry 5 — Option A implemented, and the first honest Objective 4 numbers

The project owner chose **option A** from entry 4: rate-matched synthetic
injection becomes the primary Objective 4 ground truth. CLAUDE.md §2.1 is
struck through with its original reasoning preserved, §2.1a added, and
objective 5 amended. Commit `2289719`.

### What was built

- `fault_labels.real_episode_durations()` — episode *durations* are now sampled
  from the real `Validity` episode-length distribution (264 real episodes at
  ES1679A, median 2h). EEA's flags genuinely evidence *when* an instrument
  failed and *for how long*; only the corrupted value is synthesised.
- Budget is split **by row count per fault type**, not per episode. Choosing a
  type per episode let drift take 63% of injected rows and left spike with
  1.5%, because drift's 72h floor sits far above the real 2h median.
- `windowing` carries synthetic labels and fault types in arrays reachable only
  through `labels_for(source)`, so real and synthetic cannot be merged by
  accident.
- Clean and injected builds live in separate variant directories. One is the
  forecasting reference, the other the Objective 4 ground truth.

Verified: injection rate 0.0280 against a target of 0.0280, 323–324 rows per
subtype, 37 tests passing.

### The numbers (`outputs/phase1_catalonia_injected/`)

Median across stations, by station class. Recall / AUC-PR / `fpr_on_extremes`:

| Model | class | h=1 recall | h=1 fpr_ext | h=6 recall | h=6 fpr_ext | best AUC-PR |
|---|---|---|---|---|---|---|
| persistence | urban_bg | 0.178 | 0.017 | 0.216 | **0.500** | 0.185 |
| local GRU | urban_bg | 0.219 | 0.106 | 0.286 | 0.370 | 0.322 |
| centralised GRU | urban_bg | 0.219 | 0.148 | 0.257 | 0.481 | 0.300 |
| persistence | urban_traffic | 0.361 | 0.011 | 0.418 | 0.348 | 0.370 |
| local GRU | urban_traffic | 0.386 | 0.117 | 0.481 | **0.631** | 0.508 |
| centralised GRU | urban_traffic | 0.399 | 0.117 | 0.484 | 0.520 | 0.504 |

**Three things worth keeping.**

**1. The task is genuinely hard now.** Recall 0.15–0.50, AUC-PR 0.15–0.51. The
earlier apparent AUC-PR of 0.67–0.83 was an artefact of the −999 sentinel
(entry 4) and is superseded. These are the first numbers that measure the
actual research question rather than a sentinel lookup.

**2. `fpr_on_extremes` is the headline, and it is bad.** At 6h the naive
residual detector misclassifies **up to 66.7%** of genuine high-pollution
readings as sensor faults (local GRU, urban traffic; median 0.631). Gupchup et
al. (2019) reported up to 45% and §8.2 banned recall-alone metrics because of
it — this reproduces that failure and exceeds it. Any 6h recall figure quoted
without this column beside it would be actively misleading.

**3. Better forecasting makes the false-positive problem worse, not better.**
At h=1 urban background, going from persistence to the local GRU lifts recall
0.178 → 0.219 but takes `fpr_on_extremes` 0.017 → 0.106, a six-fold rise. A
sharper forecaster produces larger residuals on real events too, so the
residual threshold alone cannot separate "unusual" from "wrong". That is
precisely the gap `spatial_agreement()` exists to close, and it is now a
quantified target rather than an argument: **rung 6 has to hold recall while
cutting `fpr_on_extremes`.**

### Caveats on these numbers — read before quoting them

- **Faults were injected into all three splits, not just test** (ES1679A: 2.9%
  train / 2.4% val / 2.6% test). The forecaster therefore trained on corrupted
  values. That is less realistic than it looks: in deployment you train on
  *verified* history, where EEA already replaced faults with −999, and run live
  on unverified readings where faults arrive as plausible numbers. A
  clean-train / injected-val / injected-test build matches deployment better,
  and may also be flattering the detector here, since a forecaster that has
  seen stuck and drifting values predicts them better and so shrinks the very
  residuals meant to expose them. **Not yet changed — open item.**
- **No per-subtype breakdown yet.** The injector balances all six subtypes and
  `fault_type` travels through the windows, but `make_records` drops it, so the
  report cannot group by it. The aggregate recall almost certainly hides a wide
  spread: spike/dropout/bias produce large immediate residuals, while drift is
  tracked by the forecaster and is expected to be near-undetectable. Drift is
  the subtype §6.7 and D'Elia et al. (2024) care most about. **Open item.**
- Forecast RMSE in this report (14.65–18.47 at h=1) is computed on *corrupted*
  test targets and is not comparable with the clean run's 7.1–9.7. Use
  `outputs/phase1_catalonia/` for forecasting accuracy.
- Per-subtype *precision* will never be definable: a false alarm on a healthy
  reading has no fault type. Only recall decomposes by subtype.

### Open items

1. Clean-train variant (above).
2. Per-subtype recall table (above).
3. CLAUDE.md still states the superseded rule in four places that now
   contradict §2.1a: the §3.1 `Concentration` row, §6.7 step 7, §8.2's anomaly
   bullet, and §9 Phase 2 step 6.
4. Persist per-reading records to disk, so later breakdowns never require
   retraining.
5. Carried forward: no meteorology; no −99 labels in this node set; flatline
   screen miscalibrated for near-detection-limit SO2; Indian values still need
   an OpenAQ API key.

---

## [2026-10-06] entry 6 — accuracy work: one win, one negative result, and the finding that reframes Phase 2

Four stale rules in CLAUDE.md fixed first (§3.1's `Concentration` row and the
`Validity` row, §6.7 step 7, §8.2's anomaly bullet, §9 Phase 2 step 6). All
four still stated the pre-§2.1a ordering; anyone reading them alone would have
got the wrong rule.

### Infrastructure: scoring is now decoupled from training

A ladder run saves `records.parquet` — truth, forecast, recent local
volatility, imputation mask, label and fault subtype, for **both** validation
and test. `src/evaluation/rescore.py` recalibrates and re-evaluates from it.
A new scoring rule now costs about a second instead of a ~50-minute retrain,
which is what made everything below measurable rather than arguable.

### Fix: training split is now clean

Injection is restricted to validation and test (`eligible_after=train_end`).
Verified: train 0.0%, val ~3.0–3.4%, test ~2.2%. This matches deployment —
train on verified history, run live on unverified readings — and removes the
effect where a forecaster trained on corrupted values learns to track the very
faults it is meant to expose.

### Win: threshold policy

Every score mode × threshold method, median across stations:

| horizon | method | recall | precision | F1 | fpr_extremes | worst station |
|---|---|---|---|---|---|---|
| 6h | mad | 0.450 | 0.133 | 0.202 | 0.850 | 0.945 |
| 6h | quantile 0.98 | 0.320 | 0.288 | 0.284 | 0.520 | 0.691 |
| 6h | val_f1_sweep | 0.259 | 0.416 | 0.310 | 0.320 | 0.547 |
| 6h | **val_fpr_extremes** | 0.216 | **0.595** | **0.326** | **0.164** | **0.306** |

`val_fpr_extremes` — the lowest threshold keeping false positives on genuine
extreme *validation* readings under 10% — gives the best F1 at 3h and 6h *and*
cuts `fpr_on_extremes` by two-thirds. Worst-station 6h false-positive rate
falls from 0.691 to 0.306. It is now the default. `mad` is catastrophic (0.850)
and should never be used. At 1h the fixed quantile still edges it on F1 (0.343
vs 0.296), at six times the false-positive rate — not a trade worth taking
given §8.2.

### Negative result: volatility scaling did not work

The hypothesis (entry 5) was that dividing the residual by the recent local
volatility would suppress false positives on genuine episodes, since real
episodes are preceded by turbulent hours. It does reduce `fpr_on_extremes` at a
matched threshold rule — but it degrades the threshold-free ranking at every
horizon:

| AUC-PR | 1h | 3h | 6h |
|---|---|---|---|
| absolute | **0.274** | **0.302** | **0.297** |
| volatility | 0.204 | 0.256 | 0.264 |

So the information it removes is worth more than the noise it suppresses. The
default is back to `absolute`; the mode is kept selectable so the result stays
reproducible rather than merely asserted. The false-positive problem is better
solved at the threshold, which costs no ranking quality.

### The finding that matters: four of six fault types are invisible

Recall per subtype (local GRU, absolute + val_fpr_extremes):

| subtype | 1h | 3h | 6h |
|---|---|---|---|
| spike | 0.590 | 0.766 | 0.755 |
| noise | 0.403 | 0.387 | 0.329 |
| dropout | 0.018 | 0.028 | 0.025 |
| bias | 0.000 | 0.009 | 0.011 |
| stuck_at | 0.000 | 0.000 | 0.000 |
| drift | 0.000 | 0.003 | 0.003 |

**The aggregate recall of ~0.2–0.3 is almost entirely spikes and noise.**
Drift, bias, stuck-at and dropout are essentially undetected. Entry 5 predicted
drift would be hard; the truth is worse and the cause is more general than
expected.

**Why, and it is structural rather than a tuning problem.** The faulty values
enter the model's *own 24-hour input window*. A stuck sensor presents a flat
history, so the GRU predicts flat and matches it. A bias offset shifts the lags
and rolling means, so the forecast shifts with it. A drift is tracked. Dropout
values near zero are continued once they are inside the window. Only faults
that are abrupt *relative to the input window* — spikes, and the variance
change of noise — ever produce a residual. Persistence (rung 1) fails the same
way, and for the same reason: it reads the last observed value, which is itself
corrupted.

**No improvement to the forecaster can fix this.** A better forecaster tracks
the fault more faithfully and makes the residual smaller. That is the opposite
of what is wanted, and it is consistent with entry 5's other finding that the
GRU raised `fpr_on_extremes` over persistence.

### What this does to Phase 2

It strengthens the case for rung 6 considerably, and sharpens what it has to
do. Cross-station spatial agreement does not depend on the station's own
history, so it is not corrupted by the fault: a station that drifts, sticks or
sits at a constant offset while its neighbours do not should be visible in
exactly the regime where residuals are blind. The graph layer is no longer a
hoped-for accuracy increment — it is the only mechanism on the table that can
catch four of the six fault types. `spatial_agreement()` now has a concrete,
quantified target: **non-zero recall on drift, bias, stuck-at and dropout**,
against a residual baseline of 0.000–0.052.

A cheaper intermediate worth trying first, because it needs no federation: a
reference that the fault cannot contaminate. A seasonal-naive comparison
against the same hour one week earlier, `|y(t) − y(t−168)|`, is immune to a
fault shorter than a week, so it should see persistent faults that the model
residual cannot. It needs `y(t−168)` carried into the records, which the
pipeline does not do yet.

### Open items

1. Seasonal-naive reference score (above) — cheap, no federation needed.
2. Meteorology still disabled; the single largest forecasting lever untried.
3. Per-subtype precision remains undefinable, by construction.
4. `is_extreme` is a quantile proxy, not a confirmed event list (§11).
5. Carried forward from entry 5: records now persisted (done), clean train
   (done), per-subtype table (done).

---

## [2026-10-12] entry 7 — the contamination hypothesis was wrong, and what actually blocks detection

Entry 6 proposed that four of six fault types are invisible because the faulty
values sit inside the model's own 24h input window, so the forecast follows the
fault. `src/evaluation/temporal_reference.py` tested that with a reference the
fault cannot reach — the median of `y(t−24k)`, k=1..7. Daily lags rather than a
single week-ago value, because the diurnal cycle is near-universal while the
weekly cycle is specific to traffic-driven cities, and India is the
generalization target (§5.8). Parameter-free, per-station scaled, and rebuilt
from the *injected* windows so the reference can itself be faulty as often as
it would be live.

**The hypothesis was wrong.** The uncontaminated reference fails on exactly the
same four subtypes (local GRU, h=1): spike 0.694, noise 0.199, bias 0.005,
drift 0.000, dropout 0.000, stuck-at 0.000. Contamination is not the binding
constraint.

### What is actually binding: the faults are inside the normal spread

Per-hour score distributions on test, h=1, local GRU:

| | threshold | normal median | normal p90 | normal p99 |
|---|---|---|---|---|
| model residual | 2.47 | 0.17 | 0.59 | 1.42 |
| multi-lag reference | 3.34 | 0.36 | 1.22 | 2.45 |

| subtype | residual median | % over threshold | reference median | % over threshold |
|---|---|---|---|---|
| spike | 3.24 | 57.2 | 5.29 | 65.3 |
| noise | 2.14 | 41.4 | 1.83 | 22.9 |
| bias | 0.31 | 0.0 | 1.39 | 0.7 |
| dropout | 0.26 | 1.1 | 0.81 | 0.0 |
| drift | — | — | 0.60 | 0.8 |
| **stuck-at** | **0.12** | 0.0 | 0.32 | 0.0 |

The four invisible subtypes produce scores **inside the distribution of normal
readings**. Bias sits at the reference's 92nd percentile of normal; drift and
stuck-at sit near its *median*. No threshold on a per-reading score can separate
them, whatever the reference — that is why swapping the reference changed
nothing. The limit is the signal, not the comparator.

**The sharpest case: stuck-at scores 0.12 against a normal median of 0.17.** A
frozen sensor is *more* predictable than a working one, so its residual is
smaller than normal. An upper threshold cannot catch it in principle; the
detector is looking in the wrong direction. Catching it needs a test for
*abnormally low* variability, which is close to free and should be near-perfect
(a stuck sensor has exactly zero variance over its episode).

Also worth noting: dropout's absolute deviation is small because dropouts are
not rare at night, when true NO2 is ~10 µg/m³ and dropping to zero is a ~0.45σ
move, not the ~1.4σ a daytime dropout would give.

### Accumulating evidence over time helps, for some subtypes

Rolling mean of the score over k hours, then recalibrate (residual score,
urban background):

| subtype | k=1 | k=6 | k=12 | k=24 |
|---|---|---|---|---|
| bias | 0.021 | 0.048 | 0.063 | **0.153** |
| dropout | 0.065 | 0.069 | 0.106 | **0.249** |
| stuck-at | 0.000 | 0.004 | 0.013 | 0.026 |
| drift | 0.000 | 0.069 | 0.000 | 0.000 |
| noise | 0.634 | **0.779** | 0.742 | 0.667 |
| spike | **0.716** | 0.661 | 0.616 | 0.502 |

Bias improves 7× and dropout 4× — a persistent offset that is invisible in one
hour becomes visible over a day. But spike *degrades* (0.716 → 0.502), because
averaging dilutes a one-hour excursion. **No single timescale works for all
subtypes**, which means the decision layer needs several, not a better choice
of one.

Drift stays at 0.000 at every timescale. Its episodes are ≥72h by construction
and ramp linearly to 0.8–2.5σ, so the per-hour increment is ~0.03σ — below the
noise floor at every window length tried, and the mean offset over any 24h slice
is still small. Drift needs a *calibration* test over weeks, not an anomaly
score over hours.

### What this means for Phase 2 — read before building rung 6

This is the main reason the diagnostic was worth 30 minutes. **Spatial agreement
computed per reading will hit the same wall.** A 1σ offset is within normal
*spatial* variability between stations just as it is within normal temporal
variability, so a per-hour neighbour comparison will not separate bias, drift or
stuck-at either. Building the graph layer and scoring it per reading would most
likely reproduce the table above and look like a failure of the graph idea, when
it would actually be a failure of the timescale.

The design that follows from the measurements is a small ensemble over
timescales and statistics, with spatial agreement supplying the reference where
it is genuinely needed:

1. **Per-reading residual** — keep for spike and noise, where it already works.
2. **Low-variance test** — for stuck-at, which no upper threshold can reach.
   Cheap, and should be near-perfect.
3. **Windowed accumulation (~24h)** — for bias and dropout, worth 4–7× recall.
4. **Long-window calibration comparison against neighbours (days to weeks)** —
   for drift. This is where cross-station agreement is irreplaceable: a drifting
   station cannot detect its own drift from its own history at any timescale,
   because every reference it owns drifts with it. A neighbour's does not.

Point 4 is the strongest argument yet for the §1.2 research gap — not that
spatial reasoning improves accuracy, but that for drift it is the *only*
available signal under the privacy constraint.

### Open items

1. Implement the four-part decision layer above; rung 6 should be evaluated at
   the timescale each subtype needs, not one global per-reading threshold.
2. Drift detection is unsolved and may stay that way — report honestly if so.
3. Meteorology still disabled.
4. India transfer still untestable (OpenAQ API key).

---

## [2026-10-12] entry 8 — ceiling test: neighbours *can* see what a station cannot

`src/evaluation/spatial_reference.py` tests the assumption the project rests on
(§1.2, §7.4), before any of rung 6 gets built. Deliberately the easy version —
raw neighbour readings, centralised, no privacy constraint, no embedding
compression, i.e. Ferrer-Cid et al. (2022). Whatever it scores is a **ceiling**
for rung 6, and the gap between the two will be the measured price of the
privacy constraint.

Each station's series is z-scored on its own training window (§6.5 — no shared
scaler), and its spatial deviation is the distance from the **median** of the
other six at that hour. Median, not mean: at a 2.8% fault rate a neighbour is
occasionally faulty itself.

### Recall by subtype, by accumulation window

| subtype | own residual | 1h | 6h | 24h | 72h | 168h |
|---|---|---|---|---|---|---|
| **drift** | **0.004** | 0.014 | 0.103 | **0.149** | 0.030 | 0.012 |
| **bias** | **0.024** | 0.040 | **0.261** | 0.235 | 0.071 | 0.101 |
| dropout | 0.081 | 0.046 | 0.144 | **0.228** | 0.026 | 0.037 |
| stuck-at | 0.000 | 0.015 | 0.015 | 0.009 | **0.060** | 0.019 |
| noise | **0.674** | 0.194 | 0.232 | 0.168 | 0.081 | 0.040 |
| spike | **0.704** | 0.512 | 0.423 | 0.282 | 0.123 | 0.074 |

**The central claim survives its first real test.** Drift goes from 0.004 to
0.149 — a 37× improvement on the subtype that was previously untouchable at
every temporal timescale. Bias goes 0.024 → 0.261, dropout 0.081 → 0.228. A
neighbour genuinely sees what the station cannot, and entry 7's reasoning for
why (every reference a drifting station owns drifts with it; a neighbour's does
not) holds up in measurement.

**And the two signals are complementary, not competing.** The station's own
residual remains far better for the transient faults — spike 0.704 vs 0.512,
noise 0.674 vs 0.232. Neither source dominates, which is exactly the shape
entry 7's four-part design predicted, now with each part's role measured rather
than assumed.

**The accumulation window matters and is subtype-specific**: 6h for bias, 24h
for drift and dropout, 1h for spike. Beyond 72h everything degrades, because
the trailing mean washes out episodes that are shorter than the window.

### The caveat, stated plainly: precision is poor

| window | recall | precision | AUC-PR | fpr_on_extremes |
|---|---|---|---|---|
| 1h | 0.169 | 0.593 | 0.274 | 0.087 |
| 6h | 0.262 | 0.270 | 0.163 | 0.060 |
| 24h | 0.203 | 0.096 | 0.075 | 0.150 |

At the 24h window where drift becomes visible, **precision is 0.096** — about
nine in ten flags are false. Against a 2.8% base rate that is still 3.4× better
than guessing, but it is nowhere near usable on its own. At 1h the spatial
score is competitive with the residual (AUC-PR 0.274 for both) but catches
different subtypes.

So the honest reading: a neighbour comparison **recovers signal that does not
otherwise exist**, at low precision. It is a component, not a detector. The
ensemble has to combine it with the residual rather than replace one with the
other, and the per-subtype windows above say how.

One genuine positive: `fpr_on_extremes` stays low (0.060–0.150) across windows.
A regional pollution episode lifts every station together, so differencing
against neighbours cancels it — which is the §8.2 failure mode the whole
project is trying to avoid, and the spatial signal is structurally resistant to
it in a way a temporal one is not.

### Limits of this ceiling

- **Seven stations, all inside Barcelona**, a few km apart and strongly
  correlated — close to the best case for neighbour comparison. A sparser
  network would do worse, which matters for the India transfer (§5.8) where
  station spacing and local source mix both differ.
- Stuck-at is barely helped (0.000 → 0.060). It still needs the low-variance
  test from entry 7; no reference-based score will find it.
- This is the raw-data ceiling. Rung 6 must reproduce it from **compressed
  embeddings**, and §11 flags that as untested. The gap is the experiment.

### Open items

1. Build rung 6 to the four-part design, evaluated per subtype at the window
   each one needs — never one global per-reading threshold.
2. Add the low-variance test for stuck-at (cheap, nothing else catches it).
3. An ensemble combining residual + spatial, rather than either alone.
4. Rungs 4 and 5 still to build (§8.1 order).
5. Carried: meteorology disabled; India untestable without an OpenAQ key.

---

## [2026-10-12] entry 9 — rung 5: federation costs almost nothing

Phase 2 begins. `src/federated/{client,server,aggregation_weights}.py`
implement §7.1–7.3 and §5.7: `region_first` weighting (locked default, which
with one region reduces to uniform — all seven nodes weighted 0.143), FedAvg
over GRU weights, and early stopping on mean client validation loss.

The privacy allowlist in `configs/federation.yaml` is enforced by
`check_upload` on **every upload of every round**, not just in tests. It
rejects any field the config does not permit and runs
`contains_normalization_statistics` over the payload. No violations across the
run.

Weight averaging needs one architecture, so clients share the **15 feature
columns common to all seven stations**, against the 15–31 a local model may
use. That is a real cost of weight sharing, and it makes the result below more
interesting rather than less.

### Results (injected variant; all rungs rescored uniformly with absolute + val_fpr_extremes)

Forecast RMSE, µg/m³ NO2, median across stations:

| model | 1h | 3h | 6h |
|---|---|---|---|
| persistence | 17.23 | 25.40 | 28.16 |
| local GRU (rung 2) | 15.10 | **19.03** | **20.02** |
| centralised GRU (rung 3) | 15.10 | 19.30 | 20.85 |
| **FedAvg GRU (rung 5)** | **15.10** | 19.38 | 20.69 |

Anomaly, median across stations:

| model | h | recall | precision | F1 | AUC-PR | fpr_ext |
|---|---|---|---|---|---|---|
| local GRU | 1 | 0.199 | 0.735 | 0.307 | 0.307 | 0.020 |
| centralised GRU | 1 | 0.189 | 0.721 | 0.296 | 0.302 | 0.025 |
| **FedAvg GRU** | 1 | **0.202** | 0.722 | **0.314** | **0.311** | 0.024 |
| local GRU | 3 | 0.234 | 0.652 | **0.351** | 0.339 | 0.142 |
| **FedAvg GRU** | 3 | 0.234 | 0.682 | 0.344 | **0.346** | 0.157 |
| **FedAvg GRU** | 6 | 0.215 | 0.692 | **0.341** | 0.339 | 0.188 |

**Federation is essentially free here.** FedAvg matches centralised training
on forecasting (identical at 1h; 19.38 vs 19.30 at 3h; and *better* at 6h,
20.69 vs 20.85) and matches or slightly beats both baselines on anomaly AUC-PR
at 1h and 3h. It does this with 15 features instead of up to 31, and with raw
data never leaving a node.

For context, Khan et al. (2024) — the §10 benchmark for this rung — report
94.5% federated against 96.7% centralised, a ~2.3% relative gap. Here the
federated-vs-centralised gap is within ±1% and reverses sign at 6h.

### Communication (§8.2)

| | |
|---|---|
| rounds run | 14 (early stopped) |
| rounds to convergence | 6 (best round 5) |
| per round | 1.595 MB |
| **total** | **22.3 MB** |

22 MB of weight traffic buys centralised-equivalent accuracy across 7 stations
and 8 years of hourly data. Measured from actual tensor sizes, uncompressed,
uplink plus downlink — `compress_weights` is deliberately off so the honest
number is the baseline.

### What this does and does not show

It shows federation is not the bottleneck: the privacy constraint costs almost
nothing in accuracy. It does **not** improve fault detection — rung 5's recall
and per-subtype profile track rung 2's closely, which is expected, since
§8.1's rung 5 is federation *without* spatial reasoning. The four invisible
subtypes stay invisible.

That is the cleanest possible setup for rung 6. With federation shown to be
accuracy-neutral, any improvement rung 6 produces is attributable to the graph
layer and the decision design, not to federation. And entry 8 already measured
what the graph layer has to reach for: drift 0.149, bias 0.261 from raw
neighbour comparison.

### Note on comparing with earlier entries

Forecast RMSE here (15.10 at 1h) is on the *injected* variant, whose test
targets contain synthetic corruption. Entry 5's clean-data figures (7.1–9.7)
are not comparable. Use `outputs/phase1_catalonia/` for forecasting accuracy
and `outputs/phase2_ladder/` for the ladder comparison.

### Open items

1. Rung 4 (centralised ST-GNN) still unbuilt — §8.1 wants it as the ceiling.
2. Rung 6: server GNN + the four-part decision layer from entries 7–8.
3. Low-variance test for stuck-at, still the only route to that subtype.
4. Carried: meteorology disabled; India untestable without an OpenAQ key.

---

## [2026-10-12] entry 10 — rung 6: the proposed model, and what it does and does not deliver

`src/models/server_gnn.py` + `src/models/decision_layer.py`, wired as rung 6.
The server reconstructs each node's hidden state **from its neighbours' hidden
states alone** and treats the reconstruction error as spatial disagreement —
the embedding-space analogue of entry 8's raw-neighbour comparison. Self-
information is excluded structurally: zero diagonal, exactly **one**
aggregation hop, depth in the MLP afterwards. Two hops would walk i → j → i and
quietly turn the reconstruction into an autoencoder.

The decision layer runs the four detectors entries 6–8 showed are needed, each
at its own timescale, and takes the strongest standardised signal.

### The headline: four previously-invisible fault types are now detectable

Recall by subtype, h=1:

| subtype | best baseline | **rung 6** | raw-neighbour ceiling (entry 8) | % of ceiling |
|---|---|---|---|---|
| bias | 0.000 | **0.148** | 0.261 | 57% |
| drift | 0.000 | **0.027** | 0.149 | 18% |
| dropout | 0.018 | **0.166** | 0.228 | 73% |
| stuck-at | 0.000 | **0.220** | 0.060 | **367%** |
| noise | 0.413 | **0.619** | 0.232 | — |
| spike | 0.590 | **0.671** | 0.512 | — |

**Rung 6 improves every single subtype**, and three that sat at exactly 0.000
across rungs 1, 2, 3 and 5 are now detected. This is the central objective
(§1, Objective 4) producing a result for the first time.

**§11's open risk did not materialise — but it did bite.** Compressing a
station's state into a 64-dim hidden vector preserves much of the spatial
signal: 57% of the raw-data ceiling for bias, 73% for dropout. It does *not*
preserve it for drift, which retains only 18% (0.027 against 0.149). Drift is
the subtlest signal and the first casualty of compression, which is the
honest, expected direction — it is now non-zero for the first time, but it
remains substantially unsolved.

Two subtypes **exceed** the raw-neighbour ceiling, and that is not an anomaly:
they are caught by detectors that are not neighbour comparisons at all.
Stuck-at (0.220 vs 0.060) comes from the low-variance test — entry 7's
observation that a frozen sensor scores *below* the normal median, so the
answer was a different statistic rather than a better reference. Spike and
noise come from the per-reading residual. The ensemble is combining genuinely
complementary mechanisms, which is what it was designed from the measurements
to do.

### The cost, stated plainly: precision collapses

Median across stations:

| model | h | recall | precision | F1 | AUC-PR | fpr_ext |
|---|---|---|---|---|---|---|
| FedAvg GRU (rung 5) | 1 | 0.202 | **0.722** | **0.314** | 0.311 | **0.024** |
| **proposed (rung 6)** | 1 | **0.433** | 0.173 | 0.240 | 0.312 | 0.190 |
| FedAvg GRU | 6 | 0.215 | **0.692** | **0.341** | 0.339 | 0.188 |
| **proposed** | 6 | **0.335** | 0.149 | 0.191 | 0.259 | **0.087** |

Recall more than doubles at h=1 (0.202 → 0.433), but precision falls from
0.722 to 0.173 — roughly one flag in six is correct — and F1 drops. **AUC-PR is
flat** (0.312 vs 0.311), which is the diagnostic that matters: the ensemble is
not ranking anomalies better overall, it is flagging far more of them. The
subtype coverage is genuinely new; the aggregate ranking quality is not.

The cause is the combination rule. Four detectors are standardised and the
**maximum** is taken, then one threshold is applied to that maximum. With four
chances to exceed a single threshold, false positives compound. The fix is
per-detector thresholds, each calibrated to its own false-positive budget,
rather than one threshold on a max — not attempted yet, and the obvious next
step.

One point in rung 6's favour: at h=6 its `fpr_on_extremes` is **0.087 against
0.188** for rung 5. Entry 8 predicted this — differencing against neighbours
cancels regional pollution episodes — and it is the §8.2 failure mode the
project exists to avoid.

### Systems (§8.2)

| | |
|---|---|
| graph edges | 42 (7 nodes, hybrid distance+correlation) |
| FedAvg weights | 22.3 MB total, converged round 6 |
| **hidden-state uploads** | **30.3 MB** |
| **total** | **52.6 MB** |
| server GNN | 40 epochs, best 37, val recon 0.0199 |

The spatial capability costs 30.3 MB on top of federation's 22.3 MB — more
than doubling communication. Worth stating when the §1.2 claim is made: the
privacy-preserving spatial channel is not free, and the hidden state is the
expensive part, not the weights.

### Honest summary

The project's central claim now has evidence: **cross-station spatial
agreement, computed server-side from compressed embeddings and never from raw
neighbour data, detects sensor faults that no amount of local modelling can
see.** That is the §1.2 gap, demonstrated end to end under the privacy
contract, with the contract enforced on every upload of every round.

It is not yet a *good* detector. Precision is poor, drift is barely moved, and
the aggregate ranking is no better than rung 5's. The result is a working
mechanism with a clearly identified next fix, not a finished system.

### Open items

1. **Per-detector thresholds** instead of one threshold on the max — the
   direct cause of the precision collapse.
2. Drift at 18% of ceiling: try a longer spatial window, or a larger hidden
   state, to test whether compression or timescale is the limiting factor.
3. Rung 4 (centralised ST-GNN) still unbuilt — the forecasting ceiling.
4. Carried: meteorology disabled; India untestable without an OpenAQ key.

---

## [2026-10-12] entry 11 — fixing the precision collapse, and the trade-off curve it exposed

Entry 10's rung 6 doubled recall but dropped precision from 0.722 to 0.173,
because four standardised detectors were combined by **maximum** against a
**single** threshold: four chances to clear one bar, so false positives
compound. Three fixes were tried; the first made it worse, and the reason is
worth keeping.

### Attempt 1 — per-detector budgets on *extremes*: much worse (0.173 → 0.027)

Giving each detector `target_fpr_extremes / k` seemed right, since §8.2 cares
most about crying wolf on real episodes. It was wrong. **Extremes are only the
top few percent of readings**, so capping false positives there leaves the
other ~95% completely unconstrained — and `val_fpr_extremes` deliberately
picks the *lowest* threshold meeting its constraint, maximising recall. Four
detectors each at their loosest permissible setting flagged almost everything:
recall rose to 0.597 and precision fell to **0.027**.

Subtype recall was excellent at that setting (drift 0.273, above entry 8's raw
ceiling of 0.149; stuck-at 0.608), which was the first hint that the detectors
were fine and only the operating point was wrong.

### Attempt 2 — budget on *all* trustworthy readings: this is the fix

`val_fpr_overall` caps false positives across every trustworthy reading, which
is what actually controls precision. At h=1 against rung 5:

| variant | recall | precision | F1 | AUC-PR | fpr_ext |
|---|---|---|---|---|---|
| rung 5 (FedAvg) | 0.202 | **0.722** | 0.314 | 0.311 | **0.024** |
| rung 6, max-threshold | 0.433 | 0.173 | 0.240 | 0.312 | 0.190 |
| rung 6, per-det on extremes | 0.597 | 0.027 | 0.051 | 0.258 | 0.134 |
| **rung 6, per-det on overall** | 0.319 | 0.322 | **0.322** | **0.322** | 0.071 |

### Attempt 3 — sweeping the budget, now free

Per-detector scores are persisted into `records.parquet` (`det_*` columns), so
the budget can be swept offline in seconds instead of a ~25-minute retrain
each. h=1, median across stations:

| budget | recall | precision | F1 | fpr_ext | drift | bias | stuck-at |
|---|---|---|---|---|---|---|---|
| **0.005** | 0.319 | **0.480** | **0.396** | 0.104 | 0.000 | 0.006 | 0.136 |
| 0.01 | 0.368 | 0.336 | 0.351 | 0.143 | 0.003 | 0.029 | 0.138 |
| 0.02 | 0.390 | 0.245 | 0.296 | 0.218 | 0.013 | 0.063 | 0.178 |
| 0.05 | 0.450 | 0.140 | 0.214 | 0.343 | 0.049 | 0.181 | 0.211 |
| 0.10 | 0.478 | 0.098 | 0.160 | 0.458 | 0.080 | 0.312 | 0.232 |
| 0.20 | 0.561 | 0.066 | 0.117 | 0.619 | 0.143 | 0.471 | 0.327 |
| 0.40 | 0.694 | 0.049 | 0.091 | 0.758 | **0.257** | **0.747** | 0.416 |

**F1 falls monotonically as the budget loosens**, so 0.005 is the default.

### Correction: the sweep is approximate, and the real run disagrees in detail

The sweep reuses the stored standardised scores with a fixed cut at zero, while
a real run re-fits each detector's scale and threshold per station *and per
horizon*. Running at 0.005 properly gives F1 **0.299** at h=1, not the 0.396 the
sweep projected. The sweep is a good guide to the *shape* of the curve and a
poor one for absolute values; quote the run, not the sweep.

Authoritative, `outputs/phase2_rung6_tuned/`:

| model | h | recall | precision | F1 | AUC-PR | fpr_ext |
|---|---|---|---|---|---|---|
| rung 5 | 1 | 0.202 | **0.722** | **0.314** | **0.311** | 0.024 |
| **rung 6 tuned** | 1 | 0.197 | 0.488 | 0.299 | 0.255 | **0.000** |
| rung 5 | 3 | 0.234 | **0.682** | 0.344 | 0.346 | **0.150** |
| **rung 6 tuned** | 3 | **0.276** | 0.527 | **0.385** | **0.384** | 0.150 |
| rung 5 | 6 | 0.215 | **0.692** | 0.341 | 0.339 | **0.188** |
| **rung 6 tuned** | 6 | **0.280** | 0.463 | **0.369** | **0.357** | 0.203 |

So rung 6 at the tight budget **beats rung 5 at 3h and 6h** on both F1 and
AUC-PR — the horizons where forecasting is hardest and the residual detector is
weakest — and is a shade behind at 1h, where it achieves `fpr_on_extremes` of
**0.000**. Subtype recall at h=1 becomes stuck-at 0.134 (from 0.000), dropout
0.090 (from 0.007), with drift and bias back at 0.000.

That last point is the trade-off in one line: **the tight budget buys aggregate
quality and gives up drift and bias entirely.**

### The finding underneath the curve

**The persistent faults are detectable, but only at very low precision.** Drift
goes from 0.000 to 0.257 across the sweep — exceeding the raw-neighbour ceiling
— but reaching it costs a precision of 0.049. That is not a tuning failure, it
is the shape of the problem: drift's per-hour signal sits inside the normal
spread (entry 7), so any threshold that catches it also catches a great deal of
ordinary variation.

Which suggests the operating point should not be one number, because the
*response* differs by fault type:

- **Spike, noise, dropout** — worth alerting on in real time, so precision
  matters and a tight budget is right.
- **Drift, bias** — these are calibration problems, not incidents. Nobody needs
  a 3am page; they need a monthly maintenance queue. A review queue with 1-in-20
  precision is perfectly workable when the check is cheap and the alternative is
  months of quietly wrong data.

So the honest recommendation is **two operating points, not one**: a tight
budget for alerting, and a loose one feeding a periodic calibration review. The
config ships the tight setting; the sweep above is how to pick the other.

### Also fixed

A real bug the new tests caught: a detector that produced nothing — `spatial`,
on any rung without a server GNN — was counted as active and took a share of
the budget, because `normalise` returned zeros for an all-NaN input, making an
absent detector look like a live one sitting exactly at its threshold. It now
returns NaN, and usability is judged on the raw detector outputs.

---

## [2026-10-12] entry 12 — rung 4 and the complete ladder

Rung 4 completes §8.1. Centralised ST-GNN: every station's raw windows held
centrally, shared GRU encoder, graph mixing across stations, shared decoder,
trained jointly. It respects none of §2.4, which is the point — it exists to
price the privacy constraint.

### Forecasting (RMSE, ug/m3 NO2, median across stations)

| rung | model | 1h | 3h | 6h |
|---|---|---|---|---|
| 1 | persistence | 17.23 | 25.40 | 28.16 |
| 2 | local GRU | 15.10 | 19.03 | 20.02 |
| 3 | centralised GRU | 15.10 | 19.30 | 20.85 |
| **4** | **centralised ST-GNN** | **14.62** | **18.10** | **19.21** |
| 5 | FedAvg GRU | 15.10 | 19.38 | 20.69 |
| 6 | proposed | 15.10 | 19.38 | 20.69 |

**Two numbers worth taking from this table.**

*Spatial information helps forecasting.* The ST-GNN beats the centralised GRU
by 3.2% / 6.2% / 7.9% at 1h / 3h / 6h, on identical data with identical
features. The only difference is that stations can see each other, so the gain
is attributable to the graph rather than to capacity or data volume.

*The privacy constraint costs 3-8% of forecast accuracy.* Federated (15.10 /
19.38 / 20.69) against the no-privacy ceiling (14.62 / 18.10 / 19.21) is
+3.3% / +7.1% / +7.7% RMSE, widening with horizon. That is the answer to
objective 6, measured rather than argued, and it is a modest price.

Note rungs 5 and 6 share a forecaster by construction, so their RMSE is
identical; rung 6 differs only in how it *decides*.

### Anomaly (median across stations)

| model | h=1 F1 | h=3 F1 | h=6 F1 | h=3 AUC-PR |
|---|---|---|---|---|
| 4 centralised ST-GNN | **0.322** | 0.357 | 0.333 | 0.334 |
| 5 FedAvg GRU | 0.314 | 0.344 | 0.341 | 0.346 |
| **6 proposed** | 0.299 | **0.385** | **0.369** | **0.384** |

Rung 6 wins at 3h and 6h on both F1 and AUC-PR — the horizons where the
forecast is weakest and a residual detector has least to work with. Rung 4
edges it at 1h, where a sharp forecast is doing most of the work.

### The subtype table is still the real result (h=1)

| subtype | rungs 1-5 | rung 6 |
|---|---|---|
| bias | 0.000 | 0.000 |
| drift | 0.000 | 0.000 |
| dropout | 0.000-0.018 | **0.090** |
| stuck-at | 0.000 | **0.134** |
| noise | 0.380-0.413 | 0.247 |
| spike | 0.449-0.590 | 0.505 |

**No rung with access to raw data from every station — including the
centralised ST-GNN — detects stuck-at or bias at all.** Only rung 6 does, and
it does so under the privacy constraint. That is not a quirk of tuning: entry
7 showed stuck-at scores *below* the normal median on any residual-based
score, so no amount of forecasting skill reaches it. It takes a different
statistic, which is what the decision layer adds.

At this (tight) budget drift and bias sit at 0.000; entry 11's sweep shows
they are reachable at 0.257 and 0.747 respectively, for a precision of 0.049.
The curve, not a single number, is the finding.

### Ladder complete

All six rungs of §8.1 are built and reported on real EEA data with rate-matched
synthetic faults. Phase 2 is done.

---

## [2026-10-12] entry 13 — closing the split-learning loop: a small gain at an enormous price

§7.2 says "Embeddings sent back down to each station", and that half had never
been built: rung 6 computed embeddings for its decision layer but shared rung
5's forecaster exactly, so their RMSE was identical by construction. Rung 4
showed roughly 7% sitting in that gap. `run_split_learning` now implements the
full §7.3 round — weights up, hidden states up, aggregate, server GNN pass,
**embeddings down** — and the node fuses them into its encoder state before
decoding, through a layer initialised to an exact no-op so it has to earn any
use of them.

### Forecasting: the gain is real and tiny

| model | 1h | 3h | 6h |
|---|---|---|---|
| 5 FedAvg GRU | 15.10 | 19.38 | 20.69 |
| **6 proposed + split learning** | **15.01** | **19.25** | **20.58** |
| 4 centralised ST-GNN (ceiling) | 14.62 | 18.10 | 19.21 |

0.6% / 0.7% / 0.5% better than rung 5 — which closes only **19% / 10% / 7%**
of the distance to the no-privacy ceiling. The embedding feedback works, in
the sense that it moves the number in the right direction every time. It does
not come close to recovering what centralised access buys.

### Communication: 146× for that

| | rung 5 | rung 6 + split learning |
|---|---|---|
| per round | 1.60 MB | **216.47 MB** |
| total | 22.3 MB | **3,247 MB** |
| rounds to convergence | 6 | 7 |

Per-window hidden states for train, validation and test, every round, across
seven stations. **3.2 GB of traffic to buy 0.6% RMSE** is not a trade any real
deployment makes, and saying so is more useful than reporting the improvement
alone.

### Where it does pay: 1h detection

| model | h=1 F1 | h=1 AUC-PR |
|---|---|---|
| 6 proposed | 0.299 | 0.255 |
| **6 proposed + split learning** | **0.358** | **0.350** |

A 20% relative F1 gain and a 37% AUC-PR gain at the shortest horizon, where
rung 6 had been its weakest and was previously behind rung 5. At 3h and 6h the
two are level. Subtype recall is essentially unchanged (stuck-at 0.134,
dropout 0.090), with small gains on noise and spike.

So the embedding is more useful as *context for deciding* than as *context for
forecasting* — consistent with everything since entry 7, where the limiting
factor was never forecast quality.

### What this says about the design

The honest conclusion is that **per-window embedding feedback is the wrong
granularity**. The spatial signal that matters is slow — neighbourhood
agreement over hours and days (entries 8, 11) — so shipping a 64-dim vector for
every hour of every split, every round, pays for temporal resolution the
decision layer then averages away.

Cheaper designs that should capture most of the same value:

1. **Send embeddings for the validation and test splits only** — training
   needs them only if the node is to *learn* from them, and the forecasting
   gain above says that is barely worth anything. This alone would cut traffic
   by roughly two-thirds.
2. **Coarsen the time axis** — one embedding per day rather than per window,
   matching the timescale the decision layer actually uses.
3. **Keep rung 6 without split learning** for the 3h/6h horizons, where it is
   already the best model on F1 and AUC-PR at 52.6 MB total.

### Simplification, restated

Gradients do not cross the server boundary. CNFGNN (Meng et al. 2021)
backpropagates through the server GNN into the node encoders; here the GNN
trains on its own reconstruction objective and the node treats the embedding
as a fixed input. A gradient-coupled version might extract more from the same
traffic — that is untested, and it is the most promising thing left in the
architecture.

---

## [2026-10-12] entry 14 — magnitude stratification and two operating points

Two changes to how results are *reported*, not to the detector. Both were
cheap, and both change what the numbers mean.

### 1. Detection rate by fault magnitude

Magnitude is recovered exactly rather than estimated: the clean and injected
processed builds differ only by the injection, so `|y_injected - y_clean|` is
the amount the fault actually moved the reading, expressed in units of that
station's own training sigma.

h=1, rung 6 + split learning, alert tier:

| magnitude | faults | recall |
|---|---|---|
| >5σ | 223 | **0.924** |
| 2–5σ | 271 | 0.583 |
| 1–2σ | 478 | 0.130 |
| 0.5–1σ | 307 | 0.134 |
| <0.5σ | 452 | 0.139 |

**92% of large faults are caught.** That is the number comparable to published
work, which generally injects large obvious faults; an aggregate F1 of 0.37
had been hiding it. Detection collapses below 2σ, which is exactly where entry
7 measured the normal spread to live.

**And bias and drift are small by construction.** Bias is injected at 0.5–2σ
and drift ramps to 0.8–2.5σ over 72+ hours, so their per-reading deviation
almost never leaves the bottom bands. Their 0.000 recall is therefore less a
detector failure than a consequence of their size — which reframes entry 7's
conclusion: they are invisible *per reading* because they are small per
reading, so detection has to come from persistence or from space, never from
magnitude.

One pleasing anomaly in the table: the <0.5σ band scores 0.139, *higher* than
1–2σ at 0.130. That is the low-variance detector catching stuck-at faults,
whose magnitude is near zero because a frozen sensor barely moves the value.
The ensemble is genuinely not magnitude-driven.

### 2. Two operating points instead of one

| | ALERT (0.005) | REVIEW (0.20) |
|---|---|---|
| h=1 recall | 0.315 | **0.575** |
| h=1 precision | **0.480** | 0.066 |
| h=1 F1 | **0.390** | 0.118 |
| fpr_on_extremes | **0.091** | 0.610 |

Recall by magnitude and subtype at each:

| | alert | review |
|---|---|---|
| >5σ | 0.924 | **1.000** |
| 2–5σ | 0.583 | **0.923** |
| 1–2σ | 0.130 | **0.642** |
| spike | 0.662 | 0.946 |
| noise | 0.546 | 0.930 |
| bias | 0.006 | **0.494** |
| dropout | 0.120 | 0.450 |
| stuck-at | 0.136 | 0.267 |
| drift | 0.000 | **0.132** |

The alert tier answers "wake someone up": F1 0.390, better than rung 5's
0.314, with 92% of serious faults caught and false alarms on genuine episodes
at 9%.

The review tier answers "what should a technician check this month": **every**
fault above 5σ, 92% above 2σ, and — for the first time at a usable rate —
bias at 0.494 and drift at 0.132. It costs a precision of 0.066, roughly
fourteen inspections per real fault, which is the right trade when an
inspection is cheap and the alternative is months of quietly wrong data.

Neither tier is the "real" answer. Reporting one number forced a choice
between them and made the system look worse than it is.

### Mechanics

Both tiers are computed offline from the `det_*` columns in
`records.parquet`, so changing either costs seconds. 8 new tests (79 total),
including that a looser budget can never flag less, and that altering test
rows cannot move a threshold.

---

## [2026-10-12] entry 15 — the CUSUM drift detector: right idea, wrong bottleneck

Entry 14 established that drift is invisible *per reading* because it is small
per reading — injected at 0.8–2.5σ spread over 72+ hours. The implied fix was a
statistic that accumulates rather than averages, so Page's CUSUM was added as a
fifth detector, running on the spatial disagreement (not the residual, since
entry 7 showed the forecaster tracks drift and its residual never grows).

### It does exactly what it was designed to do

ALERT tier, h=1, with and without:

| | 4 detectors | 5 with CUSUM |
|---|---|---|
| bias | 0.006 | **0.093** |
| stuck-at | 0.136 | **0.160** |
| 1–2σ band | 0.130 | **0.213** |
| 0.5–1σ band | 0.134 | **0.160** |
| <0.5σ band | 0.139 | **0.175** |

Small, persistent faults improve sharply — bias by a factor of fifteen, the
1–2σ magnitude band by 64%. The mechanism works.

### And it costs more than it gains

| | 4 detectors | 5 with CUSUM |
|---|---|---|
| spike | **0.662** | 0.643 |
| noise | **0.546** | 0.523 |
| 2–5σ band | **0.583** | 0.539 |
| >5σ band | **0.924** | 0.915 |
| **aggregate F1** | **0.390** | 0.310 |
| precision | 0.480 | **0.528** |

The cause is budget dilution, and it is structural rather than a tuning
accident: five detectors share the same total false-positive allowance, so each
gets a fifth instead of a quarter and **every pre-existing detector is
tightened**. The CUSUM buys small faults by making the others worse at large
ones.

**Default is therefore off** (`use_drift_cusum: false`), kept available for
deployments where small persistent faults are the priority. Shipping a 20%
aggregate F1 regression as a default would not be an improvement.

### Drift is still 0.000, and now we know why

The CUSUM did not move drift at all. The diagnosis is clear and it is not the
temporal statistic: **the bottleneck is the embedding.** Entry 8 measured
raw-neighbour comparison catching drift at 0.149; entry 10 measured the
embedding-based version at 0.027, 18% of that ceiling. A CUSUM amplifies
persistent signal, and there is almost no persistent drift signal in the
reconstruction error to amplify. You cannot accumulate information the
compression discarded.

So the open problem has moved, usefully. It is no longer "find a better
temporal statistic for drift" — that has been tried three ways now
(accumulation in entry 7, uncontaminated reference in entry 7, CUSUM here).
It is **"get a drift-sensitive signal through the privacy boundary"**.

One concrete candidate, flagged rather than implemented because it touches
§2.4: the reconstruction error is unsigned and derived from a hidden state
optimised for *forecasting*, which need not encode absolute level at all. A
drifting station's level is what diverges. Uploading a small signed summary of
the station's own normalised level deviation would carry that directly — and
whether that counts as a normalization statistic under §2.4 is a judgement
call for the project owner, not one to make silently. It is not the scaler, but
it is derived from it.

### Budget-splitting is now a known cost

A general lesson worth carrying: under a shared false-positive budget, adding a
detector is never free. Each new one must earn more than the 1/k tightening it
imposes on every existing detector. The CUSUM does not, at this operating
point, against this mix of faults.
