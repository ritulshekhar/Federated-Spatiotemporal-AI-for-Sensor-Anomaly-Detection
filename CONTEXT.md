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
