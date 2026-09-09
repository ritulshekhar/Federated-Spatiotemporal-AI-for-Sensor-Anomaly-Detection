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
