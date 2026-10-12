# FedAirGuard

**Privacy-Preserving Federated Spatiotemporal AI for Distributed Air-Quality
Monitoring and Sensor Anomaly Detection**

Each air-quality station trains a local forecasting model under federated
learning, so raw readings never leave the station. A server-side graph network
learns spatial relationships between stations from compressed embeddings alone.
A decision layer then classifies each anomalous reading as **trustworthy** (a
real pollution event) or **sensor-faulty** — without ever centralising raw data.

The classification is the point; the forecasting is the mechanism that makes it
possible.

> **Design contract:** [`CLAUDE.md`](CLAUDE.md) holds every locked decision and
> the reasoning behind it. Read it before changing anything.
> **Build log:** [`CONTEXT.md`](CONTEXT.md) is an append-only record of what was
> built, what was measured, and what is still unknown.

---

## Status

| Phase | State |
|---|---|
| **1** — data pipeline, preprocessing, baseline rungs 1–3 | Complete, on real EEA data |
| **2** — ST-GNN, FedAvg, server GNN + decision layer (rungs 4–6) | **Complete** — all six §8.1 rungs built and reported |
| **3** — dashboard, India generalization experiment | Acquisition + schema verified; not wired in |

### What the real data looks like

Measured from the EEA E1a (verified) delivery for Catalonia, 2017–2024:

| | |
|---|---|
| Stations | 7 Barcelona stations reporting NO2 |
| Rows | 990,598 |
| **Real fault rate** | **2.805%** (1.6%–4.6% per station) |
| Validity codes present | `1` and `-1` only — no `-99` maintenance flags exist in this node set |
| Verification | every row is `1` (verified) |

### Objective 4 baseline (`outputs/phase1_catalonia_injected/`)

Fault classification against rate-matched synthetic faults (§2.1a). Median
across stations, urban-background class:

| Model | h=1 recall | h=1 `fpr_on_extremes` | h=6 recall | h=6 `fpr_on_extremes` |
|---|---|---|---|---|
| Persistence | 0.178 | 0.017 | 0.216 | **0.500** |
| Local GRU | 0.219 | 0.106 | 0.286 | 0.370 |
| Centralised GRU | 0.219 | 0.148 | 0.257 | 0.481 |

Read the two columns together. At 6h the naive residual detector calls **up to
66.7%** of genuine high-pollution readings sensor faults — the Gupchup et al.
(2019) failure mode, which they measured at up to 45%. And a better forecaster
makes it *worse*: local GRU buys +0.04 recall at h=1 for a six-fold rise in
false positives on real events, because sharper forecasts produce bigger
residuals on genuine extremes too.

That is the gap rung 6 exists to close: **hold recall, cut
`fpr_on_extremes`**, using cross-station spatial agreement to tell "unusual"
from "wrong".

### Rung 6 — the proposed model (`outputs/phase2_rung6/`)

Recall by fault subtype at h=1, against the best of rungs 1–5:

| subtype | best baseline | **rung 6** |
|---|---|---|
| bias | 0.000 | **0.148** |
| drift | 0.000 | **0.027** |
| dropout | 0.018 | **0.166** |
| stuck-at | 0.000 | **0.220** |
| noise | 0.413 | **0.619** |
| spike | 0.590 | **0.671** |

Three fault types that were at exactly zero across every other rung are now
detected. Spatial agreement is computed server-side from **compressed
embeddings only** — never from raw neighbour data — which is the §1.2 gap,
demonstrated under the privacy contract.

### The complete ladder (`outputs/phase2_*`)

Forecast RMSE (µg/m³ NO2, median across stations):

| rung | model | 1h | 3h | 6h |
|---|---|---|---|---|
| 1 | persistence | 17.23 | 25.40 | 28.16 |
| 2 | local GRU | 15.10 | 19.03 | 20.02 |
| 3 | centralised GRU | 15.10 | 19.30 | 20.85 |
| **4** | **centralised ST-GNN** (no privacy) | **14.62** | **18.10** | **19.21** |
| 5 | FedAvg GRU | 15.10 | 19.38 | 20.69 |
| 6 | proposed | 15.10 | 19.38 | 20.69 |

Two results worth quoting:

- **Spatial information helps forecasting** — the ST-GNN beats the centralised
  GRU by 3.2–7.9% on identical data and features. The only difference is that
  stations can see each other.
- **Privacy costs 3–8% of forecast accuracy** — federated against the
  no-privacy ceiling, widening with horizon. That is objective 6, measured.

On fault classification, rung 6 wins at 3h and 6h (F1 0.385 / 0.369 against
rung 5's 0.344 / 0.341), and **no rung with full raw access — including the
centralised ST-GNN — detects stuck-at or bias at all.** Only rung 6 does, and
under the privacy constraint.

### How to read the detector's accuracy

**Do not use accuracy.** At a 2.8% fault rate, a classifier that answers "never
faulty" scores 0.977 — higher than this system's 0.976. That is why §8.2
specifies precision/recall/F1/AUC-PR.

The two numbers that actually describe it:

**Detection rate by fault size** (h=1, alert tier):

| magnitude | recall |
|---|---|
| >5σ | **0.924** |
| 2–5σ | 0.583 |
| 1–2σ | 0.130 |
| <1σ | ~0.135 |

92% of serious faults are caught. Detection collapses below 2σ — which is
where injected bias and drift live by construction, so their low recall is
largely a statement about their size.

**Two operating points**, because one threshold cannot serve both jobs:

| | ALERT | REVIEW |
|---|---|---|
| purpose | wake someone up | monthly technician list |
| h=1 F1 | **0.390** | 0.118 |
| precision | 0.480 | 0.066 |
| faults >5σ | 0.924 | **1.000** |
| faults 2–5σ | 0.583 | **0.923** |
| bias | 0.006 | **0.494** |
| drift | 0.000 | **0.132** |

The review tier trades precision for coverage, which is correct when an
inspection is cheap and the alternative is months of quietly wrong data. Run
either with `python -m src.evaluation.operating_points outputs/<run>`.

### How much of this is real (block bootstrap, `CONTEXT.md` entry 16)

**Aggregate differences between rungs are not statistically significant.** Rung
6 vs rung 5 at 3h: F1 +0.006, 95% interval [-0.102, 0.113]. Quote the rungs'
point estimates only with that caveat attached.

**Per subtype, resampling time only** — i.e. "would a different stretch of 2024
have said the same about these seven stations?" — three effects are real:

| subtype | difference | 95% interval |
|---|---|---|
| stuck-at | **+0.236** | [0.012, 0.382] |
| dropout | **+0.127** | [0.016, 0.221] |
| noise | **−0.125** | [−0.203, −0.045] |

**Resampling stations as well, nothing is significant.** With seven stations,
station-to-station variability swamps every effect. So rung 6 detecting
stuck-at where other rungs detect none is *demonstrated on this node set* and
*not yet established as general* — and scaling to Catalonia's ~129 stations is
the only way to close those intervals.

```bash
python -m src.evaluation.uncertainty outputs/phase2_ladder outputs/phase2_rung6_split \
    --models 5_fedavg_gru 6_proposed --horizon 3 --compare 6_proposed 5_fedavg_gru
```

See `CONTEXT.md` entries 11–16.

Two findings that shaped the configuration, both recorded in `CONTEXT.md`:

- **Fault labels only start in 2017.** 2013–2016 contains 477,114 rows with
  *zero* negative Validity codes. That is a change in reporting practice, not
  four fault-free years, so `year_from` is 2017. Including those years drops the
  measured fault rate from 2.81% to 1.91%.
- **EEA's Parquet download carries no station EoI code.** Station identity lives
  in the registry and has to be joined via `SamplingPoint`.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                       # Windows
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-optional.txt     # airbase (EEA downloader), matplotlib
```

Python 3.11+ (developed on 3.13). Everything runs on CPU.

---

## Running Phase 1

Each step writes to `data/` and prints a per-station summary. Run from the
repository root.

```bash
python -m src.data.station_metadata --refresh --print
```
Downloads EEA's station registry and checks every configured EoI code against
it. Run this first — everything else depends on station identity.

```bash
airbase verified -c ES -C Barcelona -p NO2 -p PM10 -p PM2.5 -p O3 -p SO2 -F hourly --path data/raw/eea
```
Downloads verified E1a data (~58 MB). Restricting to the Barcelona urban area
instead of all of Spain cuts this from ~2.2 GB. `python -m src.data.fetch_eea`
does the same thing through the project's own wrapper.

```bash
python -m src.data.fetch_eea
python -m src.data.harmonize --raw-parquet data/raw/eea_raw.parquet
```
Maps the download into the common schema, applies the year window, and prints
fault-label coverage per year plus the measured real fault rate.

```bash
python -m src.preprocessing.pipeline --region catalonia
```
Runs §6 independently per station: indexing, imputation (mask built *before*
filling), physical screening, features, per-station normalization fit on the
training window only, and the chronological split. Writes
`data/processed/<station>/`.

```bash
python -m src.baselines.ladder --region catalonia
```
Runs baseline rungs 1–3 and writes `outputs/phase1_catalonia/`.

### Without a download

```bash
python -m src.data.harmonize --demo
python -m src.preprocessing.pipeline --region demo
python -m src.baselines.ladder --region demo
```
Exercises the whole pipeline on simulated data. Tagged `region="demo"` and
`label_source="simulated"` so it can never be mistaken for a result.

### India (Phase 3)

```bash
$env:OPENAQ_API_KEY = "<free key from https://explore.openaq.org/register>"
python -m src.data.fetch_cpcb --discover --city Delhi
python -m src.data.fetch_cpcb --inspect <location_id>
python -m src.data.fetch_cpcb --download <location_id> --year-from 2019 --year-to 2024
```
The key is needed **once**, to map country to location ids. The bulk history
download from the public S3 archive needs no credentials.

---

## Reading the results

`outputs/<run>/` contains four CSVs:

- `forecast_per_station.csv` — RMSE / MAE / MAPE per station per horizon
- `anomaly_per_station.csv` — precision / recall / F1 / AUC-PR per station
- `*_spread.csv` — min/median/max **across** stations, never a mean

**`fpr_on_extremes` is the column to read alongside recall.** It is the fraction
of genuine high-pollution readings the detector wrongly called faults. Gupchup
et al. (2019) found naive detectors misclassify up to 45% of real events this
way, so recall on its own means nothing. `anomaly_metrics()` cannot be called
without this control.

**Nothing is ever reported as a single pooled number.** Every table is grouped
by `(region, station_class, label_source)`; `src/evaluation/report.py` raises
`PoolingError` if you try otherwise.

---

## Layout

```
configs/         data.yaml, model.yaml, federation.yaml
src/data/        acquisition + the §5.1 common schema
src/preprocessing/  the §6 pipeline, one module per stage
src/models/      node GRU (server GNN + decision layer are Phase 2)
src/federated/   Phase 2
src/baselines/   the §8.1 ladder
src/evaluation/  metrics and the grouping-enforced reporter
data/            raw / harmonized / processed  (gitignored)
outputs/         run reports (gitignored)
```

## Privacy contract

Encoded in `configs/federation.yaml` and asserted in code, not left to
convention. A node transmits GRU weight updates and encoder hidden states.
It never transmits raw readings or its local normalization statistics — a
shared scaler is itself a disclosure, since mean and standard deviation
summarise the station's raw distribution.
