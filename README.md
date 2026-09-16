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
| **1** — data pipeline, preprocessing, baseline rungs 1–3 | Running on real EEA data |
| **2** — ST-GNN, FedAvg, server GNN + decision layer (rungs 4–6) | Not started |
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
