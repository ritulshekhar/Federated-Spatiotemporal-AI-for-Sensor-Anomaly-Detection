# FedAirGuard — Project Context for Implementation

**Read this file in full before writing any code.** It captures every design decision already made
so you don't need to re-derive or re-litigate them. Where a decision is marked LOCKED, treat it as
a constraint, not a suggestion — it was chosen deliberately, usually to avoid a specific failure
mode documented alongside it.

---

## 1. What this project is

**Title:** Privacy-Preserving Federated Spatiotemporal AI for Distributed Air-Quality Monitoring
and Sensor Anomaly Detection ("FedAirGuard")

**One-sentence description:** Each air-quality monitoring station trains a local forecasting model
under federated learning (raw data never leaves the station); a server-side graph network learns
spatial relationships between stations from compressed embeddings only; and a decision layer uses
forecast residuals plus cross-station spatial agreement to classify each anomalous reading as
**trustworthy** (real event) or **sensor-faulty** — without ever centralizing raw sensor data.

**The central objective (do not lose sight of this):** Objective 4 below is what the rest of the
system exists to serve. Forecasting accuracy is the mechanism; fault-vs-event classification is the
point. If a design choice improves forecasting but makes fault-vs-event classification harder or
impossible to evaluate honestly, the classification wins.

### 1.1 Objectives, in order

1. Federated forecasting: each station is a virtual AIoT edge node, training a local model
   collaboratively without transmitting raw observations.
2. Cross-node spatial modelling on the server, from encoded hidden states only — never raw readings.
3. Multi-horizon forecasting (1h / 3h / 6h) using pollutant + meteorological variables, reported per
   station and per horizon (never pooled into one number).
4. **[CENTRAL]** Classify each anomalous reading (flagged by forecast residual) as **trustworthy**
   or **sensor-faulty**, using cross-node spatial agreement computed without sharing raw data.
5. Evaluate the classifier primarily against rate-matched synthetic faults injected at real
   `Validity` episode durations, and report real `Validity` labels alongside as a
   data-availability baseline. **AMENDED 2026-09-17 — see §2.1a.** (Originally: evaluate primarily
   against real per-reading fault labels, with synthetic injection supplementary only. That is not
   possible on this dataset: every Validity-flagged row carries the −999 no-data sentinel rather
   than a reading, so there is nothing to classify. CONTEXT.md entry 4.)
6. Compare federated vs. centralised training on accuracy, communication cost, and privacy, using
   the baseline ladder in §9.1.
7. Build a dashboard visualising station data, forecasts, and the trustworthy/faulty decision.

### 1.2 The research gap this fills

No published system performs fault-vs-event disambiguation under a federated constraint. Papers
that federate a fault detector (Khan/FedLSTM 2024, van Dreven 2026) do so without any cross-node
spatial reasoning. Papers that use cross-node spatial reasoning for fault detection (Lin 2022,
Ferrer-Cid 2022, Hammad 2026) do so centrally, requiring raw neighbour data. This project combines
both for the first time. See §12 for full citations — cite these in code comments and docs where
relevant, don't restate the argument from scratch.

---

## 2. LOCKED design decisions — do not relitigate

These were each chosen to prevent a specific, previously-identified failure mode. If you think one
should change, flag it explicitly rather than silently deviating.

1. **~~Real fault labels (EEA `Validity` ∈ {-1, -99}) are the primary evaluation ground truth.~~**
   **AMENDED 2026-09-17 by the project owner. Superseded — see 1a below.**

   ~~Synthetic fault injection (stuck-at, drift, spike, noise, dropout, bias) is a *supplementary*
   stress test for fault subtypes Validity doesn't finely distinguish (e.g. slow drift) — never the
   headline metric.~~

   *Why this was amended, kept here so the original reasoning is not lost:* the rule assumed §3.1's
   claim that `Concentration` survives a negative `Validity` flag. It does not. Measured across the
   Catalonia E1a delivery, 2017–2024, **all 27,783 Validity-flagged rows across all five pollutants
   carry exactly the −999 no-data sentinel** — the relation is biconditional, every sentinel is a
   fault and every fault is a sentinel. Those rows are therefore imputed by §6.2 and excluded from
   anomaly scoring, leaving **0 of 7,106 fault targets evaluable**. §2.1 named a primary ground
   truth containing nothing to measure. Full evidence in CONTEXT.md entry 4.

1a. **Rate-matched synthetic fault injection is the primary evaluation ground truth for
   Objective 4.** Real `Validity` labels remain reported, as an honest *data-availability*
   baseline — they record when an instrument failed, not what a wrong reading looks like.

   Three constraints keep this from becoming an arbitrary simulation, and all three are LOCKED:

   - **The injection rate matches the measured real rate** — `real_fault_rate = 0.02805`,
     measured, never a default. This is §2.7, unchanged and now load-bearing.
   - **Episode durations are sampled from the real `Validity` episode-duration distribution.**
     EEA's flags genuinely record *when* instruments fail and *for how long*; only the corrupted
     value itself is synthesised. Use the real data for what it actually evidences.
   - **`label_source` stays "synthetic" and is never blended with "real".** §5.3 and §8.3 are
     unchanged: both numbers are reported, separately, always.

   *What this costs, stated plainly rather than buried:* the headline for Objective 4 becomes
   "evaluated on realistic synthetic faults", which is weaker against the §1.2 novelty claim than
   real labels would have been. Sourcing labels that genuinely contain plausible-but-wrong
   readings — co-located reference-vs-low-cost-sensor data, as Ferrer-Cid et al. (2022) used in
   Barcelona — remains the stronger option and is the recommended stretch goal.
2. **Never pool across regions/stations into one blended dataset with one shared threshold or one
   shared normalization scaler.** Every station normalizes on its own training window. Every result
   is reported per-station and per-region — never as one pooled accuracy number. This prevents a
   data-rich or low-variance node/region from silently dominating the model or the metrics (see
   §8.5 for the mechanism).
3. **Chronological splits only, never random shuffling.** Train/val/test = 70/15/15 by date, done
   independently per region so each region's test set covers a proportionate slice of its own
   seasonal variation.
4. **The node never transmits raw readings, its local normalization statistics, or (for the spatial
   model) more than the hidden state / embedding needed by the server.** This is the entire privacy
   contract of the project — any code path that would violate it needs to be flagged, not written.
   **Narrowed 2026-10-13 — see 4a.** The prohibition on raw readings and on normalization
   statistics is unchanged and remains absolute. What changed is the "more than" clause.

4a. **One additional field may leave the node: `level_bucket`, a 5-level ordinal summary of the
   station's own level deviation, computed on already-normalized data.** AMENDED 2026-10-13 by the
   project owner, who was asked explicitly and chose the coarse-bucket option over a continuous
   scalar and over refusing outright.

   *Why this was needed.* CONTEXT.md entry 15 established that slow drift is invisible to a node's
   forecast residual, and not as a tuning problem: a sensor that drifts slowly stays *easy to
   predict*, so the residual the whole decision layer is built on shrinks rather than grows. No
   amount of server-side work over residuals or embeddings can recover a signal the residual never
   contained. Drift recall sat at 0.000 at the alert operating point for every rung, including the
   centralised model with full raw access. Either a drift-sensitive quantity crosses the boundary,
   or drift is permanently out of reach.

   *Why it is defensible, stated precisely rather than waved at.* The bucket is computed **after**
   §6.5 z-scoring, on the station's own normalized series. It therefore says "this station
   currently sits roughly one of its own standard deviations above its own training mean" and
   nothing at all about what that is in µg/m³. The scaler never leaves the node, so absolute level
   and absolute scale are both unrecoverable from an integer in 0–4. It is strictly coarser than
   the `encoder_hidden_state` already permitted, which is a continuous float vector encoding the
   same window.

   *What it costs, not buried.* The §1.2 novelty claim is "cross-node spatial reasoning under a
   federated constraint", and the constraint is now one field wider than it was. Any write-up must
   describe the uploaded set as *weights, encoder hidden states, and a coarse ordinal level
   summary* — not as "hidden states only". A reviewer is entitled to ask whether the ordinal leaks
   anything over many rounds; the honest answer is that it leaks the shape of the station's own
   normalized level trajectory, which is information about its *behaviour over time*, not about its
   measurements.

   *The mechanism it enables.* Stations in a region share weather and pollution regimes, so a
   genuine episode lifts every station's bucket together. One station's bucket rising while its
   neighbours stay put is drift — which is exactly §7.4's spatial-agreement logic applied to level
   instead of residual. Enforced in code: `configs/federation.yaml` `privacy:` lists it, and
   `check_upload` runs on every upload, every round.

   *Measured 2026-10-14, and the scope is narrower than the paragraph above implies.* At 30
   stations it identifies drifting stations with precision 0.688 and recall 0.647. At **7**
   stations it fails completely — 0 true positives — because all seven Barcelona stations sat in
   the same bucket: stations inside one city co-move, so the neighbour median moves with them and
   disagreement is zero for everyone, drifting or not. **The mechanism needs dispersion among
   neighbours, not merely neighbours**, which makes a dense single-city cluster its worst case
   rather than its best. Two further limits: it identifies *stations*, never *readings* (§2.4a
   permits one ordinal per station per upload, so per-reading drift recall stays ~0.01), and about
   one flag in three is a genuine local level shift rather than drift. CONTEXT.md entries 23-24.
5. **The citation replacing "Veera Brahmam & Gopikrishnan (2024)" is Alduais, Abdullah & Jamil
   (2019), "RDCM."** The former was retracted by Elsevier (Sept 2025) for plagiarizing the latter.
   Never re-add the retracted citation.
6. **India/CPCB data, if used at all, is a secondary generalization experiment, not primary
   training/evaluation data.** CPCB has no equivalent of EEA's Validity/Verification flags — see
   §4.2. Any fault label attached to CPCB-origin data is synthetic and must be tagged as such
   (`label_source = "synthetic"`) and never blended into a "real fault" metric.
7. **Fault injection rate on any synthetic-labeled data must match the measured real fault rate
   from EEA data** (`real_fault_rate = (eea_labels == "fault").mean()`), not an arbitrary default.
   Otherwise the model can learn "region X → more likely faulty" as a shortcut instead of learning
   genuine fault signatures.

---

## 3. Datasets

### 3.1 Primary: EEA Air Quality e-Reporting — Barcelona / Catalonia

- **Source:** European Environment Agency, https://www.eea.europa.eu/en/datahub/datahubitem-view/3b390c9c-f321-490a-b25a-ae93b2ed80c1
- **Region:** Catalonia (Generalitat de Catalunya network); primary node set = 10 Barcelona city
  stations (urban-traffic: Gràcia, Eixample; urban-background: Vall d'Hebron, Ciutadella;
  industrial/port-adjacent: Sant Adrià de Besòs). Scale-up path: 129 stations, 15 zones,
  Catalonia-wide.
- **Pollutants:** PM2.5, PM10, O3, NO2, SO2 (CO at some stations). Hourly. 2013–present.
- **Data flow: use E1a (verified) only, never E2a (Up-To-Date/unverified).** E1a is submitted
  annually by 30 September for the prior year, fully QA-reviewed. E2a flags are provisional.
- **Access (pick one):**
  - `pip install airbase`, then `AirbaseClient().request(country=["ES"], pl=[...], year_from=2013, year_to=<latest verified year>)` — downloads all of Spain; filter to the 10 station EoI codes afterward.
  - Preprocessed, gap-filled dataset with flags preserved: https://zenodo.org/records/17085064
  - Manual portal (slow, fallback only): https://eeadmz1-downloads-webapp.azurewebsites.net/

#### Column schema (as returned by the API)

| Column | Type | Notes |
|---|---|---|
| `AirQualityStationEoICode` | string | Station ID, e.g. `ES2097A`. Primary key. |
| `Countrycode` | string | ISO2, `ES`. |
| `AirPollutant` | string | e.g. `PM10`. |
| `AirPollutantCode` | string (URI) | See pollutant code table below. |
| `AveragingTime` | string | `hour` for this project. |
| `Concentration` | float | ~~Present even when Validity is negative — this is what makes real fault labels possible.~~ **FALSE, corrected 2026-10-06.** In the Catalonia E1a delivery every one of the 27,783 `Validity`-negative rows carries exactly `-999.0`, the no-data sentinel, across all five pollutants. Whatever the sensor originally reported — a wrong number, or nothing at all during an outage — was replaced during verification, and the two cases are indistinguishable afterwards. `harmonize._mask_sentinels` converts it to missing. This is why §2.1 had to be amended; see §2.1a and CONTEXT.md entry 4. |
| `UnitOfMeasurement` | string | µg/m³ for everything except CO (mg/m³ — unit trap, check before combining). |
| `DatetimeBegin` / `DatetimeEnd` | datetime, tz-aware | CET/CEST — convert to UTC. |
| `Validity` | integer | Marks readings the operator invalidated. **Not usable as fault-vs-event ground truth** — the value is always replaced by the −999 sentinel (see `Concentration` above, and §2.1a). See code table below. |
| `Verification` | integer | QA review status, independent of fault cause. See table below. |

#### Validity codes (official, dd.eionet.europa.eu/vocabulary/aq/observationvalidity)

| Code | Meaning | Use as |
|---|---|---|
| 1 | Valid | trustworthy |
| 2, 3 | Valid, below detection limit (variants) | trustworthy |
| 4 | Valid (O3 only, specific calibration) | trustworthy |
| **-1** | Not valid | **fault** (general cause) |
| **-99** | Not valid due to station maintenance | **fault** (specific, government-recorded instrument cause — the single most valuable label in the dataset) |

```python
def label_from_validity(v: int) -> str:
    return "fault" if v in (-1, -99) else "trustworthy"
```

#### Verification codes

| Code | Meaning |
|---|---|
| 1 | Verified |
| 2 | Preliminary verified |
| 3 | Not verified |

Do not confuse with Validity. Validity = is this reading correct. Verification = has it been
reviewed yet. Consider down-weighting or excluding `Verification=3` rows from *evaluation* (not
training) near the most recent end of the series.

#### Pollutant codes

| Code | Symbol | Unit |
|---|---|---|
| 1 | SO2 | µg/m³ |
| 5 | PM10 | µg/m³ |
| 6001 | PM2.5 | µg/m³ |
| 7 | O3 | µg/m³ |
| 8 | NO2 | µg/m³ |
| 10 | CO | **mg/m³** |

### 3.2 Secondary / stretch: CPCB (India) — generalization experiment only

- **Source:** CPCB CAAQMS network, via `app.cpcbccr.com` (cumbersome bulk export — 15-min data
  capped at ~1 week per request), `data.gov.in` API (AQI-level min/max/avg, coarser granularity —
  verify it matches the per-pollutant hourly structure you need before relying on it), or OpenAQ
  (free API, but only a 90-day rolling window; full historical archive on S3).
- **No Validity/Verification equivalent exists.** AQICN's own notice: *"data are unvalidated at the
  time of publication."* Any fault label on this data is synthetic — tag it as such, always.
- **Do not pool with EEA data for training or evaluation without going through §5 (the combination
  plan) in full.** If time-constrained, skip CPCB entirely for the MVP (see §10) and revisit as a
  Phase 3 stretch goal.

---

## 4. Repository structure

```
fedairguard/
├── CLAUDE.md                      # this file
├── configs/
│   ├── data.yaml                  # dataset paths, station lists, date ranges
│   ├── model.yaml                 # GRU/GNN hyperparameters
│   └── federation.yaml            # FedAvg rounds, client weighting scheme
├── data/
│   ├── raw/                       # untouched downloads (gitignored)
│   ├── harmonized/                # after §5.1 schema mapping, per-region parquet
│   └── processed/                 # after full §6 pipeline, per-station, train/val/test split
├── src/
│   ├── data/
│   │   ├── fetch_eea.py           # wraps airbase or reads Zenodo dataset
│   │   ├── fetch_cpcb.py          # stretch goal, Phase 3
│   │   ├── harmonize.py           # §5.1 — maps each source into the common schema
│   │   └── station_metadata.py    # station_id -> region, station_class lookup
│   ├── preprocessing/
│   │   ├── indexing.py            # §6.1
│   │   ├── imputation.py          # §6.2 — builds is_imputed mask BEFORE filling
│   │   ├── validity_screen.py     # §6.3
│   │   ├── features.py            # §6.4 — cyclical encodings, lags, rolling stats
│   │   ├── normalize.py           # §6.5 — per-station z-score, fit on train window only
│   │   ├── windowing.py           # §6.6 — sliding window + chronological split
│   │   └── fault_labels.py        # §6.7 — real Validity labels + rate-matched synthetic injection
│   ├── models/
│   │   ├── node_gru.py            # per-station GRU encoder-decoder
│   │   ├── server_gnn.py          # server-side Graph Network over hidden states
│   │   └── decision_layer.py      # residual + spatial agreement -> trustworthy/fault
│   ├── federated/
│   │   ├── client.py              # local training step (FedAvg round)
│   │   ├── server.py              # aggregation (FedAvg for GRU, split learning for GNN)
│   │   └── aggregation_weights.py # §5.7 — region-first weighting, not raw sample count
│   ├── baselines/
│   │   └── ladder.py              # §9.1 — persistence, local GRU, centralised GRU, centralised ST-GNN, FedAvg+GRU
│   └── evaluation/
│       ├── metrics.py             # RMSE/MAE/MAPE, precision/recall/F1/AUC-PR
│       └── report.py              # ALWAYS group by (region, station, label_source) — never pool
├── notebooks/                     # exploration only, nothing here feeds the pipeline directly
├── tests/
└── dashboard/                     # objective 7, built after the core pipeline works
```

---

## 5. Combining EEA (Catalonia) and CPCB (India), if/when you do Phase 3

Full detail — implement every step, this is not optional if CPCB is added at all.

### 5.1 Common schema

```python
import pandas as pd

# Target schema every source maps into:
# station_id | region | timestamp_utc | pollutant | concentration |
# unit | label | label_source | station_class

def harmonize_eea(df: pd.DataFrame, station_class_lookup: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "station_id": df["AirQualityStationEoICode"],
        "region": "catalonia",
        "timestamp_utc": pd.to_datetime(df["DatetimeBegin"], utc=True),
        "pollutant": df["AirPollutant"],
        "concentration": df["Concentration"],
        "unit": df["UnitOfMeasurement"],
        "label": df["Validity"].apply(lambda v: "fault" if v in (-1, -99) else "trustworthy"),
        "label_source": "real",
        "station_class": df["AirQualityStationEoICode"].map(station_class_lookup),
    })

def harmonize_cpcb(df: pd.DataFrame, station_class_lookup: dict) -> pd.DataFrame:
    # Confirm actual CPCB export column names against a real downloaded sample before trusting
    # this mapping — CPCB's export format is less standardized than EEA's.
    return pd.DataFrame({
        "station_id": df["station"],
        "region": "india",
        "timestamp_utc": pd.to_datetime(df["last_update"])
                            .dt.tz_localize("Asia/Kolkata").dt.tz_convert("UTC"),
        "pollutant": df["pollutant_id"],
        "concentration": df["pollutant_avg"],
        "unit": "ug/m3",  # verify against CPCB docs, do not assume
        "label": None,     # filled by rate-matched synthetic injection, see §6.7
        "label_source": "synthetic",
        "station_class": df["station"].map(station_class_lookup),
    })

combined = pd.concat([harmonize_eea(eea_df, sc), harmonize_cpcb(cpcb_df, sc)], ignore_index=True)
```

### 5.2 Timezones

Convert to UTC in the harmonization step itself (above), not later. Barcelona = CET/CEST, India =
IST (fixed +5:30, no DST). Skipping this makes "same hour" mean different real moments across
regions, silently corrupting any cross-station or cross-region time alignment.

### 5.3 Tag every row with `region` and `label_source` — never drop these fields downstream

Every metric in §9 must be groupable by both. This is what makes "recall on real EEA faults" and
"recall on synthetic CPCB faults" two honestly-separate numbers instead of one number that overstates
confidence in the synthetic half.

### 5.4 Per-station normalization (already required by §2.2 / §6.5 — just don't skip it here)

Fit z-score per station on that station's own training window. This absorbs the scale difference
between Barcelona and Indian pollutant levels automatically, by design, without special-casing
India. Do not fit one shared scaler "for consistency" — that reintroduces the leak §2.2 exists to
prevent, and reintroduces the threshold-miscalibration failure mode described below.

### 5.5 Match synthetic injection rate to the measured real EEA fault rate

```python
real_fault_rate = (eea_labels == "fault").mean()
inject_synthetic_faults(cpcb_df, target_rate=real_fault_rate)  # not an arbitrary default
```

### 5.6 Split chronologically, independently per region

A single global date-based split can put e.g. all of India's monsoon season in the test split while
Catalonia's test split covers an unrelated seasonal window. Split 70/15/15 by date *within* each
region separately.

### 5.7 Federated aggregation weighting — do not let it default to raw sample count

Plain FedAvg weights each client by its local data volume. If CPCB ends up with more stations or
more total readings than Catalonia's 10, unweighted FedAvg lets India dominate the global model.
Pick one explicitly:

- **Separate regional models, no cross-region sharing** — simplest, no federated benefit across
  regions.
- **Region-first weighting**: each region gets equal aggregate weight (e.g. 50/50) regardless of
  station count, split evenly among that region's own stations.
- **Cluster before aggregating**, following Hammad et al. (2026) — group stations into
  similarity-based communities (temporal correlation + spatial proximity, extend with a
  region/climate feature) rather than one flat federation across regions.

Implement region-first weighting as the default in `src/federated/aggregation_weights.py`;
clustering is the Phase 3+ stretch version.

### 5.8 The threshold-miscalibration failure mode this whole section exists to prevent

A residual threshold calibrated purely on Catalonia will be miscalibrated for India in two
compounding ways: (1) scale — Indian pollutant swings are larger in absolute terms, so even correct
forecasts produce bigger residuals; (2) relationship — the GRU learns Barcelona's specific
weather-to-pollution relationship (traffic/wind-driven), which doesn't transfer to India's different
dominant sources (biomass burning, construction dust), so forecasts are systematically worse there
even after normalization fixes the scale issue. §5.4 (per-station normalization) fixes issue (1)
automatically. Issue (2) requires either separate per-region calibration of the decision threshold,
or treating India purely as a **generalization test** (train on Catalonia, evaluate transfer to
India without retraining, and report the threshold shift required as a result in itself — this is a
legitimate, reportable finding, not a bug to hide).

---

## 6. Preprocessing pipeline (run independently per station — never concatenate raw station files)

1. **Ingestion & indexing** — merge date+hour into one datetime index per station; verify
   contiguous hourly coverage; log gap length per station.
2. **Missing values with mask retention** — build `is_imputed` boolean mask **before** filling.
   Gaps ≤3h: linear interpolation. Longer: seasonal-naive (same hour, previous day). Carry the mask
   downstream; exclude imputed points from anomaly scoring — the detector must never flag its own
   interpolation.
3. **Physical validity screening** — flag negatives, PM2.5 > PM10 (impossible), out-of-range
   pressure/temperature, flatline runs > 6 identical consecutive values.
4. **Feature engineering** — sine/cosine encode hour/day-of-week/month; wind direction from
   16-point compass to sine/cosine of degrees; lags at t-1, t-24, t-168; rolling mean/std over 6h
   and 24h windows.
5. **Local normalization** — z-score fit on each station's own training window, computed on that
   node. Never share scaling statistics across stations — a shared scaler is itself a privacy leak
   (it encodes information about every station's distribution).
6. **Windowing & split** — 24h input → 6h forecast horizon, stride 1. Chronological 70/15/15
   train/val/test split, by date, never randomly shuffled (shuffling leaks future into past).
7. **Fault labelling** — **primary: rate-matched synthetic injection** (stuck-at, drift, spike,
   noise, dropout, bias offset), at the measured real rate (§2.7) and with episode durations drawn
   from the real `Validity` episode-length distribution. Real `Validity` labels are carried
   alongside as a data-availability baseline — they mark where a reading is *absent*, not where one
   is *wrong*. **AMENDED 2026-10-06 to match §2.1a**; the original ordering (real primary,
   synthetic supplementary) is impossible on this dataset. If CPCB is in play, injection rate must
   match §5.5.

   Injection is applied to the **validation and test splits only**; the training split stays clean.
   That mirrors deployment — you train on verified history, where faults were already removed, and
   run live on unverified readings where faults arrive as plausible numbers. Injecting into
   training also lets the forecaster learn to track drifting and stuck sensors, shrinking the very
   residuals the detector depends on.

---

## 7. Model architecture

### 7.1 Node tier (per station)

- Input: 24h window of pollutant + meteorological features (after §6 preprocessing).
- Model: GRU encoder-decoder, forecasting 1h/3h/6h horizons.
- Output per timestep: forecast ŷ, hidden state h (sent to server for spatial modelling), residual
  e = y − ŷ (kept local, used in the decision layer).
- **Never transmitted from the node:** raw readings, local normalization statistics.
- **Transmitted from the node:** GRU weight updates (for FedAvg), hidden state h (for the
  server-side graph network).

### 7.2 Server tier

- **FedAvg aggregation** of the per-node GRU weights, each round.
- **Graph Network** operating over the uploaded hidden states h (never raw data, never even the
  node's full model for this part — split learning per Meng et al. 2021 / CNFGNN). Learns
  inter-station spatial relationships and produces node embeddings.
- Embeddings sent back down to each station.

### 7.3 Federated training loop (per round)

```
for round in range(n_rounds):
    for station in stations:                      # can be parallel
        local_gru_weights = train_locally(
            station.data, start_from=global_gru_weights, epochs=local_epochs
        )
        hidden_state = station.gru.encode(station.data)
        upload(local_gru_weights, hidden_state)    # weights + hidden state, NEVER raw data

    global_gru_weights = weighted_average(
        [u.local_gru_weights for u in uploads],
        weights=aggregation_weights(uploads)        # see §5.7 — region-first, not raw count
    )
    node_embeddings = server_gnn.forward([u.hidden_state for u in uploads])
    for station, embedding in zip(stations, node_embeddings):
        download(station, global_gru_weights, embedding)
```

### 7.4 Decision layer

For each reading flagged anomalous by residual magnitude:

```
if residual <= threshold:
    label = "trustworthy — baseline"
elif spatial_agreement(embedding, neighbor_embeddings) is high:
    label = "trustworthy — real event"
else:
    label = "sensor fault"
```

`spatial_agreement` is computed server-side from embeddings only — this is the mechanism that
replaces Ferrer-Cid's raw-neighbor-data approach with something that respects the privacy
constraint. This function is the actual novel contribution of the project; expect it to need the
most iteration.

**Threshold note:** if operating across regions (§5), the threshold is per-region-calibrated or the
mismatch is the explicit subject of a generalization experiment (§5.8) — never a single global
threshold silently inherited from whichever region was trained first.

---

## 8. Evaluation plan

### 8.1 Baseline ladder — build in this order, every rung before moving to the next

1. Persistence (last observed value)
2. Local-only per-station GRU (no federation)
3. Centralised GRU (all stations pooled, no privacy constraint — upper reference, not the goal)
4. Centralised ST-GNN (accuracy ceiling — what you'd get with zero privacy constraint at all)
5. FedAvg + GRU (federated forecasting, no spatial reasoning yet)
6. **Proposed model** (federated GRU + server-side GNN + decision layer)

Rungs 1–3 should be working and reported before starting the graph layer (rung 4+). The graph layer
is the highest-risk component — locking the baseline ladder first guarantees a defensible result
even if it underperforms.

### 8.2 Metrics

- **Forecasting:** RMSE, MAE, MAPE — per station, per horizon (1h/3h/6h), per station class.
- **Anomaly:** precision, recall, F1, AUC-PR — measured against rate-matched synthetic faults
  primarily, and separately against real `Validity` labels as a data-availability baseline
  (**AMENDED 2026-10-06 to match §2.1a**; the two are never merged, §5.3). Report **recall per
  fault subtype** as well as in aggregate: the subtypes differ enormously in detectability, and a
  lumped figure built mostly from spikes hides that drift is near-invisible to a residual
  threshold. Per-subtype *precision* does not exist — a false alarm on a healthy reading has no
  subtype — so only recall decomposes.
  Always report a false-positive rate on genuine (real, non-faulty) extreme readings too — recall
  alone is meaningless without this control (this is the Gupchup 2019 lesson: naive detectors
  misclassify up to 45% of real events as faults). **Measured here at up to 66.7% at a 6h horizon**
  (CONTEXT.md entry 5), so this is not a hypothetical risk in this project.
- **Systems:** communication cost per round (MB), rounds to convergence, per-node compute.

### 8.3 Reporting rule — non-negotiable

Every result table is grouped by `(region, station_class, label_source)` at minimum. Never a single
pooled number for anything. This is not extra work bolted onto evaluation — build `report.py`
around this from the start so it's structurally impossible to accidentally produce a misleading
pooled figure.

---

## 9. Build order

**Phase 1 — data & baselines (do this first, all of it, before any federated/graph code):**
1. `fetch_eea.py`, `harmonize.py`, full `preprocessing/` pipeline, on Catalonia data only.
2. Baseline ladder rungs 1–3 (persistence, local GRU, centralised GRU). Get real numbers before
   writing anything else.

**Phase 2 — the actual contribution:**
3. Centralised ST-GNN (rung 4) — establishes the accuracy ceiling.
4. FedAvg + GRU (rung 5) — federation without spatial reasoning.
5. Server-side GNN (split learning) + decision layer (rung 6, the proposed model).
6. Synthetic fault injection module — **the primary Objective 4 evaluation** (§2.1a), reported
   per fault subtype, with real `Validity` labels alongside as a data-availability baseline.
   **AMENDED 2026-10-06.** Built early, during Phase 1, because without it Objective 4 has nothing
   to measure at all.

**Phase 3 — stretch goals, in priority order:**
7. Dashboard (objective 7).
8. CPCB/India integration per §5, framed as a generalization experiment, not a dataset merge for
   primary training.
9. Clustered aggregation (Hammad-style Communities of Interest) if region-first weighting (§5.7)
   proves insufficient.

Do not start Phase 3 before Phase 1 and 2 produce real, reported numbers for every baseline rung.

---

## 10. Key literature (cite in code/docs, don't re-derive their arguments)

| # | Citation | Role in this project |
|---|---|---|
| 1 | Gupchup et al. (2019), arXiv:1902.03492 | Justifies classify-not-detect; the 45% figure is the reason recall-alone metrics are banned in §8.2 |
| 2 | Alduais, Abdullah & Jamil (2019), IEEE Access, 10.1109/ACCESS.2019.2926209 | Single-node validation precedent; replaces retracted citation (§2.5) |
| 3 | Wang et al. (2020), ACM SIGSPATIAL, 10.1145/3397536.3422208 | Justifies physically-informed graph edges for forecasting |
| 4 | Meng, Rambhatla & Liu (2021), ACM SIGKDD, 10.1145/3447548.3467371 | **Architectural base** — the FedAvg + split-learning split in §7.1–7.3 is this paper |
| 5 | Lin et al. (2022), IEEE Access, 10.1109/ACCESS.2022.3204284 | Proves spatial+temporal correlation improves AQ anomaly detection (centralised) — the result we reproduce federated |
| 6 | Ferrer-Cid et al. (2022), IEEE TNSE, 10.1109/TNSE.2022.3169220 | Direct precedent for the decision layer (§7.4) — same idea, but needs raw neighbor data; we replace that with embeddings |
| 7 | D'Elia et al. (2024), Sensors, 10.3390/s24092786 | Real-world evidence drift is its own fault subtype — informs §6.7 fault taxonomy |
| 8 | Khan, Saeed & Koo (2024), Electronics, 10.3390/electronics13244907 | Federated-vs-centralised accuracy benchmark to compare rung 5 against (94.5% vs 96.7%) |
| 9 | Masmoudi et al. (2024), EUSIPCO, 10.23919/EUSIPCO63174.2024.10715088 | Real-hardware confirmation the residual+graph approach is sound |
| 10 | Kim & Moon (2025), Atmosphere, 10.3390/atmos16101136 | Situates the project in the ML-QC-for-AQ-sensors subfield; review, not a code precedent |
| 11 | van Dreven et al. (2026), Expert Systems, 10.1111/exsy.70194 | Non-AQ precedent for the local-statistics-to-server-decision pattern (§7.4) |
| 12 | Hammad et al. (2026), arXiv:2601.05984 | Precedent for clustered/community-based aggregation — see §5.7 and §9 Phase 3 |

---

## 11. Open risks / known unknowns — do not silently resolve these, flag them

- The server-side `spatial_agreement()` function (§7.4) has no direct precedent computing it from
  compressed embeddings rather than raw data. Every cited paper does one or the other, never both.
  This is genuinely untested — expect it to need real experimentation, and report negative results
  honestly if compression turns out to destroy the signal needed.
  **RESOLVED 2026-10-14, and it works.** At 30 stations the embedding-based mechanism adds
  recall +0.119 [0.051, 0.200] over federated training without it, under resampling of both
  stations and time, and matches a centralised model with full raw access on F1
  (+0.002 [-0.034, 0.045]). Compression does **not** destroy the signal. Two honest qualifications:
  it costs precision (-0.191 [-0.288, -0.094]), so it trades rather than dominates; and at 7
  stations nothing was establishable, so the claim is specifically about node sets of this size.
  CONTEXT.md entry 24.
- CPCB's actual export schema (column names, units, whether `pollutant_avg` is hourly or some other
  aggregation) has not been verified against a real downloaded sample — confirm before trusting
  `harmonize_cpcb()` in §5.1.
  **RESOLVED 2026-10-13, and the guess was wrong in four ways.** Verified against the real OpenAQ
  archive: the columns are `location_id | sensors_id | location | datetime | lat | lon | parameter
  | units | value` — not the `data.gov.in` shape §5.1 sketched. (1) `datetime` is already
  ISO-8601 with a UTC offset, so §5.1's `tz_localize("Asia/Kolkata")` would have shifted every
  reading by +5:30. (2) Readings are **quarter-hourly**, not hourly, so they are averaged to hourly
  means. (3) A unit is a property of the *sensor*: a station exposes both a ug/m3 and a ppb sensor
  for the same gas, so any aggregation keyed on (station, pollutant, hour) that ignores `units`
  will average the two together. (4) CO's unit label is simply wrong — values are ~1000x too
  small for ppb and match ppm — so CO is dropped for this source. CONTEXT.md entry 20.
- The dataset's `Validity` field gives binary ground truth (valid/not valid) only. Nothing in EEA
  data separately confirms "this was a genuine pollution event" as opposed to "this was simply a
  normal high reading" — evaluating the event-vs-baseline distinction specifically will need
  external corroboration (e.g. documented Saharan dust intrusion dates), not a dataset field.
  **RESOLVED WORSE THAN EXPECTED, 2026-10-06:** `Validity` turned out to give *less* than binary
  per-reading ground truth, because the reading itself is destroyed (§2.1a). The
  event-vs-baseline corroboration problem above still stands and is untouched by the amendment —
  `is_extreme` in `evaluation/metrics.py` is a quantile proxy for "genuine high reading", not a
  confirmed event list, so `fpr_on_extremes` measures flagging of *unusually high* readings rather
  than of *confirmed* pollution episodes. Sourcing documented episode dates would sharpen it.
