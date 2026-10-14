# FedAirGuard — Full Technical Report

**Privacy-Preserving Federated Spatiotemporal AI for Distributed Air-Quality
Monitoring and Sensor Anomaly Detection**

*Status: living document. Written 2026-10-14, covering everything from project
start. Sections marked ⏳ are awaiting runs still in flight and will be filled
in as results land.*

---

## How to read this document

This is the long-form account: what the project is, what was built, what the
data actually turned out to be, what was measured, and — at least as
importantly — **what went wrong and what had to be abandoned**. It is written
so that someone who has never seen the repository can follow the reasoning end
to end.

Three companion documents already exist and are not duplicated here:

| File | Role |
|---|---|
| `CLAUDE.md` | The design contract. LOCKED decisions and their justifications. |
| `CONTEXT.md` | Append-only build log, 28 dated entries. The primary evidence. |
| `README.md` | Short orientation and how to run things. |

Where this report states a number, the CONTEXT entry that produced it is cited.
Nothing here is an estimate unless it says so.

**Scale of the work:** 54 Python files, ~12,000 lines, 111 tests, 63 commits,
19 experiment runs under `outputs/`.

---

## 1. The problem, and why it is not the obvious one

### 1.1 The deceptively simple framing

An air-quality station reports an unusually high reading. Two explanations:

1. **A real pollution event** — a traffic surge, a Saharan dust intrusion, a
   nearby fire. The reading is correct and matters.
2. **A sensor fault** — the instrument has drifted, frozen, or failed. The
   reading is wrong and should be discarded.

Treating these the same way is expensive in both directions. Discard real
events and you miss the pollution episodes the network exists to detect; trust
faulty readings and you raise false alarms and corrupt every downstream
average.

### 1.2 Why anomaly detection alone does not solve it

Both cases look identical to a detector that only asks *"is this reading
unusual?"* — because both **are** unusual. That is the entire difficulty, and
it is why this project's central objective is **classification**, not
detection: given that a reading is anomalous, decide *which kind* of anomalous.

Gupchup et al. (2019) quantify the cost of getting this wrong: naive detectors
misclassify **up to 45%** of genuine events as faults. This project measured
its own version of that failure at **up to 66.7%** at a 6-hour horizon early on
(CONTEXT entry 5), which is why every recall figure in this report is
accompanied by a false-positive rate on genuine extreme readings. Recall
reported alone is banned by the design contract (§8.2).

### 1.3 The discriminating signal, and the constraint that makes it hard

The signal that separates the two cases is **spatial**. A real pollution
episode is regional: neighbouring stations see it too. A sensor fault is local:
one instrument disagrees with everyone around it.

So the natural solution is to compare a station against its neighbours. The
problem is that doing so conventionally requires **centralising raw readings**,
which many operators will not do — commercially, legally, or politically.

### 1.4 The research gap

| Prior work | Federated? | Cross-node spatial reasoning? |
|---|---|---|
| Khan et al. (2024), van Dreven et al. (2026) | ✅ | ❌ |
| Lin et al. (2022), Ferrer-Cid et al. (2022), Hammad et al. (2026) | ❌ | ✅ |
| **This project** | ✅ | ✅ |

Nobody had combined them. Ferrer-Cid's decision layer is the direct precedent
for what this project does — but it needs raw neighbour data. The contribution
here is replacing that with **compressed embeddings**, so the comparison still
happens while raw readings never leave the station.

Whether that compression destroys the signal was the project's single largest
open risk (`CLAUDE.md` §11). **It does not** — see §7.2.

---

## 2. Architecture

Three tiers, with a hard privacy boundary between the first and second.

### 2.1 Node tier — one per station

Each station is treated as an AIoT edge node holding its own data.

- **Input:** a 24-hour window of pollutant and meteorological features.
- **Model:** GRU encoder–decoder forecasting 1h / 3h / 6h ahead.
- **Outputs:** a forecast ŷ; an encoder hidden state *h*; and a residual
  *e = y − ŷ* that **never leaves the node**.

Implementation: `src/models/node_gru.py`, `src/federated/client.py`.

### 2.2 The privacy contract

This is the project's entire claim, so it is enforced in code rather than by
convention. `configs/federation.yaml` lists what may leave a node, and
`check_upload()` asserts it **on every upload, every round** — a guard that
only runs in tests is not a guard.

**Permitted to leave:**

| Field | What it is |
|---|---|
| `gru_weight_update` | Model weights for FedAvg |
| `encoder_hidden_state` | Compressed window encoding for the server graph |
| `n_samples` | A scalar count, for reporting only |
| `level_bucket` | A 5-level ordinal (added 2026-10-13, see §8) |

**Never permitted:** raw readings, local normalization statistics, validity
flags, station coordinates (the server already has those from public metadata).

The subtle one is the scaler. A station's mean and standard deviation summarise
its raw distribution, so shipping them would be a disclosure dressed as a
preprocessing detail. `contains_normalization_statistics()` is deliberately
over-eager: a false positive costs a developer thirty seconds, a false negative
silently voids the project's entire claim.

### 2.3 Server tier

- **FedAvg** aggregation of node weights each round, weighted per §5.7 of the
  contract (region-first, *not* raw sample count — see §9.3).
- **Graph network** over uploaded hidden states only, following the split
  learning design of Meng et al. (2021) / CNFGNN.
- Embeddings returned to each node.

**A simplification stated rather than buried:** gradients do not cross the
server boundary. CNFGNN backpropagates through the server GNN into the node
encoders; here the GNN trains on its own objective and the node treats the
embedding as a fixed input. That is weaker, and it is the honest description of
what is implemented.

### 2.4 Decision layer — the actual contribution

```
if residual <= threshold:                      → trustworthy (baseline)
elif spatial_agreement(embedding, neighbours):  → trustworthy (real event)
else:                                          → sensor fault
```

`spatial_agreement` is computed server-side from embeddings alone. This single
function is what replaces Ferrer-Cid's raw-neighbour-data approach, and it is
where the novelty sits.

In practice the layer runs **four detectors at different timescales**, because
no single test catches every fault type: an instantaneous residual, an
accumulated residual, a low-variance test for frozen sensors, and spatial
disagreement. Each is calibrated separately against its own false-positive
budget, so by the union bound the combined rate stays under target no matter
how many run. Implementation: `src/models/decision_layer.py`.

---

## 3. The data, and the discovery that reshaped the project

### 3.1 Source

European Environment Agency Air Quality e-Reporting, Catalonia, **E1a
(verified) only** — never E2a, whose flags are provisional.

| | |
|---|---|
| Region | Catalonia (Generalitat de Catalunya network) |
| Stations available | 71 reporting NO₂ across provinces 08/17/25/43 |
| Stations preprocessed | 69 (2 fail quality gates) |
| Period | 2017–2024 |
| Target | NO₂ |
| Features | PM10, PM2.5, O₃, SO₂ + 7 ERA5 meteorological variables |
| Measured fault rate | **2.805%** |

Meteorology is **not** in the EEA data and is fetched separately from
Open-Meteo's ERA5 archive at each station's registry coordinates.

### 3.2 The finding that broke the original plan

The project was designed around EEA's `Validity` flags as ground truth.
Codes −1 ("not valid") and −99 ("not valid due to station maintenance") mark
readings an operator invalidated — government-recorded instrument faults, which
would be a far stronger evaluation basis than anything synthetic.

The design contract asserted that `Concentration` survives a negative
`Validity` flag, i.e. that you get the wrong reading *and* the label saying it
is wrong.

**That is false.** Measured across the entire Catalonia E1a delivery,
2017–2024:

> **All 27,783 `Validity`-flagged rows, across all five pollutants, carry
> exactly −999.0 — the no-data sentinel.** The relation is biconditional: every
> sentinel is a fault, every fault is a sentinel.

Whatever the instrument actually reported was **replaced during verification**.
A wrong number and a total outage are indistinguishable afterwards.

The consequence is total. Those rows are imputed by the preprocessing pipeline
and excluded from anomaly scoring, which leaves **0 of 7,106 fault targets
evaluable**. The project's primary ground truth contained nothing to measure.
(CONTEXT entry 4.)

### 3.3 What replaced it, and what that costs

The owner chose to amend the LOCKED decision rather than abandon the objective.
Rate-matched **synthetic injection** became the primary ground truth, with
three constraints that keep it from being an arbitrary simulation:

1. **The injection rate is the measured real rate** — 0.02805, never a default.
   An arbitrary rate lets the model learn "region X → more likely faulty" as a
   shortcut instead of learning fault signatures.
2. **Episode durations are sampled from the real `Validity` episode-duration
   distribution.** EEA's flags genuinely record *when* instruments failed and
   *for how long* — that part is real operator evidence, and only the corrupted
   value itself is synthesised.
3. **`label_source` stays `"synthetic"` and is never blended with `"real"`.**

**The cost, stated plainly:** Objective 4's headline becomes "evaluated on
realistic synthetic faults", which is weaker against the novelty claim than
real labels would have been. Real co-located reference-vs-low-cost-sensor
labels remain the recommended improvement and are **not yet done** (§11).

### 3.4 Six fault types

| Type | Signature | Why it matters |
|---|---|---|
| `spike` | Brief extreme excursion | Easiest; dominates naive metrics |
| `noise` | Inflated variance | Moderately detectable |
| `dropout` | Value collapse | Local models struggle |
| `stuck_at` | Frozen at last reading | **Invisible locally** — see below |
| `bias` | Constant offset | Hard |
| `drift` | Slow calibration shift | **Hardest** |

The critical insight, and the reason the aggregate recall number is misleading:
**a frozen or drifting sensor becomes *easier* to forecast, not harder.** The
residual the entire decision layer rests on *shrinks*. No amount of threshold
tuning reaches a signal the residual never contained. This is why detection
rates differ by a factor of ten across fault types and why per-subtype recall
is reported throughout.

### 3.5 The drift injector was wrong, twice

Real co-located sensors drift at roughly **0.01 σ per 72 hours** (measured,
CONTEXT entry 17). The injector drew a **fixed 0.8–2.5 σ total per episode**,
independent of duration. Two independent errors:

1. **~100× too fast.** Every drift number the project reported — including the
   zeros — was measured against a fault two orders of magnitude more obvious
   than the real thing. The zeros were *optimistic*.
2. **Backwards in duration.** A fixed total spread over a longer episode is a
   *slower* drift, so the longest episodes — the hardest cases — were the
   gentlest the injector produced.

Drift is now parameterised by **rate**: `magnitude = rate × (hours/72) × σ`,
with the rate sampled **log-uniformly** over [0.01, 2.0] σ/72h. Log-uniform
rather than uniform because uniform sampling puts half the episodes above
1 σ/72h and almost none near the real rate — which would leave the
per-magnitude table silent about exactly the band that cannot be detected.
(CONTEXT entry 19.)

### 3.6 India — the generalization set

| | |
|---|---|
| Source | CPCB via OpenAQ S3 archive |
| Stations | 7 Delhi |
| Period | 2025-02-18 → 2026-10-09 |
| Rows | 2.9M raw → **392,906 hourly** |

Four schema facts the plan had wrong, all found by inspecting real downloads
(CONTEXT entry 20):

1. **Timestamps already carry a UTC offset.** The planned
   `tz_localize("Asia/Kolkata")` would have shifted every reading by +5:30.
2. **Data is quarter-hourly, not hourly.** Unresampled, four samples would
   collapse onto one hour keeping whichever landed last — discarding three
   quarters of the data.
3. **A unit is a property of the *sensor*, not the dataset.** A station exposes
   both a µg/m³ and a ppb sensor for the same gas; any aggregation keyed on
   (station, pollutant, hour) that ignores units averages the two together.
4. **CO's unit label is simply wrong** — values are ~1000× too small for ppb
   and match ppm. CO is dropped for this source rather than guessed at.

Also measured: **2023 and 2024 are absent entirely** from the archive, for
every station surveyed.

India is a **generalization test only**. It is never trained on; every row is
`label_source="synthetic"`; and both the injection rate and the episode
durations are **borrowed** from the measured EEA values, because CPCB has no
`Validity` equivalent. The manifests record the source as
`real_validity_episodes_borrowed` so it can never be mistaken for a local
measurement.

---

## 4. The preprocessing pipeline

Run **independently per station** — raw station files are never concatenated.

| Step | What it does | Why it matters |
|---|---|---|
| 1. Indexing | One hourly datetime index per station | Gaps logged per station |
| 2. Imputation | `is_imputed` mask built **before** filling; ≤3h linear, longer seasonal-naive, >168h left NaN | The detector must never flag its own interpolation |
| 3. Physical screen | Negatives, PM2.5 > PM10, out-of-range, flatlines >6 | Catches real data errors |
| 4. Features | Cyclical hour/day/month, lags t−1/t−24/t−168, rolling 6h/24h, wind direction → sin/cos | 22 columns after alignment |
| 5. Normalization | Per-station z-score, fit on that station's **own training window** | A shared scaler is itself a privacy leak |
| 6. Windowing | 24h input → 6h horizon, stride 1; chronological 70/15/15 | Shuffling leaks future into past |
| 7. Fault labelling | Rate-matched injection into **val and test only** | See below |

**Why training stays clean** (§6.7): deployment trains on verified history,
where faults were already removed, and runs live on unverified readings where
faults arrive as plausible numbers. Injecting into training would also teach
the forecaster to track drifting and stuck sensors — shrinking the very
residuals the detector depends on.

---

## 5. The baseline ladder

Six rungs, each built and reported before the next, so a defensible result
exists even if the highest-risk component underperforms.

| Rung | Model | Purpose |
|---|---|---|
| 1 | Persistence | Trivial floor |
| 2 | Local GRU per station | No federation |
| 3 | Centralised GRU | Pooled, no privacy constraint |
| 4 | Centralised ST-GNN | Accuracy ceiling, zero privacy constraint |
| 5 | FedAvg + GRU | Federation **without** spatial reasoning |
| 6 | **Proposed** | Federated + server GNN + decision layer |

Rung 5 is the critical control: the difference between 5 and 6 isolates the
**graph layer** specifically, rather than crediting it with federation's
effects.

---

## 6. Results at 30 stations

`outputs/phase5_scale30` — corrected injector, meteorology on, GPU.

### 6.1 The headline, with confidence intervals

Paired block bootstrap at 3h, **resampling stations and time** — the strict
test. At 7 stations nothing survived it.

**Does spatial reasoning help?** (rung 6 − rung 5)

| metric | difference | 95% CI | established |
|---|---|---|---|
| **recall** | **+0.119** | [0.051, 0.200] | **yes** |
| **precision** | **−0.191** | [−0.288, −0.094] | **yes** |
| F1 | +0.059 | [−0.007, 0.136] | no |
| AUC-PR | +0.017 | [−0.051, 0.097] | no |

**The honest headline is not "the proposed model is better".** It is that
cross-station spatial reasoning **trades precision for recall**, and at 30
stations both halves of that trade are established rather than noise. F1 cannot
separate them because gain and cost roughly cancel. Which side matters is an
operating-point decision.

**What does privacy cost?** (rung 6 − rung 4, the no-privacy ceiling)

| metric | difference | 95% CI | established |
|---|---|---|---|
| **F1** | **+0.002** | **[−0.034, 0.045]** | **no — indistinguishable** |

The federated model, which never sees a raw neighbour reading, matches a
centralised model with full raw access on F1.

### 6.2 Per fault type — the sharpest result

At 30 stations with **stations resampled**:

| subtype | difference | 95% CI | established |
|---|---|---|---|
| **dropout** | **+0.348** | [0.221, 0.454] | **yes** |
| **stuck_at** | **+0.328** | [0.209, 0.443] | **yes** |
| **spike** | **−0.088** | [−0.121, −0.052] | **yes — worse** |
| drift / noise / bias | small | straddle zero | no |

The baseline detects a stuck sensor **0.000** of the time; the proposed model
reaches 0.328.

This is mechanically coherent rather than a lucky aggregate. The spatial
mechanism wins exactly where a fault makes a station *easier* to forecast — a
frozen or dropped sensor flattens the series, the local residual shrinks, and
the local detector has nothing to work with. It **loses** on spikes, where the
local residual was already the best available signal and adding a
neighbour-agreement term only dilutes it.

This also demoted an earlier claim: time-only resampling had noise at
−0.125 [−0.203, −0.045], apparently significant; with stations resampled it is
+0.024 [−0.031, 0.083] and not established at all.

### 6.3 Operating points

| tier | recall | precision | F1 | FPR on extremes |
|---|---|---|---|---|
| alert | 0.275 | 0.542 | 0.367 | **0.050** |
| review | 0.544 | 0.102 | 0.171 | 0.567 |

Detection by fault size, cleanly monotonic:

| band | alert | review |
|---|---|---|
| <0.5σ | 0.089 | 0.303 |
| 0.5–1σ | 0.201 | 0.442 |
| 1–2σ | 0.211 | 0.549 |
| 2–5σ | 0.509 | 0.891 |
| **>5σ** | **0.946** | **0.999** |

A factor of ten across magnitude bands is precisely why a single headline
recall number was always misleading.

---

## 7. Results at 69 stations — the full network

`outputs/phase8_full69`. This run was **structurally impossible** until
2026-10-14; see §9.1.

### 7.1 Forecasting — a reversal

| model | RMSE 1h | 3h | 6h |
|---|---|---|---|
| 4 centralised ST-GNN | 12.27 | 14.61 | 15.09 |
| 5 FedAvg / 6 proposed | **11.19** | **13.17** | **13.72** |

**At 69 stations the federated model beats the centralised ceiling at every
horizon** — the opposite of the 30-station result, where centralised won
(13.26 vs 13.80).

The explanation is structural rather than a modelling win. A centralised
spatial model needs stations *synchronised* — it consumes hours when the
network reported together. At 69 stations that synchronisation barely exists
(§9.1), so rung 4 trains on heavily-masked batches and is handicapped.
Federated forecasting needs no synchronisation at all: each client trains on
its own complete record.

So the privacy-preserving method wins here **not because it is better in
principle, but because the centralised alternative depends on a property real
networks do not have.** That is a stronger argument for the approach than any
accuracy delta, and it only becomes visible at realistic network scale.

### 7.2 Detection, 3h, median across stations

| model | recall | precision | F1 | AUC-PR | FPR extremes | stations |
|---|---|---|---|---|---|---|
| 4 centralised ST-GNN | 0.233 | 0.668 | 0.339 | 0.347 | 0.096 | 63 |
| 5 FedAvg | 0.208 | **0.714** | 0.322 | 0.329 | 0.097 | 69 |
| **6 proposed** | **0.345** | 0.529 | **0.401** | **0.354** | 0.111 | 69 |

The proposed model leads on F1 (0.401) and AUC-PR (0.354), both better than at
30 stations (0.389 / 0.350). The precision-for-recall trade persists.

Rung 4 scores only 63 stations — see §9.2.

### 7.3 Per fault type, 69 stations (mean across models, h=1)

| subtype | recall |
|---|---|
| spike | 0.540 |
| noise | 0.463 |
| dropout | 0.132 |
| stuck_at | 0.127 |
| drift | 0.013 |
| bias | 0.009 |

### 7.4 Confidence intervals at 69 stations — two things become established

Paired bootstrap, **stations and time resampled**, rung 6 minus rung 5:

| metric | difference | 95% CI | established |
|---|---|---|---|
| **recall** | **+0.124** | [0.076, 0.181] | **yes** |
| **precision** | **-0.182** | [-0.240, -0.115] | **yes** |
| **F1** | **+0.062** | **[0.018, 0.110]** | **yes - new at this scale** |
| AUC-PR | +0.027 | [-0.022, 0.077] | no |
| FPR on extremes | +0.016 | [-0.004, 0.038] | no |

**F1 now excludes zero.** At 30 stations it was +0.059 [-0.007, 0.136] and the
honest reading was "a trade, not an improvement". At 69 the interval clears
zero, so the proposed model is established as **net better**, not merely
differently balanced. The precision cost is unchanged and still real; what
changed is that the recall gain now outweighs it by a margin the data supports.

Per fault type, stations resampled:

| subtype | difference | 95% CI | established |
|---|---|---|---|
| **stuck_at** | **+0.402** | [0.330, 0.473] | **yes** |
| **dropout** | **+0.362** | [0.262, 0.437] | **yes** |
| **drift** | **+0.049** | **[0.010, 0.097]** | **yes - new at this scale** |
| **spike** | **-0.078** | [-0.098, -0.059] | **yes - worse** |
| noise | -0.024 | [-0.064, 0.018] | no |
| bias | -0.002 | [-0.007, 0.005] | no |

**Drift also crosses into significance** for the first time. At 30 stations it
sat on the boundary (+0.044 [0.000, 0.117]); at 69 it is +0.049 [0.010, 0.097].

The effect is real and it is *tiny*. A 0.049 improvement on a base recall of
~0.013 does not make drift detectable in any operational sense -- it means the
spatial mechanism finally has enough neighbours to register drift at all. The
honest statement is "no longer indistinguishable from zero", not "solved".

Both stuck_at and dropout also strengthened against the 30-station figures
(+0.328 -> +0.402, +0.348 -> +0.362), which is what more neighbours should do
to a mechanism that depends on neighbourhood consensus.

### 7.5 Drift identification (§2.4a)

| | n=7 | n=30 | n=69 |
|---|---|---|---|
| true positives | 0 | 11 | 23 |
| false positives | 1 | 5 | 17 |
| missed | 3 | 6 | 12 |
| precision | — | 0.688 | 0.575 |
| recall | 0.000 | 0.647 | 0.657 |

Recall holds as the network grows; precision falls. See §8.

---

## 8. The privacy amendment (§2.4a)

### 8.1 Why it was necessary

Drift recall sat at **0.000** at the alert operating point for *every* rung —
including the centralised model with full raw access. Not a tuning problem: a
slowly drifting sensor stays easy to forecast, so the residual shrinks. No
server-side work can recover a signal the residual never carried. Either a
drift-sensitive quantity crosses the privacy boundary, or drift is permanently
out of reach.

### 8.2 What was decided

The owner was offered three options — a coarse ordinal, a continuous signed
scalar, or refusing outright — and chose the **coarse ordinal**.

`level_bucket`: one integer in 0–4, one byte, computed **after** z-scoring on
the station's own series, averaged over 72 hours.

### 8.3 Why it is defensible

Because it is computed on normalized data it says *"I currently sit about one
of my own standard deviations above my own training mean"* and nothing about
what that is in µg/m³. The scaler never leaves the node, so absolute level and
scale are both unrecoverable from an integer in 0–4. It is **strictly coarser**
than the `encoder_hidden_state` already permitted, which is a continuous float
vector over the same window.

There is a test asserting that two stations with different physical levels but
the same normalized shape produce the same bucket. If that ever fails, the
field has started leaking scale and the amendment's justification is void.

### 8.4 What it costs

The novelty claim is "cross-node spatial reasoning under a federated
constraint", and the constraint is now one field wider. Any write-up must
describe the uploaded set as *weights, encoder hidden states, and a coarse
ordinal level summary* — **not** "hidden states only".

### 8.5 The measured scope is narrower than designed

The mechanism was described as "one station's bucket rising while its
neighbours stay put is drift". Measured, that requires a condition nobody
anticipated:

**At 7 stations it fails completely — 0 true positives.** All seven Barcelona
stations sat in the same bucket, because stations inside one city co-move: the
neighbour median moves with them and disagreement is zero for everybody,
drifting or not.

**The mechanism needs *dispersion* among neighbours, not merely neighbours.** A
dense single-city cluster is its worst case, not its best — the opposite of
what a reader would assume.

Two further limits: it identifies **stations**, never **readings** (one ordinal
per station per upload), so per-reading drift recall stays ~0.01; and roughly
one flag in three is a genuine local level shift rather than drift.

---

## 9. Scaling: three walls, three different causes

Worth recording because each looked like the previous one and was not.

### 9.1 Wall 1 — zero common window origins

Rungs 4 and 6 required a window origin present at **every** station. Measured:

| stations | common train origins | retained |
|---|---|---|
| ≤15 | 48,892 | 100% |
| 30 | 42,076 | 86.1% |
| 50 | 27,729 | 56.7% |
| **69** | **0** | **0%** |

At the full network **no hour at all** is reported by every station, so the
spatial half could not run. This also retroactively explains two earlier
~18-hour failures: they reached rung 5, which needs no alignment, and would
have died at rung 6 regardless of the memory exhaustion that actually killed
them.

**Fix:** an hour qualifies when ≥60% of stations reported it, with absent
stations zero-filled and flagged in a boolean mask.

**The mask is load-bearing, not cosmetic.** Features are z-scored, so the zero
fill *is* each station's own training mean — an unmasked model reads "absent"
as "sitting exactly at its average", which for a fault detector is the worst
possible default. So absent stations are zeroed before graph mixing, the
adjacency is row-renormalised per sample against who actually reported, the
loss averages over present pairs only, nothing is scattered back for an hour a
station did not report, and spatial disagreement is NaN rather than a confident
reconstruction of a zero vector. Two tests guard it.

### 9.2 Wall 2 — stations with no scoreable overlap

With masking in place, rung 4 trained successfully on all 69 stations and then
died at scoring: *"no finite validation scores to calibrate a threshold on"*.

Not a masking bug. Splits are per-station and chronological, so a station whose
record spans 2017–18 has its validation window in 2018; if the network's
densely-reported hours fall elsewhere, it shares **zero** scoreable origins
with the rest. **6 of 69 stations have exactly zero.** Masked alignment lets the
spatial rungs *see* the whole network; it cannot conjure overlap that does not
exist.

Rung 4 now excludes such stations by name and records them; rung 6 is
unaffected (its forecasts are its own and complete) but names them too, because
a station silently running without its spatial detector **is really running as
rung 5** and must not be read as rung 6.

### 9.3 Wall 3 — memory, twice, for different reasons

First: loading 69 stations when only 30 were needed. Fixed by truncating the
station list **before** any window array is read — truncating after still pays
the full peak.

Second: rung 4 materialised a dense cross-station tensor —
`(48892, 69, 24, 22)` float32 = **6.64 GiB** for training alone — duplicating
per-station arrays already resident. Fixed by assembling each batch on access:
a batch of 128 costs ~19 MB and peak memory stays flat.

### 9.4 The dGPU was never being used

Every run until 2026-10-14, including two ~18-hour ones, was CPU-bound: the
installed wheel was `torch 2.14.0+cpu` on a machine with an idle RTX 4060, so
`torch.cuda.is_available()` was `False`.

Swapped to `torch 2.14.0+cu130` — the same torch version, so a drop-in rather
than a downgrade. Measured speedups: **node GRU 5.4×**, ST-GNN 3.9×, server
reconstructor 0.8× (slower — too small to amortise kernel launches, left
alone). Losses agree to five decimals across devices, which is the check that
matters.

---

## 10. India transfer (§5.8)

A Catalonia-trained model applied to seven Delhi stations **without
retraining**, over the 22 feature columns the regions share.

### 10.1 Forecasting transfers better than predicted

| region | RMSE 1h | RMSE 6h | MAPE 1h | MAPE 6h |
|---|---|---|---|---|
| Catalonia | 15.09 | 20.08 | 34.0% | 68.3% |
| India | 48.90 | 58.32 | 41.7% | **68.7%** |

RMSE is 3.2× worse, which looks damning until you notice MAPE is **essentially
identical at 6h**. The gap is *scale* — Delhi's NO₂ is simply higher. The
design contract predicted the learned weather-to-pollution relationship would
not describe Delhi's biomass and dust sources; measured, that effect is much
smaller than expected.

### 10.2 What actually breaks: calibration stability

| region | min | median | max | spread |
|---|---|---|---|---|
| Catalonia | 1.771 | 2.473 | 2.844 | **1.6×** |
| India | 0.564 | 3.022 | 15.103 | **26.8×** |

The contract asked for the "threshold shift" as the result. That number is
×1.22 — and reporting it alone would have hidden everything that matters. The
failure is a **collapse of calibration stability**, and the median was the
pooling that concealed it.

### 10.3 The counterintuitive consequence

| region | calibration | precision | F1 |
|---|---|---|---|
| India | own validation | **0.092** | 0.107 |
| India | Catalonia's threshold | **0.564** | 0.220 |

**Delhi detects better with Catalonia's threshold than its own.** Two stations
calibrating near 0.5 flag 545 readings between them and annihilate precision;
one borrowed threshold flags 94 and keeps it.

Of the contract's two options, **per-station local calibration is the worse one
here** — Delhi's ~2,100-window validation split is too short to estimate a
0.5%-FPR threshold stably. That is a claim about **validation-window length**,
not about India, and it is testable once the archive's 2023–24 hole stops
capping the Indian window at ~1.6 years.

---

## 11. What did not work, and what remains open

### 11.1 Clustered aggregation — measured and rejected

The contract says to reach for Hammad-style communities only if `region_first`
proves insufficient. At 30 stations it has not: clustering is worse on RMSE at
every horizon (+1.2/+2.0/+3.0%) and on F1 (0.379 vs 0.389), better only on the
false-positive control (0.093 vs 0.130).

The reason matters more than the result. Clustering split the stations
**[18, 9, 1, 1, 1]**, and equal weight per community hands three singleton
stations **60% of the global model between them**. The rule exists so a crowd
cannot outvote a lone station; at this community structure it overcorrects into
exactly the opposite failure. Config stays on `region_first`; the code stays
implemented and tested.

### 11.2 Federation does not help forecasting

At 30 stations, purely local per-station models beat the federated global model
at **every** horizon (13.70/16.18/17.04 vs 13.80/16.55/17.31). One shared
forecaster across heterogeneous stations is worse than many local ones.
Federation earns its place here through the **spatial half**, not weight
sharing. (At 69 stations the comparison changes — see §7.1.)

### 11.3 Drift remains largely unsolved

Per-reading drift recall is ~0.013 even at 69 stations. The station-level
channel works (§7.4) but identifies *stations*, not *readings*.

### 11.4 Open items

| # | Item | Status |
|---|---|---|
| 1 | Level bucket sampled **over time** | Owner's decision — same byte per upload, cumulative disclosure across rounds |
| 2 | **Real co-located fault labels** | Data verified (6,763 pairs, 4 of 5 within 150 m); experiment not built |
| 3 | Masking min_fraction sensitivity | 0.6 chosen, not tuned |
| 4 | Event-vs-baseline ground truth | **Still open** — `fpr_on_extremes` is a quantile proxy, not a confirmed episode list |

Item 2 is the one that would change what the project can claim: every fault
number in this report is measured against synthetic injection.

---

## 12. Engineering lessons

Recorded because each cost real time.

1. **Never pipe a background job through `tail`.** It buffers until EOF, so
   progress is invisible — and the pipeline's exit code is `tail`'s, so a run
   that died on an ImportError was reported as *"completed, exit code 0"*.
2. **Do not edit `src/` while a long job runs.** A lazy import at rung 5 picked
   up a new `client.py` against a `normalize.py` the process had cached before
   the edit. Fifty minutes lost, rungs 1–4 discarded.
3. **A silent fallback is worse than a crash.** The magnitude table hardcoded
   7-station variant names; asked for the 30-station run it warned on 24 of 30
   stations, skipped them, and produced a table formatted identically to a
   correct one. It now refuses when more than half the node set is missing.
4. **Check what the loss is divided by.** The masked loss divided by the
   station-sample count while the numerator summed over the horizon too —
   inflating the reported loss exactly 6×. Harmless to Adam, but it made the
   curve incomparable across rungs.
5. **Non-ASCII breaks heredoc patching on Windows.** `§` and `—` decode under
   the locale codepage, so string matches silently fail.

---

## 13. Reproducing

```bash
python -m src.data.fetch_eea                       # download EEA E1a
python -m src.data.harmonize                       # -> data/harmonized/
python -m src.data.fetch_meteo --start 2017-01-01 --end 2024-12-31
python -m src.preprocessing.pipeline --region catalonia71 --inject-synthetic --variant injected71
python -m src.baselines.ladder --region catalonia71 --variant injected71 \
    --rungs 1 2 3 4 5 6 --no-split-learning --name my_run
python -m src.evaluation.uncertainty outputs/my_run --horizon 3 \
    --compare 6_proposed 5_fedavg_gru --level both
```

India requires a free `OPENAQ_API_KEY` in a gitignored `.env`.

---

## 14. Honest summary

**What is established.** Cross-station spatial reasoning from compressed
embeddings genuinely improves fault detection — recall +0.119 [0.051, 0.200]
under the strict resampling test — and genuinely costs precision,
−0.191 [−0.288, −0.094]. The privacy constraint costs nothing measurable in F1
against a centralised model with full raw access. Both effects are specific to
node sets of ~30+; at 7 stations nothing was establishable.

**What is specific and mechanical.** The mechanism works where faults make a
station easier to forecast — stuck sensors (+0.328) and dropouts (+0.348) — and
is actively worse on spikes (−0.088), where the local residual was already the
best signal.

**What is not solved.** Per-reading drift. Bias. Anything below ~1σ, where
detection is under 0.21 even at the permissive operating point.

**What the project cannot yet claim.** That any of this works on *real* faults.
Every fault number is measured against synthetic injection, because the dataset
destroys the readings its own fault labels point at. That is the honest ceiling
on the current evidence, and closing it requires co-located reference data
rather than more compute.

---

## 15. Changelog of this document

| Date | Change |
|---|---|
| 2026-10-14 | First full version: everything from project start through the 69-station run and its confidence intervals. |

*Next additions will go here as further runs complete. Nothing in this document
is an estimate unless it says so; every figure cites the CONTEXT entry or the
`outputs/` run that produced it.*
