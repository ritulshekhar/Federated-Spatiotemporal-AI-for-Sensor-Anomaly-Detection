"""§6.7 — Fault labelling.

**Primary ground truth: the real EEA `Validity` labels** (§2.1, §3.1). Those are
attached in ``src.data.harmonize`` and carried through the whole pipeline
untouched. Nothing in this module is allowed to overwrite them.

This module supplies the *supplementary* half: rate-matched synthetic fault
injection, for the fault subtypes `Validity` does not finely distinguish —
slow drift above all (D'Elia et al. 2024 gives the real-world evidence that
drift is its own subtype). Synthetic results are always a stress test, never
the headline metric.

Two hard constraints, both of which this module enforces rather than documents:

1. **The injection rate must equal the measured real fault rate** (§2.7, §5.5),
   never an arbitrary default. ``inject_faults`` has no default for
   ``target_rate``; you have to pass one, and ``measured_rate_from`` is the
   sanctioned way to get it. Otherwise the model learns "region X -> more
   likely faulty" as a shortcut instead of learning fault signatures.
2. **Injected rows are tagged ``label_source="synthetic"``** and kept in a
   separate label column from the real one, so `report.py` can never average
   the two into a single number (§5.3, §8.3).

Never injected into: rows already flagged faulty by `Validity` (they are
already faults — re-corrupting them measures nothing), and rows that were
imputed (§6.2 — corrupting an interpolation and then detecting it proves
nothing about a sensor).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.preprocessing.indexing import IMPUTED_SUFFIX, LABEL_SUFFIX
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

SYNTHETIC_LABEL_COL_SUFFIX = "__synthetic_label"
SYNTHETIC_TYPE_COL_SUFFIX = "__synthetic_fault_type"

#: The taxonomy from §6.7. `drift` is the one Validity cannot express, which is
#: the main reason this module exists at all.
FAULT_TYPES = ("stuck_at", "drift", "spike", "noise", "dropout", "bias")

# Mean episode length in hours per type. Spikes are near-instantaneous; drift is
# by definition slow, and a short "drift" is indistinguishable from a bias step.
_DURATION_HOURS: dict[str, tuple[int, int]] = {
    "stuck_at": (3, 24),
    "drift": (72, 336),
    "spike": (1, 3),
    "noise": (6, 48),
    "dropout": (2, 18),
    "bias": (12, 96),
}


@dataclass
class InjectionReport:
    station_id: str
    pollutant: str
    target_rate: float
    achieved_rate: float
    n_injected: int
    n_eligible: int
    n_episodes: int
    duration_source: str = "type_defaults"
    n_real_episodes: int = 0
    median_real_duration_h: float = float("nan")
    by_type: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "by_type"}
        d.update({f"n_{t}": self.by_type.get(t, 0) for t in FAULT_TYPES})
        return d


def real_episode_durations(frame: pd.DataFrame, pollutant: str) -> np.ndarray:
    """Lengths, in hours, of each real `Validity`-flagged fault episode.

    LOCKED by §2.1a. EEA's flags cannot tell us what a wrong reading *looks
    like* — the value is always the -999 sentinel — but they do genuinely record
    **when** an instrument failed and **for how long**. That part is real
    operator-recorded evidence, so the injector samples its episode durations
    from this distribution instead of from invented per-type ranges. Only the
    corrupted value itself is synthesised.
    """
    col = pollutant + LABEL_SUFFIX
    if col not in frame.columns:
        return np.empty(0, dtype=int)
    is_fault = (frame[col] == "fault").to_numpy()
    if not is_fault.any():
        return np.empty(0, dtype=int)

    padded = np.concatenate(([False], is_fault, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return (edges[1::2] - edges[0::2]).astype(int)


def measured_rate_from(labels: pd.Series | np.ndarray) -> float:
    """The real fault rate to match (§5.5). Pass REAL-labelled rows only."""
    s = pd.Series(labels).dropna()
    if s.empty:
        raise ValueError(
            "cannot measure a real fault rate from an empty label series. "
            "Synthetic injection must not fall back to an arbitrary default "
            "(CLAUDE.md §2.7)."
        )
    return float((s == "fault").mean())


def inject_faults(
    frame: pd.DataFrame,
    pollutant: str,
    target_rate: float,
    *,
    types: tuple[str, ...] = FAULT_TYPES,
    seed: int = 1337,
    max_attempts_factor: int = 8,
    use_real_durations: bool = True,
) -> tuple[pd.DataFrame, InjectionReport]:
    """Inject rate-matched synthetic faults into ``pollutant``, in place on a copy.

    Adds two columns:
        ``<POL>__synthetic_label``       "fault" / "trustworthy"
        ``<POL>__synthetic_fault_type``  one of FAULT_TYPES, or None

    The real ``<POL>__label`` column is left exactly as it was.
    """
    if not 0.0 < target_rate < 1.0:
        raise ValueError(
            f"target_rate={target_rate!r} must be a measured real fault rate in "
            "(0, 1) — see measured_rate_from() and CLAUDE.md §2.7/§5.5"
        )
    unknown = set(types) - set(FAULT_TYPES)
    if unknown:
        raise ValueError(f"unknown fault type(s): {sorted(unknown)}")

    frame = frame.copy()
    station_id = frame.attrs.get("station_id", "?")
    rng = np.random.default_rng(seed)

    values = frame[pollutant].to_numpy(dtype=float, copy=True)
    n = len(values)

    # ---- eligibility ------------------------------------------------------
    real_fault = (frame.get(pollutant + LABEL_SUFFIX) == "fault").to_numpy() \
        if (pollutant + LABEL_SUFFIX) in frame.columns else np.zeros(n, dtype=bool)
    was_imputed = frame[pollutant + IMPUTED_SUFFIX].to_numpy(dtype=bool) \
        if (pollutant + IMPUTED_SUFFIX) in frame.columns else np.zeros(n, dtype=bool)
    eligible = np.isfinite(values) & ~real_fault & ~was_imputed

    n_eligible = int(eligible.sum())
    if n_eligible == 0:
        raise ValueError(f"{station_id}/{pollutant}: no eligible rows to inject into")

    budget = int(round(target_rate * n_eligible))
    injected = np.zeros(n, dtype=bool)
    fault_type = np.full(n, None, dtype=object)
    by_type: dict[str, int] = {t: 0 for t in types}
    n_episodes = 0

    # §2.1a: prefer the real Validity episode-duration distribution over the
    # invented per-type ranges.
    real_durations = real_episode_durations(frame, pollutant) if use_real_durations \
        else np.empty(0, dtype=int)
    duration_source = "real_validity_episodes" if real_durations.size else "type_defaults"
    if use_real_durations and not real_durations.size:
        log.warning(
            "%s/%s | no real Validity episodes to sample durations from; falling "
            "back to the invented per-type ranges. Say so in any write-up.",
            station_id, pollutant,
        )

    # Budget is split evenly *by row count* across fault types, not by episode
    # count. Picking a type per episode instead lets drift swallow the run: its
    # 72h floor is far above the real median episode length (2h in Catalonia),
    # so uniform episode selection gave drift ~63% of injected rows and left
    # spike with 1.5%. Per-type budgets guarantee every subtype is present in
    # useful numbers, which is what makes per-subtype detection rates (§6.7)
    # reportable at all.
    per_type_budget = {t: budget // len(types) for t in types}
    for t in list(types)[: budget % len(types)]:
        per_type_budget[t] += 1

    for kind in types:
        lo, hi = _DURATION_HOURS[kind]
        attempts = 0
        max_attempts = max(per_type_budget[kind], 1) * max_attempts_factor
        while by_type[kind] < per_type_budget[kind] and attempts < max_attempts:
            attempts += 1
            if real_durations.size:
                # A real episode length, floored at what the type needs to be
                # itself: a 2h "drift" is indistinguishable from a bias step,
                # and a 40h "spike" is not a spike.
                sampled = int(rng.choice(real_durations))
                duration = int(np.clip(sampled, lo,
                                       hi if kind == "spike" else max(hi, sampled)))
            else:
                duration = int(rng.integers(lo, hi + 1))
            remaining = per_type_budget[kind] - by_type[kind]
            duration = min(duration, max(remaining, 1))
            start = int(rng.integers(0, max(n - duration, 1)))
            span = slice(start, start + duration)

            if not eligible[span].all() or injected[span].any():
                continue

            values[span] = _apply_fault(kind, values[span], values, eligible, rng)
            injected[span] = True
            fault_type[span] = kind
            by_type[kind] += duration
            n_episodes += 1

    achieved = float(injected.sum() / n_eligible)
    if injected.sum() < budget:
        log.warning(
            "%s/%s | injected %d of a %d-row budget after %d attempts "
            "(achieved rate %.4f vs target %.4f) — the series has too few "
            "contiguous eligible spans",
            station_id, pollutant, int(injected.sum()), budget, attempts,
            achieved, target_rate,
        )

    frame[pollutant] = values
    frame[pollutant + SYNTHETIC_LABEL_COL_SUFFIX] = np.where(
        injected, "fault", "trustworthy"
    )
    frame[pollutant + SYNTHETIC_TYPE_COL_SUFFIX] = fault_type

    rep = InjectionReport(
        station_id=station_id, pollutant=pollutant,
        target_rate=round(target_rate, 6), achieved_rate=round(achieved, 6),
        n_injected=int(injected.sum()), n_eligible=n_eligible,
        n_episodes=n_episodes, duration_source=duration_source,
        n_real_episodes=int(real_durations.size),
        median_real_duration_h=(float(np.median(real_durations))
                                if real_durations.size else float("nan")),
        by_type=by_type,
    )
    log.info(
        "%s/%s | injected %d rows in %d episode(s), rate %.4f (target %.4f), "
        "durations from %s (%d real episodes, median %.0fh): %s",
        station_id, pollutant, rep.n_injected, rep.n_episodes, achieved,
        target_rate, duration_source, rep.n_real_episodes,
        rep.median_real_duration_h, {k: v for k, v in by_type.items() if v},
    )
    return frame, rep


# ---------------------------------------------------------------------------
def _apply_fault(kind: str, segment: np.ndarray, full: np.ndarray,
                 eligible: np.ndarray, rng) -> np.ndarray:
    """One fault signature, applied to a contiguous segment."""
    seg = segment.copy()
    k = len(seg)
    scale = float(np.nanstd(full[eligible])) or 1.0
    level = float(np.nanmedian(full[eligible]))

    if kind == "stuck_at":
        # Sensor latched at its last good reading.
        return np.full(k, seg[0])

    if kind == "drift":
        # Slow linear calibration drift. The subtype Validity cannot express,
        # and the one a residual-threshold detector is worst at.
        magnitude = rng.uniform(0.8, 2.5) * scale * rng.choice([-1.0, 1.0])
        return seg + np.linspace(0.0, magnitude, k)

    if kind == "spike":
        return seg * rng.uniform(4.0, 12.0)

    if kind == "noise":
        # Degraded signal-to-noise, mean preserved: the variance changes, the
        # level does not, so a level-only detector misses it.
        return seg + rng.normal(0.0, rng.uniform(2.0, 5.0) * scale, k)

    if kind == "dropout":
        # Reports near-zero rather than reporting nothing. Distinct from a gap:
        # a gap is absent data, a dropout is present data that is wrong.
        return rng.uniform(0.0, 0.02 * max(level, 1.0), k)

    if kind == "bias":
        return seg + rng.uniform(0.5, 2.0) * scale * rng.choice([-1.0, 1.0])

    raise ValueError(f"unhandled fault type {kind!r}")


def combined_label_table(frame: pd.DataFrame, pollutant: str) -> pd.DataFrame:
    """Real and synthetic labels side by side, never merged into one column.

    Returned columns: ``label_real``, ``label_synthetic``, ``fault_type``.
    Evaluation groups on ``label_source`` and reads one or the other — §8.3
    forbids collapsing them, so this function deliberately does not offer a
    combined column to reach for.
    """
    out = pd.DataFrame(index=frame.index)
    out["label_real"] = frame.get(pollutant + LABEL_SUFFIX)
    out["label_synthetic"] = frame.get(pollutant + SYNTHETIC_LABEL_COL_SUFFIX)
    out["fault_type"] = frame.get(pollutant + SYNTHETIC_TYPE_COL_SUFFIX)
    return out
