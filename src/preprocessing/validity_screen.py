"""§6.3 — Physical validity screening.

Flags readings that are physically impossible or physically implausible:

* negatives on a strictly non-negative quantity;
* PM2.5 > PM10 (PM2.5 is a strict subset of PM10 — this cannot happen);
* out-of-range values against the generous outer bounds in configs/data.yaml;
* flatline runs of more than N identical consecutive values.

**This is not the fault label.** The EEA `Validity` flag is the ground truth
(§2.1); these flags are an *independent, unsupervised* screen. Keeping them
separate is what lets you ask the interesting question later — how much of the
real fault set a purely physical screen would have caught on its own — which is
the honest floor any learned detector has to beat.

Flags are recorded, never used to delete rows. Deleting them would remove
exactly the readings the classifier in §7.4 exists to adjudicate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.preprocessing.indexing import PHYSICAL_SUFFIX, _run_lengths
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

# Reason codes, one column per reason so the screen is auditable rather than a
# single opaque boolean.
REASONS = ("negative", "out_of_range", "flatline", "pm_inversion")


@dataclass
class ScreenReport:
    station_id: str
    pollutant: str
    n_total: int
    counts: dict[str, int] = field(default_factory=dict)
    n_flagged: int = 0
    # How often the physical screen and the EEA Validity flag agree. Diagnostic
    # only — see the module docstring.
    n_agree_with_validity: int = 0
    n_validity_faults: int = 0

    def as_dict(self) -> dict:
        d = {
            "station_id": self.station_id,
            "pollutant": self.pollutant,
            "n_total": self.n_total,
            "n_flagged": self.n_flagged,
            "n_validity_faults": self.n_validity_faults,
            "n_agree_with_validity": self.n_agree_with_validity,
        }
        d.update({f"n_{r}": self.counts.get(r, 0) for r in REASONS})
        return d


def screen_frame(
    frame: pd.DataFrame,
    pollutants: list[str],
    ranges: dict[str, list[float]],
    flatline_run_length: int = 6,
) -> tuple[pd.DataFrame, list[ScreenReport]]:
    """Add ``<POL>__physical_flag`` plus per-reason columns. Nothing is dropped."""
    frame = frame.copy()
    station_id = frame.attrs.get("station_id", "?")
    reports: list[ScreenReport] = []

    for pol in pollutants:
        if pol not in frame.columns:
            continue
        series = frame[pol]
        values = series.to_numpy(dtype=float)

        negative = values < 0
        lo, hi = (ranges.get(pol) or [-np.inf, np.inf])[:2]
        out_of_range = (values < lo) | (values > hi)
        flat = flatline_mask(values, run_length=flatline_run_length)

        for reason, mask in (("negative", negative), ("out_of_range", out_of_range),
                             ("flatline", flat)):
            frame[f"{pol}__{reason}"] = np.nan_to_num(mask, nan=False).astype(bool)

        flagged = negative | out_of_range | flat
        counts = {
            "negative": int(np.nansum(negative)),
            "out_of_range": int(np.nansum(out_of_range)),
            "flatline": int(flat.sum()),
        }
        frame[pol + PHYSICAL_SUFFIX] = np.nan_to_num(flagged, nan=False).astype(bool)

        rep = ScreenReport(station_id=station_id, pollutant=pol,
                           n_total=len(series), counts=counts,
                           n_flagged=int(np.nansum(flagged)))
        _attach_validity_agreement(frame, pol, flagged, rep)
        reports.append(rep)

    # ---- cross-pollutant: PM2.5 > PM10 is physically impossible --------------
    if "PM2.5" in frame.columns and "PM10" in frame.columns:
        inversion = (frame["PM2.5"] > frame["PM10"]).fillna(False).to_numpy()
        frame["PM2.5__pm_inversion"] = inversion
        frame["PM10__pm_inversion"] = inversion
        for pol in ("PM2.5", "PM10"):
            frame[pol + PHYSICAL_SUFFIX] = frame[pol + PHYSICAL_SUFFIX] | inversion
            for rep in reports:
                if rep.pollutant == pol:
                    rep.counts["pm_inversion"] = int(inversion.sum())
                    rep.n_flagged = int(frame[pol + PHYSICAL_SUFFIX].sum())
                    _attach_validity_agreement(
                        frame, pol, frame[pol + PHYSICAL_SUFFIX].to_numpy(), rep
                    )
        if inversion.any():
            log.info("%s | PM2.5 > PM10 on %d hour(s)", station_id, int(inversion.sum()))

    for rep in reports:
        log.info(
            "%s %-5s | physical flags %d (%.2f%%): %s | overlap with Validity faults "
            "%d/%d",
            rep.station_id, rep.pollutant, rep.n_flagged,
            100 * rep.n_flagged / max(rep.n_total, 1),
            {k: v for k, v in rep.counts.items() if v},
            rep.n_agree_with_validity, rep.n_validity_faults,
        )
    return frame, reports


def flatline_mask(values: np.ndarray, run_length: int = 6) -> np.ndarray:
    """True on every element of a run of more than ``run_length`` identical values.

    NaN never counts as identical to anything, including another NaN, so a gap
    does not read as a flatline.
    """
    n = len(values)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    same = np.zeros(n, dtype=bool)
    same[1:] = (values[1:] == values[:-1]) & ~np.isnan(values[1:])
    for start, end in _run_spans(same):
        # a run of k True in `same` means k+1 identical consecutive values
        if (end - start) + 1 > run_length:
            out[start - 1:end] = True
    return out


def _run_spans(mask: np.ndarray):
    """Yield ``(start, end)`` half-open spans of each maximal True run."""
    if not mask.any():
        return
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, end in zip(edges[0::2], edges[1::2]):
        yield int(start), int(end)


def _attach_validity_agreement(frame: pd.DataFrame, pollutant: str,
                               flagged: np.ndarray, rep: ScreenReport) -> None:
    """Diagnostic overlap between the physical screen and the EEA ground truth."""
    label_col = f"{pollutant}__label"
    if label_col not in frame.columns:
        return
    is_fault = (frame[label_col] == "fault").to_numpy()
    rep.n_validity_faults = int(is_fault.sum())
    rep.n_agree_with_validity = int((is_fault & flagged).sum())


# Re-exported for the run manifest.
__all__ = ["screen_frame", "flatline_mask", "ScreenReport", "REASONS"]
