"""§5.7 — how much each client counts in the FedAvg average.

Plain FedAvg weights a client by its local sample count. That is the default
almost everywhere and it is the wrong default here. With a second region in
play (§5), whichever region has more stations or more readings would dominate
the global model — and the §5.8 failure mode would then be baked into the
weights rather than merely uncorrected.

LOCKED default: ``region_first``. Every region carries equal aggregate weight
regardless of how many stations it has, split evenly among that region's own
stations. With a single region it reduces to uniform weighting, which is the
right behaviour: all seven Barcelona nodes count the same, so a station with a
longer record cannot quietly outvote one with a shorter one.

``sample_count`` is implemented anyway, because "we chose region-first" is a
claim worth being able to *test* against the alternative rather than assert.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol, Sequence

from src.utils.logging_setup import get_logger

log = get_logger(__name__)

SCHEMES = ("region_first", "sample_count", "uniform", "clustered")


class SupportsWeighting(Protocol):
    """The only fields an upload must expose to be weighted."""
    station_id: str
    region: str
    n_samples: int


def aggregation_weights(uploads: Sequence[SupportsWeighting],
                        scheme: str = "region_first") -> list[float]:
    """Normalised client weights, in the order the uploads were given."""
    if not uploads:
        raise ValueError("no uploads to weight")
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; use one of {SCHEMES}")

    if scheme == "uniform":
        weights = [1.0] * len(uploads)

    elif scheme == "sample_count":
        # Plain FedAvg. Deliberately not the default — see the module docstring.
        weights = [float(u.n_samples) for u in uploads]
        if sum(weights) <= 0:
            raise ValueError("every client reported n_samples <= 0")

    elif scheme == "region_first":
        per_region: dict[str, int] = defaultdict(int)
        for u in uploads:
            per_region[u.region] += 1
        n_regions = len(per_region)
        # Each region gets 1/n_regions, split evenly inside the region.
        weights = [1.0 / (n_regions * per_region[u.region]) for u in uploads]

    else:  # clustered
        raise NotImplementedError(
            "Clustered aggregation (Hammad et al. 2026, Communities of Interest) "
            "is the Phase 3+ stretch version — §5.7 says to reach for it only if "
            "region_first proves insufficient, which has not been shown yet."
        )

    total = sum(weights)
    return [w / total for w in weights]


def describe(uploads: Sequence[SupportsWeighting], scheme: str) -> str:
    """One-line summary for the run log, so the weighting is never implicit."""
    w = aggregation_weights(uploads, scheme)
    parts = ", ".join(f"{u.station_id}={wi:.3f}" for u, wi in zip(uploads, w))
    return f"{scheme}: {parts}"
