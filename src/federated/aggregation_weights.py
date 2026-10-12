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

``clustered`` follows Hammad et al. (2026) — Communities of Interest — and is
the §5.7 / Phase 3 stretch version. It became worth building once the node set
grew past a handful of Barcelona stations: with one region, ``region_first``
degenerates to uniform, so at 69 Catalan stations it offers no structure at
all, and a dense cluster of city-centre traffic sites outvotes a lone rural
one simply by being numerous. See ``cluster_assignments`` for what it is
allowed to cluster *on*, which is a privacy question before it is a modelling
one.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol, Sequence

import numpy as np

from src.utils.logging_setup import get_logger

log = get_logger(__name__)

SCHEMES = ("region_first", "sample_count", "uniform", "clustered")


class SupportsWeighting(Protocol):
    """The only fields an upload must expose to be weighted."""
    station_id: str
    region: str
    n_samples: int


def aggregation_weights(uploads: Sequence[SupportsWeighting],
                        scheme: str = "region_first",
                        *, clusters: dict[str, int] | None = None,
                        ) -> list[float]:
    """Normalised client weights, in the order the uploads were given.

    ``clusters`` maps station_id to a cluster id and is required by the
    ``clustered`` scheme. Build it with :func:`cluster_assignments`; it is
    passed in rather than computed here so one assignment can be reused across
    every round instead of being recomputed (and possibly drifting) each time.
    """
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

    else:  # clustered — Hammad et al. (2026), Communities of Interest
        if not clusters:
            raise ValueError(
                "scheme='clustered' needs a cluster assignment; call "
                "cluster_assignments(uploads, ...) and pass clusters=..."
            )
        missing = [u.station_id for u in uploads if u.station_id not in clusters]
        if missing:
            raise ValueError(f"no cluster assigned for station(s) {missing}")

        # Nested inside region, not instead of it. §5.7's region guarantee is
        # the reason this module exists, and clustering must not quietly undo
        # it: a region with many small communities would otherwise reclaim the
        # majority it was deliberately denied. So each region still carries
        # 1/n_regions, each community inside a region carries an equal share of
        # that, and each station an equal share of its community.
        members: dict[tuple[str, int], int] = defaultdict(int)
        region_clusters: dict[str, set[int]] = defaultdict(set)
        for u in uploads:
            key = (u.region, clusters[u.station_id])
            members[key] += 1
            region_clusters[u.region].add(clusters[u.station_id])
        n_regions = len(region_clusters)
        weights = [
            1.0 / (n_regions
                   * len(region_clusters[u.region])
                   * members[(u.region, clusters[u.station_id])])
            for u in uploads
        ]

    total = sum(weights)
    return [w / total for w in weights]


def cluster_assignments(
    uploads: Sequence[SupportsWeighting],
    *,
    coords: dict[str, tuple[float, float]] | None = None,
    embeddings: dict[str, np.ndarray] | None = None,
    n_clusters: int | None = None,
    spatial_weight: float = 0.5,
) -> dict[str, int]:
    """Group stations into communities, per §5.7 / Hammad et al. (2026).

    **What this is allowed to cluster on is a privacy question first.** Hammad
    builds communities from temporal correlation plus spatial proximity, and
    temporal correlation between two stations ordinarily means comparing their
    two raw series — which §2.4 forbids outright. So neither input here is raw
    data:

    * ``coords`` are public EEA station-registry coordinates. They are not
      observations and were never secret.
    * ``embeddings`` are the encoder hidden states the server *already*
      receives under §7.1/§7.2 to run the graph network. Clustering on them
      discloses nothing new; it reuses something already uploaded. Without
      them (rung 5 sends no hidden states) clustering falls back to geography
      alone, which is weaker but costs nothing.

    Stations are clustered **within** each region, since the weighting nests
    communities inside regions rather than across them.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    from src.models.server_gnn import haversine_km

    if spatial_weight < 0.0 or spatial_weight > 1.0:
        raise ValueError(f"spatial_weight={spatial_weight!r} must be in [0, 1]")

    by_region: dict[str, list[str]] = defaultdict(list)
    for u in uploads:
        by_region[u.region].append(u.station_id)

    out: dict[str, int] = {}
    next_id = 0
    for region, ids in by_region.items():
        n = len(ids)
        k = n_clusters if n_clusters is not None else max(1, round(np.sqrt(n)))
        k = int(np.clip(k, 1, n))
        if k == 1 or n < 3:
            for sid in ids:
                out[sid] = next_id
            next_id += 1
            log.info("clustering | region %s: %d station(s) -> 1 community",
                     region, n)
            continue

        have_coords = coords is not None and all(s in coords for s in ids)
        have_emb = embeddings is not None and all(s in embeddings for s in ids)
        if not (have_coords or have_emb):
            raise ValueError(
                f"region {region!r}: clustering needs public coordinates or "
                "server-side embeddings for every station, and has neither. "
                "Use scheme='region_first' instead of inventing a similarity."
            )

        dist = np.zeros((n, n), dtype=float)
        if have_coords:
            km = np.array([[haversine_km(coords[a], coords[b]) for b in ids]
                           for a in ids])
            # Normalised by the region's own extent, so the blend below mixes
            # two quantities that are both in [0, 1].
            dist += (spatial_weight if have_emb else 1.0) * (km / (km.max() or 1.0))
        if have_emb:
            mat = np.stack([np.asarray(embeddings[s], dtype=float).ravel()
                            for s in ids])
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            unit = mat / np.where(norms == 0.0, 1.0, norms)
            # Cosine distance, mapped from [-1, 1] onto [0, 1].
            cos = np.clip(unit @ unit.T, -1.0, 1.0)
            dist += ((1.0 - spatial_weight) if have_coords else 1.0) * (1.0 - cos) / 2.0

        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0        # linkage needs exact symmetry
        labels = fcluster(linkage(squareform(dist, checks=False), method="average"),
                          t=k, criterion="maxclust")

        for sid, lab in zip(ids, labels):
            out[sid] = next_id + int(lab) - 1
        sizes = np.bincount(labels)[1:]
        log.info("clustering | region %s: %d station(s) -> %d communities "
                 "(sizes %s) from %s", region, n, len(sizes), sizes.tolist(),
                 "coords+embeddings" if (have_coords and have_emb)
                 else "coords only" if have_coords else "embeddings only")
        next_id += int(labels.max())

    return out


def describe(uploads: Sequence[SupportsWeighting], scheme: str,
             *, clusters: dict[str, int] | None = None) -> str:
    """One-line summary for the run log, so the weighting is never implicit."""
    w = aggregation_weights(uploads, scheme, clusters=clusters)
    parts = ", ".join(f"{u.station_id}={wi:.3f}" for u, wi in zip(uploads, w))
    return f"{scheme}: {parts}"
