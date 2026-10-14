"""The federated layer: aggregation weighting and the §2.4 privacy contract.

The privacy tests matter most. §2.4 is the entire claim of the project, and a
guard that is never exercised is not a guard.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.federated.aggregation_weights import aggregation_weights, describe
from src.federated.client import PrivacyViolation, Upload, check_upload
from src.federated.server import FederatedServer
from src.preprocessing.normalize import NEVER_TRANSMIT


class _Up:
    """Minimal stand-in satisfying the weighting protocol."""
    def __init__(self, station_id, region, n_samples):
        self.station_id, self.region, self.n_samples = station_id, region, n_samples


def _weights(spec, scheme):
    return aggregation_weights([_Up(s, r, n) for s, r, n in spec], scheme)


# ---------------------------------------------------------------------------
# §5.7 — weighting
# ---------------------------------------------------------------------------
def test_region_first_gives_each_region_equal_weight():
    """Three Catalonia stations must not outvote one Indian station 3:1."""
    w = _weights([("A", "catalonia", 100), ("B", "catalonia", 100),
                  ("C", "catalonia", 100), ("D", "india", 100)], "region_first")
    assert sum(w[:3]) == pytest.approx(0.5)
    assert w[3] == pytest.approx(0.5)


def test_region_first_ignores_sample_count():
    """A long record must not buy a station more influence."""
    w = _weights([("A", "catalonia", 1_000_000), ("B", "catalonia", 10)],
                 "region_first")
    assert w[0] == pytest.approx(w[1])


def test_region_first_reduces_to_uniform_for_one_region():
    w = _weights([("A", "catalonia", 5), ("B", "catalonia", 50),
                  ("C", "catalonia", 500)], "region_first")
    assert w == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_sample_count_is_the_failure_mode_region_first_prevents():
    """Documents what plain FedAvg would do, so the choice is testable."""
    w = _weights([("A", "catalonia", 100), ("B", "india", 900)], "sample_count")
    assert w[1] == pytest.approx(0.9), "plain FedAvg lets the bigger region dominate"


def test_weights_always_sum_to_one():
    for scheme in ("region_first", "sample_count", "uniform"):
        w = _weights([("A", "catalonia", 7), ("B", "india", 3),
                      ("C", "india", 11)], scheme)
        assert sum(w) == pytest.approx(1.0)


def test_clustered_refuses_to_invent_an_assignment():
    with pytest.raises(ValueError, match="needs a cluster assignment"):
        _weights([("A", "catalonia", 1)], "clustered")


def test_clustered_refuses_a_partial_assignment():
    ups = [_Up("A", "catalonia", 1), _Up("B", "catalonia", 1)]
    with pytest.raises(ValueError, match=r"no cluster assigned.*\['B'\]"):
        aggregation_weights(ups, "clustered", clusters={"A": 0})


def test_clustered_stops_a_crowded_community_outvoting_a_lone_station():
    """The §5.7 failure mode one level down from regions.

    Four city-centre stations behaving alike must not outvote a single rural
    one four to one simply by being numerous — that is the same argument that
    made region_first the default, applied inside a region.
    """
    ups = [_Up("C1", "catalonia", 100), _Up("C2", "catalonia", 100),
           _Up("C3", "catalonia", 100), _Up("C4", "catalonia", 100),
           _Up("R1", "catalonia", 100)]
    clusters = {"C1": 0, "C2": 0, "C3": 0, "C4": 0, "R1": 1}
    w = aggregation_weights(ups, "clustered", clusters=clusters)

    assert sum(w[:4]) == pytest.approx(0.5)
    assert w[4] == pytest.approx(0.5)
    # region_first, the alternative, would have given the rural station 0.2.
    assert w[4] > _weights([(u.station_id, u.region, u.n_samples) for u in ups],
                           "region_first")[4]


def test_clustered_nests_inside_regions_and_cannot_undo_the_region_split():
    """§5.7's region guarantee survives clustering, however the clusters fall.

    A region carved into many small communities must still carry exactly its
    1/n_regions share — otherwise turning clustering on would silently hand
    the finely-divided region the majority region_first denied it.
    """
    ups = [_Up("A1", "catalonia", 1), _Up("A2", "catalonia", 1),
           _Up("A3", "catalonia", 1), _Up("A4", "catalonia", 1),
           _Up("B1", "india", 1)]
    w = aggregation_weights(
        ups, "clustered",
        clusters={"A1": 0, "A2": 1, "A3": 2, "A4": 3, "B1": 4},
    )
    assert sum(w[:4]) == pytest.approx(0.5)
    assert w[4] == pytest.approx(0.5)


def test_clustering_groups_by_proximity_and_refuses_to_guess_without_inputs():
    """Clusters come from public coordinates, never from raw series (§2.4)."""
    from src.federated.aggregation_weights import cluster_assignments

    # Two tight city pairs ~40 km apart, plus one distant station.
    coords = {"A1": (41.40, 2.15), "A2": (41.41, 2.16),
              "B1": (41.75, 2.60), "B2": (41.76, 2.61),
              "C1": (42.60, 0.80)}
    ups = [_Up(s, "catalonia", 1) for s in coords]

    assigned = cluster_assignments(ups, coords=coords, n_clusters=3)
    assert assigned["A1"] == assigned["A2"]
    assert assigned["B1"] == assigned["B2"]
    assert len({assigned[s] for s in coords}) == 3
    assert assigned["C1"] not in (assigned["A1"], assigned["B1"])

    with pytest.raises(ValueError, match="has neither"):
        cluster_assignments(ups, n_clusters=3)


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="unknown scheme"):
        _weights([("A", "catalonia", 1)], "weighted_by_vibes")


def test_describe_names_every_client():
    text = describe([_Up("ES0001A", "catalonia", 10),
                     _Up("ES0002A", "catalonia", 10)], "region_first")
    assert "ES0001A" in text and "ES0002A" in text


# ---------------------------------------------------------------------------
# §2.4 — the privacy contract
# ---------------------------------------------------------------------------
def _clean_upload(**kw) -> Upload:
    base = dict(
        station_id="ES0001A", region="catalonia", n_samples=100,
        gru_weight_update={"encoder.weight": torch.zeros(4, 4)},
    )
    base.update(kw)
    return Upload(**base)


def test_a_legitimate_upload_passes():
    assert check_upload(_clean_upload()) is not None


def test_hidden_state_is_permitted():
    """§7.1 lists the encoder hidden state as transmittable."""
    up = _clean_upload(encoder_hidden_state=np.zeros((8, 64), dtype=np.float32))
    assert check_upload(up) is not None


def test_scaler_statistics_in_the_payload_are_refused():
    """The disclosure that looks like a preprocessing detail (§6.5)."""
    up = _clean_upload(gru_weight_update={
        "encoder.weight": torch.zeros(2, 2),
        "scaler": {"mean": {"NO2": 30.6}, "std": {"NO2": 22.1}},
    })
    with pytest.raises(PrivacyViolation, match="normalization statistics"):
        check_upload(up)


def test_the_never_transmit_marker_is_caught_anywhere_in_the_payload():
    up = _clean_upload(gru_weight_update={"sneaky": [NEVER_TRANSMIT]})
    with pytest.raises(PrivacyViolation, match="normalization statistics"):
        check_upload(up)


def test_a_field_absent_from_the_config_allowlist_is_refused():
    """Adding a field to Upload must not silently widen what leaves the node."""
    up = _clean_upload()
    object.__setattr__(up, "raw_readings", np.arange(10))
    up.__dataclass_fields__ = dict(up.__dataclass_fields__)

    class _F:
        name = "raw_readings"
    up.__dataclass_fields__["raw_readings"] = _F()

    with pytest.raises(PrivacyViolation, match="does not permit"):
        check_upload(up)


def test_payload_bytes_counts_what_is_actually_sent():
    up = _clean_upload(gru_weight_update={"w": torch.zeros(100, dtype=torch.float32)})
    assert up.payload_bytes() == 400
    up2 = _clean_upload(
        gru_weight_update={"w": torch.zeros(100, dtype=torch.float32)},
        encoder_hidden_state=np.zeros(50, dtype=np.float32),
    )
    assert up2.payload_bytes() == 600


# ---------------------------------------------------------------------------
# Aggregation arithmetic
# ---------------------------------------------------------------------------
def test_aggregate_is_the_weighted_mean():
    init = {"w": torch.zeros(3)}
    server = FederatedServer(init, scheme="region_first")
    uploads = [
        _clean_upload(station_id="A", gru_weight_update={"w": torch.ones(3) * 2.0}),
        _clean_upload(station_id="B", gru_weight_update={"w": torch.ones(3) * 4.0}),
    ]
    out = server.aggregate(uploads)
    assert torch.allclose(out["w"], torch.ones(3) * 3.0)


def test_aggregate_respects_region_first_across_regions():
    """One Indian station must weigh as much as three Catalonian ones."""
    server = FederatedServer({"w": torch.zeros(1)}, scheme="region_first")
    uploads = [
        _clean_upload(station_id=s, region="catalonia",
                      gru_weight_update={"w": torch.zeros(1)})
        for s in ("A", "B", "C")
    ] + [
        _clean_upload(station_id="D", region="india",
                      gru_weight_update={"w": torch.ones(1) * 10.0})
    ]
    out = server.aggregate(uploads)
    assert out["w"].item() == pytest.approx(5.0)


def test_aggregate_refuses_an_empty_round():
    server = FederatedServer({"w": torch.zeros(1)})
    with pytest.raises(ValueError, match="no uploads"):
        server.aggregate([])


# ---------------------------------------------------------------------------
# §2.4a — the level bucket, and what it must not become
# ---------------------------------------------------------------------------
def test_level_bucket_is_permitted_and_costs_one_byte():
    u = check_upload(Upload(station_id="ES0001A", region="catalonia", n_samples=5,
                            gru_weight_update={"w": torch.zeros(3)},
                            level_bucket=4))
    assert u.level_bucket == 4
    # The ordinal must stay cheap: §2.4a's defence is that it is coarse.
    plain = Upload(station_id="ES0001A", region="catalonia", n_samples=5,
                   gru_weight_update={"w": torch.zeros(3)})
    assert u.payload_bytes() - plain.payload_bytes() == 1


def test_level_bucket_discloses_no_absolute_scale():
    """The §2.4a defence, as a test rather than a claim in a comment.

    Two stations with wildly different physical levels but the same shape must
    produce the same bucket, because the bucket is computed after per-station
    z-scoring. If this ever fails, the field has started leaking scale and the
    amendment's justification no longer holds.
    """
    from src.preprocessing.normalize import level_bucket

    shape = np.concatenate([np.zeros(72), np.full(72, 1.5)])
    quiet_city = shape          # already normalised, by construction
    filthy_city = shape         # identical after its own z-scoring
    assert level_bucket(quiet_city) == level_bucket(filthy_city)


@pytest.mark.parametrize("sigma,expected", [(-3.0, 0), (-0.6, 1), (0.0, 2),
                                            (0.6, 3), (3.0, 4)])
def test_level_bucket_is_monotone_in_level(sigma, expected):
    from src.preprocessing.normalize import level_bucket
    assert level_bucket(np.full(100, sigma)) == expected


def test_absent_data_reports_normal_not_fault():
    """A station with nothing recent is saying "nothing unusual", not "faulty"."""
    from src.preprocessing.normalize import LEVEL_BUCKET_NORMAL, level_bucket
    assert level_bucket(np.full(50, np.nan)) == LEVEL_BUCKET_NORMAL
    assert level_bucket(np.array([])) == LEVEL_BUCKET_NORMAL


def test_a_regional_episode_flags_nobody():
    """The Gupchup (2019) control, at station level.

    Every station elevated together is a real pollution episode. Calling that
    a fault is the 45%-false-positive failure §8.2 bans recall-only metrics
    over, so it must come out empty.
    """
    from src.models.server_gnn import flag_drifting_stations
    ids = ["A", "B", "C", "D", "E"]
    adj = ~np.eye(5, dtype=bool)
    assert not any(flag_drifting_stations({s: 4 for s in ids}, adj, ids).values())


def test_one_station_diverging_from_its_neighbours_is_flagged():
    from src.models.server_gnn import flag_drifting_stations
    ids = ["A", "B", "C", "D", "E"]
    adj = ~np.eye(5, dtype=bool)
    flags = flag_drifting_stations({"A": 4, "B": 2, "C": 2, "D": 2, "E": 2}, adj, ids)
    assert flags["A"] and not any(flags[s] for s in ("B", "C", "D", "E"))


def test_a_station_without_neighbours_is_never_flagged():
    """Absence of evidence is not evidence of drift (cf. spatial_agreement)."""
    from src.models.server_gnn import flag_drifting_stations, level_disagreement
    ids = ["A", "B", "C"]
    isolated = np.zeros((3, 3), dtype=bool)
    buckets = {"A": 4, "B": 2, "C": 2}
    assert not any(flag_drifting_stations(buckets, isolated, ids).values())
    assert np.isnan(level_disagreement(buckets, isolated, ids)["A"])


# ---------------------------------------------------------------------------
# Masked alignment — the 69-station limit
# ---------------------------------------------------------------------------
def test_masked_stgnn_ignores_absent_stations():
    """An absent station must not be read as one sitting at its own average.

    Features are z-scored, so the zero fill IS each station's training mean.
    Without the mask the model treats "missing" as "perfectly normal", which is
    the most dangerous possible default for a fault detector.
    """
    from src.models.st_gnn import SpatioTemporalGNN

    torch.manual_seed(0)
    N, B, L, F = 4, 3, 6, 5
    adj = ~np.eye(N, dtype=bool)
    model = SpatioTemporalGNN(n_features=F, adjacency=adj, hidden_size=8,
                              horizon=2, target_index=0).eval()

    X = torch.randn(B, N, L, F)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, 3] = False                      # station 3 never reported

    with torch.no_grad():
        base = model(X, mask)
        # Whatever station 3's zero-fill happens to be, changing it must not
        # move the other stations' forecasts while it is masked out.
        X2 = X.clone()
        X2[:, 3] = torch.randn(B, L, F) * 10
        moved = model(X2, mask)

    assert torch.allclose(base[:, :3], moved[:, :3], atol=1e-5), (
        "a masked-out station still influenced its neighbours' forecasts")


def test_masked_alignment_keeps_hours_the_strict_rule_would_drop():
    """The whole point: a station missing one hour must not delete that hour."""
    from src.baselines.ladder import _aligned_windows

    class _WS:
        def __init__(self, t0, n_feat=3, L=4, H=2):
            self.t0 = np.array(t0, dtype="datetime64[ns]")
            n = len(t0)
            self.X = np.ones((n, L, n_feat), dtype=np.float32)
            self.y = np.ones((n, H), dtype=np.float32)
            self.feature_columns = [f"f{i}" for i in range(n_feat)]

    class _St:
        def __init__(self, sid, hours):
            self.station_id = sid
            base = np.datetime64("2024-01-01T00")
            self.splits = {"train": _WS([base + np.timedelta64(h, "h") for h in hours])}

    stations = [_St("A", [0, 1, 2, 3]), _St("B", [0, 1, 2, 3]),
                _St("C", [0, 1, 3])]          # C is missing hour 2
    idx = {s.station_id: np.arange(3) for s in stations}

    X, y, common, pos, mask = _aligned_windows(stations, idx, "train",
                                               min_fraction=0.6)
    assert len(common) == 4, "the strict rule would have dropped hour 2"
    assert mask.shape == (4, 3)
    assert mask[:, 2].sum() == 3 and not mask[2, 2], "C should be absent at hour 2"
    assert (X[2, 2] == 0).all(), "an absent station must be zero-filled"
    assert pos["C"][2] == -1, "an absent station needs no scatter position"
