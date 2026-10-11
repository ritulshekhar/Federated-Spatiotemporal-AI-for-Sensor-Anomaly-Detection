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


def test_clustered_is_refused_until_region_first_proves_insufficient():
    with pytest.raises(NotImplementedError, match="Phase 3"):
        _weights([("A", "catalonia", 1)], "clustered")


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
