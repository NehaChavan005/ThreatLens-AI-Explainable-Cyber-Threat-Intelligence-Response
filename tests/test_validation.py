"""Input sanitization: negatives clamped, NaN/Inf zeroed, flags capped."""

from conftest import make_payload


def test_negative_values_clamped(client, attack_features):
    feats = dict(attack_features)
    feats["tot_fwd_pkts"] = -5.0
    feats["fwd_pkt_len_max"] = -1.0
    r = client.post("/predict", json=make_payload(feats))
    assert r.status_code == 200
    body = r.json()
    assert any("clamped" in w for w in body["validation_warnings"])


def test_nan_and_inf_sanitized(client, attack_features):
    import json as _json
    import re as _re

    feats = dict(attack_features)
    # httpx refuses NaN/Infinity via json=, so craft the raw JSON body and let
    # FastAPI parse it with Python's json.loads, which accepts NaN literals.
    raw = _re.sub(r'"flow_duration":\s*[0-9eE.\-]+', '"flow_duration": NaN', _json.dumps(feats))
    raw = _re.sub(r'"flow_iat_max":\s*[0-9eE.\-]+', '"flow_iat_max": Infinity', raw)
    r = client.post("/predict", content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 200, "NaN/Infinity must not crash prediction"
    body = r.json()
    assert isinstance(body["prediction_class"], str)
    assert body["confidence"] >= 0.0


def test_flag_values_capped_at_one(client, attack_features):
    feats = dict(attack_features)
    feats["syn_flag_cnt"] = 7.0
    feats["ack_flag_cnt"] = -3.0
    r = client.post("/predict", json=make_payload(feats))
    assert r.status_code == 200
    body = r.json()
    warns = " ".join(body["validation_warnings"])
    assert "clamped to 1" in warns
    assert "clamped to 0" in warns