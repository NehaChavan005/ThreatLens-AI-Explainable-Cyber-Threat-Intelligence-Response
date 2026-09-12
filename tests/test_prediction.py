"""End-to-end tests for the unified prediction pipeline."""

from conftest import make_payload


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"healthy", "degraded"}
    assert body["model_initialized"] is True


def test_predict_attack_detected(client, attack_features):
    r = client.post("/predict", json=make_payload(attack_features))
    assert r.status_code == 200
    body = r.json()
    assert body["is_attack"] is True
    assert body["prediction_class"]
    assert body["confidence"] > 0.5
    assert body["severity"] in {"low", "medium", "high", "critical"}
    assert body["model_backend"] == "xgboost"
    assert isinstance(body["defensive_actions"], list)
    assert len(body["defensive_actions"]) > 0
    assert body["latency_ms"] >= 0.0
    # A defensive action that is advisory-only must carry an approval flag.
    assert all("approval_required" in a for a in body["defensive_actions"])


def test_predict_benign(client, benign_features):
    r = client.post("/predict", json=make_payload(benign_features))
    assert r.status_code == 200
    body = r.json()
    assert body["is_attack"] is False
    assert body["severity"] is None


def test_predict_with_explanation(client, attack_features):
    r = client.post(
        "/predict",
        json=make_payload(attack_features, include_explanation=True),
    )
    assert r.status_code == 200
    body = r.json()
    expl = body["explanation"]
    assert expl is not None
    assert expl["status"] == "ok"
    assert len(expl["top_features"]) <= 6
    assert len(expl["top_features"]) > 0
    assert expl["explanation_source"] == "TreeExplainer"
    assert expl["evidence_summary"]
    assert expl["interpretation"]
    assert expl["prediction_class"] == body["prediction_class"]


def test_predict_with_advisory_falls_back(client, attack_features):
    r = client.post(
        "/predict",
        json=make_payload(attack_features, include_advisory=True),
    )
    assert r.status_code == 200
    body = r.json()
    adv = body["advisory"]
    assert adv is not None
    assert adv["advisory"]
    assert adv["provider"] == "heuristic"


def test_predict_get_alias(client, attack_features):
    feats = {k: v for k, v in attack_features.items() if k not in ("src_ip", "dst_ip")}
    r = client.get("/predict", params=feats)
    assert r.status_code == 200
    body = r.json()
    assert body["is_attack"] is True


def test_response_shape_backward_compatible(client, attack_features):
    r = client.post(
        "/predict",
        json=make_payload(attack_features, include_explanation=True, include_advisory=True),
    )
    assert r.status_code == 200
    body = r.json()
    for key in ("prediction", "prediction_class", "is_attack", "confidence",
                "probabilities", "model_backend", "validation_warnings"):
        assert key in body
    assert isinstance(body["prediction"], list)