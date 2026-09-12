"""Explainability API: /explain returns grounded, analyst-readable SHAP output."""

from conftest import make_payload


def test_explain_endpoint_shape(client, attack_features):
    feats = {k: v for k, v in attack_features.items() if k not in ("src_ip", "dst_ip")}
    r = client.post("/explain", json={"features": feats, "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert len(body["top_features"]) == 5
    assert body["evidence_summary"]
    assert body["interpretation"]
    assert body["explanation_source"] == "TreeExplainer"
    # Prediction context attached for dashboard/GenAI reuse.
    assert body["prediction_class"]
    assert body["is_attack"] is True
    assert body["severity"] is not None


def test_feature_contributions_have_direction(client, attack_features):
    feats = {k: v for k, v in attack_features.items() if k not in ("src_ip", "dst_ip")}
    r = client.post("/explain", json={"features": feats, "top_k": 3})
    for c in r.json()["top_features"]:
        assert c["feature"]
        assert "shap_value" in c
        assert c["direction"] in {"increase", "decrease"}