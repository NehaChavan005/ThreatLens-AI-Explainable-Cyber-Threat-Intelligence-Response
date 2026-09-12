"""GenAI advisory: LLM disabled → deterministic heuristic must still serve."""

from app.services.genai_service import genai_service


def _pred_summary():
    return {
        "class_id": 1,
        "class_name": "Infiltration",
        "confidence": 0.95,
        "is_attack": True,
        "probabilities": {"Infiltration": 0.95, "BENIGN": 0.05},
    }


def _shap_features():
    return [
        {"feature": "flow_duration", "shap_value": 1.2, "abs_shap_value": 1.2,
         "raw_value": 5000.0, "direction": "increase"},
        {"feature": "tot_fwd_pkts", "shap_value": -0.4, "abs_shap_value": 0.4,
         "raw_value": 10.0, "direction": "decrease"},
        {"feature": "fwd_pkt_len_max", "shap_value": 0.6, "abs_shap_value": 0.6,
         "raw_value": 1514.0, "direction": "increase"},
    ]


def test_advisory_endpoint_falls_back(client):
    r = client.post(
        "/explain/advisory",
        json={
            "prediction": _pred_summary(),
            "top_shap_features": _shap_features(),
            "network_context": {"src_ip": "10.0.0.5", "dst_ip": "192.168.1.9"},
            "severity": "critical",
            "evidence_summary": ["Evidence of command-and-control traffic."],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["advisory"]
    assert body["provider"] == "heuristic"


def test_heuristic_contains_attack_and_evidence():
    adv = genai_service._heuristic_advisory(
        prediction=_pred_summary(),
        top_shap_features=_shap_features(),
        network_context={"src_ip": "10.0.0.5"},
        severity="critical",
        evidence_summary=["Evidence line one.", "Evidence line two."],
        defensive_actions=[{"action": "Block host", "reason": "Infiltrating."}],
    )
    assert "Infiltration" in adv
    assert "Evidence: Evidence line one." in adv
    assert "defensive actions" in adv.lower()
    assert "analyst approval" in adv.lower()


def test_service_enabled_flag_false_when_llm_off():
    assert genai_service.enabled is False