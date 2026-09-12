"""Observability, dashboard data API, static dashboard and batch CSV."""

import io

import pandas as pd

from conftest import load_sample


def test_metrics_include_predictions(client, attack_features):
    client.post("/predict", json={"features": attack_features})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "predictions_total" in r.text


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"]


def test_static_dashboard_served(client):
    r = client.get("/dashboard/")
    assert r.status_code == 200
    assert "ThreatLens" in r.text


def test_dashboard_stats(client):
    r = client.get("/api/dashboard/stats?hours=24")
    assert r.status_code == 200
    body = r.json()
    for key in ("total_alerts", "total_incidents", "active_incidents", "alerts_by_severity"):
        assert key in body


def test_dashboard_top_attackers(client, attack_features):
    client.post("/predict", json={"features": attack_features})
    r = client.get("/api/dashboard/top-attackers?limit=5&hours=24")
    assert r.status_code == 200
    rows = r.json()
    assert rows
    assert rows[0]["src_ip"] == "10.0.0.5"
    assert rows[0]["attack_count"] >= 1


def test_dashboard_endpoints_reachable(client):
    for path in (
        "/api/dashboard/recent-alerts?limit=5",
        "/api/dashboard/attack-distribution?hours=24",
        "/api/dashboard/timeline?hours=24&interval_minutes=240",
    ):
        r = client.get(path)
        assert r.status_code == 200, path


def test_batch_csv(client):
    attack = load_sample("sample_attack")
    benign = load_sample("sample_benign")
    rows = []
    for label, src in ((attack, "10.9.9.1"), (benign, "10.9.9.2")):
        row = {**(label), "src_ip": src}
        named = ["dst_ip", "src_ip", "destination_port"] + [k for k in label.keys() if "src_ip" != k]
        rows.append(row)
    # CSV is emitted with the snake_case API keys the batch endpoint accepts.
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "dst_ip"} for r in rows])
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    r = client.post(
        "/predict/batch",
        files={"file": ("flows.csv", buf.getvalue(), "text/csv")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert body["attacks_detected"] >= 1
    assert len(body["items"]) == 2
    for item in body["items"]:
        assert "prediction_class" in item
        assert "is_attack" in item