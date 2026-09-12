"""Alert lifecycle, dedup, incident correlation and incident detail."""

from conftest import make_payload


def test_alert_created_on_attack(client, attack_features):
    r = client.post("/predict", json=make_payload(attack_features))
    body = r.json()
    assert body["alert_id"] is not None

    got = client.get(f"/api/alerts/{body['alert_id']}")
    assert got.status_code == 200
    alert = got.json()
    assert alert["attack_type"] == body["prediction_class"]
    assert alert["src_ip"] == "10.0.0.5"
    assert alert["severity"] == body["severity"]


def test_alert_dedup_returns_existing(client, attack_features):
    first = client.post("/predict", json=make_payload(attack_features)).json()
    second = client.post("/predict", json=make_payload(attack_features)).json()
    assert second["alert_id"] == first["alert_id"]


def test_no_alert_for_benign(client, benign_features):
    r = client.post("/predict", json=make_payload(benign_features))
    assert r.json()["alert_id"] is None


def test_acknowledge_alert(client, attack_features):
    alert_id = client.post("/predict", json=make_payload(attack_features)).json()["alert_id"]
    r = client.post(f"/api/alerts/{alert_id}/acknowledge")
    assert r.status_code == 200
    assert r.json()["acknowledged"] is True


def test_incident_crud(client):
    r = client.post(
        "/api/alerts/incidents",
        json={"title": "Test incident", "severity": "high", "status": "open"},
    )
    assert r.status_code == 200
    inc = r.json()
    assert inc["id"]

    lst = client.get("/api/alerts/incidents").json()
    assert any(i["id"] == inc["id"] for i in lst)

    up = client.put(
        f"/api/alerts/incidents/{inc['id']}",
        json={"status": "investigating", "assigned_to": "analyst-t"},
    )
    assert up.status_code == 200
    assert up.json()["status"] == "investigating"


def _features_with_src(features, src_ip):
    feats = dict(features)
    feats["src_ip"] = src_ip
    return feats


def test_correlation_creates_and_links(client, attack_features, settings_override):
    settings_override(ALERT_CORRELATION_MIN_ALERTS=1)
    src = "10.7.0.1"
    first = client.post(
        "/predict", json=make_payload(_features_with_src(attack_features, src))
    ).json()
    assert first["alert_id"] is not None

    incidents = [i for i in client.get("/api/alerts/incidents").json()
                 if "10.7.0.1" in i["title"]]
    assert incidents, "correlation should have created an incident"
    assert "Campaign from 10.7.0.1" in incidents[-1]["title"]
    assert len(incidents) == 1


def test_correlation_is_idempotent(client, attack_features, settings_override):
    """Dedup returns the same alert on repeat flows and correlation must NOT
    spawn a second incident for the same campaign."""
    settings_override(ALERT_CORRELATION_MIN_ALERTS=1)
    src = "10.7.0.2"
    p1 = client.post("/predict", json=make_payload(_features_with_src(attack_features, src))).json()
    p2 = client.post("/predict", json=make_payload(_features_with_src(attack_features, src))).json()
    assert p2["alert_id"] == p1["alert_id"]

    incidents = [i for i in client.get("/api/alerts/incidents").json()
                 if "10.7.0.2" in i["title"]]
    assert len(incidents) == 1, "repeat flows must not duplicate the incident"


def test_incident_detail_endpoint(client, attack_features, settings_override):
    settings_override(ALERT_CORRELATION_MIN_ALERTS=1)
    alert_id = client.post(
        "/predict", json=make_payload(_features_with_src(attack_features, "10.7.0.3"))
    ).json()["alert_id"]
    incidents = [i for i in client.get("/api/alerts/incidents").json()
                 if "10.7.0.3" in i["title"]]
    assert incidents
    detail_resp = client.get(f"/api/alerts/incidents/{incidents[-1]['id']}/detail")
    assert detail_resp.status_code == 200
    d = detail_resp.json()
    assert d["alert_ids"] and alert_id in d["alert_ids"]
    assert d["total_alerts"] >= 1
    assert d["attack_types"]
    assert d["highest_severity"] in {"low", "medium", "high", "critical"}
    assert d["related_alerts"]
    assert d["defensive_actions"], "incident detail should recommend actions"