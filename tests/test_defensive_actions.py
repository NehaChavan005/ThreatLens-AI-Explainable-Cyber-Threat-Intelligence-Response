"""Unit tests for the defensive action recommendation engine."""

from app.services.defensive_action_service import defensive_action_service


def test_benign_yields_no_actions():
    actions = defensive_action_service.recommend(
        class_name="BENIGN", confidence=0.99, severity="medium",
        src_ip="1.2.3.4", is_attack=False,
    )
    assert actions == []


def test_attack_yields_playbook():
    actions = defensive_action_service.recommend(
        class_name="Infiltration", confidence=0.95, severity="critical",
        src_ip="10.0.0.5", dst_ip="192.168.1.9", is_attack=True,
    )
    assert actions
    for a in actions:
        assert a.action
        assert a.reason
        assert a.priority in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert a.urgency
        assert a.category
        assert a.approval_required is not None


def test_negative_recommendations_require_approval():
    """The most dangerous containment actions must never auto-execute."""
    actions = defensive_action_service.recommend(
        class_name="Infiltration", confidence=0.98, severity="critical",
        src_ip="10.0.0.5", is_attack=True,
    )
    isolation = [a for a in actions if "Isolate" in a.action]
    assert isolation
    assert isolation[0].approval_required is True
    assert isolation[0].priority == "CRITICAL"
    assert isolation[0].urgency == "Immediate"


def test_dos_playbook_is_attack_specific():
    dos = defensive_action_service.recommend(
        class_name="DoS", confidence=0.9, severity="high",
        src_ip="10.0.0.4", is_attack=True,
    )
    assert dos
    rates = [a for a in dos if "rate limit" in a.action.lower() or "throttl" in a.action.lower()]
    assert rates, "DoS recommendations must mention rate limiting"


def test_supported_attack_classes_nonempty():
    classes = defensive_action_service.supported_attack_classes()
    assert classes
    assert any(c.lower() == "infiltration" for c in classes)


def test_prompt_block_renders_recs():
    actions = defensive_action_service.recommend(
        class_name="Infiltration", confidence=0.95, severity="critical",
        src_ip="10.0.0.5", dst_ip="192.168.1.9", is_attack=True,
    )
    block = defensive_action_service.as_prompt_block(actions)
    assert "1." in block
    assert "[" in block
    assert "analyst approval" in block