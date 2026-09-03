from src.evaluation.metrics import (
    area_error,
    check_schema_validity,
    extract_json,
    hard_constraint_satisfaction,
    relationship_accuracy,
    room_count_accuracy,
)


def test_extract_json_plain():
    parsed, err = extract_json('{"program": []}')
    assert err is None
    assert parsed == {"program": []}


def test_extract_json_with_markdown_fence():
    raw = '```json\n{"program": []}\n```'
    parsed, err = extract_json(raw)
    assert err is None
    assert parsed == {"program": []}


def test_extract_json_with_preamble_text():
    raw = 'Here is the SPEC:\n{"program": [], "zones": []}\nDone.'
    parsed, err = extract_json(raw)
    assert err is None
    assert parsed["program"] == []


def test_extract_json_invalid_returns_error():
    parsed, err = extract_json("not json at all")
    assert parsed is None
    assert err is not None


def test_check_schema_validity_valid():
    spec = {
        "program": [{"type": "BEDROOM", "count": 2, "area_per_room_m2": 12.0, "zone": "PRIVATE"}],
        "zones": [{"type": "PRIVATE", "room_types": ["BEDROOM"]}],
        "relationships": [],
        "circulation": [],
    }
    valid, err = check_schema_validity(spec)
    assert valid
    assert err is None


def test_check_schema_validity_invalid_room_type():
    spec = {"program": [{"type": "SWIMMING_POOL", "count": 1, "area_per_room_m2": 10.0, "zone": "PRIVATE"}], "zones": [], "relationships": [], "circulation": []}
    valid, err = check_schema_validity(spec)
    assert not valid


def test_room_count_accuracy_perfect():
    target = [{"type": "BEDROOM", "count": 2}, {"type": "KITCHEN", "count": 1}]
    predicted = [{"type": "BEDROOM", "count": 2}, {"type": "KITCHEN", "count": 1}]
    assert room_count_accuracy(predicted, target) == 1.0


def test_room_count_accuracy_partial():
    target = [{"type": "BEDROOM", "count": 2}, {"type": "KITCHEN", "count": 1}]
    predicted = [{"type": "BEDROOM", "count": 3}, {"type": "KITCHEN", "count": 1}]
    assert room_count_accuracy(predicted, target) == 0.5


def test_hard_constraint_satisfaction_none_when_no_hard_constraints():
    constraints = [{"type": "ADJACENCY", "priority": "SOFT", "target": "BEDROOM<->BATHROOM"}]
    assert hard_constraint_satisfaction(constraints, []) is None


def test_hard_constraint_satisfaction_required_room():
    constraints = [{"type": "REQUIRED_ROOM", "priority": "HARD", "target": "BALCONY"}]
    predicted = [{"type": "BALCONY", "count": 1}]
    assert hard_constraint_satisfaction(constraints, predicted) == 1.0
    assert hard_constraint_satisfaction(constraints, []) == 0.0


def test_hard_constraint_satisfaction_room_count():
    constraints = [{"type": "ROOM_COUNT", "priority": "HARD", "target": "BEDROOM", "value": 3}]
    assert hard_constraint_satisfaction(constraints, [{"type": "BEDROOM", "count": 3}]) == 1.0
    assert hard_constraint_satisfaction(constraints, [{"type": "BEDROOM", "count": 2}]) == 0.0


def test_relationship_accuracy_perfect_match():
    target = [{"a_type": "BEDROOM", "b_type": "BATHROOM", "relationship": "ADJACENT"}]
    predicted = [{"a_type": "BATHROOM", "b_type": "BEDROOM", "relationship": "ADJACENT"}]  # reversed order
    acc = relationship_accuracy(predicted, target)
    assert acc["precision"] == 1.0
    assert acc["recall"] == 1.0
    assert acc["f1"] == 1.0


def test_relationship_accuracy_no_overlap():
    target = [{"a_type": "BEDROOM", "b_type": "BATHROOM", "relationship": "ADJACENT"}]
    predicted = [{"a_type": "KITCHEN", "b_type": "LIVING", "relationship": "ADJACENT"}]
    acc = relationship_accuracy(predicted, target)
    assert acc["precision"] == 0.0
    assert acc["recall"] == 0.0


def test_area_error_computation():
    predicted = [{"count": 2, "area_per_room_m2": 15.0}, {"count": 1, "area_per_room_m2": 20.0}]  # sum=50
    result = area_error(predicted, 60.0)
    assert result["predicted_total_m2"] == 50.0
    assert result["abs_error_m2"] == 10.0
    assert result["rel_error_pct"] > 0
