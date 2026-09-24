import pytest
from agent import validate_decision

def test_decision_validation_accepts_complete_decision():
    assert validate_decision({"id":"B", "name":"Recommended", "price":100, "scope":["setup"]})["id"] == "B"

@pytest.mark.parametrize("decision", [{}, {"id":"A"}, {"id":"A","name":"A","price":100,"scope":"setup"}])
def test_decision_validation_rejects_incomplete_decision(decision):
    with pytest.raises(ValueError): validate_decision(decision)
