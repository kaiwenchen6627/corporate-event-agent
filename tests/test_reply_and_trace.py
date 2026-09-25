from unittest.mock import patch

from agent import CorporateEventAgent, validate_execution_trace

def interp(**requirements):
    return {"customer_updates": {}, "requirement_updates": requirements,
            "operational_updates": {}, "commercial_updates": {}, "commitments": {},
            "next_question": None, "same_event": {"confidence":"high","reason":"same event"}}

def test_reply_gate_stages_then_sends_human_approved_reply():
    agent = CorporateEventAgent()
    with patch.object(agent, "_llm_interpretation", return_value=interp(service="matcha workshop")), patch.object(agent, "_customer_reply", return_value="How many guests should we plan for?"):
        result = agent.message("We want a matcha workshop.", new_opportunity=True)
    oid = result["opportunity"]["id"]
    assert result["opportunity"]["pending_reply"]["draft"] == "How many guests should we plan for?"
    sent = agent.send_reply(oid, "Could you share the expected headcount?")
    assert sent["ok"] is True and sent["edited"] is True
    assert agent.db[oid].get("pending_reply") is None
    assert agent.db[oid]["conversation"][-1]["approved_by"] == "human"

def test_execution_trace_contract_is_recorded():
    agent = CorporateEventAgent()
    with patch.object(agent, "_llm_interpretation", return_value=interp(service="matcha workshop")), patch.object(agent, "_customer_reply", return_value="Please share the headcount."):
        result = agent.message("We want a matcha workshop.", new_opportunity=True)
    trace = result["opportunity"]["execution_traces"][0]
    assert validate_execution_trace(trace) is trace
    assert trace["message_id"]
    assert "llm_extraction" in trace["stages"] and "routing" in trace["stages"]
