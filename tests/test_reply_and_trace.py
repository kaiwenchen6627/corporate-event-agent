from unittest.mock import patch
import pytest

from agent import CorporateEventAgent, validate_execution_trace

def interp(**requirements):
    return {"customer_updates": {}, "requirement_updates": requirements,
            "operational_updates": {}, "commercial_updates": {}, "commitments": {},
            "next_question": None, "same_event": {"confidence":"high","reason":"same event"}}

class MemoryKnowledgeStore:
    def __init__(self):
        self.records = {"feedback": [], "principles": []}

    def add(self, collection, record):
        saved = {"id": f"{collection}_{len(self.records[collection]) + 1}", **record}
        self.records[collection].append(saved)
        return saved

    def save_record(self, collection, record):
        records = self.records[collection]
        for index, existing in enumerate(records):
            if existing.get("id") == record.get("id"):
                records[index] = dict(record)
                return records[index]
        records.append(dict(record))
        return record

    def list_records(self, collection):
        return list(self.records[collection])

def test_reply_gate_stages_then_sends_human_approved_reply():
    agent = CorporateEventAgent()
    with patch.object(agent, "_llm_interpretation", return_value=interp(service="matcha workshop")), \
         patch.object(agent, "_customer_reply", return_value="How many guests should we plan for?"), \
         patch.object(agent, "reflect", return_value={"principles": []}):
        result = agent.message("We want a matcha workshop.", new_opportunity=True)
    oid = result["opportunity"]["id"]
    assert result["opportunity"]["pending_reply"]["draft"] == "How many guests should we plan for?"
    with patch.object(agent, "reflect", return_value={"principles": []}):
        sent = agent.send_reply(oid, "Could you share the expected headcount?")
    assert sent["ok"] is True and sent["edited"] is True
    assert agent.db[oid].get("pending_reply") is None
    assert agent.db[oid]["conversation"][-1]["approved_by"] == "human"

def test_human_feedback_can_request_ai_revision_before_confirmation():
    agent = CorporateEventAgent()
    with patch.object(agent, "_llm_interpretation", return_value=interp(service="matcha workshop")), \
         patch.object(agent, "_customer_reply", return_value="Original AI draft"):
        result = agent.message("We want a matcha workshop.", new_opportunity=True)
    oid = result["opportunity"]["id"]
    with patch.object(agent, "_llm", return_value="Revised AI draft") as llm:
        revised = agent.revise_pending_reply(
            oid, "Please make this shorter.", "Use a concise, direct tone."
        )
    assert revised["draft"] == "Revised AI draft"
    assert agent.db[oid]["pending_reply"]["draft"] == "Revised AI draft"
    assert agent.db[oid]["pending_reply"]["original_draft"] == "Original AI draft"
    assert not any(m.get("role") == "agent" for m in agent.db[oid]["conversation"])
    prompt = llm.call_args.args[1]
    assert "Use a concise, direct tone." in prompt
    assert "Please make this shorter." in prompt

    with patch.object(agent, "reflect", return_value={"principles": []}) as reflect:
        sent = agent.send_reply(oid, "Revised AI draft", "Use a concise, direct tone.")
    assert sent["ok"] is True
    assert agent.db[oid]["conversation"][-1]["text"] == "Revised AI draft"
    assert reflect.call_args.kwargs["ai_draft"] == "Original AI draft"
    assert reflect.call_args.args[2] == "Round 1: Use a concise, direct tone."
    assert reflect.call_args.kwargs["feedback_notes"] == ["Use a concise, direct tone."]

def test_all_revision_round_notes_are_saved_but_intermediate_edits_are_not():
    agent = CorporateEventAgent()
    agent.persist = lambda state: None
    with patch.object(agent, "_llm_interpretation", return_value=interp(service="matcha workshop")), \
         patch.object(agent, "_customer_reply", return_value="Original AI draft"):
        result = agent.message("We want a matcha workshop.", new_opportunity=True)
    oid = result["opportunity"]["id"]
    store = MemoryKnowledgeStore()
    agent.knowledge_store = store
    notes = [f"Round {i} feedback" for i in range(1, 5)]
    intermediate_edits = [f"private intermediate edit {i}" for i in range(1, 5)]

    with patch.object(agent, "_llm", side_effect=[f"AI revised draft {i}" for i in range(1, 5)]):
        for edited_text, note in zip(intermediate_edits, notes):
            agent.revise_pending_reply(oid, edited_text, note)

    pending = agent.db[oid]["pending_reply"]
    assert pending["feedback_notes"] == notes
    assert not any(text in str(pending) for text in intermediate_edits)

    with patch.object(agent, "_summarize_reflection", return_value={
            "style": ["Use concise wording"], "commercial": [], "capability": [], "rationale": "Shorter emails"}):
        sent = agent.send_reply(oid, pending["draft"], notes[-1])

    assert sent["ok"] is True
    feedback = store.list_records("feedback")[0]
    assert feedback["feedback_notes"] == notes
    assert all(note in feedback["feedback_note"] for note in notes)
    assert feedback["human_revision"] == "AI revised draft 4"
    assert not any(text in str(feedback) for text in intermediate_edits)

def test_failed_ai_revision_keeps_original_pending_draft():
    agent = CorporateEventAgent()
    with patch.object(agent, "_llm_interpretation", return_value=interp(service="matcha workshop")), \
         patch.object(agent, "_customer_reply", return_value="Original AI draft"):
        result = agent.message("We want a matcha workshop.", new_opportunity=True)
    oid = result["opportunity"]["id"]
    with patch.object(agent, "_llm", return_value=None):
        with pytest.raises(RuntimeError, match="AI email revision failed"):
            agent.revise_pending_reply(oid, "Please make this shorter.", "Be concise.")
    assert agent.db[oid]["pending_reply"]["draft"] == "Original AI draft"

def test_execution_trace_contract_is_recorded():
    agent = CorporateEventAgent()
    with patch.object(agent, "_llm_interpretation", return_value=interp(service="matcha workshop")), patch.object(agent, "_customer_reply", return_value="Please share the headcount."):
        result = agent.message("We want a matcha workshop.", new_opportunity=True)
    trace = result["opportunity"]["execution_traces"][0]
    assert validate_execution_trace(trace) is trace
    assert trace["message_id"]
    assert "llm_extraction" in trace["stages"] and "routing" in trace["stages"]
