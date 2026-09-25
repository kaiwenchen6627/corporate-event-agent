"""Golden path tests for CorporateEventAgent.

These tests exercise the three deterministic routing paths in _process_message
without requiring a live LLM.  The LLM extraction layer is monkeypatched to
return a controlled result so the tests are stable and fast.

Path A — Discovery → Solution  (all material requirements present)
Path B — Discovery → Ask       (requirements incomplete)
Path C — Same-event low        → human_review escalation (state preserved)
"""
import json
import pytest
from unittest.mock import patch

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import CorporateEventAgent


# ---------------------------------------------------------------------------
# Shared LLM stub helpers
# ---------------------------------------------------------------------------

def _make_llm_interp(requirement_updates=None, same_event=None):
    """Return a minimal interpretation dict the way _llm_interpretation would."""
    return {
        "customer_updates": {},
        "requirement_updates": requirement_updates or {},
        "operational_updates": {},
        "commercial_updates": {},
        "commitments": {},
        "next_question": None,
        "same_event": same_event or {"confidence": "high", "reason": "same topic"},
    }


# ---------------------------------------------------------------------------
# Path A: complete requirements → solution mode, request_human_review
# ---------------------------------------------------------------------------

class TestGoldenPathComplete:
    """All material requirements supplied in a single message."""

    FULL_REQS = {
        "event_type":            "workshop",
        "pax":                   80,
        "event_date":            "2025-10-20",
        "event_time":            "10am",
        "event_duration_hours":  2,
        "venue":                 "customer office (KL)",
        "service":               "matcha workshop",
    }

    def _stub_interp(self, text, state, gaps):
        return _make_llm_interp(requirement_updates=self.FULL_REQS)

    def test_action_is_request_human_review(self):
        agent = CorporateEventAgent()
        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_interp):
            r = agent.message(
                "Matcha workshop for 80 staff at KL office on 20 Oct, 10am, 2 hours hands-on",
                new_opportunity=True,
            )
        assert r["action"] == "request_human_review", (
            f"expected request_human_review, got {r['action']}"
        )

    def test_mode_is_solution(self):
        agent = CorporateEventAgent()
        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_interp):
            r = agent.message(
                "Matcha workshop for 80 staff at KL office on 20 Oct, 10am, 2 hours",
                new_opportunity=True,
            )
        assert r["opportunity"]["mode"] == "solution"

    def test_solution_options_present(self):
        agent = CorporateEventAgent()
        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_interp):
            r = agent.message(
                "Matcha workshop for 80 staff at KL office on 20 Oct, 10am, 2 hours",
                new_opportunity=True,
            )
        sol = r["opportunity"].get("solution")
        assert sol is not None, "solution should be generated when requirements are sufficient"
        assert len(sol.get("options", [])) >= 3, "expect at least 3 pricing options"

    def test_gaps_sufficient(self):
        agent = CorporateEventAgent()
        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_interp):
            r = agent.message(
                "Matcha workshop for 80 staff at KL office on 20 Oct, 10am, 2 hours",
                new_opportunity=True,
            )
        assert r["gaps"]["sufficient"] is True

    def test_execution_trace_recorded(self):
        """execution_traces must be written and reflect the llm success."""
        agent = CorporateEventAgent()
        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_interp):
            r = agent.message(
                "Matcha workshop for 80 staff at KL office on 20 Oct, 10am, 2 hours",
                new_opportunity=True,
            )
        traces = r["opportunity"].get("execution_traces", [])
        assert len(traces) == 1, "one trace per message"
        stage = traces[0]["stages"].get("llm_extraction", {})
        assert stage.get("success") is True
        assert stage.get("fields_extracted") is not None

    def test_pending_reply_staged_for_review(self):
        """request_human_review reply must also be staged in pending_reply, not auto-sent."""
        agent = CorporateEventAgent()
        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_interp):
            r = agent.message(
                "Matcha workshop for 80 staff at KL office on 20 Oct, 10am, 2 hours",
                new_opportunity=True,
            )
        opp = r["opportunity"]
        assert opp.get("pending_reply") is not None, "pending_reply should be staged"
        assert opp["pending_reply"]["action"] == "request_human_review"
        agent_turns = [m for m in opp["conversation"] if m.get("role") == "agent"]
        assert len(agent_turns) == 0, "no agent turn in conversation before human send approval"

    def test_routing_stage_in_trace(self):
        agent = CorporateEventAgent()
        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_interp):
            r = agent.message(
                "Matcha workshop for 80 staff at KL office on 20 Oct, 10am, 2 hours",
                new_opportunity=True,
            )
        routing = r["opportunity"]["execution_traces"][0]["stages"].get("routing", {})
        assert routing.get("next_action") == "request_human_review"
        assert routing.get("sufficient") is True


# ---------------------------------------------------------------------------
# Path B: incomplete requirements → discovery mode, ask_customer
# ---------------------------------------------------------------------------

class TestGoldenPathIncomplete:
    """Only service supplied; pax, date, venue, duration all missing."""

    PARTIAL_REQS = {"service": "matcha workshop"}

    def _stub_interp(self, text, state, gaps):
        return _make_llm_interp(requirement_updates=self.PARTIAL_REQS)

    def test_action_is_ask_customer(self):
        agent = CorporateEventAgent()
        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_interp):
            r = agent.message("We want a matcha workshop for our team", new_opportunity=True)
        assert r["action"] == "ask_customer"

    def test_mode_is_discovery(self):
        agent = CorporateEventAgent()
        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_interp):
            r = agent.message("We want a matcha workshop for our team", new_opportunity=True)
        assert r["opportunity"]["mode"] == "discovery"

    def test_gaps_missing_fields(self):
        agent = CorporateEventAgent()
        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_interp):
            r = agent.message("We want a matcha workshop for our team", new_opportunity=True)
        missing = r["gaps"]["missing"]
        assert len(missing) > 0, "should report missing material fields"
        assert r["gaps"]["sufficient"] is False

    def test_reply_contains_question(self):
        agent = CorporateEventAgent()
        # stub _customer_reply too so test is LLM-free
        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_interp), \
             patch.object(agent, "_customer_reply", return_value="How many people will attend?"):
            r = agent.message("We want a matcha workshop for our team", new_opportunity=True)
        # Reply is now staged in pending_reply, not yet in conversation
        pending = r["opportunity"].get("pending_reply", {})
        assert "?" in pending.get("draft", ""), "draft reply should contain at least one question"

    def test_pending_reply_staged(self):
        """ask_customer reply must be in pending_reply, NOT in conversation."""
        agent = CorporateEventAgent()
        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_interp), \
             patch.object(agent, "_customer_reply", return_value="How many people will attend?"):
            r = agent.message("We want a matcha workshop for our team", new_opportunity=True)
        opp = r["opportunity"]
        assert opp.get("pending_reply") is not None, "pending_reply should be set"
        assert opp["pending_reply"]["action"] == "ask_customer"
        # conversation should NOT contain any agent turn yet
        agent_turns = [m for m in opp["conversation"] if m.get("role") == "agent"]
        assert len(agent_turns) == 0, "agent reply must not enter conversation before human send approval"

    def test_send_reply_promotes_to_conversation(self):
        """send_reply() should move pending_reply into conversation and clear pending."""
        agent = CorporateEventAgent()
        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_interp), \
             patch.object(agent, "_customer_reply", return_value="How many people will attend?"):
            r = agent.message("We want a matcha workshop for our team", new_opportunity=True)
        oid = r["opportunity"]["id"]
        result = agent.send_reply(oid)
        assert result.get("ok") is True
        state = agent.db[oid]
        assert state.get("pending_reply") is None, "pending_reply should be cleared after send"
        agent_turns = [m for m in state["conversation"] if m.get("role") == "agent"]
        assert len(agent_turns) == 1
        assert agent_turns[0]["approved_by"] == "human"

    def test_send_reply_with_override(self):
        """Human can edit the draft before sending."""
        agent = CorporateEventAgent()
        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_interp), \
             patch.object(agent, "_customer_reply", return_value="How many people will attend?"):
            r = agent.message("We want a matcha workshop for our team", new_opportunity=True)
        oid = r["opportunity"]["id"]
        result = agent.send_reply(oid, override_text="Could you let us know the expected headcount?")
        assert result["edited"] is True
        state = agent.db[oid]
        assert state["conversation"][-1]["text"] == "Could you let us know the expected headcount?"

    def test_execution_trace_routing_action(self):
        agent = CorporateEventAgent()
        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_interp):
            r = agent.message("We want a matcha workshop for our team", new_opportunity=True)
        routing = r["opportunity"]["execution_traces"][0]["stages"].get("routing", {})
        assert routing.get("next_action") == "ask_customer"
        assert routing.get("sufficient") is False


# ---------------------------------------------------------------------------
# Path C: same_event confidence=low → human_review, original state preserved
# ---------------------------------------------------------------------------

class TestGoldenPathDifferentEvent:
    """Second message looks like a different event — should escalate, not merge."""

    FIRST_REQS = {
        "event_type": "workshop",
        "pax": 80,
        "event_date": "2025-10-20",
        "venue": "customer office (KL)",
        "service": "matcha workshop",
    }

    def _stub_first(self, text, state, gaps):
        return _make_llm_interp(requirement_updates=self.FIRST_REQS)

    def _stub_second(self, text, state, gaps):
        # Signals a different event
        return _make_llm_interp(
            requirement_updates={"pax": 500, "event_type": "gala dinner"},
            same_event={"confidence": "low", "reason": "unrelated service and scale"},
        )

    def test_action_is_human_review(self):
        agent = CorporateEventAgent()
        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_first):
            r1 = agent.message("Matcha workshop for 80 pax on 20 Oct at KL office", new_opportunity=True)
        oid = r1["opportunity"]["id"]

        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_second):
            r2 = agent.message(
                "Also need a gala dinner for 500 people next month",
                opportunity_id=oid,
            )
        assert r2["action"] == "human_review_same_event"

    def test_original_state_not_mutated(self):
        """pax must still be 80; the 500-person gala must NOT have been merged."""
        agent = CorporateEventAgent()
        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_first):
            r1 = agent.message("Matcha workshop for 80 pax on 20 Oct at KL office", new_opportunity=True)
        oid = r1["opportunity"]["id"]

        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_second):
            r2 = agent.message(
                "Also need a gala dinner for 500 people next month",
                opportunity_id=oid,
            )
        pax = r2["opportunity"]["requirements"].get("pax", {}).get("value")
        assert pax == 80, f"original pax must be preserved; got {pax}"

    def test_needs_review_flagged(self):
        agent = CorporateEventAgent()
        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_first):
            r1 = agent.message("Matcha workshop for 80 pax on 20 Oct at KL office", new_opportunity=True)
        oid = r1["opportunity"]["id"]

        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_second):
            r2 = agent.message(
                "Also need a gala dinner for 500 people next month",
                opportunity_id=oid,
            )
        nr = r2["opportunity"].get("needs_review")
        assert nr is not None, "needs_review should be set"
        assert nr.get("reason") == "same_event_uncertain"

    def test_same_event_confidence_in_result(self):
        agent = CorporateEventAgent()
        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_first):
            r1 = agent.message("Matcha workshop for 80 pax on 20 Oct at KL office", new_opportunity=True)
        oid = r1["opportunity"]["id"]

        with patch.object(agent, "_llm_interpretation", side_effect=self._stub_second):
            r2 = agent.message(
                "Also need a gala dinner for 500 people next month",
                opportunity_id=oid,
            )
        assert r2["same_event"]["confidence"] == "low"
