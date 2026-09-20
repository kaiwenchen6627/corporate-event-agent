"""Two-stage DeepSeek extraction pipeline for historical event email threads."""
from __future__ import annotations
import json, os, uuid
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from infrastructure.knowledge_store import JsonKnowledgeStore

def now(): return datetime.now(timezone.utc).isoformat()

class KnowledgeImporter:
    def __init__(self, store: JsonKnowledgeStore):
        self.store = store
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com") + "/chat/completions"

    def _llm_json(self, system, prompt):
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for LLM knowledge import")
        payload = {"model": self.model, "temperature": 0.1, "response_format": {"type": "json_object"},
                   "messages": [{"role":"system","content":system},{"role":"user","content":prompt}]}
        request = Request(self.url, data=json.dumps(payload, ensure_ascii=False).encode(), headers={"Authorization":"Bearer "+self.api_key,"Content-Type":"application/json"}, method="POST")
        try:
            with urlopen(request, timeout=90) as response:
                return json.loads(json.loads(response.read())["choices"][0]["message"]["content"])
        except (HTTPError, URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"DeepSeek extraction failed: {exc}") from exc

    def _normalize_thread(self, raw):
        # No entity/requirement regex extraction. This only creates stable source-line references.
        return "\n".join(f"SOURCE_LINE_{i+1}: {line.rstrip()}" for i, line in enumerate(raw.splitlines()) if line.strip())

    def _stage_one(self, normalized):
        return self._llm_json("You analyze corporate event email threads. Return concise JSON only. Never invent facts.", f'''Normalize this complete chronological email thread into a compact decision timeline. Identify only information that changes the event, affects feasibility/commercial terms, creates a commitment, resolves an issue, or remains unresolved. Do not copy full email text or repeat every fact. Preserve source line references and important commercial/operational context. Use null or [] when unavailable.
Return JSON: {{"messages":[{{"message_id":"msg_001","turn":1,"speaker":"customer|company|unknown","timestamp":null,"source_lines":[],"summary":"One concise sentence.","decisions":[],"unresolved_items":[]}}],"requirement_changes":[],"commercial_discussions":[],"operational_discussions":[],"commitments":{{"company":[],"customer":[],"pending":[]}},"final_decisions":[],"outcome_evidence":[]}}
COMPLETE THREAD:\n{normalized}''')

    def _stage_two(self, timeline, source_id):
        return self._llm_json("You extract a very concise, auditable Historical Event Case. Return JSON only.", f'''Create a decision-focused Historical Event Case from this conversation. The complete raw thread is stored separately and must not be copied. Do not repeat facts across fields. Keep only information useful for future event planning decisions.
Hard limits: customer max 3 fields; initial_requirement max 6 fields; current_requirement max 8 fields; requirement_changes max 5 items; operational_state max 6 fields; commercial_state max 6 fields; each commitments list max 5 items; conversation_timeline max 5 key turns; final_solution max 6 fields; reusable_insights max 3 items.
Use null/[] when unavailable. Do not fabricate. Provenance should be compact: add source_message_ids only to important changes, commitments, commercial terms and final solution. Keep reusable insights case-specific unless clearly generalizable; never create global principles automatically.
Return exactly: {{"customer":{{}},"initial_requirement":{{}},"current_requirement":{{}},"requirement_changes":[],"operational_state":{{}},"commercial_state":{{}},"commitments":{{"company":[],"customer":[],"pending":[]}},"conversation_timeline":[],"final_solution":{{}},"outcome":null,"reusable_insights":[]}}
SOURCE_ID: {source_id}
NORMALIZED TIMELINE:\n{json.dumps(timeline, ensure_ascii=False)}''')

    def import_text(self, raw, source_type="email_thread", metadata=None):
        if not raw or not raw.strip(): raise ValueError("raw source is empty")
        source = self.store.add_source(source_type, raw, metadata)
        timeline = self._stage_one(self._normalize_thread(raw))
        case = self._stage_two(timeline, source["id"])
        # Keep extraction metadata light. Full raw source is available through source_ids;
        # the structured case should remain focused enough for retrieval and review.
        compact_timeline = [{k:v for k,v in m.items() if k != "raw_text"} for m in timeline.get("messages", [])]
        record = {"id":"case_candidate_"+uuid.uuid4().hex[:10], "title":"Imported historical event case", "source_ids":[source["id"]], "tags":["imported","historical_event_case"], "extraction":{"method":"deepseek_two_stage","created_at":now(),"timeline":compact_timeline}, "content":case, "review_status":"pending"}
        return self.store.add("historical_cases", record)
