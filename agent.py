#!/usr/bin/env python3
"""Runnable bounded corporate-event collaboration agent.

Usage:
  python3 agent.py demo
  python3 agent.py serve --port 8080
  python3 agent.py message --opportunity-id opp_1 --text '...'
"""
from __future__ import annotations
import argparse, json, os, re, uuid
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from infrastructure.knowledge_store import JsonKnowledgeStore
from ingestion.importer import KnowledgeImporter

ROOT = Path(__file__).parent
STORE = ROOT / "data"
DB = STORE / "opportunities.json"          # legacy single-file store (migrated on first run)
OPPS = STORE / "opportunities"             # one JSON file per opportunity
ROUTES = STORE / "email_routes.json"
KNOWLEDGE = ROOT / "knowledge.json"
EXAMPLES = ROOT / "examples"               # anonymized seeds (committed); bootstrap copies into STORE on first run
EXAMPLE_OPPS = EXAMPLES / "seed_opportunities"
EXAMPLE_KNOWLEDGE = EXAMPLES / "seed_knowledge"

def load_env_file(path=ROOT / ".env"):
    """Load KEY=VALUE pairs from .env; real environment variables always win."""
    if not path.exists(): return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ: os.environ[key] = value

load_env_file()

def now(): return datetime.now(timezone.utc).isoformat()
def load(path, default):
    if not path.exists(): return default
    return json.loads(path.read_text())
def save(path, value):
    path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(value, indent=2, ensure_ascii=False))

class StateUpdater:
    """Pure state boundary: parses facts and records auditable changes."""
    def __init__(self, owner): self.owner = owner
    def apply(self, state, text): return self.owner._extract(text, state)

class EvidenceRetriever:
    """Knowledge access only. It never chooses an action or edits Event State.

    Single retrieval path: the JsonKnowledgeStore (approved records only).
    Keyword scoring happens inside the store; this layer adds the domain map
    and the historical-case pax similarity filter."""
    DOMAINS = {"historical_cases": "historical_cases", "capabilities": "capabilities",
               "sales_principles": "principles"}
    def __init__(self, store): self.store = store
    def retrieve(self, state, domains, tags=None):
        req = state["requirements"]
        pax = req.get("pax", {}).get("value") or 0
        # Query = requirements values + latest customer message + service type.
        # The customer message often carries the actual topic (more specific than req keys).
        query_parts = [str(v.get("value", "")) for v in req.values() if isinstance(v, dict)]
        convo = state.get("conversation") or []
        if convo:
            last_customer = next((m.get("text", "") for m in reversed(convo) if m.get("role") == "customer"), "")
            if last_customer: query_parts.append(last_customer)
        query = " ".join(p for p in query_parts if p) or "event"
        result = {}
        for domain in domains:
            collection = self.DOMAINS.get(domain)
            if not collection: continue
            records = [h["record"] for h in self.store.search([collection], query, 8, tags=tags)]
            if collection == "historical_cases":
                records = [r for r in records if self._pax_similar(r, pax)]
            result[domain] = records
        return result
    @staticmethod
    def _pax_similar(record, pax):
        value = record.get("pax")
        if isinstance(value, dict): value = value.get("value")
        if value is None:
            content = record.get("content")
            if isinstance(content, dict):
                p = content.get("pax")
                value = p.get("value") if isinstance(p, dict) else p
        try: case_pax = int(value)
        except (TypeError, ValueError): return True  # unknown pax: keyword match decides
        return abs(case_pax - pax) <= max(30, pax * .35)

class CorporateEventAgent:
    def __init__(self):
        self.db = self._load_opportunities()
        self.routes = load(ROUTES, {})   # email -> active opportunity_id
        self.deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.deepseek_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com") + "/chat/completions"
        self.state_updater = StateUpdater(self)
        self.knowledge_store = JsonKnowledgeStore(STORE / "knowledge")
        self.knowledge_importer = KnowledgeImporter(self.knowledge_store)
        self.retriever = EvidenceRetriever(self.knowledge_store)
        self._bootstrap_examples_if_empty()
        self._seed_legacy_knowledge()

    # ---------- one-time bootstrap ----------

    def _bootstrap_examples_if_empty(self):
        """On a fresh checkout, copy the committed examples/ fixtures into data/
        so the agent starts with a populated review UI and demo data. Idempotent:
        if any collection under data/ already has content, do nothing."""
        if not EXAMPLES.exists(): return
        # First copy opportunities only if data/opportunities/ is empty
        opps_target = STORE / "opportunities"
        opps_was_empty = (not opps_target.exists()) or (not any(opps_target.glob("*.json")))
        if opps_was_empty and EXAMPLE_OPPS.exists():
            opps_target.mkdir(parents=True, exist_ok=True)
            n = 0
            for src in sorted(EXAMPLE_OPPS.glob("*.json")):
                rec = json.loads(src.read_text())
                (opps_target / src.name).write_text(json.dumps(rec, indent=2, ensure_ascii=False))
                n += 1
            if n: print(f"[agent] bootstrapped {n} example opportunities into {opps_target}")
        # Then knowledge collections
        for coll_name in ("historical_cases", "capabilities", "principles"):
            target = STORE / "knowledge" / coll_name
            example_coll = EXAMPLE_KNOWLEDGE / coll_name
            target_exists = target.exists() and any(target.glob("*.json"))
            if target_exists: continue
            if not example_coll.exists(): continue
            target.mkdir(parents=True, exist_ok=True)
            n = 0
            for src in sorted(example_coll.glob("*.json")):
                rec = json.loads(src.read_text())
                (target / src.name).write_text(json.dumps(rec, indent=2, ensure_ascii=False))
                n += 1
            if n: print(f"[agent] bootstrapped {n} example records into {coll_name}")

    def _seed_legacy_knowledge(self):
        """One-time, idempotent import of the old flat knowledge.json into the store.

        The legacy file was only ever read by the retired in-memory retriever;
        seeding keeps its demo data alive on the single retrieval path. In
        packaged checkouts the canonical source is examples/legacy_knowledge.json;
        we fall back to the repo-root knowledge.json for local development."""
        seed_path = ROOT / "examples" / "legacy_knowledge.json"
        if not seed_path.exists(): seed_path = KNOWLEDGE
        legacy = load(seed_path, None)
        if not legacy: return
        def seed(collection, items, id_prefix):
            count = 0
            for i, item in enumerate(items):
                if isinstance(item, str): item = {"text": item}
                slug = re.sub(r"[^a-z0-9]+", "_", str(item.get("name") or item.get("text") or i).lower()).strip("_")
                rid = id_prefix + (slug or str(i))
                if self.knowledge_store.get(collection, rid): continue
                self.knowledge_store.add(collection, {**item, "id": rid, "review_status": "approved", "origin": "legacy_seed"})
                count += 1
            return count
        n = (seed("historical_cases", legacy.get("cases", []), "seed_case_")
             + seed("capabilities", legacy.get("service", []) + legacy.get("operations", []), "seed_cap_")
             + seed("principles", legacy.get("principles", []), "seed_principle_"))
        if n: print(f"[agent] seeded {n} legacy knowledge records into the knowledge store")

    def _load_opportunities(self):
        """Directory store: one JSON file per opportunity (like the knowledge
        store). Migrates the legacy single-file DB once — keeping it as
        *.migrated backup — then never reads it again. Also patches older
        records to the new reflection/list schema."""
        OPPS.mkdir(parents=True, exist_ok=True)
        db = {}
        for p in sorted(OPPS.glob("*.json")):
            try: rec = json.loads(p.read_text()); db[rec["id"]] = rec
            except (json.JSONDecodeError, KeyError): continue
        if not db and DB.exists():
            for oid, st in load(DB, {}).items():
                (OPPS / f"{oid}.json").write_text(json.dumps(st, indent=2, ensure_ascii=False)); db[oid] = st
            DB.rename(STORE / "opportunities.json.migrated")
            print(f"[agent] migrated {len(db)} opportunities into {OPPS}")
        for st in db.values():
            patched = False
            if "reflection" in st and "reflections" not in st:
                legacy = st.pop("reflection")
                if legacy: st["reflections"] = [legacy]
                else: st["reflections"] = []
                patched = True
            for k in ("style_lessons", "commercial_lessons", "capability_lessons"):
                if k not in st: st[k] = []; patched = True
            if patched:
                (OPPS / f"{st['id']}.json").write_text(json.dumps(st, indent=2, ensure_ascii=False))
        return db

    def _save_opp(self, state):
        OPPS.mkdir(exist_ok=True)
        (OPPS / f"{state['id']}.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))

    def persist(self, state=None):
        """Write one opportunity's file; without an argument, all of them."""
        if state is not None: self._save_opp(state); return
        for st in self.db.values(): self._save_opp(st)

    def _principle_tags_for_state(self, state):
        """Infer which principle buckets matter for this state. capability is
        always relevant (scope constraints are scope-bound); style applies
        whenever the agent will write a reply; commercial applies once we
        have a pax to price against. Returns a set of bucket names."""
        req = state.get("requirements", {}) if isinstance(state.get("requirements"), dict) else {}
        tags = {"capability"}
        if req.get("pax"): tags.add("commercial")
        if state.get("conversation"): tags.add("style")
        return tags

    def _plan(self, state, gaps):
        """Deterministic decision layer (replaces the former LLM decide call).
        Routing derives from the gap analysis; no remote call, no latency, no drift."""
        if gaps["sufficient"]:
            return {"matters_now": ["prepare a feasible first solution from the approved state"],
                    "missing_information": [],
                    "retrieval_needed": ["historical_cases", "capabilities", "sales_principles"],
                    "next_action": "request_human_review",
                    "reason": "all material requirements present (deterministic gap analysis)",
                    "customer_reply_strategy": "Acknowledge the confirmed details and signal readiness to prepare a first solution."}
        missing = gaps["missing"]
        return {"matters_now": ["resolve the highest-value missing requirement"],
                "missing_information": missing,
                "retrieval_needed": ["capabilities"] if "service" in missing else [],
                "next_action": "ask_customer",
                "reason": "missing material fields: " + ", ".join(missing),
                "customer_reply_strategy": "Thank the customer, confirm what is already known, and ask only the highest-value question."}

    def change_impact_analysis(self, state, changes):
        """Deterministic guardrail invoked when an existing commitment changes."""
        impacts=[]
        for change in changes:
            field = change.get("field", "")
            if field in {"event_date", "event_time"}: impacts += ["staff availability", "setup schedule", "supplier deadlines", "quotation validity"]
            if field in {"pax", "service", "event_duration_hours"}: impacts += ["throughput", "staffing", "equipment", "commercial estimate"]
            if field in {"venue", "workshop_format"}: impacts += ["venue access", "footprint", "operational feasibility"]
        return {"triggered": bool(impacts), "changes": changes, "impacts": sorted(set(impacts)), "human_review_required": bool(impacts), "reason": "existing state or commitment may be affected"}

    def _llm(self, system, user, json_mode=False):
        """Call DeepSeek only when configured; workflow remains usable without it."""
        if not self.deepseek_key: return None
        payload = {"model": self.deepseek_model, "temperature": 0.2,
                   "messages": [{"role":"system","content":system},{"role":"user","content":user}]}
        if json_mode: payload["response_format"] = {"type":"json_object"}
        req = Request(self.deepseek_url, data=json.dumps(payload).encode(), headers={
            "Authorization": "Bearer " + self.deepseek_key,
            "Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=45) as response:
                return json.loads(response.read()) ["choices"][0]["message"]["content"]
        except (HTTPError, URLError, TimeoutError, KeyError, json.JSONDecodeError):
            return None

    def _llm_interpretation(self, text, state, gaps):
        """LLM PRIMARY extraction: reads the message first and drives state updates.
        Regex runs afterwards only as a fallback for what the LLM missed."""
        prompt = f'''Current Event Opportunity state: {json.dumps(self._project_for_llm(state), ensure_ascii=False)}
New customer message: {text}
Fields still missing from state (extraction priorities): {json.dumps(gaps)}
You are the PRIMARY extraction layer for state updates. Extract every fact the message
supports, prioritising the missing fields above. A later message may correct an earlier
value — when it does, extract the customer's latest statement.
Return JSON with keys: customer_updates (object, e.g. company/contact_person; do NOT include contact_email — it is set from the mail header),
requirement_updates (object; canonical keys: event_type, pax, event_date, event_time,
event_duration_hours, venue, service, workshop_format; numbers as numbers),
operational_updates (object), commercial_updates (object),
commitments (object with keys company/customer, each a list of plain strings),
next_question (string|null — the single highest-value question to ask next),
same_event (object with keys confidence (high|medium|low) and reason):
judge whether this message is about the SAME event already described in the state,
or a DIFFERENT/new event. Continuity signals: same topic/service, incremental
details (dates, headcount tweaks), references to prior discussion. Divergence
signals: a brand-new purpose, unrelated service or scale, "another event",
"separately", "in addition". Adjustments to the same event (new headcount, new
date) are SAME-event, not new. Use low ONLY when the message looks like a
different event; when unsure use medium.
Leave a key empty rather than guessing. Do not invent prices, commitments, dates, or capabilities.'''
        raw = self._llm("You are a bounded corporate event collaboration agent. Preserve existing state; never overwrite an agreed commercial term silently.", prompt, True)
        if not raw: return None
        try: return json.loads(raw)
        except json.JSONDecodeError: return None

    def _merge_interpretation(self, state, interp):
        """Apply LLM interpretation: PRIMARY layer — may set or correct any field.
        Every change is recorded in history for audit. Returns the list of applied fields."""
        def empty(v): return v is None or v == "" or v == [] or v == {}
        if not isinstance(interp, dict): return []
        applied = []
        req = state["requirements"]
        for key, value in (interp.get("requirement_updates") or {}).items():
            if empty(value): continue
            existing = req.get(key)
            if existing and existing.get("value") == value: continue  # no-op
            req[key] = {"value": value, "confidence": "llm", "source": "llm_interpretation", "explicit": False, "updated_at": now()}
            state["history"].append({"at": now(), "type": "state_change", "field": key,
                                     "from": existing.get("value") if existing else None, "to": value,
                                     "source": "llm_interpretation"})
            applied.append(key)
        for key, value in (interp.get("customer_updates") or {}).items():
            if empty(value): continue
            old = state["customer"].get(key)
            if old == value: continue
            state["customer"][key] = value
            state["history"].append({"at": now(), "type": "state_change", "field": "customer." + key,
                                     "from": old, "to": value, "source": "llm_interpretation"})
            applied.append("customer." + key)
        for section in ("operational", "commercial"):
            for key, value in (interp.get(section + "_updates") or {}).items():
                if empty(value): continue
                old = state[section].get(key)
                if old == value: continue
                state[section][key] = value
                state["history"].append({"at": now(), "type": "state_change", "field": f"{section}.{key}",
                                         "from": old, "to": value, "source": "llm_interpretation"})
                applied.append(f"{section}.{key}")
        comms = interp.get("commitments")
        if isinstance(comms, dict):
            for side in ("company", "customer"):
                items = comms.get(side)
                if isinstance(items, str): items = [items]
                if not isinstance(items, list): continue
                for item in items:
                    if not isinstance(item, str) or not item.strip(): continue
                    bucket = state["commitments"].setdefault(side, [])
                    if item.strip() in bucket: continue
                    bucket.append(item.strip())
                    state["history"].append({"at": now(), "type": "state_change", "field": f"commitments.{side}",
                                             "from": None, "to": item.strip(), "source": "llm_interpretation"})
                    applied.append(f"commitments.{side}")
        state.setdefault("interpretations", []).append({"at": now(), "interpretation": interp, "applied": applied})
        return applied

    def _new(self, text):
        oid = "opp_" + uuid.uuid4().hex[:8]
        state = {"id": oid, "created_at": now(), "updated_at": now(), "mode": "discovery",
          "customer": {}, "requirements": {}, "operational": {}, "commercial": {},
          "commitments": {"company": [], "customer": [], "pending": []}, "conversation": [],
          "history": [], "decision": None, "reflections": [],
          "style_lessons": [], "commercial_lessons": [], "capability_lessons": []}
        self.db[oid] = state; return state

    def _extract(self, text, state):
        """Regex FALLBACK: runs AFTER the LLM layer. Fills only keys the LLM
        left empty; never overwrites an existing value. Returns applied keys."""
        req = state["requirements"]; lower = text.lower()
        applied = []
        def put(key, value, confidence="medium"):
            if key in req: return  # LLM layer already decided this field
            req[key] = {"value": value, "confidence": confidence, "source": "regex_fallback", "explicit": True, "updated_at": now()}
            state["history"].append({"at": now(), "type": "state_change", "field": key, "from": None, "to": value, "source": "regex_fallback"})
            applied.append(key)
        company = re.search(r"(?:from|at|for) ([A-Z][\w &.-]{2,40})", text)
        if company and not state["customer"].get("company"):
            name = re.split(r"\.\s", company.group(1))[0].strip(" .,")  # trim greedy sentence capture
            state["customer"] = {"company": name, "source": "regex_fallback"}
        pax = re.search(r"(?:around|about|for|of)\s*(\d+)\s*(?:employees|people|pax|guests|participants)?", lower)
        if pax: put("pax", int(pax.group(1)), "high")
        date = re.search(r"(?:\bon\b|date[: ]+)\s*((?:\d{1,2}\s+)?[A-Za-z]+(?:\s+\d{4})?|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)", text, re.I)
        if date: put("event_date", date.group(1), "high")
        time = re.search(r"(?:at|around|^|,\s*)(\d{1,2}(?::\d{2})?\s*(?:am|pm))", lower)
        if time: put("event_time", time.group(1), "high")
        duration = re.search(r"(\d+(?:\.\d+)?)\s*[- ]?\s*(?:hour|hours|hr|hrs)", lower)
        if duration: put("event_duration_hours", float(duration.group(1)), "high")
        if "matcha" in lower: put("service", "matcha workshop", "high")
        if "workshop" in lower: put("event_type", "workshop", "high")
        if "office" in lower or "kl" in lower or "kuala lumpur" in lower: put("venue", "customer office (KL)", "medium")
        if "hands-on" in lower or "hands on" in lower: put("workshop_format", "hands-on", "high")
        return applied

    def _gaps(self, state):
        req = state["requirements"]
        material = ["event_type", "pax", "event_date", "venue", "service", "event_time", "event_duration_hours"]
        missing = [x for x in material if x not in req]
        sufficient = not [x for x in ["event_type", "pax", "event_date", "venue", "service", "event_duration_hours"] if x not in req]
        return {"missing": missing, "sufficient": sufficient, "next_question": ("What time would work best, and how long should the workshop run?" if "event_time" in missing or "event_duration_hours" in missing else None)}

    def _retrieve(self, state, tags=None):
        """Single retrieval path: the knowledge store via EvidenceRetriever."""
        return self.retriever.retrieve(state, ["historical_cases", "capabilities", "sales_principles"], tags=tags)

    def _project_evidence(self, evidence):
        """Replace full knowledge records with lean references: {id, name, pax,
        summary, insights}. Full records stay in data/knowledge/ for on-demand
        lookup; prompts never need the 26-turn timelines. Idempotent: already
        lean entries pass through untouched."""
        def one(r):
            if not isinstance(r, dict): return {"summary": str(r)[:120]}
            if "content" not in r and "extraction" not in r:  # already projected, or plain text record
                out = {k: r[k] for k in ("id", "name", "pax", "summary", "insights", "text", "tags") if r.get(k) is not None}
                if not out.get("name") and r.get("text"): out["name"] = str(r["text"])[:80]
                return out
            c = r.get("content") if isinstance(r.get("content"), dict) else {}
            name = r.get("name") or r.get("title")
            if not name and isinstance(c.get("customer"), dict): name = c["customer"].get("company")
            fs = c.get("final_solution")
            summary = (fs if isinstance(fs, str) else json.dumps(fs, ensure_ascii=False))[:220] if fs else None
            if not summary and r.get("text"): summary = str(r["text"])[:220]
            insights = c.get("reusable_insights") or []
            insights = [(i if isinstance(i, str) else (i.get("insight") or i.get("text") or json.dumps(i, ensure_ascii=False))[:140]) for i in insights][:3]
            out = {"id": r.get("id")}
            if name: out["name"] = name
            if r.get("pax") is not None: out["pax"] = r["pax"]
            if summary: out["summary"] = summary
            if insights: out["insights"] = insights
            return out
        return {domain: [one(r) for r in records] for domain, records in (evidence or {}).items() if isinstance(records, list)}

    def _project_for_llm(self, state):
        """Lean view of state for LLM prompts: business facts + recent context.
        Full history/interpretations/evidence stay on disk for audit but never
        enter a prompt. Requirements collapse to {key: value}."""
        p = {"id": state.get("id"), "mode": state.get("mode"),
             "customer": state.get("customer", {}),
             "requirements": {k: (v.get("value") if isinstance(v, dict) else v) for k, v in state.get("requirements", {}).items()},
             "operational": state.get("operational", {}), "commercial": state.get("commercial", {}),
             "commitments": state.get("commitments", {}),
             "conversation_recent": state.get("conversation", [])[-8:],
             "history_recent": state.get("history", [])[-12:]}
        if state.get("needs_review"): p["needs_review"] = state["needs_review"]
        if state.get("decision"): p["decision"] = state["decision"]
        if state.get("retrieved_evidence"): p["evidence"] = self._project_evidence(state["retrieved_evidence"])
        sol = state.get("solution")
        if isinstance(sol, dict) and sol.get("options"):
            options = sol.get("options", [])
            rec = next((o for o in options if o.get("recommended")), options[0])
            p["solution"] = {"summary": sol.get("summary"),
                             "recommended_option": {"id": rec.get("id"), "name": rec.get("name"), "price": rec.get("price"), "scope": rec.get("scope", [])},
                             "other_options": [{"id": o.get("id"), "name": o.get("name"), "price": o.get("price")} for o in options if o is not rec],
                             "risks": sol.get("risks", []), "assumptions": sol.get("assumptions", [])}
        return p

    def _solution(self, state):
        """Pricing is deterministic; references are no longer duplicated here —
        the single source of truth for case references is state.retrieved_evidence."""
        req = state["requirements"]; pax = req["pax"]["value"]
        unit = 12; base = pax * unit
        return {"summary": f"{pax}-person {req.get('service', {}).get('value', 'event')} at {req.get('venue', {}).get('value', 'venue')}.",
          "options": [
            {"id":"A","name":"Essential workshop","price":round(base*.75),"scope":["1 facilitator","demonstration and tasting","basic setup"]},
            {"id":"B","name":"Guided hands-on lab","price":base,"recommended":True,"scope":["2 facilitators","hands-on tasting kits","setup and teardown"]},
            {"id":"C","name":"Premium team experience","price":round(base*1.55),"scope":["2 facilitators","branded take-home set","extended tasting bar"]}],
          "risks":["Confirm venue loading access and hot water/power.","Final dietary requirements can be collected before proposal."],
          "assumptions":["Two-hour session","Venue can support a 2m x 2m footprint"]}

    def message(self, text, opportunity_id=None, from_email=None, new_opportunity=False):
        email = (from_email or "").strip().lower()
        routed_via = None
        # email routing: sender -> existing opportunity (explicit id always wins)
        if not opportunity_id and email and not new_opportunity:
            opportunity_id = self.routes.get(email)
            if opportunity_id: routed_via = "email_route"
        state = self.db.get(opportunity_id) if opportunity_id else None
        if state and state.get("closed"):
            # this opportunity already ended; a new mail from the same sender starts a fresh one (repeat business)
            state = None; routed_via = "new_opportunity_after_close"
        if not state:
            state = self._new(text)
            if not routed_via: routed_via = "new_opportunity"
        elif not routed_via: routed_via = "explicit_id"
        # register/refresh the route and record contact_email in the state
        if email:
            if self.routes.get(email) != state["id"]:
                self.routes[email] = state["id"]; save(ROUTES, self.routes)
            if state["customer"].get("contact_email") != email:
                state["history"].append({"at": now(), "type": "state_change", "field": "customer.contact_email",
                                         "from": state["customer"].get("contact_email"), "to": email, "source": "email_header"})
                state["customer"]["contact_email"] = email
        before = json.loads(json.dumps(state["requirements"]))
        history_start = len(state["history"])
        state["conversation"].append({"id": uuid.uuid4().hex[:8], "role":"customer", "text":text, "at":now()})
        pre_gaps = self._gaps(state)
        interp = self._llm_interpretation(text, state, pre_gaps)     # 1) LLM primary extraction
        # same-event guard: follow-up mail must still be about THIS event
        same_event = interp.get("same_event") if isinstance(interp, dict) else None
        if not isinstance(same_event, dict): same_event = {"confidence": "high", "reason": "not assessed"}
        if len(state["conversation"]) > 1 and same_event.get("confidence") == "low":
            # low confidence this is the same event: do NOT merge anything into state; escalate
            state["needs_review"] = {"at": now(), "reason": "same_event_uncertain",
                                     "detail": same_event.get("reason", ""),
                                     "message_id": state["conversation"][-1].get("id"), "interpretation": interp}
            state["history"].append({"at": now(), "type": "needs_review", "reason": same_event.get("reason", "")})
            reply = "Thank you for your message — I have received it and will confirm the details with our team before getting back to you shortly."
            state["conversation"].append({"role": "agent", "text": reply, "at": now(), "action": "human_review_same_event"})
            state["updated_at"] = now(); action = "human_review_same_event"
            self.persist(state)
            return {"opportunity": state, "action": action, "reply": reply, "gaps": pre_gaps,
                    "llm_filled": [], "regex_filled": [], "routed_via": routed_via, "same_event": same_event}
        if same_event.get("confidence") == "medium":
            state["history"].append({"at": now(), "type": "same_event_check", "confidence": "medium", "reason": same_event.get("reason", "")})
        llm_filled = self._merge_interpretation(state, interp)
        regex_filled = self.state_updater.apply(state, text)        # 2) regex fallback: only fills what the LLM left empty
        gaps = self._gaps(state); state["updated_at"] = now()
        plan = self._plan(state, gaps)                                # deterministic routing, no LLM call
        # principles tag pre-filter: drop principles whose tags don't overlap with state needs.
        # Only consulted when principles are in retrieval_needed; passes through otherwise.
        need_tags = self._principle_tags_for_state(state) if "sales_principles" in plan.get("retrieval_needed", []) else None
        evidence = self.retriever.retrieve(state, plan.get("retrieval_needed", []), tags=need_tags)
        state["agent_plan"] = plan
        state["retrieved_evidence"] = self._project_evidence(evidence)  # lean references only; full records live in data/knowledge/
        changes = [h for h in state["history"][history_start:] if h.get("type")=="state_change"]
        if changes:
            state["change_impact"] = self.change_impact_analysis(state, changes)
        next_q = (interp or {}).get("next_question") if isinstance(interp, dict) else None
        if gaps["sufficient"]:
            state["mode"] = "solution"; state["solution"] = self._solution(state); action = "request_human_review"
            reply = self._customer_reply(state, plan["customer_reply_strategy"])
        else:
            state["mode"] = "discovery"; action = "ask_customer"
            reply = self._customer_reply(state, plan["customer_reply_strategy"] + " " + (next_q or gaps["next_question"] or "Could you share a little more about the event format?"))
        state["conversation"].append({"role":"agent","text":reply,"at":now(),"action":action})
        state["history"].append({"at":now(),"type":"agent_action","action":action,"changed":before != state["requirements"]})
        self.persist(state); return {"opportunity":state,"action":action,"reply":reply,"gaps":gaps,"llm_filled":llm_filled,"regex_filled":regex_filled,"routed_via":routed_via,"same_event":same_event}

    def _customer_reply(self, state, strategy):
        """Response generation is downstream of state update + agent strategy.
        Feeds the PROJECTED state: business facts, recent dialogue, lean evidence
        references and the recommended option — never the full audit trail."""
        raw = self._llm("Write a concise, human corporate-event email. You may naturally draw on the historical cases and knowledge in the evidence (e.g. mention we hosted a similar session for a comparable team), but never expose internal IDs, retrieval mechanics, or internal reasoning. Prices must come from the solution options only; do not invent prices, commitments, or capabilities.", json.dumps({"state": self._project_for_llm(state), "strategy": strategy}, ensure_ascii=False))
        return raw or strategy

    def approve(self, oid, option_id="B", edits=None):
        state = self.db[oid]; solution = state.get("solution") or self._solution(state)
        option = next(x for x in solution["options"] if x["id"] == option_id); edits = edits or {}
        approved = {**option, **edits, "approved_at": now()}; state["decision"] = approved; state["mode"] = "collaboration"; state["updated_at"] = now()
        state["proposal"] = {"subject":"Proposal — corporate event collaboration","body":f"We’re pleased to propose the {option['name']} for your event. Scope: {', '.join(option['scope'])}. Estimated fee: S${approved['price']}. Next step: confirm venue access and final requirements."}
        self.persist(state); return {"opportunity":state,"proposal":state["proposal"]}

    def _summarize_reflection(self, state, human_output, note=""):
        """LLM distills a human edit into reusable principles bucketed as
        style / commercial / capability. Raises on failure — the caller is
        expected to surface 400 to the user so they can retry, per the
        'no silent fall-back to the old keyword path' decision."""
        if not human_output or not human_output.strip():
            raise ValueError("human_output is required for AI reflection summary")
        ai = (state.get("proposal") or {}).get("body", "")
        if not ai:
            tail = (state.get("conversation") or [])
            ai = next((m.get("text", "") for m in reversed(tail) if m.get("role") == "agent"), "")
        ctx = {"company": (state.get("customer") or {}).get("company"),
               "service": ((state.get("requirements") or {}).get("service") or {}).get("value"),
               "pax": ((state.get("requirements") or {}).get("pax") or {}).get("value")}
        prompt = f'''AI draft:
{ai}

Human edited version:
{human_output}

Human note: {note or "(none)"}

Opportunity context: {json.dumps(ctx, ensure_ascii=False)}

Distill each adjustment into one of three buckets:
- style: tone, emphasis, format of outbound communications (e.g., "Lead with confirmed details", "Avoid exclamation marks in formal replies")
- commercial: pricing baseline, scope inclusions, payment terms, what to include in quotes (e.g., "Default quote excludes taxes unless explicitly asked")
- capability: which kinds of events we accept or decline, and how we want to handle certain event types (e.g., "We do not host wedding receptions; refer to partner vendors")

Rules:
- Each principle must be GENERALISABLE — applicable beyond this specific client.
- If an adjustment is client-specific (e.g., "this client dislikes formal greetings"), do NOT produce a principle; skip that bucket for that adjustment.
- Cap 2-4 insights per bucket. If a bucket has no generalisable adjustment, return an empty list for it.
- Each principle is a single declarative sentence.

Return JSON: {{"style": [...], "commercial": [...], "capability": [...], "rationale": "one-sentence summary of what changed"}}'''
        raw = self._llm("You are a feedback summarizer for a corporate event sales team. Extract reusable principles; ignore client-specific observations.", prompt, True)
        if not raw:
            raise RuntimeError("AI reflection summary unavailable — please retry")
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            raise RuntimeError("AI reflection summary returned invalid JSON — please retry")
        for k in ("style", "commercial", "capability"):
            if not isinstance(result.get(k), list):
                result[k] = []
        result["rationale"] = result.get("rationale") if isinstance(result.get("rationale"), str) else ""
        return result

    def reflect(self, oid, human_output, note=""):
        """AI-driven reflection. LLM buckets the human edit into style/commercial/
        capability; each bucket's insights append to a per-state list AND become
        a pending principle record (tagged by bucket) in the knowledge store.
        On LLM failure, raise — caller returns 400 so the user retries."""
        state = self.db[oid]
        summary = self._summarize_reflection(state, human_output, note)
        now_ts = now()
        bucket_to_list = {"style": "style_lessons", "commercial": "commercial_lessons", "capability": "capability_lessons"}
        for k in bucket_to_list.values():
            state.setdefault(k, [])
        state.setdefault("reflections", [])
        applied_buckets, new_principles = [], []
        existing_texts = {p.get("text") for p in self.knowledge_store.list_records("principles")}
        for bucket, list_key in bucket_to_list.items():
            for text in (summary.get(bucket) or []):
                if not isinstance(text, str): continue
                t = text.strip()
                if not t: continue
                state[list_key].append({"text": t, "at": now_ts, "source": "ai_reflection"})
                if t not in existing_texts:
                    principle = {"text": t, "tags": [bucket], "source_opportunity": oid, "at": now_ts, "origin": "ai_reflection"}
                    self.knowledge_store.add("principles", principle)
                    new_principles.append(principle)
                    existing_texts.add(t)
                applied_buckets.append(bucket)
        reflection_record = {
            "at": now_ts, "human_output": human_output, "note": note,
            "buckets": {b: list(summary.get(b) or []) for b in bucket_to_list},
            "rationale": summary.get("rationale", ""),
            "applied": sorted(set(applied_buckets)),
        }
        state["reflections"].append(reflection_record)
        state["history"].append({"at": now_ts, "type": "ai_reflection_summary",
                                 "buckets": sorted(set(applied_buckets)),
                                 "insight_count": sum(len(summary.get(b) or []) for b in bucket_to_list)})
        self.persist(state)
        return {"reflection": reflection_record,
                "lessons_appended": len(applied_buckets),
                "principles_created": len(new_principles),
                "principles": new_principles}

    def _distill_insights(self, state, outcome, note):
        """LLM drafts 2-4 reusable insights from the negotiation history.
        Fallback: the human's close note, else empty — never blocks a close."""
        convo = "\n".join(f"{m.get('role','?')}: {m.get('text','')}" for m in state.get("conversation", [])[-20:])
        changes = [f"{h.get('field')}: {h.get('from')!r} -> {h.get('to')!r}" for h in state.get("history", []) if h.get("type") == "state_change"]
        prompt = f'''Opportunity outcome: {outcome}. Human close note: {note or "(none)"}
Final requirements: {json.dumps({k: v.get("value") for k, v in state.get("requirements", {}).items()}, ensure_ascii=False)}
Commitments: {json.dumps(state.get("commitments", {}), ensure_ascii=False)}
Requirement change history: {json.dumps(changes, ensure_ascii=False)}
Full dialogue:
{convo}

You are distilling this completed corporate-event opportunity into a reusable case
for the team knowledge base. Return JSON: {{"reusable_insights": [2-4 short
generalisable lessons for winning similar future deals, each one sentence]}}.
Focus on negotiation patterns, what this customer cared about, and operational
gotchas. Do not invent prices or commitments.'''
        raw = self._llm("You are a knowledge distillation layer for a corporate events team.", prompt, True)
        if raw:
            try:
                insights = json.loads(raw).get("reusable_insights")
                if isinstance(insights, list) and insights:
                    return [str(i)[:300] for i in insights if str(i).strip()][:4]
            except json.JSONDecodeError: pass
        return [note.strip()] if note.strip() else []

    def _distill_case(self, state, outcome, note, insights):
        """Deterministic state -> case mapping (10 fields) + AI-drafted insights.
        Enters the knowledge store as pending; a human approves it in the review UI."""
        req = state.get("requirements", {})
        # initial requirement: the value each field held when FIRST set (first change "to")
        initial = {}
        for k in req:
            firsts = [h for h in state.get("history", []) if h.get("type") == "state_change" and h.get("field") == k]
            initial[k] = firsts[0].get("to") if firsts else req[k].get("value")
        changes = [{"at": h.get("at"), "field": h.get("field"), "from": h.get("from"), "to": h.get("to"), "source": h.get("source")}
                   for h in state.get("history", []) if h.get("type") == "state_change" and "." not in str(h.get("field"))]
        timeline = [{"turn": i + 1, "speaker": m.get("role", "?"), "text": m.get("text", ""), "at": m.get("at")}
                    for i, m in enumerate(state.get("conversation", []))]
        decision = state.get("decision")
        if decision:
            final_solution = {"selected_option": decision.get("name"), "price": decision.get("price"),
                              "scope": decision.get("scope", []), "approved_at": decision.get("approved_at")}
        else:
            sol = state.get("solution") or {}
            rec = next((o for o in sol.get("options", []) if o.get("recommended")), None)
            final_solution = ({"selected_option": rec.get("name"), "price": rec.get("price"), "scope": rec.get("scope", []),
                               "note": "recommended, never approved"} if rec else None)
        record = {
            "name": f"{(state.get('customer') or {}).get('company') or 'Unknown'} — {req.get('event_type', {}).get('value') or 'event'}",
            "source_type": "opportunity_distillation", "source_opportunity": state["id"],
            "review_status": "pending", "tags": ["distilled", outcome],
            "pax": req.get("pax", {}).get("value"),
            "content": {
                "customer": state.get("customer", {}),
                "initial_requirement": initial,
                "current_requirement": {k: v.get("value") for k, v in req.items()},
                "requirement_changes": changes,
                "operational_state": state.get("operational", {}),
                "commercial_state": state.get("commercial", {}),
                "commitments": state.get("commitments", {}),
                "conversation_timeline": timeline,
                "final_solution": final_solution,
                "outcome": {"result": outcome, "note": note or None, "closed_at": now()},
                "reusable_insights": insights,
            },
        }
        return self.knowledge_store.add("historical_cases", record)

    def close(self, oid, outcome, note=""):
        """End an opportunity (won/lost) and distill it into a pending case.
        The case then waits for human approval in /knowledge/review."""
        if outcome not in ("won", "lost"): raise ValueError("outcome must be won or lost")
        state = self.db.get(oid)
        if not state: raise ValueError(f"unknown opportunity {oid}")
        if state.get("closed"): return {"opportunity": state, "case": None, "already_closed": True}
        insights = self._distill_insights(state, outcome, note)
        case = self._distill_case(state, outcome, note, insights)
        state["closed"] = {"outcome": outcome, "note": note, "at": now(), "distilled_case_id": case["id"]}
        state["mode"] = "closed"; state["updated_at"] = now()
        state["history"].append({"at": now(), "type": "closed", "outcome": outcome, "distilled_case_id": case["id"]})
        self.persist(state)
        return {"opportunity": state, "case": {"id": case["id"], "review_status": "pending",
                "reusable_insights": insights}}

class Handler(BaseHTTPRequestHandler):
    agent = CorporateEventAgent()
    def send_json(self, data, code=200):
        raw=json.dumps(data, ensure_ascii=False).encode(); self.send_response(code); self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def send_html(self, path):
        raw=Path(path).read_bytes(); self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.send_header("Content-Length",str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        parsed = urlparse(self.path); q = parse_qs(parsed.query); route = parsed.path
        if route == "/health": return self.send_json({"ok":True,"service":"corporate-event-agent"})
        if route == "/knowledge/review": return self.send_html(ROOT / "review" / "knowledge_browser.html")
        if route == "/knowledge/pending": return self.send_json(self.agent.knowledge_store.list_pending())
        if route == "/knowledge/collections":
            counts = {c: len(self.agent.knowledge_store.list_records(c)) for c in ("historical_cases","capabilities","principles")}
            counts["sources"] = len(self.agent.knowledge_store.list_sources())
            return self.send_json(counts)
        if route == "/knowledge/records":
            return self.send_json(self.agent.knowledge_store.list_records(q.get("collection",[""])[0]))
        if route == "/knowledge/sources": return self.send_json(self.agent.knowledge_store.list_sources())
        if route == "/knowledge/search":
            return self.send_json(self.agent.knowledge_store.search(["historical_cases","capabilities","principles"],q.get("q",[""])[0]))
        if route == "/opportunities": return self.send_json(list(self.agent.db.values()))
        if route == "/opportunities/review": return self.send_html(ROOT / "review" / "opportunities_browser.html")
        if route.startswith("/opportunities/"):
            return self.send_json(self.agent.db.get(route.split("/")[-1], {}))
        self.send_json({"error":"not found"},404)
    def do_POST(self):
        try:
            body=json.loads(self.rfile.read(int(self.headers.get("Content-Length",0))) or b"{}")
            parts=self.path.strip("/").split("/")
            if self.path=="/messages": return self.send_json(self.agent.message(body["text"],body.get("opportunity_id"),body.get("from_email"),body.get("new_opportunity")),201)
            if self.path=="/knowledge/import": return self.send_json(self.agent.knowledge_importer.import_text(body["raw"],body.get("source_type","email_thread"),body.get("metadata")),201)
            if self.path=="/knowledge/save": return self.send_json(self.agent.knowledge_store.save_record(body["collection"], body["record"]),201)
            if self.path=="/knowledge/delete": return self.send_json(self.agent.knowledge_store.delete_record(body["collection"], body["id"]))
            if len(parts)==3 and parts[0]=="knowledge" and parts[1]=="review": return self.send_json(self.agent.knowledge_store.review(parts[2],body["action"],body.get("reviewer","workspace-user"),body.get("patch"),body.get("reason")))
            if len(parts)==3 and parts[0]=="opportunities" and parts[2]=="approve": return self.send_json(self.agent.approve(parts[1],body.get("option_id","B"),body.get("edits")))
            if len(parts)==3 and parts[0]=="opportunities" and parts[2]=="close": return self.send_json(self.agent.close(parts[1],body["outcome"],body.get("note","")))
            if len(parts)==3 and parts[0]=="opportunities" and parts[2]=="reflect": return self.send_json(self.agent.reflect(parts[1],body["human_output"],body.get("note","")))
            self.send_json({"error":"not found"},404)
        except Exception as e: self.send_json({"error":str(e)},400)

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    s=sub.add_parser("serve"); s.add_argument("--port",type=int,default=8080)
    m=sub.add_parser("message"); m.add_argument("--text",required=True); m.add_argument("--opportunity-id"); m.add_argument("--from-email"); m.add_argument("--new-opportunity",action="store_true")
    c=sub.add_parser("close"); c.add_argument("--id",required=True); c.add_argument("--outcome",required=True,choices=["won","lost"]); c.add_argument("--note",default="")
    sub.add_parser("demo")
    a=p.parse_args(); agent=CorporateEventAgent()
    if a.cmd=="serve": print(f"Agent listening on http://localhost:{a.port}"); ThreadingHTTPServer(("127.0.0.1",a.port),Handler).serve_forever()
    elif a.cmd=="message": print(json.dumps(agent.message(a.text,a.opportunity_id,a.from_email,a.new_opportunity),ensure_ascii=False,indent=2))
    elif a.cmd=="close": print(json.dumps(agent.close(a.id,a.outcome,a.note),ensure_ascii=False,indent=2))
    else:
        r=agent.message("We are planning a wellness event for around 80 employees at our office in KL on 20 October and would like a matcha workshop."); print(json.dumps(r,ensure_ascii=False,indent=2)); print("\n--- simulated reply ---\n"); print(json.dumps(agent.message("10am would be ideal, and we are happy with a 2-hour hands-on session.",r["opportunity"]["id"]),ensure_ascii=False,indent=2))
if __name__=="__main__": main()
