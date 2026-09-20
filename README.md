# Corporate Event Collaboration Agent

A runnable, bounded agent for **corporate event lead handling**. It keeps an
**Event Opportunity** as the primary object, performs incremental state
updates and gap analysis, retrieves evidence across three collections, drafts
multi-option solutions, supports human approval, and stores human
feedback as approved principles. When `DEEPSEEK_API_KEY` is configured,
DeepSeek handles message interpretation and customer-facing generation
through its OpenAI-compatible API.

## Quick start

```bash
# 1. Clone and configure
git clone <repo-url>
cd <project>
cp .env.example .env
echo "DEEPSEEK_API_KEY=your-key-here" >> .env

# 2. Run the demo (uses a built-in wellness-workshop prompt)
python3 agent.py demo

# 3. Or start the HTTP service
python3 agent.py serve --port 8080
```

The service exposes a tiny HTTP API and three review UIs:

| URL | Purpose |
|---|---|
| `http://localhost:8080/opportunities/review`   | Browse every active opportunity, its state, evidence, and reflections |
| `http://localhost:8080/knowledge/review`       | Approve / reject pending knowledge records and tag principles |
| `http://localhost:8080/`                       | A small standalone HTML page (no real data behind it — use the URLs above) |

## On first run

`agent.py` auto-bootstraps an empty `data/` directory from `examples/`:

* `data/opportunities/` gets three anonymized opportunities.
* `data/knowledge/{historical_cases,capabilities,principles}/` gets the
  approved seed fixtures.
* The legacy flat file `examples/legacy_knowledge.json` is imported through
  `_seed_legacy_knowledge` so the original demo cases survive the move.

The bootstrap is **idempotent**: if `data/` already has content it is left
alone. Delete files under `data/` to re-trigger a copy.

## Mental model

Every business email or message becomes an **Opportunity** — a JSON document
with these top-level sections:

| Section | Holds | Updated how |
|---|---|---|
| `requirements`    | pax, service, event_date, workshop_format … | extracted by regex + LLM, merged into prior state |
| `operational`     | facilitator count, AV needs, dietary etc.    | extracted when present |
| `commercial`      | price, GST, payment terms                   | only ever updated from human approval |
| `commitments`     | confirmed vs pending promises to the client  | tracked through approval flow |
| `conversation`    | chronological customer / agent messages    | appended on every turn |
| `retrieved_evidence` | slice of historical cases + capabilities + principles used as context for the most recent LLM call | refreshed each turn |
| `decision`        | the chosen option (`A` / `B` / `C`)         | set on `POST /opportunities/:id/approve` |
| `reflections`     | history of (human_output, note, AI bucket summary) | appended on `POST /opportunities/:id/reflect` |
| `style_lessons` / `commercial_lessons` / `capability_lessons` | structured per-domain lessons pulled out of human edits | appended on every reflection |
| `history`         | event log (state changes, agent actions)    | appended on every turn |

### Two-call LLM flow

1. **Call 1 — interpretation.** The regex layer first fills obvious slots
   (`pax`, `service`, `event_date`), then DeepSeek interprets the message
   against the projected prior state and the gap list. It produces a
   *merged* `requirements` update and a `next_question` if anything is
   still missing.
2. **Call 2 — drafting.** With the updated state and the retrieved
   evidence (historical cases ranked by TF-IDF + keyword count, principles
   pre-filtered by tag), the agent drafts three solution options
   with names, prices, scope, and references; it returns one of them as
   the outbound customer reply.

### Reflection loop

Human edits feed back as:

1. Append a `reflections[]` entry containing the human output, the
   optional note, and the AI-summarised three-bucket lessons
   (`style`, `commercial`, `capability`).
2. Promote each lesson into the matching lesson list on the
   opportunity state.
3. Record the same lesson into the `principles` knowledge collection as a
   `pending` record, tagged so that the next retrieval can target it.

Tag inference:

| Bucket | Tag applied when… |
|---|---|
| `capability` | always |
| `commercial` | the opportunity has any `pax` value |
| `style`     | the opportunity has any `conversation` entry |

Records without tags remain eligible for any future query (empty-tag
fallback).

### Knowledge stores

Three collections under `data/knowledge/`:

| Collection           | Source                                  | Tag-filtered at retrieval? |
|---|---|---|
| `historical_cases`   | Past approved engagements, with pax, scope, commercial outcomes | No |
| `capabilities`       | Playbooks: venue checklist, throughput, dedicated workshop recipes | No |
| `principles`         | Approved AI- and human-derived rules   | **Yes** — see above |

Hybrid retrieval is a 50/50 blend of:

* **Keyword count** (`text.count(term)`) — exact-term fallback.
* **TF-IDF cosine** (sklearn `TfidfVectorizer` with `char_wb` 2–4 ngrams) —
  fuzzy match.

`principles` records are pre-filtered to those whose `tags` intersect the
opportunity's inferred tags before the two scores are blended.

## API

```bash
# Create or continue an opportunity from a customer message
curl -X POST http://localhost:8080/messages \
  -H 'content-type: application/json' \
  -d '{"text":"We are planning a wellness event for 80 employees at our KL office on 20 October and would like a matcha workshop."}'

# Choose a solution option
curl -X POST http://localhost:8080/opportunities/opp_xxxxxxxx/approve \
  -H 'content-type: application/json' -d '{"option_id":"B"}'

# Capture human edit + free-text note
curl -X POST http://localhost:8080/opportunities/opp_xxxxxxxx/reflect \
  -H 'content-type: application/json' -d '{"human_output":"...","note":"Adjust tone here only"}'

# Search the knowledge store (tags optional list)
curl -X POST http://localhost:8080/knowledge/search \
  -H 'content-type: application/json' -d '{"query":"matcha workshop pricing","tags":["style"]}'

# Import an old email thread as a candidate historical case
curl -X POST http://localhost:8080/knowledge/import \
  -H 'content-type: application/json' \
  -d '{"source_type":"email_thread","raw":"ACME Corp planned a 90 pax matcha workshop …"}'
```

The full list of routes is at the bottom of `agent.py` `Handler.do_POST`.

## Project layout

```
.
├── agent.py                  # main service + CLI
├── infrastructure/           # JsonKnowledgeStore, EvidenceRetriever, services
│   └── knowledge_store.py
├── ingestion/                # KnowledgeImporter (source extraction)
├── review/                   # 2 static browser UIs (opportunities_browser.html, knowledge_browser.html)
├── index.html / app.js / styles.css  # standalone landing page
├── examples/                 # committed, anonymized seed data (no PII)
│   ├── seed_opportunities/   # three representative opportunity records
│   ├── seed_knowledge/       # historical_cases + capabilities + principles
│   └── legacy_knowledge.json # flat legacy seed used by the inline importer
├── data/                     # runtime state (gitignored, auto-bootstrapped from examples/)
├── .env.example              # copy → .env, fill in DEEPSEEK_API_KEY
├── LICENSE                   # MIT
└── README.md
```

## Configuration

Environment variables (read by `agent.py` on startup, with `.env` as
fallback):

| Variable            | Default                  | Notes |
|---|---|---|
| `DEEPSEEK_API_KEY`  | *(required for LLM)*     | Without it the agent falls back to regex-only paths |
| `DEEPSEEK_MODEL`    | `deepseek-chat`          | Any OpenAI-compatible model name |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | Override for self-hosted or proxies |

Never commit `DEEPSEEK_API_KEY`. The supported defaults are
`deepseek-chat` and `https://api.deepseek.com`.

## Design notes

* The reasoning layer is deliberately bounded and deterministic for V1. The
  regex extraction, retrieval, and solution functions are isolated so an
  LLM provider can replace only those components later without changing
  the workflow or state model.
* The `commercial` state section is **read-only** outside of human
  approval — the agent never rewrites pricing on its own.
* Approved principles are what the second LLM call actually sees; pending
  records show up in the review UI, not in prompts.

## License

MIT — see `LICENSE`.
