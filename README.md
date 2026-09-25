# Corporate Event Collaboration Agent

A runnable, bounded agent for **corporate event lead handling**. It keeps an
**Event Opportunity** as the primary object, performs incremental state
updates and gap analysis, retrieves evidence across three collections, drafts
multi-option solutions, supports human approval, and stages outbound replies
for human send approval. Human feedback becomes pending principles that require
Knowledge Review before retrieval. When `DEEPSEEK_API_KEY` is configured,
DeepSeek handles message interpretation and customer-facing generation
through its OpenAI-compatible API.

## Quick start

Requires **Python 3.9+**. The only third-party dependencies are scikit-learn and
numpy (hybrid retrieval); everything else is stdlib.

```bash
# 1. Clone and install dependencies
git clone <repo-url>
cd <project>
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
echo "DEEPSEEK_API_KEY=your-key-here" >> .env

# 3. Run the demo (uses a built-in wellness-workshop prompt)
python3 agent.py demo

# 4. Or start the HTTP service
python3 agent.py serve --port 8080

# 5. Run the test suite
pytest -q
```

Without `DEEPSEEK_API_KEY` the service still starts — extraction falls back to
regex-only paths and drafting is skipped. The startup banner tells you which
mode you are in.

If `scikit-learn` is missing, the service also keeps running: retrieval drops to
the keyword layer only (no TF-IDF cosine), prints a one-line warning, and every
route stays up. Install `requirements.txt` for full hybrid ranking.

The service exposes a tiny HTTP API, two review UIs, and one standalone landing page:

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
| `commercial`      | `customer_claims` / `discussed_terms` / `approved_terms` | claims extracted from customer text; `approved_terms` only ever written by human approval (`POST /opportunities/:id/commercial/approve`, or option approval) |
| `commitments`     | confirmed vs pending promises to the client  | tracked through approval flow |
| `conversation`    | chronological customer / agent messages    | appended on every turn |
| `retrieved_evidence` | slice of historical cases + capabilities + principles used as context for the most recent LLM call | refreshed each turn |
| `decision`        | the chosen option (`A` / `B` / `C`)         | set on `POST /opportunities/:id/approve` |
| `pending_reply`   | AI-generated customer reply awaiting human send approval | set on every inbound message; promoted by `POST /opportunities/:id/reply` |
| `execution_traces`| compact per-message extraction, routing, fallback, and evidence trace | appended on every inbound message |
| `reflections`     | history of (human_output, note, AI bucket summary) | appended on `POST /opportunities/:id/reflect` |
| `style_lessons` / `commercial_lessons` / `capability_lessons` | structured per-domain lessons pulled out of human edits | appended on every reflection |
| `history`         | event log (state changes, agent actions)    | appended on every turn |

### Two-call LLM flow

1. **Call 1 — interpretation.** The regex layer first fills obvious slots
   (`pax`, `service`, `event_date`), then DeepSeek interprets the message
   against the projected prior state and the gap list. It produces a
   *merged* `requirements` update and a `next_question` if anything is
   still missing.
2. **Planning and solution.** The deterministic bounded planner uses the
   updated state to choose the next action and retrieval domains. When the
   requirements are sufficient, the workflow generates three solution options
   with names, prices, scope, and references.
3. **Call 2 — drafting.** The LLM drafts customer-facing language from the
   projected state, selected evidence, and strategy. The draft is staged in
   `pending_reply`; it is not added to conversation or treated as sent until
   a human calls the reply endpoint.

### Reflection loop

Human edits feed back as:

1. Append a `reflections[]` entry containing the human output, the
   optional note, and the AI-summarised three-bucket lessons
   (`style`, `commercial`, `capability`).
2. Promote each lesson into the matching lesson list on the
   opportunity state.
3. Record the same lesson into the `principles` knowledge collection as a
   `pending` record, tagged so that a reviewer can approve it before retrieval.

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

# Choose a solution option (optionally edit scope/price; the proposal
# follows the EDITED decision, and approved price lands in commercial.approved_terms)
curl -X POST http://localhost:8080/opportunities/opp_xxxxxxxx/approve \
  -H 'content-type: application/json' -d '{"option_id":"B"}'

# Promote a customer-claimed commercial term into an agreed term (or set one
# directly); reject drops the claim as not agreed
curl -X POST http://localhost:8080/opportunities/opp_xxxxxxxx/commercial/approve \
  -H 'content-type: application/json' -d '{"key":"payment_terms","value":"50% deposit"}'

# Capture human edit + free-text note
curl -X POST http://localhost:8080/opportunities/opp_xxxxxxxx/reflect \
  -H 'content-type: application/json' -d '{"human_output":"...","note":"Adjust tone here only"}'

# Human-review and send the staged customer reply; optional text edits the draft
curl -X POST http://localhost:8080/opportunities/opp_xxxxxxxx/reply \
  -H 'content-type: application/json' -d '{"text":"Could you share the expected headcount?"}'

# Search the knowledge store (GET, query string)
curl 'http://localhost:8080/knowledge/search?q=matcha+workshop+pricing'

# Import an old email thread as a candidate historical case
curl -X POST http://localhost:8080/knowledge/import \
  -H 'content-type: application/json' \
  -d '{"source_type":"email_thread","raw":"ACME Corp planned a 90 pax matcha workshop …"}'
```

With `AGENT_AUTH` set, add `-u user:pass` to every request. The full list of
routes is at the bottom of `agent.py` — `Handler._route_get` and `Handler.do_POST`.

## Project layout

```
.
├── agent.py                  # main service + CLI
├── healthcheck.py            # container healthcheck probe (stdlib only)
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
├── tests/                    # unit, retrieval regression, and golden-path tests
├── requirements.txt          # scikit-learn + numpy + pytest
├── Dockerfile                # python:3.12-slim, non-root, healthchecked
├── docker-compose.yml        # single-service deploy with a persistent data volume
├── .dockerignore
├── .env.example              # copy → .env, fill in DEEPSEEK_API_KEY
├── LICENSE                   # MIT
└── README.md
```

## Deployment

Three supported paths. Pick by how much isolation you need.

### A. Local process (fastest)

```bash
pip install -r requirements.txt
cp .env.example .env && echo "DEEPSEEK_API_KEY=..." >> .env
python3 agent.py serve --port 8080        # binds 127.0.0.1 only
```

Nothing is exposed to the network, so `AGENT_AUTH` is not needed.

### B. Docker Compose (self-hosting, recommended)

```bash
cp .env.example .env
# fill in DEEPSEEK_API_KEY, and set AGENT_AUTH="user:pass"
docker compose up -d --build
```

State lives in the named volume `agent-data`, so opportunities and approved
knowledge survive rebuilds. To wipe it and return to the shipped seed:
`docker compose down -v`.

By default compose publishes the port on **127.0.0.1 only**. To expose it on
your LAN, set `BIND_ADDR` in `.env`:

```bash
BIND_ADDR=0.0.0.0
HOST_PORT=8080
AGENT_AUTH=admin:change-me     # now mandatory
```

### C. Platform as a service (Railway / Render / Fly.io)

The container reads `HOST` and `PORT` from the environment, so those platforms
work without code changes. Two things to configure:

1. **Attach a persistent volume mounted at `/app/data`.** Without it the
   filesystem resets on every deploy and all opportunities are lost.
2. **Set `AGENT_AUTH`.** A public URL without auth exposes every customer
   conversation and the whole knowledge store to anyone who guesses the domain.

Health checks can point at `GET /health` (returns 401 with auth on, which is
expected — it proves the process is alive).

> **Security note.** The review UIs and JSON APIs have no built-in account
> system beyond the single shared `AGENT_AUTH` credential, and they are not
> designed for untrusted multi-user exposure. Treat them as an internal tool:
> keep them on localhost, a private network, or behind an authenticating
> reverse proxy.

## Configuration

Environment variables (read by `agent.py` on startup, with `.env` as
fallback):

| Variable             | Default                  | Notes |
|---|---|---|
| `LLM_GATEWAY_URL`    | *(empty)*                | Company OpenAI-compatible gateway, e.g. an AWS-backed Claude proxy |
| `LLM_GATEWAY_API_KEY`| *(empty)*                | When set, takes precedence over all other providers |
| `LLM_MODEL`          | *(empty)*                | Model name as issued by the gateway, e.g. `global.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| `ANTHROPIC_API_KEY`  | *(empty)*                | Claude direct via the official SDK (requires `pip install anthropic`) |
| `ANTHROPIC_MODEL`    | `claude-opus-5`          | Any Anthropic model name |
| `DEEPSEEK_API_KEY`   | *(empty)*                | DeepSeek direct |
| `DEEPSEEK_MODEL`     | `deepseek-chat`          | Any OpenAI-compatible model name |
| `DEEPSEEK_BASE_URL`  | `https://api.deepseek.com` | Override for self-hosted or proxies |
| `HOST`               | `127.0.0.1`              | Bind address. Default is local-only; use `0.0.0.0` in containers |
| `PORT`               | `8080`                   | Bind port; PaaS providers usually inject this |
| `AGENT_AUTH`         | *(empty = no auth)*      | `user:pass` enables HTTP Basic auth on every route |

Provider precedence: `LLM_GATEWAY_API_KEY` → `ANTHROPIC_API_KEY` → `DEEPSEEK_API_KEY`
→ regex-only. The gateway speaks the OpenAI `/v1/chat/completions` protocol
(Bearer auth), so the Anthropic SDK is not needed for it.

`--host` / `--port` flags override the corresponding environment variables.

Never commit any API key. `.env` is gitignored — keep real keys there.

## Design notes

* The reasoning layer is deliberately bounded and deterministic for V1. The
  regex extraction, retrieval, and solution functions are isolated so an
  LLM provider can replace only those components later without changing
  the workflow or state model.
* The `commercial.approved_terms` section is **read-only** outside of human
  approval — the agent never rewrites pricing on its own. Customer-asserted
  money facts land in `commercial.customer_claims` (claims, not agreements)
  and are promoted only through `POST /opportunities/:id/commercial/approve`.
* Approved principles are what the customer-drafting LLM call actually sees;
  pending records show up in the review UI, not in prompts.

## License

MIT — see `LICENSE`.
