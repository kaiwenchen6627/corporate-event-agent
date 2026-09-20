# Seed data

Static, anonymized fixtures used to bootstrap an empty `data/` directory the
first time the agent is started. All client names have been replaced with the
placeholders **ACME Corp**, **Globex Co**, and **Initech Pte Ltd**, and any
real-looking email addresses are rewritten to `client@example.invalid`.

```
examples/
├── legacy_knowledge.json                  # Old flat-style legacy seed (cases + principles)
├── seed_opportunities/                    # Three representative opportunity records
│   ├── seed_opp_alpha.json                #   plain discovery stage, all required fields
│   ├── seed_opp_bravo.json                #   in solution stage with full retrieved_evidence
│   └── seed_opp_charlie.json              #   carries an AI reflection + bucket summary
└── seed_knowledge/
    ├── historical_cases/                  # Approved past workshop and drinks gigs
    ├── capabilities/                      # Playbooks: venue checklist, throughput, etc.
    └── principles/                        # Approved style / commercial / capability rules
```

## When this is loaded

`agent.py` looks at this folder **once** at startup, only if:

* `data/opportunities/` is empty (no live records), **and**
* `data/knowledge/{historical_cases,capabilities,principles}/` is empty.

If both conditions hold, the boot path copies each file into the matching
collection under `data/` and the seeds appear in the review UI. The seed copy
becomes editable state from then on; this folder is **not re-read** unless the
user deletes the destination.

## What is intentionally *not* here

* Real customer names, emails, phone numbers
* Live conversation transcripts with real participants
* Unapproved review candidates (only `review_status: "approved"` records)
