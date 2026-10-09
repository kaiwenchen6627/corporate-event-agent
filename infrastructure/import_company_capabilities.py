"""Import the original products.md and services.md entries into capabilities.

The Markdown itself is preserved verbatim as a source. Searchable records are
cut only at the source document's existing numbered-entry headings; text is
never summarized or rewritten.

Usage:
  python3 -m infrastructure.import_company_capabilities /path/to/company
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from infrastructure.knowledge_store import JsonKnowledgeStore


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"


def source_for(store, document_id, filename, raw):
    for source in store.list_sources():
        if source.get("metadata", {}).get("document_id") == document_id:
            # Keep the source id stable across re-imports while refreshing its
            # exact raw contents to match the specified local source document.
            source["raw"] = raw
            source["metadata"].update({"filename": filename, "document_id": document_id})
            return store.save_source(source)
    return store.add_source("business_document_markdown", raw,
                            {"filename": filename, "document_id": document_id})


def exact_sections(raw: str, code: str):
    """Return source-authored entry blocks plus the source-authored pending section."""
    lines = raw.splitlines(keepends=True)
    starts = []
    for index, line in enumerate(lines):
        match = re.match(r"^###\s+((?:PRD|SRV)-[A-Z]\d+)\s*$", line.rstrip("\r\n"))
        if match and match.group(1).startswith(code + "-"):
            starts.append((index, match.group(1)))

    sections = []
    for n, (start, original_id) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        # Stop at the next second-level document section, not at nested
        # markdown, so every original record (including tables/YAML) stays whole.
        for i in range(start + 1, end):
            if re.match(r"^##\s+", lines[i]):
                end = i
                break
        text = "".join(lines[start:end]).rstrip()
        sections.append((original_id, text))

    for index, line in enumerate(lines):
        if re.match(r"^##\s+待确认\s*$", line.rstrip("\r\n")):
            end = next((i for i in range(index + 1, len(lines))
                        if re.match(r"^##\s+", lines[i])), len(lines))
            text = "".join(lines[index:end]).rstrip()
            sections.append((code + "-PENDING", text))
            break
    return sections


def remove_inline_yaml_metadata(text: str) -> str:
    """Remove only the source-authored YAML metadata block after an entry heading.

    The complete original document remains available in the source collection;
    this keeps searchable capability text focused on the human-readable entry.
    """
    lines = text.splitlines()
    if not lines or not re.match(r"^###\s+(?:PRD|SRV)-[A-Z]\d+\s*$", lines[0]):
        return text
    index = 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines) or lines[index].strip() != "```yaml":
        return text
    end = next((i for i in range(index + 1, len(lines)) if lines[i].strip() == "```"), None)
    if end is None:
        return text
    del lines[index:end + 1]
    while index < len(lines) and not lines[index].strip():
        del lines[index]
    return "\n".join(lines).rstrip()


def import_documents(company_dir: Path):
    store = JsonKnowledgeStore(KNOWLEDGE_DIR)
    total = 0
    for kind, code, filename, document_id in (
        ("product", "PRD", "products.md", "moss.company.products"),
        ("service", "SRV", "services.md", "moss.company.services"),
    ):
        path = company_dir / filename
        raw = path.read_text(encoding="utf-8")
        source = source_for(store, document_id, filename, raw)
        sections = exact_sections(raw, code)
        for original_id, text in sections:
            record_id = f"company_{kind}_{original_id.lower().replace('-', '_')}"
            existing = store.get("capabilities", record_id)
            if existing and source["id"] not in existing.get("source_ids", []):
                raise RuntimeError(f"Refusing to overwrite unrelated capability record {record_id}")
            record = {
                "id": record_id,
                "title": f"{filename} · {original_id}",
                "text": remove_inline_yaml_metadata(text),
                "capability_type": kind,
                "source_document_id": document_id,
                "source_record_id": original_id,
                "source_ids": [source["id"]],
                "tags": [kind, "original_markdown"],
                "origin": "company_markdown_import",
            }
            if existing:
                record["created_at"] = existing.get("created_at")
                store.save_record("capabilities", record)
            else:
                store.add("capabilities", record)
            total += 1
        print(f"{filename}: preserved raw source {source['id']}; imported {len(sections)} verbatim records")
    print(f"Total capability records imported/updated: {total}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 -m infrastructure.import_company_capabilities /path/to/company")
    import_documents(Path(sys.argv[1]).expanduser().resolve())
