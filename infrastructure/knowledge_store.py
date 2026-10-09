import json, os, re, threading, uuid
from datetime import datetime, timezone
from pathlib import Path

# Hybrid retrieval uses scikit-learn + numpy for the semantic layer. They are
# declared in requirements.txt, but a missing install must not take the whole
# service down — we degrade to the keyword layer and say so once.
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - depends on the runtime environment
    _HAS_SKLEARN = False
    _warned_missing_sklearn = False

    def _warn_once():
        global _warned_missing_sklearn
        if not _warned_missing_sklearn:
            _warned_missing_sklearn = True
            print("[knowledge_store] scikit-learn not installed — retrieval is "
                  "running on the keyword layer only. "
                  "Install with: pip install -r requirements.txt")

COLLECTIONS = ("historical_cases", "capabilities", "principles", "feedback")
def now(): return datetime.now(timezone.utc).isoformat()

# Process-wide IO lock: ThreadingHTTPServer workers must not interleave
# read-modify-write cycles on knowledge files (lost update), and a crash
# must never leave a half-written record behind.
_IO_LOCK = threading.RLock()

def _atomic_write(path, text):
    """Write via tmp file + os.replace so readers see either the old or the
    new complete file — never a truncated JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _IO_LOCK:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

# Collections that should be pre-filtered by `tags` before scoring. Other
# collections (historical_cases, capabilities) are tag-agnostic — their
# records describe what they describe, not a strategic bucket.
TAG_FILTER_COLLECTIONS = ("principles",)

class KnowledgeStore:
    """Stable interface; JSON keyword search can later be replaced by hybrid/vector RAG."""
    def add(self, collection, record): raise NotImplementedError
    def get(self, collection, record_id): raise NotImplementedError
    def search(self, collections, query, limit=10, tags=None, topics=None): raise NotImplementedError
    def save_source(self, source): raise NotImplementedError

class JsonKnowledgeStore(KnowledgeStore):
    """Directory-per-collection, file-per-record storage.

    Layout:
        root/<collection>/<record_id>.json   one record per file
        root/sources/<source_id>.json       one raw source per file
    Legacy flat files (root/<collection>.json, root/sources.json) are migrated
    into the directory layout on first load, then moved to root/_legacy/.
    """
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        for c in COLLECTIONS: (self.root / c).mkdir(exist_ok=True)
        (self.root / "sources").mkdir(exist_ok=True)
        self._tfidf_cache = {}  # collection -> (vec, matrix, records, texts_lower) or None when empty
        self._migrate_legacy()
        self._remove_knowledge_review_metadata()

    @staticmethod
    def _without_review_metadata(record):
        # Knowledge records are directly editable and immediately searchable;
        # opportunity/commitment approval remains a separate business workflow.
        return {k: v for k, v in record.items() if k not in {
            "review_status", "approved_by", "approved_at", "rejected_by",
            "rejected_at", "rejection_reason",
        }}

    def _remove_knowledge_review_metadata(self):
        """Clean up records created by the old knowledge review UI."""
        for collection in COLLECTIONS:
            for record in self._records(collection):
                clean = self._without_review_metadata(record)
                if clean != record:
                    _atomic_write(self._record_path(collection, record["id"]),
                                  json.dumps(clean, indent=2, ensure_ascii=False))

    # ---------- migration ----------
    def _migrate_legacy(self):
        legacy_dir = self.root / "_legacy"; moved = False
        for c in COLLECTIONS:
            legacy = self.root / (c + ".json")
            if legacy.exists():
                for record in json.loads(legacy.read_text(encoding="utf-8")):
                    if "id" in record: _atomic_write(self._record_path(c, record["id"]), json.dumps(record, indent=2, ensure_ascii=False))
                legacy_dir.mkdir(exist_ok=True); legacy.rename(legacy_dir / legacy.name); moved = True
        legacy_sources = self.root / "sources.json"
        if legacy_sources.exists():
            for source in json.loads(legacy_sources.read_text(encoding="utf-8")):
                if "id" in source: _atomic_write(self._source_path(source["id"]), json.dumps(source, indent=2, ensure_ascii=False))
            legacy_dir.mkdir(exist_ok=True); legacy_sources.rename(legacy_dir / legacy_sources.name); moved = True
        if moved: print(f"[knowledge_store] migrated legacy flat files into per-record layout (originals kept in {legacy_dir})")

    # ---------- paths & io ----------
    def _record_path(self, collection, record_id): return self.root / collection / (record_id + ".json")
    def _source_path(self, source_id): return self.root / "sources" / (source_id + ".json")
    def _records(self, collection):
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((self.root / collection).glob("*.json"))]
    def _sources(self):
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((self.root / "sources").glob("*.json"))]

    # ---------- sources ----------
    def add_source(self, source_type, raw, metadata=None):
        source = {"id": "source_" + uuid.uuid4().hex[:10], "type": source_type, "raw": raw, "metadata": metadata or {}, "created_at": now()}
        _atomic_write(self._source_path(source["id"]), json.dumps(source, indent=2, ensure_ascii=False))
        return source

    def save_source(self, source):
        if not source.get("id"): raise ValueError("source id is required")
        _atomic_write(self._source_path(source["id"]), json.dumps(source, indent=2, ensure_ascii=False))
        return source

    # ---------- records ----------
    def add(self, collection, record):
        if collection not in COLLECTIONS: raise ValueError("unknown knowledge collection")
        record = self._without_review_metadata(record)
        record = {**record, "id": record.get("id") or collection + "_" + uuid.uuid4().hex[:8],
                  "collection": collection, "created_at": record.get("created_at", now())}
        _atomic_write(self._record_path(collection, record["id"]), json.dumps(record, indent=2, ensure_ascii=False))
        self._tfidf_cache.pop(collection, None)   # a new record can shift every score
        return record

    def get(self, collection, record_id):
        path = self._record_path(collection, record_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def _build_tfidf(self, collection):
        """Lazy-build / rebuild TF-IDF index for one collection.
        Returns (vec, matrix, records, texts_lower) or None when the collection is empty
        or when scikit-learn is unavailable (keyword-only mode)."""
        if not _HAS_SKLEARN:
            _warn_once()
            self._tfidf_cache[collection] = None
            return None
        records = self._records(collection)
        if not records:
            self._tfidf_cache[collection] = None
            return None
        texts_lower = [json.dumps(r, ensure_ascii=False).lower() for r in records]
        # char_wb analyzer: handles English word boundaries AND Chinese character n-grams,
        # so "team-building" and "team_building" share substrings; "报价有效期" and
        # "价格有效时间" share character 3-grams. min_df=1 keeps small corpora working.
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, lowercase=True)
        matrix = vec.fit_transform(texts_lower)
        self._tfidf_cache[collection] = (vec, matrix, records, texts_lower)
        return self._tfidf_cache[collection]

    def search(self, collections, query, limit=10, tags=None, topics=None):
        """Hybrid search:
          - Tag pre-filter: any collection in TAG_FILTER_COLLECTIONS drops records whose
            tags are non-empty AND have no overlap with `tags`. Records without a tags
            field always pass (back-compat with legacy seed records).
          - Optional precise-topic filter: matches a record's primary or related topic.
          - Token-frequency layer (exact match) — fast, deterministic, strong on IDs/numbers.
          - TF-IDF cosine layer (char_wb n-grams) — semantic tolerance for Chinese and
            English variants of the same concept.
          - Final score: 0.5 * kw + 0.5 * tfidf (tfidf scaled to kw-magnitude).
        When scikit-learn is missing the TF-IDF layer contributes 0 and the
        keyword layer carries the ranking on its own.
        """
        terms = set(re.findall(r"[a-z0-9]+", (query or "").lower()))
        q_lower = (query or "").lower()
        hits = []
        for c in collections:
            # 1) Tag pre-filter
            records = self._records(c)
            if c in TAG_FILTER_COLLECTIONS and tags:
                tagset = set(tags)
                records = [r for r in records
                           if not r.get("tags") or set(r["tags"]) & tagset]
            if topics:
                topicset = set(topics)
                records = [r for r in records
                           if not r.get("primary_topic")
                           or r.get("primary_topic") in topicset
                           or topicset.intersection(r.get("related_topics") or [])]
            if not records: continue
            # 2) TF-IDF layer (cached)
            cached = self._tfidf_cache.get(c)
            if cached is None and c not in self._tfidf_cache:
                cached = self._build_tfidf(c)
            if cached is None:
                # No TF-IDF available (empty collection, or scikit-learn missing).
                # Plain list rather than np.zeros so this path needs no numpy either.
                tfidf_scores = [0.0] * len(records)
                texts_lower = [json.dumps(r, ensure_ascii=False).lower() for r in records]
            else:
                vec, matrix, cached_records, cached_texts_lower = cached
                q_vec = vec.transform([q_lower])
                all_scores = cosine_similarity(q_vec, matrix).flatten()
                # `records` may be a tag-filtered subset of cached_records. Never
                # index cached scores by the subset position: that makes record B
                # inherit record A's score and attribution.
                cached_index = {r["id"]: i for i, r in enumerate(cached_records)}
                # A hand-edited/deleted record can make a stale cache miss an id.
                # Rebuild once instead of silently assigning a wrong score.
                if any(r["id"] not in cached_index for r in records):
                    cached = self._build_tfidf(c)
                    if cached is None:
                        tfidf_scores = [0.0] * len(records)
                    else:
                        vec, matrix, cached_records, _ = cached
                        q_vec = vec.transform([q_lower]); all_scores = cosine_similarity(q_vec, matrix).flatten()
                        cached_index = {r["id"]: i for i, r in enumerate(cached_records)}
                        tfidf_scores = [float(all_scores[cached_index[r["id"]]]) for r in records]
                else:
                    tfidf_scores = [float(all_scores[cached_index[r["id"]]]) for r in records]
                texts_lower = [json.dumps(r, ensure_ascii=False).lower() for r in records]
            # 3) Score per record
            for i, r in enumerate(records):
                kw_score = sum(texts_lower[i].count(t) for t in terms) if terms else 0
                tfidf_score = float(tfidf_scores[i])
                # tfidf is 0-1; kw can be much larger. Scale tfidf to ~kw-magnitude.
                score = 0.5 * kw_score + 0.5 * (tfidf_score * 10)
                if score > 0:
                    hits.append({"record_id": r["id"], "collection": c, "score": score,
                                 "kw_score": kw_score, "tfidf_score": tfidf_score,
                                 "source_ids": r.get("source_ids", []), "record": r})
        return sorted(hits, key=lambda x: x["score"], reverse=True)[:limit]

    # ---------- full-record management (for the knowledge browser UI) ----------
    def list_records(self, collection):
        if collection not in COLLECTIONS: raise ValueError("unknown knowledge collection")
        return self._records(collection)

    def list_sources(self):
        return self._sources()

    def save_record(self, collection, record):
        """Create or fully replace a record by its id. Holds the IO lock for
        the whole read-modify-write cycle so concurrent saves cannot interleave."""
        if collection not in COLLECTIONS: raise ValueError("unknown knowledge collection")
        with _IO_LOCK:
            rid = record.get("id")
            if not rid: rid = record["id"] = collection + "_" + uuid.uuid4().hex[:8]
            existing = self.get(collection, rid)
            if existing and not record.get("created_at"): record["created_at"] = existing.get("created_at")
            record.setdefault("collection", collection)
            record.setdefault("created_at", now())
            record = self._without_review_metadata(record)
            _atomic_write(self._record_path(collection, rid), json.dumps(record, indent=2, ensure_ascii=False))
            self._tfidf_cache.pop(collection, None)
            return record

    def delete_record(self, collection, record_id):
        if collection not in COLLECTIONS: raise ValueError("unknown knowledge collection")
        path = self._record_path(collection, record_id)
        if not path.exists(): raise KeyError(record_id)
        path.unlink(); self._tfidf_cache.pop(collection, None); return {"deleted": record_id}
