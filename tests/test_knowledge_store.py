import json
from infrastructure.knowledge_store import JsonKnowledgeStore

def record(rid, tags, text):
    return {"id": rid, "tags": tags, "text": text, "review_status": "approved", "source_ids": ["src_"+rid]}

def test_tag_filtered_tfidf_uses_the_record_own_score(tmp_path):
    store = JsonKnowledgeStore(tmp_path / "knowledge")
    store.add("principles", record("principle_a", ["style"], "gift basket presentation"))
    store.add("principles", record("principle_b", ["capability"], "venue staffing throughput"))
    unfiltered = {x["record_id"]: x["tfidf_score"] for x in store.search(["principles"], "gift basket")}
    filtered = store.search(["principles"], "gift basket", tags=["capability"])
    assert filtered == []
    filtered = store.search(["principles"], "venue staffing", tags=["capability"])
    assert filtered and filtered[0]["record_id"] == "principle_b"
    assert filtered[0]["tfidf_score"] > 0

def test_pending_and_rejected_are_not_retrievable(tmp_path):
    store = JsonKnowledgeStore(tmp_path / "knowledge")
    pending = store.add("principles", {"id":"pending", "text":"hidden", "review_status":"pending"})
    rejected = store.add("principles", {"id":"rejected", "text":"hidden", "review_status":"rejected"})
    assert store.search(["principles"], "hidden") == []

def test_add_after_cache_build_does_not_crash_or_misalign(tmp_path):
    """Regression: add() must invalidate the TF-IDF cache. Before the fix,
    a record added after the cache was built raised KeyError in search()
    (id-mapping lookup); the defensive .get() also covers hand-edited files."""
    store = JsonKnowledgeStore(tmp_path / "knowledge")
    store.add("principles", record("p_a", ["style"], "gift basket"))
    store.search(["principles"], "gift basket")            # builds the cache
    store.add("principles", record("p_b", ["style"], "venue staffing throughput"))
    hits = store.search(["principles"], "venue staffing")  # must not raise
    assert hits and hits[0]["record_id"] == "p_b"
    assert hits[0]["tfidf_score"] > 0                      # rebuilt cache scored it
    # and the old record still scores on its own text
    hits2 = store.search(["principles"], "gift basket")
    assert {h["record_id"] for h in hits2} == {"p_a"}
