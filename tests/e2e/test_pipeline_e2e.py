"""End-to-end pipeline run through the real orchestrator and registry components.

Uses an LLM-free pipeline (load -> chunk -> BM25 index -> retrieve -> build
context) so it exercises the full init + run flow without external services.
"""
from pathlib import Path

from pipeline.orchestrator import RAGOrchestrator
from pipeline.results import extract_answer, extract_contexts


def _config(tmp_path: Path) -> dict:
    coarse_path = str(tmp_path / "coarse_index.json")
    return {
        "runtime": {"mode": "api"},
        "intermediate": {"enabled": False},
        "cache": {"enabled": False},
        "chunking": {"recursive": {"chunk_size": 200, "chunk_overlap": 20}},
        "indexers": {"coarse": {"path": coarse_path}},
        "retrieval": {"top_k": 3},
        "init_pipeline": {
            "steps": [
                {"name": "load", "component": "text_loader"},
                {"name": "chunk", "component": "recursive_chunker"},
                {"name": "index", "component": "coarse_indexer"},
            ]
        },
        "pipeline": {
            "steps": [
                {"name": "retrieve", "component": "coarse_retriever"},
                {"name": "context", "component": "context_builder"},
            ]
        },
    }


def test_end_to_end_bm25_pipeline(tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text(
        "Indexing builds a searchable structure over documents. "
        "Retrieval finds relevant chunks for a query before generation."
    )

    config = _config(tmp_path)
    orchestrator = RAGOrchestrator(config)

    state = {"query": "what does retrieval do", "sources": [str(doc)]}
    state = orchestrator.initialize(state)
    assert state["indexed_count"] >= 1
    assert Path(config["indexers"]["coarse"]["path"]).exists()

    state = orchestrator.run(state)

    # context built from BM25-retrieved chunks
    assert state["retrieved"], "expected BM25 retrieval to return chunks"
    assert "retrieval" in state["context"].lower()
    contexts = extract_contexts(state)
    assert contexts and any("retrieval" in c.lower() for c in contexts)
    # no generator in this pipeline, so the answer is empty
    assert extract_answer(state) == ""


def test_end_to_end_reindex_is_idempotent(tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("Alpha beta gamma delta. Retrieval and indexing working together.")

    config = _config(tmp_path)
    state = RAGOrchestrator(config).initialize({"query": "alpha", "sources": [str(doc)]})
    first_count = state["indexed_count"]

    # a fresh orchestrator over the same config/sources re-indexes consistently
    state2 = RAGOrchestrator(config).initialize({"query": "alpha", "sources": [str(doc)]})
    assert state2["indexed_count"] == first_count
