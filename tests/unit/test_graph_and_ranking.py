"""Unit tests for the code-graph indexer/expander, rank fusion, and BM25 retriever."""
import json

import pytest

from components.indexer.repo_graph_indexer import RepoGraphIndexer, RepoGraphIndexerSettings
from components.ranking.rank_fusion import RankFusion, RankFusionSettings
from components.retrieval.coarse_retriever import CoarseRetriever, CoarseRetrieverSettings
from components.retrieval.graph_expander import GraphExpander, GraphExpanderSettings
from components.shared_types import RetrievedChunk


def _chunk(text, **meta):
    return {"text": text, "metadata": meta}


# --------------------------------------------------------------------------- #
# RepoGraphIndexer
# --------------------------------------------------------------------------- #
def _build_graph(tmp_path):
    chunks = [
        _chunk(
            "import os\nfrom collections import deque\nclass Foo:\n    def bar(self): pass",
            relative_path="a.py",
            source_id="repo",
            chunk_id="chunk:c1",
            symbol="Foo.bar",
            chunk_type="method",
            start_line=1,
            end_line=4,
        ),
        _chunk("def baz(): pass", relative_path="a.py", source_id="repo", chunk_id="chunk:c2", symbol="baz", chunk_type="function"),
        _chunk("key1: value\nkey2: value", relative_path="config.yaml", source_id="repo", chunk_id="chunk:c3"),
        _chunk("def test_bar():\n    Foo().bar()", relative_path="test_a.py", source_id="repo", chunk_id="chunk:c4"),
        _chunk("   ", relative_path="empty.py"),  # skipped (blank)
    ]
    settings = RepoGraphIndexerSettings(path=str(tmp_path / "graph.json"))
    indexer = RepoGraphIndexer(settings)
    records = indexer.index(chunks)
    return records, json.loads((tmp_path / "graph.json").read_text())


def test_repo_graph_indexer_builds_nodes_edges_chunks(tmp_path):
    records, payload = _build_graph(tmp_path)
    assert len(records) == 4  # blank chunk skipped
    node_types = {n["type"] for n in payload["nodes"]}
    assert {"Repository", "File", "Method", "Module", "ConfigKey"} <= node_types
    relations = {e["relation"] for e in payload["edges"]}
    assert {"CONTAINS", "DEFINES", "IMPORTS", "DEFINES_CONFIG"} <= relations
    # test-name heuristic links test_a.py to the bar symbol
    assert any(e["relation"] == "TESTS" for e in payload["edges"])


def test_repo_graph_indexer_chunk_helpers():
    from types import SimpleNamespace

    assert RepoGraphIndexer._chunk_text(SimpleNamespace(text="x")) == "x"
    assert RepoGraphIndexer._chunk_text({"text": "y"}) == "y"
    assert RepoGraphIndexer._chunk_text("z") == "z"
    assert RepoGraphIndexer._chunk_text(123) == ""
    assert RepoGraphIndexer._chunk_metadata({"metadata": {"a": 1}}) == {"a": 1}
    assert RepoGraphIndexer._chunk_metadata(123) == {}
    assert RepoGraphIndexer._symbol_node_type("class") == "Class"
    assert RepoGraphIndexer._symbol_node_type("other") == "Symbol"
    assert RepoGraphIndexer._extract_imports("import os\nfrom a.b import c") == ["a", "os"]
    assert RepoGraphIndexer._extract_config_keys("k: v", "x.py") == []
    assert "key1" in RepoGraphIndexer._extract_config_keys("key1: v\n[sec]\nk2 = 3", "c.toml")


# --------------------------------------------------------------------------- #
# GraphExpander
# --------------------------------------------------------------------------- #
def test_graph_expander_expands_neighbors(tmp_path):
    _build_graph(tmp_path)
    expander = GraphExpander(GraphExpanderSettings(path=str(tmp_path / "graph.json")))
    seed = RetrievedChunk(id="chunk:c1", text="seed", metadata={"chunk_id": "chunk:c1", "path": "a.py"})
    expanded = expander.expand([seed], top_k=10)
    ids = {c.id for c in expanded}
    assert "chunk:c1" not in ids  # original excluded
    assert ids  # reached other chunks in the file
    assert all(c.metadata["retrieval_source"] == "graph_expander" for c in expanded)


def test_graph_expander_empty_cases(tmp_path):
    missing = GraphExpander(GraphExpanderSettings(path=str(tmp_path / "none.json")))
    assert missing.expand([RetrievedChunk(id="x", text="t")]) == []
    _build_graph(tmp_path)
    exp = GraphExpander(GraphExpanderSettings(path=str(tmp_path / "graph.json")))
    assert exp.expand([]) == []
    # seed that matches nothing -> no seeds -> empty
    assert exp.expand([RetrievedChunk(id="zzz", text="t", metadata={"chunk_id": "nope"})]) == []


def test_graph_expander_handles_corrupt_graph(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json")
    exp = GraphExpander(GraphExpanderSettings(path=str(p)))
    assert exp.expand([RetrievedChunk(id="x", text="t")]) == []


# --------------------------------------------------------------------------- #
# RankFusion
# --------------------------------------------------------------------------- #
def test_rank_fusion_combines_and_normalizes():
    fusion = RankFusion(RankFusionSettings())
    set_a = [RetrievedChunk(id="x", text="x"), RetrievedChunk(id="y", text="y")]
    set_b = [RetrievedChunk(id="x", text="x"), RetrievedChunk(id="z", text="z")]
    fused = fusion.fuse([set_a, set_b])
    assert fused[0].id == "x"  # appears in both -> top
    assert 0.0 <= min(c.score for c in fused) and max(c.score for c in fused) == 1.0


def test_rank_fusion_empty_and_weights():
    fusion = RankFusion(RankFusionSettings())
    assert fusion.fuse([]) == []
    assert fusion._resolve_weights(3) == [1.0, 1.0, 1.0]
    weighted = RankFusion(RankFusionSettings(weights=[0.0, 1.0]))
    fused = weighted.fuse([[RetrievedChunk(id="a", text="a")], [RetrievedChunk(id="b", text="b")]])
    assert {c.id for c in fused} == {"b"}  # zero-weight set skipped
    with pytest.raises(ValueError):
        RankFusion(RankFusionSettings(weights=[1.0])).fuse([[], []])


def test_rank_fusion_dedup_key_and_equal_scores():
    fusion = RankFusion(RankFusionSettings())
    assert fusion._dedup_key(RetrievedChunk(id="i", text="t")) == "id::i"
    assert fusion._dedup_key(RetrievedChunk(id="", text="t")).startswith("text::")
    # single set, single item -> span 0 -> score forced to 1.0
    fused = fusion.fuse([[RetrievedChunk(id="only", text="t")]])
    assert fused[0].score == 1.0


# --------------------------------------------------------------------------- #
# CoarseRetriever (BM25)
# --------------------------------------------------------------------------- #
def _write_index(path, docs):
    path.write_text(json.dumps({"documents": docs}))


def test_coarse_retriever_retrieves_ranked(tmp_path):
    idx = tmp_path / "coarse.json"
    _write_index(
        idx,
        [
            {"id": "d1", "text": "the quick brown fox", "metadata": {"k": "v"}},
            {"id": "d2", "text": "lazy dog sleeps", "metadata": {}},
            {"id": "d3", "text": "  ", "metadata": {}},  # blank -> dropped
        ],
    )
    retriever = CoarseRetriever(CoarseRetrieverSettings(path=str(idx)))
    out = retriever.retrieve("quick fox", top_k=2)
    assert out and out[0].id == "d1"
    assert out[0].metadata["source"] == "bm25"
    # second call hits the signature cache
    assert retriever.retrieve("quick fox", top_k=2)[0].id == "d1"


def test_coarse_retriever_missing_and_empty(tmp_path):
    missing = CoarseRetriever(CoarseRetrieverSettings(path=str(tmp_path / "none.json")))
    assert missing.retrieve("q") == []
    empty = tmp_path / "empty.json"
    _write_index(empty, [{"id": "x", "text": "   "}])
    assert CoarseRetriever(CoarseRetrieverSettings(path=str(empty))).retrieve("q") == []
