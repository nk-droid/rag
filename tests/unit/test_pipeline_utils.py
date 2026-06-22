"""Unit tests for the pure helpers in pipeline.registry_utils / validator /
workspace / results — no external services touched."""
import types

import pytest

from components.shared_types import RetrievedChunk
from pipeline import registry_utils as ru
from pipeline.results import extract_answer, extract_contexts
from pipeline.validator import validate_config
from pipeline.workspace import apply_workspace, workspace_id


# --------------------------------------------------------------------------- #
# registry_utils — cache config helpers
# --------------------------------------------------------------------------- #
def test_config_cache_key_is_stable_and_includes_relevant_slices():
    cfg = {"vector_store": {"provider": "faiss"}, "models": {"a": 1}, "extra": "ignored"}
    key1 = ru._config_cache_key(cfg)
    key2 = ru._config_cache_key(dict(cfg))
    assert key1 == key2
    assert "faiss" in key1
    assert "ignored" not in key1


def test_cache_enabled_respects_global_and_feature_flags():
    assert ru._cache_enabled({"cache": {"enabled": False}}, "retrieval") is False
    # enabled globally, no feature map -> default True
    assert ru._cache_enabled({"cache": {"enabled": True}}, "retrieval") is True
    # feature explicitly disabled
    cfg = {"cache": {"enabled": True, "features": {"retrieval": False}}}
    assert ru._cache_enabled(cfg, "retrieval") is False
    assert ru._cache_enabled(cfg, "other") is True


def test_cache_ttl_per_feature_default_and_none():
    cfg = {"cache": {"ttl_sec": {"retrieval": 30, "prompt": None}, "default_ttl_sec": 900}}
    assert ru._cache_ttl(cfg, "retrieval") == 30
    assert ru._cache_ttl(cfg, "prompt") is None
    assert ru._cache_ttl(cfg, "missing") == 900
    assert ru._cache_ttl({"cache": {}}, "x", fallback=None) is None


def test_cache_env_and_key():
    assert ru._cache_env({"app": {"env": "prod"}}) == "prod"
    assert ru._cache_env({}) == "default"
    assert ru._cache_env({"app": "notadict"}) == "default"
    key = ru._cache_key({"cache": {"namespace": "ns", "version": "v2"}}, "feat", {"q": "hi"})
    assert isinstance(key, str) and key


def test_mark_cache_hit_initializes_and_updates():
    payload: dict = {}
    ru._mark_cache_hit(payload, "retrieval", True)
    assert payload["cache_hit"] == {"retrieval": True}
    payload["cache_hit"] = "corrupt"
    ru._mark_cache_hit(payload, "ranking", False)
    assert payload["cache_hit"] == {"ranking": False}


def test_get_cache_disabled_returns_none():
    assert ru._get_cache({"cache": {"enabled": False}}) is None


def test_get_cache_in_memory_is_cached_per_signature():
    ru._CACHE_CLIENTS.clear()
    cfg = {"cache": {"enabled": True, "type": "in_memory", "max_entries": 5}}
    first = ru._get_cache(cfg)
    second = ru._get_cache(cfg)
    assert first is second
    from infra.cache.in_memory_cache import InMemoryCache

    assert isinstance(first, InMemoryCache)


def test_get_cache_redis_builds_client(monkeypatch):
    ru._CACHE_CLIENTS.clear()
    built = {}

    class _FakeRedis:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr(ru, "RedisCache", _FakeRedis)
    client = ru._get_cache({"cache": {"enabled": True, "type": "redis"}})
    assert isinstance(client, _FakeRedis)
    assert "redis_url" in built


def test_get_cache_unsupported_type_raises():
    ru._CACHE_CLIENTS.clear()
    with pytest.raises(ValueError):
        ru._get_cache({"cache": {"enabled": True, "type": "memcached"}})


# --------------------------------------------------------------------------- #
# registry_utils — chunk (de)serialization
# --------------------------------------------------------------------------- #
def test_serialize_chunks_handles_all_input_shapes():
    obj = types.SimpleNamespace(id="o1", text="objtext", score=0.5, metadata={"k": "v"})
    chunks = [
        RetrievedChunk(id="c1", text="t1", score=1.0, metadata={"a": 1}),
        {"id": "c2", "content": "t2", "score": None, "metadata": "bad"},
        obj,
    ]
    out = ru._serialize_chunks(chunks)
    assert out[0]["id"] == "c1"
    assert out[1]["text"] == "t2" and out[1]["score"] == 0.0 and out[1]["metadata"] == {}
    assert out[2]["id"] == "o1" and out[2]["metadata"] == {"k": "v"}


def test_deserialize_chunks_filters_and_coerces():
    assert ru._deserialize_chunks("notalist") == []
    payload = [
        {"id": "c1", "text": "t1", "score": "1.5", "metadata": {"a": 1}},
        "skip-me",
        {"text": "t2", "score": "bad"},
    ]
    out = ru._deserialize_chunks(payload)
    assert len(out) == 2
    assert out[0].score == 1.5
    assert out[1].score == 0.0 and out[1].id == "chunk-2"


# --------------------------------------------------------------------------- #
# registry_utils — answer / query helpers
# --------------------------------------------------------------------------- #
def test_answer_text_variants():
    assert ru._answer_text(None) == ""
    assert ru._answer_text("hello") == "hello"
    assert ru._answer_text(types.SimpleNamespace(content="c")) == "c"
    assert ru._answer_text(types.SimpleNamespace(content=None)) != ""  # falls back to str(obj)
    listed = types.SimpleNamespace(content=[{"text": "a"}, "b", {"nope": 1}])
    assert ru._answer_text(listed) == "a\nb"


def test_retrieval_queries_dedupes_case_insensitively():
    payload = {"query": "Hello", "queries": ["hello", "World", "world", ""]}
    assert ru._retrieval_queries(payload) == ["Hello", "World"]
    assert ru._retrieval_queries({"queries": "notalist"}) == []


def test_as_retrieved_chunk_branches():
    rc = RetrievedChunk(id="x", text="t", score=1.0)
    assert ru._as_retrieved_chunk(rc, 0) is rc
    assert ru._as_retrieved_chunk({"text": "  "}, 0) is None
    made = ru._as_retrieved_chunk({"content": "hi", "score": "bad"}, 3)
    assert made.id == "chunk-3" and made.score == 0.0
    obj = types.SimpleNamespace(id="o", text="ot", score="bad", metadata="nope")
    out = ru._as_retrieved_chunk(obj, 0)
    assert out.text == "ot" and out.score == 0.0 and out.metadata == {}
    assert ru._as_retrieved_chunk(types.SimpleNamespace(text=""), 0) is None


def test_merge_retrieval_chunks_dedupes_keeps_best_and_respects_top_k():
    chunks = [
        {"id": "a", "text": "ta", "score": 0.2, "metadata": {"x": 1}},
        {"id": "a", "text": "ta", "score": 0.9, "metadata": {"y": 2}},
        {"id": "b", "text": "tb", "score": 0.5},
    ]
    merged = ru._merge_retrieval_chunks(chunks, top_k=10)
    assert [c.id for c in merged] == ["a", "b"]
    assert merged[0].score == 0.9 and merged[0].metadata == {"x": 1, "y": 2}
    assert ru._merge_retrieval_chunks(chunks, top_k=0) == []
    assert len(ru._merge_retrieval_chunks(chunks, top_k=1)) == 1


# --------------------------------------------------------------------------- #
# registry_utils — index paths / fingerprints / misc
# --------------------------------------------------------------------------- #
def test_get_index_path_priority_order():
    assert ru._get_index_path({"indexers": {"embedding": {"path": "p1"}}}, "embedding_indexer") == "p1"
    assert ru._get_index_path({"vector_store": {"embedding_indexer": {"path": "p2"}}}, "embedding_indexer") == "p2"
    assert ru._get_index_path({"vector_store": {"path": "p3"}}, "embedding_indexer") == "p3"
    assert ru._get_index_path({}, "embedding_indexer") == "data/indices/faiss_index"
    assert ru._get_index_path({"coarse_index": {"path": "cp"}}, "coarse_indexer") == "cp"
    assert ru._get_index_path({}, "coarse_indexer") == "data/indices/coarse_index.json"
    assert ru._get_index_path({}, "repo_graph_indexer") == "data/indices/repo_graph.json"
    assert ru._get_index_path({}, "weird_indexer") == "data/indices/weird.json"


def test_pipeline_component_names_flattens():
    cfg = {
        "pipeline": {
            "steps": [
                {"component": "a"},
                {"component": ["b", "c"]},
                "not-a-dict",
                {"name": "no-component"},
            ]
        }
    }
    assert ru._pipeline_component_names(cfg, "pipeline") == ["a", "b", "c"]
    assert ru._pipeline_component_names({"pipeline": "bad"}, "pipeline") == []


def test_index_fingerprint_is_deterministic(tmp_path):
    cfg = {
        "indexers": {
            "embedding": {"path": str(tmp_path / "emb")},
            "coarse": {"path": str(tmp_path / "coarse.json")},
        },
        "models": {"embedding": {"model_name": "m"}},
    }
    assert ru._index_fingerprint(cfg) == ru._index_fingerprint(dict(cfg))


def test_index_fingerprint_handles_directory_index(tmp_path):
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    (emb_dir / "index.faiss").write_text("x")
    (emb_dir / "index.pkl").write_text("y")
    cfg = {"indexers": {"embedding": {"path": str(emb_dir)}, "coarse": {"path": str(tmp_path / "c.json")}}}
    assert isinstance(ru._index_fingerprint(cfg), str)


def test_generation_cacheable():
    assert ru._generation_cacheable({"models": {"llm": {"temperature": 0}}}) is True
    assert ru._generation_cacheable({"models": {"llm": {"temperature": 0.7}}}) is False
    assert ru._generation_cacheable(
        {"models": {"llm": {"temperature": 0.7}}, "cache": {"allow_nondeterministic_generation": True}}
    ) is True
    assert ru._generation_cacheable({"models": {"llm": {"temperature": "bad"}}}) is True


def test_document_to_payload_variants():
    doc = types.SimpleNamespace(text="t", source="s.py", metadata={"lang": "py"})
    out = ru._document_to_payload(doc)
    assert out["text"] == "t" and out["metadata"]["source"] == "s.py"
    assert ru._document_to_payload({"body": "b", "source": "x"})["text"] == "b"
    assert ru._document_to_payload("raw") == {"text": "raw", "metadata": {}}
    assert ru._document_to_payload(123) == {"text": "", "metadata": {}}


def test_extract_chunk_inputs_priority(tmp_path):
    # documents/data_sources win
    out = ru._extract_chunk_inputs({"documents": [{"text": "doc1"}]})
    assert out == [("doc1", {})]
    # sources string that is not a path
    out = ru._extract_chunk_inputs({"sources": "just text"})
    assert out == [("just text", {})]
    # sources string that IS a path -> skipped, falls through to text
    real = tmp_path / "f.txt"
    real.write_text("hi")
    out = ru._extract_chunk_inputs({"sources": str(real), "text": "fallback"})
    assert out == [("fallback", {})]
    # query fallback
    out = ru._extract_chunk_inputs({"query": "q"})
    assert out == [("q", {"source": "query"})]
    assert ru._extract_chunk_inputs({}) == []


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
def test_extract_answer_precedence():
    assert extract_answer({"parsed_output": types.SimpleNamespace(answer="A")}) == "A"
    assert extract_answer({"parsed_output": {"answer": "B"}}) == "B"
    assert extract_answer({"answer": "C"}) == "C"
    assert extract_answer({"result": 42}) == "42"
    assert extract_answer({}) == ""


def test_extract_contexts_handles_shapes():
    state = {
        "retrieved": [
            RetrievedChunk(id="a", text="ta"),
            {"text": "tb"},
            types.SimpleNamespace(text="tc"),
        ]
    }
    assert extract_contexts(state) == ["ta", "tb", "tc"]
    assert extract_contexts({}) == []


# --------------------------------------------------------------------------- #
# workspace
# --------------------------------------------------------------------------- #
def test_workspace_id_changes_with_shape():
    a = {"pipeline": {"steps": [{"component": "x"}]}}
    b = {"pipeline": {"steps": [{"component": "y"}]}}
    assert workspace_id(a) != workspace_id(b)
    assert len(workspace_id(a)) == 16


def test_apply_workspace_rewrites_managed_paths():
    cfg = {"pipeline": {"steps": [{"component": "x"}]}, "indexers": {"embedding": {"path": "old/custom_name"}}}
    out = apply_workspace(cfg)
    wid = out["workspace"]["id"]
    assert out["indexers"]["embedding"]["path"].endswith(f"{wid}/custom_name")
    assert out["indexers"]["coarse"]["path"].endswith("coarse_index.json")
    assert out["workspace"]["path"].endswith(wid)
    # original config not mutated
    assert cfg["indexers"]["embedding"]["path"] == "old/custom_name"


def test_apply_workspace_disabled_is_noop():
    cfg = {"workspace": {"enabled": False}, "indexers": {"embedding": {"path": "keep"}}}
    out = apply_workspace(cfg)
    assert out["indexers"]["embedding"]["path"] == "keep"


def test_apply_workspace_custom_root():
    out = apply_workspace({"workspace": {"root": "/tmp/ws"}, "pipeline": {"steps": []}})
    assert out["workspace"]["root"] == "/tmp/ws"
    assert out["indexers"]["embedding"]["path"].startswith("/tmp/ws/")


# --------------------------------------------------------------------------- #
# validator
# --------------------------------------------------------------------------- #
def test_validate_config_flags_empty_pipeline():
    errors = validate_config({"pipeline": {"steps": []}})
    assert any("nothing to run" in e for e in errors)


def test_validate_config_unknown_component():
    errors = validate_config({"pipeline": {"steps": [{"name": "s", "component": "nope_xyz"}]}})
    assert any("unknown component" in e for e in errors)


def test_validate_config_missing_component_and_bad_step():
    errors = validate_config({"pipeline": {"steps": [{"name": "s"}, "notadict"]}})
    assert any("missing 'component'" in e for e in errors)
    assert any("not a mapping" in e for e in errors)


def test_validate_config_detects_missing_producer():
    # generator typically requires context/prompt produced by an earlier step
    errors = validate_config({"pipeline": {"steps": [{"name": "g", "component": "llm_generator"}]}})
    assert any("requires" in e for e in errors)


def test_validate_config_valid_simple_pipeline_has_no_requirement_errors():
    cfg = {
        "pipeline": {
            "steps": [
                {"name": "clean", "component": "query_cleaner"},
                {"name": "retrieve", "component": "coarse_retriever"},
            ]
        }
    }
    errors = validate_config(cfg)
    assert not any("requires" in e for e in errors)
