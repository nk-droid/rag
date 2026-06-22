"""Unit coverage for every handler in pipeline.registry_handlers using stub
components. Cache-dependent handlers are exercised on both miss and hit paths."""
import types

import pytest

from components.shared_types import Chunk, RetrievedChunk
from pipeline import registry_handlers as rh


# --------------------------------------------------------------------------- #
# Stub components
# --------------------------------------------------------------------------- #
class _Loader:
    def load(self, source):
        return [types.SimpleNamespace(text=f"doc::{source}", source=source, metadata={})]


class _Normalizer:
    def normalize(self, documents):
        return [{"text": getattr(d, "text", ""), "metadata": {}} for d in documents]


class _Chunker:
    def chunk(self, text):
        return [Chunk(text=text, index=0, metadata={})]


class _Indexer:
    index_path = "data/indices/x.json"

    def index(self, chunks):
        return [object() for _ in chunks]


class _Retriever:
    def retrieve(self, query, top_k=5):
        return [RetrievedChunk(id=f"{query}-{i}", text=f"t{i}", score=1.0 - i * 0.1) for i in range(top_k)]


class _HybridRetriever:
    candidate_multiplier = 2

    def retrieve_candidates(self, query, top_k=5):
        return {
            "sparse": [RetrievedChunk(id=f"s-{query}", text="s", score=0.5)],
            "dense": [RetrievedChunk(id=f"d-{query}", text="d", score=0.6)],
        }

    def retrieve(self, query, top_k=5):
        return [RetrievedChunk(id="combined", text="c", score=0.9)]


class _Fusion:
    def fuse(self, groups):
        out = []
        for g in groups:
            out.extend(g)
        return out


class _Ranker:
    def rank(self, query, candidates):
        return list(reversed(candidates))


class _Generator:
    def generate(self, prompt, inputs):
        return f"answer::{inputs['query']}::{inputs['context'][:5]}"


class _StreamGen:
    def stream(self, prompt, inputs):
        yield "a"
        yield "b"


class _Expander:
    settings = types.SimpleNamespace(max_expanded_chunks=10)

    def expand(self, chunks, top_k=None):
        return [RetrievedChunk(id="exp", text="expanded", score=0.3)]


def _cfg(**over):
    base = {"retrieval": {"top_k": 3}, "cache": {"enabled": False}, "models": {"llm": {"temperature": 0}}}
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# ingest / normalize / chunk / index
# --------------------------------------------------------------------------- #
def test_ingest_with_string_and_list_sources():
    out = rh._ingest_with(_Loader(), {"sources": "s1"}, _cfg())
    assert len(out["documents"]) == 1 and out["ingestion_loader"] == "_Loader"
    out = rh._ingest_with(_Loader(), {"data_sources": ["a", "b"]}, _cfg())
    assert len(out["documents"]) == 2


def test_normalize_sources_with():
    out = rh._normalize_sources_with(_Normalizer(), {"documents": [types.SimpleNamespace(text="x")]}, _cfg())
    assert out["data_sources"] == [{"text": "x", "metadata": {}}]


def test_chunk_with_assigns_ids_and_skips_metadata_less_chunks():
    class _BadChunker:
        def chunk(self, text):
            return [types.SimpleNamespace(text=text)]  # no metadata attr

    out = rh._chunk_with(_Chunker(), {"text": "hello world"}, _cfg())
    assert out["chunks"][0].metadata["chunk_id"].startswith("chunk:")
    out2 = rh._chunk_with(_BadChunker(), {"text": "hello"}, _cfg())
    assert out2["chunks"][0].text == "hello"  # passed through, no chunk_id


def test_index_with():
    out = rh._index_with(_Indexer(), {"chunks": [1, 2, 3]}, _cfg())
    assert out["indexed_count"] == 3 and out["vector_store_path"] == "data/indices/x.json"


# --------------------------------------------------------------------------- #
# retrieve
# --------------------------------------------------------------------------- #
def test_retrieve_with_basic():
    out = rh._retrieve_with(_Retriever(), {"query": "q"}, _cfg())
    assert len(out["retrieved"]) == 3
    assert out["cache_hit"]["retrieval"] is False


def test_retrieve_with_if_under_skip():
    state = {"query": "q", "retrieved": [1, 2, 3], "_step": {"if_under": 2}}
    out = rh._retrieve_with(_Retriever(), state, _cfg())
    assert out["retrieval_skipped"] == "_Retriever"


def test_retrieve_with_multi_query_merges():
    out = rh._retrieve_with(_Retriever(), {"query": "q", "queries": ["q", "q2"]}, _cfg())
    assert out["retrieval_queries"] == ["q", "q2"]


def test_retrieve_with_no_query_returns_empty():
    out = rh._retrieve_with(_Retriever(), {}, _cfg())
    assert out["retrieved"] == []


def test_retrieve_with_cache_miss_then_hit():
    rh._get_cache.__wrapped__ if hasattr(rh._get_cache, "__wrapped__") else None
    cfg = _cfg(cache={"enabled": True, "type": "in_memory", "features": {"retrieval": True}})
    state = {"query": "cachetest"}
    out1 = rh._retrieve_with(_Retriever(), dict(state), cfg)
    assert out1["cache_hit"]["retrieval"] is False
    out2 = rh._retrieve_with(_Retriever(), dict(state), cfg)
    assert out2["cache_hit"]["retrieval"] is True


def test_retrieve_with_merge_with_existing():
    existing = [RetrievedChunk(id="old", text="o", score=0.95)]
    state = {"query": "q", "retrieved": existing, "_step": {"merge_with_existing": True}}
    out = rh._retrieve_with(_Retriever(), state, _cfg())
    ids = [c.id for c in out["retrieved"]]
    assert "old" in ids


# --------------------------------------------------------------------------- #
# hybrid retrieve
# --------------------------------------------------------------------------- #
def test_hybrid_retrieve_non_fuse():
    out = rh._hybrid_retrieve_with(_HybridRetriever(), {"query": "q"}, _cfg())
    assert out["retrieved"][0].id == "combined"
    assert out["sparse_retrieved"] and out["dense_retrieved"]


def test_hybrid_retrieve_fuse(monkeypatch):
    monkeypatch.setattr(rh, "_build_aux_component", lambda name, config: _Fusion())
    out = rh._hybrid_retrieve_with(_HybridRetriever(), {"query": "q", "_step": {"fuse": True}}, _cfg())
    assert out["retrieved"]


def test_hybrid_retrieve_cache_hit():
    cfg = _cfg(cache={"enabled": True, "type": "in_memory", "features": {"retrieval": True}})
    state = {"query": "hybridcache"}
    rh._hybrid_retrieve_with(_HybridRetriever(), dict(state), cfg)
    out = rh._hybrid_retrieve_with(_HybridRetriever(), dict(state), cfg)
    assert out["cache_hit"]["retrieval"] is True


# --------------------------------------------------------------------------- #
# rank fusion / rank
# --------------------------------------------------------------------------- #
def test_rank_fusion_with_sparse_dense():
    state = {
        "sparse_retrieved": [RetrievedChunk(id="s", text="s", score=0.5)],
        "dense_retrieved": [RetrievedChunk(id="d", text="d", score=0.6)],
        "_step": {"top_k": 1},
    }
    out = rh._rank_fusion_with(_Fusion(), state, _cfg())
    assert len(out["retrieved"]) == 1


def test_rank_fusion_with_fallback_to_retrieved():
    out = rh._rank_fusion_with(_Fusion(), {"retrieved": [1, 2, 3]}, _cfg())
    assert out["retrieved"] == [1, 2, 3]


def test_rank_with():
    out = rh._rank_with(_Ranker(), {"query": "q", "retrieved": [1, 2, 3]}, _cfg())
    assert out["ranked"] == [3, 2, 1]


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #
def _prompt():
    return types.SimpleNamespace(template="tmpl", partial_variables={})


def test_generate_with_builds_context_from_ranked():
    state = {"query": "q", "ranked": [RetrievedChunk(id="a", text="ranktext")], "prompt": _prompt()}
    out = rh._generate_with(_Generator(), state, _cfg())
    assert out["answer"].startswith("answer::q::")
    assert out["cache_hit"]["generation"] is False


def test_generate_with_cache_hit():
    cfg = _cfg(cache={"enabled": True, "type": "in_memory", "features": {"generation": True}})
    state = {"query": "gcache", "context": "ctx", "prompt": _prompt()}
    rh._generate_with(_Generator(), dict(state), cfg)
    out = rh._generate_with(_Generator(), dict(state), cfg)
    assert out["cache_hit"]["generation"] is True


# --------------------------------------------------------------------------- #
# graph expand / context handlers
# --------------------------------------------------------------------------- #
def test_graph_expand_with():
    state = {"retrieved": [RetrievedChunk(id="r", text="r", score=0.9)]}
    out = rh._graph_expand_with(_Expander(), state, _cfg())
    assert out["graph_expanded"][0].id == "exp"
    assert any(c.id == "exp" for c in out["retrieved"])


def test_context_build_merge_truncate():
    builder = types.SimpleNamespace(build=lambda chunks: "built")
    assert rh._context_build_with(builder, {"retrieved": [1]}, _cfg())["context"] == "built"
    merger = types.SimpleNamespace(merge=lambda chunks: ["m"])
    assert rh._context_merge_with(merger, {"retrieved": [1, 2]}, _cfg())["retrieved"] == ["m"]
    trunc = types.SimpleNamespace(truncate=lambda ctx, mx: ctx[:mx] if mx else ctx)
    out = rh._context_truncate_with(trunc, {"context": "abcdef", "max_tokens": 3}, _cfg())
    assert out["context"] == "abc"


# --------------------------------------------------------------------------- #
# query transforms
# --------------------------------------------------------------------------- #
def test_query_transform_handlers():
    cleaner = types.SimpleNamespace(clean=lambda q: q.strip())
    assert rh._clean_query_with(cleaner, {"query": " hi "}, _cfg())["query"] == "hi"
    rewriter = types.SimpleNamespace(rewrite=lambda q: q + "?")
    assert rh._rewrite_query_with(rewriter, {"query": "hi"}, _cfg())["query"] == "hi?"
    mq = types.SimpleNamespace(generate=lambda q: [q, q + "2"])
    assert rh._multi_query_with(mq, {"query": "hi"}, _cfg())["queries"] == ["hi", "hi2"]


# --------------------------------------------------------------------------- #
# streaming
# --------------------------------------------------------------------------- #
def test_stream_generate_with():
    state = {"query": "q", "ranked": [RetrievedChunk(id="a", text="x")], "prompt": _prompt()}
    out = rh._stream_generate_with(_StreamGen(), state, _cfg())
    assert out["stream"] == ["a", "b"] and out["answer"] == "ab"


def test_stream_generate_requires_prompt():
    with pytest.raises(ValueError):
        rh._stream_generate_with(_StreamGen(), {"query": "q"}, _cfg())


# --------------------------------------------------------------------------- #
# prompt / parse / memory / eval / critique / refine
# --------------------------------------------------------------------------- #
def test_build_prompt_and_parse_output():
    builder = types.SimpleNamespace(build=lambda template_name, parser_model: f"P:{template_name}")
    out = rh._build_prompt_with(builder, {"_step": {"template_name": "t.yaml"}}, _cfg())
    assert out["prompt"] == "P:t.yaml"
    parser = types.SimpleNamespace(parse=lambda text, parser_model: {"answer": text})
    out = rh._parse_output_with(parser, {"answer": "hi"}, _cfg())
    assert out["parsed_output"] == {"answer": "hi"}


def test_memory_handlers():
    writer = types.SimpleNamespace(write=lambda interaction: interaction)
    out = rh._memory_write_with(writer, {"answer": "a", "memory_id": "m1"}, _cfg())
    assert out["memory_record"]["id"] == "m1"
    store = types.SimpleNamespace(search=lambda q, top_k: ["mem"])
    assert rh._memory_retrieve_with(store, {"query": "q"}, _cfg())["memories"] == ["mem"]
    mfilter = types.SimpleNamespace(filter=lambda mems: mems[:1])
    assert rh._memory_filter_with(mfilter, {"memories": ["a", "b"]}, _cfg())["memories"] == ["a"]


def test_evaluate_critique_refine():
    evaluator = types.SimpleNamespace(evaluate=lambda payload: {"score": 1})
    assert rh._evaluate_with(evaluator, {"x": 1}, _cfg())["evaluation"] == {"score": 1}
    critic = types.SimpleNamespace(critique=lambda ans, ctx: {"ok": True})
    assert rh._critique_with(critic, {"answer": "a", "context": "c"}, _cfg())["critique"] == {"ok": True}
    refiner = types.SimpleNamespace(refine=lambda ans, crit: "refined")
    assert rh._refine_with(refiner, {"answer": "a", "critique": {}}, _cfg())["answer"] == "refined"


def test_build_aux_component_is_cached(monkeypatch):
    calls = {"n": 0}

    def _factory(config):
        calls["n"] += 1
        return object()

    monkeypatch.setitem(rh.COMPONENT_FACTORIES, "rank_fusion", _factory)
    rh._AUX_COMPONENT_CACHE.clear()
    a = rh._build_aux_component("rank_fusion", _cfg())
    b = rh._build_aux_component("rank_fusion", _cfg())
    assert a is b and calls["n"] == 1
