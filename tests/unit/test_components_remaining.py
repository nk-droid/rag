"""Unit tests for components that wrap LLMs / embeddings / vector stores, using
stub dependencies, plus the registry component builder/override logic."""
import json
import types

import pytest

from components.shared_types import Chunk, RetrievedChunk
from pipeline import registry as reg


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_build_component_is_cached():
    reg._COMPONENT_CACHE.clear()
    cfg = {"ranking": {"fusion": {"method": "rrf"}}}
    a = reg._build_component("rank_fusion", cfg)
    b = reg._build_component("rank_fusion", cfg)
    assert a is b


def test_bind_runs_handler():
    run = reg.bind("rank_fusion", lambda component, state, config: {"handled": True})
    assert run({}, {"ranking": {}}) == {"handled": True}


def test_apply_step_overrides():
    from components.ranking.rank_fusion import RankFusionSettings

    component = types.SimpleNamespace(settings=RankFusionSettings(rrf_k=60))
    same = reg._apply_step_overrides(component, {"_step": {"name": "n", "component": "c"}})
    assert same is component  # no overrides
    changed = reg._apply_step_overrides(component, {"_step": {"name": "n", "component": "c", "rrf_k": 99}})
    assert changed.settings.rrf_k == 99 and changed is not component
    # non-ComponentSettings -> returned unchanged
    plain = types.SimpleNamespace(settings=object())
    assert reg._apply_step_overrides(plain, {"_step": {"x": 1}}) is plain


# --------------------------------------------------------------------------- #
# indexers
# --------------------------------------------------------------------------- #
def test_coarse_indexer_writes_json(tmp_path):
    from components.indexer.coarse_indexer import CoarseIndexer, CoarseIndexerSettings

    indexer = CoarseIndexer(CoarseIndexerSettings(path=str(tmp_path / "c.json")))
    assert indexer.index([]) == []
    records = indexer.index([Chunk(text="hello", index=0, metadata={"source": "a"}), {"text": "  "}])
    assert len(records) == 1
    payload = json.loads((tmp_path / "c.json").read_text())
    assert payload["documents"][0]["text"] == "hello"


def test_embedding_indexer_uses_vector_store(tmp_path):
    from components.indexer.embedding_indexer import EmbeddingIndexer, EmbeddingIndexerSettings

    added = {}

    class _FakeStore:
        def add_documents(self, documents=None, ids=None):
            added["docs"] = documents
            added["ids"] = ids

    indexer = EmbeddingIndexer(EmbeddingIndexerSettings(path=str(tmp_path / "f")), vector_store=_FakeStore())
    assert indexer.index([]) == []
    assert indexer.index([{"text": ""}]) == []  # empty text skipped
    records = indexer.index([Chunk(text="t", index=0, metadata={"source": "s"})])
    assert len(records) == 1 and len(added["ids"]) == 1


# --------------------------------------------------------------------------- #
# prompt_builder
# --------------------------------------------------------------------------- #
def test_prompt_builder_build_and_cache():
    from components.generation.prompt_builder import PromptBuilder, PromptBuilderSettings

    builder = PromptBuilder(PromptBuilderSettings())
    prompt = builder.build("summarize.yaml")
    assert prompt.template
    cached = builder.build("summarize.yaml")
    assert cached is prompt  # cache hit
    with_parser = builder.build("summarize.yaml", parser_model="Answer")
    assert "format_instructions" in with_parser.partial_variables
    with pytest.raises(ValueError):
        builder.build("summarize.yaml", parser_model="NopeModel")


# --------------------------------------------------------------------------- #
# query rewriter / multi-query
# --------------------------------------------------------------------------- #
def _llm_stub(parsed):
    generator = types.SimpleNamespace(generate=lambda prompt, inputs: "raw")
    prompt_builder = types.SimpleNamespace(build=lambda name, model: object())
    parser = types.SimpleNamespace(parse=lambda text, model: parsed)
    return generator, prompt_builder, parser


def test_query_rewriter():
    from components.query.rewriter import QueryRewriter, QueryRewriterSettings

    gen, pb, parser = _llm_stub(types.SimpleNamespace(query="rewritten"))
    rw = QueryRewriter(QueryRewriterSettings(), gen, pb, parser)
    assert rw.rewrite("  hello ") == "rewritten"
    assert rw.rewrite("") == ""
    # parser raises -> fallback to cleaned
    bad = QueryRewriter(QueryRewriterSettings(), gen, pb, types.SimpleNamespace(parse=lambda t, m: (_ for _ in ()).throw(ValueError())))
    assert bad.rewrite("hi") == "hi"
    assert QueryRewriter._to_text(types.SimpleNamespace(content="c")) == "c"


def test_multi_query():
    from components.query.multi_query import MultiQueryGenerator, MultiQueryGeneratorSettings

    gen, pb, parser = _llm_stub(types.SimpleNamespace(queries=["a", "a", "b"]))
    mq = MultiQueryGenerator(MultiQueryGeneratorSettings(max_queries=3), gen, pb, parser)
    out = mq.generate("orig")
    assert out[0] == "orig" and "a" in out and "b" in out
    assert mq.generate("") == []
    bad = MultiQueryGenerator(MultiQueryGeneratorSettings(), gen, pb, types.SimpleNamespace(parse=lambda t, m: (_ for _ in ()).throw(ValueError())))
    assert bad.generate("q") == ["q"]


# --------------------------------------------------------------------------- #
# postprocessing
# --------------------------------------------------------------------------- #
def test_refiner():
    from components.postprocessing.refiner import Refiner, RefinerSettings

    gen, pb, parser = _llm_stub(types.SimpleNamespace(answer="better"))
    refiner = Refiner(RefinerSettings(), gen, pb, parser)
    # no refine needed -> returns raw
    assert refiner.refine("orig", {"needs_refine": False}) == "orig"
    # needs refine -> json with refined answer
    out = refiner.refine("orig", {"needs_refine": True})
    assert json.loads(out)["answer"] == "better"
    # extract answer from json input
    assert refiner._extract_answer_text('{"answer": "x"}') == "x"
    assert refiner._extract_answer_text("plain") == "plain"


def test_self_critic():
    from components.postprocessing.self_critic import SelfCritic, SelfCriticSettings

    from components.generation.output_parser import SelfCritique

    gen, pb, parser = _llm_stub(SelfCritique(needs_refine=True, grounded=False, issues=["x"], suggestions=["y"]))
    critic = SelfCritic(SelfCriticSettings(), gen, pb, parser)
    out = critic.critique("an answer", "context")
    assert out["needs_refine"] is True
    # empty answer -> needs refine without calling llm
    empty = critic.critique("", "ctx")
    assert empty["needs_refine"] is True and "empty_answer" in empty["issues"]
    # parser error -> safe default
    bad = SelfCritic(SelfCriticSettings(), gen, pb, types.SimpleNamespace(parse=lambda t, m: (_ for _ in ()).throw(ValueError())))
    assert bad.critique("ans", "ctx")["needs_refine"] is False


# --------------------------------------------------------------------------- #
# rankers
# --------------------------------------------------------------------------- #
def test_embedding_ranker(monkeypatch):
    import components.ranking.embedding_ranker as er

    class _FakeEmbeddings:
        def __init__(self, model):
            pass

        def embed_query(self, q):
            return [1.0, 0.0]

        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(er, "OllamaEmbeddings", _FakeEmbeddings)
    ranker = er.EmbeddingRanker(er.EmbeddingRankerSettings(strategy="cosine", top_n=2))
    cands = [RetrievedChunk(id="a", text="t1"), RetrievedChunk(id="b", text="t2")]
    assert ranker.rank("q", []) == []
    out = ranker.rank("q", cands)
    assert len(out) <= 2
    # second call hits the embedding caches
    ranker.rank("q", cands)
    nocache = er.EmbeddingRanker(er.EmbeddingRankerSettings(strategy="cosine", use_cache=False))
    assert nocache.rank("q", cands)


def test_cross_encoder_ranker():
    from components.ranking.cross_encoder_ranker import CrossEncoderRanker, CrossEncoderRankerSettings

    ranker = CrossEncoderRanker(CrossEncoderRankerSettings(top_n=1))
    ranker.model = types.SimpleNamespace(predict=lambda pairs: [0.1, 0.9])
    assert ranker.rank("q", []) == []
    out = ranker.rank("q", [RetrievedChunk(id="a", text="t1"), RetrievedChunk(id="b", text="t2")])
    assert len(out) == 1 and out[0].id == "b"


# --------------------------------------------------------------------------- #
# retrievers
# --------------------------------------------------------------------------- #
def test_fine_retriever():
    from components.retrieval.fine_retriever import FineRetriever, FineRetrieverSettings

    class _FakeStore:
        def similarity_search_with_score(self, query, k):
            return [
                (types.SimpleNamespace(page_content="hit", metadata={"id": "d1"}), 0.9),
                (types.SimpleNamespace(page_content="  ", metadata={}), 0.1),
            ]

    retriever = FineRetriever(FineRetrieverSettings(), store=_FakeStore())
    out = retriever.retrieve("q", top_k=5)
    assert len(out) == 1 and out[0].id == "d1"
    assert FineRetriever(FineRetrieverSettings(), store=None).retrieve("q") == []
    assert retriever.retrieve("  ") == []


def test_hybrid_retriever():
    from components.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverSettings

    dense = types.SimpleNamespace(retrieve=lambda q, top_k: [RetrievedChunk(id="d", text="dt", score=0.8)])
    sparse = types.SimpleNamespace(retrieve=lambda q, top_k: [RetrievedChunk(id="s", text="st", score=0.4)])
    hybrid = HybridRetriever(HybridRetrieverSettings(), dense_retriever=dense, sparse_retriever=sparse)
    candidates = hybrid.retrieve_candidates("q", top_k=3)
    assert candidates["dense"] and candidates["sparse"]
    out = hybrid.retrieve("q", top_k=5)
    assert {c.id for c in out} == {"d", "s"}
    # _normalize: equal scores -> all 1.0
    normed = HybridRetriever._normalize([RetrievedChunk(id="a", text="a", score=0.5), RetrievedChunk(id="b", text="b", score=0.5)])
    assert all(c.score == 1.0 for c in normed)
    assert HybridRetriever._normalize([]) == []
