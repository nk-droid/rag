import json

import numpy as np
import pytest

from components.chunking.chunk_utils import estimate_tokens, merge_small_chunks
from components.chunking.code_aware_chunker import CodeAwareChunker, CodeAwareChunkerSettings
from components.chunking.late_chunker import LateChunker, LateChunkerSettings
from components.context.context_builder import ContextBuilder, ContextBuilderSettings
from components.context.context_merger import ContextMerger, ContextMergerSettings
from components.context.context_truncator import ContextTruncator, ContextTruncatorSettings
from components.evaluation.evaluator import Evaluator
from components.generation.streaming_generator import StreamingGenerator, StreamingGeneratorSettings
from components.indexer.local_vector_db import LocalVectorDB
from components.memory.memory_store import MemoryStore, MemoryStoreSettings
from components.query.query_cleaner import QueryCleaner, QueryCleanerSettings
from components.ranking.colbert_ranker import ColBERTRanker, ColBERTRankerSettings
from components.ranking.scoring_utils import CosineScoring, MMRScoring, normalize_scores, sort_by_score
from components.retrieval.filters import dedupe_results, filter_by_metadata, filter_by_score
from components.retrieval.graph_retriever import GraphRetriever, GraphRetrieverSettings
from components.retrieval.memory_retriever import MemoryRetriever, MemoryRetrieverSettings
from components.shared_types import MemoryRecord, RetrievedChunk

def test_query_cleaner_normalizes_whitespace_and_punctuation() -> None:
    cleaner = QueryCleaner(QueryCleanerSettings())
    assert cleaner.clean("  hello   world??  ") == "hello world"
    assert cleaner.clean("") == ""

def test_context_components_roundtrip() -> None:
    chunks = [
        RetrievedChunk(id="1", text="alpha", score=0.8),
        RetrievedChunk(id="2", text="alpha", score=0.7),
        RetrievedChunk(id="3", text="beta", score=0.5),
    ]

    builder = ContextBuilder(ContextBuilderSettings())
    merger = ContextMerger(ContextMergerSettings())
    truncator = ContextTruncator(ContextTruncatorSettings())

    merged = merger.merge(chunks)
    assert [chunk.text for chunk in merged] == ["alpha", "beta"]

    context = builder.build(merged)
    assert context == "[source: unknown]\nalpha\n\n[source: unknown]\nbeta"

    assert truncator.truncate("one two three", max_tokens=2) == "one two"

def test_chunk_utils() -> None:
    assert estimate_tokens("one two three") == 3
    assert merge_small_chunks(["a", "bc", "defgh"], min_length=4) == ["a bc", "defgh"]

def test_code_aware_chunker_extracts_python_symbols_and_line_ranges() -> None:
    source = "\n".join(
        [
            "import os",
            "from pathlib import Path",
            "",
            "class Service:",
            "    def run(self):",
            "        return Path(os.getcwd())",
            "",
            "async def fetch():",
            "    return 'ok'",
        ]
    )

    chunks = CodeAwareChunker(CodeAwareChunkerSettings()).chunk(source)

    assert [chunk.metadata["chunk_type"] for chunk in chunks] == [
        "imports",
        "class",
        "method",
        "function",
    ]
    assert [chunk.metadata["symbol"] for chunk in chunks] == [
        None,
        "Service",
        "Service.run",
        "fetch",
    ]
    assert chunks[0].text == "import os\nfrom pathlib import Path"
    assert chunks[2].metadata["start_line"] == 5
    assert chunks[2].metadata["end_line"] == 6
    assert all(chunk.metadata["chunk_id"].startswith("chunk:") for chunk in chunks)

def test_code_aware_chunker_splits_markdown_by_heading() -> None:
    source = "# Intro\nalpha\n## Details\nbeta\n"

    chunks = CodeAwareChunker(CodeAwareChunkerSettings()).chunk(source)

    assert [chunk.text for chunk in chunks] == ["# Intro\nalpha", "## Details\nbeta"]
    assert [chunk.metadata["title"] for chunk in chunks] == ["Intro", "Details"]
    assert [chunk.metadata["chunk_type"] for chunk in chunks] == ["section", "section"]
    assert [chunk.metadata["start_line"] for chunk in chunks] == [1, 3]

def test_code_aware_chunker_falls_back_to_overlapping_text_windows() -> None:
    source = "a" * 250
    settings = CodeAwareChunkerSettings(chunk_size=100, chunk_overlap=10)

    chunks = CodeAwareChunker(settings).chunk(source)

    assert [len(chunk.text) for chunk in chunks] == [100, 100, 70]
    assert [chunk.metadata["title"] for chunk in chunks] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
    ]
    assert all(chunk.metadata["chunk_type"] == "text" for chunk in chunks)

def test_retrieval_filters() -> None:
    chunks = [
        RetrievedChunk(id="1", text="alpha", score=0.2, metadata={"source": "a"}),
        RetrievedChunk(id="2", text="beta", score=0.9, metadata={"source": "b"}),
        RetrievedChunk(id="1", text="alpha-dup", score=0.4, metadata={"source": "a"}),
    ]

    assert [c.id for c in filter_by_score(chunks, min_score=0.3)] == ["2", "1"]
    assert [c.id for c in filter_by_metadata(chunks, {"source": "a"})] == ["1", "1"]
    assert [c.id for c in dedupe_results(chunks)] == ["1", "2"]

def test_scoring_utils() -> None:
    chunks = [
        RetrievedChunk(id="a", text="a", score=1.0),
        RetrievedChunk(id="b", text="b", score=0.5),
    ]

    normalized = normalize_scores(chunks)
    assert normalized[0].score == pytest.approx(1.0)
    assert normalized[1].score == pytest.approx(0.5)
    assert [c.id for c in sort_by_score(chunks)] == ["a", "b"]

    docs = [
        RetrievedChunk(id="1", text="doc1"),
        RetrievedChunk(id="2", text="doc2"),
        RetrievedChunk(id="3", text="doc3"),
    ]
    query_vec = np.array([[1.0, 0.0]])
    doc_vecs = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

    cosine_selected = CosineScoring().select(query_vec, doc_vecs, docs, top_n=2)
    assert len(cosine_selected) == 2

    mmr_selected = MMRScoring(lambda_param=0.7).select(query_vec, doc_vecs, docs, top_n=2)
    assert len(mmr_selected) == 2

def test_memory_store_basic_ops() -> None:
    store = MemoryStore(MemoryStoreSettings())
    record = store.add(MemoryRecord(id="m1", content="hello", metadata={"topic": "greet"}))
    assert record.id == "m1"
    assert len(store.all()) == 1
    assert len(store.search("hello", top_k=1)) == 1

def test_graph_retriever_uses_graph_nodes_and_evidence_chunks(tmp_path) -> None:
    graph_path = tmp_path / "repo_graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "version": 1,
                "nodes": [
                    {
                        "id": "file:services/user.py",
                        "type": "File",
                        "label": "services/user.py",
                        "metadata": {"path": "services/user.py"},
                    },
                    {
                        "id": "symbol:services/user.py:UserService",
                        "type": "Class",
                        "label": "UserService",
                        "metadata": {"path": "services/user.py"},
                    },
                ],
                "edges": [
                    {
                        "source": "file:services/user.py",
                        "relation": "DEFINES",
                        "target": "symbol:services/user.py:UserService",
                        "evidence_chunk_id": "chunk:user-service",
                        "metadata": {},
                    }
                ],
                "chunks": [
                    {
                        "id": "chunk:user-service",
                        "text": "class UserService:\n    pass",
                        "metadata": {
                            "path": "services/user.py",
                            "symbol": "UserService",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    retriever = GraphRetriever(GraphRetrieverSettings(path=str(graph_path)))
    results = retriever.retrieve("Where is UserService defined?", top_k=3)

    assert [chunk.id for chunk in results] == ["chunk:user-service"]
    assert results[0].metadata["retrieval_source"] == "graph_retriever"
    assert results[0].metadata["graph_score"] > 0

def test_streaming_generator_streams_llm_pieces() -> None:
    class _FakeStreamLLM:
        def stream(self, prompt: str):
            assert "hello" in prompt
            for piece in ["Hi", " there", "!"]:
                yield piece

    from langchain_core.prompts import PromptTemplate

    template = PromptTemplate.from_template("greet {query}")
    generator = StreamingGenerator(StreamingGeneratorSettings(), llm=_FakeStreamLLM())
    pieces = list(generator.stream(template, {"query": "hello"}))
    assert pieces == ["Hi", " there", "!"]
    assert "".join(pieces) == "Hi there!"

def test_known_stubs_raise_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        LateChunker(LateChunkerSettings()).chunk("text")

    with pytest.raises(NotImplementedError):
        MemoryRetriever(MemoryRetrieverSettings()).retrieve("query", top_k=3)

    with pytest.raises(NotImplementedError):
        ColBERTRanker(ColBERTRankerSettings()).rank("query", [])

    with pytest.raises(NotImplementedError):
        LocalVectorDB("tmp").load()

    with pytest.raises(NotImplementedError):
        LocalVectorDB("tmp").add_records([])

    with pytest.raises(NotImplementedError):
        LocalVectorDB("tmp").count()

    with pytest.raises(NotImplementedError):
        Evaluator().evaluate({})
