from components.shared_types import Chunk, RetrievedChunk
from pipeline.registry_handlers import _chunk_with, _retrieve_with


class _Chunker:
    def chunk(self, text: str) -> list[Chunk]:
        return [
            Chunk(
                text=f"{text} chunk",
                index=0,
                metadata={"chunk_type": "unit"},
            )
        ]


class _Retriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        if query == "primary":
            return [
                RetrievedChunk(
                    id="shared",
                    text="lower scoring duplicate",
                    score=0.2,
                    metadata={"source": "primary"},
                ),
                RetrievedChunk(id="primary-only", text="primary hit", score=0.4),
            ]

        return [
            RetrievedChunk(
                id="shared",
                text="higher scoring duplicate",
                score=0.9,
                metadata={"source": "variant"},
            ),
            RetrievedChunk(id="variant-only", text="variant hit", score=0.8),
        ]


def test_chunk_handler_merges_source_metadata_and_assigns_stable_ids() -> None:
    state = {
        "documents": [
            {
                "text": "alpha",
                "metadata": {
                    "relative_path": "src/app.py",
                    "language": "python",
                },
            }
        ]
    }

    result = _chunk_with(_Chunker(), state, {"app": {"env": "test"}})

    assert result["chunker"] == "_Chunker"
    assert len(result["chunks"]) == 1

    chunk = result["chunks"][0]
    assert chunk.text == "alpha chunk"
    assert chunk.metadata["relative_path"] == "src/app.py"
    assert chunk.metadata["language"] == "python"
    assert chunk.metadata["source_index"] == 0
    assert chunk.metadata["chunk_index"] == 0
    assert chunk.metadata["chunk_type"] == "unit"
    assert chunk.metadata["chunk_id"].startswith("chunk:")


def test_retrieve_handler_expands_queries_dedupes_and_keeps_top_scores() -> None:
    retriever = _Retriever()
    state = {
        "query": "primary",
        "queries": ["primary", "variant"],
        "top_k": 2,
    }

    result = _retrieve_with(retriever, state, {"retrieval": {"top_k": 5}})

    assert retriever.calls == [("primary", 2), ("variant", 2)]
    assert result["retrieval_queries"] == ["primary", "variant"]
    assert [chunk.id for chunk in result["retrieved"]] == [
        "shared",
        "variant-only",
    ]
    assert result["retrieved"][0].text == "higher scoring duplicate"
    assert result["retrieved"][0].score == 0.9
    assert result["retrieved"][0].metadata == {"source": "variant"}
    assert result["cache_hit"]["retrieval"] is False


def test_retrieve_handler_honors_if_under_skip_gate() -> None:
    retriever = _Retriever()
    existing = [
        RetrievedChunk(id="a", text="one"),
        RetrievedChunk(id="b", text="two"),
    ]
    state = {
        "query": "primary",
        "retrieved": existing,
        "_step": {"if_under": 2},
    }

    result = _retrieve_with(retriever, state, {"retrieval": {"top_k": 5}})

    assert retriever.calls == []
    assert result["retrieved"] == existing
    assert result["retrieval_skipped"] == "_Retriever"
