import pytest

import pipeline.component_factories as component_factories


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-mark tests by the directory they live in (unit/integration/e2e)."""
    for item in items:
        parts = set(item.path.parts)
        if "e2e" in parts:
            item.add_marker(pytest.mark.e2e)
        elif "integration" in parts:
            item.add_marker(pytest.mark.integration)
        elif "unit" in parts:
            item.add_marker(pytest.mark.unit)

class _DummyLLM:
    def invoke(self, prompt):
        return "dummy-response"

class _DummyVectorStore:
    def add_documents(self, documents=None, ids=None):
        return None

class _StubCrossEncoderRanker:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.settings = kwargs.get("settings")

    def rank(self, query, candidates):
        return list(candidates)

class _StubEmbeddingRanker:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.settings = kwargs.get("settings")

    def rank(self, query, candidates):
        return list(candidates)

class _StubRagasEvaluator:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.settings = kwargs.get("settings")

    def evaluate(self, samples):
        return {"rows": len(samples.get("question", []))}

class _StubSemanticChunker:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.settings = kwargs.get("settings")

    def chunk(self, text):
        return []

@pytest.fixture()
def minimal_config() -> dict:
    return {
        "app": {"env": "test"},
        "models": {
            "llm": {
                "provider": "ollama",
                "model_name": "mock-llm",
                "temperature": 0.0,
                "max_tokens": 128,
            },
            "embedding": {
                "provider": "ollama",
                "model_name": "mock-embed",
            },
        },
        "vector_store": {"provider": "faiss"},
        "retrieval": {
            "top_k": 5,
            "hybrid": {"candidate_multiplier": 2},
            "query_expansion": {"max_queries": 3},
        },
        "ranking": {
            "embedding": {
                "model_name": "mock-embed",
                "top_n": 3,
            },
            "cross_encoder": {
                "model_name": "mock-cross-encoder",
                "top_n": 3,
            },
        },
        "indexers": {
            "embedding": {
                "path": "data/indices/faiss_index",
                "vector_store": {"provider": "faiss"},
            },
            "coarse": {"path": "data/indices/coarse_index.json"},
        },
        "chunking": {
            "recursive": {"chunk_size": 256, "chunk_overlap": 32},
            "semantic": {
                "template_name": "chunk.yaml",
                "parser_model": "SemanticChunks",
            },
        },
        "cache": {"enabled": False},
    }

@pytest.fixture()
def patched_factory_dependencies(monkeypatch):
    monkeypatch.setattr(component_factories, "get_llm", lambda config: _DummyLLM())
    monkeypatch.setattr(component_factories, "get_vector_store", lambda config: _DummyVectorStore())
    monkeypatch.setattr(component_factories, "CrossEncoderRanker", _StubCrossEncoderRanker)
    monkeypatch.setattr(component_factories, "EmbeddingRanker", _StubEmbeddingRanker)
    monkeypatch.setattr(component_factories, "RagasEvaluator", _StubRagasEvaluator)
    monkeypatch.setattr(component_factories, "SemanticChunker", _StubSemanticChunker)
    return monkeypatch
