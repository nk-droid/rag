from dotenv import load_dotenv
from langchain_tavily import TavilySearch

from components.retrieval.base_retriever import BaseRetriever, BaseRetrieverSettings
from components.shared_types import RetrievedChunk

load_dotenv()

class ExternalRetrieverSettings(BaseRetrieverSettings):
    _CONFIG_PATH = "retrieval.external"

    topic: str = "general"
    max_results: int = 5

class ExternalRetriever(BaseRetriever):
    def __init__(self, settings: ExternalRetrieverSettings) -> None:
        super().__init__(settings=settings, store=None)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        max_results = top_k if top_k is not None else self.settings.max_results
        search_engine = TavilySearch(
            max_results=max_results,
            topic=self.settings.topic,
        )

        searched_results = search_engine.invoke({"query": query}).get("results", [])

        return [
            RetrievedChunk(
                id=str(idx),
                text=result.get("content", ""),
                score=round(result.get("score", 0), 2),
                metadata={},
            )
            for idx, result in enumerate(searched_results)
        ]
