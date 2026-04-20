from components.shared_types import RetrievedChunk

from components.shared_types import RetrievedChunk
from components.retrieval.base_retriever import BaseRetriever
from langchain_tavily import TavilySearch

from dotenv import load_dotenv
load_dotenv()

class ExternalRetriever(BaseRetriever):
    """Retrieve context from external systems or APIs."""

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        search_engine = TavilySearch(
            max_results = top_k,
            topic = "general"
        )

        searched_results = search_engine.invoke({"query": query}).get("results", [])
        
        return [
            RetrievedChunk(
                id=idx,
                text=result.get("content", ""),
                score=round(result.get("score", 0), 2),
                metadata={}
            ) for idx, result in enumerate(searched_results)
        ]
