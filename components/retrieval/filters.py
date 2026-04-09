from collections.abc import Iterable

from components.shared_types import RetrievedChunk

def filter_by_score(
    results: Iterable[RetrievedChunk],
    min_score: float = 0.0,
) -> list[RetrievedChunk]:
    return [result for result in results if result.score >= min_score]

def filter_by_metadata(
    results: Iterable[RetrievedChunk],
    required_metadata: dict[str, object] | None = None,
) -> list[RetrievedChunk]:
    if not required_metadata:
        return list(results)
    return [
        result
        for result in results
        if all(result.metadata.get(key) == value for key, value in required_metadata.items())
    ]

def dedupe_results(results: Iterable[RetrievedChunk]) -> list[RetrievedChunk]:
    seen_ids: set[str] = set()
    deduped: list[RetrievedChunk] = []
    for result in results:
        if result.id in seen_ids:
            continue
        seen_ids.add(result.id)
        deduped.append(result)
    return deduped
