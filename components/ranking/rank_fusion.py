from components.shared_types import RetrievedChunk

class RankFusion:
    """Combine multiple ranked result sets into one."""

    def fuse(self, result_sets: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
        fused: list[RetrievedChunk] = []
        for result_set in result_sets:
            fused.extend(result_set)
        return fused
