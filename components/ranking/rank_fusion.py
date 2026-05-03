from components._base import ComponentSettings
from components.shared_types import RetrievedChunk

class RankFusionSettings(ComponentSettings):
    _CONFIG_PATH = "ranking.fusion"

class RankFusion:
    def __init__(self, settings: RankFusionSettings) -> None:
        self.settings = settings

    def fuse(self, result_sets: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
        fused: list[RetrievedChunk] = []
        for result_set in result_sets:
            fused.extend(result_set)
        return fused
