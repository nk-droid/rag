from components._base import ComponentSettings
from components.ingestion.ingestion_schema import SourceDocument

class SourceNormalizerSettings(ComponentSettings):
    _CONFIG_PATH = "ingestion.source_normalizer"

class SourceNormalizer:
    def __init__(self, settings: SourceNormalizerSettings) -> None:
        self.settings = settings

    def normalize(self, documents: list[SourceDocument]) -> list[dict[str, object]]:
        return [
            {
                "text": document.text,
                "metadata": {
                    "source": document.source,
                    **document.metadata,
                },
            }
            for document in documents
        ]
