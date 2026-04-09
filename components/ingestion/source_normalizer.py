from components.ingestion.ingestion_schema import SourceDocument

class SourceNormalizer:
    """Convert ingestion documents into pipeline-friendly payloads."""

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
