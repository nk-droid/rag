class QueryCleaner:
    """Normalize raw user queries before downstream processing."""

    def clean(self, query: str) -> str:
        if not query:
            return ""

        cleaned = " ".join(query.split()).strip()
        return cleaned.rstrip("?.!,:;")