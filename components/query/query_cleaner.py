from components._base import ComponentSettings

class QueryCleanerSettings(ComponentSettings):
    _CONFIG_PATH = "retrieval.query_clean"

class QueryCleaner:
    def __init__(self, settings: QueryCleanerSettings) -> None:
        self.settings = settings

    def clean(self, query: str) -> str:
        if not query:
            return ""

        cleaned = " ".join(query.split()).strip()
        return cleaned.rstrip("?.!,:;")
