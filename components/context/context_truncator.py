from components._base import ComponentSettings

class ContextTruncatorSettings(ComponentSettings):
    _CONFIG_PATH = "context.truncate"

    max_tokens: int = 256

class ContextTruncator:
    def __init__(self, settings: ContextTruncatorSettings) -> None:
        self.settings = settings

    def truncate(self, context: str, max_tokens: int | None = None) -> str:
        limit = max_tokens if max_tokens is not None else self.settings.max_tokens
        words = context.split()
        return " ".join(words[: int(limit)])
