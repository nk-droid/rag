class ContextTruncator:
    """Trim context to fit a token budget."""

    def truncate(self, context: str, max_tokens: int) -> str:
        words = context.split()
        return " ".join(words[:max_tokens])
