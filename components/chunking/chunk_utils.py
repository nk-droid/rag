def estimate_tokens(text: str) -> int:
    return len(text.split())

def merge_small_chunks(chunks: list[str], min_length: int = 20) -> list[str]:
    merged: list[str] = []
    buffer = ""
    for chunk in chunks:
        candidate = f"{buffer} {chunk}".strip() if buffer else chunk
        if len(candidate) < min_length:
            buffer = candidate
            continue
        merged.append(candidate)
        buffer = ""
    if buffer:
        merged.append(buffer)
    return merged
