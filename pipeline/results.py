from typing import Any

from components.shared_types import RetrievedChunk
from pipeline.registry_utils import _answer_text

def extract_answer(state: dict[str, Any]) -> str:
    parsed = state.get("parsed_output")
    if parsed is not None:
        answer = getattr(parsed, "answer", None)
        if answer:
            return str(answer)
        if isinstance(parsed, dict) and parsed.get("answer"):
            return str(parsed["answer"])

    answer = state.get("answer")
    if answer:
        return _answer_text(answer)

    result = state.get("result")
    if result is not None:
        return str(result)

    return ""

def extract_contexts(state: dict[str, Any]) -> list[str]:
    chunks = state.get("retrieved") or []
    texts: list[str] = []
    for chunk in chunks:
        if isinstance(chunk, RetrievedChunk):
            texts.append(chunk.text)
        elif isinstance(chunk, dict):
            texts.append(str(chunk.get("text", "")))
        else:
            texts.append(str(getattr(chunk, "text", chunk)))
    return texts