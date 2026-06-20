from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate as ragas_evaluate
from ragas.metrics import _answer_relevancy, _context_precision, _context_recall, _faithfulness
from ragas.run_config import RunConfig

from components._base import ComponentSettings
from components.evaluation.evaluator import Evaluator

RAGAS_METRICS = {
    "faithfulness": _faithfulness,
    "answer_relevancy": _answer_relevancy,
    "context_precision": _context_precision,
    "context_recall": _context_recall,
}

_NEEDS_REFERENCE = {"context_precision", "context_recall"}
_DEFAULT_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

class RagasEvaluatorSettings(ComponentSettings):
    _CONFIG_PATH = "evaluation.ragas"

    llm_model: str = "gemma4:31b-cloud"
    embedding_model: str = "qwen3-embedding:0.6b"
    metrics: list[str] = _DEFAULT_METRICS
    max_workers: int = 2
    timeout: int = 600
    max_retries: int = 3

class RagasEvaluator(Evaluator):
    def __init__(self, settings: RagasEvaluatorSettings) -> None:
        self.settings = settings
        self.llm = ChatOllama(model=settings.llm_model)
        self.embeddings = OllamaEmbeddings(model=settings.embedding_model)

    def score_frame(self, samples: dict, metrics: list[str] | None = None):
        names = metrics or list(self.settings.metrics)

        questions = list(samples.get("question", []))
        answers = list(samples.get("answer", []))
        contexts = list(samples.get("retrieved_contexts", samples.get("contexts", [])))
        references = list(samples.get("ground_truth", []))
        if not questions or not answers:
            return None, {}

        has_reference = bool(references) and all(bool(r) for r in references)
        selected = [
            name
            for name in names
            if name in RAGAS_METRICS and (has_reference or name not in _NEEDS_REFERENCE)
        ]
        if not selected:
            return None, {}

        columns = {
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": [list(c or []) for c in contexts],
        }
        if has_reference:
            columns["reference"] = references

        dataset = Dataset.from_dict(columns)
        run_config = RunConfig(
            timeout=self.settings.timeout,
            max_workers=self.settings.max_workers,
            max_retries=self.settings.max_retries,
        )
        result = ragas_evaluate(
            dataset,
            metrics=[RAGAS_METRICS[name] for name in selected],
            llm=self.llm,
            embeddings=self.embeddings,
            run_config=run_config,
        )

        frame = result.to_pandas()
        name_map = {name: getattr(RAGAS_METRICS[name], "name", name) for name in selected}
        return frame, name_map

    def evaluate(self, samples: dict, metrics: list[str] | None = None) -> dict[str, float]:
        frame, name_map = self.score_frame(samples, metrics)
        if frame is None:
            return {}
        scores: dict[str, float] = {}
        for key, column in name_map.items():
            col = column if column in frame else (key if key in frame else None)
            if col is not None:
                scores[key] = float(frame[col].mean(skipna=True))
        return scores
