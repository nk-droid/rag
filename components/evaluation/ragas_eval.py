from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate as ragas_evaluate
from ragas.metrics import _answer_relevancy, _context_precision, _context_recall, _faithfulness

from components._base import ComponentSettings
from components.evaluation.evaluator import Evaluator
from components.shared_types import EvaluationResult

class RagasEvaluatorSettings(ComponentSettings):
    _CONFIG_PATH = "evaluation.ragas"

    llm_model: str = "llama3.2:latest"
    embedding_model: str = "qwen3-embedding:4b"

class RagasEvaluator(Evaluator):
    def __init__(self, settings: RagasEvaluatorSettings) -> None:
        self.settings = settings
        self.llm = ChatOllama(model=settings.llm_model)
        self.embeddings = OllamaEmbeddings(model=settings.embedding_model)

    def evaluate(self, samples: dict) -> list[EvaluationResult]:
        dataset = Dataset.from_dict(samples)
        return ragas_evaluate(
            dataset,
            metrics=[
                _faithfulness,
                _answer_relevancy,
                _context_precision,
                _context_recall,
            ],
            llm=self.llm,
            embeddings=self.embeddings,
        )
