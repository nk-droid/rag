from components._base import ComponentSettings
from components.shared_types import EvaluationResult

class EvaluatorSettings(ComponentSettings):
    _CONFIG_PATH = "evaluation.base"

class Evaluator:
    def __init__(self, settings: EvaluatorSettings | None = None) -> None:
        self.settings = settings or EvaluatorSettings()

    def evaluate(self, samples: dict) -> list[EvaluationResult]:
        raise NotImplementedError
