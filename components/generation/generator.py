from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from components._base import ComponentSettings
from infra.llm.llm_wrapper import BaseLLM

class GeneratorSettings(ComponentSettings):
    _CONFIG_PATH = "generation.generator"

class Generator:
    def __init__(self, settings: GeneratorSettings, llm: BaseLLM) -> None:
        self.settings = settings
        self.llm = llm

    def _get_llm_runnable(self):
        llm_runnable = getattr(self.llm, "llm", None)
        if llm_runnable is not None:
            return llm_runnable

        invoke = getattr(self.llm, "invoke", None)
        if callable(invoke):
            return RunnableLambda(lambda prompt: invoke(prompt))

        raise TypeError(
            f"Expected llm to expose a runnable or callable invoke method. Got: {type(self.llm)}"
        )

    def generate(self, prompt: PromptTemplate, inputs: dict) -> str:
        llm_runnable = self._get_llm_runnable()
        chain = prompt | llm_runnable
        return chain.invoke(inputs)
