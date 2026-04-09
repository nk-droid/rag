from langchain_openai import ChatOpenAI
from infra.llm.llm_wrapper import BaseLLM

class OpenAILLM(BaseLLM):
    def __init__(self, model, temperature=0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def invoke(self, prompt, **kwargs):
        return self.llm.invoke(prompt).content

    def stream(self, prompt, **kwargs):
        for chunk in self.llm.stream(prompt):
            yield chunk.content