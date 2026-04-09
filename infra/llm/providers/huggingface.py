from langchain_community.llms import HuggingFaceEndpoint
from infra.llm.llm_wrapper import BaseLLM

class HuggingFaceLLM(BaseLLM):
    def __init__(self, repo_id):
        self.llm = HuggingFaceEndpoint(repo_id=repo_id)

    def invoke(self, prompt, **kwargs):
        return self.llm.invoke(prompt, **kwargs)

    def stream(self, prompt, **kwargs):
        stream = getattr(self.llm, "stream", None)
        if callable(stream):
            for chunk in stream(prompt, **kwargs):
                yield chunk
            return

        yield self.invoke(prompt, **kwargs)
