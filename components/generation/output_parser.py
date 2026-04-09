from typing import List
from langchain_core.output_parsers import PydanticOutputParser

# TODO: Move to separate files
from pydantic import Field, BaseModel

class Answer(BaseModel):
    answer: str = Field(..., description="Answer to the query")

class SemanticChunks(BaseModel):
    chunks: List[str] = Field(
        ...,
        description="Ordered list of semantic chunks as plain text strings only (no topic keys, no nested objects).",
    )

pydantic_models = {
    "Answer": Answer,
    "SemanticChunks": SemanticChunks
}
    
class OutputParser:
    """Convert raw model output into a structured payload."""

    def parse(self, text: str, parser_model: str):
        parser = PydanticOutputParser(pydantic_object=pydantic_models.get(parser_model))
        return parser.parse(text)
