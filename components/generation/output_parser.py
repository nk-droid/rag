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

class RewrittenQuery(BaseModel):
    query: str = Field(..., description="Single rewritten retrieval-ready query.")

class QueryVariants(BaseModel):
    queries: List[str] = Field(..., description="List of rewritten search query variants.")

class SelfCritique(BaseModel):
    needs_refine: bool = Field(..., description="Whether the answer needs refinement based on self-critique.")
    grounded: bool = Field(..., description="Whether the answer is grounded in the provided context.")
    issues: List[str] = Field(..., description="List of issues found in the answer.")
    suggestions: List[str] = Field(..., description="List of suggestions for improving the answer.")

class RefinedAnswer(BaseModel):
    answer: str = Field(..., description="Refined answer to the query based on self-critique.")

pydantic_models = {
    "Answer": Answer,
    "SemanticChunks": SemanticChunks,
    "RewrittenQuery": RewrittenQuery,
    "QueryVariants": QueryVariants,
    "SelfCritique": SelfCritique,
    "RefinedAnswer": RefinedAnswer,
}
    
class OutputParser:
    """Convert raw model output into a structured payload."""

    def parse(self, text: str, parser_model: str):
        parser = PydanticOutputParser(pydantic_object=pydantic_models.get(parser_model))
        return parser.parse(text)
