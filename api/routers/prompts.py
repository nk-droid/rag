from fastapi import APIRouter, HTTPException

from api.prompt_service import create_prompt_template, list_prompt_templates
from api.schemas import (
    PromptCreateRequest,
    PromptCreateResponse,
    PromptTemplatesResponse,
)

router = APIRouter(prefix="/api/prompts", tags=["prompts"])

@router.get("", response_model=PromptTemplatesResponse)
def get_prompts() -> PromptTemplatesResponse:
    return PromptTemplatesResponse(prompts=list_prompt_templates())

@router.post("", response_model=PromptCreateResponse)
def create_prompt(payload: PromptCreateRequest) -> PromptCreateResponse:
    try:
        prompt = create_prompt_template(payload.name, payload.template, payload.overwrite)
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return PromptCreateResponse(prompt=prompt)
