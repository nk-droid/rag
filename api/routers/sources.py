from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.schemas import (
    RegisterPathRequest,
    RegisterPathResponse,
    RegisterRepoRequest,
    RegisterRepoResponse,
    SourcesResponse,
    UploadSourcesResponse,
)
from api.source_store import SourceStore, validate_public_repo_url
from components.ingestion.repo_cloner import RepoCloneError, RepoCloner, RepoClonerSettings

router = APIRouter(prefix="/api/sources", tags=["sources"])

def get_store() -> SourceStore:
    return SourceStore()

def get_repo_cloner() -> RepoCloner:
    return RepoCloner(RepoClonerSettings())

@router.get("", response_model=SourcesResponse)
def list_sources(store: SourceStore = Depends(get_store)) -> SourcesResponse:
    return SourcesResponse(sources=store.list_sources())

@router.post("/register-path", response_model=RegisterPathResponse)
def register_path(
    payload: RegisterPathRequest,
    store: SourceStore = Depends(get_store),
) -> RegisterPathResponse:
    resolved = Path(payload.path).expanduser().resolve()
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {resolved}")

    source_type, loader = store.resolve_loader_for_path(resolved)
    if source_type == "file" and loader == "document_loader":
        raise HTTPException(
            status_code=400,
            detail="Unsupported file extension. Supported: .md, .markdown, .txt, .log",
        )

    source = store.add_source(
        name=resolved.name,
        source_type=source_type,
        loader=loader,
        path=str(resolved),
        size_bytes=resolved.stat().st_size if resolved.is_file() else None,
    )
    return RegisterPathResponse(source=source)

@router.post("/register-repo", response_model=RegisterRepoResponse)
def register_repo(
    payload: RegisterRepoRequest,
    store: SourceStore = Depends(get_store),
    repo_cloner: RepoCloner = Depends(get_repo_cloner),
) -> RegisterRepoResponse:
    try:
        repo_url = validate_public_repo_url(payload.repo_url)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        checkout = repo_cloner.clone_or_update(
            repo_url=repo_url,
            branch=payload.branch.strip() if payload.branch and payload.branch.strip() else None,
        )
    except RepoCloneError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    source = store.add_repository_source(checkout)
    return RegisterRepoResponse(source=source)

@router.post("/upload", response_model=UploadSourcesResponse)
async def upload_sources(
    files: list[UploadFile] = File(...),
    store: SourceStore = Depends(get_store),
) -> UploadSourcesResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    saved = []
    for item in files:
        contents = await item.read()
        try:
            source = store.persist_uploaded_file(filename=item.filename or "", contents=contents)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        saved.append(source)

    return UploadSourcesResponse(sources=saved)
