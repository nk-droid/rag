from fastapi import APIRouter

from api.catalog import as_json_payload

router = APIRouter(prefix="/api/components", tags=["components"])

@router.get("/catalog")
def get_components_catalog() -> dict[str, object]:
    return as_json_payload()
