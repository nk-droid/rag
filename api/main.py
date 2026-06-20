from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers.components import router as components_router
from api.routers.experiments import router as experiments_router
from api.routers.pipelines import router as pipelines_router
from api.routers.prompts import router as prompts_router
from api.routers.sources import router as sources_router

app = FastAPI(title="Modular RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

app.include_router(components_router)
app.include_router(sources_router)
app.include_router(pipelines_router)
app.include_router(prompts_router)
app.include_router(experiments_router)
