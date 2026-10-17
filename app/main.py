import asyncio
import os
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import FastAPI, HTTPException, UploadFile
from PIL import UnidentifiedImageError
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.tagger import ClipTagger


class AnalyzeResponse(BaseModel):
    filename: str | None
    content_type: str | None
    bytes: int
    tags: list[str]
    embedding: list[float]


class Tagger(Protocol):
    """Anything with a predict method like ClipTagger's (tests use a fake)."""

    def predict(self, data: bytes) -> tuple[list[str], list[float]]: ...


models: dict[str, Tagger] = {}

# torch already spreads one prediction across every CPU core, so running many at once
# only makes them compete. Extra requests wait here for a free slot.
inference_slots = asyncio.Semaphore(int(os.environ.get("INFERENCE_CONCURRENCY", "1")))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load heavy ML models once at startup, not per request.
    models["tagger"] = ClipTagger()
    yield
    models.clear()


app = FastAPI(title="smart-gallery-ml", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile) -> AnalyzeResponse:
    data = await file.read()
    try:
        async with inference_slots:
            # Running the model is slow CPU work; do it in a thread so other requests aren't blocked.
            tags, embedding = await run_in_threadpool(models["tagger"].predict, data)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image") from None
    return AnalyzeResponse(
        filename=file.filename,
        content_type=file.content_type,
        bytes=len(data),
        tags=tags,
        embedding=embedding,
    )
