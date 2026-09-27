from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile

from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    filename: str | None
    content_type: str | None
    bytes: int
    tags: list[str]

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load heavy ML models once at startup, not per request.
    # models["tagger"] = load_tagger()
    yield
    models.clear()


app = FastAPI(title="smart-gallery-ml", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile) -> AnalyzeResponse:
    data = await file.read()
    # result = models["tagger"].predict(data)
    return AnalyzeResponse(
        filename=file.filename,
        content_type=file.content_type,
        bytes=len(data),
        tags=[]
    )

