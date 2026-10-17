import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, models

client = TestClient(app)


class FakeTagger:
    """Stands in for ClipTagger so tests don't download or run the real model."""

    def predict(self, data):
        Image.open(io.BytesIO(data))  # raises for non-images, like the real tagger
        return ["dog", "beach"]


@pytest.fixture(autouse=True)
def fake_tagger():
    models["tagger"] = FakeTagger()
    yield
    models.clear()


def png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_returns_tags():
    image = png_bytes()
    response = client.post(
        "/analyze",
        files={"file": ("dog.png", image, "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "dog.png",
        "content_type": "image/png",
        "bytes": len(image),
        "tags": ["dog", "beach"],
    }


def test_analyze_rejects_non_images():
    response = client.post(
        "/analyze",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "File is not a valid image"}
