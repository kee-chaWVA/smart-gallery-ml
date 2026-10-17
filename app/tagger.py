import io
import os

import torch
from PIL import Image
from torch.nn.functional import normalize
from transformers import CLIPModel, CLIPProcessor

DEFAULT_MODEL = "openai/clip-vit-base-patch32"

# Candidate tags. CLIP scores every photo against each of these.
LABELS = [
    "person",
    "group of people",
    "baby",
    "dog",
    "cat",
    "bird",
    "horse",
    "car",
    "bicycle",
    "boat",
    "airplane",
    "train",
    "beach",
    "mountain",
    "forest",
    "lake",
    "snow",
    "desert",
    "city",
    "street",
    "building",
    "house",
    "garden",
    "park",
    "sunset",
    "night sky",
    "food",
    "drink",
    "cake",
    "flowers",
    "concert",
    "sports",
    "wedding",
    "party",
    "birthday",
    "selfie",
    "document",
    "screenshot",
]


class ClipTagger:
    """Zero-shot image tagger: picks the LABELS that best match a photo."""

    def __init__(
        self,
        model_name: str | None = None,
        labels: list[str] = LABELS,
        top_k: int = 5,
        min_score: float = 0.1,
    ):
        model_name = model_name or os.environ.get("CLIP_MODEL", DEFAULT_MODEL)
        self.model = CLIPModel.from_pretrained(model_name).eval()  # pyright: ignore[reportUnknownMemberType]
        self.processor = CLIPProcessor.from_pretrained(model_name)  # pyright: ignore[reportUnknownMemberType]
        self.labels = labels
        self.top_k = top_k
        self.min_score = min_score
        # The labels never change, so embed them once here instead of per request.
        self.label_features = self._embed_labels()

    @torch.inference_mode()
    def _embed_labels(self):
        prompts = [f"a photo of a {label}" for label in self.labels]
        inputs = self.processor.tokenizer(prompts, return_tensors="pt", padding=True)
        pooled = self.model.text_model(**inputs).pooler_output
        return normalize(self.model.text_projection(pooled), dim=-1)

    @torch.inference_mode()
    def predict(self, data: bytes) -> tuple[list[str], list[float]]:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        inputs = self.processor.image_processor(image, return_tensors="pt")
        pooled = self.model.vision_model(**inputs).pooler_output
        image_features = normalize(self.model.visual_projection(pooled), dim=-1)

        # Similarity between the photo and every label, turned into scores that sum to 1.
        logits = self.model.logit_scale.exp() * image_features @ self.label_features.T
        scores = logits.softmax(dim=-1)[0]

        best = scores.topk(min(self.top_k, len(self.labels)))
        tags = [
            self.labels[int(i)]
            for score, i in zip(best.values, best.indices, strict=True)
            if float(score) >= self.min_score
        ]
        # The photo's 512-number CLIP vector, for similarity search.
        embedding = [float(x) for x in image_features[0]]
        return tags, embedding
