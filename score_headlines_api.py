


"""
score_headlines_api.py

FastAPI service to score a list of news headlines with a sentiment model.

Endpoints:
- GET  /status -> {"status": "OK"}
- POST /score_headlines -> {"labels": ["Optimistic", "Neutral", ...]}
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, conlist



logger = logging.getLogger("headline_api")


def setup_logging() -> None:
    """Configure logging once for the app."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logger.info("Logging initialized at level=%s", log_level)


# -----------------------------
# Request/Response Schemas
# -----------------------------
class ScoreHeadlinesRequest(BaseModel):
    # Require at least 1 headline; cap can be adjusted
    headlines: conlist(str, min_length=1) = Field(
        ..., description="List of news headlines to score"
    )


class ScoreHeadlinesResponse(BaseModel):
    labels: List[str]


# -----------------------------
# Model Loading (global)
# -----------------------------
app = FastAPI(title="Headline Sentiment API", version="1.0.0")

# Global objects initialized at startup
MODEL = None
TOKENIZER = None
PIPELINE = None

# Map model outputs to required labels if needed
# Adjust this mapping to match your trained model.
LABEL_MAP = {
    "LABEL_0": "Pessimistic",
    "LABEL_1": "Neutral",
    "LABEL_2": "Optimistic",
    # Sometimes models output lowercase or different names:
    "negative": "Pessimistic",
    "neutral": "Neutral",
    "positive": "Optimistic",
}


def load_sentiment_pipeline():
    """
    Load your model once.

    Replace this with the *same* model-loading logic you used in your batch job.
    The key requirement: do NOT load inside the request handler.
    """
    # Option A (common): HuggingFace pipeline
    from transformers import pipeline  # pylint: disable=import-outside-toplevel

    model_name_or_path = os.getenv("MODEL_NAME_OR_PATH", "").strip()

    if not model_name_or_path:
   
        raise RuntimeError(
            "MODEL_NAME_OR_PATH is not set. "
            "Set it to your local saved model directory or HF model id."
        )

    logger.info("Loading sentiment pipeline from: %s", model_name_or_path)
    return pipeline("text-classification", model=model_name_or_path, tokenizer=model_name_or_path)


def normalize_label(raw_label: str) -> str:
    """Convert model output label into one of: Optimistic/Neutral/Pessimistic."""
    if raw_label in LABEL_MAP:
        return LABEL_MAP[raw_label]

    # If the model already outputs required labels, pass through:
    if raw_label in {"Optimistic", "Neutral", "Pessimistic"}:
        return raw_label

    # Fallback: unknown label
    logger.warning("Unknown label from model: %s", raw_label)
    return raw_label


@app.on_event("startup")
def on_startup() -> None:
    setup_logging()

    global PIPELINE  # noqa: PLW0603
    try:
        PIPELINE = load_sentiment_pipeline()
        logger.info("Model pipeline loaded successfully.")
    except Exception as exc:  # pylint: disable=broad-except
        logger.critical("Failed to load model pipeline: %s", exc, exc_info=True)
        # Re-raise so the server fails fast instead of running broken
        raise


# -----------------------------
# Routes
# -----------------------------
@app.get("/status")
def status() -> dict:
    return {"status": "OK"}


@app.post("/score_headlines", response_model=ScoreHeadlinesResponse)
async def score_headlines(payload: ScoreHeadlinesRequest, request: Request) -> ScoreHeadlinesResponse:
    client_host: Optional[str] = getattr(request.client, "host", None)
    logger.info("POST /score_headlines from %s (n=%d)", client_host, len(payload.headlines))

    if PIPELINE is None:
        logger.error("Model pipeline is not initialized.")
        raise HTTPException(status_code=500, detail="Model not initialized")

    # Basic input sanitation
    headlines = [h.strip() for h in payload.headlines]
    if any(not h for h in headlines):
        logger.warning("Received empty/blank headline(s).")
        raise HTTPException(status_code=400, detail="Headlines must be non-empty strings")

    try:
        # Many pipelines accept a list directly and return list[dict]
        results = PIPELINE(headlines)

        labels: List[str] = []
        for item in results:
            raw = item.get("label", "")
            labels.append(normalize_label(raw))

        return ScoreHeadlinesResponse(labels=labels)

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Scoring failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Scoring failed") from exc


# -----------------------------
# Local / Server run
# -----------------------------
if __name__ == "__main__":
    # Use port 8087 for linsabones
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8087"))
    logger.info("Starting server on %s:%d", host, port)
    uvicorn.run("score_headlines_api:app", host=host, port=port, log_level="info")
