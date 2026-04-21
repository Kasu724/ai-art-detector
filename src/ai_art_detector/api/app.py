"""FastAPI application factory for AI art detector inference."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile

from ai_art_detector.api.schemas import (
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
)
from ai_art_detector.config import load_experiment_config
from ai_art_detector.inference.predictor import (
    InvalidImageError,
    PredictionResult,
    load_predictor_from_environment,
)


def _to_response(payload: PredictionResult) -> PredictionResponse:
    return PredictionResponse(
        predicted_label=payload.predicted_label,
        decision=payload.decision,
        probability_ai=payload.probability_ai,
        probabilities=payload.probabilities,
        confidence=payload.confidence,
        threshold=payload.threshold,
        model_name=payload.model_name,
        backend=payload.backend,
        calibrated=payload.calibrated,
    )


def create_app(predictor: Any | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor
        app.state.startup_error = None
        if app.state.predictor is None:
            try:
                config_path = os.getenv("AIAD_CONFIG_PATH")
                config = load_experiment_config(config_path) if config_path else None
                app.state.predictor = load_predictor_from_environment(config=config)
            except Exception as exc:  # pragma: no cover - startup fallback
                app.state.startup_error = str(exc)
        yield

    app = FastAPI(
        title="AI Art Detector API",
        version="0.1.0",
        description="Inference API for classifying images as AI-generated or human-made.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        predictor_obj = getattr(app.state, "predictor", None)
        startup_error = getattr(app.state, "startup_error", None)
        if predictor_obj is not None:
            return HealthResponse(
                status="ok",
                model_loaded=True,
                backend=getattr(predictor_obj, "backend", None),
                model_name=getattr(predictor_obj, "model_name", None),
            )
        status = "error" if startup_error else "not_ready"
        return HealthResponse(status=status, model_loaded=False)

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def predict(file: UploadFile = File(...)) -> PredictionResponse:
        predictor_obj = getattr(app.state, "predictor", None)
        if predictor_obj is None:
            raise HTTPException(status_code=503, detail="Model is not loaded.")

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

        payload = await file.read()
        try:
            prediction = predictor_obj.predict_bytes(payload)
        except InvalidImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _to_response(prediction)

    @app.post(
        "/predict-batch",
        response_model=BatchPredictionResponse,
        responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def predict_batch(files: list[UploadFile] = File(...)) -> BatchPredictionResponse:
        predictor_obj = getattr(app.state, "predictor", None)
        if predictor_obj is None:
            raise HTTPException(status_code=503, detail="Model is not loaded.")

        results = []
        for file in files:
            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="All uploaded files must be images.")
            payload = await file.read()
            try:
                prediction = predictor_obj.predict_bytes(payload)
            except InvalidImageError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            results.append(_to_response(prediction))
        return BatchPredictionResponse(results=results)

    return app


app = create_app()
