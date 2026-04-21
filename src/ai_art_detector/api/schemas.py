"""Pydantic schemas for the FastAPI surface."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    backend: str | None = None
    model_name: str | None = None


class PredictionProbabilities(BaseModel):
    human: float = Field(..., ge=0.0, le=1.0)
    ai: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_label: str
    decision: str
    probability_ai: float = Field(..., ge=0.0, le=1.0)
    probabilities: PredictionProbabilities
    confidence: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    model_name: str
    backend: str
    calibrated: bool


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]


class ErrorResponse(BaseModel):
    detail: str
