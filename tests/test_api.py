from __future__ import annotations

import asyncio
from io import BytesIO

from ai_art_detector.api.app import create_app
from ai_art_detector.inference.predictor import InvalidImageError, PredictionResult


def _get_route(app, path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route
    raise AssertionError(f"Route not found: {path}")


def _upload_file(filename: str, payload: bytes, content_type: str):
    from starlette.datastructures import Headers, UploadFile

    return UploadFile(
        file=BytesIO(payload),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


class DummyPredictor:
    backend = "dummy"
    model_name = "dummy-model"

    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail

    def predict_bytes(self, payload: bytes) -> PredictionResult:
        if self.should_fail:
            raise InvalidImageError("bad image")
        return PredictionResult(
            predicted_label="ai",
            decision="ai",
            probability_ai=0.91,
            probabilities={"human": 0.09, "ai": 0.91},
            confidence=0.91,
            threshold=0.5,
            model_name=self.model_name,
            backend=self.backend,
            calibrated=False,
        )


def test_health_endpoint_reports_loaded_predictor() -> None:
    app = create_app()
    app.state.predictor = DummyPredictor()
    route = _get_route(app, "/health")
    payload = asyncio.run(route.endpoint())
    assert payload.model_loaded is True


def test_predict_endpoint_returns_structured_payload() -> None:
    app = create_app()
    app.state.predictor = DummyPredictor()
    route = _get_route(app, "/predict")
    response = asyncio.run(route.endpoint(_upload_file("example.png", b"fake-bytes", "image/png")))
    assert response.predicted_label == "ai"
    assert response.probabilities.ai == 0.91


def test_predict_endpoint_rejects_bad_images() -> None:
    from fastapi import HTTPException

    app = create_app()
    app.state.predictor = DummyPredictor(should_fail=True)
    route = _get_route(app, "/predict")
    try:
        asyncio.run(route.endpoint(_upload_file("bad.png", b"bad-bytes", "image/png")))
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "bad image" in str(exc.detail)
    else:
        raise AssertionError("Expected HTTPException for invalid input.")
