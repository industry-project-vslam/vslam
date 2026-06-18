from __future__ import annotations

import threading
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse

from .config import Settings, load_settings
from .pipeline import ImageDecodeError, PostureDetectionPipeline


app = FastAPI(title="VSLAM Object Detection API", version="1.0.0")

_settings: Settings = load_settings()
_pipeline: PostureDetectionPipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> PostureDetectionPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = PostureDetectionPipeline(_settings)
    return _pipeline


def _model_status() -> dict[str, Any]:
    return {
        "detector_model": str(_settings.detector_model),
        "classifier_model": str(_settings.classifier_model),
        "class_map": str(_settings.class_map),
        "detector_model_exists": _settings.detector_model.exists(),
        "classifier_model_exists": _settings.classifier_model.exists(),
        "class_map_exists": _settings.class_map.exists(),
        "device": _settings.device,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    models = _model_status()
    return {
        "status": "ok" if all(models[key] for key in ("detector_model_exists", "classifier_model_exists", "class_map_exists")) else "missing_models",
        "models": models,
    }


@app.post("/predict")
async def predict(request: Request) -> JSONResponse:
    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Request body is empty.")
    if len(image_bytes) > _settings.max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image is too large. Max size is {_settings.max_image_bytes} bytes.",
        )

    try:
        result = get_pipeline().predict_bytes(image_bytes)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return JSONResponse(result)


@app.websocket("/stream")
async def stream(websocket: WebSocket) -> None:
    await websocket.accept()
    frame_id = 0

    try:
        while True:
            message = await websocket.receive()
            if "bytes" not in message or message["bytes"] is None:
                await websocket.send_json({"error": "Send image frames as binary JPEG or PNG bytes."})
                continue

            image_bytes = message["bytes"]
            frame_id += 1
            if len(image_bytes) > _settings.max_image_bytes:
                await websocket.send_json(
                    {
                        "frame_id": frame_id,
                        "error": f"Image is too large. Max size is {_settings.max_image_bytes} bytes.",
                    }
                )
                continue

            try:
                result = get_pipeline().predict_bytes(image_bytes)
            except ImageDecodeError as exc:
                await websocket.send_json({"frame_id": frame_id, "error": str(exc)})
                continue

            result["frame_id"] = frame_id
            await websocket.send_json(result)
    except WebSocketDisconnect:
        return
